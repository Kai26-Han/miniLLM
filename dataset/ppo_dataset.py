#!/usr/bin/env python3
"""Prompt-only dataset for miniLLM PPO rollouts.

Each JSONL row contains a ``conversations`` list whose final message is an
empty assistant placeholder.  The placeholder is removed and the remaining
conversation is rendered with the tokenizer chat template plus a generation
prompt.  PPO then samples a fresh answer from the current policy.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from dataset.lm_dataset import validate_tokenizer
from dataset.sft_dataset import normalize_conversations


PPO_STAT_KEYS = ("source_indices", "prompt_tokens", "prompt_truncated")
AGENT_STAT_KEYS = ("source_indices", "difficulties")
_PPO_PREFLIGHT_CACHE: dict[
    tuple[tuple[str, int, int], ...], tuple[int, ...]
] = {}


def _stable_probability(seed: int, index: int, namespace: str) -> float:
    payload = f"{seed}:{index}:{namespace}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value / 2**64


def _prepare_prompt_messages(raw: Any) -> tuple[list[dict[str, Any]], int]:
    """Validate one rollout row and return its prompt plus repair count.

    A few public RLAIF rows end in ``assistant(answer), assistant("")``.
    The second assistant cannot respond to the first one, so the only valid
    rollout prompt is the nearest preceding user/tool turn.  Plain trailing
    assistant answers are therefore removed deterministically.  Unfinished
    tool calls remain hard errors because discarding them would change the
    tool interaction semantics.
    """

    messages = normalize_conversations(raw)
    final = messages[-1]
    if final["role"] != "assistant":
        raise ValueError("conversation must end with an assistant placeholder")
    if final.get("content", "").strip():
        raise ValueError("final assistant placeholder must have empty content")
    if final.get("reasoning_content", "").strip() or final.get("tool_calls"):
        raise ValueError("final assistant placeholder cannot contain reasoning or tools")
    prompt_messages = messages[:-1]
    removed_assistant_answers = 0
    while prompt_messages and prompt_messages[-1]["role"] == "assistant":
        trailing = prompt_messages[-1]
        if trailing.get("tool_calls"):
            raise ValueError(
                "assistant tool call before the placeholder has no tool response"
            )
        prompt_messages.pop()
        removed_assistant_answers += 1
    if not prompt_messages:
        raise ValueError("conversation has no prompt before the assistant placeholder")
    if prompt_messages[-1]["role"] not in {"user", "tool"}:
        raise ValueError("the message before the placeholder must be user or tool")
    return prompt_messages, removed_assistant_answers


def _extract_prompt_messages(raw: Any) -> list[dict[str, Any]]:
    """Return the valid context preceding an empty assistant placeholder."""

    return _prepare_prompt_messages(raw)[0]


def _preflight_ppo_rows(samples: Any, data_paths: Sequence[Path]) -> tuple[int, ...]:
    """Validate every row before expensive policy/reward models are loaded."""

    cache_key = tuple(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in data_paths
    )
    cached = _PPO_PREFLIGHT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    repaired: list[int] = []
    failures: list[tuple[int, str]] = []
    for source_index in range(len(samples)):
        try:
            _, repair_count = _prepare_prompt_messages(
                samples[source_index]["conversations"]
            )
            if repair_count:
                repaired.append(source_index)
        except Exception as exc:
            failures.append((source_index, str(exc)))

    if failures:
        preview = "; ".join(
            f"row {source_index}: {reason}"
            for source_index, reason in failures[:8]
        )
        suffix = f"; plus {len(failures) - 8} more" if len(failures) > 8 else ""
        raise ValueError(
            "PPO dataset preflight failed before training: "
            f"{preview}{suffix}"
        )

    result = tuple(repaired)
    _PPO_PREFLIGHT_CACHE[cache_key] = result
    return result


class PPODataset(Dataset[dict[str, Any]]):
    """Load RLAIF-style conversations and return rendered rollout prompts."""

    def __init__(
        self,
        data_paths: str | Path | Sequence[str | Path],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_len: int = 512,
        split: str = "train",
        val_ratio: float = 0.02,
        expected_vocab_size: int = 8192,
        thinking_ratio: float = 0.0,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be one of: train, validation, all")
        if max_prompt_len < 16:
            raise ValueError("max_prompt_len must be at least 16")
        if split != "all" and not 0.0 < val_ratio < 0.5:
            raise ValueError("val_ratio must be between 0 and 0.5")
        if not 0.0 <= thinking_ratio <= 1.0:
            raise ValueError("thinking_ratio must be in [0, 1]")
        if not tokenizer.chat_template:
            raise ValueError("PPO requires a tokenizer chat template")

        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        self.data_paths = [Path(path).expanduser().resolve() for path in data_paths]
        missing = [str(path) for path in self.data_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("PPO data file(s) not found: " + ", ".join(missing))
        validate_tokenizer(tokenizer, expected_vocab_size)

        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.split = split
        self.thinking_ratio = thinking_ratio
        self.seed = seed
        self.samples = load_dataset(
            "json",
            data_files=[str(path) for path in self.data_paths],
            split="train",
            keep_in_memory=False,
        )
        if "conversations" not in self.samples.column_names:
            raise ValueError("PPO JSONL must contain a 'conversations' column")
        if len(self.samples) == 0:
            raise ValueError("PPO dataset is empty")
        self.repaired_source_indices = _preflight_ppo_rows(
            self.samples, self.data_paths
        )
        sample_count = len(self.samples)
        if split != "all" and sample_count < 2:
            raise ValueError("PPO train/validation split requires at least 2 samples")
        self.validation_length = (
            0
            if split == "all"
            else min(sample_count - 1, max(1, round(sample_count * val_ratio)))
        )
        multiplier = (2 * abs(seed) + 1) % sample_count or 1
        while math.gcd(multiplier, sample_count) != 1:
            multiplier = (multiplier + 2) % sample_count or 1
        self.permutation_multiplier = multiplier
        self.permutation_offset = int(
            _stable_probability(seed, sample_count, "split-offset") * sample_count
        )

    def __len__(self) -> int:
        if self.split == "all":
            return len(self.samples)
        if self.split == "validation":
            return self.validation_length
        return len(self.samples) - self.validation_length

    def _source_index(self, index: int) -> int:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)
        if self.split == "all":
            return index
        position = index if self.split == "validation" else self.validation_length + index
        return (
            self.permutation_multiplier * position + self.permutation_offset
        ) % len(self.samples)

    def _render(self, messages: Sequence[Mapping[str, Any]], open_thinking: bool) -> str:
        tools: list[Any] = []
        for message in messages:
            tools.extend(message.get("tools", []))
        return self.tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=open_thinking,
            tools=tools or None,
        )

    def _fit_prompt(
        self,
        messages: list[dict[str, Any]],
        open_thinking: bool,
    ) -> tuple[str, int, bool, list[dict[str, Any]]]:
        """Drop complete old turns, then left-truncate an oversized newest turn."""

        def encode(text: str) -> list[int]:
            return list(
                self.tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                    verbose=False,
                )["input_ids"]
            )

        prompt = self._render(messages, open_thinking)
        token_ids = encode(prompt)
        if len(token_ids) <= self.max_prompt_len:
            return prompt, len(token_ids), False, messages

        leading_system: list[dict[str, Any]] = []
        body_start = 0
        while body_start < len(messages) and messages[body_start]["role"] == "system":
            leading_system.append(messages[body_start])
            body_start += 1
        user_starts = [
            index
            for index in range(body_start, len(messages))
            if messages[index]["role"] == "user"
        ]
        for start in user_starts[1:]:
            candidate = [*leading_system, *messages[start:]]
            candidate_prompt = self._render(candidate, open_thinking)
            candidate_ids = encode(candidate_prompt)
            if len(candidate_ids) <= self.max_prompt_len:
                return candidate_prompt, len(candidate_ids), True, candidate

        # If the newest complete turn is still too large, keep the most recent
        # token suffix. Decoding makes the returned prompt and token count agree;
        # this fallback is explicitly surfaced through prompt_truncated.
        fallback_start = user_starts[-1] if user_starts else body_start
        candidate = [*leading_system, *messages[fallback_start:]]
        candidate_prompt = self._render(candidate, open_thinking)
        candidate_ids = encode(candidate_prompt)
        fitted_ids = candidate_ids[-self.max_prompt_len :]
        fitted_prompt = self.tokenizer.decode(
            fitted_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        # The learned reward must not see context hidden from the Actor. This
        # fallback represents the exact truncated model input as one user turn.
        reward_messages = [{"role": "user", "content": fitted_prompt}]
        return fitted_prompt, len(fitted_ids), True, reward_messages

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = self._source_index(index)
        try:
            messages = _extract_prompt_messages(
                self.samples[source_index]["conversations"]
            )
            open_thinking = (
                _stable_probability(self.seed, source_index, "open-thinking")
                < self.thinking_ratio
            )
            prompt, prompt_tokens, truncated, reward_messages = self._fit_prompt(
                messages, open_thinking
            )
        except Exception as exc:
            raise ValueError(f"Invalid PPO row {source_index}: {exc}") from exc
        return {
            "prompt": prompt,
            "messages": reward_messages,
            "source_index": torch.tensor(source_index, dtype=torch.long),
            "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
            "prompt_truncated": torch.tensor(int(truncated), dtype=torch.long),
        }


class PPODataCollator:
    """Keep variable-length prompt text/messages and stack scalar statistics."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty feature list")
        return {
            "prompts": [feature["prompt"] for feature in features],
            "messages": [feature["messages"] for feature in features],
            "source_indices": torch.stack(
                [feature["source_index"] for feature in features]
            ),
            "prompt_tokens": torch.stack(
                [feature["prompt_tokens"] for feature in features]
            ),
            "prompt_truncated": torch.stack(
                [feature["prompt_truncated"] for feature in features]
            ),
        }


class AgentRLDataset(Dataset[dict[str, Any]]):
    """Prompt-only Tool-Use tasks for multi-turn Agentic RL.

    The public MiniMind-style file mixes open-ended prompts with verifiable
    tool tasks.  Agentic RL deliberately keeps only rows that expose tools and
    contain at least one ground-truth value; open-ended rows need a reward
    model and are outside this verifier-based training stage.
    """

    def __init__(
        self,
        data_paths: str | Path | Sequence[str | Path],
        tokenizer: PreTrainedTokenizerBase,
        split: str = "train",
        val_ratio: float = 0.02,
        expected_vocab_size: int = 8192,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be one of: train, validation, all")
        if split != "all" and not 0.0 < val_ratio < 0.5:
            raise ValueError("val_ratio must be between 0 and 0.5")
        if not tokenizer.chat_template:
            raise ValueError("Agentic RL requires a tokenizer chat template")
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        paths = [Path(path).expanduser().resolve() for path in data_paths]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Agentic RL data file(s) not found: " + ", ".join(missing)
            )
        validate_tokenizer(tokenizer, expected_vocab_size)
        self.samples = load_dataset(
            "json",
            data_files=[str(path) for path in paths],
            split="train",
            keep_in_memory=False,
        )
        required = {"conversations", "gt"}
        missing_columns = required.difference(self.samples.column_names)
        if missing_columns:
            raise ValueError(
                "Agentic RL JSONL is missing column(s): "
                + ", ".join(sorted(missing_columns))
            )

        # Keep source row numbers so diagnostics still point to the original
        # JSONL even after verifier-only filtering and deterministic splitting.
        self.eligible_indices: list[int] = []
        for index, sample in enumerate(self.samples):
            raw_gt = sample.get("gt")
            conversations = sample.get("conversations") or []
            has_tools = any(
                isinstance(message, Mapping) and bool(message.get("tools"))
                for message in conversations
            )
            if has_tools and isinstance(raw_gt, Sequence) and not isinstance(
                raw_gt, (str, bytes)
            ) and len(raw_gt) > 0:
                self.eligible_indices.append(index)
        if not self.eligible_indices:
            raise ValueError("Agentic RL dataset has no verifiable tool-use rows")
        if split != "all" and len(self.eligible_indices) < 2:
            raise ValueError("Agentic RL split requires at least 2 eligible rows")

        self.split = split
        self.seed = seed
        count = len(self.eligible_indices)
        self.validation_length = (
            0 if split == "all" else min(count - 1, max(1, round(count * val_ratio)))
        )
        multiplier = (2 * abs(seed) + 1) % count or 1
        while math.gcd(multiplier, count) != 1:
            multiplier = (multiplier + 2) % count or 1
        self.permutation_multiplier = multiplier
        self.permutation_offset = int(
            _stable_probability(seed, count, "agent-split-offset") * count
        )

    def __len__(self) -> int:
        if self.split == "all":
            return len(self.eligible_indices)
        if self.split == "validation":
            return self.validation_length
        return len(self.eligible_indices) - self.validation_length

    def _source_index(self, index: int) -> int:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)
        if self.split == "all":
            eligible_position = index
        else:
            position = index if self.split == "validation" else self.validation_length + index
            eligible_position = (
                self.permutation_multiplier * position + self.permutation_offset
            ) % len(self.eligible_indices)
        return self.eligible_indices[eligible_position]

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = self._source_index(index)
        try:
            sample = self.samples[source_index]
            messages = _extract_prompt_messages(sample["conversations"])
            tools: list[Any] = []
            for message in messages:
                tools.extend(message.get("tools", []))
            if not tools:
                raise ValueError("no tools found after conversation normalization")
            gt = sample["gt"]
            if not isinstance(gt, Sequence) or isinstance(gt, (str, bytes)) or not gt:
                raise ValueError("gt must be a non-empty array")
            normalized_gt = [str(value).strip() for value in gt]
            if any(not value for value in normalized_gt):
                raise ValueError("gt values cannot be empty")
        except Exception as exc:
            raise ValueError(f"Invalid Agentic RL row {source_index}: {exc}") from exc
        return {
            "messages": messages,
            "tools": tools,
            "gt": normalized_gt,
            "source_index": torch.tensor(source_index, dtype=torch.long),
            "difficulty": torch.tensor(len(normalized_gt), dtype=torch.long),
        }


class AgentRLDataCollator:
    """Keep variable-length conversations as Python objects."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty Agentic RL batch")
        return {
            "messages": [feature["messages"] for feature in features],
            "tools": [feature["tools"] for feature in features],
            "gt": [feature["gt"] for feature in features],
            "source_indices": torch.stack(
                [feature["source_index"] for feature in features]
            ),
            "difficulties": torch.stack(
                [feature["difficulty"] for feature in features]
            ),
        }

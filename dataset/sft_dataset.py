#!/usr/bin/env python3
"""Conversation dataset for miniLLM supervised fine-tuning.

The dataset renders ChatML conversations, keeps system/user/tool messages as
context, and computes language-model loss only on assistant output spans.  It
supports the MiniMind-style ``reasoning_content``, ``tools`` and ``tool_calls``
fields used by ``dataset/sft/sft.jsonl``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from datasets import Features, Value, load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from dataset.lm_dataset import validate_tokenizer


ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
DEFAULT_SYSTEM_PROMPTS = (
    "你是 miniLLM，一个可靠、友善的人工智能助手。请准确地回答用户的问题。",
    "你是一个知识丰富的AI助手，请尽力提供准确、有价值的回答。",
    "你是 miniLLM，请认真理解用户意图并给出清晰的回答。",
    "You are miniLLM, a helpful and reliable AI assistant.",
    (
        "You are a knowledgeable AI assistant. "
        "Answer the user carefully and accurately."
    ),
)
TOOL_INSTRUCTIONS = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "Function signatures are provided inside <tools></tools> tags:\n"
    "<tools>\n{tools}\n</tools>\n\n"
    "Return each function call as JSON inside <tool_call></tool_call> tags."
)
SFT_FEATURES = Features(
    {
        "conversations": [
            {
                "role": Value("string"),
                "content": Value("string"),
                "reasoning_content": Value("string"),
                "tools": Value("string"),
                "tool_calls": Value("string"),
            }
        ]
    }
)


@dataclass(frozen=True)
class RenderedConversation:
    """Rendered text plus character spans that should contribute to CLM loss."""

    text: str
    assistant_spans: tuple[tuple[int, int], ...]


def _stable_probability(seed: int, index: int, namespace: str) -> float:
    """Return a deterministic pseudo-random value in ``[0, 1)``."""

    payload = f"{seed}:{index}:{namespace}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value / 2**64


def _parse_jsonish(value: Any, field: str) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON string in '{field}': {exc}") from exc


def _json_dump(value: Any) -> str:
    # Match Jinja's ``tojson`` spacing so training and inference render tool
    # definitions/calls identically.
    return json.dumps(value, ensure_ascii=False)


def _normalize_tool_arguments(value: Any) -> Any:
    """Normalize heterogeneous public-dataset tool arguments without data loss.

    Function arguments are usually a JSON object, but public SFT corpora also
    contain Python-literal dictionaries and occasional raw strings. A malformed
    nested argument should not abort a multi-million-row training run.
    """

    if not isinstance(value, str):
        return {} if value is None else value
    stripped = value.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        literal = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return value
    # Keep only JSON-serializable structured literals. Other literal types are
    # preserved as their original string instead of being silently rewritten.
    if not isinstance(literal, (dict, list)):
        return value
    try:
        json.dumps(literal, ensure_ascii=False)
    except (TypeError, ValueError):
        return value
    return literal


def _normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    parsed = _parse_jsonish(tool_call, "tool_calls")
    if not isinstance(parsed, Mapping):
        raise TypeError("Each tool_call must be an object")
    function = parsed.get("function", parsed)
    if not isinstance(function, Mapping):
        raise TypeError("tool_call.function must be an object")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Each tool_call must contain a non-empty function name")
    arguments = _normalize_tool_arguments(function.get("arguments", {}))
    return {"name": name, "arguments": arguments}


def normalize_conversations(conversations: Any) -> list[dict[str, Any]]:
    """Validate a raw conversation and normalize optional JSON-encoded fields."""

    if not isinstance(conversations, Sequence) or isinstance(conversations, (str, bytes)):
        raise TypeError("'conversations' must be a non-empty list of messages")
    if not conversations:
        raise ValueError("'conversations' cannot be empty")

    normalized: list[dict[str, Any]] = []
    for message_index, raw_message in enumerate(conversations):
        if not isinstance(raw_message, Mapping):
            raise TypeError(f"Message {message_index} must be an object")
        role = raw_message.get("role")
        if role not in ALLOWED_ROLES:
            raise ValueError(
                f"Message {message_index} has unsupported role {role!r}; "
                f"expected one of {sorted(ALLOWED_ROLES)}"
            )
        content = raw_message.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = _json_dump(content)

        message: dict[str, Any] = {"role": role, "content": content}
        reasoning = raw_message.get("reasoning_content", "")
        if reasoning is not None and not isinstance(reasoning, str):
            reasoning = _json_dump(reasoning)
        message["reasoning_content"] = reasoning or ""

        tools = _parse_jsonish(raw_message.get("tools"), "tools")
        if tools is not None:
            if not isinstance(tools, (list, tuple)):
                raise TypeError("'tools' must be a JSON array")
            message["tools"] = list(tools)

        raw_calls = _parse_jsonish(raw_message.get("tool_calls"), "tool_calls")
        if raw_calls:
            if not isinstance(raw_calls, (list, tuple)):
                raw_calls = [raw_calls]
            message["tool_calls"] = [_normalize_tool_call(call) for call in raw_calls]
        normalized.append(message)

    if not any(message["role"] == "assistant" for message in normalized):
        raise ValueError("Conversation contains no assistant message to supervise")
    return normalized


def _tools_from_messages(messages: Sequence[Mapping[str, Any]]) -> list[Any]:
    tools: list[Any] = []
    for message in messages:
        if message.get("tools"):
            tools.extend(message["tools"])
    return tools


def _render_tool_instructions(tools: Sequence[Any]) -> str:
    definitions = "\n".join(_json_dump(tool) for tool in tools)
    return TOOL_INSTRUCTIONS.format(tools=definitions)


def render_conversation(
    messages: Sequence[Mapping[str, Any]],
    *,
    include_empty_think: bool = False,
    injected_system_prompt: str | None = None,
) -> RenderedConversation:
    """Render messages and record exact assistant target character spans."""

    normalized = [dict(message) for message in messages]
    tools = _tools_from_messages(normalized)
    if injected_system_prompt and normalized[0].get("role") != "system":
        normalized.insert(
            0,
            {
                "role": "system",
                "content": injected_system_prompt,
                "reasoning_content": "",
            },
        )

    chunks: list[str] = []
    assistant_spans: list[tuple[int, int]] = []
    current_offset = 0
    tools_attached = False

    for message in normalized:
        role = str(message["role"])
        content = str(message.get("content", ""))
        if role == "system" and tools and not tools_attached:
            tool_text = _render_tool_instructions(tools)
            content = f"{content}\n\n{tool_text}" if content else tool_text
            tools_attached = True

        if role == "assistant":
            prefix = "<|im_start|>assistant\n"
            target_parts: list[str] = []
            reasoning = str(message.get("reasoning_content", "")).strip("\n")
            if reasoning:
                target_parts.append(f"<think>\n{reasoning}\n</think>\n\n")
            elif include_empty_think:
                target_parts.append("<think>\n\n</think>\n\n")
            target_parts.append(content.lstrip("\n"))
            for tool_call in message.get("tool_calls", []):
                if target_parts and target_parts[-1] and not target_parts[-1].endswith("\n"):
                    target_parts.append("\n")
                target_parts.append(
                    "<tool_call>\n"
                    + _json_dump(_normalize_tool_call(tool_call))
                    + "\n</tool_call>"
                )
            target_parts.append("<|im_end|>")
            target = "".join(target_parts)
            segment = prefix + target + "\n"
            start = current_offset + len(prefix)
            assistant_spans.append((start, start + len(target)))
        elif role == "tool":
            segment = (
                "<|im_start|>tool\n<tool_response>\n"
                + content
                + "\n</tool_response><|im_end|>\n"
            )
        else:
            segment = f"<|im_start|>{role}\n{content}<|im_end|>\n"

        chunks.append(segment)
        current_offset += len(segment)

    # Some tool datasets put tools on a user message and contain no system role.
    if tools and not tools_attached:
        system_segment = (
            "<|im_start|>system\n"
            + _render_tool_instructions(tools)
            + "<|im_end|>\n"
        )
        shift = len(system_segment)
        chunks.insert(0, system_segment)
        assistant_spans = [(start + shift, end + shift) for start, end in assistant_spans]

    return RenderedConversation("".join(chunks), tuple(assistant_spans))


class SFTDataset(Dataset[dict[str, torch.Tensor]]):
    """Load conversation JSONL and produce assistant-only CLM samples."""

    def __init__(
        self,
        data_paths: str | Path | Sequence[str | Path],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 1024,
        split: str = "train",
        val_ratio: float = 0.001,
        expected_vocab_size: int = 8192,
        empty_think_ratio: float = 0.2,
        system_prompt_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be one of: train, validation, all")
        if max_seq_len < 32:
            raise ValueError("max_seq_len must be at least 32")
        if split != "all" and not 0.0 < val_ratio < 0.5:
            raise ValueError("val_ratio must be between 0 and 0.5")
        if not 0.0 <= empty_think_ratio <= 1.0:
            raise ValueError("empty_think_ratio must be in [0, 1]")
        if not 0.0 <= system_prompt_ratio <= 1.0:
            raise ValueError("system_prompt_ratio must be in [0, 1]")
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("SFTDataset requires a fast tokenizer for offset mapping")

        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        self.data_paths = [Path(path).expanduser().resolve() for path in data_paths]
        missing = [str(path) for path in self.data_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("SFT data file(s) not found: " + ", ".join(missing))
        validate_tokenizer(tokenizer, expected_vocab_size)

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.empty_think_ratio = empty_think_ratio
        self.system_prompt_ratio = system_prompt_ratio if split == "train" else 0.0
        self.seed = seed
        self.samples = load_dataset(
            "json",
            data_files=[str(path) for path in self.data_paths],
            split="train",
            features=SFT_FEATURES,
            keep_in_memory=False,
        )
        if "conversations" not in self.samples.column_names:
            raise ValueError("SFT JSONL must contain a 'conversations' column")
        if len(self.samples) == 0:
            raise ValueError("SFT dataset is empty")
        sample_count = len(self.samples)
        if split != "all" and sample_count < 2:
            raise ValueError("SFT train/validation split requires at least 2 samples")
        self.validation_length = (
            0
            if split == "all"
            else min(sample_count - 1, max(1, round(sample_count * val_ratio)))
        )
        # A seeded affine permutation produces a deterministic, globally spread
        # validation partition without allocating a 5-million-element index list.
        multiplier = (2 * abs(seed) + 1) % sample_count
        multiplier = multiplier or 1
        while math.gcd(multiplier, sample_count) != 1:
            multiplier = (multiplier + 2) % sample_count or 1
        self.permutation_multiplier = multiplier
        self.permutation_offset = int(
            _stable_probability(seed, sample_count, "split-offset") * sample_count
        )

    def __len__(self) -> int:
        length = len(self.samples)
        if self.split == "all":
            return length
        return (
            self.validation_length
            if self.split == "validation"
            else length - self.validation_length
        )

    def _source_index(self, index: int) -> int:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)
        if self.split == "all":
            return index
        permutation_position = (
            index if self.split == "validation" else self.validation_length + index
        )
        return (
            self.permutation_multiplier * permutation_position
            + self.permutation_offset
        ) % len(self.samples)

    def _augmentation_choices(
        self, source_index: int, messages: Sequence[Mapping[str, Any]]
    ) -> tuple[bool, str | None]:
        include_empty_think = (
            _stable_probability(self.seed, source_index, "empty-think")
            < self.empty_think_ratio
        )
        injected_system = None
        has_system = bool(messages and messages[0].get("role") == "system")
        has_tools = bool(_tools_from_messages(messages))
        if (
            not has_system
            and not has_tools
            and _stable_probability(self.seed, source_index, "system-prompt")
            < self.system_prompt_ratio
        ):
            prompt_index = int(
                _stable_probability(self.seed, source_index, "system-prompt-choice")
                * len(DEFAULT_SYSTEM_PROMPTS)
            )
            injected_system = DEFAULT_SYSTEM_PROMPTS[
                min(prompt_index, len(DEFAULT_SYSTEM_PROMPTS) - 1)
            ]
        return include_empty_think, injected_system

    def _tokenize_rendered(
        self, rendered: RenderedConversation
    ) -> tuple[list[int], list[int]]:
        encoded = self.tokenizer(
            rendered.text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            # Long conversations are deliberately tokenized in full first so
            # `_encode_messages` can remove complete old turns instead of
            # blindly cutting through a message. The final sample is bounded
            # by `max_seq_len` before it reaches the model, so suppress the
            # tokenizer's premature model_max_length warning here.
            verbose=False,
        )
        input_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        labels = [-100] * len(input_ids)
        span_index = 0
        spans = rendered.assistant_spans
        for token_index, (start, end) in enumerate(offsets):
            while span_index < len(spans) and start >= spans[span_index][1]:
                span_index += 1
            if span_index >= len(spans):
                break
            span_start, span_end = spans[span_index]
            if end > span_start and start < span_end:
                labels[token_index] = input_ids[token_index]
        return input_ids, labels

    def _encode_messages(
        self,
        messages: list[dict[str, Any]],
        include_empty_think: bool,
        injected_system: str | None,
    ) -> tuple[list[int], list[int], bool, bool]:
        rendered = render_conversation(
            messages,
            include_empty_think=include_empty_think,
            injected_system_prompt=injected_system,
        )
        input_ids, labels = self._tokenize_rendered(rendered)
        if len(input_ids) <= self.max_seq_len:
            return input_ids, labels, False, False

        # Keep explicit system messages, then remove the oldest complete turns.
        leading_system: list[dict[str, Any]] = []
        body_start = 0
        while body_start < len(messages) and messages[body_start]["role"] == "system":
            leading_system.append(messages[body_start])
            body_start += 1
        user_starts = [
            index
            for index in range(body_start, len(messages))
            if messages[index]["role"] == "user"
            and any(
                later["role"] == "assistant" for later in messages[index + 1 :]
            )
        ]
        for start in user_starts[1:]:
            candidate = [*leading_system, *messages[start:]]
            candidate_rendered = render_conversation(
                candidate,
                include_empty_think=include_empty_think,
                injected_system_prompt=injected_system,
            )
            candidate_ids, candidate_labels = self._tokenize_rendered(candidate_rendered)
            if len(candidate_ids) <= self.max_seq_len:
                return candidate_ids, candidate_labels, True, False

        # The newest turn itself is too long. Prefer the final answer over an
        # optional long reasoning trace, then keep recent prompt context plus the
        # assistant header and answer start. A clipped answer intentionally has no
        # synthetic early EOS.
        fallback_start = user_starts[-1] if user_starts else body_start
        fallback_messages = [
            {
                **message,
                "reasoning_content": (
                    ""
                    if message["role"] == "assistant"
                    and (message.get("content") or message.get("tool_calls"))
                    else message.get("reasoning_content", "")
                ),
            }
            for message in [*leading_system, *messages[fallback_start:]]
        ]
        fallback_rendered = render_conversation(
            fallback_messages,
            include_empty_think=include_empty_think,
            injected_system_prompt=injected_system,
        )
        fallback_ids, fallback_labels = self._tokenize_rendered(fallback_rendered)
        if len(fallback_ids) <= self.max_seq_len:
            return fallback_ids, fallback_labels, True, False

        supervised_indices = [
            index for index, label in enumerate(fallback_labels) if label != -100
        ]
        if not supervised_indices:
            return (
                fallback_ids[: self.max_seq_len],
                fallback_labels[: self.max_seq_len],
                True,
                True,
            )
        first_target = supervised_indices[0]
        assistant_header_start = max(
            (
                index
                for index in range(first_target)
                if fallback_ids[index] == self.tokenizer.bos_token_id
            ),
            default=first_target,
        )
        context_budget = min(256, self.max_seq_len // 4)
        window_start = max(0, assistant_header_start - context_budget)
        window_end = min(len(fallback_ids), window_start + self.max_seq_len)
        # Near the end of a sequence, shift the window left to use all available
        # context without exceeding max_seq_len.
        window_start = max(0, window_end - self.max_seq_len)
        answer_truncated = any(
            label != -100 for label in fallback_labels[window_end:]
        )
        return (
            fallback_ids[window_start:window_end],
            fallback_labels[window_start:window_end],
            True,
            answer_truncated,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = self._source_index(index)
        try:
            sample = self.samples[source_index]
            messages = normalize_conversations(sample["conversations"])
            include_empty_think, injected_system = self._augmentation_choices(
                source_index, messages
            )
            input_ids, labels, truncated, answer_truncated = self._encode_messages(
                messages,
                include_empty_think=include_empty_think,
                injected_system=injected_system,
            )
        except Exception as exc:
            raise ValueError(f"Invalid SFT row {source_index}: {exc}") from exc
        if len(input_ids) > self.max_seq_len:
            raise RuntimeError(
                f"SFT row {source_index} produced {len(input_ids)} tokens after "
                f"truncation; expected at most {self.max_seq_len}"
            )
        if not any(label != -100 for label in labels[1:]):
            raise ValueError(
                f"SFT row {source_index} has no assistant target token after truncation"
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            # Keep the original JSONL row number outside the model inputs so a
            # numerical/data failure can identify the exact source sample.
            "source_index": torch.tensor(source_index, dtype=torch.long),
            "assistant_tokens": torch.tensor(
                sum(label != -100 for label in labels[1:]), dtype=torch.long
            ),
            "truncated": torch.tensor(int(truncated), dtype=torch.long),
            "answer_truncated": torch.tensor(int(answer_truncated), dtype=torch.long),
        }


class SFTDataCollator:
    """Dynamically right-pad SFT samples to the longest sequence in a batch."""

    def __init__(
        self,
        pad_token_id: int,
        pad_to_multiple_of: int = 8,
    ) -> None:
        if pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty feature list")
        max_length = max(feature["input_ids"].numel() for feature in features)
        multiple = self.pad_to_multiple_of
        padded_length = ((max_length + multiple - 1) // multiple) * multiple
        batch_size = len(features)
        input_ids = torch.full(
            (batch_size, padded_length), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full((batch_size, padded_length), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, padded_length), dtype=torch.long)
        for row, feature in enumerate(features):
            length = feature["input_ids"].numel()
            input_ids[row, :length] = feature["input_ids"]
            labels[row, :length] = feature["labels"]
            attention_mask[row, :length] = 1

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source_indices": torch.stack(
                [feature["source_index"] for feature in features]
            ),
            "assistant_tokens": torch.stack(
                [feature["assistant_tokens"] for feature in features]
            ),
            "truncated": torch.stack([feature["truncated"] for feature in features]),
            "answer_truncated": torch.stack(
                [feature["answer_truncated"] for feature in features]
            ),
            "sequence_tokens": attention_mask.sum(dim=1),
        }
        return result

#!/usr/bin/env python3
"""Preference-pair dataset for miniLLM Direct Preference Optimization.

Each JSONL row contains a ``chosen`` and ``rejected`` conversation.  Their
history must be identical and only the final assistant responses may differ.
The dataset renders both branches with miniLLM's ChatML renderer, supervises
only the final response, and truncates both branches with the same prompt
window so their conditional probabilities remain comparable.
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
from dataset.sft_dataset import normalize_conversations, render_conversation


DPO_STAT_KEYS = (
    "source_indices",
    "chosen_response_tokens",
    "rejected_response_tokens",
    "chosen_sequence_tokens",
    "rejected_sequence_tokens",
    "chosen_truncated",
    "rejected_truncated",
    "chosen_answer_truncated",
    "rejected_answer_truncated",
)


def _stable_probability(seed: int, index: int, namespace: str) -> float:
    payload = f"{seed}:{index}:{namespace}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value / 2**64


def _last_assistant_pair(
    raw_chosen: Any,
    raw_rejected: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize a pair and require an identical prompt plus two final answers."""

    chosen = normalize_conversations(raw_chosen)
    rejected = normalize_conversations(raw_rejected)
    if chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
        raise ValueError("chosen and rejected must both end with an assistant message")
    if chosen[:-1] != rejected[:-1]:
        raise ValueError(
            "chosen and rejected must have identical messages before the final "
            "assistant response"
        )
    if not chosen[:-1]:
        raise ValueError("a DPO pair must contain context before the final response")
    return chosen, rejected


class DPODataset(Dataset[dict[str, torch.Tensor]]):
    """Load preference JSONL and return variable-length chosen/rejected pairs."""

    def __init__(
        self,
        data_paths: str | Path | Sequence[str | Path],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 1024,
        split: str = "train",
        val_ratio: float = 0.02,
        expected_vocab_size: int = 8192,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be one of: train, validation, all")
        if max_seq_len < 32:
            raise ValueError("max_seq_len must be at least 32")
        if split != "all" and not 0.0 < val_ratio < 0.5:
            raise ValueError("val_ratio must be between 0 and 0.5")
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("DPODataset requires a fast tokenizer for offset mapping")

        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        self.data_paths = [Path(path).expanduser().resolve() for path in data_paths]
        missing = [str(path) for path in self.data_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("DPO data file(s) not found: " + ", ".join(missing))
        validate_tokenizer(tokenizer, expected_vocab_size)

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.seed = seed
        self.samples = load_dataset(
            "json",
            data_files=[str(path) for path in self.data_paths],
            split="train",
            keep_in_memory=False,
        )
        required = {"chosen", "rejected"}
        missing_columns = required.difference(self.samples.column_names)
        if missing_columns:
            raise ValueError(
                "DPO JSONL is missing column(s): " + ", ".join(sorted(missing_columns))
            )
        if len(self.samples) == 0:
            raise ValueError("DPO dataset is empty")
        sample_count = len(self.samples)
        if split != "all" and sample_count < 2:
            raise ValueError("DPO train/validation split requires at least 2 pairs")
        self.validation_length = (
            0
            if split == "all"
            else min(sample_count - 1, max(1, round(sample_count * val_ratio)))
        )

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
        if self.split == "validation":
            return self.validation_length
        return length - self.validation_length

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

    def _tokenize_final_response(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> tuple[list[int], list[int]]:
        rendered = render_conversation(messages)
        if not rendered.assistant_spans:
            raise ValueError("DPO conversation contains no assistant response")
        target_start, target_end = rendered.assistant_spans[-1]
        encoded = self.tokenizer(
            rendered.text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        input_ids = list(encoded["input_ids"])
        labels = [-100] * len(input_ids)
        for token_index, (start, end) in enumerate(encoded["offset_mapping"]):
            if end > target_start and start < target_end:
                labels[token_index] = input_ids[token_index]
        if not any(label != -100 for label in labels[1:]):
            raise ValueError("final assistant response produced no target token")
        return input_ids, labels

    def _truncate_pair(
        self,
        chosen_ids: list[int],
        chosen_labels: list[int],
        rejected_ids: list[int],
        rejected_labels: list[int],
    ) -> tuple[
        tuple[list[int], list[int], bool, bool],
        tuple[list[int], list[int], bool, bool],
    ]:
        chosen_first = next(
            index for index, label in enumerate(chosen_labels) if label != -100
        )
        rejected_first = next(
            index for index, label in enumerate(rejected_labels) if label != -100
        )
        if chosen_first != rejected_first:
            raise ValueError(
                "chosen/rejected prompt token lengths differ after chat rendering"
            )
        prefix_length = chosen_first
        if chosen_ids[:prefix_length] != rejected_ids[:prefix_length]:
            raise ValueError(
                "chosen/rejected prompt tokens differ after chat rendering"
            )

        if max(len(chosen_ids), len(rejected_ids)) <= self.max_seq_len:
            return (
                (chosen_ids, chosen_labels, False, False),
                (rejected_ids, rejected_labels, False, False),
            )

        chosen_response_length = len(chosen_ids) - prefix_length
        rejected_response_length = len(rejected_ids) - prefix_length
        longest_response = max(chosen_response_length, rejected_response_length)

        # Preserve the largest common prompt suffix possible while reserving at
        # least 75% of the window for a long response. For short responses the
        # unused response budget flows back to the prompt automatically.
        minimum_prompt = min(prefix_length, min(256, self.max_seq_len // 4))
        response_budget = self.max_seq_len - minimum_prompt
        response_to_keep = min(longest_response, response_budget)
        prompt_to_keep = min(prefix_length, self.max_seq_len - response_to_keep)
        window_start = prefix_length - prompt_to_keep

        def truncate(
            input_ids: list[int], labels: list[int]
        ) -> tuple[list[int], list[int], bool, bool]:
            window_end = min(len(input_ids), window_start + self.max_seq_len)
            answer_truncated = any(label != -100 for label in labels[window_end:])
            result_ids = input_ids[window_start:window_end]
            result_labels = labels[window_start:window_end]
            if not any(label != -100 for label in result_labels[1:]):
                raise ValueError("DPO truncation removed every response target token")
            return result_ids, result_labels, True, answer_truncated

        return truncate(chosen_ids, chosen_labels), truncate(
            rejected_ids, rejected_labels
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = self._source_index(index)
        sample = self.samples[source_index]
        try:
            chosen, rejected = _last_assistant_pair(
                sample["chosen"], sample["rejected"]
            )
            chosen_ids, chosen_labels = self._tokenize_final_response(chosen)
            rejected_ids, rejected_labels = self._tokenize_final_response(rejected)
            chosen_result, rejected_result = self._truncate_pair(
                chosen_ids,
                chosen_labels,
                rejected_ids,
                rejected_labels,
            )
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"DPO row {source_index}: {exc}") from exc

        chosen_ids, chosen_labels, chosen_truncated, chosen_answer_truncated = (
            chosen_result
        )
        rejected_ids, rejected_labels, rejected_truncated, rejected_answer_truncated = (
            rejected_result
        )
        return {
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
            "source_index": torch.tensor(source_index, dtype=torch.long),
            "chosen_response_tokens": torch.tensor(
                sum(label != -100 for label in chosen_labels[1:]), dtype=torch.long
            ),
            "rejected_response_tokens": torch.tensor(
                sum(label != -100 for label in rejected_labels[1:]), dtype=torch.long
            ),
            "chosen_truncated": torch.tensor(int(chosen_truncated), dtype=torch.long),
            "rejected_truncated": torch.tensor(
                int(rejected_truncated), dtype=torch.long
            ),
            "chosen_answer_truncated": torch.tensor(
                int(chosen_answer_truncated), dtype=torch.long
            ),
            "rejected_answer_truncated": torch.tensor(
                int(rejected_answer_truncated), dtype=torch.long
            ),
        }


class DPODataCollator:
    """Right-pad pairs and concatenate chosen rows before rejected rows."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        if pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty feature list")
        ordered = [
            *(feature["chosen_input_ids"] for feature in features),
            *(feature["rejected_input_ids"] for feature in features),
        ]
        ordered_labels = [
            *(feature["chosen_labels"] for feature in features),
            *(feature["rejected_labels"] for feature in features),
        ]
        max_length = max(tensor.numel() for tensor in ordered)
        multiple = self.pad_to_multiple_of
        padded_length = ((max_length + multiple - 1) // multiple) * multiple
        batch_size = len(ordered)
        input_ids = torch.full(
            (batch_size, padded_length), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full((batch_size, padded_length), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, padded_length), dtype=torch.long)
        for row, (tokens, targets) in enumerate(zip(ordered, ordered_labels)):
            length = tokens.numel()
            input_ids[row, :length] = tokens
            labels[row, :length] = targets
            attention_mask[row, :length] = 1

        pair_count = len(features)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source_indices": torch.stack(
                [feature["source_index"] for feature in features]
            ),
            "chosen_response_tokens": torch.stack(
                [feature["chosen_response_tokens"] for feature in features]
            ),
            "rejected_response_tokens": torch.stack(
                [feature["rejected_response_tokens"] for feature in features]
            ),
            "chosen_sequence_tokens": attention_mask[:pair_count].sum(dim=1),
            "rejected_sequence_tokens": attention_mask[pair_count:].sum(dim=1),
            "chosen_truncated": torch.stack(
                [feature["chosen_truncated"] for feature in features]
            ),
            "rejected_truncated": torch.stack(
                [feature["rejected_truncated"] for feature in features]
            ),
            "chosen_answer_truncated": torch.stack(
                [feature["chosen_answer_truncated"] for feature in features]
            ),
            "rejected_answer_truncated": torch.stack(
                [feature["rejected_answer_truncated"] for feature in features]
            ),
        }


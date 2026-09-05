#!/usr/bin/env python3
"""JSONL dataset for miniLLM causal-language-model pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


EXPECTED_SPECIAL_TOKENS = {
    "pad_token_id": 0,
    "unk_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
}


def validate_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    expected_vocab_size: int = 8192,
) -> None:
    """Fail early when model/tokenizer IDs would be incompatible."""

    actual_size = len(tokenizer)
    if actual_size != expected_vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary is {actual_size}, expected {expected_vocab_size}. "
            "Use the tokenizer exported with the input model or change "
            "model vocab_size deliberately."
        )
    mismatches = []
    for attribute, expected in EXPECTED_SPECIAL_TOKENS.items():
        actual = getattr(tokenizer, attribute, None)
        if actual != expected:
            mismatches.append(f"{attribute}={actual!r} (expected {expected})")
    if mismatches:
        raise ValueError("Tokenizer special-token mismatch: " + ", ".join(mismatches))


class PretrainDataset(Dataset[dict[str, torch.Tensor]]):
    """Load ``{"text": ...}`` JSONL and produce fixed-length CLM samples.

    Validation rows are selected deterministically by index, so no second data
    file or in-memory random split is required.  With ``val_ratio=0.001``, every
    1000th source row belongs to validation and all other rows belong to train.
    """

    def __init__(
        self,
        data_paths: str | Path | Sequence[str | Path],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 512,
        split: str = "train",
        val_ratio: float = 0.001,
        expected_vocab_size: int = 8192,
    ) -> None:
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be one of: train, validation, all")
        if max_seq_len < 3:
            raise ValueError("max_seq_len must be at least 3")
        if split != "all" and not 0.0 < val_ratio < 0.5:
            raise ValueError("val_ratio must be between 0 and 0.5")

        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        self.data_paths = [Path(path).expanduser().resolve() for path in data_paths]
        missing = [str(path) for path in self.data_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Pretraining data file(s) not found: " + ", ".join(missing))
        empty = [str(path) for path in self.data_paths if path.stat().st_size == 0]
        if empty:
            raise ValueError("Pretraining data file(s) are empty: " + ", ".join(empty))

        validate_tokenizer(tokenizer, expected_vocab_size)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.val_stride = max(2, round(1.0 / val_ratio)) if split != "all" else 0

        self.samples = load_dataset(
            "json",
            data_files=[str(path) for path in self.data_paths],
            split="train",
            keep_in_memory=False,
        )
        if "text" not in self.samples.column_names:
            raise ValueError(
                "Pretraining JSONL must contain a 'text' field; found columns: "
                + ", ".join(self.samples.column_names)
            )
        if len(self.samples) == 0:
            raise ValueError("Pretraining dataset contains no rows")

    def __len__(self) -> int:
        total = len(self.samples)
        if self.split == "all":
            return total
        validation_count = (total + self.val_stride - 1) // self.val_stride
        if self.split == "validation":
            return validation_count
        return total - validation_count

    def _source_index(self, index: int) -> int:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)

        if self.split == "all":
            return index
        if self.split == "validation":
            return index * self.val_stride

        # Validation owns 0, stride, 2*stride, ...; map compact train indices
        # to every remaining source row without allocating a huge index list.
        group, offset = divmod(index, self.val_stride - 1)
        return group * self.val_stride + 1 + offset

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = self._source_index(index)
        text = self.samples[source_index]["text"]
        if not isinstance(text, str):
            raise TypeError(
                f"Row {source_index} has non-string 'text': {type(text).__name__}"
            )

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        token_ids = token_ids[: self.max_seq_len - 2]
        token_ids = [self.tokenizer.bos_token_id, *token_ids, self.tokenizer.eos_token_id]

        valid_length = len(token_ids)
        input_ids = torch.full(
            (self.max_seq_len,),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        input_ids[:valid_length] = torch.tensor(token_ids, dtype=torch.long)

        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        attention_mask[:valid_length] = 1
        labels = input_ids.clone()
        labels[valid_length:] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

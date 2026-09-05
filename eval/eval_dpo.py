#!/usr/bin/env python3
"""Evaluate a miniLLM policy against its frozen SFT reference on DPO pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dpo_dataset import DPODataCollator, DPODataset  # noqa: E402
from dataset.lm_dataset import validate_tokenizer  # noqa: E402
from eval.eval_sft import resolve_device, resolve_dtype  # noqa: E402
from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM  # noqa: E402
from trainer.train_dpo import (  # noqa: E402
    DPO_BETA,
    DPO_LABEL_SMOOTHING,
    evaluate,
)
from trainer.trainer_utils import (  # noqa: E402
    DistributedContext,
    ensure_tokenizer_matches_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate miniLLM DPO preferences.")
    parser.add_argument(
        "--policy-path", type=Path, default=PROJECT_ROOT / "out" / "dpo"
    )
    parser.add_argument(
        "--reference-path", type=Path, default=PROJECT_ROOT / "out" / "sft"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Tokenizer directory; defaults to --policy-path.",
    )
    parser.add_argument(
        "--data-path",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "dataset" / "rl" / "dpo.jsonl"],
    )
    parser.add_argument("--split", choices=["validation", "all"], default="validation")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--beta", type=float, default=DPO_BETA)
    parser.add_argument(
        "--label-smoothing", type=float, default=DPO_LABEL_SMOOTHING
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--attention-backend", choices=["eager", "auto", "math"], default="eager"
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_model(
    path: Path,
    device: torch.device,
    attention_backend: str,
) -> MiniLLMForCausalLM:
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Model config not found: {path / 'config.json'}")
    config = MiniLLMConfig.from_pretrained(path, local_files_only=True)
    config.attention_backend = attention_backend
    config.use_cache = False
    model = MiniLLMForCausalLM.from_pretrained(
        path, config=config, local_files_only=True
    )
    return model.to(device)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_seq_len < 32 or args.beta <= 0:
        raise ValueError("invalid batch size, sequence length, or beta")
    if args.num_workers < 0 or args.eval_samples < 0:
        raise ValueError("worker and sample limits cannot be negative")
    if not 0.0 < args.val_ratio < 0.5:
        raise ValueError("--val-ratio must be between 0 and 0.5")
    if not 0.0 <= args.label_smoothing < 0.5:
        raise ValueError("--label-smoothing must be in [0, 0.5)")

    policy_path = project_path(args.policy_path)
    reference_path = project_path(args.reference_path)
    tokenizer_path = (
        policy_path
        if args.tokenizer_path is None
        else project_path(args.tokenizer_path)
    )
    data_paths = [project_path(path) for path in args.data_path]
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    ensure_tokenizer_matches_models(
        tokenizer_path,
        {
            "DPO policy": policy_path,
            "DPO reference": reference_path,
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, use_fast=True
    )
    template_path = tokenizer_path / "chat_template.jinja"
    if template_path.is_file():
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    if not tokenizer.chat_template:
        raise ValueError(f"Tokenizer chat template not found in {tokenizer_path}")
    validate_tokenizer(tokenizer, expected_vocab_size=len(tokenizer))

    dataset: Any = DPODataset(
        data_paths=data_paths,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        split=args.split,
        val_ratio=args.val_ratio,
        expected_vocab_size=len(tokenizer),
        seed=args.seed,
    )
    if args.eval_samples:
        dataset = Subset(dataset, range(min(args.eval_samples, len(dataset))))
    collator = DPODataCollator(
        tokenizer.pad_token_id, pad_to_multiple_of=args.pad_to_multiple_of
    )
    loader_options: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collator,
    }
    if args.num_workers > 0:
        loader_options["multiprocessing_context"] = "spawn"
    loader = DataLoader(dataset, **loader_options)
    if len(loader) == 0:
        raise ValueError("Evaluation DataLoader is empty")

    policy_model = load_model(policy_path, device, args.attention_backend)
    reference_model = load_model(reference_path, device, args.attention_backend)
    policy_model.eval()
    reference_model.eval()
    reference_model.requires_grad_(False)
    context = DistributedContext(
        device=device,
        rank=0,
        local_rank=0,
        world_size=1,
        distributed=False,
    )
    metrics = evaluate(
        policy_model,
        reference_model,
        loader,
        context,
        dtype,
        args.beta,
        args.label_smoothing,
    )
    result = {
        "policy_path": str(policy_path),
        "reference_path": str(reference_path),
        "data_paths": [str(path) for path in data_paths],
        "split": args.split,
        "beta": args.beta,
        "label_smoothing": args.label_smoothing,
        **metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

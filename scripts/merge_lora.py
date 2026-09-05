#!/usr/bin/env python3
"""Merge a miniLLM LoRA adapter into a standalone Transformers model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from model.model_lora import load_adapter, merge_lora  # noqa: E402
from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM  # noqa: E402
from trainer.trainer_utils import ensure_tokenizer_matches_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge miniLLM LoRA weights.")
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument(
        "--no-verify-base",
        action="store_true",
        help="Skip the adapter/base SHA-256 check (unsafe).",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    base_path = resolve_path(args.base_model_path)
    adapter_path = resolve_path(args.adapter_path)
    output_dir = resolve_path(args.output_dir)
    if output_dir in {base_path, adapter_path}:
        raise ValueError("--output-dir must not overwrite the base model or adapter")
    if not (base_path / "config.json").is_file():
        raise FileNotFoundError(f"Base model config not found: {base_path / 'config.json'}")

    config = MiniLLMConfig.from_pretrained(base_path, local_files_only=True)
    config.use_cache = False
    model = MiniLLMForCausalLM.from_pretrained(
        base_path, config=config, local_files_only=True
    ).eval()
    adapter_config = load_adapter(
        model,
        adapter_path,
        base_model_path=base_path,
        verify_base=not args.no_verify_base,
    )

    # Verify the algebra on deterministic token IDs before writing a full model.
    length = min(8, config.max_position_embeddings)
    input_ids = torch.arange(length, dtype=torch.long).unsqueeze(0) % config.vocab_size
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        adapter_logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits.float()
    merged_layers = merge_lora(model, unload=True)
    with torch.no_grad():
        merged_logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits.float()
    max_difference = (adapter_logits - merged_logits).abs().max().item()
    if max_difference > 1e-4:
        raise RuntimeError(
            f"LoRA merge parity check failed: max logit difference={max_difference:.6g}"
        )

    dtype = getattr(torch, args.dtype)
    model = model.to(dtype=dtype)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer_source = (
        resolve_path(args.tokenizer_path)
        if args.tokenizer_path is not None
        else adapter_path
    )
    if not (tokenizer_source / "tokenizer_config.json").is_file():
        tokenizer_source = base_path
    ensure_tokenizer_matches_models(
        tokenizer_source,
        {"LoRA base model": base_path},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, local_files_only=True, use_fast=True
    )
    tokenizer.save_pretrained(output_dir)
    print(f"Base model     : {base_path}")
    print(f"Adapter        : {adapter_path}")
    print(f"LoRA rank      : {adapter_config.rank}")
    print(f"Merged layers  : {len(merged_layers)}")
    print(f"Parity max diff: {max_difference:.6g}")
    print(f"Output model   : {output_dir}")


if __name__ == "__main__":
    main()

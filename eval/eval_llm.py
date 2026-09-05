#!/usr/bin/env python3
"""Evaluate a pretrained miniLLM with validation loss and text completion."""

from __future__ import annotations

import argparse
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lm_dataset import PretrainDataset, validate_tokenizer  # noqa: E402
from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM  # noqa: E402
from trainer.trainer_utils import (  # noqa: E402
    ensure_config_tokenizer_fingerprint,
    ensure_tokenizer_matches_models,
    tokenizer_fingerprint,
)


DEFAULT_PROMPTS = [
    "中国的首都是",
    "人工智能的发展",
    "The history of computer science",
    "def quick_sort(",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained miniLLM.")
    parser.add_argument(
        "--mode", choices=["loss", "generate", "both"], default="both"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "out" / "pretrain",
        help="Transformers-format directory exported by train_pretrain.py.",
    )
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="Full training checkpoint such as checkpoints/pretrain/latest.pt.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Tokenizer directory; defaults to --model-path.",
    )
    parser.add_argument(
        "--data-path",
        nargs="*",
        type=Path,
        default=[PROJECT_ROOT / "dataset" / "pretrain" / "pretrain.jsonl"],
        help="Required for loss mode; JSONL files with a string 'text' field.",
    )
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.001)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda" or name == "float32":
        return torch.float32
    if name == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype != torch.float32:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def load_checkpoint_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(args: argparse.Namespace, device: torch.device) -> MiniLLMForCausalLM:
    if args.checkpoint is not None:
        checkpoint = load_checkpoint_file(project_path(args.checkpoint))
        if "config" not in checkpoint or "model" not in checkpoint:
            raise ValueError("Checkpoint must contain 'config' and 'model' entries")
        config = MiniLLMConfig(**checkpoint["config"])
        model = MiniLLMForCausalLM(config)
        model.load_state_dict(checkpoint["model"], strict=True)
        print(f"Loaded checkpoint: {project_path(args.checkpoint)}")
    else:
        model_path = project_path(args.model_path)
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"Exported model config not found: {model_path / 'config.json'}"
            )
        model = MiniLLMForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
        )
        print(f"Loaded model     : {model_path}")
    return model.to(device).eval()


@torch.no_grad()
def evaluate_loss(
    model: MiniLLMForCausalLM,
    tokenizer: Any,
    data_paths: list[Path],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, int]:
    if not data_paths:
        raise ValueError("--data-path is required when --mode is loss or both")
    dataset: Any = PretrainDataset(
        data_paths,
        tokenizer,
        max_seq_len=args.max_seq_len,
        split="validation",
        val_ratio=args.val_ratio,
        expected_vocab_size=model.config.vocab_size,
    )
    if args.eval_samples:
        dataset = Subset(dataset, range(min(args.eval_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with autocast_context(device, dtype):
            output = model(**batch)
        valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
        # MoE 的 Router 辅助损失不属于语言困惑度，只统计 next-token loss。
        lm_loss = output.lm_loss if output.lm_loss is not None else output.loss
        total_loss += lm_loss.float().item() * valid_tokens
        total_tokens += valid_tokens
    mean_loss = total_loss / max(1, total_tokens)
    return mean_loss, math.exp(min(mean_loss, 20.0)), total_tokens


@torch.no_grad()
def generate_text(
    model: MiniLLMForCausalLM,
    tokenizer: Any,
    prompts: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    print("\nText completion")
    print("-" * 72)
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor(
            [[tokenizer.bos_token_id, *prompt_ids]],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        do_sample = args.temperature > 0
        generation_options = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": args.repetition_penalty,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": False,
        }
        if do_sample:
            generation_options.update(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
        eval_dtype = resolve_dtype(args.dtype, device)
        with autocast_context(device, eval_dtype):
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_options,
            )
        completion_ids = generated[0, input_ids.shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        print(f"Prompt    : {prompt}")
        print(f"Completion: {completion}")
        print("-" * 72)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model_path = project_path(args.model_path)
    tokenizer_path = (
        model_path
        if args.tokenizer_path is None
        else project_path(args.tokenizer_path)
    )
    if args.checkpoint is None:
        fingerprint = ensure_tokenizer_matches_models(
            tokenizer_path,
            {"pretrained model": model_path},
        )
    else:
        fingerprint = tokenizer_fingerprint(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        use_fast=True,
    )
    validate_tokenizer(tokenizer, expected_vocab_size=8192)
    model = load_model(args, device)
    ensure_config_tokenizer_fingerprint(
        model.config,
        fingerprint,
        label="pretrained model",
    )
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError(
            f"Tokenizer size {len(tokenizer)} != model vocab_size {model.config.vocab_size}"
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Tokenizer        : {tokenizer_path}")
    print(f"Device           : {device}")
    architecture = "Dense"
    active_parameter_count = parameter_count
    if model.config.use_moe:
        architecture = (
            f"MoE ({model.config.num_experts} Experts, "
            f"Top-{model.config.num_experts_per_tok})"
        )
        one_expert_group = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if ".mlp.experts.0." in name
        )
        active_parameter_count -= one_expert_group * (
            model.config.num_experts - model.config.num_experts_per_tok
        )
    print(f"Architecture     : {architecture}")
    print(f"Parameters       : {parameter_count:,} total")
    if model.config.use_moe:
        print(f"Active parameters: {active_parameter_count:,} per token")

    if args.mode in {"loss", "both"}:
        data_paths = [project_path(path) for path in args.data_path]
        loss, perplexity, tokens = evaluate_loss(
            model, tokenizer, data_paths, args, device, dtype
        )
        print(f"Validation tokens: {tokens:,}")
        print(f"Validation loss  : {loss:.6f}")
        print(f"Perplexity       : {perplexity:.4f}")

    if args.mode in {"generate", "both"}:
        generate_text(
            model,
            tokenizer,
            args.prompt or DEFAULT_PROMPTS,
            args,
            device,
        )


if __name__ == "__main__":
    main()

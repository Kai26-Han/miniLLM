#!/usr/bin/env python3
"""Evaluate an SFT miniLLM with assistant-only loss and chat generation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lm_dataset import validate_tokenizer  # noqa: E402
from dataset.sft_dataset import SFTDataCollator, SFTDataset  # noqa: E402
from model.model_lora import load_adapter, lora_parameters  # noqa: E402
from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM  # noqa: E402
from trainer.trainer_utils import (  # noqa: E402
    ensure_config_tokenizer_fingerprint,
    ensure_tokenizer_matches_models,
    tokenizer_fingerprint,
)


DEFAULT_PROMPTS = (
    "请用三句话介绍一下北京。",
    "解释为什么天空通常是蓝色的。",
    "Write a Python function that checks whether a string is a palindrome.",
    "把“人工智能正在改变软件开发方式”翻译成英文。",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SFT miniLLM.")
    parser.add_argument(
        "--mode", choices=["loss", "generate", "both"], default="both"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "out" / "sft",
    )
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="Optional LoRA adapter directory to apply on top of --model-path.",
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Evaluate base loss before applying the adapter, then evaluate both.",
    )
    parser.add_argument(
        "--no-verify-adapter-base",
        action="store_true",
        help="Skip the adapter/base SHA-256 check (unsafe).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Tokenizer directory; defaults to --model-path.",
    )
    parser.add_argument(
        "--data-path",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "dataset" / "sft" / "sft_mini.jsonl"],
    )
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--val-ratio", type=float, default=0.001)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--empty-think-ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument(
        "--open-thinking", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--tools-json",
        type=Path,
        help="Optional JSON file containing a list of function definitions.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
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
        checkpoint_path = project_path(args.checkpoint)
        checkpoint = load_checkpoint_file(checkpoint_path)
        if "config" not in checkpoint or "model" not in checkpoint:
            raise ValueError("Checkpoint must contain 'config' and 'model' entries")
        model = MiniLLMForCausalLM(MiniLLMConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"], strict=True)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        model_path = project_path(args.model_path)
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"Model config not found: {model_path / 'config.json'}")
        model = MiniLLMForCausalLM.from_pretrained(
            model_path, local_files_only=True
        )
        print(f"Loaded model     : {model_path}")
    model.config.use_cache = False
    return model.to(device).eval()


@torch.no_grad()
def evaluate_loss(
    model: MiniLLMForCausalLM,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    dataset: Any = SFTDataset(
        [project_path(path) for path in args.data_path],
        tokenizer,
        max_seq_len=args.max_seq_len,
        split="validation",
        val_ratio=args.val_ratio,
        expected_vocab_size=model.config.vocab_size,
        empty_think_ratio=args.empty_think_ratio,
        system_prompt_ratio=0.0,
        seed=args.seed,
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
        collate_fn=SFTDataCollator(tokenizer.pad_token_id),
    )
    total_loss = 0.0
    total_assistant_tokens = 0
    total_sequence_tokens = 0
    total_slots = 0
    truncated = 0
    answer_truncated = 0
    samples = 0
    for batch in loader:
        batch.pop("source_indices")
        assistant_tokens = batch.pop("assistant_tokens")
        truncated += int(batch.pop("truncated").sum().item())
        answer_truncated += int(batch.pop("answer_truncated").sum().item())
        sequence_tokens = batch.pop("sequence_tokens")
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with autocast_context(device, dtype):
            output = model(**batch)
        valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
        lm_loss = output.lm_loss if output.lm_loss is not None else output.loss
        total_loss += lm_loss.float().item() * valid_tokens
        total_assistant_tokens += int(assistant_tokens.sum().item())
        total_sequence_tokens += int(sequence_tokens.sum().item())
        total_slots += batch["input_ids"].numel()
        samples += batch["input_ids"].shape[0]
    mean_loss = total_loss / max(1, total_assistant_tokens)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "assistant_tokens": float(total_assistant_tokens),
        "padding_efficiency": total_sequence_tokens / max(1, total_slots),
        "truncated_ratio": truncated / max(1, samples),
        "answer_truncated_ratio": answer_truncated / max(1, samples),
    }


def load_tools(path: Path | None) -> list[Any] | None:
    if path is None:
        return None
    resolved = project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        tools = json.load(handle)
    if not isinstance(tools, list):
        raise ValueError("--tools-json must contain a JSON array")
    return tools


@torch.no_grad()
def generate_chats(
    model: MiniLLMForCausalLM,
    tokenizer: Any,
    prompts: list[str],
    tools: list[Any] | None,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    print("\nChat generation")
    print("-" * 80)
    for prompt in prompts:
        messages: list[dict[str, str]] = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": prompt})
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
            open_thinking=args.open_thinking,
        )
        encoded = tokenizer(chat_text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        do_sample = args.temperature > 0
        options: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": args.repetition_penalty,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": False,
        }
        if do_sample:
            options.update(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
        with autocast_context(device, dtype):
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **options,
            )
        completion_ids = generated[0, input_ids.shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
        completion = completion.replace("<|im_end|>", "").strip()
        if args.open_thinking:
            completion = "<think>\n" + completion
        print(f"User      : {prompt}")
        print(f"Assistant : {completion}")
        print("-" * 80)


def main() -> None:
    args = parse_args()
    if args.max_seq_len < 32 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid sequence length, batch size, or worker count")
    if args.eval_samples < 0 or not 0.0 < args.val_ratio < 0.5:
        raise ValueError("invalid evaluation sample limit or validation ratio")
    if not 0.0 <= args.empty_think_ratio <= 1.0:
        raise ValueError("--empty-think-ratio must be in [0, 1]")
    if args.compare_base and args.adapter_path is None:
        raise ValueError("--compare-base requires --adapter-path")
    if args.compare_base and args.mode == "generate":
        raise ValueError("--compare-base requires --mode loss or --mode both")
    if args.adapter_path is not None and args.checkpoint is not None:
        raise ValueError(
            "LoRA evaluation requires a Transformers --model-path, not --checkpoint"
        )
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
            {"SFT model": model_path},
        )
    else:
        fingerprint = tokenizer_fingerprint(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, use_fast=True
    )
    template_path = tokenizer_path / "chat_template.jinja"
    if template_path.is_file():
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    if not tokenizer.chat_template:
        raise ValueError(f"Tokenizer chat template not found in {tokenizer_path}")
    validate_tokenizer(tokenizer, expected_vocab_size=len(tokenizer))
    model = load_model(args, device)
    ensure_config_tokenizer_fingerprint(
        model.config,
        fingerprint,
        label="SFT model",
    )
    if model.config.vocab_size != len(tokenizer):
        raise ValueError(
            f"Model vocab_size={model.config.vocab_size} but tokenizer has {len(tokenizer)} tokens"
        )
    if args.compare_base and args.mode in {"loss", "both"}:
        base_result = evaluate_loss(model, tokenizer, args, device, dtype)
        print("\nBase model evaluation")
        print(f"Assistant loss   : {base_result['loss']:.6f}")
        print(f"Assistant PPL    : {base_result['perplexity']:.3f}")

    adapter_config = None
    if args.adapter_path is not None:
        adapter_path = project_path(args.adapter_path)
        adapter_tokenizer = adapter_path / "tokenizer.json"
        if adapter_tokenizer.is_file():
            adapter_fingerprint = tokenizer_fingerprint(adapter_path)
            if adapter_fingerprint != fingerprint:
                raise ValueError(
                    "LoRA adapter tokenizer does not match its SFT base model: "
                    f"adapter={adapter_fingerprint}, base={fingerprint}"
                )
        adapter_config = load_adapter(
            model,
            adapter_path,
            base_model_path=project_path(args.model_path),
            verify_base=not args.no_verify_adapter_base,
        )
        model.eval()
        print(f"Loaded adapter   : {adapter_path}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Tokenizer        : {tokenizer_path}")
    print(f"Device           : {device}")
    print(f"Precision        : {str(dtype).removeprefix('torch.')}")
    print(f"Architecture     : {'MoE' if model.config.use_moe else 'Dense'}")
    print(f"Parameters       : {parameter_count:,}")
    if adapter_config is not None:
        trainable_adapter = sum(parameter.numel() for parameter in lora_parameters(model))
        print(f"LoRA rank/alpha  : {adapter_config.rank}/{adapter_config.alpha:g}")
        print(f"LoRA parameters  : {trainable_adapter:,}")

    if args.mode in {"loss", "both"}:
        result = evaluate_loss(model, tokenizer, args, device, dtype)
        if adapter_config is not None:
            print("\nBase + LoRA evaluation")
        print(f"Assistant loss   : {result['loss']:.6f}")
        print(f"Assistant PPL    : {result['perplexity']:.3f}")
        print(f"Assistant tokens : {int(result['assistant_tokens']):,}")
        print(f"Padding efficiency: {result['padding_efficiency']:.1%}")
        print(f"Truncated samples: {result['truncated_ratio']:.1%}")
        print(f"Clipped answers  : {result['answer_truncated_ratio']:.1%}")
    if args.mode in {"generate", "both"}:
        generate_chats(
            model,
            tokenizer,
            args.prompt or list(DEFAULT_PROMPTS),
            load_tools(args.tools_json),
            args,
            device,
            dtype,
        )


if __name__ == "__main__":
    main()

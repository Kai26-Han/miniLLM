#!/usr/bin/env python3
"""Parameter-efficient supervised fine-tuning for miniLLM with LoRA.

The base model is frozen. Checkpoints and final exports contain adapter tensors
only, so the original SFT model is never overwritten.

Recommended usage from the project root::

    python trainer/train_lora.py \
        --model-path out/sft \
        --data-path dataset/lora/lora_medical.jsonl \
        --adapter-name medical
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, PreTrainedTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lm_dataset import validate_tokenizer  # noqa: E402
from model.model_lora import (  # noqa: E402
    LoRAConfig,
    adapter_state_dict,
    apply_lora,
    count_parameters,
    load_adapter_state_dict,
    lora_parameters,
    model_directory_sha256,
    resolve_target_modules,
    save_adapter,
)
from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM  # noqa: E402
from trainer.train_sft import (  # noqa: E402
    autocast_context,
    build_optimizer,
    configure_attention_backend,
    ensure_finite_forward,
    ensure_finite_model_state,
    evaluate,
    make_dataloaders,
    pop_batch_stats,
    register_numerics_hooks,
    resolve_amp,
)
from trainer.trainer_utils import (  # noqa: E402
    DistributedContext,
    ExperimentTracker,
    atomic_torch_save,
    build_cosine_scheduler,
    cleanup_distributed,
    distributed_sum,
    ensure_checkpoint_tokenizer_fingerprint,
    ensure_tokenizer_matches_models,
    init_experiment_tracker,
    resolve_resume_path,
    seed_everything,
    setup_distributed,
    unwrap_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for miniLLM.")
    parser.add_argument(
        "--data-path",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "dataset" / "lora" / "lora_medical.jsonl"],
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "out" / "sft",
        help="SFT Transformers directory used as the frozen base model.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Tokenizer directory; defaults to --model-path.",
    )
    parser.add_argument("--adapter-name", type=str, default="medical")
    parser.add_argument(
        "--save-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "lora"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "out" / "lora"
    )

    parser.add_argument(
        "--target-preset",
        choices=["attention", "all-linear"],
        default="attention",
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help="Explicit Linear module suffixes; overrides --target-preset.",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--empty-think-ratio", type=float, default=0.0)
    parser.add_argument("--system-prompt-ratio", type=float, default=0.2)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume a LoRA checkpoint, or use save-dir/adapter-name/latest.pt.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--attention-backend",
        choices=["eager", "auto", "math"],
        default="eager",
    )
    parser.add_argument("--debug-numerics", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument(
        "--tracker", choices=["none", "swanlab", "wandb"], default="none"
    )
    parser.add_argument("--tracker-project", type=str, default="miniLLM-LoRA")
    parser.add_argument("--tracker-run-name", type=str, default=None)
    parser.add_argument("--tracker-entity", type=str, default=None)
    parser.add_argument("--tracker-group", type=str, default=None)
    parser.add_argument("--tracker-tags", nargs="*", default=[])
    parser.add_argument(
        "--tracker-mode", choices=["online", "offline"], default="online"
    )
    parser.add_argument(
        "--tracker-log-dir", type=Path, default=PROJECT_ROOT / "logs" / "lora"
    )
    parser.add_argument("--tracker-run-id", type=str, default=None)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_tokenizer(path: Path) -> Any:
    """Load tokenizers exported by both Transformers 4.x and 5.x."""

    options = {"local_files_only": True, "use_fast": True}
    try:
        return AutoTokenizer.from_pretrained(path, **options)
    except ValueError as exc:
        if "Tokenizer class TokenizersBackend" not in str(exc):
            raise
        # Transformers 5 may write ``TokenizersBackend`` into the config,
        # while older releases only expose its compatible public class name.
        return PreTrainedTokenizerFast.from_pretrained(path, **options)


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "epochs",
        "batch_size",
        "accumulation_steps",
        "learning_rate",
        "max_seq_len",
        "pad_to_multiple_of",
        "log_interval",
        "eval_interval",
        "save_interval",
        "lora_rank",
        "lora_alpha",
    )
    for field in positive:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if not args.adapter_name.strip() or Path(args.adapter_name).name != args.adapter_name:
        raise ValueError("--adapter-name must be a non-empty directory name")
    if args.max_steps < 0 or args.max_train_samples < 0 or args.eval_samples < 0:
        raise ValueError("sample and step limits cannot be negative")
    if args.max_seq_len < 32 or args.num_workers < 0:
        raise ValueError("invalid sequence length or worker count")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if not 0.0 < args.min_learning_rate <= args.learning_rate:
        raise ValueError("--min-learning-rate must be > 0 and <= learning-rate")
    if not 0.0 < args.val_ratio < 0.5:
        raise ValueError("--val-ratio must be between 0 and 0.5")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in [0, 1)")
    for field in ("empty_think_ratio", "system_prompt_ratio"):
        if not 0.0 <= getattr(args, field) <= 1.0:
            raise ValueError(f"--{field.replace('_', '-')} must be in [0, 1]")
    if args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("weight decay cannot be negative and grad clip must be positive")
    if args.debug_numerics and args.compile:
        raise ValueError("--debug-numerics cannot be combined with --compile")


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_lora_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    *,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    args: argparse.Namespace,
    lora_config: LoRAConfig,
    tracker_state: dict[str, Any],
) -> None:
    payload = {
        "adapter": adapter_state_dict(model),
        "adapter_config": lora_config.to_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "args": vars(args).copy(),
        "rng_state": _rng_state(),
        "tracker": tracker_state,
    }
    atomic_torch_save(payload, path)


def load_lora_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    expected_config: LoRAConfig,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    saved_config = LoRAConfig.from_dict(checkpoint["adapter_config"])
    comparable_saved = (
        saved_config.rank,
        saved_config.alpha,
        saved_config.dropout,
        saved_config.target_modules,
        saved_config.base_model_sha256,
    )
    comparable_expected = (
        expected_config.rank,
        expected_config.alpha,
        expected_config.dropout,
        expected_config.target_modules,
        expected_config.base_model_sha256,
    )
    if comparable_saved != comparable_expected:
        raise ValueError(
            "Resume checkpoint LoRA configuration differs from command line: "
            f"saved={comparable_saved}, requested={comparable_expected}"
        )
    load_adapter_state_dict(model, checkpoint["adapter"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    _restore_rng_state(checkpoint.get("rng_state"))
    return checkpoint


def print_setup(
    args: argparse.Namespace,
    model: torch.nn.Module,
    injected: tuple[str, ...],
    context: DistributedContext,
    train_batches: int,
    validation_batches: int,
    total_steps: int,
    amp_dtype: torch.dtype,
) -> None:
    if not context.is_main:
        return
    raw_model = unwrap_model(model)
    total = sum(parameter.numel() for parameter in raw_model.parameters())
    trainable = count_parameters(lora_parameters(raw_model))
    print(f"Project root    : {PROJECT_ROOT}")
    print(f"Frozen base     : {args.model_path}")
    print(f"Tokenizer       : {args.tokenizer_path}")
    print(f"Tokenizer SHA   : {args.tokenizer_fingerprint}")
    print(f"Adapter output  : {args.output_dir / args.adapter_name}")
    print(f"Architecture    : {'MoE' if raw_model.config.use_moe else 'Dense'}")
    print(f"Device          : {context.device} (world_size={context.world_size})")
    print(f"Precision       : {str(amp_dtype).removeprefix('torch.')}")
    target_names = resolve_target_modules(args.target_preset, args.target_modules)
    print(f"LoRA targets    : {', '.join(target_names)}")
    print(f"LoRA layers     : {len(injected)}")
    print(f"LoRA rank/alpha : {args.lora_rank}/{args.lora_alpha:g}")
    print(f"Trainable       : {trainable:,} / {total:,} ({trainable / total:.3%})")
    print(f"Global batch    : {args.batch_size * args.accumulation_steps * context.world_size}")
    print(f"Train/valid     : {train_batches:,} / {validation_batches:,} batches per rank")
    print(f"Update steps    : {total_steps:,}")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    validate_args(args)
    args.data_path = [project_path(path) for path in args.data_path]
    args.model_path = project_path(args.model_path)
    args.tokenizer_path = (
        args.model_path
        if args.tokenizer_path is None
        else project_path(args.tokenizer_path)
    )
    args.save_dir = project_path(args.save_dir) / args.adapter_name
    args.output_dir = project_path(args.output_dir)
    args.tracker_log_dir = project_path(args.tracker_log_dir)

    context = setup_distributed(args.device)
    tracker = ExperimentTracker()
    finish_state = "crashed"
    finish_error: str | None = "LoRA training stopped before completion."
    try:
        seed_everything(args.seed, context.rank)
        if context.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        configure_attention_backend(args.attention_backend, context)
        if not (args.model_path / "config.json").is_file():
            hint = (
                " Run trainer/train_sft.py first, or explicitly pass a valid "
                "Transformers model directory with --model-path."
            )
            raise FileNotFoundError(
                f"SFT base config not found: {args.model_path / 'config.json'}.{hint}"
            )

        args.tokenizer_fingerprint = ensure_tokenizer_matches_models(
            args.tokenizer_path,
            {"SFT base model": args.model_path},
        )

        tokenizer = load_tokenizer(args.tokenizer_path)
        template_path = args.tokenizer_path / "chat_template.jinja"
        if template_path.is_file():
            tokenizer.chat_template = template_path.read_text(encoding="utf-8")
        if not tokenizer.chat_template:
            raise ValueError(f"Tokenizer chat template not found in {args.tokenizer_path}")
        validate_tokenizer(tokenizer, expected_vocab_size=len(tokenizer))
        dataloader_bundle = make_dataloaders(args, tokenizer, context)
        if not isinstance(dataloader_bundle, tuple) or len(dataloader_bundle) < 3:
            raise TypeError(
                "make_dataloaders() must return at least "
                "(train_loader, validation_loader, train_sampler)"
            )
        # Some project revisions also return validation samplers or datasets.
        # LoRA only needs the first three values shared by all revisions.
        train_loader, validation_loader, train_sampler = dataloader_bundle[:3]

        model_config = MiniLLMConfig.from_pretrained(
            args.model_path, local_files_only=True
        )
        model_config.tokenizer_fingerprint = args.tokenizer_fingerprint
        model_config.attention_backend = args.attention_backend
        model: torch.nn.Module = MiniLLMForCausalLM.from_pretrained(
            args.model_path, config=model_config, local_files_only=True
        )
        if model.config.vocab_size != len(tokenizer):
            raise ValueError(
                f"Model vocab_size={model.config.vocab_size}, tokenizer={len(tokenizer)}"
            )
        model.config.use_cache = False
        model = model.to(context.device)
        ensure_finite_model_state(model, "LoRA base model")

        targets = resolve_target_modules(args.target_preset, args.target_modules)
        if context.is_main:
            print("Base fingerprint : computing SHA-256 of config and weight shards")
        if context.distributed:
            fingerprint_box: list[str | None] = [
                model_directory_sha256(args.model_path) if context.is_main else None
            ]
            torch.distributed.broadcast_object_list(fingerprint_box, src=0)
            base_fingerprint = fingerprint_box[0]
            if base_fingerprint is None:
                raise RuntimeError("Rank 0 did not broadcast the base model fingerprint")
        else:
            base_fingerprint = model_directory_sha256(args.model_path)
        lora_config = LoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=targets,
            base_model_name_or_path=str(args.model_path),
            base_model_sha256=base_fingerprint,
        )
        injected = apply_lora(model, lora_config)
        numeric_hooks = register_numerics_hooks(model)
        if args.gradient_checkpointing:
            model.model.gradient_checkpointing = True
        if args.compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("--compile requires PyTorch 2.x")
            model = torch.compile(model)
        if context.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        optimizer = build_optimizer(model, args)
        updates_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
        total_steps = args.max_steps or args.epochs * updates_per_epoch
        training_epochs = (
            math.ceil(total_steps / updates_per_epoch) if args.max_steps else args.epochs
        )
        scheduler = build_cosine_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_ratio=args.warmup_ratio,
            min_lr_ratio=args.min_learning_rate / args.learning_rate,
        )
        amp_dtype, needs_scaler = resolve_amp(args.dtype, context)
        scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)

        start_epoch = 0
        start_batch = 0
        global_step = 0
        checkpoint: dict[str, Any] = {}
        resume_path = resolve_resume_path(args.save_dir, args.resume)
        if resume_path is not None:
            checkpoint = load_lora_checkpoint(
                resume_path, model, optimizer, scheduler, scaler, lora_config
            )
            ensure_checkpoint_tokenizer_fingerprint(
                checkpoint, args.tokenizer_fingerprint
            )
            start_epoch = int(checkpoint.get("epoch", 0))
            start_batch = int(checkpoint.get("batch_in_epoch", 0))
            global_step = int(checkpoint.get("global_step", 0))
            if context.is_main:
                print(f"Resumed         : {resume_path} (step={global_step:,})")
        elif args.resume == "auto" and context.is_main:
            print("Resume          : no latest.pt found; starting a new run")

        checkpoint_tracker = checkpoint.get("tracker", {})
        restored_id = (
            checkpoint_tracker.get("run_id")
            if checkpoint_tracker.get("backend") == args.tracker
            else None
        )
        tracker = init_experiment_tracker(
            backend=args.tracker,
            context=context,
            project=args.tracker_project,
            run_name=args.tracker_run_name or f"miniLLM-LoRA-{args.adapter_name}",
            entity=args.tracker_entity,
            group=args.tracker_group,
            tags=args.tracker_tags,
            mode=args.tracker_mode,
            log_dir=args.tracker_log_dir,
            config=vars(args).copy(),
            run_id=args.tracker_run_id or restored_id,
            strict_resume=bool(restored_id and not args.tracker_run_id),
        )
        print_setup(
            args,
            model,
            injected,
            context,
            len(train_loader),
            len(validation_loader),
            total_steps,
            amp_dtype,
        )
        if global_step >= total_steps:
            raise ValueError(
                f"Checkpoint step {global_step} already reached total_steps={total_steps}"
            )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        stop_training = False
        interval_loss = 0.0
        interval_lm_loss = 0.0
        interval_aux_loss = 0.0
        interval_batches = 0
        interval_tokens = 0
        interval_start = time.perf_counter()
        latest_validation: dict[str, float] | None = None

        for epoch in range(start_epoch, training_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            skip_before = start_batch if epoch == start_epoch else 0
            last_batch_index = -1
            for batch_index, batch in enumerate(train_loader):
                last_batch_index = batch_index
                if batch_index < skip_before:
                    continue
                window_start = (batch_index // args.accumulation_steps) * args.accumulation_steps
                window_size = min(
                    args.accumulation_steps, len(train_loader) - window_start
                )
                should_update = (
                    (batch_index + 1) % args.accumulation_steps == 0
                    or batch_index + 1 == len(train_loader)
                )
                sync_context = (
                    nullcontext()
                    if should_update or not context.distributed
                    else model.no_sync()
                )
                stats = pop_batch_stats(batch)
                batch = {
                    key: value.to(context.device, non_blocking=True)
                    for key, value in batch.items()
                }
                with sync_context:
                    with autocast_context(context.device, amp_dtype):
                        output = model(**batch)
                        scaled_loss = output.loss / window_size
                    ensure_finite_forward(
                        output,
                        batch,
                        stats,
                        context=context,
                        phase="lora-train",
                        epoch=epoch,
                        batch_index=batch_index,
                        global_step=global_step,
                    )
                    if numeric_hooks and not args.debug_numerics:
                        for handle in numeric_hooks:
                            handle.remove()
                        numeric_hooks.clear()
                    if needs_scaler:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                interval_loss += output.loss.detach().float().item()
                interval_lm_loss += output.lm_loss.detach().float().item()
                if output.router_aux_loss is not None:
                    interval_aux_loss += output.router_aux_loss.detach().float().item()
                interval_batches += 1
                interval_tokens += int(stats["assistant_tokens"].sum().item())
                if not should_update:
                    continue

                if needs_scaler:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    lora_parameters(model),
                    args.grad_clip,
                    error_if_nonfinite=not needs_scaler,
                )
                if needs_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_interval == 0:
                    elapsed = max(time.perf_counter() - interval_start, 1e-6)
                    totals = torch.tensor(
                        [
                            interval_loss,
                            interval_lm_loss,
                            interval_aux_loss,
                            interval_batches,
                            interval_tokens,
                        ],
                        device=context.device,
                        dtype=torch.float64,
                    )
                    distributed_sum(totals, context)
                    if context.is_main:
                        batches = max(1.0, totals[3].item())
                        metrics = {
                            "train/loss": totals[0].item() / batches,
                            "train/lm_loss": totals[1].item() / batches,
                            "train/router_aux_loss": totals[2].item() / batches,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/grad_norm": float(grad_norm.detach().float().item()),
                            "train/assistant_tokens_per_second": totals[4].item() / elapsed,
                        }
                        print(
                            f"epoch={epoch + 1}/{training_epochs} "
                            f"step={global_step:,}/{total_steps:,} "
                            f"loss={metrics['train/loss']:.4f} "
                            f"lm={metrics['train/lm_loss']:.4f} "
                            f"aux={metrics['train/router_aux_loss']:.6f} "
                            f"lr={metrics['train/learning_rate']:.2e}"
                        )
                        tracker.log(metrics, step=global_step)
                    interval_loss = interval_lm_loss = interval_aux_loss = 0.0
                    interval_batches = interval_tokens = 0
                    interval_start = time.perf_counter()

                if global_step % args.eval_interval == 0:
                    latest_validation = evaluate(
                        model, validation_loader, context, amp_dtype
                    )
                    if context.is_main:
                        print(
                            f"validation step={global_step:,} "
                            f"loss={latest_validation['loss']:.4f} "
                            f"ppl={latest_validation['perplexity']:.2f}"
                        )
                        tracker.log(
                            {
                                f"validation/{key}": value
                                for key, value in latest_validation.items()
                            },
                            step=global_step,
                        )

                if global_step % args.save_interval == 0 and context.is_main:
                    save_lora_checkpoint(
                        args.save_dir / "latest.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch=epoch,
                        batch_in_epoch=batch_index + 1,
                        global_step=global_step,
                        args=args,
                        lora_config=lora_config,
                        tracker_state=tracker.state_dict(),
                    )
                    print(f"Checkpoint      : {args.save_dir / 'latest.pt'}")

                if global_step >= total_steps:
                    stop_training = True
                    break

            start_batch = 0
            if context.is_main:
                checkpoint_epoch = epoch if stop_training else epoch + 1
                checkpoint_batch = last_batch_index + 1 if stop_training else 0
                save_lora_checkpoint(
                    args.save_dir / "latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=checkpoint_epoch,
                    batch_in_epoch=checkpoint_batch,
                    global_step=global_step,
                    args=args,
                    lora_config=lora_config,
                    tracker_state=tracker.state_dict(),
                )
            if stop_training:
                break

        if context.distributed:
            torch.distributed.barrier()
        if context.is_main:
            adapter_dir = args.output_dir / args.adapter_name
            saved_config = save_adapter(
                model,
                adapter_dir,
                lora_config,
                tokenizer=tokenizer,
                base_model_path=args.model_path,
            )
            summary = {
                "global_step": global_step,
                "trainable_parameters": count_parameters(lora_parameters(model)),
                "adapter_config": saved_config.to_dict(),
                "latest_validation": latest_validation,
            }
            (adapter_dir / "training_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Training complete: {global_step:,} optimizer steps")
            print(f"Adapter         : {adapter_dir}")
            print(f"Checkpoint      : {args.save_dir / 'latest.pt'}")
        finish_state = "success"
        finish_error = None
    except KeyboardInterrupt:
        finish_state = "aborted"
        finish_error = "LoRA training interrupted by user."
        raise
    except BaseException:
        finish_state = "crashed"
        finish_error = traceback.format_exc()
        raise
    finally:
        try:
            tracker.finish(state=finish_state, error=finish_error)
        finally:
            cleanup_distributed(context)


if __name__ == "__main__":
    main()

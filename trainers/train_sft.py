#!/usr/bin/env python3
"""Full-parameter supervised fine-tuning for miniLLM.

The script loads an exported pretrained miniLLM, renders conversation JSONL,
and optimizes only assistant output tokens.  It reuses miniLLM's DDP, AMP,
checkpoint, tracker and Hugging Face export infrastructure.

Single GPU example (run from the project root):

    python trainer/train_sft.py \
        --model-path out/pretrain

Six GPU example:

    torchrun --nproc_per_node 6 trainer/train_sft.py \
        --model-path out/pretrain
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    DistributedSampler,
    RandomSampler,
    Sampler,
    Subset,
)
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lm_dataset import validate_tokenizer  # noqa: E402
from dataset.sft_dataset import SFTDataCollator, SFTDataset  # noqa: E402
from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM  # noqa: E402
from trainer.trainer_utils import (  # noqa: E402
    DistributedContext,
    ExperimentTracker,
    build_cosine_scheduler,
    cleanup_distributed,
    distributed_sum,
    ensure_checkpoint_tokenizer_fingerprint,
    ensure_tokenizer_matches_models,
    export_pretrained,
    init_experiment_tracker,
    load_checkpoint,
    resolve_resume_path,
    save_checkpoint,
    seed_everything,
    setup_distributed,
    unwrap_model,
)


STAT_KEYS = (
    "source_indices",
    "assistant_tokens",
    "truncated",
    "answer_truncated",
    "sequence_tokens",
)
ADAM_BETAS = (0.9, 0.95)


class SkipBatchSampler(Sampler[list[int]]):
    """Skip completed batches without loading or tokenizing their samples."""

    def __init__(self, batch_sampler: Sampler[list[int]]) -> None:
        self.batch_sampler = batch_sampler
        self.skip_batches = 0

    def set_skip_batches(self, skip_batches: int) -> None:
        if not 0 <= skip_batches <= len(self.batch_sampler):
            raise ValueError(
                f"skip_batches={skip_batches} is outside "
                f"[0, {len(self.batch_sampler)}]"
            )
        self.skip_batches = skip_batches

    def __iter__(self) -> Iterator[list[int]]:
        iterator = iter(self.batch_sampler)
        for _ in range(self.skip_batches):
            next(iterator)
        yield from iterator

    def __len__(self) -> int:
        return len(self.batch_sampler) - self.skip_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-parameter SFT for miniLLM.")
    parser.add_argument(
        "--data-path",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "dataset" / "sft" / "sft_mini.jsonl"],
        help="Conversation JSONL file(s); defaults to the curated mini SFT set.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "out" / "pretrain",
        help="Transformers directory exported by train_pretrain.py.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Tokenizer directory; defaults to --model-path.",
    )
    parser.add_argument(
        "--save-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "sft"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "out" / "sft"
    )

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--val-ratio", type=float, default=0.001)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--empty-think-ratio", type=float, default=0.2)
    parser.add_argument("--system-prompt-ratio", type=float, default=0.2)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="float32",
        help=(
            "Training precision. float32 is the stable SFT default; reduced "
            "precision remains available for separately validated runs."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume an SFT checkpoint, or use save-dir/latest.pt with --resume.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Trade compute for activation memory. Disabled by default to keep "
            "the SFT forward path identical to the pretrained model."
        ),
    )
    parser.add_argument(
        "--attention-backend",
        choices=["eager", "auto", "math"],
        default="eager",
        help=(
            "Attention implementation. 'eager' is the stable default and "
            "computes attention in FP32. 'auto' and 'math' use PyTorch SDPA "
            "only for batches without padding."
        ),
    )
    parser.add_argument(
        "--debug-numerics",
        action="store_true",
        help="Check every leaf module input/output and report the first NaN/Inf.",
    )
    parser.add_argument("--compile", action="store_true")

    parser.add_argument(
        "--tracker", choices=["none", "swanlab", "wandb"], default="none"
    )
    parser.add_argument("--tracker-project", type=str, default="miniLLM-SFT")
    parser.add_argument("--tracker-run-name", type=str, default=None)
    parser.add_argument("--tracker-entity", type=str, default=None)
    parser.add_argument("--tracker-group", type=str, default=None)
    parser.add_argument("--tracker-tags", nargs="*", default=[])
    parser.add_argument(
        "--tracker-mode", choices=["online", "offline"], default="online"
    )
    parser.add_argument(
        "--tracker-log-dir", type=Path, default=PROJECT_ROOT / "logs" / "sft"
    )
    parser.add_argument("--tracker-run-id", type=str, default=None)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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
    )
    for field in positive:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.max_steps < 0 or args.max_train_samples < 0 or args.eval_samples < 0:
        raise ValueError("sample and step limits cannot be negative")
    if args.max_seq_len < 32:
        raise ValueError("--max-seq-len must be at least 32")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if not 0.0 < args.min_learning_rate <= args.learning_rate:
        raise ValueError("--min-learning-rate must be > 0 and <= learning-rate")
    if not 0.0 < args.val_ratio < 0.5:
        raise ValueError("--val-ratio must be between 0 and 0.5")
    for field in ("empty_think_ratio", "system_prompt_ratio"):
        if not 0.0 <= getattr(args, field) <= 1.0:
            raise ValueError(f"--{field.replace('_', '-')} must be in [0, 1]")
    if args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("weight decay cannot be negative and grad clip must be positive")
    if args.debug_numerics and args.compile:
        raise ValueError("--debug-numerics cannot be combined with --compile")


def resolve_amp(
    requested: str, context: DistributedContext
) -> tuple[torch.dtype, bool]:
    if requested == "float32" or context.device.type != "cuda":
        if requested != "float32" and context.is_main:
            print(f"Warning: {requested} autocast is disabled on {context.device.type}.")
        return torch.float32, False
    if requested == "bfloat16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
        if context.is_main:
            print("Warning: bfloat16 is unsupported; falling back to float16.")
        return torch.float16, True
    return torch.float16, True


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype != torch.float32:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def configure_attention_backend(
    backend: str, context: DistributedContext
) -> None:
    """Configure global SDPA flags when the Math backend is requested."""

    if backend != "math" or context.device.type != "cuda":
        return
    # PyTorch's math SDPA keeps intermediates in FP32 for BF16/FP16 inputs.
    # Disable every fused backend explicitly because their availability and
    # dispatch priority vary across PyTorch/CUDA/GPU combinations.
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def _walk_floating_tensors(
    value: Any, prefix: str = "tensor"
) -> list[tuple[str, torch.Tensor]]:
    tensors: list[tuple[str, torch.Tensor]] = []
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            tensors.append((prefix, value))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            tensors.extend(_walk_floating_tensors(item, f"{prefix}[{index}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            tensors.extend(_walk_floating_tensors(item, f"{prefix}.{key}"))
    return tensors


def register_numerics_hooks(model: torch.nn.Module) -> list[Any]:
    """Instrument leaf modules to identify where finite activations first fail."""

    handles: list[Any] = []
    recent_finite_ranges: list[str] = []

    def check(module_name: str, stage: str, value: Any) -> None:
        for tensor_name, tensor in _walk_floating_tensors(value):
            detached = tensor.detach()
            if bool(torch.isfinite(detached).all().item()):
                if stage == "output" and detached.numel():
                    recent_finite_ranges.append(
                        f"{module_name}:{tensor_name} "
                        f"abs_max={detached.float().abs().max().item():.6g}"
                    )
                    del recent_finite_ranges[:-12]
                continue
            recent = (
                "\nRecent finite module outputs:\n  "
                + "\n  ".join(recent_finite_ranges)
                if recent_finite_ranges
                else ""
            )
            raise FloatingPointError(
                f"First non-finite module {stage}: {module_name} ({tensor_name})\n"
                + _tensor_nonfinite_summary(tensor_name, tensor)
                + recent
            )

    for module_name, module in unwrap_model(model).named_modules():
        if not module_name or any(module.children()):
            continue

        def pre_hook(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
            *,
            name: str = module_name,
        ) -> None:
            check(name, "input", inputs)

        def post_hook(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            name: str = module_name,
        ) -> None:
            check(name, "output", output)

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    return handles


def pop_batch_stats(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: batch.pop(key) for key in STAT_KEYS}


def _tensor_nonfinite_summary(name: str, tensor: torch.Tensor) -> str:
    """Return a compact diagnostic without printing model/data values."""

    detached = tensor.detach()
    flat = detached.reshape(-1)
    total = flat.numel()
    finite_count = 0
    nan_count = 0
    posinf_count = 0
    neginf_count = 0
    finite_min = math.inf
    finite_max = -math.inf
    # Chunk the scan so diagnostics do not allocate a second logits-sized
    # tensor when memory is already tight.
    for start in range(0, total, 1_000_000):
        chunk = flat[start : start + 1_000_000]
        finite_mask = torch.isfinite(chunk)
        chunk_finite_count = int(finite_mask.sum().item())
        finite_count += chunk_finite_count
        nan_count += int(torch.isnan(chunk).sum().item())
        posinf_count += int(torch.isposinf(chunk).sum().item())
        neginf_count += int(torch.isneginf(chunk).sum().item())
        if chunk_finite_count:
            finite_values = chunk[finite_mask].float()
            finite_min = min(finite_min, finite_values.min().item())
            finite_max = max(finite_max, finite_values.max().item())
    finite_range = "none"
    if finite_count:
        finite_range = f"[{finite_min:.6g}, {finite_max:.6g}]"
    return (
        f"{name}: shape={tuple(detached.shape)}, dtype={detached.dtype}, "
        f"finite={finite_count}/{total}, nan={nan_count}, "
        f"+inf={posinf_count}, -inf={neginf_count}, finite_range={finite_range}"
    )


@torch.no_grad()
def ensure_finite_model_state(model: torch.nn.Module, source: str) -> None:
    """Fail before training if a parameter or floating buffer is non-finite."""

    bad: list[str] = []
    bad_tensor_count = 0
    raw_model = unwrap_model(model)
    named_state = (
        ("parameter", raw_model.named_parameters()),
        ("buffer", raw_model.named_buffers()),
    )
    for state_kind, tensors in named_state:
        for name, tensor in tensors:
            if not tensor.is_floating_point():
                continue
            if bool(torch.isfinite(tensor).all().item()):
                continue
            bad_tensor_count += 1
            if len(bad) < 10:
                bad.append(
                    _tensor_nonfinite_summary(f"{state_kind}:{name}", tensor)
                )
    if bad_tensor_count:
        omitted = bad_tensor_count - len(bad)
        suffix = f"\n... and {omitted} more state tensors" if omitted else ""
        raise FloatingPointError(
            f"{source} contains non-finite model state; SFT was aborted "
            "before the first update. Re-export a finite pretraining checkpoint.\n"
            + "\n".join(bad)
            + suffix
        )


def ensure_finite_forward(
    output: Any,
    batch: dict[str, torch.Tensor],
    stats: dict[str, torch.Tensor],
    *,
    context: DistributedContext,
    phase: str,
    epoch: int | None,
    batch_index: int,
    global_step: int,
) -> None:
    """Stop before backward and report the batch that produced NaN/Inf."""

    loss_tensors = {
        "loss": output.loss,
        "lm_loss": output.lm_loss,
        "router_aux_loss": output.router_aux_loss,
    }
    bad_losses = [
        name
        for name, value in loss_tensors.items()
        if value is None or not bool(torch.isfinite(value.detach()).all().item())
    ]
    if not bad_losses:
        return

    loss_values = ", ".join(
        f"{name}={None if value is None else value.detach().float().item()}"
        for name, value in loss_tensors.items()
    )
    source_indices = stats["source_indices"].tolist()
    sequence_lengths = batch["attention_mask"].sum(dim=1).tolist()
    target_counts = (batch["labels"][:, 1:] != -100).sum(dim=1).tolist()
    diagnostics = [
        "Non-finite forward result; this update was aborted before backward.",
        (
            f"phase={phase}, rank={context.rank}, epoch={epoch}, "
            f"batch_index={batch_index}, completed_updates={global_step}"
        ),
        f"bad_values={bad_losses}; {loss_values}",
        f"source_jsonl_rows={source_indices}",
        f"sequence_lengths={sequence_lengths}",
        f"assistant_target_counts={target_counts}",
        _tensor_nonfinite_summary("logits", output.logits),
    ]
    raise FloatingPointError("\n".join(diagnostics))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    context: DistributedContext,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    # loss sum, assistant tokens, sequence tokens, padded slots, truncated,
    # answer-truncated, sample count
    totals = torch.zeros(7, device=context.device, dtype=torch.float64)
    for batch_index, batch in enumerate(loader):
        stats = pop_batch_stats(batch)
        batch = {
            key: value.to(context.device, non_blocking=True)
            for key, value in batch.items()
        }
        with autocast_context(context.device, amp_dtype):
            output = model(**batch)
        ensure_finite_forward(
            output,
            batch,
            stats,
            context=context,
            phase="validation",
            epoch=None,
            batch_index=batch_index,
            global_step=-1,
        )
        assistant_tokens = (batch["labels"][:, 1:] != -100).sum()
        lm_loss = output.lm_loss if output.lm_loss is not None else output.loss
        totals[0] += lm_loss.detach().double() * assistant_tokens
        totals[1] += assistant_tokens
        totals[2] += stats["sequence_tokens"].sum().to(context.device)
        totals[3] += batch["input_ids"].numel()
        totals[4] += stats["truncated"].sum().to(context.device)
        totals[5] += stats["answer_truncated"].sum().to(context.device)
        totals[6] += batch["input_ids"].shape[0]
    distributed_sum(totals, context)
    model.train()
    loss = (totals[0] / totals[1].clamp_min(1)).item()
    samples = totals[6].clamp_min(1)
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "assistant_tokens": totals[1].item(),
        "padding_efficiency": (totals[2] / totals[3].clamp_min(1)).item(),
        "truncated_ratio": (totals[4] / samples).item(),
        "answer_truncated_ratio": (totals[5] / samples).item(),
    }


def make_dataloaders(
    args: argparse.Namespace,
    tokenizer: Any,
    context: DistributedContext,
) -> tuple[
    DataLoader,
    DataLoader,
    DistributedSampler | None,
    SkipBatchSampler,
]:
    common = {
        "data_paths": args.data_path,
        "tokenizer": tokenizer,
        "max_seq_len": args.max_seq_len,
        "val_ratio": args.val_ratio,
        "expected_vocab_size": len(tokenizer),
        "empty_think_ratio": args.empty_think_ratio,
        "system_prompt_ratio": args.system_prompt_ratio,
        "seed": args.seed,
    }
    train_dataset: Any = SFTDataset(split="train", **common)
    validation_dataset: Any = SFTDataset(split="validation", **common)
    if args.max_train_samples:
        train_dataset = Subset(
            train_dataset, range(min(args.max_train_samples, len(train_dataset)))
        )
    if args.eval_samples:
        validation_dataset = Subset(
            validation_dataset, range(min(args.eval_samples, len(validation_dataset)))
        )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
        )
        if context.distributed
        else None
    )
    validation_sampler = (
        DistributedSampler(
            validation_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=False,
        )
        if context.distributed
        else None
    )
    collator = SFTDataCollator(
        tokenizer.pad_token_id, pad_to_multiple_of=args.pad_to_multiple_of
    )
    loader_options: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": context.device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collator,
    }
    if args.num_workers > 0:
        loader_options["multiprocessing_context"] = "spawn"
    train_index_sampler: Sampler[int] = (
        train_sampler if train_sampler is not None else RandomSampler(train_dataset)
    )
    train_batch_sampler = SkipBatchSampler(
        BatchSampler(
            train_index_sampler,
            batch_size=args.batch_size,
            drop_last=True,
        )
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=validation_sampler,
        drop_last=False,
        **loader_options,
    )
    if len(train_loader) == 0:
        raise ValueError("Training DataLoader is empty; reduce --batch-size")
    if len(validation_loader) == 0:
        raise ValueError("Validation DataLoader is empty")
    return train_loader, validation_loader, train_sampler, train_batch_sampler


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=ADAM_BETAS,
    )


def print_setup(
    args: argparse.Namespace,
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    context: DistributedContext,
    total_steps: int,
    amp_dtype: torch.dtype,
) -> None:
    if not context.is_main:
        return
    raw_model = unwrap_model(model)
    parameter_count = sum(parameter.numel() for parameter in raw_model.parameters())
    global_batch = args.batch_size * args.accumulation_steps * context.world_size
    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Pretrained    : {args.model_path}")
    print(f"Tokenizer     : {args.tokenizer_path}")
    print(f"Tokenizer SHA : {args.tokenizer_fingerprint}")
    print(f"Data files    : {len(args.data_path)}")
    print(f"Device        : {context.device} (world_size={context.world_size})")
    print(f"PyTorch/CUDA  : {torch.__version__} / {torch.version.cuda}")
    if context.device.type == "cuda":
        properties = torch.cuda.get_device_properties(context.device)
        print(
            f"GPU           : {properties.name} "
            f"(sm_{properties.major}{properties.minor})"
        )
    print(f"Precision     : {str(amp_dtype).removeprefix('torch.')}")
    if context.device.type == "cuda":
        print(
            "TF32          : "
            f"{'enabled' if torch.backends.cuda.matmul.allow_tf32 else 'disabled'}"
        )
    print(f"Attention     : {args.attention_backend}")
    print(
        "Optimizer     : AdamW "
        f"(betas={ADAM_BETAS[0]:g}/{ADAM_BETAS[1]:g}, "
        f"weight_decay={args.weight_decay:g})"
    )
    print(
        "Grad checkpoint: "
        f"{'enabled' if args.gradient_checkpointing else 'disabled'}"
    )
    numeric_check = "every forward" if args.debug_numerics else "first forward"
    print(f"Numeric check : {numeric_check}")
    print(f"Architecture  : {'MoE' if raw_model.config.use_moe else 'Dense'}")
    print(f"Parameters    : {parameter_count:,} trainable")
    print(f"Sequence      : up to {args.max_seq_len} (dynamic padding)")
    print(f"Global batch  : {global_batch} sequences/update")
    print(f"Train batches : {len(train_loader):,} per rank")
    print(f"Valid batches : {len(validation_loader):,} per rank")
    print(f"Update steps  : {total_steps:,}")


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
    args.save_dir = project_path(args.save_dir)
    args.output_dir = project_path(args.output_dir)
    args.tracker_log_dir = project_path(args.tracker_log_dir)

    context = setup_distributed(args.device)
    tracker = ExperimentTracker()
    tracker_finish_state = "crashed"
    tracker_finish_error: str | None = "Training stopped before completion."
    try:
        seed_everything(args.seed, context.rank)
        if context.device.type == "cuda":
            # `--dtype float32` must be genuine IEEE FP32. Forcing TF32 here
            # previously kept CUDA matmuls in reduced mantissa precision even
            # during the FP32 diagnostic run. MiniMind's SFT entry does not
            # force TF32 either, so use the reproducible baseline by default.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        configure_attention_backend(args.attention_backend, context)

        if not (args.model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"Pretrained model config not found: {args.model_path / 'config.json'}"
            )
        args.tokenizer_fingerprint = ensure_tokenizer_matches_models(
            args.tokenizer_path,
            {"pretrained model": args.model_path},
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, local_files_only=True, use_fast=True
        )
        template_path = args.tokenizer_path / "chat_template.jinja"
        if template_path.is_file():
            # transformers 4.x commonly stores the template in tokenizer_config,
            # while newer releases also discover this standalone file.
            tokenizer.chat_template = template_path.read_text(encoding="utf-8")
        if not tokenizer.chat_template:
            raise ValueError(
                f"Tokenizer chat template not found in {args.tokenizer_path}"
            )
        validate_tokenizer(tokenizer, expected_vocab_size=len(tokenizer))
        (
            train_loader,
            validation_loader,
            train_sampler,
            train_batch_sampler,
        ) = make_dataloaders(args, tokenizer, context)
        train_batches_per_epoch = len(train_loader)

        model_config = MiniLLMConfig.from_pretrained(
            args.model_path,
            local_files_only=True,
        )
        model_config.tokenizer_fingerprint = args.tokenizer_fingerprint
        model_config.attention_backend = args.attention_backend
        model: torch.nn.Module = MiniLLMForCausalLM.from_pretrained(
            args.model_path,
            config=model_config,
            local_files_only=True,
        )
        if model.config.vocab_size != len(tokenizer):
            raise ValueError(
                f"Model vocab_size={model.config.vocab_size} but tokenizer has "
                f"{len(tokenizer)} tokens"
            )
        model.config.use_cache = False
        model = model.to(context.device)
        ensure_finite_model_state(model, "Pretrained model")
        if context.is_main:
            print("State check   : parameters and floating buffers are finite")
        # Always trace the first training forward. The hooks are removed after
        # that batch succeeds, so a full run pays the synchronization cost only
        # once. `--debug-numerics` intentionally keeps them for every batch.
        numeric_hook_handles = register_numerics_hooks(model)
        if args.gradient_checkpointing:
            model.model.gradient_checkpointing = True
        if args.compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("--compile requires PyTorch 2.x")
            if model.config.use_moe and context.is_main:
                print(
                    "Warning: dynamic MoE routing may cause torch.compile graph breaks."
                )
            model = torch.compile(model)
        if context.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                # MiniLLMSparseMoE explicitly links unselected experts with a
                # zero-valued graph dependency, so every parameter is present.
                find_unused_parameters=False,
            )

        optimizer = build_optimizer(model, args)
        updates_per_epoch = math.ceil(
            train_batches_per_epoch / args.accumulation_steps
        )
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
            checkpoint = load_checkpoint(
                resume_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            ensure_checkpoint_tokenizer_fingerprint(
                checkpoint, args.tokenizer_fingerprint
            )
            start_epoch = int(checkpoint.get("epoch", 0))
            start_batch = int(checkpoint.get("batch_in_epoch", 0))
            global_step = int(checkpoint.get("global_step", 0))
            ensure_finite_model_state(model, f"SFT checkpoint {resume_path}")
            if context.is_main:
                print("Resume check  : checkpoint model state is finite")
                print(f"Resumed       : {resume_path} (step={global_step:,})")
        elif args.resume == "auto" and context.is_main:
            print("Resume       : no latest.pt found; starting a new SFT run.")

        checkpoint_tracker = checkpoint.get("tracker", {})
        restored_tracker_id = (
            checkpoint_tracker.get("run_id")
            if checkpoint_tracker.get("backend") == args.tracker
            else None
        )
        use_checkpoint_tracker = bool(restored_tracker_id and not args.tracker_run_id)
        tracker_run_name = args.tracker_run_name or (
            f"miniLLM-SFT-Seq{args.max_seq_len}-"
            f"BS{args.batch_size}x{args.accumulation_steps}-LR{args.learning_rate:g}"
        )
        tracker = init_experiment_tracker(
            backend=args.tracker,
            context=context,
            project=(
                checkpoint_tracker.get("project") or args.tracker_project
                if use_checkpoint_tracker
                else args.tracker_project
            ),
            run_name=(
                checkpoint_tracker.get("run_name") or tracker_run_name
                if use_checkpoint_tracker
                else tracker_run_name
            ),
            entity=(
                checkpoint_tracker.get("entity")
                if use_checkpoint_tracker
                else args.tracker_entity
            ),
            group=args.tracker_group,
            tags=args.tracker_tags,
            mode=(
                checkpoint_tracker.get("mode") or args.tracker_mode
                if use_checkpoint_tracker
                else args.tracker_mode
            ),
            log_dir=args.tracker_log_dir,
            config=vars(args).copy(),
            run_id=args.tracker_run_id or restored_tracker_id,
            strict_resume=use_checkpoint_tracker,
        )
        print_setup(
            args,
            model,
            train_loader,
            validation_loader,
            context,
            total_steps,
            amp_dtype,
        )
        if global_step >= total_steps:
            raise ValueError(
                f"Checkpoint step {global_step} already reached total_steps={total_steps}"
            )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw_config = unwrap_model(model).config
        use_moe = bool(raw_config.use_moe)
        num_experts = int(raw_config.num_experts) if use_moe else 0
        interval_loss = 0.0
        interval_lm_loss = 0.0
        interval_aux_loss = 0.0
        interval_micro_batches = 0
        interval_updates = 0
        interval_assistant_tokens = 0
        interval_sequence_tokens = 0
        interval_padded_slots = 0
        interval_samples = 0
        interval_truncated = 0
        interval_answer_truncated = 0
        interval_expert_counts = torch.zeros(
            num_experts, device=context.device, dtype=torch.float64
        )
        interval_router_prob_sums = torch.zeros_like(interval_expert_counts)
        interval_router_entropy_sum = torch.zeros(
            (), device=context.device, dtype=torch.float64
        )
        interval_routed_token_count = torch.zeros_like(
            interval_router_entropy_sum
        )
        interval_start = time.perf_counter()
        latest_grad_norm = 0.0
        stop_training = False

        for epoch in range(start_epoch, training_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            skip_before = start_batch if epoch == start_epoch else 0
            train_batch_sampler.set_skip_batches(skip_before)
            if skip_before and context.is_main:
                print(
                    f"Resume data  : skipping {skip_before:,} completed "
                    "batches at sampler level"
                )
            last_batch_in_epoch = skip_before

            for batch_index, batch in enumerate(
                train_loader, start=skip_before
            ):
                last_batch_in_epoch = batch_index + 1
                window_start = (
                    batch_index // args.accumulation_steps
                ) * args.accumulation_steps
                accumulation_window = min(
                    args.accumulation_steps,
                    train_batches_per_epoch - window_start,
                )
                should_update = (
                    (batch_index + 1) % args.accumulation_steps == 0
                    or batch_index + 1 == train_batches_per_epoch
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
                    try:
                        with autocast_context(context.device, amp_dtype):
                            output = model(**batch)
                            # The last window can contain fewer micro-batches.
                            # Divide by its actual size so that update is not
                            # underweighted when loader length is not divisible.
                            scaled_loss = output.loss / accumulation_window
                    except FloatingPointError as exc:
                        raise FloatingPointError(
                            f"{exc}\nphase=train, rank={context.rank}, epoch={epoch}, "
                            f"batch_index={batch_index}, "
                            f"completed_updates={global_step}\n"
                            f"source_jsonl_rows={stats['source_indices'].tolist()}\n"
                            f"sequence_lengths={batch['attention_mask'].sum(dim=1).tolist()}\n"
                            "assistant_target_counts="
                            f"{(batch['labels'][:, 1:] != -100).sum(dim=1).tolist()}"
                        ) from exc
                    ensure_finite_forward(
                        output,
                        batch,
                        stats,
                        context=context,
                        phase="train",
                        epoch=epoch,
                        batch_index=batch_index,
                        global_step=global_step,
                    )
                    if numeric_hook_handles and not args.debug_numerics:
                        for handle in numeric_hook_handles:
                            handle.remove()
                        numeric_hook_handles.clear()
                    if needs_scaler:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                interval_loss += output.loss.detach().float().item()
                interval_lm_loss += output.lm_loss.detach().float().item()
                if output.router_aux_loss is not None:
                    interval_aux_loss += output.router_aux_loss.detach().float().item()
                if use_moe:
                    interval_expert_counts += output.expert_counts.double()
                    interval_router_prob_sums += output.router_prob_sums.double()
                    interval_router_entropy_sum += output.router_entropy_sum.double()
                    interval_routed_token_count += output.routed_token_count.double()
                interval_micro_batches += 1
                interval_assistant_tokens += int(stats["assistant_tokens"].sum().item())
                interval_sequence_tokens += int(stats["sequence_tokens"].sum().item())
                interval_padded_slots += batch["input_ids"].numel()
                interval_samples += batch["input_ids"].shape[0]
                interval_truncated += int(stats["truncated"].sum().item())
                interval_answer_truncated += int(
                    stats["answer_truncated"].sum().item()
                )

                if not should_update:
                    continue
                if needs_scaler:
                    scaler.unscale_(optimizer)
                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.grad_clip,
                        # FP16 GradScaler handles overflow by skipping an
                        # update. BF16/FP32 have no scaler and must fail fast.
                        error_if_nonfinite=not needs_scaler,
                    )
                except RuntimeError as exc:
                    raise FloatingPointError(
                        "Non-finite gradient norm; optimizer update was aborted. "
                        f"rank={context.rank}, epoch={epoch}, "
                        f"batch_index={batch_index}, completed_updates={global_step}, "
                        f"last_source_jsonl_rows={stats['source_indices'].tolist()}"
                    ) from exc
                latest_grad_norm = float(grad_norm.detach().float().item())
                if needs_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                interval_updates += 1

                if global_step % args.log_interval == 0:
                    elapsed = max(time.perf_counter() - interval_start, 1e-6)
                    totals = torch.tensor(
                        [
                            interval_loss,
                            interval_lm_loss,
                            interval_aux_loss,
                            interval_micro_batches,
                            interval_assistant_tokens,
                            interval_sequence_tokens,
                            interval_padded_slots,
                            interval_samples,
                            interval_truncated,
                            interval_answer_truncated,
                            interval_updates,
                        ],
                        device=context.device,
                        dtype=torch.float64,
                    )
                    distributed_sum(totals, context)
                    moe_totals = None
                    if use_moe:
                        moe_totals = torch.cat(
                            [
                                interval_expert_counts,
                                interval_router_prob_sums,
                                interval_router_entropy_sum.reshape(1),
                                interval_routed_token_count.reshape(1),
                            ]
                        )
                        distributed_sum(moe_totals, context)
                    if context.is_main:
                        micro_batches = max(1.0, totals[3].item())
                        samples = max(1.0, totals[7].item())
                        mean_loss = totals[0].item() / micro_batches
                        mean_lm_loss = totals[1].item() / micro_batches
                        mean_aux_loss = totals[2].item() / micro_batches
                        world_elapsed = elapsed
                        metrics = {
                            "train/loss": mean_loss,
                            "train/lm_loss": mean_lm_loss,
                            "train/router_aux_loss": mean_aux_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/grad_norm": latest_grad_norm,
                            "train/assistant_tokens_per_second": totals[4].item()
                            / world_elapsed,
                            "train/sequence_tokens_per_second": totals[5].item()
                            / world_elapsed,
                            "data/padding_efficiency": totals[5].item()
                            / max(1.0, totals[6].item()),
                            "data/truncated_ratio": totals[8].item() / samples,
                            "data/answer_truncated_ratio": totals[9].item() / samples,
                        }
                        if use_moe and moe_totals is not None:
                            count_end = num_experts
                            prob_end = count_end + num_experts
                            expert_counts = moe_totals[:count_end]
                            router_prob_sums = moe_totals[count_end:prob_end]
                            entropy_sum = moe_totals[prob_end]
                            routed_tokens = moe_totals[prob_end + 1].clamp_min(1.0)
                            expert_usage = expert_counts / expert_counts.sum().clamp_min(1.0)
                            mean_router_probs = router_prob_sums / routed_tokens
                            normalized_entropy = entropy_sum / (
                                routed_tokens * math.log(num_experts)
                            )
                            metrics["moe/router_entropy_normalized"] = (
                                normalized_entropy.item()
                            )
                            metrics["moe/max_load_ratio"] = expert_usage.max().item()
                            metrics["moe/min_load_ratio"] = expert_usage.min().item()
                            for expert_index in range(num_experts):
                                metrics[f"moe/expert_{expert_index}_usage"] = (
                                    expert_usage[expert_index].item()
                                )
                                metrics[
                                    f"moe/expert_{expert_index}_router_probability"
                                ] = mean_router_probs[expert_index].item()
                        print(
                            f"epoch={epoch + 1}/{training_epochs} "
                            f"step={global_step:,}/{total_steps:,} "
                            f"loss={mean_loss:.4f} lm={mean_lm_loss:.4f} "
                            f"aux={mean_aux_loss:.6f} "
                            f"lr={scheduler.get_last_lr()[0]:.2e} "
                            f"assistant_tok/s={metrics['train/assistant_tokens_per_second']:.0f} "
                            f"pad_eff={metrics['data/padding_efficiency']:.1%}"
                        )
                        tracker.log(metrics, step=global_step)
                    interval_loss = 0.0
                    interval_lm_loss = 0.0
                    interval_aux_loss = 0.0
                    interval_micro_batches = 0
                    interval_updates = 0
                    interval_assistant_tokens = 0
                    interval_sequence_tokens = 0
                    interval_padded_slots = 0
                    interval_samples = 0
                    interval_truncated = 0
                    interval_answer_truncated = 0
                    interval_expert_counts.zero_()
                    interval_router_prob_sums.zero_()
                    interval_router_entropy_sum.zero_()
                    interval_routed_token_count.zero_()
                    interval_start = time.perf_counter()

                if global_step % args.eval_interval == 0:
                    validation = evaluate(
                        model, validation_loader, context, amp_dtype
                    )
                    if context.is_main:
                        print(
                            f"validation step={global_step:,} "
                            f"loss={validation['loss']:.4f} "
                            f"ppl={validation['perplexity']:.2f} "
                            f"pad_eff={validation['padding_efficiency']:.1%} "
                            f"truncated={validation['truncated_ratio']:.1%}"
                        )
                        tracker.log(
                            {
                                "validation/loss": validation["loss"],
                                "validation/perplexity": validation["perplexity"],
                                "validation/assistant_tokens": validation[
                                    "assistant_tokens"
                                ],
                                "validation/padding_efficiency": validation[
                                    "padding_efficiency"
                                ],
                                "validation/truncated_ratio": validation[
                                    "truncated_ratio"
                                ],
                                "validation/answer_truncated_ratio": validation[
                                    "answer_truncated_ratio"
                                ],
                            },
                            step=global_step,
                        )

                if global_step % args.save_interval == 0 and context.is_main:
                    save_checkpoint(
                        args.save_dir / "latest.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch=epoch,
                        batch_in_epoch=batch_index + 1,
                        global_step=global_step,
                        args=args,
                        tracker_state=tracker.state_dict(),
                    )
                    print(f"Checkpoint    : {args.save_dir / 'latest.pt'}")

                if global_step >= total_steps:
                    stop_training = True
                    break

            start_batch = 0
            train_batch_sampler.set_skip_batches(0)
            checkpoint_epoch = epoch if stop_training else epoch + 1
            checkpoint_batch = last_batch_in_epoch if stop_training else 0
            if context.is_main:
                save_checkpoint(
                    args.save_dir / "latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=checkpoint_epoch,
                    batch_in_epoch=checkpoint_batch,
                    global_step=global_step,
                    args=args,
                    tracker_state=tracker.state_dict(),
                )
            if stop_training:
                break

        if context.distributed:
            torch.distributed.barrier()
        if context.is_main:
            export_pretrained(
                args.output_dir, model, tokenizer, args.tokenizer_path
            )
            print(f"Training complete at optimizer step {global_step:,}.")
            print(f"Checkpoint    : {args.save_dir / 'latest.pt'}")
            print(f"Exported model: {args.output_dir}")
        tracker_finish_state = "success"
        tracker_finish_error = None
    except KeyboardInterrupt:
        tracker_finish_state = "aborted"
        tracker_finish_error = "SFT interrupted by user (KeyboardInterrupt)."
        raise
    except BaseException:
        tracker_finish_state = "crashed"
        tracker_finish_error = traceback.format_exc()
        raise
    finally:
        try:
            tracker.finish(state=tracker_finish_state, error=tracker_finish_error)
        finally:
            cleanup_distributed(context)


if __name__ == "__main__":
    main()

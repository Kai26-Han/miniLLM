#!/usr/bin/env python3
"""Full-parameter Direct Preference Optimization for miniLLM.

The policy and frozen reference model start from the same exported SFT model.
Chosen/rejected responses are concatenated into one policy forward and one
reference forward, and only final assistant tokens contribute to sequence
log-probabilities.

Example from the project root::

    python trainer/train_dpo.py \
        --model-path out/sft \
        --data-path dataset/rl/dpo.jsonl
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dpo_dataset import (  # noqa: E402
    DPO_STAT_KEYS,
    DPODataCollator,
    DPODataset,
)
from dataset.lm_dataset import validate_tokenizer  # noqa: E402
from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM  # noqa: E402
from trainer.train_sft import (  # noqa: E402
    autocast_context,
    configure_attention_backend,
    ensure_finite_model_state,
    register_numerics_hooks,
    resolve_amp,
)
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


DPO_ADAM_BETAS = (0.9, 0.999)
DPO_LEARNING_RATE = 4e-8
DPO_MIN_LEARNING_RATE = 4e-9
DPO_BETA = 0.15
DPO_LABEL_SMOOTHING = 0.0
DPO_BASELINE_LOSS = math.log(2.0)


@dataclass
class DPOObjectiveOutput:
    losses: torch.Tensor
    chosen_rewards: torch.Tensor
    rejected_rewards: torch.Tensor
    reward_margins: torch.Tensor
    preference_accuracies: torch.Tensor
    policy_chosen_logps: torch.Tensor
    policy_rejected_logps: torch.Tensor
    reference_chosen_logps: torch.Tensor
    reference_rejected_logps: torch.Tensor


@dataclass
class DPOForwardOutput:
    total_loss: torch.Tensor
    dpo_loss: torch.Tensor
    router_aux_loss: torch.Tensor
    objective: DPOObjectiveOutput
    expert_counts: torch.Tensor | None = None
    router_prob_sums: torch.Tensor | None = None
    router_entropy_sum: torch.Tensor | None = None
    routed_token_count: torch.Tensor | None = None


@dataclass
class DPOTrainingState:
    """Validation state needed for best-checkpoint selection and early stopping."""

    best_validation_loss: float = math.inf
    best_step: int = 0
    bad_evaluations: int = 0
    last_evaluation_step: int = -1

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "DPOTrainingState":
        saved = checkpoint.get("extra_state", {})
        return cls(
            best_validation_loss=float(
                saved.get("best_validation_loss", math.inf)
            ),
            best_step=int(saved.get("best_step", 0)),
            bad_evaluations=int(saved.get("bad_evaluations", 0)),
            last_evaluation_step=int(saved.get("last_evaluation_step", -1)),
        )

    def state_dict(self) -> dict[str, float | int]:
        return {
            "best_validation_loss": self.best_validation_loss,
            "best_step": self.best_step,
            "bad_evaluations": self.bad_evaluations,
            "last_evaluation_step": self.last_evaluation_step,
        }

    def update(self, validation_loss: float, step: int, min_delta: float) -> bool:
        improved = validation_loss < self.best_validation_loss - min_delta
        self.last_evaluation_step = step
        if improved:
            self.best_validation_loss = validation_loss
            self.best_step = step
            self.bad_evaluations = 0
        else:
            self.bad_evaluations += 1
        return improved


def sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum next-token log-probabilities at response-only label positions."""

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be [batch, sequence, vocab] and labels [batch, sequence]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels batch/sequence dimensions must match")
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    target_mask = shift_labels.ne(-100)
    if bool((target_mask.sum(dim=1) == 0).any().item()):
        raise ValueError("every DPO sequence must contain at least one response token")
    safe_labels = shift_labels.masked_fill(~target_mask, 0)
    token_logps = F.log_softmax(shift_logits.float(), dim=-1).gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logps * target_mask).sum(dim=-1)


def compute_dpo_objective(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> DPOObjectiveOutput:
    """Compute sequence-level conservative DPO for possibly noisy preferences."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")
    if policy_logps.ndim != 1 or reference_logps.ndim != 1:
        raise ValueError("policy and reference log-probabilities must be vectors")
    if policy_logps.shape != reference_logps.shape or policy_logps.numel() % 2:
        raise ValueError("DPO vectors must have equal, even lengths")
    pair_count = policy_logps.numel() // 2
    policy_chosen, policy_rejected = policy_logps.split(pair_count)
    reference_chosen, reference_rejected = reference_logps.split(pair_count)
    policy_logratios = policy_chosen - policy_rejected
    reference_logratios = reference_chosen - reference_rejected
    preference_logits = policy_logratios - reference_logratios
    scaled_logits = beta * preference_logits
    losses = -(
        (1.0 - label_smoothing) * F.logsigmoid(scaled_logits)
        + label_smoothing * F.logsigmoid(-scaled_logits)
    )
    chosen_rewards = beta * (policy_chosen - reference_chosen).detach()
    rejected_rewards = beta * (policy_rejected - reference_rejected).detach()
    reward_margins = chosen_rewards - rejected_rewards
    return DPOObjectiveOutput(
        losses=losses,
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        reward_margins=reward_margins,
        preference_accuracies=reward_margins.gt(0).float(),
        policy_chosen_logps=policy_chosen.detach(),
        policy_rejected_logps=policy_rejected.detach(),
        reference_chosen_logps=reference_chosen.detach(),
        reference_rejected_logps=reference_rejected.detach(),
    )


def dpo_forward(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    beta: float,
    label_smoothing: float = 0.0,
) -> DPOForwardOutput:
    """Run frozen-reference and trainable-policy forwards for one paired batch."""

    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    with torch.no_grad():
        reference_logits = reference_model(**model_inputs).logits
        reference_logps = sequence_log_probs(reference_logits, batch["labels"])
        del reference_logits
    policy_output = policy_model(**model_inputs)
    policy_logps = sequence_log_probs(policy_output.logits, batch["labels"])
    objective = compute_dpo_objective(
        policy_logps,
        reference_logps,
        beta,
        label_smoothing=label_smoothing,
    )
    dpo_loss = objective.losses.mean()
    router_aux_loss = policy_output.router_aux_loss
    if router_aux_loss is None:
        router_aux_loss = dpo_loss.new_zeros(())
    return DPOForwardOutput(
        total_loss=dpo_loss + router_aux_loss,
        dpo_loss=dpo_loss,
        router_aux_loss=router_aux_loss,
        objective=objective,
        expert_counts=policy_output.expert_counts,
        router_prob_sums=policy_output.router_prob_sums,
        router_entropy_sum=policy_output.router_entropy_sum,
        routed_token_count=policy_output.routed_token_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-parameter DPO for miniLLM.")
    parser.add_argument(
        "--data-path",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "dataset" / "rl" / "dpo.jsonl"],
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "out" / "sft",
        help="Exported SFT model used to initialize policy and reference.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Tokenizer directory; defaults to --model-path.",
    )
    parser.add_argument(
        "--save-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "dpo"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "out" / "dpo"
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Preference pairs/GPU"
    )
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=DPO_LEARNING_RATE)
    parser.add_argument("--min-learning-rate", type=float, default=DPO_MIN_LEARNING_RATE)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=DPO_ADAM_BETAS[0])
    parser.add_argument("--adam-beta2", type=float, default=DPO_ADAM_BETAS[1])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=DPO_BETA)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=DPO_LABEL_SMOOTHING,
        help="Conservative DPO preference-noise probability; 0 is standard DPO.",
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2,
        help="Preference pairs/GPU during validation; does not affect training batch.",
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Training precision; bfloat16 matches the MiniMind DPO default.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving validations; 0 disables it.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation DPO-loss decrease counted as an improvement.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume a DPO checkpoint, or use save-dir/latest.pt with --resume.",
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
    parser.add_argument("--tracker-project", type=str, default="miniLLM-DPO")
    parser.add_argument("--tracker-run-name", type=str, default=None)
    parser.add_argument("--tracker-entity", type=str, default=None)
    parser.add_argument("--tracker-group", type=str, default=None)
    parser.add_argument("--tracker-tags", nargs="*", default=[])
    parser.add_argument(
        "--tracker-mode", choices=["online", "offline"], default="online"
    )
    parser.add_argument(
        "--tracker-log-dir", type=Path, default=PROJECT_ROOT / "logs" / "dpo"
    )
    parser.add_argument("--tracker-run-id", type=str, default=None)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    for field in (
        "epochs",
        "batch_size",
        "accumulation_steps",
        "eval_batch_size",
        "learning_rate",
        "beta",
        "max_seq_len",
        "pad_to_multiple_of",
        "log_interval",
        "eval_interval",
        "save_interval",
    ):
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.max_steps < 0 or args.max_train_samples < 0 or args.eval_samples < 0:
        raise ValueError("sample and step limits cannot be negative")
    if args.early_stopping_patience < 0 or args.early_stopping_min_delta < 0:
        raise ValueError("early-stopping patience and min-delta cannot be negative")
    if args.max_seq_len < 32 or args.num_workers < 0:
        raise ValueError("invalid sequence length or worker count")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if not 0.0 < args.min_learning_rate <= args.learning_rate:
        raise ValueError("--min-learning-rate must be > 0 and <= learning-rate")
    if not 0.0 < args.val_ratio < 0.5:
        raise ValueError("--val-ratio must be between 0 and 0.5")
    if not 0.0 <= args.label_smoothing < 0.5:
        raise ValueError("--label-smoothing must be in [0, 0.5)")
    if args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("weight decay cannot be negative and grad clip must be positive")
    for field in ("adam_beta1", "adam_beta2"):
        if not 0.0 <= getattr(args, field) < 1.0:
            raise ValueError(f"--{field.replace('_', '-')} must be in [0, 1)")
    if args.debug_numerics and args.compile:
        raise ValueError("--debug-numerics cannot be combined with --compile")


def pop_batch_stats(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: batch.pop(key) for key in DPO_STAT_KEYS}


def build_dpo_optimizer(
    model: torch.nn.Module, args: argparse.Namespace
) -> AdamW:
    """Build the MiniMind-style AdamW optimizer for DPO only."""

    return AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )


def ensure_finite_dpo(
    output: DPOForwardOutput,
    stats: dict[str, torch.Tensor],
    *,
    context: DistributedContext,
    phase: str,
    epoch: int | None,
    batch_index: int,
    global_step: int,
) -> None:
    values = {
        "total_loss": output.total_loss,
        "dpo_loss": output.dpo_loss,
        "router_aux_loss": output.router_aux_loss,
        "chosen_rewards": output.objective.chosen_rewards,
        "rejected_rewards": output.objective.rejected_rewards,
        "policy_chosen_logps": output.objective.policy_chosen_logps,
        "policy_rejected_logps": output.objective.policy_rejected_logps,
        "reference_chosen_logps": output.objective.reference_chosen_logps,
        "reference_rejected_logps": output.objective.reference_rejected_logps,
    }
    bad = [
        name
        for name, value in values.items()
        if not bool(torch.isfinite(value.detach()).all().item())
    ]
    if bad:
        raise FloatingPointError(
            "Non-finite DPO forward result; update aborted before backward. "
            f"phase={phase}, rank={context.rank}, epoch={epoch}, "
            f"batch_index={batch_index}, completed_updates={global_step}, "
            f"source_jsonl_rows={stats['source_indices'].tolist()}, bad_values={bad}"
        )


def make_dataloaders(
    args: argparse.Namespace,
    tokenizer: Any,
    context: DistributedContext,
) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    common = {
        "data_paths": args.data_path,
        "tokenizer": tokenizer,
        "max_seq_len": args.max_seq_len,
        "val_ratio": args.val_ratio,
        "expected_vocab_size": len(tokenizer),
        "seed": args.seed,
    }
    train_dataset: Any = DPODataset(split="train", **common)
    validation_dataset: Any = DPODataset(split="validation", **common)
    if args.max_train_samples:
        train_dataset = Subset(
            train_dataset, range(min(args.max_train_samples, len(train_dataset)))
        )
    if args.eval_samples:
        validation_dataset = Subset(
            validation_dataset, range(min(args.eval_samples, len(validation_dataset)))
        )
    # Use the epoch-seeded sampler on one GPU as well. This makes batch order
    # reproducible when a checkpoint resumes by skipping completed batches.
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=args.seed,
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
    collator = DPODataCollator(
        tokenizer.pad_token_id, pad_to_multiple_of=args.pad_to_multiple_of
    )
    options: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": context.device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collator,
    }
    if args.num_workers > 0:
        options["multiprocessing_context"] = "spawn"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        drop_last=True,
        **options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        sampler=validation_sampler,
        drop_last=False,
        **options,
    )
    if len(train_loader) == 0:
        raise ValueError("Training DataLoader is empty; reduce --batch-size")
    if len(validation_loader) == 0:
        raise ValueError("Validation DataLoader is empty")
    return train_loader, validation_loader, train_sampler


@torch.no_grad()
def evaluate(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    loader: DataLoader,
    context: DistributedContext,
    amp_dtype: torch.dtype,
    beta: float,
    label_smoothing: float = 0.0,
) -> dict[str, float]:
    policy_model.eval()
    reference_model.eval()
    totals = torch.zeros(22, device=context.device, dtype=torch.float64)
    for batch_index, batch in enumerate(loader):
        stats = pop_batch_stats(batch)
        batch = {
            key: value.to(context.device, non_blocking=True)
            for key, value in batch.items()
        }
        with autocast_context(context.device, amp_dtype):
            output = dpo_forward(
                policy_model,
                reference_model,
                batch,
                beta,
                label_smoothing=label_smoothing,
            )
        ensure_finite_dpo(
            output,
            stats,
            context=context,
            phase="validation",
            epoch=None,
            batch_index=batch_index,
            global_step=-1,
        )
        pairs = stats["source_indices"].numel()
        objective = output.objective
        totals[0] += output.dpo_loss.double() * pairs
        totals[1] += output.router_aux_loss.double() * pairs
        totals[2] += output.total_loss.double() * pairs
        totals[3] += objective.chosen_rewards.double().sum()
        totals[4] += objective.rejected_rewards.double().sum()
        totals[5] += objective.reward_margins.double().sum()
        totals[6] += objective.preference_accuracies.double().sum()
        totals[7] += objective.policy_chosen_logps.double().sum()
        totals[8] += objective.policy_rejected_logps.double().sum()
        totals[9] += objective.reference_chosen_logps.double().sum()
        totals[10] += objective.reference_rejected_logps.double().sum()
        totals[11] += pairs
        totals[12] += stats["chosen_response_tokens"].sum().to(context.device)
        totals[13] += stats["rejected_response_tokens"].sum().to(context.device)
        totals[14] += (
            stats["chosen_sequence_tokens"].sum()
            + stats["rejected_sequence_tokens"].sum()
        ).to(context.device)
        totals[15] += batch["input_ids"].numel()
        totals[16] += stats["chosen_truncated"].sum().to(context.device)
        totals[17] += stats["rejected_truncated"].sum().to(context.device)
        totals[18] += stats["chosen_answer_truncated"].sum().to(context.device)
        totals[19] += stats["rejected_answer_truncated"].sum().to(context.device)
        totals[20] += (
            objective.policy_chosen_logps - objective.reference_chosen_logps
        ).abs().double().sum()
        totals[21] += (
            objective.policy_rejected_logps - objective.reference_rejected_logps
        ).abs().double().sum()
    distributed_sum(totals, context)
    policy_model.train()
    pairs = totals[11].clamp_min(1.0)
    return {
        "dpo_loss": (totals[0] / pairs).item(),
        "router_aux_loss": (totals[1] / pairs).item(),
        "total_loss": (totals[2] / pairs).item(),
        "reward_chosen": (totals[3] / pairs).item(),
        "reward_rejected": (totals[4] / pairs).item(),
        "reward_margin": (totals[5] / pairs).item(),
        "preference_accuracy": (totals[6] / pairs).item(),
        "policy_chosen_logp": (totals[7] / pairs).item(),
        "policy_rejected_logp": (totals[8] / pairs).item(),
        "reference_chosen_logp": (totals[9] / pairs).item(),
        "reference_rejected_logp": (totals[10] / pairs).item(),
        "pairs": totals[11].item(),
        "mean_chosen_response_tokens": (totals[12] / pairs).item(),
        "mean_rejected_response_tokens": (totals[13] / pairs).item(),
        "padding_efficiency": (totals[14] / totals[15].clamp_min(1.0)).item(),
        "chosen_truncated_ratio": (totals[16] / pairs).item(),
        "rejected_truncated_ratio": (totals[17] / pairs).item(),
        "chosen_answer_truncated_ratio": (totals[18] / pairs).item(),
        "rejected_answer_truncated_ratio": (totals[19] / pairs).item(),
        "policy_reference_chosen_logp_mae": (totals[20] / pairs).item(),
        "policy_reference_rejected_logp_mae": (totals[21] / pairs).item(),
    }


def _resume_compatibility(checkpoint: dict[str, Any], args: argparse.Namespace) -> None:
    saved = checkpoint.get("args", {})
    if "label_smoothing" not in saved:
        raise ValueError(
            "This checkpoint predates the conservative DPO objective and cannot "
            "be resumed safely. Start a fresh run from --model-path instead."
        )
    previous_data_paths = [str(path) for path in saved.get("data_path", [])]
    current_data_paths = [str(path) for path in args.data_path]
    if previous_data_paths != current_data_paths:
        raise ValueError(
            "Cannot resume with changed data_path: "
            f"checkpoint={previous_data_paths!r}, current={current_data_paths!r}"
        )
    checks = {
        "model_path": str(args.model_path),
        "max_seq_len": args.max_seq_len,
        "val_ratio": args.val_ratio,
        "beta": args.beta,
        "label_smoothing": args.label_smoothing,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "eval_samples": args.eval_samples,
    }
    for key, current in checks.items():
        previous = saved.get(key)
        if previous is not None and str(previous) != str(current):
            raise ValueError(
                f"Cannot resume with changed {key}: checkpoint={previous!r}, "
                f"current={current!r}"
            )


def print_setup(
    args: argparse.Namespace,
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    context: DistributedContext,
    total_steps: int,
    amp_dtype: torch.dtype,
) -> None:
    if not context.is_main:
        return
    policy = unwrap_model(policy_model)
    policy_parameters = sum(parameter.numel() for parameter in policy.parameters())
    reference_parameters = sum(
        parameter.numel() for parameter in reference_model.parameters()
    )
    global_batch = args.batch_size * args.accumulation_steps * context.world_size
    objective_name = "conservative" if args.label_smoothing > 0 else "standard"
    print(f"Project root  : {PROJECT_ROOT}")
    print(f"SFT model     : {args.model_path}")
    print(f"Tokenizer     : {args.tokenizer_path}")
    print(f"Tokenizer SHA : {args.tokenizer_fingerprint}")
    print(f"Data files    : {len(args.data_path)}")
    print(f"Device        : {context.device} (world_size={context.world_size})")
    print(f"Precision     : {str(amp_dtype).removeprefix('torch.')}")
    print(f"Attention     : {args.attention_backend}")
    print(
        "Optimizer     : AdamW "
        f"(betas={args.adam_beta1:g}/{args.adam_beta2:g}, "
        f"weight_decay={args.weight_decay:g})"
    )
    print(f"Architecture  : {'MoE' if policy.config.use_moe else 'Dense'}")
    print(f"Policy params : {policy_parameters:,} trainable")
    print(f"Ref params    : {reference_parameters:,} frozen")
    print(f"Sequence      : up to {args.max_seq_len} (dynamic pair padding)")
    print(
        f"DPO objective : {objective_name} (beta={args.beta:g}, "
        f"label_smoothing={args.label_smoothing:g})"
    )
    print(f"Global batch  : {global_batch} preference pairs/update")
    print(f"Train batches : {len(train_loader):,} per rank")
    print(
        f"Validation    : {len(validation_loader):,} batches/rank "
        f"(batch={args.eval_batch_size})"
    )
    print(f"Update steps  : {total_steps:,}")
    if args.early_stopping_patience:
        print(
            "Early stopping: "
            f"patience={args.early_stopping_patience}, "
            f"min_delta={args.early_stopping_min_delta:g}"
        )


def validate_and_update_best(
    *,
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    validation_loader: DataLoader,
    context: DistributedContext,
    amp_dtype: torch.dtype,
    args: argparse.Namespace,
    training_state: DPOTrainingState,
    optimizer: AdamW,
    scheduler: Any,
    scaler: Any,
    tracker: ExperimentTracker,
    best_checkpoint_path: Path,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
) -> dict[str, float]:
    """Evaluate on all ranks and atomically update the rank-zero best checkpoint."""

    if context.is_main:
        print(f"Validation    : starting at step {global_step:,} ...")
    validation = evaluate(
        policy_model,
        reference_model,
        validation_loader,
        context,
        amp_dtype,
        args.beta,
        args.label_smoothing,
    )
    improved = training_state.update(
        validation["dpo_loss"],
        global_step,
        args.early_stopping_min_delta,
    )
    if not context.is_main:
        return validation

    print(
        f"validation step={global_step:,} "
        f"dpo={validation['dpo_loss']:.4f} "
        f"margin={validation['reward_margin']:.4f} "
        f"acc={validation['preference_accuracy']:.1%} "
        f"pad_eff={validation['padding_efficiency']:.1%}"
    )
    tracker.log(
        {f"validation/{key}": value for key, value in validation.items()},
        step=global_step,
    )
    if improved:
        save_checkpoint(
            best_checkpoint_path,
            policy_model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            global_step=global_step,
            args=args,
            tracker_state=tracker.state_dict(),
            extra_state=training_state.state_dict(),
        )
        print(
            f"Best checkpoint: {best_checkpoint_path} "
            f"(dpo={training_state.best_validation_loss:.6f})"
        )
    else:
        print(
            "No improvement: "
            f"{training_state.bad_evaluations}/"
            f"{args.early_stopping_patience or 'disabled'}"
        )
    return validation


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
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        configure_attention_backend(args.attention_backend, context)

        if not (args.model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"SFT model config not found: {args.model_path / 'config.json'}. "
                "Finish SFT and export out/sft before starting DPO."
            )
        args.tokenizer_fingerprint = ensure_tokenizer_matches_models(
            args.tokenizer_path,
            {"SFT policy/reference model": args.model_path},
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, local_files_only=True, use_fast=True
        )
        template_path = args.tokenizer_path / "chat_template.jinja"
        if template_path.is_file():
            tokenizer.chat_template = template_path.read_text(encoding="utf-8")
        if not tokenizer.chat_template:
            raise ValueError(f"Tokenizer chat template not found in {args.tokenizer_path}")
        validate_tokenizer(tokenizer, expected_vocab_size=len(tokenizer))
        train_loader, validation_loader, train_sampler = make_dataloaders(
            args, tokenizer, context
        )

        config = MiniLLMConfig.from_pretrained(args.model_path, local_files_only=True)
        config.tokenizer_fingerprint = args.tokenizer_fingerprint
        config.attention_backend = args.attention_backend
        config.dpo_beta = args.beta
        config.dpo_label_smoothing = args.label_smoothing
        config.dpo_reference_model = str(args.model_path)
        policy_model: torch.nn.Module = MiniLLMForCausalLM.from_pretrained(
            args.model_path, config=config, local_files_only=True
        )
        reference_config = MiniLLMConfig.from_pretrained(
            args.model_path, local_files_only=True
        )
        reference_config.tokenizer_fingerprint = args.tokenizer_fingerprint
        reference_config.attention_backend = args.attention_backend
        reference_model: torch.nn.Module = MiniLLMForCausalLM.from_pretrained(
            args.model_path, config=reference_config, local_files_only=True
        )
        if policy_model.config.vocab_size != len(tokenizer):
            raise ValueError(
                f"Model vocab_size={policy_model.config.vocab_size} but tokenizer "
                f"has {len(tokenizer)} tokens"
            )
        policy_model.config.use_cache = False
        reference_model.config.use_cache = False
        policy_model = policy_model.to(context.device)
        reference_model = reference_model.to(context.device)
        reference_model.eval()
        reference_model.requires_grad_(False)
        ensure_finite_model_state(policy_model, "SFT policy model")
        ensure_finite_model_state(reference_model, "SFT reference model")
        numeric_hook_handles = register_numerics_hooks(policy_model)
        if args.gradient_checkpointing:
            policy_model.model.gradient_checkpointing = True
        if args.compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("--compile requires PyTorch 2.x")
            policy_model = torch.compile(policy_model)
        if context.distributed:
            policy_model = DistributedDataParallel(
                policy_model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        optimizer = build_dpo_optimizer(policy_model, args)
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
            checkpoint = load_checkpoint(
                resume_path,
                policy_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            _resume_compatibility(checkpoint, args)
            ensure_checkpoint_tokenizer_fingerprint(
                checkpoint, args.tokenizer_fingerprint
            )
            start_epoch = int(checkpoint.get("epoch", 0))
            start_batch = int(checkpoint.get("batch_in_epoch", 0))
            global_step = int(checkpoint.get("global_step", 0))
            ensure_finite_model_state(policy_model, f"DPO checkpoint {resume_path}")
            if context.is_main:
                print(f"Resumed       : {resume_path} (step={global_step:,})")
        elif args.resume == "auto" and context.is_main:
            print("Resume        : no latest.pt found; starting a new DPO run")
        training_state = DPOTrainingState.from_checkpoint(checkpoint)
        best_checkpoint_path = args.save_dir / "best.pt"

        checkpoint_tracker = checkpoint.get("tracker", {})
        restored_tracker_id = (
            checkpoint_tracker.get("run_id")
            if checkpoint_tracker.get("backend") == args.tracker
            else None
        )
        use_checkpoint_tracker = bool(restored_tracker_id and not args.tracker_run_id)
        objective_tag = "cDPO" if args.label_smoothing > 0 else "DPO"
        tracker_run_name = args.tracker_run_name or (
            f"miniLLM-{objective_tag}-Beta{args.beta:g}-LS{args.label_smoothing:g}-"
            f"Seq{args.max_seq_len}-"
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
            policy_model,
            reference_model,
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

        if resume_path is None:
            if context.is_main:
                print("Baseline      : validating policy/reference parity before training")
            baseline = evaluate(
                policy_model,
                reference_model,
                validation_loader,
                context,
                amp_dtype,
                args.beta,
                args.label_smoothing,
            )
            logp_mae = max(
                baseline["policy_reference_chosen_logp_mae"],
                baseline["policy_reference_rejected_logp_mae"],
            )
            if (
                abs(baseline["dpo_loss"] - DPO_BASELINE_LOSS) > 5e-4
                or abs(baseline["reward_margin"]) > 1e-5
                or logp_mae > 1e-4
            ):
                raise RuntimeError(
                    "Policy/reference parity check failed before the first DPO update: "
                    f"dpo_loss={baseline['dpo_loss']:.8f}, "
                    f"expected={DPO_BASELINE_LOSS:.8f}, "
                    f"reward_margin={baseline['reward_margin']:.3e}, "
                    f"logp_mae={logp_mae:.3e}. Start policy and reference from the "
                    "same untouched SFT export."
                )
            training_state.best_validation_loss = baseline["dpo_loss"]
            training_state.best_step = 0
            training_state.bad_evaluations = 0
            training_state.last_evaluation_step = 0
            if context.is_main:
                print(
                    f"Baseline pass : dpo={baseline['dpo_loss']:.6f} "
                    f"margin={baseline['reward_margin']:.2e} logp_mae={logp_mae:.2e}"
                )
                print(
                    "Validation data: "
                    f"pairs={baseline['pairs']:.0f}, "
                    f"chosen/rejected tokens="
                    f"{baseline['mean_chosen_response_tokens']:.1f}/"
                    f"{baseline['mean_rejected_response_tokens']:.1f}, "
                    f"answer_clipped="
                    f"{baseline['chosen_answer_truncated_ratio']:.1%}/"
                    f"{baseline['rejected_answer_truncated_ratio']:.1%}"
                )
                if max(
                    baseline["chosen_answer_truncated_ratio"],
                    baseline["rejected_answer_truncated_ratio"],
                ) > 0.05:
                    print(
                        "Warning       : more than 5% of validation answers are "
                        "clipped; increase max_seq_len or filter long pairs"
                    )
                tracker.log(
                    {f"validation/{key}": value for key, value in baseline.items()},
                    step=0,
                )
                save_checkpoint(
                    best_checkpoint_path,
                    policy_model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=0,
                    batch_in_epoch=0,
                    global_step=0,
                    args=args,
                    tracker_state=tracker.state_dict(),
                    extra_state=training_state.state_dict(),
                )
                print(f"Best checkpoint: {best_checkpoint_path} (baseline)")
        elif not best_checkpoint_path.is_file():
            # A resumable latest checkpoint without its paired best checkpoint
            # must establish a new best on the next validation.
            training_state.best_validation_loss = math.inf
            training_state.best_step = global_step
            training_state.bad_evaluations = 0
            if context.is_main:
                print(
                    f"Warning       : {best_checkpoint_path} is missing; the next "
                    "validation will recreate it"
                )

        policy_model.train()
        reference_model.eval()
        optimizer.zero_grad(set_to_none=True)
        raw_config = unwrap_model(policy_model).config
        use_moe = bool(raw_config.use_moe)
        num_experts = int(raw_config.num_experts) if use_moe else 0
        interval = torch.zeros(21, device=context.device, dtype=torch.float64)
        interval_expert_counts = torch.zeros(
            num_experts, device=context.device, dtype=torch.float64
        )
        interval_router_prob_sums = torch.zeros_like(interval_expert_counts)
        interval_router_entropy_sum = torch.zeros(
            (), device=context.device, dtype=torch.float64
        )
        interval_routed_token_count = torch.zeros_like(interval_router_entropy_sum)
        interval_start = time.perf_counter()
        latest_grad_norm = 0.0
        stop_training = False

        for epoch in range(start_epoch, training_epochs):
            train_sampler.set_epoch(epoch)
            skip_before = start_batch if epoch == start_epoch else 0
            for batch_index, batch in enumerate(train_loader):
                if batch_index < skip_before:
                    continue
                window_start = (
                    batch_index // args.accumulation_steps
                ) * args.accumulation_steps
                accumulation_window = min(
                    args.accumulation_steps, len(train_loader) - window_start
                )
                should_update = (
                    (batch_index + 1) % args.accumulation_steps == 0
                    or batch_index + 1 == len(train_loader)
                )
                sync_context = (
                    nullcontext()
                    if should_update or not context.distributed
                    else policy_model.no_sync()
                )
                stats = pop_batch_stats(batch)
                batch = {
                    key: value.to(context.device, non_blocking=True)
                    for key, value in batch.items()
                }
                with sync_context:
                    with autocast_context(context.device, amp_dtype):
                        output = dpo_forward(
                            policy_model,
                            reference_model,
                            batch,
                            args.beta,
                            label_smoothing=args.label_smoothing,
                        )
                        scaled_loss = output.total_loss / accumulation_window
                    ensure_finite_dpo(
                        output,
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

                pairs = stats["source_indices"].numel()
                objective = output.objective
                interval[0] += output.total_loss.detach().double()
                interval[1] += output.dpo_loss.detach().double()
                interval[2] += output.router_aux_loss.detach().double()
                interval[3] += objective.chosen_rewards.double().sum()
                interval[4] += objective.rejected_rewards.double().sum()
                interval[5] += objective.reward_margins.double().sum()
                interval[6] += objective.preference_accuracies.double().sum()
                interval[7] += objective.policy_chosen_logps.double().sum()
                interval[8] += objective.policy_rejected_logps.double().sum()
                interval[9] += objective.reference_chosen_logps.double().sum()
                interval[10] += objective.reference_rejected_logps.double().sum()
                interval[11] += 1
                interval[12] += pairs
                interval[13] += (
                    stats["chosen_response_tokens"].sum()
                    + stats["rejected_response_tokens"].sum()
                ).to(context.device)
                interval[14] += (
                    stats["chosen_sequence_tokens"].sum()
                    + stats["rejected_sequence_tokens"].sum()
                ).to(context.device)
                interval[15] += batch["input_ids"].numel()
                interval[16] += stats["chosen_truncated"].sum().to(context.device)
                interval[17] += stats["rejected_truncated"].sum().to(context.device)
                interval[18] += stats["chosen_answer_truncated"].sum().to(
                    context.device
                )
                interval[19] += stats["rejected_answer_truncated"].sum().to(
                    context.device
                )
                if use_moe:
                    interval_expert_counts += output.expert_counts.double()
                    interval_router_prob_sums += output.router_prob_sums.double()
                    interval_router_entropy_sum += output.router_entropy_sum.double()
                    interval_routed_token_count += output.routed_token_count.double()

                if not should_update:
                    continue
                if needs_scaler:
                    scaler.unscale_(optimizer)
                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy_model.parameters(),
                        args.grad_clip,
                        error_if_nonfinite=not needs_scaler,
                    )
                except RuntimeError as exc:
                    raise FloatingPointError(
                        "Non-finite DPO gradient norm; optimizer update aborted. "
                        f"rank={context.rank}, epoch={epoch}, "
                        f"batch_index={batch_index}, completed_updates={global_step}, "
                        f"source_jsonl_rows={stats['source_indices'].tolist()}"
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
                interval[20] += 1

                if global_step % args.log_interval == 0:
                    elapsed = max(time.perf_counter() - interval_start, 1e-6)
                    distributed_sum(interval, context)
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
                        micro_batches = max(1.0, interval[11].item())
                        pair_count = max(1.0, interval[12].item())
                        metrics = {
                            "train/total_loss": interval[0].item() / micro_batches,
                            "train/dpo_loss": interval[1].item() / micro_batches,
                            "train/router_aux_loss": interval[2].item()
                            / micro_batches,
                            "train/reward_chosen": interval[3].item() / pair_count,
                            "train/reward_rejected": interval[4].item() / pair_count,
                            "train/reward_margin": interval[5].item() / pair_count,
                            "train/preference_accuracy": interval[6].item()
                            / pair_count,
                            "train/policy_chosen_logp": interval[7].item()
                            / pair_count,
                            "train/policy_rejected_logp": interval[8].item()
                            / pair_count,
                            "train/reference_chosen_logp": interval[9].item()
                            / pair_count,
                            "train/reference_rejected_logp": interval[10].item()
                            / pair_count,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/grad_norm": latest_grad_norm,
                            "train/response_tokens_per_second": interval[13].item()
                            / elapsed,
                            "train/sequence_tokens_per_second": interval[14].item()
                            / elapsed,
                            "data/padding_efficiency": interval[14].item()
                            / max(1.0, interval[15].item()),
                            "data/chosen_truncated_ratio": interval[16].item()
                            / pair_count,
                            "data/rejected_truncated_ratio": interval[17].item()
                            / pair_count,
                            "data/chosen_answer_truncated_ratio": interval[18].item()
                            / pair_count,
                            "data/rejected_answer_truncated_ratio": interval[19].item()
                            / pair_count,
                        }
                        if use_moe and moe_totals is not None:
                            count_end = num_experts
                            prob_end = count_end + num_experts
                            expert_counts = moe_totals[:count_end]
                            probability_sums = moe_totals[count_end:prob_end]
                            routed_tokens = moe_totals[prob_end + 1].clamp_min(1.0)
                            expert_usage = expert_counts / expert_counts.sum().clamp_min(1.0)
                            metrics["moe/router_entropy_normalized"] = (
                                moe_totals[prob_end]
                                / (routed_tokens * math.log(num_experts))
                            ).item()
                            metrics["moe/max_load_ratio"] = expert_usage.max().item()
                            metrics["moe/min_load_ratio"] = expert_usage.min().item()
                            for expert_index in range(num_experts):
                                metrics[f"moe/expert_{expert_index}_usage"] = (
                                    expert_usage[expert_index].item()
                                )
                                metrics[
                                    f"moe/expert_{expert_index}_router_probability"
                                ] = (probability_sums[expert_index] / routed_tokens).item()
                        print(
                            f"epoch={epoch + 1}/{training_epochs} "
                            f"step={global_step:,}/{total_steps:,} "
                            f"dpo={metrics['train/dpo_loss']:.4f} "
                            f"margin={metrics['train/reward_margin']:.4f} "
                            f"acc={metrics['train/preference_accuracy']:.1%} "
                            f"lr={metrics['train/learning_rate']:.2e} "
                            f"response_tok/s={metrics['train/response_tokens_per_second']:.0f}"
                        )
                        tracker.log(metrics, step=global_step)
                    interval.zero_()
                    interval_expert_counts.zero_()
                    interval_router_prob_sums.zero_()
                    interval_router_entropy_sum.zero_()
                    interval_routed_token_count.zero_()
                    interval_start = time.perf_counter()

                if global_step % args.eval_interval == 0:
                    validate_and_update_best(
                        policy_model=policy_model,
                        reference_model=reference_model,
                        validation_loader=validation_loader,
                        context=context,
                        amp_dtype=amp_dtype,
                        args=args,
                        training_state=training_state,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        tracker=tracker,
                        best_checkpoint_path=best_checkpoint_path,
                        epoch=epoch,
                        batch_in_epoch=batch_index + 1,
                        global_step=global_step,
                    )

                    if (
                        args.early_stopping_patience > 0
                        and training_state.bad_evaluations
                        >= args.early_stopping_patience
                    ):
                        stop_training = True
                        if context.is_main:
                            print(
                                "Early stopping: validation DPO loss did not improve "
                                f"for {training_state.bad_evaluations} evaluations."
                            )

                if global_step % args.save_interval == 0 and context.is_main:
                    save_checkpoint(
                        args.save_dir / "latest.pt",
                        policy_model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch=epoch,
                        batch_in_epoch=batch_index + 1,
                        global_step=global_step,
                        args=args,
                        tracker_state=tracker.state_dict(),
                        extra_state=training_state.state_dict(),
                    )
                    print(f"Checkpoint    : {args.save_dir / 'latest.pt'}")

                if stop_training or global_step >= total_steps:
                    stop_training = True
                    break

            start_batch = 0
            checkpoint_epoch = epoch if stop_training else epoch + 1
            checkpoint_batch = batch_index + 1 if stop_training else 0
            if context.is_main:
                save_checkpoint(
                    args.save_dir / "latest.pt",
                    policy_model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=checkpoint_epoch,
                    batch_in_epoch=checkpoint_batch,
                    global_step=global_step,
                    args=args,
                    tracker_state=tracker.state_dict(),
                    extra_state=training_state.state_dict(),
                )
            if stop_training:
                break

        if training_state.last_evaluation_step != global_step:
            validate_and_update_best(
                policy_model=policy_model,
                reference_model=reference_model,
                validation_loader=validation_loader,
                context=context,
                amp_dtype=amp_dtype,
                args=args,
                training_state=training_state,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                tracker=tracker,
                best_checkpoint_path=best_checkpoint_path,
                epoch=checkpoint_epoch,
                batch_in_epoch=checkpoint_batch,
                global_step=global_step,
            )
            if context.is_main:
                save_checkpoint(
                    args.save_dir / "latest.pt",
                    policy_model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=checkpoint_epoch,
                    batch_in_epoch=checkpoint_batch,
                    global_step=global_step,
                    args=args,
                    tracker_state=tracker.state_dict(),
                    extra_state=training_state.state_dict(),
                )

        if context.distributed:
            torch.distributed.barrier()
        if context.is_main:
            if not best_checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Best DPO checkpoint was not created: {best_checkpoint_path}"
                )
            load_checkpoint(
                best_checkpoint_path,
                policy_model,
                restore_rng=False,
            )
            export_pretrained(
                args.output_dir, policy_model, tokenizer, args.tokenizer_path
            )
            print(f"Training complete at optimizer step {global_step:,}.")
            print(f"Latest checkpoint: {args.save_dir / 'latest.pt'}")
            print(
                f"Best checkpoint  : {best_checkpoint_path} "
                f"(step={training_state.best_step:,}, "
                f"dpo={training_state.best_validation_loss:.6f})"
            )
            print(f"Exported best model: {args.output_dir}")
        tracker_finish_state = "success"
        tracker_finish_error = None
    except KeyboardInterrupt:
        tracker_finish_state = "aborted"
        tracker_finish_error = "DPO interrupted by user (KeyboardInterrupt)."
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

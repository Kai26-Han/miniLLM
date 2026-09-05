#!/usr/bin/env python3
"""Full-parameter GRPO training for miniLLM after SFT, DPO, or PPO.

The implementation follows MiniMind's compact RLAIF recipe: sample several
answers for every prompt, score them with a frozen reward model plus small
rule rewards, normalize rewards inside each prompt group, and optimize a
PPO-clipped policy objective with a frozen-reference KL penalty.

Example from the project root::

    python trainer/train_grpo.py \
        --model-path out/ppo \
        --reward-model-path ../internlm2-1_8b-reward
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lm_dataset import validate_tokenizer  # noqa: E402
from dataset.ppo_dataset import PPODataCollator, PPODataset  # noqa: E402
from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM  # noqa: E402
from trainer.rollout_engine import (  # noqa: E402
    TorchRolloutEngine,
    gather_completion_log_probs,
)
from trainer.train_sft import (  # noqa: E402
    autocast_context,
    build_optimizer,
    configure_attention_backend,
    ensure_finite_model_state,
    resolve_amp,
)
from trainer.train_ppo import ExternalRewardScorer, resolve_reward_dtype  # noqa: E402
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


@dataclass
class RewardOutput:
    total: torch.Tensor
    model: torch.Tensor
    length: torch.Tensor
    thinking: torch.Tensor
    repetition: torch.Tensor


@dataclass
class AdvantageOutput:
    advantages: torch.Tensor
    group_means: torch.Tensor
    group_stds: torch.Tensor


@dataclass
class GRPOObjectiveOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_penalty: torch.Tensor
    clip_fraction: torch.Tensor
    mean_ratio: torch.Tensor


class LMRewardModel:
    """GRPO wrapper around the validated PPO reward-model adapter.

    Keeping one reward adapter for PPO and GRPO ensures both stages use the
    same tokenizer-ID repair, RoPE-buffer restoration, cache compatibility,
    precision handling, finite-state checks, and input validation.
    """

    def __init__(self, path: Path, device: torch.device, dtype: torch.dtype) -> None:
        self.scorer = ExternalRewardScorer(
            str(path),
            device,
            dtype,
            reward_clip=3.0,
            allow_remote=False,
            trust_remote_code=True,
        )
        self.model = self.scorer.model
        self.tokenizer = self.scorer.tokenizer
        self.dtype = self.scorer.dtype
        self.token_id_remap = self.scorer.token_id_remap
        self.restored_rope_buffers = self.scorer.restored_rope_buffers
        probe = self.scorer.score(
            [[{"role": "user", "content": "你好"}]],
            ["你好！有什么可以帮你？"],
        )
        self.probe_score = float(probe[0].item())

    @torch.no_grad()
    def get_score(self, messages: Sequence[dict[str, Any]], answer: str) -> float:
        return float(self.scorer.score([messages], [answer])[0].item())


def repetition_penalty(text: str, n: int = 3, cap: float = 0.5) -> float:
    """Return MiniMind-style repeated n-gram penalty in ``[0, cap]``."""

    if n <= 0 or cap < 0:
        raise ValueError("n must be positive and cap cannot be negative")
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    duplicates = len(ngrams) - len(set(ngrams))
    return min(cap, duplicates * cap * 2 / len(ngrams))


def repeat_by_group(values: Sequence[Any], num_generations: int) -> list[Any]:
    """Repeat each prompt contiguously so reward reshape preserves groups."""

    if num_generations < 2:
        raise ValueError("GRPO requires at least two generations per prompt")
    return [value for value in values for _ in range(num_generations)]


def calculate_rewards(
    messages: Sequence[Sequence[dict[str, Any]]],
    responses: Sequence[str],
    reward_model: Any,
    num_generations: int,
    device: torch.device,
) -> RewardOutput:
    """Score grouped responses with the compact MiniMind reward recipe."""

    expected = len(messages) * num_generations
    if len(responses) != expected:
        raise ValueError(f"Expected {expected} responses, received {len(responses)}")
    model_scores: list[float] = []
    length_scores: list[float] = []
    thinking_scores: list[float] = []
    repetition_scores: list[float] = []
    with torch.no_grad():
        for index, response in enumerate(responses):
            answer = response.strip()
            length_score = 0.5 if 20 <= len(answer) <= 800 else -0.5
            thinking_score = 0.0
            if "</think>" in response:
                thinking, answer = response.split("</think>", 1)
                thinking_score += 1.0 if 20 <= len(thinking.strip()) <= 300 else -0.5
                thinking_score += 0.25 if response.count("</think>") == 1 else -0.25
                answer = answer.strip()
            model_scores.append(
                reward_model.get_score(messages[index // num_generations], answer)
            )
            length_scores.append(length_score)
            thinking_scores.append(thinking_score)
            repetition_scores.append(repetition_penalty(answer))

    model_tensor = torch.tensor(model_scores, device=device, dtype=torch.float32)
    length_tensor = torch.tensor(length_scores, device=device, dtype=torch.float32)
    thinking_tensor = torch.tensor(thinking_scores, device=device, dtype=torch.float32)
    repetition_tensor = torch.tensor(
        repetition_scores, device=device, dtype=torch.float32
    )
    return RewardOutput(
        total=model_tensor + length_tensor + thinking_tensor - repetition_tensor,
        model=model_tensor,
        length=length_tensor,
        thinking=thinking_tensor,
        repetition=repetition_tensor,
    )


def compute_group_advantages(
    rewards: torch.Tensor, num_generations: int, eps: float = 1e-4
) -> AdvantageOutput:
    """Normalize scalar rewards independently inside every prompt group."""

    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("rewards must be a non-empty vector")
    if num_generations < 2 or rewards.numel() % num_generations:
        raise ValueError("rewards must contain complete GRPO groups")
    if eps <= 0:
        raise ValueError("eps must be positive")
    grouped = rewards.reshape(-1, num_generations)
    means = grouped.mean(dim=1, keepdim=True)
    stds = grouped.std(dim=1, unbiased=False, keepdim=True)
    advantages = ((grouped - means) / (stds + eps)).reshape(-1)
    return AdvantageOutput(advantages, means.squeeze(1), stds.squeeze(1))


def compute_grpo_objective(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    epsilon: float,
    beta: float,
) -> GRPOObjectiveOutput:
    """Compute token-clipped GRPO with a non-negative sampled KL penalty."""

    if current_log_probs.ndim != 2:
        raise ValueError("log-probability tensors must have shape [batch, response]")
    if not (
        current_log_probs.shape
        == old_log_probs.shape
        == reference_log_probs.shape
        == completion_mask.shape
    ):
        raise ValueError("log-probability tensors and completion_mask must match")
    if advantages.shape != (current_log_probs.shape[0],):
        raise ValueError("advantages must have one value per generated response")
    if not 0.0 < epsilon < 1.0 or beta < 0:
        raise ValueError("epsilon must be in (0, 1) and beta cannot be negative")
    mask = completion_mask.float()
    if bool((mask.sum(dim=1) == 0).any().item()):
        raise ValueError("every response must contain at least one valid token")

    old = old_log_probs.detach()
    reference = reference_log_probs.detach()
    advantage = advantages.detach().unsqueeze(1)
    ratio = torch.exp(current_log_probs - old)
    clipped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    log_ratio = reference - current_log_probs
    per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0
    per_token_loss = -surrogate + beta * per_token_kl
    token_counts = mask.sum(dim=1).clamp_min(1.0)
    policy_loss = ((-surrogate * mask).sum(dim=1) / token_counts).mean()
    kl_penalty = ((per_token_kl * mask).sum(dim=1) / token_counts).mean()
    loss = ((per_token_loss * mask).sum(dim=1) / token_counts).mean()
    clip_fraction = (
        (((ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)).float() * mask).sum()
        / mask.sum().clamp_min(1.0)
    )
    mean_ratio = (ratio * mask).sum() / mask.sum().clamp_min(1.0)
    return GRPOObjectiveOutput(loss, policy_loss, kl_penalty, clip_fraction, mean_ratio)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-parameter GRPO for miniLLM.")
    parser.add_argument(
        "--data-path",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "dataset" / "rl" / "rlaif.jsonl"],
    )
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "out" / "dpo")
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=None,
        help="Frozen KL anchor; defaults to --model-path.",
    )
    parser.add_argument(
        "--reward-model-path",
        type=Path,
        default=PROJECT_ROOT.parent / "internlm2-1_8b-reward",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Tokenizer directory; defaults to --model-path.",
    )
    parser.add_argument(
        "--save-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "grpo"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "out" / "grpo"
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Prompts per GPU")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-7)
    parser.add_argument("--min-learning-rate", type=float, default=1e-8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--advantage-eps", type=float, default=1e-4)

    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--thinking-ratio", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=0)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--reward-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Reward inference precision; FP16 is the verified InternLM2 default.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None,
        help="Resume a checkpoint, or use save-dir/latest.pt with --resume.",
    )
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--attention-backend", choices=["eager", "auto", "math"], default="eager"
    )

    parser.add_argument(
        "--tracker", choices=["none", "swanlab", "wandb"], default="none"
    )
    parser.add_argument("--tracker-project", type=str, default="miniLLM-GRPO")
    parser.add_argument("--tracker-run-name", type=str, default=None)
    parser.add_argument("--tracker-entity", type=str, default=None)
    parser.add_argument("--tracker-group", type=str, default=None)
    parser.add_argument("--tracker-tags", nargs="*", default=[])
    parser.add_argument("--tracker-mode", choices=["online", "offline"], default="online")
    parser.add_argument(
        "--tracker-log-dir", type=Path, default=PROJECT_ROOT / "logs" / "grpo"
    )
    parser.add_argument("--tracker-run-id", type=str, default=None)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    for field in (
        "epochs", "batch_size", "num_generations", "accumulation_steps",
        "learning_rate", "max_prompt_len", "max_new_tokens", "log_interval",
        "eval_interval", "save_interval",
    ):
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_generations < 2:
        raise ValueError("--num-generations must be at least 2")
    if args.max_steps < 0 or args.max_train_samples < 0 or args.eval_samples < 0:
        raise ValueError("sample and step limits cannot be negative")
    if not 0.0 < args.min_learning_rate <= args.learning_rate:
        raise ValueError("--min-learning-rate must be > 0 and <= learning-rate")
    if not 0.0 <= args.warmup_ratio < 1.0 or not 0.0 < args.val_ratio < 0.5:
        raise ValueError("invalid warmup or validation ratio")
    if not 0.0 <= args.thinking_ratio <= 1.0:
        raise ValueError("--thinking-ratio must be in [0, 1]")
    if not 0.0 < args.epsilon < 1.0 or args.beta < 0 or args.advantage_eps <= 0:
        raise ValueError("invalid GRPO epsilon, beta, or advantage epsilon")
    if args.weight_decay < 0 or args.grad_clip <= 0 or args.num_workers < 0:
        raise ValueError("invalid optimizer or worker setting")
    if args.temperature <= 0 or not 0.0 < args.top_p <= 1.0 or args.top_k < 0:
        raise ValueError("GRPO requires valid stochastic sampling parameters")


def make_dataloaders(
    args: argparse.Namespace, tokenizer: Any, context: DistributedContext
) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    common = dict(
        data_paths=args.data_path,
        tokenizer=tokenizer,
        max_prompt_len=args.max_prompt_len,
        val_ratio=args.val_ratio,
        expected_vocab_size=len(tokenizer),
        thinking_ratio=args.thinking_ratio,
        seed=args.seed,
    )
    train_dataset: Any = PPODataset(split="train", **common)
    validation_dataset: Any = PPODataset(split="validation", **common)
    if context.is_main and train_dataset.repaired_source_indices:
        repaired = ", ".join(
            str(index) for index in train_dataset.repaired_source_indices[:8]
        )
        suffix = " ..." if len(train_dataset.repaired_source_indices) > 8 else ""
        print(
            "RL data repair : removed trailing assistant answers from "
            f"{len(train_dataset.repaired_source_indices)} rows "
            f"({repaired}{suffix})"
        )
    if args.max_train_samples:
        train_dataset = Subset(
            train_dataset, range(min(args.max_train_samples, len(train_dataset)))
        )
    if args.eval_samples:
        validation_dataset = Subset(
            validation_dataset, range(min(args.eval_samples, len(validation_dataset)))
        )
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
    options: dict[str, Any] = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=context.device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=PPODataCollator(),
    )
    if args.num_workers > 0:
        options["multiprocessing_context"] = "spawn"
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, shuffle=False, drop_last=True, **options
    )
    validation_loader = DataLoader(
        validation_dataset,
        sampler=validation_sampler,
        shuffle=False,
        drop_last=False,
        **options,
    )
    if not len(train_loader) or not len(validation_loader):
        raise ValueError("GRPO DataLoader is empty; reduce --batch-size")
    return train_loader, validation_loader, train_sampler


def reference_log_probs(
    reference_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    action_positions: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        output = reference_model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = gather_completion_log_probs(
            output.logits, input_ids, action_positions
        )
        del output
    return log_probs


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    rollout_engine: TorchRolloutEngine,
    reference_model: torch.nn.Module,
    reward_model: LMRewardModel,
    args: argparse.Namespace,
    context: DistributedContext,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    # reward, model reward, length reward, think reward, repetition penalty,
    # response tokens, eos count, responses, zero-std groups, groups, sampled KL
    totals = torch.zeros(11, device=context.device, dtype=torch.float64)
    for batch in loader:
        expanded_prompts = repeat_by_group(batch["prompts"], args.num_generations)
        rollout = rollout_engine.rollout(expanded_prompts)
        rewards = calculate_rewards(
            batch["messages"], rollout.responses, reward_model,
            args.num_generations, context.device,
        )
        advantages = compute_group_advantages(
            rewards.total, args.num_generations, args.advantage_eps
        )
        with autocast_context(context.device, amp_dtype):
            ref_log_probs = reference_log_probs(
                reference_model, rollout.input_ids, rollout.attention_mask,
                rollout.action_positions,
            )
        mask = rollout.completion_mask
        sampled_kl = (
            ((rollout.old_log_probs - ref_log_probs) * mask).sum()
            / mask.sum().clamp_min(1.0)
        )
        response_count = rewards.total.numel()
        totals[0] += rewards.total.double().sum()
        totals[1] += rewards.model.double().sum()
        totals[2] += rewards.length.double().sum()
        totals[3] += rewards.thinking.double().sum()
        totals[4] += rewards.repetition.double().sum()
        totals[5] += rollout.response_lengths.double().sum()
        totals[6] += rollout.has_eos.double().sum()
        totals[7] += response_count
        totals[8] += advantages.group_stds.le(args.advantage_eps).double().sum()
        totals[9] += advantages.group_stds.numel()
        totals[10] += sampled_kl.double() * response_count
        del rollout, rewards, advantages, ref_log_probs
    distributed_sum(totals, context)
    responses = totals[7].clamp_min(1.0)
    groups = totals[9].clamp_min(1.0)
    return {
        "reward": (totals[0] / responses).item(),
        "reward_model": (totals[1] / responses).item(),
        "reward_length": (totals[2] / responses).item(),
        "reward_thinking": (totals[3] / responses).item(),
        "repetition_penalty": (totals[4] / responses).item(),
        "response_length": (totals[5] / responses).item(),
        "eos_rate": (totals[6] / responses).item(),
        "zero_std_group_rate": (totals[8] / groups).item(),
        "sampled_kl": (totals[10] / responses).item(),
    }


def _resume_compatibility(checkpoint: dict[str, Any], args: argparse.Namespace) -> None:
    saved = checkpoint.get("args", {})
    checks = {
        "model_path": str(args.model_path),
        "reference_path": str(args.reference_path),
        "reward_model_path": str(args.reward_model_path),
        "reward_dtype": args.reward_dtype,
        "num_generations": args.num_generations,
        "max_prompt_len": args.max_prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "beta": args.beta,
        "epsilon": args.epsilon,
    }
    for key, current in checks.items():
        previous = saved.get(key)
        if previous is not None and str(previous) != str(current):
            raise ValueError(
                f"Cannot resume with changed {key}: checkpoint={previous!r}, "
                f"current={current!r}"
            )


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    validate_args(args)
    args.data_path = [project_path(path) for path in args.data_path]
    args.model_path = project_path(args.model_path)
    args.reference_path = (
        args.model_path if args.reference_path is None else project_path(args.reference_path)
    )
    args.reward_model_path = project_path(args.reward_model_path)
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
    finish_state = "crashed"
    finish_error: str | None = "Training stopped before completion."
    try:
        seed_everything(args.seed, context.rank)
        configure_attention_backend(args.attention_backend, context)
        for name, path in (
            ("GRPO policy", args.model_path),
            ("reference", args.reference_path),
            ("reward model", args.reward_model_path),
        ):
            if not (path / "config.json").is_file():
                raise FileNotFoundError(f"{name} config not found: {path / 'config.json'}")

        args.tokenizer_fingerprint = ensure_tokenizer_matches_models(
            args.tokenizer_path,
            {
                "GRPO policy": args.model_path,
                "GRPO reference": args.reference_path,
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, local_files_only=True, use_fast=True
        )
        template_path = args.tokenizer_path / "chat_template.jinja"
        if template_path.is_file():
            tokenizer.chat_template = template_path.read_text(encoding="utf-8")
        if not tokenizer.chat_template:
            raise ValueError("GRPO tokenizer has no chat template")
        validate_tokenizer(tokenizer, expected_vocab_size=len(tokenizer))
        train_loader, validation_loader, train_sampler = make_dataloaders(
            args, tokenizer, context
        )

        config = MiniLLMConfig.from_pretrained(args.model_path, local_files_only=True)
        config.tokenizer_fingerprint = args.tokenizer_fingerprint
        config.attention_backend = args.attention_backend
        policy_model: torch.nn.Module = MiniLLMForCausalLM.from_pretrained(
            args.model_path, config=config, local_files_only=True
        )
        reference_config = MiniLLMConfig.from_pretrained(
            args.reference_path, local_files_only=True
        )
        reference_config.tokenizer_fingerprint = args.tokenizer_fingerprint
        reference_config.attention_backend = args.attention_backend
        reference_model = MiniLLMForCausalLM.from_pretrained(
            args.reference_path, config=reference_config, local_files_only=True
        )
        if config.vocab_size != len(tokenizer):
            raise ValueError("Policy and tokenizer vocabulary sizes do not match")
        if args.max_prompt_len + args.max_new_tokens > config.max_position_embeddings:
            raise ValueError("Prompt plus completion exceeds model position capacity")
        policy_model.config.use_cache = False
        reference_model.config.use_cache = False
        policy_model = policy_model.to(context.device)
        reference_model = reference_model.to(context.device).eval().requires_grad_(False)
        ensure_finite_model_state(policy_model, "GRPO policy model")
        ensure_finite_model_state(reference_model, "GRPO reference model")
        if args.gradient_checkpointing:
            unwrap_model(policy_model).model.gradient_checkpointing = True
        if context.distributed:
            policy_model = DistributedDataParallel(
                policy_model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        amp_dtype, needs_scaler = resolve_amp(args.dtype, context)
        scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)
        # Keep reward inference precision independent of policy autocast.
        reward_dtype = resolve_reward_dtype(args.reward_dtype, context.device)
        reward_model = LMRewardModel(
            args.reward_model_path, context.device, reward_dtype
        )
        rollout_engine = TorchRolloutEngine(
            policy_model,
            tokenizer,
            context.device,
            max_prompt_len=args.max_prompt_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            autocast_factory=lambda: autocast_context(context.device, amp_dtype),
        )
        optimizer = build_optimizer(policy_model, args)
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

        start_epoch = start_batch = global_step = 0
        checkpoint: dict[str, Any] = {}
        resume_path = resolve_resume_path(args.save_dir, args.resume)
        if resume_path is not None:
            checkpoint = load_checkpoint(
                resume_path, policy_model, optimizer, scheduler, scaler
            )
            _resume_compatibility(checkpoint, args)
            ensure_checkpoint_tokenizer_fingerprint(
                checkpoint, args.tokenizer_fingerprint
            )
            start_epoch = int(checkpoint.get("epoch", 0))
            start_batch = int(checkpoint.get("batch_in_epoch", 0))
            global_step = int(checkpoint.get("global_step", 0))

        tracker_state = checkpoint.get("tracker", {})
        restored_id = (
            tracker_state.get("run_id")
            if tracker_state.get("backend") == args.tracker
            else None
        )
        tracker = init_experiment_tracker(
            backend=args.tracker,
            context=context,
            project=args.tracker_project,
            run_name=args.tracker_run_name or "miniLLM-GRPO",
            entity=args.tracker_entity,
            group=args.tracker_group,
            tags=args.tracker_tags,
            mode=args.tracker_mode,
            log_dir=args.tracker_log_dir,
            config=vars(args).copy(),
            run_id=args.tracker_run_id or restored_id,
            strict_resume=bool(restored_id and not args.tracker_run_id),
        )
        if global_step >= total_steps:
            raise ValueError(
                f"Checkpoint step {global_step} already reached total_steps={total_steps}"
            )
        if context.is_main:
            print(f"Policy        : {args.model_path}")
            print(f"Reference     : {args.reference_path}")
            print(f"Tokenizer     : {args.tokenizer_path}")
            print(f"Tokenizer SHA : {args.tokenizer_fingerprint}")
            print(f"Reward model  : {args.reward_model_path}")
            print(f"Reward dtype  : {str(reward_model.dtype).removeprefix('torch.')}")
            print(f"Reward check  : finite (probe={reward_model.probe_score:.4f})")
            if reward_model.token_id_remap:
                print(
                    "Reward ID map : "
                    f"{len(reward_model.token_id_remap)} special tokens restored"
                )
            if reward_model.restored_rope_buffers:
                print(
                    "Reward RoPE   : "
                    f"{reward_model.restored_rope_buffers} FP32 buffers rebuilt"
                )
            print(f"Device        : {context.device} (world_size={context.world_size})")
            print(f"Groups        : {args.batch_size} prompts/GPU x {args.num_generations}")
            print(f"Lengths       : {args.max_prompt_len} + {args.max_new_tokens}")
            print(f"GRPO          : beta={args.beta:g}, epsilon={args.epsilon:g}")
            print(f"Update steps  : {total_steps:,}")

        policy_model.train()
        optimizer.zero_grad(set_to_none=True)
        stop_training = False
        interval = torch.zeros(16, device=context.device, dtype=torch.float64)
        interval_start = time.perf_counter()
        latest_grad_norm = 0.0

        for epoch in range(start_epoch, training_epochs):
            train_sampler.set_epoch(epoch)
            skip_before = start_batch if epoch == start_epoch else 0
            for batch_index, batch in enumerate(train_loader):
                if batch_index < skip_before:
                    continue
                window_start = (batch_index // args.accumulation_steps) * args.accumulation_steps
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
                expanded_prompts = repeat_by_group(
                    batch["prompts"], args.num_generations
                )
                rollout = rollout_engine.rollout(expanded_prompts)
                rewards = calculate_rewards(
                    batch["messages"], rollout.responses, reward_model,
                    args.num_generations, context.device,
                )
                advantage_output = compute_group_advantages(
                    rewards.total, args.num_generations, args.advantage_eps
                )

                with autocast_context(context.device, amp_dtype):
                    ref_log_probs = reference_log_probs(
                        reference_model,
                        rollout.input_ids,
                        rollout.attention_mask,
                        rollout.action_positions,
                    )
                with sync_context:
                    with autocast_context(context.device, amp_dtype):
                        policy_output = policy_model(
                            input_ids=rollout.input_ids,
                            attention_mask=rollout.attention_mask,
                        )
                        current_log_probs = gather_completion_log_probs(
                            policy_output.logits,
                            rollout.input_ids,
                            rollout.action_positions,
                        )
                        objective = compute_grpo_objective(
                            current_log_probs,
                            rollout.old_log_probs,
                            ref_log_probs,
                            advantage_output.advantages,
                            rollout.completion_mask,
                            epsilon=args.epsilon,
                            beta=args.beta,
                        )
                        router_aux = policy_output.router_aux_loss
                        if router_aux is None:
                            router_aux = objective.loss.new_zeros(())
                        total_loss = objective.loss + router_aux
                        scaled_loss = total_loss / accumulation_window
                    finite = (
                        total_loss, objective.loss, objective.kl_penalty,
                        rewards.total, current_log_probs,
                    )
                    if not all(torch.isfinite(value.detach()).all() for value in finite):
                        raise FloatingPointError(
                            f"Non-finite GRPO value at epoch={epoch}, batch={batch_index}"
                        )
                    if needs_scaler:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                responses = rewards.total.numel()
                groups = advantage_output.group_stds.numel()
                interval[0] += total_loss.detach().double()
                interval[1] += objective.policy_loss.detach().double()
                interval[2] += objective.kl_penalty.detach().double()
                interval[3] += router_aux.detach().double()
                interval[4] += rewards.total.double().sum()
                interval[5] += rewards.model.double().sum()
                interval[6] += rewards.length.double().sum()
                interval[7] += rewards.thinking.double().sum()
                interval[8] += rewards.repetition.double().sum()
                interval[9] += rollout.response_lengths.double().sum()
                interval[10] += rollout.has_eos.double().sum()
                interval[11] += advantage_output.group_stds.le(
                    args.advantage_eps
                ).double().sum()
                interval[12] += responses
                interval[13] += groups
                interval[14] += 1
                interval[15] += objective.clip_fraction.detach().double()

                del (
                    expanded_prompts,
                    rollout,
                    rewards,
                    advantage_output,
                    ref_log_probs,
                    policy_output,
                    current_log_probs,
                    objective,
                    router_aux,
                    total_loss,
                    scaled_loss,
                )

                if not should_update:
                    continue
                if needs_scaler:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(), args.grad_clip, error_if_nonfinite=True
                )
                latest_grad_norm = float(grad_norm.detach().float().item())
                if needs_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_interval == 0:
                    distributed_sum(interval, context)
                    if context.is_main:
                        micro_batches = max(interval[14].item(), 1.0)
                        response_count = max(interval[12].item(), 1.0)
                        group_count = max(interval[13].item(), 1.0)
                        metrics = {
                            "train/total_loss": interval[0].item() / micro_batches,
                            "train/policy_loss": interval[1].item() / micro_batches,
                            "train/kl_penalty": interval[2].item() / micro_batches,
                            "train/router_aux_loss": interval[3].item() / micro_batches,
                            "train/reward": interval[4].item() / response_count,
                            "train/reward_model": interval[5].item() / response_count,
                            "train/reward_length": interval[6].item() / response_count,
                            "train/reward_thinking": interval[7].item() / response_count,
                            "train/repetition_penalty": interval[8].item() / response_count,
                            "train/response_length": interval[9].item() / response_count,
                            "train/eos_rate": interval[10].item() / response_count,
                            "train/zero_std_group_rate": interval[11].item() / group_count,
                            "train/clip_fraction": interval[15].item() / micro_batches,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/grad_norm": latest_grad_norm,
                            "train/responses_per_second": response_count
                            / max(time.perf_counter() - interval_start, 1e-6),
                        }
                        print(
                            f"epoch={epoch + 1}/{training_epochs} "
                            f"step={global_step:,}/{total_steps:,} "
                            f"reward={metrics['train/reward']:.3f} "
                            f"loss={metrics['train/policy_loss']:.4f} "
                            f"kl={metrics['train/kl_penalty']:.4f} "
                            f"zero_groups={metrics['train/zero_std_group_rate']:.1%}"
                        )
                        tracker.log(metrics, step=global_step)
                    interval.zero_()
                    interval_start = time.perf_counter()

                if global_step % args.eval_interval == 0:
                    metrics = evaluate(
                        validation_loader, rollout_engine, reference_model,
                        reward_model, args, context, amp_dtype,
                    )
                    if context.is_main:
                        print(
                            f"validation step={global_step:,} "
                            f"reward={metrics['reward']:.3f} "
                            f"kl={metrics['sampled_kl']:.4f} "
                            f"eos={metrics['eos_rate']:.1%}"
                        )
                        tracker.log(
                            {f"validation/{key}": value for key, value in metrics.items()},
                            step=global_step,
                        )

                if global_step % args.save_interval == 0 and context.is_main:
                    save_checkpoint(
                        args.save_dir / "latest.pt", policy_model, optimizer,
                        scheduler, scaler, epoch, batch_index + 1, global_step,
                        args, tracker.state_dict(),
                    )
                if global_step >= total_steps:
                    stop_training = True
                    break

            start_batch = 0
            checkpoint_epoch = epoch if stop_training else epoch + 1
            checkpoint_batch = batch_index + 1 if stop_training else 0
            if context.is_main:
                save_checkpoint(
                    args.save_dir / "latest.pt", policy_model, optimizer,
                    scheduler, scaler, checkpoint_epoch, checkpoint_batch,
                    global_step, args, tracker.state_dict(),
                )
            if stop_training:
                break

        if context.distributed:
            torch.distributed.barrier()
        if context.is_main:
            export_pretrained(
                args.output_dir, policy_model, tokenizer, args.tokenizer_path
            )
            print(f"Training complete at optimizer step {global_step:,}.")
            print(f"Checkpoint    : {args.save_dir / 'latest.pt'}")
            print(f"Exported model: {args.output_dir}")
        finish_state = "success"
        finish_error = None
    except KeyboardInterrupt:
        finish_state = "aborted"
        finish_error = "GRPO interrupted by user."
        raise
    except BaseException:
        finish_error = traceback.format_exc()
        raise
    finally:
        try:
            tracker.finish(state=finish_state, error=finish_error)
        finally:
            cleanup_distributed(context)


if __name__ == "__main__":
    main()

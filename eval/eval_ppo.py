#!/usr/bin/env python3
"""Compare an exported PPO policy with its frozen pre-PPO reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lm_dataset import validate_tokenizer  # noqa: E402
from dataset.ppo_dataset import PPODataCollator, PPODataset  # noqa: E402
from eval.eval_sft import (  # noqa: E402
    autocast_context,
    resolve_device,
    resolve_dtype,
)
from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM  # noqa: E402
from trainer.rollout_engine import (  # noqa: E402
    RolloutBatch,
    TorchRolloutEngine,
    gather_completion_log_probs,
)
from trainer.train_ppo import (  # noqa: E402
    ExternalRewardScorer,
    calculate_rewards,
    resolve_reward_dtype,
)
from trainer.train_sft import ensure_finite_model_state  # noqa: E402
from trainer.trainer_utils import ensure_tokenizer_matches_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PPO and pre-PPO policies on held-out online rollouts."
    )
    parser.add_argument("--policy-path", type=Path, default=PROJECT_ROOT / "out" / "ppo")
    parser.add_argument(
        "--reference-path", type=Path, default=PROJECT_ROOT / "out" / "sft"
    )
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument(
        "--reward-model-path", type=Path, required=True
    )
    parser.add_argument(
        "--data-path",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "dataset" / "rl" / "rlaif.jsonl"],
    )
    parser.add_argument("--split", choices=["validation", "all"], default="validation")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--reward-clip", type=float, default=3.0)
    parser.add_argument("--repetition-penalty-cap", type=float, default=0.5)
    parser.add_argument("--missing-eos-penalty", type=float, default=0.2)
    parser.add_argument("--show-samples", type=int, default=4)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=4,
        help="Print progress every N evaluation batches.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--reward-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--attention-backend", choices=["eager", "auto", "math"], default="eager"
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    for field in (
        "eval_samples",
        "batch_size",
        "max_prompt_len",
        "max_new_tokens",
        "reward_clip",
    ):
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.show_samples < 0:
        raise ValueError("worker and displayed-sample counts cannot be negative")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")
    if not 0.0 < args.val_ratio < 0.5:
        raise ValueError("--val-ratio must be between 0 and 0.5")
    if args.temperature < 0 or not 0.0 < args.top_p <= 1.0 or args.top_k < 0:
        raise ValueError("invalid generation sampling parameters")
    if args.generation_repetition_penalty <= 0:
        raise ValueError("--generation-repetition-penalty must be positive")
    if args.repetition_penalty_cap < 0 or args.missing_eos_penalty < 0:
        raise ValueError("reward penalties cannot be negative")


def weight_fingerprint(path: Path) -> str:
    """Hash exported weight files so an unchanged PPO export is unmistakable."""

    weight_files = sorted(path.glob("*.safetensors"))
    if not weight_files:
        weight_files = sorted(path.glob("pytorch_model*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"No exported model weights found in {path}")
    digest = hashlib.sha256()
    for weight_file in weight_files:
        digest.update(weight_file.name.encode("utf-8"))
        with weight_file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_policy(
    path: Path,
    device: torch.device,
    attention_backend: str,
    label: str,
) -> MiniLLMForCausalLM:
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{label} config not found: {path / 'config.json'}")
    config = MiniLLMConfig.from_pretrained(path, local_files_only=True)
    config.attention_backend = attention_backend
    config.use_cache = False
    model = MiniLLMForCausalLM.from_pretrained(
        path, config=config, local_files_only=True
    ).to(device)
    model.eval().requires_grad_(False)
    ensure_finite_model_state(model, label)
    return model


def make_engine(
    model: MiniLLMForCausalLM,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> TorchRolloutEngine:
    return TorchRolloutEngine(
        model,
        tokenizer,
        device,
        max_prompt_len=args.max_prompt_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.generation_repetition_penalty,
        autocast_factory=lambda: autocast_context(device, dtype),
    )


def reset_sampling_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def reference_kl(
    policy_rollout: RolloutBatch,
    reference: MiniLLMForCausalLM,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float]:
    with autocast_context(device, dtype):
        logits = reference(
            input_ids=policy_rollout.input_ids,
            attention_mask=policy_rollout.attention_mask,
        ).logits
    reference_log_probs = gather_completion_log_probs(
        logits,
        policy_rollout.input_ids,
        policy_rollout.action_positions,
    )
    mask = policy_rollout.completion_mask.float()
    log_ratio = (reference_log_probs - policy_rollout.old_log_probs).clamp(-20.0, 20.0)
    sampled_kl = (log_ratio.exp() - log_ratio - 1.0) * mask
    return float(sampled_kl.sum().item()), float(mask.sum().item())


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    policy_engine: TorchRolloutEngine,
    reference_engine: TorchRolloutEngine,
    reference_model: MiniLLMForCausalLM,
    scorer: ExternalRewardScorer,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    policy_total: list[float] = []
    policy_model: list[float] = []
    policy_repetition: list[float] = []
    policy_lengths: list[float] = []
    policy_eos: list[float] = []
    policy_empty: list[float] = []
    reference_total: list[float] = []
    reference_model_rewards: list[float] = []
    reference_repetition: list[float] = []
    reference_lengths: list[float] = []
    reference_eos: list[float] = []
    reference_empty: list[float] = []
    reward_deltas: list[float] = []
    model_reward_deltas: list[float] = []
    kl_sum = 0.0
    kl_tokens = 0.0
    wins = ties = losses = 0
    examples: list[dict[str, str]] = []
    processed = 0
    started = time.perf_counter()
    total_samples = len(loader.dataset)
    print(
        "Evaluation      : generating PPO/SFT pairs and scoring rewards "
        f"(0/{total_samples})",
        flush=True,
    )

    for batch_index, batch in enumerate(loader):
        sampling_seed = args.seed + batch_index
        reset_sampling_seed(sampling_seed, device)
        policy_rollout = policy_engine.rollout(batch["prompts"])
        reset_sampling_seed(sampling_seed, device)
        reference_rollout = reference_engine.rollout(batch["prompts"])

        policy_rewards, policy_components = calculate_rewards(
            scorer,
            batch["messages"],
            policy_rollout,
            repetition_penalty_cap=args.repetition_penalty_cap,
            missing_eos_penalty=args.missing_eos_penalty,
        )
        reference_rewards, reference_components = calculate_rewards(
            scorer,
            batch["messages"],
            reference_rollout,
            repetition_penalty_cap=args.repetition_penalty_cap,
            missing_eos_penalty=args.missing_eos_penalty,
        )
        batch_kl, batch_kl_tokens = reference_kl(
            policy_rollout, reference_model, device, dtype
        )
        kl_sum += batch_kl
        kl_tokens += batch_kl_tokens

        policy_values = policy_rewards.float().cpu().tolist()
        reference_values = reference_rewards.float().cpu().tolist()
        policy_model_values = policy_components["model_reward"].float().cpu().tolist()
        reference_model_values = (
            reference_components["model_reward"].float().cpu().tolist()
        )
        policy_total.extend(policy_values)
        reference_total.extend(reference_values)
        policy_model.extend(policy_model_values)
        reference_model_rewards.extend(reference_model_values)
        policy_repetition.extend(
            policy_components["repetition_penalty"].float().cpu().tolist()
        )
        reference_repetition.extend(
            reference_components["repetition_penalty"].float().cpu().tolist()
        )
        policy_lengths.extend(policy_rollout.response_lengths.float().cpu().tolist())
        reference_lengths.extend(
            reference_rollout.response_lengths.float().cpu().tolist()
        )
        policy_eos.extend(policy_rollout.has_eos.float().cpu().tolist())
        reference_eos.extend(reference_rollout.has_eos.float().cpu().tolist())
        policy_empty.extend(
            [float(not response.strip()) for response in policy_rollout.responses]
        )
        reference_empty.extend(
            [float(not response.strip()) for response in reference_rollout.responses]
        )
        processed += len(policy_rollout.responses)

        for (
            prompt,
            messages,
            policy_response,
            reference_response,
            policy_reward,
            reference_reward,
            policy_rm,
            reference_rm,
        ) in zip(
            batch["prompts"],
            batch["messages"],
            policy_rollout.responses,
            reference_rollout.responses,
            policy_values,
            reference_values,
            policy_model_values,
            reference_model_values,
        ):
            delta = policy_reward - reference_reward
            reward_deltas.append(delta)
            model_reward_deltas.append(policy_rm - reference_rm)
            if delta > 1e-6:
                wins += 1
            elif delta < -1e-6:
                losses += 1
            else:
                ties += 1
            if len(examples) < args.show_samples:
                visible_prompt = (
                    str(messages[-1].get("content", "")) if messages else prompt
                )
                examples.append(
                    {
                        "prompt": visible_prompt,
                        "ppo_response": policy_response,
                        "sft_response": reference_response,
                        "ppo_reward": f"{policy_reward:.4f}",
                        "sft_reward": f"{reference_reward:.4f}",
                    }
                )

        completed_batches = batch_index + 1
        if (
            completed_batches % args.progress_interval == 0
            or processed >= total_samples
        ):
            elapsed = max(time.perf_counter() - started, 1e-6)
            rate = processed / elapsed
            remaining = max(0, total_samples - processed)
            eta = remaining / max(rate, 1e-9)
            print(
                f"Evaluation      : {processed}/{total_samples} "
                f"({processed / total_samples:.1%}), "
                f"{rate:.2f} samples/s, ETA {eta / 60:.1f} min",
                flush=True,
            )

    sample_count = len(policy_total)
    if not sample_count:
        raise RuntimeError("PPO evaluation produced no samples")
    if not all(
        math.isfinite(value)
        for values in (
            policy_total,
            policy_model,
            reference_total,
            reference_model_rewards,
            reward_deltas,
        )
        for value in values
    ):
        raise FloatingPointError("PPO evaluation produced non-finite metrics")

    result = {
        "samples": sample_count,
        "policy": {
            "total_reward": mean(policy_total),
            "model_reward": mean(policy_model),
            "response_tokens": mean(policy_lengths),
            "eos_rate": mean(policy_eos),
            "empty_rate": mean(policy_empty),
            "repetition_penalty": mean(policy_repetition),
        },
        "reference": {
            "total_reward": mean(reference_total),
            "model_reward": mean(reference_model_rewards),
            "response_tokens": mean(reference_lengths),
            "eos_rate": mean(reference_eos),
            "empty_rate": mean(reference_empty),
            "repetition_penalty": mean(reference_repetition),
        },
        "comparison": {
            "total_reward_delta": mean(reward_deltas),
            "model_reward_delta": mean(model_reward_deltas),
            "ppo_win_rate": wins / sample_count,
            "tie_rate": ties / sample_count,
            "ppo_loss_rate": losses / sample_count,
            "reference_kl_per_token": kl_sum / max(1.0, kl_tokens),
        },
    }
    return result, examples


def main() -> None:
    args = parse_args()
    validate_args(args)
    policy_path = project_path(args.policy_path)
    reference_path = project_path(args.reference_path)
    tokenizer_path = (
        policy_path if args.tokenizer_path is None else project_path(args.tokenizer_path)
    )
    reward_model_path = project_path(args.reward_model_path)
    data_paths = [project_path(path) for path in args.data_path]
    output_json = project_path(args.output_json) if args.output_json else None
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    reward_dtype = resolve_reward_dtype(args.reward_dtype, device)
    policy_sha = weight_fingerprint(policy_path)
    reference_sha = weight_fingerprint(reference_path)

    ensure_tokenizer_matches_models(
        tokenizer_path,
        {"PPO policy": policy_path, "SFT reference": reference_path},
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

    dataset: Any = PPODataset(
        data_paths=data_paths,
        tokenizer=tokenizer,
        max_prompt_len=args.max_prompt_len,
        split=args.split,
        val_ratio=args.val_ratio,
        expected_vocab_size=len(tokenizer),
        thinking_ratio=0.0,
        seed=args.seed,
    )
    dataset = Subset(dataset, range(min(args.eval_samples, len(dataset))))
    loader_options: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": PPODataCollator(),
    }
    if args.num_workers > 0:
        loader_options["multiprocessing_context"] = "spawn"
    loader = DataLoader(dataset, **loader_options)
    if not len(loader):
        raise ValueError("PPO evaluation DataLoader is empty")

    policy_model = load_policy(policy_path, device, args.attention_backend, "PPO policy")
    reference_model = load_policy(
        reference_path, device, args.attention_backend, "SFT reference"
    )
    if policy_model.config.vocab_size != len(tokenizer):
        raise ValueError("PPO policy and tokenizer vocabulary sizes do not match")
    if args.max_prompt_len + args.max_new_tokens > policy_model.config.max_position_embeddings:
        raise ValueError("Prompt plus completion exceeds policy position capacity")

    scorer = ExternalRewardScorer(
        str(reward_model_path),
        device,
        reward_dtype,
        reward_clip=args.reward_clip,
        allow_remote=False,
        trust_remote_code=True,
    )
    probe = scorer.score(
        [[{"role": "user", "content": "你好"}]],
        ["你好！有什么可以帮你？"],
    )
    print(f"PPO policy      : {policy_path}")
    print(f"SFT reference   : {reference_path}")
    print(f"Reward model    : {reward_model_path}")
    print(f"PPO weights SHA : {policy_sha}")
    print(f"SFT weights SHA : {reference_sha}")
    print(f"Weights changed : {'yes' if policy_sha != reference_sha else 'NO'}")
    print(f"Reward check    : finite (probe={float(probe[0].item()):.4f})")
    print(f"Device          : {device}")
    print(f"Policy precision: {str(dtype).removeprefix('torch.')}")
    print(f"Reward precision: {str(scorer.dtype).removeprefix('torch.')}")
    print(f"Samples         : {len(dataset)}")

    result, examples = evaluate(
        loader,
        make_engine(policy_model, tokenizer, device, dtype, args),
        make_engine(reference_model, tokenizer, device, dtype, args),
        reference_model,
        scorer,
        device,
        dtype,
        args,
    )
    report = {
        "policy_path": str(policy_path),
        "reference_path": str(reference_path),
        "reward_model_path": str(reward_model_path),
        "policy_weights_sha256": policy_sha,
        "reference_weights_sha256": reference_sha,
        "weights_changed": policy_sha != reference_sha,
        "data_paths": [str(path) for path in data_paths],
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        **result,
        "examples": examples,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print("\nPPO quality report")
    print(rendered)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved report    : {output_json}")


if __name__ == "__main__":
    main()

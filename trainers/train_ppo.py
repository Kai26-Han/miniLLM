#!/usr/bin/env python3
"""Native Actor-Critic PPO training for miniLLM after SFT or DPO.

The trainable Actor, frozen Reference and Critic backbone all start from the
same exported policy base model. A local reward model scores freshly generated
responses, while PPO Clip, value clipping, GAE and reference KL constrain each
online update.

Example from the project root::

    python trainer/train_ppo.py \
        --model-path out/sft \
        --data-path dataset/rl/rlaif.jsonl \
        --reward-model-path /path/to/internlm2-1_8b-reward
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import AutoConfig, AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lm_dataset import validate_tokenizer  # noqa: E402
from dataset.ppo_dataset import PPODataCollator, PPODataset  # noqa: E402
from model.model_minillm import (  # noqa: E402
    MiniLLMConfig,
    MiniLLMForCausalLM,
    MiniLLMModel,
)
from trainer.rollout_engine import (  # noqa: E402
    RolloutBatch,
    TorchRolloutEngine,
    gather_completion_log_probs,
)
from trainer.train_sft import (  # noqa: E402
    autocast_context,
    configure_attention_backend,
    ensure_finite_model_state,
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
    load_ppo_checkpoint,
    resolve_resume_path,
    save_ppo_checkpoint,
    seed_everything,
    setup_distributed,
    unwrap_model,
)


# InternLM2 reserves these embedding rows for its chat/reward protocol.  Some
# tokenizer snapshots keep placeholder names in tokenizer.model and register
# the public token strings as added tokens, which incorrectly assigns IDs just
# beyond the 92,544-row embedding table on recent Transformers releases.
INTERNLM2_PROTOCOL_TOKEN_IDS = {
    "<|reward|>": 92527,
    "<|plugin|>": 92538,
    "<|interpreter|>": 92539,
    "<|action_end|>": 92540,
    "<|action_start|>": 92541,
    "<|im_end|>": 92542,
    "<|im_start|>": 92543,
}


@dataclass
class CriticOutput:
    values: torch.Tensor
    router_aux_loss: torch.Tensor
    expert_counts: torch.Tensor | None = None
    router_prob_sums: torch.Tensor | None = None
    router_entropy_sum: torch.Tensor | None = None
    routed_token_count: torch.Tensor | None = None


class CriticModel(nn.Module):
    """A policy-initialized miniLLM decoder with a scalar value per token."""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.model = MiniLLMModel(config)
        self.value_head = nn.Linear(config.hidden_size, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def initialize_backbone(self, actor: MiniLLMForCausalLM) -> None:
        self.model.load_state_dict(actor.model.state_dict(), strict=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> CriticOutput:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return CriticOutput(
            values=self.value_head(output.last_hidden_state).squeeze(-1),
            router_aux_loss=output.router_aux_loss,
            expert_counts=output.expert_counts,
            router_prob_sums=output.router_prob_sums,
            router_entropy_sum=output.router_entropy_sum,
            routed_token_count=output.routed_token_count,
        )


class RewardScorer:
    """Small interface shared by the external and smoke-test reward backends."""

    def score(
        self,
        messages: Sequence[Sequence[Mapping[str, Any]]],
        responses: Sequence[str],
    ) -> torch.Tensor:
        raise NotImplementedError


def _normalize_legacy_rope_scaling(config: Any) -> None:
    """Keep old remote model code compatible with normalized RoPE configs.

    Recent Transformers releases normalize ``rope_scaling["type"]`` to
    ``rope_scaling["rope_type"]``.  InternLM2-Reward's remote modeling code
    still reads the legacy ``type``/``factor`` pair directly.  They may also
    turn an original ``rope_scaling=null`` into the internal
    ``{"rope_type": "default"}`` sentinel; the legacy model interprets every
    dictionary as enabled scaling and then fails while reading ``factor``.
    """

    rope_scaling = getattr(config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict):
        return
    rope_type = rope_scaling.get("type", rope_scaling.get("rope_type"))

    # InternLM2-1.8B-Reward ships with rope_scaling=null.  Preserve that
    # meaning when a newer Transformers release materializes its default
    # sentinel as a dictionary without a scaling factor.
    if "factor" not in rope_scaling:
        if rope_type in (None, "default"):
            config.rope_scaling = None
            return
        raise ValueError(
            "Reward model has an incomplete rope_scaling configuration: "
            f"{rope_scaling!r}. Expected both a scaling type and factor."
        )

    if rope_type is None:
        return
    rope_scaling.setdefault("type", rope_type)
    rope_scaling.setdefault("rope_type", rope_type)


def _sentencepiece_token_id(tokenizer: Any, token: str) -> int | None:
    """Return a token's original SentencePiece ID, rejecting unknown aliases."""

    sentencepiece = getattr(tokenizer, "sp_model", None)
    if sentencepiece is None:
        return None
    piece_to_id = getattr(sentencepiece, "piece_to_id", None) or getattr(
        sentencepiece, "PieceToId", None
    )
    id_to_piece = getattr(sentencepiece, "id_to_piece", None) or getattr(
        sentencepiece, "IdToPiece", None
    )
    if not callable(piece_to_id) or not callable(id_to_piece):
        return None
    token_id = int(piece_to_id(token))
    if token_id < 0 or str(id_to_piece(token_id)) != token:
        return None
    return token_id


def _build_reward_token_id_remap(
    tokenizer: Any,
    embedding_vocab_size: int,
) -> dict[int, int]:
    """Map duplicated added special tokens back to trained SentencePiece IDs."""

    remap: dict[int, int] = {}
    unresolved: list[tuple[str, int]] = []
    for token, raw_token_id in tokenizer.get_vocab().items():
        token_id = int(raw_token_id)
        if token_id < embedding_vocab_size:
            continue
        sentencepiece_id = _sentencepiece_token_id(tokenizer, str(token))
        if sentencepiece_id is None:
            sentencepiece_id = INTERNLM2_PROTOCOL_TOKEN_IDS.get(str(token))
        if sentencepiece_id is None or not 0 <= sentencepiece_id < embedding_vocab_size:
            unresolved.append((str(token), token_id))
            continue
        remap[token_id] = sentencepiece_id

    if unresolved:
        preview = ", ".join(
            f"{token!r}:{token_id}" for token, token_id in unresolved[:8]
        )
        raise ValueError(
            "Reward tokenizer contains token IDs outside the model embedding "
            f"table that have no trained SentencePiece alias: {preview}; "
            f"embedding_vocab_size={embedding_vocab_size}. Re-download the "
            "tokenizer and weights from the same reward-model snapshot."
        )
    return remap


@torch.no_grad()
def _restore_internlm2_rope_buffers(model: nn.Module) -> int:
    """Rebuild InternLM2's non-persistent RoPE frequencies in FP32.

    InternLM2 registers ``rotary_emb.inv_freq`` with ``persistent=False``, so
    those tensors are initialized by the trusted remote Python code rather
    than loaded from the checkpoint.  Fast/meta initialization in newer
    Transformers versions can leave these legacy buffers uninitialized.  A
    malformed finite value is enough to make ``cos``/``sin`` return NaN, and a
    non-finite value poisons the first attention layer immediately.
    """

    if getattr(getattr(model, "config", None), "model_type", None) != "internlm2":
        return 0

    restored = 0
    for module in model.modules():
        current = getattr(module, "inv_freq", None)
        dim = getattr(module, "dim", None)
        base = getattr(module, "base", None)
        if not isinstance(current, torch.Tensor) or dim is None or base is None:
            continue
        dim = int(dim)
        base = float(base)
        if dim <= 0 or dim % 2 or not math.isfinite(base) or base <= 0:
            raise ValueError(
                "Invalid InternLM2 rotary embedding configuration: "
                f"dim={dim}, base={base}."
            )
        exponent = torch.arange(
            0,
            dim,
            2,
            dtype=torch.float32,
            device=current.device,
        ) / dim
        inv_freq = torch.pow(
            torch.tensor(base, dtype=torch.float32, device=current.device),
            -exponent,
        )
        module.register_buffer("inv_freq", inv_freq, persistent=False)
        restored += 1

    expected = int(getattr(model.config, "num_hidden_layers", 0))
    if expected and restored != expected:
        raise RuntimeError(
            "Could not restore every InternLM2 RoPE buffer: "
            f"restored={restored}, expected={expected}."
        )
    return restored


class ExternalRewardScorer(RewardScorer):
    """Safe adapter for InternLM-style Hugging Face reward models."""

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
        *,
        reward_clip: float,
        allow_remote: bool,
        trust_remote_code: bool,
    ) -> None:
        load_options = {
            "local_files_only": not allow_remote,
            "trust_remote_code": trust_remote_code,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            **load_options,
        )
        reward_config = AutoConfig.from_pretrained(model_path, **load_options)
        _normalize_legacy_rope_scaling(reward_config)
        # Reward scoring is a single full-sequence forward pass.  It does not
        # benefit from a generation KV cache, and disabling it also avoids the
        # legacy InternLM2 remote code calling cache APIs removed by newer
        # Transformers releases.
        reward_config.use_cache = False
        self.model = AutoModel.from_pretrained(
            model_path,
            config=reward_config,
            torch_dtype=dtype if device.type == "cuda" else torch.float32,
            **load_options,
        ).to(device)
        self.model.eval().requires_grad_(False)
        self.device = device
        self.dtype = next(self.model.parameters()).dtype
        self.reward_clip = reward_clip
        self.probe_score: float | None = None
        self.restored_rope_buffers = _restore_internlm2_rope_buffers(self.model)

        bad_state_names = [
            name
            for name, tensor in (
                *self.model.named_parameters(),
                *self.model.named_buffers(),
            )
            if tensor.is_floating_point()
            and not bool(torch.isfinite(tensor).all().item())
        ]
        if bad_state_names:
            preview = ", ".join(bad_state_names[:8])
            suffix = " ..." if len(bad_state_names) > 8 else ""
            raise FloatingPointError(
                "Reward model contains non-finite parameters or buffers before "
                f"its first forward pass: {preview}{suffix}. Re-download the "
                "reward-model checkpoint."
            )

        input_embeddings = self.model.get_input_embeddings()
        if input_embeddings is None or not hasattr(input_embeddings, "weight"):
            raise TypeError("Reward model must expose token input embeddings.")
        self.embedding_vocab_size = int(input_embeddings.weight.shape[0])
        self.reward_token_id = int(
            getattr(
                self.model,
                "reward_token_id",
                getattr(self.model.config, "reward_token_id", -1),
            )
        )
        if not 0 <= self.reward_token_id < self.embedding_vocab_size:
            raise ValueError(
                "Reward token ID is outside the reward model embedding table: "
                f"reward_token_id={self.reward_token_id}, "
                f"embedding_vocab_size={self.embedding_vocab_size}."
            )

        self.token_id_remap = _build_reward_token_id_remap(
            self.tokenizer,
            self.embedding_vocab_size,
        )

        self.max_sequence_length = int(
            getattr(self.model.config, "max_position_embeddings", 32768)
        )
        if self.max_sequence_length < 2:
            raise ValueError(
                "Reward model max_position_embeddings must be at least 2, got "
                f"{self.max_sequence_length}."
            )

    @staticmethod
    def _reward_messages(
        messages: Sequence[Mapping[str, Any]], response: str
    ) -> list[dict[str, str]]:
        history = messages[:-1]
        latest = messages[-1].get("content", "") if messages else ""
        history_text = "\n".join(
            f"{message.get('role', '')}: {message.get('content', '')}"
            for message in history
        )
        query = (
            f"{history_text}\n以上是对话历史。我的新问题是：\n{latest}"
            if history_text
            else str(latest)
        )
        return [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ]

    def _encode_reward_input(
        self,
        messages: Sequence[Mapping[str, Any]],
        response: str,
        *,
        sample_index: int,
    ) -> torch.Tensor:
        """Render one reward conversation and validate IDs before CUDA lookup."""

        conversation = self._reward_messages(messages, response)
        rendered = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        token_ids = [
            self.token_id_remap.get(int(token_id), int(token_id))
            for token_id in self.tokenizer.encode(
                rendered,
                add_special_tokens=False,
            )
        ]
        if not token_ids or token_ids[-1] != self.reward_token_id:
            token_ids.append(self.reward_token_id)

        # Preserve the response and terminal reward token if two tokenizers
        # expand the same prompt differently or a malformed row is very long.
        if len(token_ids) > self.max_sequence_length:
            token_ids = token_ids[-self.max_sequence_length :]

        minimum_id = min(token_ids)
        maximum_id = max(token_ids)
        if minimum_id < 0 or maximum_id >= self.embedding_vocab_size:
            raise ValueError(
                "Reward input contains an out-of-range token before CUDA forward: "
                f"sample_index={sample_index}, sequence_length={len(token_ids)}, "
                f"token_id_range=[{minimum_id}, {maximum_id}], "
                f"embedding_vocab_size={self.embedding_vocab_size}."
            )
        return torch.tensor([token_ids], dtype=torch.long)

    @torch.no_grad()
    def score(
        self,
        messages: Sequence[Sequence[Mapping[str, Any]]],
        responses: Sequence[str],
    ) -> torch.Tensor:
        if len(messages) != len(responses):
            raise ValueError(
                "Reward messages and responses must have identical batch sizes."
            )
        scores: list[float] = []
        pairs = zip(messages, responses)
        for sample_index, (context, response) in enumerate(pairs):
            input_ids = self._encode_reward_input(
                context,
                response,
                sample_index=sample_index,
            )
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                use_cache=False,
                return_dict=True,
            )
            score_tensor = getattr(outputs, "logits", None)
            if score_tensor is None:
                score_tensor = outputs[0]
            if score_tensor.numel() != 1:
                raise ValueError(
                    "Reward model must return one scalar per conversation; "
                    f"sample_index={sample_index}, shape={tuple(score_tensor.shape)}."
                )
            score = float(score_tensor.detach().float().cpu().item())
            if not math.isfinite(score):
                input_min = int(input_ids.min().item())
                input_max = int(input_ids.max().item())
                raise FloatingPointError(
                    "Reward model returned a non-finite score before PPO update: "
                    f"sample_index={sample_index}, score={score}, "
                    f"reward_dtype={getattr(self, 'dtype', 'unknown')}, "
                    f"sequence_length={input_ids.shape[1]}, "
                    f"token_id_range=[{input_min}, {input_max}], "
                    f"embedding_vocab_size={self.embedding_vocab_size}."
                )
            scores.append(score)
        return torch.tensor(scores, device=self.device, dtype=torch.float32).clamp(
            -self.reward_clip, self.reward_clip
        )


class RuleRewardScorer(RewardScorer):
    """Dependency-free reward used only for tests and pipeline smoke runs."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @torch.no_grad()
    def score(
        self,
        messages: Sequence[Sequence[Mapping[str, Any]]],
        responses: Sequence[str],
    ) -> torch.Tensor:
        del messages
        values = [1.0 if 20 <= len(response.strip()) <= 800 else -0.5 for response in responses]
        return torch.tensor(values, device=self.device, dtype=torch.float32)


@dataclass
class PPOObjectiveOutput:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    reference_kl: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor
    value_clip_fraction: torch.Tensor
    actor_router_aux_loss: torch.Tensor
    critic_router_aux_loss: torch.Tensor


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.shape != mask.shape:
        raise ValueError("value and mask must have identical shapes")
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def masked_whiten(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = masked_mean(value, mask)
    variance = masked_mean((value - mean).square(), mask)
    return (value - mean) * torch.rsqrt(variance + 1e-8) * mask


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute terminal-aware token-level Generalized Advantage Estimation."""

    if rewards.shape != values.shape or rewards.shape != mask.shape:
        raise ValueError("rewards, values and mask must have identical shapes")
    if rewards.ndim != 2:
        raise ValueError("GAE tensors must have shape [batch, response]")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    rewards = rewards.float()
    values = values.float()
    mask = mask.float()
    advantages = torch.zeros_like(values)
    last_advantage = torch.zeros(values.shape[0], device=values.device)
    for position in reversed(range(values.shape[1])):
        current_mask = mask[:, position]
        if position + 1 < values.shape[1]:
            next_mask = mask[:, position + 1]
            next_value = values[:, position + 1] * next_mask
        else:
            next_mask = torch.zeros_like(current_mask)
            next_value = torch.zeros_like(current_mask)
        delta = (
            rewards[:, position] + gamma * next_value - values[:, position]
        ) * current_mask
        last_advantage = (
            delta + gamma * gae_lambda * last_advantage * next_mask
        ) * current_mask
        advantages[:, position] = last_advantage
    returns = (advantages + values) * mask
    return advantages, returns


def compute_ppo_objective(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_epsilon: float,
    value_clip: float,
    value_loss_coef: float,
    kl_coef: float,
    actor_router_aux_loss: torch.Tensor | None = None,
    critic_router_aux_loss: torch.Tensor | None = None,
) -> PPOObjectiveOutput:
    """Compute MiniMind-style clipped PPO, value and reference-KL losses."""

    tensors = (
        new_log_probs,
        old_log_probs,
        reference_log_probs,
        advantages,
        new_values,
        old_values,
        returns,
        mask,
    )
    if any(tensor.shape != mask.shape for tensor in tensors):
        raise ValueError("all PPO token tensors must have identical shapes")
    if clip_epsilon <= 0 or value_clip <= 0:
        raise ValueError("PPO clip ranges must be positive")
    if value_loss_coef < 0 or kl_coef < 0:
        raise ValueError("loss coefficients cannot be negative")

    new_log_probs = new_log_probs.float()
    old_log_probs = old_log_probs.float()
    reference_log_probs = reference_log_probs.float()
    advantages = advantages.float()
    new_values = new_values.float()
    old_values = old_values.float()
    returns = returns.float()
    mask = mask.float()

    log_ratio = (new_log_probs - old_log_probs).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    unclipped_policy = -advantages * ratio
    clipped_policy = -advantages * ratio.clamp(
        1.0 - clip_epsilon, 1.0 + clip_epsilon
    )
    policy_loss = masked_mean(torch.maximum(unclipped_policy, clipped_policy), mask)
    clip_fraction = masked_mean(
        ((ratio - 1.0).abs() > clip_epsilon).float(), mask
    )
    approximate_kl = 0.5 * masked_mean(log_ratio.square(), mask)

    reference_log_ratio = (reference_log_probs - new_log_probs).clamp(-20.0, 20.0)
    reference_kl = masked_mean(
        reference_log_ratio.exp() - reference_log_ratio - 1.0,
        mask,
    )

    clipped_values = old_values + (new_values - old_values).clamp(
        -value_clip, value_clip
    )
    value_loss_unclipped = (new_values - returns).square()
    value_loss_clipped = (clipped_values - returns).square()
    value_loss = 0.5 * masked_mean(
        torch.maximum(value_loss_unclipped, value_loss_clipped), mask
    )
    value_clip_fraction = masked_mean(
        ((new_values - old_values).abs() > value_clip).float(), mask
    )

    zero = policy_loss.new_zeros(())
    actor_aux = actor_router_aux_loss.float() if actor_router_aux_loss is not None else zero
    critic_aux = critic_router_aux_loss.float() if critic_router_aux_loss is not None else zero
    total_loss = (
        policy_loss
        + kl_coef * reference_kl
        + value_loss_coef * value_loss
        + actor_aux
        + critic_aux
    )
    return PPOObjectiveOutput(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        reference_kl=reference_kl,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
        value_clip_fraction=value_clip_fraction,
        actor_router_aux_loss=actor_aux,
        critic_router_aux_loss=critic_aux,
    )


def repetition_penalty(text: str, ngram: int = 3, cap: float = 0.5) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(tokens[index : index + ngram]) for index in range(len(tokens) - ngram + 1)]
    if not grams:
        return 0.0
    duplicates = len(grams) - len(set(grams))
    return min(cap, duplicates * cap * 2.0 / len(grams))


def calculate_rewards(
    scorer: RewardScorer,
    messages: Sequence[Sequence[Mapping[str, Any]]],
    rollout: RolloutBatch,
    *,
    repetition_penalty_cap: float,
    missing_eos_penalty: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model_reward = scorer.score(messages, rollout.responses)
    if model_reward.shape != (len(rollout.responses),):
        raise ValueError("reward scorer must return one scalar per response")
    repetition = torch.tensor(
        [
            repetition_penalty(response, cap=repetition_penalty_cap)
            for response in rollout.responses
        ],
        device=model_reward.device,
        dtype=torch.float32,
    )
    eos_penalty = (~rollout.has_eos).float() * missing_eos_penalty
    total = model_reward - repetition - eos_penalty
    return total, {
        "model_reward": model_reward,
        "repetition_penalty": repetition,
        "missing_eos_penalty": eos_penalty,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO training for miniLLM after SFT or DPO."
    )
    parser.add_argument(
        "--data-path",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "dataset" / "rl" / "rlaif.jsonl"],
    )
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "out" / "sft")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Tokenizer directory; defaults to --model-path.",
    )
    parser.add_argument(
        "--save-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "ppo"
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "out" / "ppo")

    parser.add_argument("--reward-backend", choices=["external", "rule"], default="external")
    parser.add_argument("--reward-model-path", type=str, default=None)
    parser.add_argument("--allow-remote-reward-model", action="store_true")
    parser.add_argument("--trust-reward-remote-code", action="store_true")
    parser.add_argument(
        "--reward-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help=(
            "External reward-model precision. InternLM2 Reward is published and "
            "used by MiniMind in float16; this is intentionally independent of "
            "the policy --dtype."
        ),
    )
    parser.add_argument("--reward-clip", type=float, default=3.0)
    parser.add_argument("--repetition-penalty-cap", type=float, default=0.5)
    parser.add_argument("--missing-eos-penalty", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-rollout-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2, help="Prompts per GPU rollout")
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-7)
    parser.add_argument("--actor-min-learning-rate", type=float, default=1e-8)
    parser.add_argument("--critic-learning-rate", type=float, default=5e-7)
    parser.add_argument("--critic-min-learning-rate", type=float, default=5e-8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-update-iters", type=int, default=2)
    parser.add_argument("--early-stop-kl", type=float, default=0.10)

    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--thinking-ratio", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume a PPO checkpoint, or use save-dir/latest.pt with --resume.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--attention-backend", choices=["eager", "auto", "math"], default="eager"
    )

    parser.add_argument("--tracker", choices=["none", "swanlab", "wandb"], default="none")
    parser.add_argument("--tracker-project", type=str, default="miniLLM-PPO")
    parser.add_argument("--tracker-run-name", type=str, default=None)
    parser.add_argument("--tracker-entity", type=str, default=None)
    parser.add_argument("--tracker-group", type=str, default=None)
    parser.add_argument("--tracker-tags", nargs="*", default=[])
    parser.add_argument("--tracker-mode", choices=["online", "offline"], default="online")
    parser.add_argument(
        "--tracker-log-dir", type=Path, default=PROJECT_ROOT / "logs" / "ppo"
    )
    parser.add_argument("--tracker-run-id", type=str, default=None)
    return parser.parse_args(argv)


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "epochs",
        "batch_size",
        "mini_batch_size",
        "accumulation_steps",
        "actor_learning_rate",
        "actor_min_learning_rate",
        "critic_learning_rate",
        "critic_min_learning_rate",
        "grad_clip",
        "clip_epsilon",
        "value_clip",
        "ppo_update_iters",
        "early_stop_kl",
        "max_prompt_len",
        "max_new_tokens",
        "reward_clip",
        "log_interval",
        "eval_interval",
        "save_interval",
    )
    for field in positive:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.reward_backend == "external" and not args.reward_model_path:
        raise ValueError("--reward-model-path is required for --reward-backend external")
    if args.max_rollout_steps < 0 or args.max_train_samples < 0:
        raise ValueError("step and sample limits cannot be negative")
    if args.eval_samples < 0 or args.eval_batches < 0 or args.num_workers < 0:
        raise ValueError("evaluation and worker limits cannot be negative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if not 0.0 < args.val_ratio < 0.5 or not 0.0 <= args.thinking_ratio <= 1.0:
        raise ValueError("invalid validation or thinking ratio")
    if not 0.0 <= args.gamma <= 1.0 or not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("gamma and GAE lambda must be in [0, 1]")
    if args.value_loss_coef < 0 or args.kl_coef < 0:
        raise ValueError("loss coefficients cannot be negative")
    if args.repetition_penalty_cap < 0 or args.missing_eos_penalty < 0:
        raise ValueError("reward penalties cannot be negative")
    if args.temperature < 0 or not 0.0 < args.top_p <= 1.0 or args.top_k < 0:
        raise ValueError("invalid sampling parameters")
    if args.actor_min_learning_rate > args.actor_learning_rate:
        raise ValueError("actor minimum learning rate exceeds initial rate")
    if args.critic_min_learning_rate > args.critic_learning_rate:
        raise ValueError("critic minimum learning rate exceeds initial rate")


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> AdamW:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )


def resolve_reward_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Resolve reward inference precision independently from policy AMP."""

    if device.type != "cuda":
        return torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[requested]


def make_dataloaders(
    args: argparse.Namespace,
    tokenizer: Any,
    context: DistributedContext,
) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    common = {
        "data_paths": args.data_path,
        "tokenizer": tokenizer,
        "max_prompt_len": args.max_prompt_len,
        "val_ratio": args.val_ratio,
        "expected_vocab_size": len(tokenizer),
        "thinking_ratio": args.thinking_ratio,
        "seed": args.seed,
    }
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
    options: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": context.device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": PPODataCollator(),
    }
    if args.num_workers > 0:
        options["multiprocessing_context"] = "spawn"
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=False,
        drop_last=True,
        **options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        sampler=validation_sampler,
        shuffle=False,
        drop_last=False,
        **options,
    )
    if len(train_loader) == 0:
        raise ValueError("Training DataLoader is empty; reduce --batch-size")
    if len(validation_loader) == 0:
        raise ValueError("Validation DataLoader is empty")
    return train_loader, validation_loader, train_sampler


def _gather_values(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return values.gather(1, positions)


@torch.no_grad()
def _reference_log_probs(
    reference_model: nn.Module,
    rollout: RolloutBatch,
    context: DistributedContext,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    with autocast_context(context.device, amp_dtype):
        logits = reference_model(
            input_ids=rollout.input_ids,
            attention_mask=rollout.attention_mask,
        ).logits
    return gather_completion_log_probs(
        logits, rollout.input_ids, rollout.action_positions
    ) * rollout.completion_mask


@torch.no_grad()
def _old_values(
    critic_model: nn.Module,
    rollout: RolloutBatch,
    context: DistributedContext,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    raw_critic = unwrap_model(critic_model)
    was_training = raw_critic.training
    raw_critic.eval()
    try:
        with autocast_context(context.device, amp_dtype):
            values = raw_critic(
                rollout.input_ids, rollout.attention_mask
            ).values
        return _gather_values(values, rollout.action_positions).float() * rollout.completion_mask
    finally:
        raw_critic.train(was_training)


def _terminal_token_rewards(
    sequence_rewards: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    token_rewards = torch.zeros_like(mask, dtype=torch.float32)
    lengths = mask.sum(dim=1).long()
    if bool((lengths <= 0).any().item()):
        raise ValueError("each PPO trajectory needs at least one valid response token")
    token_rewards[
        torch.arange(mask.shape[0], device=mask.device), lengths - 1
    ] = sequence_rewards
    return token_rewards


def _mean_across_processes(value: torch.Tensor, context: DistributedContext) -> float:
    result = value.detach().float().clone()
    if context.distributed:
        dist.all_reduce(result, op=dist.ReduceOp.AVG)
    return float(result.item())


def _divide_gradients(model: nn.Module, divisor: int) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(divisor)


def _optimizer_step(
    actor_model: nn.Module,
    critic_model: nn.Module,
    actor_optimizer: AdamW,
    critic_optimizer: AdamW,
    actor_scheduler: Any,
    critic_scheduler: Any,
    scaler: Any,
    *,
    needs_scaler: bool,
    accumulation_count: int,
    grad_clip: float,
) -> tuple[float, float]:
    if accumulation_count <= 0:
        return 0.0, 0.0
    if needs_scaler:
        scaler.unscale_(actor_optimizer)
        scaler.unscale_(critic_optimizer)
    _divide_gradients(actor_model, accumulation_count)
    _divide_gradients(critic_model, accumulation_count)
    actor_norm = torch.nn.utils.clip_grad_norm_(
        actor_model.parameters(), grad_clip, error_if_nonfinite=not needs_scaler
    )
    critic_norm = torch.nn.utils.clip_grad_norm_(
        critic_model.parameters(), grad_clip, error_if_nonfinite=not needs_scaler
    )
    if needs_scaler:
        scaler.step(actor_optimizer)
        scaler.step(critic_optimizer)
        scaler.update()
    else:
        actor_optimizer.step()
        critic_optimizer.step()
    actor_scheduler.step()
    critic_scheduler.step()
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    return float(actor_norm.detach().float()), float(critic_norm.detach().float())


def _build_reward_scorer(
    args: argparse.Namespace,
    context: DistributedContext,
    reward_dtype: torch.dtype,
) -> RewardScorer:
    if args.reward_backend == "rule":
        return RuleRewardScorer(context.device)
    scorer = ExternalRewardScorer(
        args.reward_model_path,
        context.device,
        reward_dtype,
        reward_clip=args.reward_clip,
        allow_remote=args.allow_remote_reward_model,
        trust_remote_code=args.trust_reward_remote_code,
    )
    if context.is_main and scorer.token_id_remap:
        print(
            "Reward ID map  : "
            f"{len(scorer.token_id_remap)} duplicated special tokens restored"
        )
    if context.is_main and scorer.restored_rope_buffers:
        print(
            "Reward RoPE    : "
            f"{scorer.restored_rope_buffers} FP32 buffers rebuilt"
        )
    probe = scorer.score(
        [[{"role": "user", "content": "你好"}]],
        ["你好！有什么可以帮你？"],
    )
    scorer.probe_score = float(probe[0].item())
    return scorer


def _resume_compatibility(checkpoint: dict[str, Any], args: argparse.Namespace) -> None:
    saved = checkpoint.get("args", {})
    checks = {
        "model_path": str(args.model_path),
        "reward_backend": args.reward_backend,
        "reward_model_path": args.reward_model_path,
        "reward_dtype": args.reward_dtype,
        "max_prompt_len": args.max_prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "clip_epsilon": args.clip_epsilon,
        "value_clip": args.value_clip,
        "kl_coef": args.kl_coef,
    }
    for key, current in checks.items():
        previous = saved.get(key)
        if previous is not None and str(previous) != str(current):
            raise ValueError(
                f"Cannot resume with changed {key}: checkpoint={previous!r}, current={current!r}"
            )


@torch.no_grad()
def evaluate_rollouts(
    loader: DataLoader,
    rollout_engine: TorchRolloutEngine,
    reference_model: nn.Module,
    scorer: RewardScorer,
    context: DistributedContext,
    amp_dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, float]:
    totals = torch.zeros(8, device=context.device, dtype=torch.float64)
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.eval_batches:
            break
        rollout = rollout_engine.rollout(batch["prompts"])
        rewards, components = calculate_rewards(
            scorer,
            batch["messages"],
            rollout,
            repetition_penalty_cap=args.repetition_penalty_cap,
            missing_eos_penalty=args.missing_eos_penalty,
        )
        reference_log_probs = _reference_log_probs(
            reference_model, rollout, context, amp_dtype
        )
        ref_ratio = (reference_log_probs - rollout.old_log_probs).clamp(-20.0, 20.0)
        reference_kl = masked_mean(
            ref_ratio.exp() - ref_ratio - 1.0, rollout.completion_mask
        )
        count = len(rollout.responses)
        totals[0] += rewards.double().sum()
        totals[1] += components["model_reward"].double().sum()
        totals[2] += reference_kl.double() * count
        totals[3] += rollout.response_lengths.double().sum()
        totals[4] += rollout.has_eos.double().sum()
        totals[5] += batch["prompt_truncated"].sum().to(context.device)
        totals[6] += count
        totals[7] += components["repetition_penalty"].double().sum()
    distributed_sum(totals, context)
    count = totals[6].clamp_min(1.0)
    return {
        "reward": (totals[0] / count).item(),
        "model_reward": (totals[1] / count).item(),
        "reference_kl": (totals[2] / count).item(),
        "response_length": (totals[3] / count).item(),
        "eos_ratio": (totals[4] / count).item(),
        "prompt_truncated_ratio": (totals[5] / count).item(),
        "repetition_penalty": (totals[7] / count).item(),
        "samples": totals[6].item(),
    }


def _print_setup(
    args: argparse.Namespace,
    actor_model: nn.Module,
    critic_model: nn.Module,
    train_loader: DataLoader,
    context: DistributedContext,
    max_rollouts: int,
    max_optimizer_steps: int,
    amp_dtype: torch.dtype,
    scorer: RewardScorer,
) -> None:
    if not context.is_main:
        return
    actor_parameters = sum(parameter.numel() for parameter in actor_model.parameters())
    critic_parameters = sum(parameter.numel() for parameter in critic_model.parameters())
    print(f"Policy base    : {args.model_path}")
    print(f"Tokenizer      : {args.tokenizer_path}")
    print(f"Tokenizer SHA  : {args.tokenizer_fingerprint}")
    print(f"Reward backend : {args.reward_backend}")
    if args.reward_model_path:
        print(f"Reward model   : {args.reward_model_path}")
    print(f"Device         : {context.device} (world_size={context.world_size})")
    print(f"Policy dtype   : {str(amp_dtype).removeprefix('torch.')}")
    if isinstance(scorer, ExternalRewardScorer):
        print(f"Reward dtype   : {str(scorer.dtype).removeprefix('torch.')}")
        print(f"Reward check   : finite (score={scorer.probe_score:.4f})")
    print(f"Architecture   : {'MoE' if unwrap_model(actor_model).config.use_moe else 'Dense'}")
    print(f"Actor params   : {actor_parameters:,}")
    print(f"Critic params  : {critic_parameters:,}")
    print(f"Prompt/response: {args.max_prompt_len}/{args.max_new_tokens}")
    print(f"Rollout batch  : {args.batch_size} prompts/GPU")
    print(f"Train batches  : {len(train_loader):,} per rank")
    print(f"Rollout steps  : {max_rollouts:,}")
    print(f"Max opt steps  : {max_optimizer_steps:,}")


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
    tracker_state = "crashed"
    tracker_error: str | None = "PPO stopped before completion."
    try:
        seed_everything(args.seed, context.rank)
        if context.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        configure_attention_backend(args.attention_backend, context)
        if not (args.model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"Policy base config not found: {args.model_path / 'config.json'}. "
                "Finish SFT or DPO and export a Transformers model before PPO."
            )

        args.tokenizer_fingerprint = ensure_tokenizer_matches_models(
            args.tokenizer_path,
            {"PPO actor/reference base": args.model_path},
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, local_files_only=True, use_fast=True
        )
        template_path = args.tokenizer_path / "chat_template.jinja"
        if template_path.is_file():
            tokenizer.chat_template = template_path.read_text(encoding="utf-8")
        if not tokenizer.chat_template:
            raise ValueError("PPO tokenizer has no chat template")
        validate_tokenizer(tokenizer, expected_vocab_size=len(tokenizer))
        train_loader, validation_loader, train_sampler = make_dataloaders(
            args, tokenizer, context
        )

        actor_config = MiniLLMConfig.from_pretrained(args.model_path, local_files_only=True)
        actor_config.tokenizer_fingerprint = args.tokenizer_fingerprint
        actor_config.attention_backend = args.attention_backend
        actor_config.use_cache = False
        actor_model: nn.Module = MiniLLMForCausalLM.from_pretrained(
            args.model_path,
            config=actor_config,
            local_files_only=True,
        )
        reference_config = MiniLLMConfig.from_pretrained(
            args.model_path, local_files_only=True
        )
        reference_config.tokenizer_fingerprint = args.tokenizer_fingerprint
        reference_config.attention_backend = args.attention_backend
        reference_config.use_cache = False
        reference_model: nn.Module = MiniLLMForCausalLM.from_pretrained(
            args.model_path,
            config=reference_config,
            local_files_only=True,
        )
        critic_config = MiniLLMConfig.from_pretrained(args.model_path, local_files_only=True)
        critic_config.tokenizer_fingerprint = args.tokenizer_fingerprint
        critic_config.attention_backend = args.attention_backend
        critic_config.use_cache = False
        if actor_config.attention_dropout or actor_config.hidden_dropout:
            raise ValueError(
                "PPO requires attention_dropout=hidden_dropout=0 so rollout and "
                "first-update log probabilities are identical."
            )
        critic_model: nn.Module = CriticModel(critic_config)
        critic_model.initialize_backbone(actor_model)
        if actor_config.vocab_size != len(tokenizer):
            raise ValueError("Policy base model and tokenizer vocabulary sizes differ")
        if args.max_prompt_len + args.max_new_tokens > actor_config.max_position_embeddings:
            raise ValueError("prompt + response length exceeds model position capacity")

        actor_model = actor_model.to(context.device)
        reference_model = reference_model.to(context.device).eval().requires_grad_(False)
        critic_model = critic_model.to(context.device)
        ensure_finite_model_state(actor_model, "PPO Actor")
        ensure_finite_model_state(reference_model, "PPO Reference")
        ensure_finite_model_state(critic_model, "Policy-initialized Critic")
        if args.gradient_checkpointing:
            actor_model.model.gradient_checkpointing = True
            critic_model.model.gradient_checkpointing = True

        amp_dtype, needs_scaler = resolve_amp(args.dtype, context)
        reward_dtype = resolve_reward_dtype(args.reward_dtype, context.device)
        scorer = _build_reward_scorer(args, context, reward_dtype)
        autocast_factory = lambda: autocast_context(context.device, amp_dtype)
        rollout_engine = TorchRolloutEngine(
            actor_model,
            tokenizer,
            context.device,
            max_prompt_len=args.max_prompt_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.generation_repetition_penalty,
            autocast_factory=autocast_factory,
        )
        eval_engine = TorchRolloutEngine(
            actor_model,
            tokenizer,
            context.device,
            max_prompt_len=args.max_prompt_len,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=args.generation_repetition_penalty,
            autocast_factory=autocast_factory,
        )

        if context.distributed:
            actor_model = DistributedDataParallel(
                actor_model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            critic_model = DistributedDataParallel(
                critic_model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            rollout_engine.policy_model = actor_model
            eval_engine.policy_model = actor_model

        actor_optimizer = build_optimizer(
            actor_model, args.actor_learning_rate, args.weight_decay
        )
        critic_optimizer = build_optimizer(
            critic_model, args.critic_learning_rate, args.weight_decay
        )
        available_rollouts = args.epochs * len(train_loader)
        max_rollouts = (
            min(args.max_rollout_steps, available_rollouts)
            if args.max_rollout_steps
            else available_rollouts
        )
        minibatches_per_rollout = math.ceil(args.batch_size / args.mini_batch_size)
        updates_per_rollout = math.ceil(
            args.ppo_update_iters
            * minibatches_per_rollout
            / args.accumulation_steps
        )
        max_optimizer_steps = max(1, max_rollouts * updates_per_rollout)
        actor_scheduler = build_cosine_scheduler(
            actor_optimizer,
            max_optimizer_steps,
            args.warmup_ratio,
            args.actor_min_learning_rate / args.actor_learning_rate,
        )
        critic_scheduler = build_cosine_scheduler(
            critic_optimizer,
            max_optimizer_steps,
            args.warmup_ratio,
            args.critic_min_learning_rate / args.critic_learning_rate,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)

        start_epoch = 0
        start_batch = 0
        rollout_step = 0
        global_step = 0
        checkpoint: dict[str, Any] = {}
        resume_path = resolve_resume_path(args.save_dir, args.resume)
        if resume_path is not None:
            checkpoint = load_ppo_checkpoint(
                resume_path,
                actor_model,
                critic_model,
                actor_optimizer,
                critic_optimizer,
                actor_scheduler,
                critic_scheduler,
                scaler,
            )
            _resume_compatibility(checkpoint, args)
            ensure_checkpoint_tokenizer_fingerprint(
                checkpoint, args.tokenizer_fingerprint
            )
            start_epoch = int(checkpoint.get("epoch", 0))
            start_batch = int(checkpoint.get("batch_in_epoch", 0))
            rollout_step = int(checkpoint.get("rollout_step", 0))
            global_step = int(checkpoint.get("global_step", 0))
            if context.is_main:
                print(f"Resumed        : {resume_path} (rollout={rollout_step}, step={global_step})")

        checkpoint_tracker = checkpoint.get("tracker", {})
        restored_tracker_id = (
            checkpoint_tracker.get("run_id")
            if checkpoint_tracker.get("backend") == args.tracker
            else None
        )
        run_name = args.tracker_run_name or (
            f"miniLLM-PPO-BS{args.batch_size}-LR{args.actor_learning_rate:g}-KL{args.kl_coef:g}"
        )
        tracker = init_experiment_tracker(
            backend=args.tracker,
            context=context,
            project=args.tracker_project,
            run_name=run_name,
            entity=args.tracker_entity,
            group=args.tracker_group,
            tags=args.tracker_tags,
            mode=args.tracker_mode,
            log_dir=args.tracker_log_dir,
            config=vars(args).copy(),
            run_id=args.tracker_run_id or restored_tracker_id,
            strict_resume=bool(restored_tracker_id and not args.tracker_run_id),
        )
        _print_setup(
            args,
            actor_model,
            critic_model,
            train_loader,
            context,
            max_rollouts,
            max_optimizer_steps,
            amp_dtype,
            scorer,
        )

        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        stop_training = rollout_step >= max_rollouts
        latest_actor_grad_norm = 0.0
        latest_critic_grad_norm = 0.0
        use_moe = bool(unwrap_model(actor_model).config.use_moe)
        num_experts = int(unwrap_model(actor_model).config.num_experts) if use_moe else 0

        for epoch in range(start_epoch, args.epochs):
            if stop_training:
                break
            train_sampler.set_epoch(epoch)
            skip_before = start_batch if epoch == start_epoch else 0
            for batch_index, batch in enumerate(train_loader):
                if batch_index < skip_before:
                    continue
                rollout_started = time.perf_counter()
                rollout = rollout_engine.rollout(batch["prompts"])
                rewards, reward_components = calculate_rewards(
                    scorer,
                    batch["messages"],
                    rollout,
                    repetition_penalty_cap=args.repetition_penalty_cap,
                    missing_eos_penalty=args.missing_eos_penalty,
                )
                reference_log_probs = _reference_log_probs(
                    reference_model, rollout, context, amp_dtype
                )
                old_values = _old_values(critic_model, rollout, context, amp_dtype)
                token_rewards = _terminal_token_rewards(
                    rewards, rollout.completion_mask
                )
                raw_advantages, returns = compute_gae(
                    token_rewards,
                    old_values,
                    rollout.completion_mask,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                )
                advantages = masked_whiten(
                    raw_advantages, rollout.completion_mask
                )

                metric_sums = torch.zeros(9, device=context.device, dtype=torch.float64)
                actor_expert_counts = torch.zeros(
                    num_experts, device=context.device, dtype=torch.float64
                )
                critic_expert_counts = torch.zeros_like(actor_expert_counts)
                actor_router_totals = torch.zeros(
                    2, device=context.device, dtype=torch.float64
                )
                critic_router_totals = torch.zeros_like(actor_router_totals)
                metric_count = 0
                accumulation_count = 0
                stopped_for_kl = False
                batch_size = rollout.input_ids.shape[0]
                for _ in range(args.ppo_update_iters):
                    permutation = torch.randperm(batch_size, device=context.device)
                    for start in range(0, batch_size, args.mini_batch_size):
                        indices = permutation[start : start + args.mini_batch_size]
                        mini_input_ids = rollout.input_ids[indices]
                        mini_attention = rollout.attention_mask[indices]
                        mini_positions = rollout.action_positions[indices]
                        mini_mask = rollout.completion_mask[indices]
                        with autocast_context(context.device, amp_dtype):
                            actor_output = actor_model(
                                input_ids=mini_input_ids,
                                attention_mask=mini_attention,
                            )
                            critic_output = critic_model(
                                input_ids=mini_input_ids,
                                attention_mask=mini_attention,
                            )
                        new_log_probs = gather_completion_log_probs(
                            actor_output.logits, mini_input_ids, mini_positions
                        )
                        new_values = _gather_values(
                            critic_output.values, mini_positions
                        )
                        objective = compute_ppo_objective(
                            new_log_probs,
                            rollout.old_log_probs[indices],
                            reference_log_probs[indices],
                            advantages[indices],
                            new_values,
                            old_values[indices],
                            returns[indices],
                            mini_mask,
                            clip_epsilon=args.clip_epsilon,
                            value_clip=args.value_clip,
                            value_loss_coef=args.value_loss_coef,
                            kl_coef=args.kl_coef,
                            actor_router_aux_loss=actor_output.router_aux_loss,
                            critic_router_aux_loss=critic_output.router_aux_loss,
                        )
                        if not bool(torch.isfinite(objective.total_loss).item()):
                            raise FloatingPointError(
                                f"Non-finite PPO loss at source rows {batch['source_indices'].tolist()}"
                            )
                        synchronized_kl = _mean_across_processes(
                            objective.approximate_kl, context
                        )
                        if synchronized_kl > args.early_stop_kl:
                            stopped_for_kl = True
                            break
                        if needs_scaler:
                            scaler.scale(objective.total_loss).backward()
                        else:
                            objective.total_loss.backward()
                        accumulation_count += 1
                        metric_sums += torch.stack(
                            (
                                objective.total_loss.detach(),
                                objective.policy_loss.detach(),
                                objective.value_loss.detach(),
                                objective.reference_kl.detach(),
                                objective.approximate_kl.detach(),
                                objective.clip_fraction.detach(),
                                objective.value_clip_fraction.detach(),
                                objective.actor_router_aux_loss.detach(),
                                objective.critic_router_aux_loss.detach(),
                            )
                        ).double()
                        if use_moe:
                            actor_expert_counts += actor_output.expert_counts.double()
                            critic_expert_counts += critic_output.expert_counts.double()
                            actor_router_totals += torch.stack(
                                (
                                    actor_output.router_entropy_sum.detach(),
                                    actor_output.routed_token_count.detach(),
                                )
                            ).double()
                            critic_router_totals += torch.stack(
                                (
                                    critic_output.router_entropy_sum.detach(),
                                    critic_output.routed_token_count.detach(),
                                )
                            ).double()
                        metric_count += 1
                        if accumulation_count == args.accumulation_steps:
                            latest_actor_grad_norm, latest_critic_grad_norm = _optimizer_step(
                                actor_model,
                                critic_model,
                                actor_optimizer,
                                critic_optimizer,
                                actor_scheduler,
                                critic_scheduler,
                                scaler,
                                needs_scaler=needs_scaler,
                                accumulation_count=accumulation_count,
                                grad_clip=args.grad_clip,
                            )
                            accumulation_count = 0
                            global_step += 1
                    if stopped_for_kl:
                        break
                if accumulation_count:
                    latest_actor_grad_norm, latest_critic_grad_norm = _optimizer_step(
                        actor_model,
                        critic_model,
                        actor_optimizer,
                        critic_optimizer,
                        actor_scheduler,
                        critic_scheduler,
                        scaler,
                        needs_scaler=needs_scaler,
                        accumulation_count=accumulation_count,
                        grad_clip=args.grad_clip,
                    )
                    global_step += 1

                rollout_step += 1
                if rollout_step % args.log_interval == 0:
                    count_tensor = torch.tensor(
                        float(metric_count), device=context.device, dtype=torch.float64
                    )
                    distributed_sum(metric_sums, context)
                    distributed_sum(count_tensor, context)
                    if use_moe:
                        distributed_sum(actor_expert_counts, context)
                        distributed_sum(critic_expert_counts, context)
                        distributed_sum(actor_router_totals, context)
                        distributed_sum(critic_router_totals, context)
                    denominator = count_tensor.clamp_min(1.0)
                    elapsed = max(time.perf_counter() - rollout_started, 1e-6)
                    metrics = {
                        "train/total_loss": (metric_sums[0] / denominator).item(),
                        "train/policy_loss": (metric_sums[1] / denominator).item(),
                        "train/value_loss": (metric_sums[2] / denominator).item(),
                        "train/reference_kl": (metric_sums[3] / denominator).item(),
                        "train/approximate_kl": (metric_sums[4] / denominator).item(),
                        "train/clip_fraction": (metric_sums[5] / denominator).item(),
                        "train/value_clip_fraction": (metric_sums[6] / denominator).item(),
                        "train/actor_router_aux_loss": (metric_sums[7] / denominator).item(),
                        "train/critic_router_aux_loss": (metric_sums[8] / denominator).item(),
                        "train/reward": _mean_across_processes(rewards.mean(), context),
                        "train/model_reward": _mean_across_processes(
                            reward_components["model_reward"].mean(), context
                        ),
                        "train/repetition_penalty": _mean_across_processes(
                            reward_components["repetition_penalty"].mean(), context
                        ),
                        "train/missing_eos_penalty": _mean_across_processes(
                            reward_components["missing_eos_penalty"].mean(), context
                        ),
                        "train/reward_std": _mean_across_processes(
                            rewards.std(unbiased=False), context
                        ),
                        "train/raw_advantage_mean": _mean_across_processes(
                            masked_mean(raw_advantages, rollout.completion_mask), context
                        ),
                        "train/response_length": _mean_across_processes(
                            rollout.response_lengths.float().mean(), context
                        ),
                        "train/eos_ratio": _mean_across_processes(
                            rollout.has_eos.float().mean(), context
                        ),
                        "train/prompt_truncated_ratio": _mean_across_processes(
                            batch["prompt_truncated"].float().mean().to(context.device), context
                        ),
                        "train/actor_learning_rate": actor_scheduler.get_last_lr()[0],
                        "train/critic_learning_rate": critic_scheduler.get_last_lr()[0],
                        "train/actor_grad_norm": latest_actor_grad_norm,
                        "train/critic_grad_norm": latest_critic_grad_norm,
                        "train/response_tokens_per_second": _mean_across_processes(
                            rollout.response_lengths.float().sum() / elapsed, context
                        ),
                        "train/early_stopped_for_kl": float(stopped_for_kl),
                        "train/optimizer_step": global_step,
                    }
                    if use_moe:
                        actor_usage = actor_expert_counts / actor_expert_counts.sum().clamp_min(1.0)
                        critic_usage = critic_expert_counts / critic_expert_counts.sum().clamp_min(1.0)
                        normalizer = max(math.log(num_experts), 1.0)
                        metrics.update(
                            {
                                "moe/actor_max_load_ratio": actor_usage.max().item(),
                                "moe/actor_min_load_ratio": actor_usage.min().item(),
                                "moe/critic_max_load_ratio": critic_usage.max().item(),
                                "moe/critic_min_load_ratio": critic_usage.min().item(),
                                "moe/actor_router_entropy_normalized": (
                                    actor_router_totals[0]
                                    / actor_router_totals[1].clamp_min(1.0)
                                    / normalizer
                                ).item(),
                                "moe/critic_router_entropy_normalized": (
                                    critic_router_totals[0]
                                    / critic_router_totals[1].clamp_min(1.0)
                                    / normalizer
                                ).item(),
                            }
                        )
                    if context.is_main:
                        print(
                            f"epoch={epoch + 1}/{args.epochs} rollout={rollout_step}/{max_rollouts} "
                            f"step={global_step} reward={metrics['train/reward']:.4f} "
                            f"kl_ref={metrics['train/reference_kl']:.4f} "
                            f"kl_old={metrics['train/approximate_kl']:.4f} "
                            f"clip={metrics['train/clip_fraction']:.1%}"
                        )
                        tracker.log(metrics, step=rollout_step)

                if args.eval_batches and rollout_step % args.eval_interval == 0:
                    validation = evaluate_rollouts(
                        validation_loader,
                        eval_engine,
                        reference_model,
                        scorer,
                        context,
                        amp_dtype,
                        args,
                    )
                    if context.is_main:
                        print(
                            f"validation rollout={rollout_step} reward={validation['reward']:.4f} "
                            f"kl_ref={validation['reference_kl']:.4f} "
                            f"eos={validation['eos_ratio']:.1%}"
                        )
                        tracker.log(
                            {f"validation/{key}": value for key, value in validation.items()},
                            step=rollout_step,
                        )

                if rollout_step % args.save_interval == 0:
                    if context.is_main:
                        save_ppo_checkpoint(
                            args.save_dir / "latest.pt",
                            actor_model,
                            critic_model,
                            actor_optimizer,
                            critic_optimizer,
                            actor_scheduler,
                            critic_scheduler,
                            scaler,
                            epoch=epoch,
                            batch_in_epoch=batch_index + 1,
                            rollout_step=rollout_step,
                            global_step=global_step,
                            args=args,
                            tracker_state=tracker.state_dict(),
                        )
                    if context.distributed:
                        dist.barrier()

                if rollout_step >= max_rollouts:
                    stop_training = True
                    break

            start_batch = 0
            checkpoint_epoch = epoch if stop_training else epoch + 1
            checkpoint_batch = batch_index + 1 if stop_training else 0
            if context.is_main:
                save_ppo_checkpoint(
                    args.save_dir / "latest.pt",
                    actor_model,
                    critic_model,
                    actor_optimizer,
                    critic_optimizer,
                    actor_scheduler,
                    critic_scheduler,
                    scaler,
                    epoch=checkpoint_epoch,
                    batch_in_epoch=checkpoint_batch,
                    rollout_step=rollout_step,
                    global_step=global_step,
                    args=args,
                    tracker_state=tracker.state_dict(),
                )
            if context.distributed:
                dist.barrier()

        if context.distributed:
            dist.barrier()
        if context.is_main:
            export_pretrained(
                args.output_dir, actor_model, tokenizer, args.tokenizer_path
            )
            print(f"PPO complete    : rollout={rollout_step}, optimizer_step={global_step}")
            print(f"Checkpoint      : {args.save_dir / 'latest.pt'}")
            print(f"Exported Actor  : {args.output_dir}")
        tracker_state = "success"
        tracker_error = None
    except KeyboardInterrupt:
        tracker_state = "aborted"
        tracker_error = "PPO interrupted by user."
        raise
    except BaseException:
        tracker_state = "crashed"
        tracker_error = traceback.format_exc()
        raise
    finally:
        try:
            tracker.finish(state=tracker_state, error=tracker_error)
        finally:
            cleanup_distributed(context)


if __name__ == "__main__":
    main()

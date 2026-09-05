#!/usr/bin/env python3
"""Native PyTorch rollout engine used by miniLLM PPO and Agentic RL."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from trainer.trainer_utils import unwrap_model


@dataclass
class RolloutBatch:
    """Generated trajectories and the behavior-policy statistics they need."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_log_probs: torch.Tensor
    action_positions: torch.Tensor
    responses: list[str]
    raw_responses: list[str]
    has_eos: torch.Tensor

    @property
    def response_lengths(self) -> torch.Tensor:
        return self.completion_mask.sum(dim=1)


def completion_mask_from_ids(
    completion_ids: torch.Tensor,
    *,
    eos_token_id: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Include the first EOS and exclude every later generation-fill token.

    A policy can sample ``pad_token_id`` before EOS because it remains part of
    the vocabulary. Such an action is valid. Pads inserted by ``generate`` are
    distinguishable because they only appear after that row has emitted EOS.
    """

    if completion_ids.ndim != 2:
        raise ValueError("completion_ids must have shape [batch, response]")
    batch, response = completion_ids.shape
    mask = torch.zeros_like(completion_ids, dtype=torch.bool)
    alive = torch.ones(batch, dtype=torch.bool, device=completion_ids.device)
    has_eos = torch.zeros_like(alive)
    for position in range(response):
        tokens = completion_ids[:, position]
        valid = alive
        mask[:, position] = valid
        is_eos = tokens.eq(eos_token_id) & valid
        has_eos |= is_eos
        alive &= ~is_eos
    del pad_token_id  # Kept in the public signature to document generation IDs.
    return mask, has_eos


def gather_completion_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    action_positions: torch.Tensor,
) -> torch.Tensor:
    """Gather log P(completion token) at its causal prediction position."""

    if logits.ndim != 3 or input_ids.ndim != 2 or action_positions.ndim != 2:
        raise ValueError("invalid PPO log-probability tensor rank")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must share batch/sequence dimensions")
    batch = input_ids.shape[0]
    if action_positions.shape[0] != batch:
        raise ValueError("action_positions batch dimension does not match input_ids")
    if action_positions.numel() == 0:
        raise ValueError("a rollout must contain at least one completion position")
    if int(action_positions.min().item()) < 0:
        raise ValueError("action positions cannot be negative")
    if int(action_positions.max().item()) >= input_ids.shape[1] - 1:
        raise ValueError("action position does not have a following target token")

    next_token_ids = input_ids.gather(1, action_positions + 1)
    minimum_token_id = int(next_token_ids.min().item())
    maximum_token_id = int(next_token_ids.max().item())
    if minimum_token_id < 0 or maximum_token_id >= logits.shape[-1]:
        raise ValueError(
            "completion target token is outside the policy vocabulary: "
            f"token_id_range=[{minimum_token_id}, {maximum_token_id}], "
            f"vocab_size={logits.shape[-1]}"
        )
    selected_logits = logits.gather(
        1,
        action_positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1]),
    )
    return F.log_softmax(selected_logits.float(), dim=-1).gather(
        2, next_token_ids.unsqueeze(-1)
    ).squeeze(-1)


class TorchRolloutEngine:
    """Generate on-policy responses and recompute exact raw-model log-probs."""

    def __init__(
        self,
        policy_model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        *,
        max_prompt_len: int,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        autocast_factory: Callable[[], AbstractContextManager[Any]] | None = None,
    ) -> None:
        if max_prompt_len <= 0 or max_new_tokens <= 0:
            raise ValueError("prompt and response lengths must be positive")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0.0 < top_p <= 1.0 or top_k < 0:
            raise ValueError("top_p must be in (0, 1] and top_k cannot be negative")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.max_prompt_len = max_prompt_len
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.autocast_factory = autocast_factory or nullcontext

    @torch.no_grad()
    def rollout(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> RolloutBatch:
        if not prompts:
            raise ValueError("Cannot rollout an empty prompt batch")
        previous_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(
                list(prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_prompt_len,
                add_special_tokens=False,
            ).to(self.device)
            return self.rollout_tokenized(
                encoded.input_ids,
                encoded.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        finally:
            self.tokenizer.padding_side = previous_padding_side

    @torch.no_grad()
    def rollout_tokenized(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> RolloutBatch:
        """Generate from already-rendered token IDs without re-tokenization.

        Agentic RL uses this path between tool turns so generated actions and
        inserted observations remain one exact token trajectory.
        """

        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, seq]")
        if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("tokenized rollout input cannot be empty")
        if input_ids.shape[1] > self.max_prompt_len:
            raise ValueError(
                f"tokenized prompt length {input_ids.shape[1]} exceeds "
                f"max_prompt_len={self.max_prompt_len}"
            )
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        requested_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        requested_temperature = self.temperature if temperature is None else temperature
        requested_top_p = self.top_p if top_p is None else top_p
        requested_top_k = self.top_k if top_k is None else top_k
        if requested_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if requested_temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0.0 < requested_top_p <= 1.0 or requested_top_k < 0:
            raise ValueError("invalid top_p or top_k")

        raw_policy = unwrap_model(self.policy_model)
        was_training = raw_policy.training
        raw_policy.eval()
        try:
            prompt_width = input_ids.shape[1]
            generation_args: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": requested_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": self.repetition_penalty,
                "use_cache": False,
            }
            if requested_temperature > 0:
                generation_args.update(
                    do_sample=True,
                    temperature=requested_temperature,
                    top_p=requested_top_p,
                    top_k=requested_top_k,
                )
            else:
                generation_args["do_sample"] = False
            with self.autocast_factory():
                generated = raw_policy.generate(**generation_args)
            completion_ids = generated[:, prompt_width:]
            completion_mask, has_eos = completion_mask_from_ids(
                completion_ids,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if bool((completion_mask.sum(dim=1) == 0).any().item()):
                raise RuntimeError("policy generated an empty PPO response")
            full_attention_mask = torch.cat(
                [attention_mask.bool(), completion_mask], dim=1
            ).long()
            response_width = completion_ids.shape[1]
            action_positions = (
                torch.arange(response_width, device=self.device)
                .unsqueeze(0)
                .expand(generated.shape[0], -1)
                + prompt_width
                - 1
            )
            with self.autocast_factory():
                logits = raw_policy(
                    input_ids=generated,
                    attention_mask=full_attention_mask,
                ).logits
            old_log_probs = gather_completion_log_probs(
                logits, generated, action_positions
            )
            responses = [
                self.tokenizer.decode(
                    row[mask],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for row, mask in zip(completion_ids, completion_mask)
            ]
            raw_responses = [
                self.tokenizer.decode(
                    row[mask],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                for row, mask in zip(completion_ids, completion_mask)
            ]
            return RolloutBatch(
                input_ids=generated,
                attention_mask=full_attention_mask,
                completion_ids=completion_ids,
                completion_mask=completion_mask.float(),
                old_log_probs=old_log_probs * completion_mask,
                action_positions=action_positions,
                responses=responses,
                raw_responses=raw_responses,
                has_eos=has_eos,
            )
        finally:
            raw_policy.train(was_training)

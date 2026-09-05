#!/usr/bin/env python3
"""Multi-turn Tool-Use Agentic RL for miniLLM.

This intentionally follows MiniMind's compact layout: tool execution, reward,
multi-turn rollout, GRPO/CISPO objectives, evaluation and the training entry
point live in one file.  The policy starts from a completed PPO actor while a
frozen copy of that actor supplies the KL anchor.

Example from the project root::

    python trainer/train_agent.py \
        --model-path out/ppo \
        --data-path dataset/rl/agent_rl.jsonl

Or load the actor directly from a native PPO checkpoint::

    python trainer/train_agent.py --ppo-checkpoint checkpoints/ppo/latest.pt
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import operator
import os
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lm_dataset import validate_tokenizer  # noqa: E402
from dataset.ppo_dataset import AgentRLDataCollator, AgentRLDataset  # noqa: E402
from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM  # noqa: E402
from trainer.rollout_engine import TorchRolloutEngine  # noqa: E402
from trainer.train_sft import (  # noqa: E402
    autocast_context,
    build_optimizer,
    configure_attention_backend,
    resolve_amp,
)
from trainer.trainer_utils import (  # noqa: E402
    DistributedContext,
    ExperimentTracker,
    build_cosine_scheduler,
    cleanup_distributed,
    distributed_sum,
    ensure_checkpoint_tokenizer_fingerprint,
    ensure_config_tokenizer_fingerprint,
    ensure_tokenizer_matches_models,
    export_pretrained,
    init_experiment_tracker,
    load_checkpoint,
    resolve_resume_path,
    save_checkpoint,
    seed_everything,
    setup_distributed,
    tokenizer_fingerprint,
    unwrap_model,
)


WEATHER_DATA = {
    "北京": ("28°C", "晴"),
    "上海": ("15°C", "多云"),
    "广州": ("32°C", "闷热"),
    "深圳": ("30°C", "晴"),
    "杭州": ("22°C", "阴"),
    "成都": ("18°C", "小雨"),
    "武汉": ("25°C", "多云"),
    "南京": ("20°C", "晴"),
    "西安": ("16°C", "大风"),
    "重庆": ("26°C", "阴"),
    "Tokyo": ("12°C", "晴"),
    "New York": ("8°C", "多云"),
    "London": ("5°C", "小雨"),
    "Paris": ("10°C", "阴"),
    "Sydney": ("25°C", "晴朗"),
}
TIME_DATA = {
    "Asia/Shanghai": "2025-03-07 14:30:00",
    "America/New_York": "2025-03-07 01:30:00",
    "Europe/London": "2025-03-07 06:30:00",
    "Asia/Tokyo": "2025-03-07 15:30:00",
    "Europe/Paris": "2025-03-07 07:30:00",
    "Australia/Sydney": "2025-03-07 17:30:00",
}
EXCHANGE_DATA = {
    ("USD", "CNY"): 7.21,
    ("EUR", "CNY"): 7.85,
    ("GBP", "CNY"): 9.12,
    ("JPY", "CNY"): 0.048,
    ("USD", "EUR"): 0.92,
    ("USD", "GBP"): 0.79,
    ("CNY", "JPY"): 20.83,
    ("AUD", "CNY"): 4.72,
}
TRANSLATE_DATA = {
    ("你好世界", "english"): "Hello World",
    ("Good morning", "chinese"): "早上好",
    ("今天天气真好", "english"): "The weather is nice today",
    ("I love programming", "chinese"): "我喜欢编程",
    ("机器学习很有趣", "english"): "Machine learning is interesting",
    ("Happy birthday", "chinese"): "生日快乐",
}
UNIT_FACTORS = {
    ("km", "miles"): 0.621371,
    ("miles", "km"): 1.60934,
    ("kg", "pounds"): 2.20462,
    ("pounds", "kg"): 0.453592,
    ("meters", "feet"): 3.28084,
    ("feet", "meters"): 0.3048,
}
UNIT_ALIASES = {
    "kilometer": "km",
    "kilometers": "km",
    "mile": "miles",
    "kilogram": "kg",
    "kilograms": "kg",
    "pound": "pounds",
    "meter": "meters",
    "metre": "meters",
    "metres": "meters",
    "foot": "feet",
}
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])")


@dataclass(frozen=True)
class ToolExecution:
    name: str
    arguments: dict[str, Any]
    success: bool
    output: str
    signature: str


@dataclass
class AgentTrajectory:
    input_ids: list[int]
    action_mask: list[int]
    old_log_probs: list[float]
    turns: list[str]
    tool_outputs: list[str]
    valid_calls: int = 0
    invalid_calls: int = 0
    successful_calls: int = 0
    repeated_calls: int = 0
    parse_errors: int = 0
    unfinished: bool = False
    final_answer: str = ""
    prompt_truncated: bool = False


@dataclass(frozen=True)
class AgentReward:
    total: float
    outcome: float
    tool: float
    format: float
    efficiency: float
    matched: int
    observation_matched: int


@dataclass(frozen=True)
class AgentObjective:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl: torch.Tensor
    clip_fraction: torch.Tensor
    mean_ratio: torch.Tensor
    active_trajectories: torch.Tensor


def _tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function", tool)
    return str(function.get("name", "")) if isinstance(function, Mapping) else ""


def parse_tool_calls(text: str) -> tuple[list[dict[str, Any]], int]:
    """Parse MiniMind-style tool tags and count malformed calls/tags."""

    calls: list[dict[str, Any]] = []
    errors = abs(text.count("<tool_call>") - text.count("</tool_call>"))
    for payload in TOOL_CALL_PATTERN.findall(text):
        try:
            raw = json.loads(payload.strip())
            if not isinstance(raw, Mapping):
                raise TypeError("tool call is not an object")
            function = raw.get("function", raw)
            if not isinstance(function, Mapping):
                raise TypeError("tool call function is not an object")
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, Mapping):
                raise TypeError("tool arguments are not an object")
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("tool name is empty")
            calls.append({"name": name.strip(), "arguments": dict(arguments)})
        except (json.JSONDecodeError, TypeError, ValueError):
            errors += 1
    return calls, errors


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calculate(expression: str) -> int | float:
    """Evaluate basic arithmetic without names, calls, attributes or indexing."""

    normalized = (
        expression.replace("^", "**")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("（", "(")
        .replace("）", ")")
    )
    if len(normalized) > 256:
        raise ValueError("expression is too long")
    tree = ast.parse(normalized, mode="eval")

    def evaluate(node: ast.AST, depth: int = 0) -> int | float:
        if depth > 32:
            raise ValueError("expression is too deeply nested")
        if isinstance(node, ast.Expression):
            return evaluate(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = node.value
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            value = _UNARY_OPERATORS[type(node.op)](evaluate(node.operand, depth + 1))
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left, depth + 1)
            right = evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("exponent is too large")
            value = _BINARY_OPERATORS[type(node.op)](left, right)
        else:
            raise ValueError("unsupported arithmetic syntax")
        if not math.isfinite(float(value)) or abs(float(value)) > 1e100:
            raise ValueError("numeric result is out of range")
        return value

    result = evaluate(tree)
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


def _require_string(arguments: Mapping[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def execute_tool_call(
    call: Mapping[str, Any], allowed_names: set[str]
) -> ToolExecution:
    """Validate and execute one deterministic training-environment tool call."""

    name = str(call.get("name", ""))
    raw_arguments = call.get("arguments", {})
    arguments = dict(raw_arguments) if isinstance(raw_arguments, Mapping) else {}
    signature = json.dumps(
        {"name": name, "arguments": arguments}, ensure_ascii=False, sort_keys=True
    )
    try:
        if name not in allowed_names:
            raise ValueError("tool is not available for this task")
        if name == "calculate_math":
            result: Any = {"result": str(safe_calculate(_require_string(arguments, "expression")))}
        elif name == "unit_converter":
            value = arguments.get("value")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("value must be numeric")
            source = UNIT_ALIASES.get(
                _require_string(arguments, "from_unit").lower(),
                _require_string(arguments, "from_unit").lower(),
            )
            target = UNIT_ALIASES.get(
                _require_string(arguments, "to_unit").lower(),
                _require_string(arguments, "to_unit").lower(),
            )
            if source == "celsius" and target == "fahrenheit":
                converted = float(value) * 1.8 + 32
            elif source == "fahrenheit" and target == "celsius":
                converted = (float(value) - 32) / 1.8
            elif (source, target) in UNIT_FACTORS:
                converted = float(value) * UNIT_FACTORS[(source, target)]
            else:
                raise ValueError("unsupported unit conversion")
            result = {"result": round(converted, 4)}
        elif name == "get_current_weather":
            location = _require_string(arguments, "location")
            if location not in WEATHER_DATA:
                raise ValueError("unknown weather location")
            temperature, condition = WEATHER_DATA[location]
            result = {
                "city": location,
                "temperature": temperature,
                "humidity": "65%",
                "condition": condition,
            }
        elif name == "get_current_time":
            timezone = arguments.get("timezone", "Asia/Shanghai")
            if not isinstance(timezone, str) or timezone not in TIME_DATA:
                raise ValueError("unknown timezone")
            result = {"datetime": TIME_DATA[timezone], "timezone": timezone}
        elif name == "get_exchange_rate":
            source_currency = _require_string(arguments, "from_currency").upper()
            target_currency = _require_string(arguments, "to_currency").upper()
            pair = (source_currency, target_currency)
            if pair not in EXCHANGE_DATA:
                raise ValueError("unsupported currency pair")
            result = {
                "from": source_currency,
                "to": target_currency,
                "rate": EXCHANGE_DATA[pair],
            }
        elif name == "translate_text":
            text = _require_string(arguments, "text")
            language = _require_string(arguments, "target_language").lower()
            key = (text, language)
            if key not in TRANSLATE_DATA:
                raise ValueError("translation is not available")
            result = {"translated_text": TRANSLATE_DATA[key]}
        else:
            raise ValueError("tool implementation is unavailable")
        output = json.dumps(result, ensure_ascii=False)
        return ToolExecution(name, arguments, True, output[:2048], signature)
    except (ArithmeticError, SyntaxError, ValueError) as exc:
        output = json.dumps({"error": str(exc)}, ensure_ascii=False)
        return ToolExecution(name, arguments, False, output[:2048], signature)


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
    )


def _render_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    open_thinking: bool,
) -> str:
    return tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=open_thinking,
        tools=list(tools),
    )


def fit_initial_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    open_thinking: bool,
    max_prompt_len: int,
) -> tuple[list[int], bool]:
    """Drop old complete turns, then shorten the newest message if necessary."""

    current = [dict(message) for message in messages]
    prompt_ids = _encode(tokenizer, _render_prompt(tokenizer, current, tools, open_thinking))
    if len(prompt_ids) <= max_prompt_len:
        return prompt_ids, False

    leading_system: list[dict[str, Any]] = []
    body_start = 0
    while body_start < len(current) and current[body_start]["role"] == "system":
        leading_system.append(current[body_start])
        body_start += 1
    user_starts = [
        index for index in range(body_start, len(current)) if current[index]["role"] == "user"
    ]
    for start in reversed(user_starts):
        candidate = [*leading_system, *current[start:]]
        candidate_ids = _encode(
            tokenizer, _render_prompt(tokenizer, candidate, tools, open_thinking)
        )
        if len(candidate_ids) <= max_prompt_len:
            return candidate_ids, True

    # The newest turn itself is too large. Keep the tool-bearing system prompt
    # and as much of the newest message suffix as fits.
    target_index = user_starts[-1] if user_starts else len(current) - 1
    target = dict(current[target_index])
    content_ids = _encode(tokenizer, str(target.get("content", "")))
    low, high = 0, len(content_ids)
    best: list[int] | None = None
    while low <= high:
        keep = (low + high) // 2
        shortened = dict(target)
        shortened["content"] = tokenizer.decode(
            content_ids[-keep:] if keep else [],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        candidate = [*leading_system, shortened, *current[target_index + 1 :]]
        candidate_ids = _encode(
            tokenizer, _render_prompt(tokenizer, candidate, tools, open_thinking)
        )
        if len(candidate_ids) <= max_prompt_len:
            best = candidate_ids
            low = keep + 1
        else:
            high = keep - 1
    if best is None:
        raise ValueError(
            "tool definitions and generation prompt exceed --max-prompt-len"
        )
    return best, True


def _assistant_suffix(has_eos: bool) -> str:
    return "\n" if has_eos else "<|im_end|>\n"


def _next_turn_context(
    executions: Sequence[ToolExecution], open_thinking: bool
) -> str:
    chunks = []
    for execution in executions:
        chunks.append(
            "<|im_start|>tool\n<tool_response>\n"
            + execution.output
            + "\n</tool_response><|im_end|>\n"
        )
    chunks.append("<|im_start|>assistant\n")
    chunks.append("<think>\n" if open_thinking else "<think>\n\n</think>\n\n")
    return "".join(chunks)


def clean_final_answer(text: str) -> str:
    text = text.replace("<|im_end|>", "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    return TOOL_CALL_PATTERN.sub("", text).strip()


@torch.no_grad()
def rollout_agent(
    engine: TorchRolloutEngine,
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    max_prompt_len: int,
    max_total_len: int,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    open_thinking: bool,
) -> AgentTrajectory:
    """Run one exact multi-turn trajectory and mark policy/environment tokens."""

    prompt_ids, prompt_truncated = fit_initial_prompt(
        tokenizer, messages, tools, open_thinking, max_prompt_len
    )
    input_ids = list(prompt_ids)
    action_mask = [0] * len(input_ids)
    old_log_probs = [0.0] * len(input_ids)
    trajectory = AgentTrajectory(
        input_ids=input_ids,
        action_mask=action_mask,
        old_log_probs=old_log_probs,
        turns=[],
        tool_outputs=[],
        prompt_truncated=prompt_truncated,
    )
    allowed_names = {_tool_name(tool) for tool in tools}
    allowed_names.discard("")
    seen_calls: set[str] = set()

    for turn_index in range(max_turns):
        remaining = max_total_len - len(trajectory.input_ids)
        if remaining <= 0:
            trajectory.unfinished = True
            break
        turn_budget = min(max_new_tokens, remaining)
        device = engine.device
        ids = torch.tensor([trajectory.input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(ids)
        result = engine.rollout_tokenized(
            ids,
            attention_mask,
            max_new_tokens=turn_budget,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        completion_mask = result.completion_mask[0].bool()
        completion_ids = result.completion_ids[0][completion_mask].tolist()
        completion_logps = result.old_log_probs[0][completion_mask].float().tolist()
        raw_text = result.raw_responses[0]
        has_eos = bool(result.has_eos[0].item())
        trajectory.input_ids.extend(completion_ids)
        trajectory.action_mask.extend([1] * len(completion_ids))
        trajectory.old_log_probs.extend(completion_logps)
        trajectory.turns.append(raw_text)

        calls, parse_errors = parse_tool_calls(raw_text)
        trajectory.parse_errors += parse_errors
        if not calls:
            trajectory.final_answer = clean_final_answer(raw_text)
            trajectory.unfinished = not has_eos
            break

        executions: list[ToolExecution] = []
        for call in calls:
            execution = execute_tool_call(call, allowed_names)
            executions.append(execution)
            trajectory.tool_outputs.append(execution.output)
            if execution.signature in seen_calls:
                trajectory.repeated_calls += 1
            seen_calls.add(execution.signature)
            if execution.success:
                trajectory.valid_calls += 1
                trajectory.successful_calls += 1
            else:
                trajectory.invalid_calls += 1

        if turn_index == max_turns - 1:
            trajectory.unfinished = True
            break
        environment_text = _assistant_suffix(has_eos) + _next_turn_context(
            executions, open_thinking
        )
        environment_ids = _encode(tokenizer, environment_text)
        if len(trajectory.input_ids) + len(environment_ids) >= max_total_len:
            trajectory.unfinished = True
            break
        trajectory.input_ids.extend(environment_ids)
        trajectory.action_mask.extend([0] * len(environment_ids))
        trajectory.old_log_probs.extend([0.0] * len(environment_ids))
    else:
        trajectory.unfinished = True

    if not any(trajectory.action_mask):
        raise RuntimeError("Agent rollout produced no policy action tokens")
    return trajectory


def validate_gt_in_text(text: str, gt: Sequence[str]) -> set[str]:
    """Return GT values found by normalized string or tolerant numeric match."""

    lowered = text.casefold()
    numeric_text = text.replace(",", "")
    numbers = [float(value) for value in NUMBER_PATTERN.findall(numeric_text)]
    matched: set[str] = set()
    for raw in gt:
        target = str(raw).strip()
        if not target:
            continue
        if target.casefold() in lowered:
            matched.add(target)
            continue
        normalized = target.replace(",", "")
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", normalized):
            expected = float(normalized)
            tolerance = max(1e-6, abs(expected) * 1e-6)
            if any(abs(expected - actual) <= tolerance for actual in numbers):
                matched.add(target)
    return matched


def _repetition_penalty(text: str) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text.casefold())
    grams = [tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    if not grams:
        return 0.0
    duplicate_ratio = (len(grams) - len(set(grams))) / len(grams)
    return min(0.5, duplicate_ratio)


def calculate_agent_reward(
    trajectory: AgentTrajectory, gt: Sequence[str]
) -> AgentReward:
    """Score verified outcome first, then protocol quality and efficiency."""

    final_matches = validate_gt_in_text(trajectory.final_answer, gt)
    observation_matches = validate_gt_in_text("\n".join(trajectory.tool_outputs), gt)
    denominator = max(len(gt), 1)
    # A correct guess without executing a tool is not an Agent success.
    outcome = (
        2.5 * len(final_matches) / denominator
        if trajectory.successful_calls > 0
        else 0.0
    )
    if trajectory.successful_calls > 0:
        tool_score = 0.25 + 0.25 * min(
            1.0,
            trajectory.successful_calls
            / max(trajectory.valid_calls + trajectory.invalid_calls, 1),
        )
    else:
        tool_score = -1.0
    tool_score -= min(1.0, 0.35 * trajectory.invalid_calls)

    format_score = 0.2 if trajectory.valid_calls and not trajectory.parse_errors else 0.0
    format_score -= min(0.8, 0.25 * trajectory.parse_errors)
    if trajectory.unfinished:
        format_score -= 0.5

    efficiency = -min(
        0.75,
        0.05 * trajectory.invalid_calls + 0.2 * trajectory.repeated_calls,
    )
    efficiency -= _repetition_penalty(trajectory.final_answer)
    total = max(-3.0, min(3.0, outcome + tool_score + format_score + efficiency))
    return AgentReward(
        total=total,
        outcome=outcome,
        tool=tool_score,
        format=format_score,
        efficiency=efficiency,
        matched=len(final_matches),
        observation_matched=len(observation_matches),
    )


def compute_group_advantages(
    rewards: torch.Tensor, group_size: int, eps: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("rewards must be a non-empty vector")
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("group_size must divide rewards and be at least 2")
    grouped = rewards.view(-1, group_size)
    means = grouped.mean(dim=1, keepdim=True)
    stds = grouped.std(dim=1, unbiased=False, keepdim=True)
    valid_groups = stds.squeeze(1).gt(1e-6)
    advantages = (grouped - means) / (stds + eps)
    advantages = advantages * valid_groups.unsqueeze(1)
    return advantages.reshape(-1), valid_groups.repeat_interleave(group_size)


def compute_per_token_logps(
    logits: torch.Tensor, input_ids: torch.Tensor
) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must share [batch, sequence]")
    return F.log_softmax(logits[:, :-1, :].float(), dim=-1).gather(
        2, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)


def snapshot_on_policy_logps(policy_logps: torch.Tensor) -> torch.Tensor:
    """Freeze a same-forward denominator for one-pass on-policy updates."""

    if policy_logps.ndim != 2:
        raise ValueError("policy_logps must have shape [batch, sequence]")
    return policy_logps.detach()


def compute_agent_objective(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    action_mask: torch.Tensor,
    advantages: torch.Tensor,
    active_samples: torch.Tensor,
    *,
    loss_type: str,
    beta: float,
    epsilon: float,
    epsilon_high: float,
) -> AgentObjective:
    """Compute token-masked GRPO or MiniMind-style CISPO."""

    if policy_logps.shape != old_logps.shape or policy_logps.shape != reference_logps.shape:
        raise ValueError("policy, old and reference log-probabilities must align")
    if action_mask.shape != policy_logps.shape:
        raise ValueError("action_mask must align with token log-probabilities")
    if advantages.shape != active_samples.shape or advantages.ndim != 1:
        raise ValueError("advantages and active_samples must be aligned vectors")
    if policy_logps.shape[0] != advantages.numel():
        raise ValueError("trajectory and advantage batch dimensions must match")
    if loss_type not in {"grpo", "cispo"}:
        raise ValueError("loss_type must be grpo or cispo")
    if beta < 0 or not 0 < epsilon < 1 or epsilon_high <= 0:
        raise ValueError("invalid Agent objective hyperparameters")

    log_ratio = (policy_logps - old_logps).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    kl_delta = (reference_logps - policy_logps).clamp(-20.0, 20.0)
    per_token_kl = kl_delta.exp() - kl_delta - 1.0
    advantage = advantages.unsqueeze(1)
    if loss_type == "cispo":
        importance = ratio.clamp(max=epsilon_high).detach()
        policy_term = importance * advantage * policy_logps
        clipped = ratio.gt(epsilon_high)
    else:
        clipped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
        policy_term = torch.minimum(ratio * advantage, clipped_ratio * advantage)
        clipped = ratio.ne(clipped_ratio)

    mask = action_mask.float() * active_samples.float().unsqueeze(1)
    token_counts = mask.sum(dim=1)
    valid = token_counts.gt(0)
    per_sequence_policy = (-(policy_term) * mask).sum(dim=1) / token_counts.clamp_min(1)
    per_sequence_kl = (per_token_kl * mask).sum(dim=1) / token_counts.clamp_min(1)
    if bool(valid.any().item()):
        policy_loss = per_sequence_policy[valid].mean()
        kl = per_sequence_kl[valid].mean()
        loss = policy_loss + beta * kl
        clip_fraction = (clipped.float() * mask).sum() / mask.sum().clamp_min(1)
        mean_ratio = (ratio * mask).sum() / mask.sum().clamp_min(1)
    else:
        zero = policy_logps.sum() * 0.0
        loss = policy_loss = kl = clip_fraction = zero
        mean_ratio = zero + 1.0
    return AgentObjective(
        loss=loss,
        policy_loss=policy_loss,
        kl=kl,
        clip_fraction=clip_fraction,
        mean_ratio=mean_ratio,
        active_trajectories=valid.sum(),
    )


def pack_trajectories(
    trajectories: Sequence[AgentTrajectory], pad_token_id: int, device: torch.device
) -> dict[str, torch.Tensor]:
    if not trajectories:
        raise ValueError("cannot pack an empty trajectory list")
    max_length = max(len(trajectory.input_ids) for trajectory in trajectories)
    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    action_rows: list[list[int]] = []
    old_rows: list[list[float]] = []
    for trajectory in trajectories:
        length = len(trajectory.input_ids)
        if not (
            length == len(trajectory.action_mask) == len(trajectory.old_log_probs)
        ):
            raise ValueError("trajectory token, mask and log-prob lengths differ")
        padding = max_length - length
        input_rows.append(trajectory.input_ids + [pad_token_id] * padding)
        attention_rows.append([1] * length + [0] * padding)
        action_rows.append(trajectory.action_mask + [0] * padding)
        old_rows.append(trajectory.old_log_probs + [0.0] * padding)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(
            attention_rows, dtype=torch.long, device=device
        ),
        # Shift because logits[:, t] predicts token t+1.
        "action_mask": torch.tensor(
            action_rows, dtype=torch.float32, device=device
        )[:, 1:],
        "old_logps": torch.tensor(old_rows, dtype=torch.float32, device=device)[
            :, 1:
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="miniLLM multi-turn Agentic RL")
    parser.add_argument(
        "--data-path",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "dataset" / "rl" / "agent_rl.jsonl"],
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "out" / "ppo",
        help="Exported PPO actor directory.",
    )
    parser.add_argument(
        "--ppo-checkpoint",
        type=Path,
        default=None,
        help="Native PPO checkpoint containing actor_model and actor_config.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Tokenizer directory; defaults to --model-path.",
    )
    parser.add_argument(
        "--save-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "agent"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "out" / "agent"
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2, help="Prompts per GPU")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-7)
    parser.add_argument("--min-learning-rate", type=float, default=3e-8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--loss-type", choices=["grpo", "cispo"], default="grpo")
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-high", type=float, default=5.0)
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-total-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--thinking-ratio", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)

    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume checkpoints/agent/latest.pt or an explicit Agent checkpoint.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--attention-backend", choices=["eager", "auto", "math"], default="eager"
    )
    parser.add_argument(
        "--tracker", choices=["none", "swanlab", "wandb"], default="none"
    )
    parser.add_argument("--tracker-project", default="miniLLM-Agent-RL")
    parser.add_argument("--tracker-run-name", default=None)
    parser.add_argument("--tracker-entity", default=None)
    parser.add_argument("--tracker-group", default=None)
    parser.add_argument("--tracker-tags", nargs="*", default=[])
    parser.add_argument(
        "--tracker-mode", choices=["online", "offline"], default="online"
    )
    parser.add_argument(
        "--tracker-log-dir", type=Path, default=PROJECT_ROOT / "logs" / "agent"
    )
    return parser.parse_args()


def project_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "epochs",
        "batch_size",
        "accumulation_steps",
        "learning_rate",
        "grad_clip",
        "num_generations",
        "epsilon",
        "epsilon_high",
        "max_prompt_len",
        "max_total_len",
        "max_new_tokens",
        "max_turns",
        "eval_interval",
        "log_interval",
        "save_interval",
    )
    for field in positive:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_generations < 2:
        raise ValueError("--num-generations must be at least 2")
    if args.max_total_len <= args.max_prompt_len:
        raise ValueError("--max-total-len must exceed --max-prompt-len")
    if args.max_steps < 0 or args.max_train_samples < 0 or args.eval_samples < 0:
        raise ValueError("step and sample limits cannot be negative")
    if not 0 <= args.beta or not 0 <= args.weight_decay:
        raise ValueError("beta and weight decay cannot be negative")
    if not 0 <= args.thinking_ratio <= 1:
        raise ValueError("--thinking-ratio must be in [0, 1]")
    if not 0 < args.top_p <= 1 or args.top_k < 0 or args.temperature < 0:
        raise ValueError("invalid sampling parameters")
    if not 0 < args.val_ratio < 0.5 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("invalid validation or warmup ratio")
    if not 0 < args.min_learning_rate <= args.learning_rate:
        raise ValueError("min learning rate must be positive and <= learning rate")
    if args.num_workers < 0 or args.repetition_penalty <= 0:
        raise ValueError("invalid worker count or repetition penalty")


def make_dataloaders(
    args: argparse.Namespace, tokenizer: Any, context: DistributedContext
) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    common = {
        "data_paths": args.data_path,
        "tokenizer": tokenizer,
        "val_ratio": args.val_ratio,
        "expected_vocab_size": len(tokenizer),
        "seed": args.seed,
    }
    train_dataset: Any = AgentRLDataset(split="train", **common)
    validation_dataset: Any = AgentRLDataset(split="validation", **common)
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
        "collate_fn": AgentRLDataCollator(),
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
    if not len(train_loader) or not len(validation_loader):
        raise ValueError("Agent train/validation loader is empty")
    return train_loader, validation_loader, train_sampler


def load_actor(args: argparse.Namespace) -> MiniLLMForCausalLM:
    if args.ppo_checkpoint is None:
        config_path = args.model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"PPO actor config not found: {config_path}. Export PPO actor first "
                "or pass --ppo-checkpoint."
            )
        config = MiniLLMConfig.from_pretrained(args.model_path, local_files_only=True)
        config.attention_backend = args.attention_backend
        return MiniLLMForCausalLM.from_pretrained(
            args.model_path, config=config, local_files_only=True
        )

    if not args.ppo_checkpoint.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {args.ppo_checkpoint}")
    try:
        checkpoint = torch.load(
            args.ppo_checkpoint, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(args.ppo_checkpoint, map_location="cpu")
    if "actor_model" not in checkpoint or "actor_config" not in checkpoint:
        raise ValueError(
            "PPO checkpoint must contain actor_model and actor_config fields"
        )
    config = MiniLLMConfig(**checkpoint["actor_config"])
    config.attention_backend = args.attention_backend
    model = MiniLLMForCausalLM(config)
    model.load_state_dict(checkpoint["actor_model"], strict=True)
    return model


def _rollout_kwargs(args: argparse.Namespace, *, evaluation: bool) -> dict[str, Any]:
    return {
        "max_prompt_len": args.max_prompt_len,
        "max_total_len": args.max_total_len,
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0 if evaluation else args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }


def collect_grouped_rollouts(
    engine: TorchRolloutEngine,
    tokenizer: Any,
    batch: dict[str, Any],
    args: argparse.Namespace,
    *,
    evaluation: bool,
) -> tuple[list[AgentTrajectory], list[AgentReward]]:
    trajectories: list[AgentTrajectory] = []
    rewards: list[AgentReward] = []
    group_size = 1 if evaluation else args.num_generations
    for messages, tools, gt in zip(batch["messages"], batch["tools"], batch["gt"]):
        for _ in range(group_size):
            trajectory = rollout_agent(
                engine,
                tokenizer,
                messages,
                tools,
                open_thinking=(
                    False
                    if evaluation
                    else random.random() < args.thinking_ratio
                ),
                **_rollout_kwargs(args, evaluation=evaluation),
            )
            trajectories.append(trajectory)
            rewards.append(calculate_agent_reward(trajectory, gt))
    return trajectories, rewards


@torch.no_grad()
def evaluate_agent(
    model: torch.nn.Module,
    tokenizer: Any,
    loader: DataLoader,
    args: argparse.Namespace,
    context: DistributedContext,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    engine = TorchRolloutEngine(
        model,
        tokenizer,
        context.device,
        max_prompt_len=args.max_total_len,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        autocast_factory=lambda: autocast_context(context.device, amp_dtype),
    )
    totals = torch.zeros(15, dtype=torch.float64, device=context.device)
    for batch in loader:
        trajectories, rewards = collect_grouped_rollouts(
            engine, tokenizer, batch, args, evaluation=True
        )
        for index, (trajectory, reward) in enumerate(zip(trajectories, rewards)):
            gt_count = len(batch["gt"][index])
            calls = trajectory.valid_calls + trajectory.invalid_calls
            totals[0] += reward.total
            totals[1] += reward.outcome
            totals[2] += int(
                reward.matched == gt_count
                and trajectory.successful_calls > 0
                and not trajectory.unfinished
            )
            totals[3] += reward.matched
            totals[4] += gt_count
            totals[5] += trajectory.valid_calls
            totals[6] += calls
            totals[7] += trajectory.successful_calls
            totals[8] += trajectory.invalid_calls
            totals[9] += int(trajectory.unfinished)
            totals[10] += len(trajectory.turns)
            totals[11] += calls
            totals[12] += int(trajectory.prompt_truncated)
            totals[13] += trajectory.parse_errors
            totals[14] += 1
    distributed_sum(totals, context)
    model.train(was_training)
    samples = totals[14].clamp_min(1)
    return {
        "reward": (totals[0] / samples).item(),
        "outcome_reward": (totals[1] / samples).item(),
        "task_success_rate": (totals[2] / samples).item(),
        "gt_match_rate": (totals[3] / totals[4].clamp_min(1)).item(),
        "valid_tool_call_rate": (totals[5] / totals[6].clamp_min(1)).item(),
        "tool_execution_success_rate": (
            totals[7] / totals[6].clamp_min(1)
        ).item(),
        "invalid_calls_per_sample": (totals[8] / samples).item(),
        "unfinished_rate": (totals[9] / samples).item(),
        "mean_turns": (totals[10] / samples).item(),
        "mean_calls": (totals[11] / samples).item(),
        "prompt_truncated_rate": (totals[12] / samples).item(),
        "parse_errors_per_sample": (totals[13] / samples).item(),
        "samples": totals[14].item(),
    }


def _format_metrics(prefix: str, metrics: Mapping[str, float]) -> str:
    def render(value: float) -> str:
        return f"{value:.3e}" if value and abs(value) < 1e-3 else f"{value:.4f}"

    values = ", ".join(f"{key}={render(value)}" for key, value in metrics.items())
    return f"{prefix}: {values}"


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    validate_args(args)
    args.data_path = [project_path(path) for path in args.data_path]
    args.model_path = project_path(args.model_path)
    args.ppo_checkpoint = project_path(args.ppo_checkpoint)
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
    finish_error: str | None = "Agent training stopped before completion."
    try:
        seed_everything(args.seed, context.rank)
        configure_attention_backend(args.attention_backend, context)
        if args.ppo_checkpoint is None:
            args.tokenizer_fingerprint = ensure_tokenizer_matches_models(
                args.tokenizer_path,
                {"PPO actor export": args.model_path},
            )
        else:
            args.tokenizer_fingerprint = tokenizer_fingerprint(args.tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, local_files_only=True, use_fast=True
        )
        template_path = args.tokenizer_path / "chat_template.jinja"
        if template_path.is_file():
            tokenizer.chat_template = template_path.read_text(encoding="utf-8")
        validate_tokenizer(tokenizer, expected_vocab_size=len(tokenizer))
        train_loader, validation_loader, train_sampler = make_dataloaders(
            args, tokenizer, context
        )

        policy_model: torch.nn.Module = load_actor(args)
        ensure_config_tokenizer_fingerprint(
            policy_model.config,
            args.tokenizer_fingerprint,
            label="PPO actor",
        )
        if policy_model.config.vocab_size != len(tokenizer):
            raise ValueError("PPO actor and tokenizer vocabulary sizes differ")
        if args.max_total_len > policy_model.config.max_position_embeddings:
            raise ValueError("max total length exceeds model position capacity")
        policy_model.config.use_cache = False
        reference_model = copy.deepcopy(policy_model)
        policy_model = policy_model.to(context.device)
        reference_model = reference_model.to(context.device)
        reference_model.eval().requires_grad_(False)
        if args.gradient_checkpointing:
            policy_model.model.gradient_checkpointing = True
        if context.distributed:
            policy_model = DistributedDataParallel(
                policy_model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
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
        amp_dtype, needs_scaler = resolve_amp(args.dtype, context)
        scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)

        start_epoch = start_batch = global_step = 0
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
            ensure_checkpoint_tokenizer_fingerprint(
                checkpoint, args.tokenizer_fingerprint
            )
            start_epoch = int(checkpoint.get("epoch", 0))
            start_batch = int(checkpoint.get("batch_in_epoch", 0))
            global_step = int(checkpoint.get("global_step", 0))
            if context.is_main:
                print(f"Resumed: {resume_path} (step={global_step})")
        if not args.eval_only and global_step >= total_steps:
            raise ValueError(
                f"Checkpoint step {global_step} already reached total_steps={total_steps}"
            )

        tracker = init_experiment_tracker(
            backend=args.tracker,
            context=context,
            project=args.tracker_project,
            run_name=args.tracker_run_name,
            entity=args.tracker_entity,
            group=args.tracker_group,
            tags=args.tracker_tags,
            mode=args.tracker_mode,
            log_dir=args.tracker_log_dir,
            config=vars(args).copy(),
        )
        if context.is_main:
            source = args.ppo_checkpoint or args.model_path
            print(f"PPO actor     : {source}")
            print(f"Tokenizer     : {args.tokenizer_path}")
            print(f"Tokenizer SHA : {args.tokenizer_fingerprint}")
            print(f"Data          : {', '.join(map(str, args.data_path))}")
            print(f"Device        : {context.device} (world_size={context.world_size})")
            print(f"Algorithm     : {args.loss_type.upper()} (G={args.num_generations})")
            print(f"Trajectory    : {args.max_turns} turns, {args.max_total_len} tokens")
            print(f"Train batches : {len(train_loader):,} per rank")
            print(f"Update steps  : {total_steps:,}")

        if args.eval_only:
            metrics = evaluate_agent(
                policy_model, tokenizer, validation_loader, args, context, amp_dtype
            )
            if context.is_main:
                print(_format_metrics("Agent evaluation", metrics))
            finish_state, finish_error = "success", None
            return

        engine = TorchRolloutEngine(
            policy_model,
            tokenizer,
            context.device,
            max_prompt_len=args.max_total_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            autocast_factory=lambda: autocast_context(context.device, amp_dtype),
        )
        policy_model.train()
        optimizer.zero_grad(set_to_none=True)
        interval_start = time.perf_counter()
        last_eval_step = -1
        stop = False

        for epoch in range(start_epoch, training_epochs):
            train_sampler.set_epoch(epoch)
            for batch_index, batch in enumerate(train_loader):
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                trajectories, reward_parts = collect_grouped_rollouts(
                    engine, tokenizer, batch, args, evaluation=False
                )
                rewards = torch.tensor(
                    [reward.total for reward in reward_parts],
                    dtype=torch.float32,
                    device=context.device,
                )
                advantages, active_samples = compute_group_advantages(
                    rewards, args.num_generations
                )
                packed = pack_trajectories(
                    trajectories, tokenizer.pad_token_id, context.device
                )
                with torch.no_grad(), autocast_context(context.device, amp_dtype):
                    reference_logits = reference_model(
                        input_ids=packed["input_ids"],
                        attention_mask=packed["attention_mask"],
                    ).logits
                    reference_logps = compute_per_token_logps(
                        reference_logits, packed["input_ids"]
                    )
                    del reference_logits
                with autocast_context(context.device, amp_dtype):
                    policy_output = policy_model(
                        input_ids=packed["input_ids"],
                        attention_mask=packed["attention_mask"],
                    )
                    policy_logps = compute_per_token_logps(
                        policy_output.logits, packed["input_ids"]
                    )
                    # Each trajectory is sampled and optimized exactly once
                    # before the next parameter update, so this denominator is
                    # an on-policy snapshot. ``rollout_agent`` records log-probs
                    # one turn at a time, while this forward uses a right-padded
                    # packed batch. With BF16 Top-1 MoE, those different kernel
                    # shapes can route borderline tokens to different experts
                    # and produce a spurious importance ratio even though the
                    # weights are unchanged. Taking the detached snapshot from
                    # this same packed forward keeps the correct policy gradient
                    # and makes a fresh on-policy ratio exactly one.
                    optimization_old_logps = snapshot_on_policy_logps(policy_logps)
                    objective = compute_agent_objective(
                        policy_logps,
                        optimization_old_logps,
                        reference_logps,
                        packed["action_mask"],
                        advantages,
                        active_samples,
                        loss_type=args.loss_type,
                        beta=args.beta,
                        epsilon=args.epsilon,
                        epsilon_high=args.epsilon_high,
                    )
                    router_aux = policy_output.router_aux_loss
                    if router_aux is None:
                        router_aux = objective.loss.new_zeros(())
                    # A constant-reward group has no RL signal. Do not let the
                    # MoE auxiliary term update the actor from such groups on
                    # its own; scale it by the effective trajectory fraction.
                    router_aux = router_aux * active_samples.float().mean()
                    total_loss = objective.loss + router_aux
                    scaled_loss = total_loss / args.accumulation_steps
                if not bool(torch.isfinite(total_loss.detach()).item()):
                    raise FloatingPointError("non-finite Agentic RL loss")
                if needs_scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                is_last_batch = batch_index + 1 == len(train_loader)
                should_update = (
                    (batch_index + 1) % args.accumulation_steps == 0 or is_last_batch
                )
                if not should_update:
                    continue
                if needs_scaler:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(), args.grad_clip, error_if_nonfinite=True
                )
                if needs_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                success = sum(
                    int(
                        reward.matched == len(batch["gt"][index // args.num_generations])
                        and trajectories[index].successful_calls > 0
                        and not trajectories[index].unfinished
                    )
                    for index, reward in enumerate(reward_parts)
                ) / len(reward_parts)
                metrics = {
                    "loss": float(total_loss.detach()),
                    "policy_loss": float(objective.policy_loss.detach()),
                    "kl": float(objective.kl.detach()),
                    "reward": float(rewards.mean()),
                    "task_success_rate": success,
                    "effective_group_rate": float(
                        active_samples.view(-1, args.num_generations)[:, 0]
                        .float()
                        .mean()
                    ),
                    "clip_fraction": float(objective.clip_fraction.detach()),
                    "mean_ratio": float(objective.mean_ratio.detach()),
                    "mean_action_tokens": sum(
                        sum(trajectory.action_mask) for trajectory in trajectories
                    )
                    / len(trajectories),
                    "grad_norm": float(grad_norm.detach()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                if global_step % args.log_interval == 0:
                    elapsed = max(time.perf_counter() - interval_start, 1e-6)
                    metrics["updates_per_second"] = args.log_interval / elapsed
                    if context.is_main:
                        print(_format_metrics(f"Step {global_step}", metrics))
                    tracker.log(metrics, step=global_step)
                    interval_start = time.perf_counter()

                if global_step % args.eval_interval == 0:
                    validation = evaluate_agent(
                        policy_model,
                        tokenizer,
                        validation_loader,
                        args,
                        context,
                        amp_dtype,
                    )
                    if context.is_main:
                        print(_format_metrics("Validation", validation))
                    tracker.log(
                        {f"validation/{key}": value for key, value in validation.items()},
                        step=global_step,
                    )
                    last_eval_step = global_step

                if global_step % args.save_interval == 0 and context.is_main:
                    save_checkpoint(
                        args.save_dir / "latest.pt",
                        policy_model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        batch_index + 1,
                        global_step,
                        args,
                        tracker.state_dict(),
                    )
                if global_step >= total_steps:
                    stop = True
                    break
            start_batch = 0
            if stop:
                break

        if last_eval_step != global_step:
            validation = evaluate_agent(
                policy_model,
                tokenizer,
                validation_loader,
                args,
                context,
                amp_dtype,
            )
            if context.is_main:
                print(_format_metrics("Final validation", validation))
            tracker.log(
                {f"validation/{key}": value for key, value in validation.items()},
                step=global_step,
            )
        if context.distributed:
            torch.distributed.barrier()
        if context.is_main:
            save_checkpoint(
                args.save_dir / "latest.pt",
                policy_model,
                optimizer,
                scheduler,
                scaler,
                training_epochs,
                0,
                global_step,
                args,
                tracker.state_dict(),
            )
            export_pretrained(
                args.output_dir, policy_model, tokenizer, args.tokenizer_path
            )
            print(f"Exported Agent actor: {args.output_dir}")
        finish_state, finish_error = "success", None
    except BaseException as exc:
        finish_error = f"{type(exc).__name__}: {exc}"
        if context.is_main:
            traceback.print_exc()
        raise
    finally:
        tracker.finish(finish_state, finish_error)
        cleanup_distributed(context)


if __name__ == "__main__":
    main()

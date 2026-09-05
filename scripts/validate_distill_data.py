#!/usr/bin/env python3
"""Validate black-box distillation JSONL before starting a long SFT run.

Structural validation scans every selected row.  Token-length statistics are
sampled because the full SFT corpus can contain millions of conversations.
The report is written next to the distilled file by default and the process
returns a non-zero exit code when invalid training rows are found.

Example:

    python scripts/validate_distill_data.py \
        --data-path dataset/distill/qwen3_1_7b_sft.jsonl \
        --model-path out/pretrain \
        --max-seq-len 1024
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "dataset" / "distill" / "qwen3_1_7b_sft.jsonl"
DEFAULT_MODEL = PROJECT_ROOT / "out" / "pretrain"
ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
LEAKED_SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def stable_sample(seed: int, row_index: int, every: int) -> bool:
    if every <= 1:
        return True
    payload = f"{seed}:{row_index}:token-stats".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value % every == 0


def parse_jsonish(value: Any, field_name: str) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {field_name}: {exc}") from exc


def normalize_for_validation(raw_messages: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("'conversations' must be a non-empty list")
    messages: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, Mapping):
            raise TypeError(f"message {index} must be an object")
        role = raw.get("role")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"message {index} has invalid role {role!r}")
        content = raw.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise TypeError(f"message {index}.content must be a string")
        reasoning = raw.get("reasoning_content", "") or ""
        if not isinstance(reasoning, str):
            raise TypeError(f"message {index}.reasoning_content must be a string")
        tools = parse_jsonish(raw.get("tools"), f"message[{index}].tools")
        tool_calls = parse_jsonish(
            raw.get("tool_calls"), f"message[{index}].tool_calls"
        )
        messages.append(
            {
                "role": role,
                "content": content,
                "reasoning_content": reasoning,
                "tools": tools,
                "tool_calls": tool_calls,
            }
        )
    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError("conversation contains no assistant turn")
    return messages


def validate_conversation(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str], Counter[str], int]:
    """Return errors, warnings, counters and total assistant characters."""

    errors: list[str] = []
    warnings: list[str] = []
    counters: Counter[str] = Counter()
    assistant_characters = 0

    for index, message in enumerate(messages):
        role = str(message["role"])
        counters[f"role_{role}"] += 1
        content = str(message.get("content", ""))
        reasoning = str(message.get("reasoning_content", ""))
        if any(token in content or token in reasoning for token in LEAKED_SPECIAL_TOKENS):
            errors.append(f"message {index} contains leaked Qwen chat special tokens")
        if role == "assistant":
            counters["assistant_turns"] += 1
            assistant_characters += len(content) + len(reasoning)
            if reasoning:
                counters["thinking_turns"] += 1
            if message.get("tool_calls"):
                counters["tool_call_turns"] += 1
            if not content.strip() and not message.get("tool_calls"):
                errors.append(f"assistant message {index} has no content or tool call")
            if "<think>" in content or "</think>" in content:
                warnings.append(
                    f"assistant message {index} keeps think tags inside content"
                )
            if len(content) > 20_000:
                warnings.append(f"assistant message {index} is unusually long")
        elif reasoning:
            warnings.append(f"non-assistant message {index} contains reasoning_content")

        if message.get("tools") or role == "tool":
            counters["tool_context_messages"] += 1

    if messages[-1]["role"] not in {"assistant", "tool"}:
        warnings.append("conversation does not end with assistant/tool")
    return errors, warnings, counters, assistant_characters


def load_token_tools(tokenizer_path: Path, model_path: Path):
    """Load tokenizer and the exact renderer used by SFT, only when requested."""

    try:
        from transformers import AutoTokenizer

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from dataset.sft_dataset import normalize_conversations, render_conversation
        from trainer.trainer_utils import ensure_tokenizer_matches_models
    except ImportError as exc:
        raise RuntimeError(
            "Token statistics require the project torch/transformers dependencies. "
            "Use --skip-token-stats for structural validation only."
        ) from exc

    ensure_tokenizer_matches_models(
        tokenizer_path,
        {"student base model": model_path},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("token statistics require the fast miniLLM tokenizer")
    return tokenizer, normalize_conversations, render_conversation


def count_rendered_tokens(
    raw_messages: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    normalize_conversations: Any,
    render_conversation: Any,
) -> tuple[int, int]:
    messages = normalize_conversations(raw_messages)
    rendered = render_conversation(messages)
    encoded = tokenizer(
        rendered.text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        verbose=False,
    )
    offsets = encoded["offset_mapping"]
    target_tokens = 0
    span_index = 0
    for start, end in offsets:
        while (
            span_index < len(rendered.assistant_spans)
            and start >= rendered.assistant_spans[span_index][1]
        ):
            span_index += 1
        if span_index >= len(rendered.assistant_spans):
            break
        span_start, span_end = rendered.assistant_spans[span_index]
        if end > span_start and start < span_end:
            target_tokens += 1
    return len(encoded["input_ids"]), target_tokens


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate miniLLM distillation JSONL.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Student tokenizer directory; defaults to --model-path.",
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--token-sample-every",
        type=int,
        default=100,
        help="Compute exact token stats for approximately one in N rows.",
    )
    parser.add_argument("--max-token-samples", type=int, default=10_000)
    parser.add_argument("--max-reported-issues", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-token-stats", action="store_true")
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.data_path = project_path(args.data_path)
    args.model_path = project_path(args.model_path)
    args.tokenizer_path = (
        args.model_path
        if args.tokenizer_path is None
        else project_path(args.tokenizer_path)
    )
    args.report_path = (
        project_path(args.report_path)
        if args.report_path is not None
        else args.data_path.with_suffix(args.data_path.suffix + ".validation.json")
    )
    if not args.data_path.is_file():
        raise FileNotFoundError(f"distillation data not found: {args.data_path}")
    if not args.skip_token_stats and not args.tokenizer_path.is_dir():
        raise FileNotFoundError(f"tokenizer directory not found: {args.tokenizer_path}")
    if not args.skip_token_stats and not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"student model config not found: {args.model_path / 'config.json'}"
        )
    for name in (
        "max_seq_len",
        "token_sample_every",
        "max_token_samples",
        "max_reported_issues",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_samples < 0:
        raise ValueError("--max-samples cannot be negative")


def main() -> None:
    args = parse_args()
    validate_args(args)
    token_tools = None
    if not args.skip_token_stats:
        token_tools = load_token_tools(args.tokenizer_path, args.model_path)

    counters: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    total_assistant_characters = 0
    token_count_sum = 0
    target_token_sum = 0
    min_tokens: int | None = None
    max_tokens = 0
    started = time.perf_counter()

    with args.data_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if args.max_samples and row_index >= args.max_samples:
                break
            counters["rows"] += 1
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("row must be an object")
                messages = normalize_for_validation(row.get("conversations"))
                row_errors, row_warnings, row_counts, assistant_chars = (
                    validate_conversation(messages)
                )
                counters.update(row_counts)
                total_assistant_characters += assistant_chars
                if row_errors:
                    counters["invalid_rows"] += 1
                if row_warnings:
                    counters["warning_rows"] += 1
                for issue in row_errors:
                    if len(errors) < args.max_reported_issues:
                        errors.append({"line": row_index + 1, "message": issue})
                for issue in row_warnings:
                    if len(warnings) < args.max_reported_issues:
                        warnings.append({"line": row_index + 1, "message": issue})

                if (
                    token_tools is not None
                    and counters["token_sample_rows"] < args.max_token_samples
                    and stable_sample(args.seed, row_index, args.token_sample_every)
                ):
                    tokenizer, normalize_conversations, render_conversation = token_tools
                    total_tokens, target_tokens = count_rendered_tokens(
                        row["conversations"],
                        tokenizer,
                        normalize_conversations,
                        render_conversation,
                    )
                    counters["token_sample_rows"] += 1
                    counters["overlength_sample_rows"] += int(
                        total_tokens > args.max_seq_len
                    )
                    counters["zero_target_sample_rows"] += int(target_tokens == 0)
                    token_count_sum += total_tokens
                    target_token_sum += target_tokens
                    min_tokens = (
                        total_tokens if min_tokens is None else min(min_tokens, total_tokens)
                    )
                    max_tokens = max(max_tokens, total_tokens)
            except Exception as exc:
                counters["invalid_rows"] += 1
                if len(errors) < args.max_reported_issues:
                    errors.append({"line": row_index + 1, "message": str(exc)})

    elapsed = time.perf_counter() - started
    rows = counters["rows"]
    token_samples = counters["token_sample_rows"]
    report = {
        "status": "failed" if counters["invalid_rows"] else "passed",
        "data_path": str(args.data_path),
        "model_path": None if token_tools is None else str(args.model_path),
        "tokenizer_path": None if token_tools is None else str(args.tokenizer_path),
        "max_seq_len": args.max_seq_len,
        "elapsed_seconds": round(elapsed, 3),
        "counts": dict(counters),
        "ratios": {
            "invalid_rows": ratio(counters["invalid_rows"], rows),
            "warning_rows": ratio(counters["warning_rows"], rows),
            "thinking_turns": ratio(
                counters["thinking_turns"], counters["assistant_turns"]
            ),
            "overlength_token_sample": ratio(
                counters["overlength_sample_rows"], token_samples
            ),
        },
        "averages": {
            "assistant_characters_per_row": ratio(total_assistant_characters, rows),
            "tokens_per_sampled_row": ratio(token_count_sum, token_samples),
            "assistant_target_tokens_per_sampled_row": ratio(
                target_token_sum, token_samples
            ),
        },
        "token_sample_range": {
            "minimum": min_tokens,
            "maximum": max_tokens if token_samples else None,
        },
        "errors": errors,
        "warnings": warnings,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Rows             : {rows:,}")
    print(f"Invalid rows     : {counters['invalid_rows']:,}")
    print(f"Warning rows     : {counters['warning_rows']:,}")
    print(f"Assistant turns  : {counters['assistant_turns']:,}")
    print(
        f"Thinking ratio   : "
        f"{report['ratios']['thinking_turns']:.2%}"
    )
    if token_samples:
        print(f"Token samples    : {token_samples:,}")
        print(
            f"Overlength ratio : "
            f"{report['ratios']['overlength_token_sample']:.2%}"
        )
    print(f"Report           : {args.report_path}")
    if counters["invalid_rows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

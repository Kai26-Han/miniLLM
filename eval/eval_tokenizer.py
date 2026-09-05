#!/usr/bin/env python3
"""Evaluate the miniLLM tokenizer independently from the training script.

Examples:
    python eval/eval_tokenizer.py
    python eval/eval_tokenizer.py --input dataset/tokenizer_eval.jsonl
    python eval/eval_tokenizer.py --input dataset/tokenizer_eval --show-examples 8
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO

try:
    import jinja2  # noqa: F401 - required by transformers.apply_chat_template
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install them with:\n"
        "  pip install -r requirements.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = PROJECT_ROOT / "model" / "tokenizer"
DEFAULT_REPORT = PROJECT_ROOT / "eval" / "tokenizer_eval_report.json"
SUPPORTED_SUFFIXES = (".jsonl", ".jsonl.gz", ".txt", ".txt.gz")

EXPECTED_CORE_IDS = {
    "<|endoftext|>": 0,
    "<|im_start|>": 1,
    "<|im_end|>": 2,
}
STRUCTURAL_TOKENS = [
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
]

BUILTIN_SAMPLES = {
    "chinese": [
        "人工智能是计算机科学的重要研究方向。",
        "清晨的阳光穿过窗帘，桌上的书页被风轻轻翻动。",
        "北京、上海、深圳和杭州都是中国的重要城市。",
    ],
    "english": [
        "Large language models predict the next token in a sequence.",
        "A tokenizer converts text into integer identifiers and back again.",
    ],
    "mixed": [
        "miniLLM 使用 ByteLevel BPE 构建一个 8192 词表的 tokenizer。",
        "Transformer 可以处理中文、English、数字 12345 和 URL: https://example.com。",
    ],
    "code": [
        "def add(a, b):\n    return a + b",
        "SELECT name FROM users WHERE active = true ORDER BY id DESC;",
        "const message = `hello ${user.name}`;\nconsole.log(message);",
    ],
    "edge_cases": [
        "生僻字：𠮷、龘、靐；emoji：🙂🚀🧠。",
        "第一行\n    第二行使用四个空格\n\t第三行使用 Tab",
        "数学：E=mc²，x∈[0,1]，增长率为 12.5%，价格是 ¥19.90。",
        "<think>分析过程</think><tool_call>{\"name\":\"search\"}</tool_call>",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate miniLLM's tokenizer.")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help="Tokenizer directory (default: model/tokenizer).",
    )
    parser.add_argument(
        "--input",
        nargs="*",
        type=Path,
        default=[],
        help="Optional JSONL/TXT evaluation files or directories.",
    )
    parser.add_argument(
        "--expected-vocab-size",
        type=int,
        default=8192,
        help="Expected vocabulary size (default: 8192; use 0 to disable).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000,
        help="Maximum external samples to evaluate (default: 10000).",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="Print tokenization details for this many samples (default: 5).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="JSON report path (default: eval/tokenizer_eval_report.json).",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def is_supported(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def resolve_inputs(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = project_path(raw_path)
        if path.is_file():
            if not is_supported(path):
                raise ValueError(f"Unsupported evaluation file: {path}")
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file() and is_supported(candidate)
            )
        else:
            raise FileNotFoundError(f"Evaluation path does not exist: {path}")
    return sorted(set(files), key=str)


def open_text(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def extract_message_content(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts) or None
    return None


def extract_text(row: Any) -> str | None:
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return None

    for key in ("text", "content", "code"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value

    for key in ("conversations", "messages"):
        messages = row.get(key)
        if isinstance(messages, list):
            parts = [extract_message_content(message) for message in messages]
            parts = [part for part in parts if part]
            if parts:
                return "\n".join(parts)

    parts = [
        row[key]
        for key in ("instruction", "input", "output", "response")
        if isinstance(row.get(key), str) and row[key].strip()
    ]
    return "\n".join(parts) if parts else None


def iter_file_samples(path: Path) -> Iterator[str]:
    name = path.name.lower()
    is_jsonl = name.endswith(".jsonl") or name.endswith(".jsonl.gz")
    with open_text(path) as handle:
        for line in handle:
            if not is_jsonl:
                text = line.rstrip("\n")
            else:
                try:
                    text = extract_text(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if text and text.strip():
                yield text


def percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def token_preview(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_tokens: int = 80,
) -> dict[str, Any]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    visible_ids = ids[:max_tokens]
    return {
        "text": text,
        "token_count": len(ids),
        "ids": visible_ids,
        "tokens": tokenizer.convert_ids_to_tokens(visible_ids),
        "truncated": len(ids) > max_tokens,
    }


def evaluate_samples(
    tokenizer: PreTrainedTokenizerBase,
    samples: IterableSample,
    show_examples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    token_lengths: list[int] = []
    total_characters = 0
    total_bytes = 0
    total_tokens = 0
    unknown_tokens = 0
    roundtrip_failures: list[dict[str, str]] = []
    previews: list[dict[str, Any]] = []
    unk_id = tokenizer.unk_token_id

    for category, text in samples:
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)

        token_lengths.append(len(ids))
        total_characters += len(text)
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(ids)
        if unk_id is not None:
            unknown_tokens += sum(token_id == unk_id for token_id in ids)

        if decoded != text and len(roundtrip_failures) < 20:
            roundtrip_failures.append(
                {"category": category, "input": text, "decoded": decoded}
            )

        if len(previews) < show_examples:
            preview = token_preview(tokenizer, text)
            preview["category"] = category
            previews.append(preview)

    sample_count = len(token_lengths)
    metrics = {
        "sample_count": sample_count,
        "total_characters": total_characters,
        "total_utf8_bytes": total_bytes,
        "total_tokens": total_tokens,
        "characters_per_token": round(total_characters / max(total_tokens, 1), 6),
        "bytes_per_token": round(total_bytes / max(total_tokens, 1), 6),
        "unknown_tokens": unknown_tokens,
        "unknown_token_rate": round(unknown_tokens / max(total_tokens, 1), 8),
        "roundtrip_failures": len(roundtrip_failures),
        "roundtrip_failure_examples": roundtrip_failures,
        "tokens_per_sample": {
            "mean": round(statistics.mean(token_lengths), 4) if token_lengths else 0.0,
            "median": round(statistics.median(token_lengths), 4) if token_lengths else 0.0,
            "p95": round(percentile(token_lengths, 0.95), 4),
            "maximum": max(token_lengths, default=0),
        },
    }
    return metrics, previews


# An alias keeps the function annotation readable on Python 3.10+.
IterableSample = Iterator[tuple[str, str]]


def iter_all_samples(input_files: Sequence[Path], max_external: int) -> IterableSample:
    for category, texts in BUILTIN_SAMPLES.items():
        for text in texts:
            yield category, text

    external_count = 0
    for path in input_files:
        for text in iter_file_samples(path):
            if max_external and external_count >= max_external:
                return
            yield "external", text
            external_count += 1


def check_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    expected_vocab_size: int,
) -> tuple[dict[str, Any], bool]:
    checks: dict[str, Any] = {}
    passed = True

    checks["vocab_size"] = {
        "actual": len(tokenizer),
        "expected": expected_vocab_size or None,
        "passed": not expected_vocab_size or len(tokenizer) == expected_vocab_size,
    }
    passed &= checks["vocab_size"]["passed"]

    core_ids = {
        token: tokenizer.convert_tokens_to_ids(token) for token in EXPECTED_CORE_IDS
    }
    checks["core_token_ids"] = {
        "actual": core_ids,
        "expected": EXPECTED_CORE_IDS,
        "passed": core_ids == EXPECTED_CORE_IDS,
    }
    passed &= checks["core_token_ids"]["passed"]

    structural_results = {}
    for token in STRUCTURAL_TOKENS:
        expected_id = tokenizer.convert_tokens_to_ids(token)
        ids = tokenizer.encode(token, add_special_tokens=False)
        structural_results[token] = {
            "id": expected_id,
            "encoded_ids": ids,
            "atomic": len(ids) == 1 and ids[0] == expected_id,
        }
    checks["structural_tokens"] = {
        "tokens": structural_results,
        "passed": all(result["atomic"] for result in structural_results.values()),
    }
    passed &= checks["structural_tokens"]["passed"]

    visible_text = "<think>测试</think>"
    visible_ids = tokenizer.encode(visible_text, add_special_tokens=False)
    visible_decoded = tokenizer.decode(visible_ids, skip_special_tokens=True)
    checks["structural_tokens_visible"] = {
        "decoded": visible_decoded,
        "expected": visible_text,
        "passed": visible_decoded == visible_text,
    }
    passed &= checks["structural_tokens_visible"]["passed"]

    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "你好"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        expected_parts = [
            "<|im_start|>user\n",
            "你好<|im_end|>\n",
            "<|im_start|>assistant\n",
        ]
        template_passed = all(part in rendered for part in expected_parts)
        checks["chat_template"] = {
            "rendered": rendered,
            "passed": template_passed,
        }
    except Exception as exc:  # Report template errors without losing other metrics.
        checks["chat_template"] = {"error": str(exc), "passed": False}
    passed &= checks["chat_template"]["passed"]

    checks["overall_passed"] = bool(passed)
    return checks, bool(passed)


def print_summary(
    tokenizer_path: Path,
    checks: dict[str, Any],
    metrics: dict[str, Any],
    previews: Sequence[dict[str, Any]],
) -> None:
    status = "PASS" if checks["overall_passed"] and not metrics["roundtrip_failures"] else "FAIL"
    print(f"Tokenizer : {tokenizer_path}")
    print(f"Status    : {status}")
    print(f"Vocab     : {checks['vocab_size']['actual']}")
    print(f"Samples   : {metrics['sample_count']}")
    print(f"Tokens    : {metrics['total_tokens']}")
    print(f"Chars/tok : {metrics['characters_per_token']}")
    print(f"Bytes/tok : {metrics['bytes_per_token']}")
    print(f"UNK rate  : {metrics['unknown_token_rate']:.8f}")
    print(f"Round-trip failures: {metrics['roundtrip_failures']}")

    if previews:
        print("\nTokenization examples:")
    for index, preview in enumerate(previews, start=1):
        print(f"\n[{index}] {preview['category']} | {preview['token_count']} tokens")
        print(f"text   = {preview['text']!r}")
        print(f"tokens = {preview['tokens']}")


def main() -> int:
    args = parse_args()
    if args.max_samples < 0 or args.show_examples < 0:
        print("ERROR: --max-samples and --show-examples cannot be negative.", file=sys.stderr)
        return 2

    try:
        tokenizer_path = project_path(args.tokenizer)
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"Tokenizer directory does not exist: {tokenizer_path}")

        input_files = resolve_inputs(args.input)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        checks, checks_passed = check_tokenizer(tokenizer, args.expected_vocab_size)
        metrics, previews = evaluate_samples(
            tokenizer,
            iter_all_samples(input_files, args.max_samples),
            args.show_examples,
        )

        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tokenizer_path": str(tokenizer_path),
            "input_files": [str(path) for path in input_files],
            "checks": checks,
            "metrics": metrics,
            "examples": previews,
        }
        report_path = project_path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print_summary(tokenizer_path, checks, metrics, previews)
        print(f"\nReport: {report_path}")
        return 0 if checks_passed and metrics["roundtrip_failures"] == 0 else 1
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

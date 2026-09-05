#!/usr/bin/env python3
"""Extract a 150 MB multi-language The Stack Smol code corpus.

The output contains only ``{"text": "..."}`` rows, with the source code stored
in ``text`` so miniLLM's tokenizer trainer can read it directly.

Before first use, accept the dataset terms in a browser and authenticate:
    https://huggingface.co/datasets/bigcode/the-stack-smol
    hf auth login

Default language byte targets:
    Python 60%; JavaScript 10%; TypeScript 5%; Java 10%; C++ 5%; C 5%;
    Shell 2%; Go 2%; Rust 1%.

Run:
    python scripts/extract_stack_smol_code.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'datasets'. Install project dependencies with:\n"
        "  pip install -r requirements.txt"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def resolve_project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

def clean_text(value: Any) -> str | None:
    """Normalize line endings and remove NULs without rewriting content."""
    if not isinstance(value, str):
        return None
    text = value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = text.strip()
    return text or None

def format_size(byte_count: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(byte_count)
    for unit in units:
        if value < 1000 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1000
    return f"{value:.2f} TB"


def write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def validate_common_args(args: argparse.Namespace) -> None:
    if args.target_mb <= 0:
        raise ValueError("--target-mb must be greater than 0.")
    if args.min_chars < 1:
        raise ValueError("--min-chars must be at least 1.")
    if args.max_chars < 0 or (args.max_chars and args.max_chars < args.min_chars):
        raise ValueError("--max-chars must be 0 or greater than/equal to --min-chars.")
    if args.max_documents < 0:
        raise ValueError("--max-documents cannot be negative.")
    if args.shuffle_buffer < 0:
        raise ValueError("--shuffle-buffer cannot be negative.")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1.")


def prepare_output(args: argparse.Namespace) -> tuple[Path, Path, int]:
    output_path = resolve_project_path(args.output)
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError("--output must end with .jsonl.")
    metadata_path = output_path.with_suffix(".metadata.json")

    if (output_path.exists() or metadata_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use a different --output or pass --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_bytes = int(args.target_mb * 1_000_000)
    free_bytes = shutil.disk_usage(output_path.parent).free
    required_bytes = int(target_bytes * 1.05)
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient free disk space: need about {format_size(required_bytes)}, "
            f"available {format_size(free_bytes)}."
        )
    return output_path, metadata_path, target_bytes


DATASET_NAME = "bigcode/the-stack-smol"
DATASET_SPLIT = "train"
DATASET_URL = "https://huggingface.co/datasets/bigcode/the-stack-smol"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "dataset" / "tokenizer_corpus" / "stack_smol_code_150mb.jsonl"
)

# Ordered cumulative quotas keep the overall output close to the requested
# size: any overshoot or shortfall from one language is carried into the next.
LANGUAGE_MIX: tuple[tuple[str, float], ...] = (
    ("python", 0.60),
    ("javascript", 0.10),
    ("typescript", 0.05),
    ("java", 0.10),
    ("c++", 0.05),
    ("c", 0.05),
    ("shell", 0.02),
    ("go", 0.02),
    ("rust", 0.01),
)

ACCESS_HINT = (
    "The Stack Smol requires accepting its terms and authenticating. First open\n"
    "  https://huggingface.co/datasets/bigcode/the-stack-smol\n"
    "and click Access repository, then run:\n"
    "  hf auth login"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream 150 MB of The Stack Smol into tokenizer-ready JSONL."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output JSONL file "
            "(default: dataset/tokenizer_corpus/stack_smol_code_150mb.jsonl)."
        ),
    )
    parser.add_argument(
        "--target-mb",
        type=float,
        default=150.0,
        help="Approximate output size in decimal MB, where 1 MB=10^6 bytes (default: 150).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help=(
            "Hugging Face branch, tag, or commit revision "
            "(default: main; pass a commit hash for strict reproducibility)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Per-language streaming shuffle seed (default: 42).",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=5_000,
        help="Per-language shuffle buffer; use 0 to keep source order (default: 5000).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=100,
        help="Skip code files shorter than this many characters (default: 100).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=100_000,
        help="Skip code files longer than this many characters; 0 disables (default: 100000).",
    )
    parser.add_argument(
        "--max-line-length",
        type=int,
        default=1_000,
        help=(
            "Skip files whose source metadata reports a longer line; 0 disables "
            "(default: 1000)."
        ),
    )
    parser.add_argument(
        "--min-alphanum-fraction",
        type=float,
        default=0.20,
        help=(
            "Skip files below this alphanumeric-character fraction; 0 disables "
            "(default: 0.20)."
        ),
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=0,
        help="Optional written-document limit for testing; 0 disables.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1_000,
        help="Print and checkpoint progress every N written files (default: 1000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output and metadata file.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    validate_common_args(args)
    if args.max_line_length < 0:
        raise ValueError("--max-line-length cannot be negative.")
    if not 0 <= args.min_alphanum_fraction <= 1:
        raise ValueError("--min-alphanum-fraction must be between 0 and 1.")
    mix_total = sum(ratio for _, ratio in LANGUAGE_MIX)
    if abs(mix_total - 1.0) > 1e-9:
        raise RuntimeError(f"Internal language mix must sum to 1.0, got {mix_total}.")


def looks_like_access_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in message
        for marker in (
            "gated",
            "unauthorized",
            "401",
            "403",
            "access to dataset",
            "authentication",
        )
    )


def code_jsonl_line(text: str, row: dict[str, Any], language: str) -> str:
    """Keep license/provenance fields while exposing code through ``text``."""
    record = {
        "text": text,
        "language": language,
        "licenses": row.get("licenses"),
        "repository_name": row.get("repository_name"),
        "path": row.get("path"),
    }
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"


def extract(args: argparse.Namespace) -> Path:
    validate_args(args)
    output_path, metadata_path, target_bytes = prepare_output(args)

    print(f"Dataset      : {DATASET_NAME}")
    print(f"Revision     : {args.revision}")
    print(f"Target       : {format_size(target_bytes)}")
    print(f"Output       : {output_path}")
    print("Access       : gated; accepted terms and HF authentication are required")
    print(f"Shuffle      : buffer={args.shuffle_buffer}, seed={args.seed}")
    print("Language mix : " + ", ".join(f"{lang}={ratio:.0%}" for lang, ratio in LANGUAGE_MIX))

    # Check gated access before creating the output file. This prevents an
    # authentication failure from leaving an empty JSONL that would require
    # --overwrite on the next attempt.
    first_language = LANGUAGE_MIX[0][0]
    first_dataset = load_dataset(
        DATASET_NAME,
        data_dir=f"data/{first_language}",
        split=DATASET_SPLIT,
        streaming=True,
        revision=args.revision,
    )
    if args.shuffle_buffer:
        first_dataset = first_dataset.shuffle(
            seed=args.seed,
            buffer_size=args.shuffle_buffer,
        )

    started_at = time.monotonic()
    source_rows_seen = 0
    documents_written = 0
    short_documents_skipped = 0
    long_documents_skipped = 0
    long_line_files_skipped = 0
    low_alphanum_files_skipped = 0
    invalid_documents_skipped = 0
    bytes_written = 0
    status = "running"
    last_error: str | None = None
    current_language: str | None = None
    language_stats: dict[str, dict[str, int | float | str]] = {
        language: {
            "target_ratio": ratio,
            "target_bytes": int(target_bytes * ratio),
            "source_rows_seen": 0,
            "documents_written": 0,
            "bytes_written": 0,
            "status": "pending",
        }
        for language, ratio in LANGUAGE_MIX
    }

    def metadata(current_status: str, error: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": current_status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "dataset": DATASET_NAME,
                "split": DATASET_SPLIT,
                "revision": args.revision,
                "url": DATASET_URL,
                "terms": "The Stack Terms of Use and each source file's license",
                "streaming": True,
            },
            "sampling": {
                "seed": args.seed,
                "shuffle_buffer": args.shuffle_buffer,
                "min_chars": args.min_chars,
                "max_chars": args.max_chars,
                "max_line_length": args.max_line_length,
                "min_alphanum_fraction": args.min_alphanum_fraction,
                "max_documents": args.max_documents,
                "language_mix": {
                    language: ratio for language, ratio in LANGUAGE_MIX
                },
            },
            "output": {
                "path": str(output_path),
                "format": "jsonl",
                "schema": {
                    "text": "string",
                    "language": "string",
                    "licenses": "source value",
                    "repository_name": "source value",
                    "path": "source value",
                },
                "target_bytes": target_bytes,
                "bytes_written": bytes_written,
            },
            "progress": {
                "current_language": current_language,
                "source_rows_seen": source_rows_seen,
                "documents_written": documents_written,
                "short_documents_skipped": short_documents_skipped,
                "long_documents_skipped": long_documents_skipped,
                "long_line_files_skipped": long_line_files_skipped,
                "low_alphanum_files_skipped": low_alphanum_files_skipped,
                "invalid_documents_skipped": invalid_documents_skipped,
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
                "languages": language_stats,
            },
        }
        if error:
            result["error"] = error
        return result

    def checkpoint(current_status: str, error: str | None = None) -> None:
        write_metadata(metadata_path, metadata(current_status, error))

    cumulative_ratio = 0.0
    try:
        with output_path.open("w", encoding="utf-8", buffering=1024 * 1024) as output:
            checkpoint("running")
            for language_index, (language, ratio) in enumerate(LANGUAGE_MIX):
                current_language = language
                cumulative_ratio += ratio
                cumulative_target = (
                    target_bytes
                    if language_index == len(LANGUAGE_MIX) - 1
                    else int(target_bytes * cumulative_ratio)
                )
                stats = language_stats[language]
                stats["status"] = "loading"
                checkpoint("running")
                print(
                    f"Loading {language} (cumulative target "
                    f"{format_size(cumulative_target)})...",
                    flush=True,
                )

                if language_index == 0:
                    dataset = first_dataset
                else:
                    dataset = load_dataset(
                        DATASET_NAME,
                        data_dir=f"data/{language}",
                        split=DATASET_SPLIT,
                        streaming=True,
                        revision=args.revision,
                    )
                    if args.shuffle_buffer:
                        dataset = dataset.shuffle(
                            seed=args.seed + language_index,
                            buffer_size=args.shuffle_buffer,
                        )

                stats["status"] = "running"
                for row in dataset:
                    source_rows_seen += 1
                    stats["source_rows_seen"] = int(stats["source_rows_seen"]) + 1

                    text = clean_text(row.get("content") if isinstance(row, dict) else None)
                    if text is None:
                        invalid_documents_skipped += 1
                        continue
                    if len(text) < args.min_chars:
                        short_documents_skipped += 1
                        continue
                    if args.max_chars and len(text) > args.max_chars:
                        long_documents_skipped += 1
                        continue

                    reported_max_line = row.get("max_line_length")
                    if (
                        args.max_line_length
                        and isinstance(reported_max_line, (int, float))
                        and reported_max_line > args.max_line_length
                    ):
                        long_line_files_skipped += 1
                        continue

                    alphanum_fraction = row.get("alphanum_fraction")
                    if (
                        args.min_alphanum_fraction
                        and isinstance(alphanum_fraction, (int, float))
                        and alphanum_fraction < args.min_alphanum_fraction
                    ):
                        low_alphanum_files_skipped += 1
                        continue

                    line = code_jsonl_line(text, row, language)
                    line_bytes = len(line.encode("utf-8"))
                    output.write(line)
                    bytes_written += line_bytes
                    documents_written += 1
                    stats["bytes_written"] = int(stats["bytes_written"]) + line_bytes
                    stats["documents_written"] = int(stats["documents_written"]) + 1

                    if documents_written % args.log_every == 0:
                        output.flush()
                        checkpoint("running")
                        elapsed = max(time.monotonic() - started_at, 1e-9)
                        percentage = min(bytes_written / target_bytes * 100, 100.0)
                        print(
                            f"[{percentage:6.2f}%] files={documents_written:,} "
                            f"language={language} size={format_size(bytes_written)} "
                            f"speed={format_size(int(bytes_written / elapsed))}/s",
                            flush=True,
                        )

                    if bytes_written >= cumulative_target:
                        stats["status"] = "quota_reached"
                        break
                    if args.max_documents and documents_written >= args.max_documents:
                        stats["status"] = "document_limit_reached"
                        status = "document_limit_reached"
                        break
                else:
                    stats["status"] = "source_exhausted"

                output.flush()
                checkpoint("running" if status == "running" else status)

                if status == "document_limit_reached":
                    break
                if bytes_written >= target_bytes:
                    status = "complete"
                    break
            else:
                status = "complete" if bytes_written >= target_bytes else "source_exhausted"
            output.flush()
    except KeyboardInterrupt:
        status = "interrupted"
        print("\nInterrupted by user; partial JSONL remains valid.", file=sys.stderr)
    except Exception as exc:
        status = "error"
        last_error = f"{type(exc).__name__}: {exc}"
        if looks_like_access_error(exc):
            last_error += f"\n\n{ACCESS_HINT}"
        raise RuntimeError(last_error) from exc
    finally:
        checkpoint(status, last_error)

    elapsed = time.monotonic() - started_at
    print("\nExtraction finished.")
    print(f"Status       : {status}")
    print(f"Source rows  : {source_rows_seen:,}")
    print(f"Code files   : {documents_written:,}")
    print(f"Output size  : {format_size(bytes_written)}")
    print(f"Elapsed      : {elapsed / 60:.2f} minutes")
    print(f"JSONL        : {output_path}")
    print(f"Metadata     : {metadata_path}")

    if status == "interrupted":
        raise KeyboardInterrupt
    if status != "complete" and not args.max_documents:
        print(
            f"WARNING: extraction ended with status={status} before reaching target size.",
            file=sys.stderr,
        )
    return output_path


def main() -> int:
    try:
        extract(parse_args())
    except KeyboardInterrupt:
        return 130
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        message = f"{exc}"
        if looks_like_access_error(exc) and ACCESS_HINT not in message:
            message += f"\n\n{ACCESS_HINT}"
        print(f"ERROR: {message}", file=sys.stderr)
        return 1
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if looks_like_access_error(exc):
            message += f"\n\n{ACCESS_HINT}"
        print(f"ERROR: The Stack Smol extraction failed: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Stream Chinese Wikipedia into miniLLM tokenizer-ready JSONL.

The script reads the cleaned Chinese Wikipedia snapshot from Hugging Face in
streaming mode, so it does not download the full dataset before extraction. It
stops when the output JSONL reaches the requested UTF-8 byte size.

Default output:
    dataset/tokenizer_corpus/wikipedia_zh_600mb.jsonl

Run the 600 MB extraction:
    python scripts/extract_wikipedia_zh.py

Small connectivity test:
    python scripts/extract_wikipedia_zh.py \
        --target-mb 1 \
        --max-documents 100 \
        --output dataset/tokenizer_corpus/wikipedia_zh_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'datasets'. Install project dependencies with:\n"
        "  pip install -r requirements.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "dataset"
    / "tokenizer_corpus"
    / "wikipedia_zh_600mb.jsonl"
)
DATASET_NAME = "wikimedia/wikipedia"
DATASET_SUBSET = "20231101.zh"
DATASET_SPLIT = "train"
DATASET_URL = "https://huggingface.co/datasets/wikimedia/wikipedia"
DATASET_LICENSES = ["CC BY-SA 3.0", "GFDL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream cleaned Chinese Wikipedia into tokenizer-ready JSONL."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output JSONL file "
            "(default: dataset/tokenizer_corpus/wikipedia_zh_600mb.jsonl)."
        ),
    )
    parser.add_argument(
        "--target-mb",
        type=float,
        default=600.0,
        help=(
            "Approximate JSONL output size in decimal MB, where 1 MB=10^6 bytes "
            "(default: 600)."
        ),
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
        help="Streaming shuffle seed (default: 42).",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=10_000,
        help="Streaming shuffle buffer; use 0 to keep source order (default: 10000).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=100,
        help="Skip articles shorter than this many characters (default: 100).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=200_000,
        help=(
            "Skip articles longer than this many characters; 0 disables "
            "(default: 200000)."
        ),
    )
    parser.add_argument(
        "--exclude-title",
        action="store_true",
        help="Do not prepend each article title to its body text.",
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
        help="Print and checkpoint progress every N written articles (default: 1000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output and metadata file.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def clean_text(value: Any) -> str | None:
    """Normalize line endings and remove NULs without rewriting article text."""
    if not isinstance(value, str):
        return None
    text = value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = text.strip()
    return text or None


def article_text(row: Any, include_title: bool) -> str | None:
    if not isinstance(row, dict):
        return None

    body = clean_text(row.get("text"))
    if body is None:
        return None
    if not include_title:
        return body

    title = clean_text(row.get("title"))
    if title is None:
        return body

    # Avoid duplicating the title if a future dataset revision puts it at the
    # beginning of the article body.
    first_line = body.split("\n", 1)[0].strip()
    return body if first_line == title else f"{title}\n\n{body}"


def format_size(byte_count: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(byte_count)
    for unit in units:
        if value < 1000 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1000
    return f"{value:.2f} TB"


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Atomically update metadata so interrupted runs remain diagnosable."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def build_metadata(
    args: argparse.Namespace,
    output_path: Path,
    status: str,
    target_bytes: int,
    source_rows_seen: int,
    documents_written: int,
    short_documents_skipped: int,
    long_documents_skipped: int,
    invalid_documents_skipped: int,
    bytes_written: int,
    started_at: float,
    error: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": DATASET_NAME,
            "subset": DATASET_SUBSET,
            "split": DATASET_SPLIT,
            "revision": args.revision,
            "url": DATASET_URL,
            "licenses": DATASET_LICENSES,
            "streaming": True,
        },
        "sampling": {
            "seed": args.seed,
            "shuffle_buffer": args.shuffle_buffer,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "include_title": not args.exclude_title,
            "max_documents": args.max_documents,
        },
        "output": {
            "path": str(output_path),
            "format": "jsonl",
            "schema": {"text": "string"},
            "target_bytes": target_bytes,
            "bytes_written": bytes_written,
        },
        "progress": {
            "source_rows_seen": source_rows_seen,
            "documents_written": documents_written,
            "short_documents_skipped": short_documents_skipped,
            "long_documents_skipped": long_documents_skipped,
            "invalid_documents_skipped": invalid_documents_skipped,
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
        },
    }
    if error:
        metadata["error"] = error
    return metadata


def validate_args(args: argparse.Namespace) -> None:
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


def extract(args: argparse.Namespace) -> Path:
    validate_args(args)
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

    print(f"Dataset      : {DATASET_NAME}")
    print(f"Subset       : {DATASET_SUBSET}")
    print(f"Revision     : {args.revision}")
    print(f"Target       : {format_size(target_bytes)}")
    print(f"Output       : {output_path}")
    print(f"Title        : {'included' if not args.exclude_title else 'excluded'}")
    print(f"Shuffle      : buffer={args.shuffle_buffer}, seed={args.seed}")
    print("Loading streaming dataset. The first connection may take a while...", flush=True)

    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_SUBSET,
        split=DATASET_SPLIT,
        streaming=True,
        revision=args.revision,
    )
    if args.shuffle_buffer:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    started_at = time.monotonic()
    source_rows_seen = 0
    documents_written = 0
    short_documents_skipped = 0
    long_documents_skipped = 0
    invalid_documents_skipped = 0
    bytes_written = 0
    status = "running"
    last_error: str | None = None

    def checkpoint(current_status: str, error: str | None = None) -> None:
        write_metadata(
            metadata_path,
            build_metadata(
                args=args,
                output_path=output_path,
                status=current_status,
                target_bytes=target_bytes,
                source_rows_seen=source_rows_seen,
                documents_written=documents_written,
                short_documents_skipped=short_documents_skipped,
                long_documents_skipped=long_documents_skipped,
                invalid_documents_skipped=invalid_documents_skipped,
                bytes_written=bytes_written,
                started_at=started_at,
                error=error,
            ),
        )

    try:
        with output_path.open("w", encoding="utf-8", buffering=1024 * 1024) as output:
            checkpoint("running")
            for row in dataset:
                source_rows_seen += 1
                text = article_text(row, include_title=not args.exclude_title)
                if text is None:
                    invalid_documents_skipped += 1
                    continue
                if len(text) < args.min_chars:
                    short_documents_skipped += 1
                    continue
                if args.max_chars and len(text) > args.max_chars:
                    long_documents_skipped += 1
                    continue

                line = json.dumps(
                    {"text": text},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ) + "\n"
                output.write(line)
                bytes_written += len(line.encode("utf-8"))
                documents_written += 1

                if documents_written % args.log_every == 0:
                    output.flush()
                    checkpoint("running")
                    elapsed = max(time.monotonic() - started_at, 1e-9)
                    speed = bytes_written / elapsed
                    percentage = min(bytes_written / target_bytes * 100, 100.0)
                    print(
                        f"[{percentage:6.2f}%] articles={documents_written:,} "
                        f"size={format_size(bytes_written)} "
                        f"speed={format_size(int(speed))}/s",
                        flush=True,
                    )

                if bytes_written >= target_bytes:
                    status = "complete"
                    break
                if args.max_documents and documents_written >= args.max_documents:
                    status = "document_limit_reached"
                    break
            else:
                status = "source_exhausted"

            output.flush()
    except KeyboardInterrupt:
        status = "interrupted"
        print("\nInterrupted by user; partial JSONL remains valid.", file=sys.stderr)
    except Exception as exc:
        status = "error"
        last_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        checkpoint(status, last_error)

    elapsed = time.monotonic() - started_at
    print("\nExtraction finished.")
    print(f"Status       : {status}")
    print(f"Source rows  : {source_rows_seen:,}")
    print(f"Articles     : {documents_written:,}")
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
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"ERROR: Wikipedia extraction failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

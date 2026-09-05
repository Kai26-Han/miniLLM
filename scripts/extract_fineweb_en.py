#!/usr/bin/env python3
"""Extract a 300 MB English FineWeb sample for tokenizer training.

The sample-10BT configuration is streamed and the script stops after the local
JSONL reaches 300 MB; the complete sample is not downloaded.

Run:
    python scripts/extract_fineweb_en.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DatasetConfig:
    label: str
    dataset_name: str
    subset: str | None
    split: str
    url: str
    licenses: tuple[str, ...]
    default_output: Path
    default_target_mb: float
    default_shuffle_buffer: int = 20_000
    default_min_chars: int = 200
    default_max_chars: int = 200_000
    text_field: str = "text"
    gated_access_hint: str | None = None


def build_parser(config: DatasetConfig, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.default_output,
        help=f"Output JSONL file (default: {relative_display(config.default_output)}).",
    )
    parser.add_argument(
        "--target-mb",
        type=float,
        default=config.default_target_mb,
        help=(
            "Approximate JSONL output size in decimal MB, where 1 MB=10^6 bytes "
            f"(default: {config.default_target_mb:g})."
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
        default=config.default_shuffle_buffer,
        help=(
            "Streaming shuffle buffer; use 0 to keep source order "
            f"(default: {config.default_shuffle_buffer})."
        ),
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=config.default_min_chars,
        help=(
            "Skip documents shorter than this many characters "
            f"(default: {config.default_min_chars})."
        ),
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=config.default_max_chars,
        help=(
            "Skip documents longer than this many characters; 0 disables "
            f"(default: {config.default_max_chars})."
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
        help="Print and checkpoint progress every N written documents (default: 1000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output and metadata file.",
    )
    return parser


def relative_display(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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


def jsonl_line(text: str) -> str:
    return json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")) + "\n"


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


def access_error_message(config: DatasetConfig, exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    lowered = message.lower()
    gated_markers = (
        "gated",
        "unauthorized",
        "401",
        "403",
        "access to dataset",
        "authentication",
    )
    if config.gated_access_hint and any(marker in lowered for marker in gated_markers):
        return f"{message}\n\n{config.gated_access_hint}"
    return message


def extract(args: argparse.Namespace, config: DatasetConfig) -> Path:
    validate_common_args(args)
    output_path, metadata_path, target_bytes = prepare_output(args)

    print(f"Dataset      : {config.dataset_name}")
    print(f"Subset       : {config.subset or 'default'}")
    print(f"Revision     : {args.revision}")
    print(f"Target       : {format_size(target_bytes)}")
    print(f"Output       : {output_path}")
    print(f"Shuffle      : buffer={args.shuffle_buffer}, seed={args.seed}")
    if config.gated_access_hint:
        print("Access       : gated; accepted terms and HF authentication are required")
    print("Loading streaming dataset. The first connection may take a while...", flush=True)

    load_kwargs: dict[str, Any] = {
        "path": config.dataset_name,
        "split": config.split,
        "streaming": True,
        "revision": args.revision,
    }
    if config.subset:
        load_kwargs["name"] = config.subset
    dataset = load_dataset(**load_kwargs)
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

    def metadata(current_status: str, error: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": current_status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "dataset": config.dataset_name,
                "subset": config.subset,
                "split": config.split,
                "revision": args.revision,
                "url": config.url,
                "licenses": list(config.licenses),
                "streaming": True,
            },
            "sampling": {
                "seed": args.seed,
                "shuffle_buffer": args.shuffle_buffer,
                "min_chars": args.min_chars,
                "max_chars": args.max_chars,
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
            result["error"] = error
        return result

    def checkpoint(current_status: str, error: str | None = None) -> None:
        write_metadata(metadata_path, metadata(current_status, error))

    try:
        with output_path.open("w", encoding="utf-8", buffering=1024 * 1024) as output:
            checkpoint("running")
            for row in dataset:
                source_rows_seen += 1
                value = row.get(config.text_field) if isinstance(row, dict) else None
                text = clean_text(value)
                if text is None:
                    invalid_documents_skipped += 1
                    continue
                if len(text) < args.min_chars:
                    short_documents_skipped += 1
                    continue
                if args.max_chars and len(text) > args.max_chars:
                    long_documents_skipped += 1
                    continue

                line = jsonl_line(text)
                output.write(line)
                bytes_written += len(line.encode("utf-8"))
                documents_written += 1

                if documents_written % args.log_every == 0:
                    output.flush()
                    checkpoint("running")
                    elapsed = max(time.monotonic() - started_at, 1e-9)
                    percentage = min(bytes_written / target_bytes * 100, 100.0)
                    print(
                        f"[{percentage:6.2f}%] docs={documents_written:,} "
                        f"size={format_size(bytes_written)} "
                        f"speed={format_size(int(bytes_written / elapsed))}/s",
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
        last_error = access_error_message(config, exc)
        raise RuntimeError(last_error) from exc
    finally:
        checkpoint(status, last_error)

    elapsed = time.monotonic() - started_at
    print("\nExtraction finished.")
    print(f"Status       : {status}")
    print(f"Source rows  : {source_rows_seen:,}")
    print(f"Documents    : {documents_written:,}")
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


def main(config: DatasetConfig, description: str) -> int:
    parser = build_parser(config, description)
    try:
        extract(parser.parse_args(), config)
    except KeyboardInterrupt:
        return 130
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {access_error_message(config, exc)}", file=sys.stderr)
        return 1
    except Exception as exc:
        message = access_error_message(config, exc)
        print(f"ERROR: {config.label} extraction failed: {message}", file=sys.stderr)
        return 1
    return 0


CONFIG = DatasetConfig(
    label="FineWeb English",
    dataset_name="HuggingFaceFW/fineweb",
    subset="sample-10BT",
    split="train",
    url="https://huggingface.co/datasets/HuggingFaceFW/fineweb",
    licenses=("ODC-By 1.0",),
    default_output=(
        PROJECT_ROOT / "dataset" / "tokenizer_corpus" / "fineweb_en_300mb.jsonl"
    ),
    default_target_mb=300.0,
    default_shuffle_buffer=20_000,
    default_min_chars=200,
    default_max_chars=200_000,
)


if __name__ == "__main__":
    raise SystemExit(
        main(CONFIG, "Stream 300 MB of English FineWeb into tokenizer-ready JSONL.")
    )


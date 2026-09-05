#!/usr/bin/env python3
"""Explicit training entry for offline black-box sequence distillation.

Teacher inference is intentionally not performed in this process.  Use
``scripts/generate_distill_data.py`` first; this entry validates the resulting
JSONL/manifest, selects distillation-specific output directories and then
executes the stable SFT trainer.  The actual objective is assistant-only CE on
teacher-generated hard labels, which is exactly the black-box distillation
objective.

Example:

    python trainer/train_distillation.py \
        --data-path dataset/distill/qwen3_1_7b_sft.jsonl \
        --batch-size 4 \
        --accumulation-steps 16 \
        --epochs 1

All arguments not declared by this wrapper are forwarded unchanged to
``trainer/train_sft.py``.  It therefore supports the same AMP, DDP, tracker,
checkpoint and resume options.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SFT_TRAINER = PROJECT_ROOT / "trainer" / "train_sft.py"
DEFAULT_DATA = PROJECT_ROOT / "dataset" / "distill" / "qwen3_1_7b_sft.jsonl"
DEFAULT_MODEL = PROJECT_ROOT / "out" / "pretrain"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "checkpoints" / "sft_distilled"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "out" / "sft_distilled"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "sft_distilled"


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def manifest_path(data_path: Path) -> Path:
    return data_path.with_suffix(data_path.suffix + ".manifest.json")


def load_distillation_manifest(
    data_path: Path,
    *,
    require_manifest: bool,
    allow_incomplete: bool,
) -> Mapping[str, Any] | None:
    """Validate provenance produced by ``generate_distill_data.py``."""

    sidecar = manifest_path(data_path)
    if not sidecar.is_file():
        if require_manifest:
            raise FileNotFoundError(
                f"distillation manifest not found: {sidecar}. Generate the data "
                "with scripts/generate_distill_data.py or pass "
                "--no-require-manifest for a manually prepared dataset."
            )
        return None
    try:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid distillation manifest {sidecar}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise TypeError(f"distillation manifest must be an object: {sidecar}")

    status = manifest.get("status")
    if status != "complete" and not allow_incomplete:
        raise ValueError(
            f"distillation manifest is {status!r}, expected 'complete': {sidecar}. "
            "Finish generation or pass --allow-incomplete-manifest deliberately."
        )
    config = manifest.get("config")
    stats = manifest.get("stats")
    if not isinstance(config, Mapping) or not isinstance(stats, Mapping):
        raise ValueError(f"manifest is missing config/stats objects: {sidecar}")

    recorded_output = config.get("output_path")
    if recorded_output and Path(str(recorded_output)).expanduser().resolve() != data_path:
        raise ValueError(
            f"manifest output_path does not match data file: {recorded_output!r} "
            f"!= {str(data_path)!r}"
        )
    processed_rows = int(stats.get("processed_rows", 0))
    distilled_rows = int(stats.get("distilled_rows", 0))
    if processed_rows <= 0:
        raise ValueError(f"manifest reports no processed rows: {sidecar}")
    if distilled_rows <= 0:
        raise ValueError(
            f"manifest reports no teacher-distilled rows: {sidecar}. This entry "
            "is for black-box distillation rather than ordinary SFT replay."
        )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Train miniLLM from out/pretrain on offline Qwen hard-label "
            "distillation data. Unknown options are forwarded to train_sft.py."
        ),
        epilog=(
            "Examples of forwarded options: --max-steps, --batch-size, --dtype, "
            "--tracker, --resume and --gradient-checkpointing."
        ),
    )
    parser.add_argument("--data-path", nargs="+", type=Path, default=[DEFAULT_DATA])
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Student tokenizer directory; defaults to --model-path.",
    )
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tracker-project", default="miniLLM-BlackBox-Distillation")
    parser.add_argument("--tracker-run-name", default=None)
    parser.add_argument("--tracker-log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--require-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require generation provenance next to every data file.",
    )
    parser.add_argument(
        "--allow-incomplete-manifest",
        action="store_true",
        help="Allow training on a deliberately partial generation output.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip manifest validation; file existence checks still run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved SFT command without executing it.",
    )
    return parser.parse_known_args(argv)


def validate_paths(args: argparse.Namespace) -> None:
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

    if not SFT_TRAINER.is_file():
        raise FileNotFoundError(f"underlying SFT trainer not found: {SFT_TRAINER}")
    for data_path in args.data_path:
        if not data_path.is_file():
            raise FileNotFoundError(f"distillation data not found: {data_path}")
        if data_path.stat().st_size == 0:
            raise ValueError(f"distillation data is empty: {data_path}")
    if not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"student base model config not found: {args.model_path / 'config.json'}"
        )
    if not args.tokenizer_path.is_dir():
        raise FileNotFoundError(f"student tokenizer not found: {args.tokenizer_path}")


def build_train_argv(
    args: argparse.Namespace,
    forwarded: Sequence[str],
) -> list[str]:
    """Build an explicit argument list for the shared SFT implementation."""

    train_argv = [
        str(SFT_TRAINER),
        "--data-path",
        *(str(path) for path in args.data_path),
        "--model-path",
        str(args.model_path),
        "--tokenizer-path",
        str(args.tokenizer_path),
        "--save-dir",
        str(args.save_dir),
        "--output-dir",
        str(args.output_dir),
        "--tracker-project",
        args.tracker_project,
        "--tracker-log-dir",
        str(args.tracker_log_dir),
    ]
    run_name = args.tracker_run_name or (
        f"miniLLM-BlackBoxKD-{args.data_path[0].stem}"
    )
    train_argv.extend(["--tracker-run-name", run_name])
    train_argv.extend(forwarded)
    return train_argv


def print_preflight(
    args: argparse.Namespace,
    manifests: Sequence[Mapping[str, Any] | None],
) -> None:
    print("Training mode  : offline black-box sequence distillation", flush=True)
    print("Online teacher : disabled (hard labels are already in JSONL)", flush=True)
    print(f"Student init   : {args.model_path}", flush=True)
    print(f"Output model   : {args.output_dir}", flush=True)
    for index, (data_path, manifest) in enumerate(
        zip(args.data_path, manifests, strict=True), start=1
    ):
        print(f"Data {index:<9}: {data_path}", flush=True)
        if manifest is None:
            print("  provenance   : unchecked manual dataset", flush=True)
            continue
        config = manifest["config"]
        stats = manifest["stats"]
        processed = int(stats.get("processed_rows", 0))
        distilled = int(stats.get("distilled_rows", 0))
        replay = int(stats.get("replay_rows", 0))
        tools = int(stats.get("preserved_tool_rows", 0))
        print(f"  teacher      : {config.get('model_path', 'unknown')}", flush=True)
        print(
            f"  rows         : {processed:,} total; {distilled:,} distilled; "
            f"{replay:,} replay; {tools:,} preserved tool",
            flush=True,
        )
        print(
            f"  thinking     : {float(config.get('thinking_ratio', 0.0)):.1%}",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> None:
    args, forwarded = parse_args(argv)
    validate_paths(args)
    manifests: list[Mapping[str, Any] | None] = []
    for data_path in args.data_path:
        if args.skip_preflight:
            manifests.append(None)
        else:
            manifests.append(
                load_distillation_manifest(
                    data_path,
                    require_manifest=args.require_manifest,
                    allow_incomplete=args.allow_incomplete_manifest,
                )
            )
    print_preflight(args, manifests)
    train_argv = build_train_argv(args, forwarded)
    command = [sys.executable, *train_argv]
    if args.dry_run:
        print(f"Resolved command: {shlex.join(command)}", flush=True)
        return

    # Replace this lightweight wrapper with the training process.  torchrun's
    # rank environment is preserved, so DDP behaves exactly as train_sft.py.
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()

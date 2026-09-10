#!/usr/bin/env python3
"""单张 RTX 4090 的 Projector pretrain；命令路径相对 miniLLM-vlm 根目录。"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import minillm_vlm  # noqa: F401,E402

import torch

from minillm_vlm.dataset.vlm_dataset import (CaptionEncoder,
    VLMCollator, VLMDataset, atomic_json, load_parquet, prepare_index)
from minillm_vlm.trainer.trainer_utils import (VLM_ROOT, add_data_args, add_resource_args,
    autocast, data_identity, export_adapter, json_hash, load_resources, model_report,
    project_path, resolve_device, to_device, evaluate, make_loader,
    DistributedContext, _restore_rng_state, _rng_state, atomic_torch_save,
    build_cosine_scheduler, init_experiment_tracker, seed_everything)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_resource_args(parser)
    add_data_args(parser)
    parser.add_argument("--check-only", action="store_true", help="Run preflight without training or optimizer updates.")
    parser.add_argument("--prepare-index", action="store_true", help="With --check-only, validate all rows and build the data index.")
    parser.add_argument("--samples", type=int, default=8, help="Number of samples for --check-only forward/backward.")
    parser.add_argument("--report-path", type=project_path, default=VLM_ROOT / "out/preflight.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="Schedule horizon cap; 0 uses epochs.")
    parser.add_argument("--stop-after-steps", type=int, default=0,
                        help="Gracefully pause at this global step without changing the schedule.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--min-learning-rate", type=float, default=4e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--router-aux-weight", type=float, default=1.0,
                        help="Multiplier of Base's already-weighted router_aux_loss; 0 is a CE-only ablation.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--save-dir", type=project_path, default=VLM_ROOT / "checkpoints/pretrain")
    parser.add_argument("--output-dir", type=project_path, default=VLM_ROOT / "out/pretrain")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume a trusted local checkpoint; no argument uses save-dir/latest.pt.")
    parser.add_argument("--tracker", choices=["none", "swanlab", "wandb"], default="none")
    parser.add_argument("--tracker-project", default="miniLLM-VLM-Pretrain")
    parser.add_argument("--tracker-run-name", default=None)
    parser.add_argument("--tracker-mode", choices=["online", "offline"], default="online")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def validate_args(args):
    if args.check_only and args.resume is not None:
        raise ValueError("--check-only cannot be combined with --resume")
    if args.prepare_index and not args.check_only:
        raise ValueError("--prepare-index requires --check-only; training prepares its index automatically")
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This first pretrain implementation is single-GPU; do not launch multi-process torchrun")
    for key in ("epochs", "batch_size", "accumulation_steps", "learning_rate", "grad_clip",
                "max_seq_len", "eval_samples", "log_interval", "eval_interval", "save_interval"):
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    for key in ("max_steps", "stop_after_steps", "max_train_samples", "num_workers", "weight_decay", "router_aux_weight"):
        if getattr(args, key) < 0:
            raise ValueError(f"--{key.replace('_', '-')} must be nonnegative")
    if not 0 < args.val_ratio < 0.5 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("Require 0 < val-ratio < .5 and 0 <= warmup-ratio < 1")
    if not 0 < args.min_learning_rate <= args.learning_rate:
        raise ValueError("Require 0 < min-learning-rate <= learning-rate")


def check_resources(args, model, tokenizer, processor, identity, resources,
                    source, encoder, index, device, dtype):
    """仅检查前向/反向及冻结状态，不创建优化器或训练断点。"""
    report = {**model_report(model), "resources": resources, "identity": identity,
              "device": str(device), "dtype": str(dtype), "dataset_rows": len(source)}
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name()
        report["gpu_memory_gib"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if index is not None:
        report["data_stats"] = index["stats"]
        report["invalid_examples"] = index["invalid_examples"]
        report["split_sizes"] = {key: len(index[key]) for key in ("train", "validation")}
        if not index["train"] or not index["validation"]:
            raise ValueError("Empty train/validation split; inspect dataset size or val-ratio")
        indices = index["train"][:args.samples]
    else:
        indices = list(range(min(args.samples, len(source))))
    if not indices:
        raise ValueError("Dataset is empty")
    dataset = VLMDataset(source, indices, encoder, processor)
    samples = [dataset[i] for i in range(len(dataset))]
    report["sample_lengths"] = [{key: sample[key] for key in
                                 ("row_index", "original_length", "prompt_length", "truncated")}
                                for sample in samples]
    cpu_batch = VLMCollator(tokenizer.pad_token_id)(samples)
    model.to(device).train()
    batch = to_device(cpu_batch, device)
    with autocast(device, dtype):
        output = model(**batch)
    if not torch.isfinite(output.loss):
        raise FloatingPointError("Preflight forward produced a non-finite loss")
    output.loss.backward()
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable_names or any(not name.startswith("projector.") for name in trainable_names):
        raise RuntimeError("Freeze policy violation")
    if any(p.grad is not None for p in model.llm.parameters()) or any(p.grad is not None for p in model.vision_encoder.parameters()):
        raise RuntimeError("Frozen parameters unexpectedly received gradients")
    gradients = [p.grad for p in model.projector.parameters()]
    if any(g is None or not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Projector gradient is missing or non-finite")
    norm = sum(g.float().square().sum().item() for g in gradients) ** 0.5
    if norm <= 0:
        raise RuntimeError("Projector gradient is zero")
    report.update({"lm_loss": output.lm_loss.item(), "router_aux_loss": output.router_aux_loss.item(),
                   "projector_grad_norm": norm, "freeze_check": "passed",
                   "batch_shapes": {k: list(v.shape) for k, v in cpu_batch.items()}})
    if device.type == "cuda":
        report["peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
    atomic_json(report, args.report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Preflight passed. Report: {args.report_path}")
    model.zero_grad(set_to_none=True)
    return report


def train_update(model, cpu_batches, optimizer, scaler, device, dtype, grad_clip=1.0, aux_weight=1.0):
    """按本次更新的有效目标 token 归一化 CE，正确处理短回答和最后不足的累积组。"""
    token_counts = [int((b["labels"][:, 1:] != -100).sum()) for b in cpu_batches]
    total_tokens = sum(token_counts)
    sample_count = sum(b["input_ids"].shape[0] for b in cpu_batches)
    if total_tokens == 0 or any(n == 0 for n in token_counts):
        raise ValueError("Every microbatch must have supervised caption tokens")
    optimizer.zero_grad(set_to_none=True)
    ce_total = aux_total = 0.0
    expert_counts = None
    for cpu_batch, count in zip(cpu_batches, token_counts):
        batch = to_device(cpu_batch, device)
        sample_weight = batch["input_ids"].shape[0] / sample_count
        with autocast(device, dtype):
            result = model(**batch)
            # miniLLM 的 result.loss 已含辅助项，这里明确拆开，避免重复计算。
            loss = result.lm_loss * (count / total_tokens) + result.router_aux_loss * (aux_weight * sample_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; latest committed checkpoint remains intact")
        scaler.scale(loss).backward()
        ce_total += result.lm_loss.item() * count / total_tokens
        aux_total += result.router_aux_loss.item() * sample_weight
        if result.expert_counts is not None:
            current = result.expert_counts.detach().float()
            expert_counts = current if expert_counts is None else expert_counts + current
        del result, loss, batch
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not any(p.grad is not None for p in parameters):
        raise RuntimeError("Trainable parameters received no gradients")
    group_norms = {}
    for name, module in (("projector", model.projector), ("llm", model.llm)):
        squares = [p.grad.detach().float().square().sum() for p in module.parameters()
                   if p.requires_grad and p.grad is not None]
        if squares:
            group_norms[f"train/{name}_grad_norm"] = float(torch.stack(squares).sum().sqrt())
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    metrics = {"train/lm_loss": ce_total, "train/router_aux_loss": aux_total,
               "train/loss": ce_total + aux_weight * aux_total,
               "train/grad_norm": float(grad_norm), "train/samples": sample_count,
               "train/supervised_tokens": total_tokens}
    metrics.update(group_norms)
    if expert_counts is not None and expert_counts.sum() > 0:
        for i, fraction in enumerate((expert_counts / expert_counts.sum()).tolist()):
            metrics[f"router/expert_{i}_fraction"] = fraction
    return metrics


def run(args):
    validate_args(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device, dtype = resolve_device(args.device, args.dtype)
    if not args.check_only and args.resume is None and ((args.save_dir / "latest.pt").exists() or
                                (args.output_dir / "run_config.json").exists()):
        raise FileExistsError("Run already exists: use --resume or choose new save/output directories")
    seed_everything(args.seed)
    model, tokenizer, processor, identity, resources = load_resources(args)
    if args.max_seq_len > model.llm.config.max_position_embeddings:
        raise ValueError("max-seq-len exceeds Base max_position_embeddings")
    source = load_parquet(args.data_path, args.cache_dir / "arrow")
    encoder = CaptionEncoder(tokenizer, model.config, args.max_seq_len)
    index = None
    if not args.check_only or args.prepare_index:
        signature = {"data": data_identity(args.data_path), "tokenizer": identity["tokenizer"],
                     "implementation": identity["implementation"]}
        index = prepare_index(source, encoder, args.index_path, signature, args.seed, args.val_ratio)
    if args.check_only:
        return check_resources(args, model, tokenizer, processor, identity, resources,
                               source, encoder, index, device, dtype)
    train_indices = list(index["train"])
    # 限量短跑时随机选子集，避免数据源按语言/来源排序产生偏差。
    if args.max_train_samples and args.max_train_samples < len(train_indices):
        order = torch.randperm(len(train_indices), generator=torch.Generator().manual_seed(args.seed)).tolist()
        train_indices = [train_indices[i] for i in order[:args.max_train_samples]]
    validation_indices = index["validation"]
    order = torch.randperm(len(validation_indices), generator=torch.Generator().manual_seed(args.seed + 1)).tolist()
    validation_indices = [validation_indices[i] for i in order[:args.eval_samples]]
    if not train_indices or not validation_indices:
        raise ValueError("Empty train/validation split. For tiny data, use more unique images or a larger --val-ratio")
    train_ds = VLMDataset(source, train_indices, encoder, processor)
    val_ds = VLMDataset(source, validation_indices, encoder, processor)
    batches_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps = math.ceil(batches_per_epoch / args.accumulation_steps) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    if args.stop_after_steps > total_steps:
        raise ValueError("stop-after-steps exceeds the schedule horizon")
    model.to(device)
    decay, no_decay = [], []
    for parameter in model.projector.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.learning_rate)
    scheduler = build_cosine_scheduler(optimizer, total_steps, args.warmup_ratio,
                                       args.min_learning_rate / args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    config_keys = ("epochs", "batch_size", "accumulation_steps", "max_steps", "seed", "max_seq_len",
                   "learning_rate", "min_learning_rate", "warmup_ratio", "weight_decay", "grad_clip",
                   "router_aux_weight", "attention_backend", "tracker", "tracker_mode")
    contract = {"identity": identity, "vlm_config": model.config.to_dict(), "data": signature,
                "index_spec": index["spec"], "train_indices": json_hash(train_indices),
                "validation_indices": json_hash(validation_indices), "dtype": str(dtype),
                "device_type": device.type, "total_steps": total_steps,
                "training": {key: getattr(args, key) for key in config_keys}}
    epoch = batch_cursor = global_step = 0
    best_loss = float("inf")
    checkpoint = None
    if args.resume:
        resume_path = args.save_dir / "latest.pt" if args.resume == "auto" else project_path(args.resume)
        # Checkpoints 含 Python/NumPy RNG，只读取本人生成且可信的本地文件。
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("contract") != contract:
            raise ValueError("Resume contract mismatch: model/data/tokenizer/config/schedule changed")
        model.projector.load_state_dict(checkpoint["projector"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        epoch, batch_cursor, global_step = (checkpoint[key] for key in ("epoch", "batch_cursor", "global_step"))
        best_loss = checkpoint["best_loss"]
        updates_per_epoch = math.ceil(batches_per_epoch / args.accumulation_steps)
        expected_epoch, update_in_epoch = divmod(global_step, updates_per_epoch)
        if not 0 <= global_step <= total_steps or epoch != expected_epoch or batch_cursor != update_in_epoch * args.accumulation_steps:
            raise ValueError("Invalid checkpoint progress: cursor is not an optimizer-update boundary")
    context = DistributedContext(device, 0, 0, 1, False)
    tracker = init_experiment_tracker(
        args.tracker, context, args.tracker_project, args.tracker_run_name, None, None, [],
        args.tracker_mode, args.output_dir / "tracker", vars(args),
        run_id=(checkpoint or {}).get("tracker", {}).get("run_id"),
        strict_resume=bool(checkpoint and checkpoint.get("tracker", {}).get("run_id")),
    )
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {**model_report(model), "device": str(device), "dtype": str(dtype),
              "train_samples": len(train_ds), "validation_samples": len(val_ds),
              "total_steps": total_steps, "data_stats": index["stats"]}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    atomic_json({"args": {k: str(v) if isinstance(v, Path) else [str(p) for p in v] if isinstance(v, list) else v
                           for k, v in vars(args).items()}, "report": report, "contract": contract},
                args.output_dir / "run_config.json")
    if checkpoint:
        _restore_rng_state(checkpoint["rng_state"])
        print(f"Resumed optimizer step {global_step}, epoch {epoch}, next batch {batch_cursor}", flush=True)
    metrics_path = args.output_dir / "metrics.jsonl"

    def log(metrics):
        payload = dict(metrics, global_step=global_step)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        tracker.log(metrics, global_step)

    def save(name="latest.pt"):
        atomic_torch_save({"format_version": 1, "contract": contract,
            "projector": model.projector.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "batch_cursor": batch_cursor, "global_step": global_step,
            "best_loss": best_loss, "rng_state": _rng_state(), "tracker": tracker.state_dict()},
            args.save_dir / name)

    try:
        if not checkpoint:
            baseline = evaluate(model, make_loader(val_ds, args), device, dtype, paired=True)
            log(baseline)
            # step 0 是随机 Projector 基线，不参与最佳训练权重选择。
            save()
        if global_step >= total_steps:
            print("Requested training horizon already completed.", flush=True)
        stop = global_step >= total_steps or (args.stop_after_steps and global_step >= args.stop_after_steps)
        while epoch < args.epochs and not stop:
            loader = make_loader(train_ds, args, epoch, batch_cursor, training=True)
            iterator = iter(loader)
            model.train()
            while True:
                started = time.monotonic()
                group = list(islice(iterator, args.accumulation_steps))
                if not group:
                    break
                lr = optimizer.param_groups[0]["lr"]
                metrics = train_update(model, group, optimizer, scaler, device, dtype, args.grad_clip, args.router_aux_weight)
                scheduler.step()
                global_step += 1
                batch_cursor += len(group)
                epoch_finished = batch_cursor >= batches_per_epoch
                if epoch_finished:
                    epoch += 1
                    batch_cursor = 0
                stop = global_step >= total_steps or (args.stop_after_steps and global_step >= args.stop_after_steps)
                metrics.update({"train/learning_rate": lr, "train/epoch": epoch,
                                "train/samples_per_second": metrics["train/samples"] / max(1e-9, time.monotonic() - started)})
                if device.type == "cuda":
                    metrics["train/peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
                if global_step % args.log_interval == 0 or stop or epoch_finished:
                    log(metrics)
                if global_step % args.eval_interval == 0 or epoch_finished or stop:
                    validation_metrics = evaluate(model, make_loader(val_ds, args), device, dtype, paired=True)
                    log(validation_metrics)
                    if validation_metrics["val/lm_loss"] < best_loss:
                        best_loss = validation_metrics["val/lm_loss"]
                        export_adapter(args.output_dir / "best_adapter.pt", model, identity, resources, global_step, validation_metrics)
                        save("best.pt")
                    save()
                if global_step % args.save_interval == 0 or stop or epoch_finished:
                    save()
                    export_adapter(args.output_dir / "last_adapter.pt", model, identity, resources, global_step)
                if stop or epoch_finished:
                    break
            del iterator, loader
        export_adapter(args.output_dir / "last_adapter.pt", model, identity, resources, global_step)
        tracker.finish()
        print(f"{'Completed' if global_step >= total_steps else 'Paused'} at step {global_step}; artifacts: {args.output_dir}", flush=True)
        return report
    except BaseException as exc:
        # 不把可能只完成部分反向/更新的状态保存为可恢复断点。
        print(f"Stopped: {exc}. Resume from the last committed latest.pt.", file=sys.stderr)
        tracker.finish(state="crashed", error=str(exc))
        raise


if __name__ == "__main__":
    run(parse_args())

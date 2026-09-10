#!/usr/bin/env python3
"""独立 miniLLM-vlm SFT：继承 Pretrain，训练 Projector 与指定语言层。"""
from __future__ import annotations

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

from minillm_vlm.dataset.vlm_dataset import (SFTEncoder, SFTCollator, SFTDataset,
    atomic_json, load_parquet, prepare_sft_index)
from minillm_vlm.trainer.train_pretrain_vlm import build_parser, validate_args, train_update
from minillm_vlm.trainer.trainer_utils import (VLM_ROOT, autocast, data_identity,
    export_sft, json_hash, load_sft_resources, model_report, project_path, resolve_device,
    to_device, evaluate_sft, make_sft_loader, checked_state_load, directory_hash,
    DistributedContext, _restore_rng_state, _rng_state, atomic_torch_save,
    build_cosine_scheduler, init_experiment_tracker, seed_everything)


def parse_args(argv=None):
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(data_path=[VLM_ROOT / "dataset/sft_i2t.parquet"],
        index_path=VLM_ROOT / "cache/sft_index.json", max_seq_len=768, batch_size=4,
        accumulation_steps=16, learning_rate=5e-6, min_learning_rate=5e-7,
        save_dir=VLM_ROOT / "checkpoints/sft", output_dir=VLM_ROOT / "out/sft",
        report_path=VLM_ROOT / "out/sft_preflight.json", tracker_project="miniLLM-VLM-SFT")
    parser.add_argument("--from-pretrain", type=project_path, required=True,
                        help="Formal Pretrain best_adapter.pt (also retained as provenance on resume).")
    parser.add_argument("--freeze-llm", type=int, choices=[0, 1, 2], default=1,
                        help="0: full LLM; 1: first/last decoder blocks; 2: projector only. SigLIP always frozen.")
    parser.add_argument("--scan-samples", type=int, default=0,
                        help="Fixed random subset for a smoke index; 0 scans all rows. Use a separate index-path.")
    parser.add_argument("--generation-samples", type=int, default=4)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    return parser.parse_args(argv)


def check_sft_resources(args, model, tokenizer, processor, provenance,
                        source, encoder, index, device, dtype):
    indices = index["train"][:args.samples] if index is not None else list(range(min(args.samples, len(source))))
    if not indices:
        raise ValueError("No SFT samples available for preflight")
    dataset = SFTDataset(source, indices, encoder, processor)
    samples = [dataset[i] for i in range(len(dataset))]
    cpu_batch = SFTCollator(tokenizer.pad_token_id)(samples)
    model.to(device).train()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    last = len(model.llm.model.layers) - 1
    allowed = (("projector.", "llm.") if args.freeze_llm == 0 else
               ("projector.", "llm.model.layers.0.", f"llm.model.layers.{last}.") if args.freeze_llm == 1 else
               ("projector.",))
    if not trainable or any(not name.startswith(allowed) for name in trainable):
        raise RuntimeError("SFT freeze policy violation")
    with autocast(device, dtype):
        output = model(**to_device(cpu_batch, device))
    if not torch.isfinite(output.loss):
        raise FloatingPointError("Non-finite SFT preflight loss")
    if not output.loss.requires_grad:
        raise ValueError("No trainable path in these samples (projector-only policy requires images)")
    output.loss.backward()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and parameter.grad is not None:
            raise RuntimeError(f"Frozen parameter received gradient: {name}")
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient: {name}")
    norms = {}
    for name, module in (("projector", model.projector), ("llm", model.llm)):
        norm = sum(p.grad.float().square().sum().item() for p in module.parameters() if p.grad is not None) ** .5
        norms[name] = norm
        required = bool(cpu_batch["has_image"].any()) if name == "projector" else args.freeze_llm != 2
        if required and norm <= 0:
            raise RuntimeError(f"Missing/nonpositive gradient in {name}")
    report = {**model_report(model), "stage": "sft", "freeze_check": "passed",
        "freeze_llm": args.freeze_llm, "trainable_names": trainable, "pretrain": provenance,
        "lm_loss": output.lm_loss.item(), "grad_norms": norms,
        "visual_samples": int(cpu_batch["has_image"].sum()),
        "text_samples": int((~cpu_batch["has_image"]).sum()),
        "batch_shapes": {key: list(value.shape) for key, value in cpu_batch.items()},
        "data_stats": index["stats"] if index is not None else None}
    if device.type == "cuda":
        report["peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
    atomic_json(report, args.report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"SFT preflight passed. Report: {args.report_path}", flush=True)
    model.zero_grad(set_to_none=True)
    return report


@torch.no_grad()
def generate_examples(model, dataset, args, device, dtype, step):
    records = []
    for index in range(min(args.generation_samples, len(dataset))):
        sample = dataset[index]
        length = sample["prompt_length"]
        ids = sample["input_ids"][:length].unsqueeze(0).to(device)
        budget = min(args.generation_max_new_tokens, model.llm.config.max_position_embeddings - length)
        if budget < 1:
            continue
        pixels = sample["pixel_values"].unsqueeze(0).to(device) if sample["has_image"] else None
        with autocast(device, dtype):
            result = model.generate_caption(ids, pixels, budget)
        answer_labels = sample["labels"][length:]
        end = next((i for i, token in enumerate(answer_labels) if token == -100), len(answer_labels))
        records.append({"global_step": step, "row_index": sample["row_index"],
            "prompt": dataset.encoder.tokenizer.decode(ids[0], skip_special_tokens=False),
            "reference": dataset.encoder.tokenizer.decode(answer_labels[:end], skip_special_tokens=True),
            "generated": dataset.encoder.tokenizer.decode(result[0], skip_special_tokens=True)})
    if records:
        with (args.output_dir / "generations.jsonl").open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args):
    validate_args(args)
    if args.scan_samples < 0 or args.generation_samples < 0 or args.generation_max_new_tokens < 1:
        raise ValueError("Invalid scan/generation settings")
    if not args.from_pretrain.is_file():
        raise FileNotFoundError(f"Pretrain adapter not found: {args.from_pretrain}")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device, dtype = resolve_device(args.device, args.dtype)
    if not args.check_only and args.resume is None and ((args.save_dir / "latest.pt").exists() or
                                (args.output_dir / "run_config.json").exists()):
        raise FileExistsError("Run already exists: use --resume or choose new save/output directories")
    seed_everything(args.seed)
    model, tokenizer, processor, identity, resources, provenance = load_sft_resources(args)
    if args.max_seq_len > model.llm.config.max_position_embeddings:
        raise ValueError("max-seq-len exceeds Base max_position_embeddings")
    source = load_parquet(args.data_path, args.cache_dir / "arrow")
    encoder = SFTEncoder(tokenizer, model.config, args.max_seq_len)
    index = None
    if not args.check_only or args.prepare_index:
        signature = {"data": data_identity(args.data_path), "tokenizer": identity["tokenizer"],
                     "implementation": identity["implementation"]}
        index = prepare_sft_index(source, encoder, args.index_path, signature, args.seed, args.val_ratio, args.scan_samples)
    if args.check_only:
        return check_sft_resources(args, model, tokenizer, processor, provenance,
                                   source, encoder, index, device, dtype)
    if args.freeze_llm == 2 and index["stats"].get("text", 0):
        raise ValueError("--freeze-llm 2 requires image-only data; use 1 or 0 for mixed text SFT")
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
    train_ds = SFTDataset(source, train_indices, encoder, processor)
    val_ds = SFTDataset(source, validation_indices, encoder, processor)
    batches_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps = math.ceil(batches_per_epoch / args.accumulation_steps) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    if args.stop_after_steps > total_steps:
        raise ValueError("stop-after-steps exceeds the schedule horizon")
    model.to(device)
    decay, no_decay = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.learning_rate)
    scheduler = build_cosine_scheduler(optimizer, total_steps, args.warmup_ratio,
                                       args.min_learning_rate / args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    config_keys = ("epochs", "batch_size", "accumulation_steps", "max_steps", "seed", "max_seq_len",
                   "learning_rate", "min_learning_rate", "warmup_ratio", "weight_decay", "grad_clip",
                   "router_aux_weight", "attention_backend", "tracker", "tracker_mode", "freeze_llm",
                   "tracker_project", "eval_interval", "save_interval", "num_workers",
                   "generation_samples", "generation_max_new_tokens")
    contract = {"stage": "sft", "pretrain": provenance, "identity": identity, "vlm_config": model.config.to_dict(), "data": signature,
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
        checked_state_load(model.llm, checkpoint["llm"], "SFT resume LLM")
        checked_state_load(model.projector, checkpoint["projector"], "SFT resume projector")
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
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    resources = dict(resources, exported_tokenizer_identity=directory_hash(args.output_dir / "tokenizer"))
    report = {**model_report(model), "stage": "sft", "freeze_llm": args.freeze_llm,
              "trainable_names": [name for name, p in model.named_parameters() if p.requires_grad], "device": str(device), "dtype": str(dtype),
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
            "llm": model.llm.state_dict(), "projector": model.projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "batch_cursor": batch_cursor, "global_step": global_step,
            "best_loss": best_loss, "rng_state": _rng_state(), "tracker": tracker.state_dict()},
            args.save_dir / name)

    try:
        if not checkpoint:
            baseline = evaluate_sft(model, make_sft_loader(val_ds, args, metadata=True), device, dtype)
            log(baseline)
            generate_examples(model, val_ds, args, device, dtype, global_step)
            # step 0 是继承 Pretrain 的基线；best_sft 仅选择完成更新后的模型。
            save()
        if global_step >= total_steps:
            print("Requested training horizon already completed.", flush=True)
        stop = global_step >= total_steps or (args.stop_after_steps and global_step >= args.stop_after_steps)
        while epoch < args.epochs and not stop:
            loader = make_sft_loader(train_ds, args, epoch, batch_cursor, training=True)
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
                    validation_metrics = evaluate_sft(model, make_sft_loader(val_ds, args, metadata=True), device, dtype)
                    log(validation_metrics)
                    generate_examples(model, val_ds, args, device, dtype, global_step)
                    if validation_metrics["val/selection_loss"] < best_loss:
                        best_loss = validation_metrics["val/selection_loss"]
                        export_sft(args.output_dir / "best_sft.pt", model, identity, resources, provenance, global_step, validation_metrics)
                        save("best.pt")
                    save()
                if global_step % args.save_interval == 0 or stop or epoch_finished:
                    save()
                    export_sft(args.output_dir / "last_sft.pt", model, identity, resources, provenance, global_step)
                if stop or epoch_finished:
                    break
            del iterator, loader
        export_sft(args.output_dir / "last_sft.pt", model, identity, resources, provenance, global_step)
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

#!/usr/bin/env python3
"""使用 ``{"text": ...}`` JSONL，从随机权重开始预训练 miniLLM。

学习时可以把本文件理解成一条完整的数据流水线：

1. 读取命令行参数，并初始化单卡或 DDP 多卡环境；
2. 用已经训练好的 Tokenizer 把文本转换成 token id；
3. 创建随机初始化的 Causal Language Model；
4. 执行前向传播、反向传播、梯度累积和参数更新；
5. 定期评估、记录 SwanLab/W&B 指标、保存断点；
6. 训练结束后导出 Hugging Face 格式的模型和 Tokenizer。

这个阶段训练的是模型参数，不再修改 Tokenizer 的词表。

Single-GPU example (run from the project root):

    python trainer/train_pretrain.py \
        --data-path dataset/pretrain/pretrain.jsonl \
        --tokenizer-path model/tokenizer

MoE example (4 Experts, Top-1, about 200M total / 65M active parameters):

    python trainer/train_pretrain.py \
        --data-path dataset/pretrain/pretrain.jsonl \
        --use-moe \
        --save-dir checkpoints/pretrain_moe \
        --output-dir out/pretrain_moe

Multi-GPU example:

    torchrun --nproc_per_node 2 trainer/train_pretrain.py \
        --data-path dataset/pretrain/pretrain.jsonl
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lm_dataset import PretrainDataset, validate_tokenizer  # noqa: E402
from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM  # noqa: E402
from trainer.trainer_utils import (  # noqa: E402
    DistributedContext,
    ExperimentTracker,
    build_cosine_scheduler,
    cleanup_distributed,
    distributed_sum,
    ensure_checkpoint_tokenizer_fingerprint,
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


def parse_args() -> argparse.Namespace:
    """定义预训练参数；所有参数都有默认值，可只覆盖需要调整的部分。"""

    parser = argparse.ArgumentParser(description="Pretrain the miniLLM base model.")

    # 输入输出路径：断点用于继续训练，output-dir 用于推理或后续微调。
    parser.add_argument(
        "--data-path",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "dataset" / "pretrain" / "pretrain.jsonl"],
        help="One or more JSONL files containing a string 'text' field.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=PROJECT_ROOT / "model" / "tokenizer",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "pretrain",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "out" / "pretrain",
    )

    # 训练策略：global_step 表示“优化器更新次数”，不是 DataLoader 批次数。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-steps", type=int, default=0, help="Optimizer steps; 0 uses epochs."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.001)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int, default=0)

    # 模型结构。vocab-size 必须与 Tokenizer 的实际词表大小一致。
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=0)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--num-key-value-heads", type=int, default=4)
    parser.add_argument("--max-position-embeddings", type=int, default=32768)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--hidden-dropout", type=float, default=0.0)
    # 正式方案默认使用 4 Expert / Top-1 MoE；--no-use-moe 可运行 Dense 对照实验。
    parser.add_argument(
        "--use-moe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sparse MoE layers (default: enabled; use --no-use-moe for Dense).",
    )
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-experts-per-tok", type=int, default=1)
    parser.add_argument("--moe-intermediate-size", type=int, default=0)
    parser.add_argument("--router-aux-loss-coef", type=float, default=5e-4)
    parser.add_argument(
        "--norm-topk-prob",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize selected Top-K router probabilities (default: enabled).",
    )

    # 运行环境：精度、DataLoader 进程、断点恢复及显存优化开关。
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Resume from an explicit checkpoint path, or use --resume to load "
            "save-dir/latest.pt when it exists and otherwise start fresh."
        ),
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    # 实验监控：none、SwanLab 和 W&B 共用这一组参数。
    parser.add_argument(
        "--tracker",
        choices=["none", "swanlab", "wandb"],
        default="none",
        help="Optional experiment monitoring backend.",
    )
    parser.add_argument("--tracker-project", type=str, default="miniLLM-Pretrain")
    parser.add_argument("--tracker-run-name", type=str, default=None)
    parser.add_argument(
        "--tracker-entity",
        type=str,
        default=None,
        help="SwanLab workspace or W&B entity/team.",
    )
    parser.add_argument("--tracker-group", type=str, default=None)
    parser.add_argument("--tracker-tags", nargs="*", default=[])
    parser.add_argument(
        "--tracker-mode", choices=["online", "offline"], default="online"
    )
    parser.add_argument(
        "--tracker-log-dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "pretrain",
    )
    parser.add_argument(
        "--tracker-run-id",
        type=str,
        default=None,
        help="Explicit run ID; normally restored automatically from checkpoint.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    """把相对路径统一解释为相对于项目根目录，而非当前终端目录。"""

    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    """尽早检查参数，避免运行数小时后才因明显配置错误退出。"""

    positive_fields = (
        "epochs",
        "batch_size",
        "accumulation_steps",
        "learning_rate",
        "max_seq_len",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "num_experts",
        "num_experts_per_tok",
        "log_interval",
        "eval_interval",
        "save_interval",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if not 0.0 < args.min_learning_rate <= args.learning_rate:
        raise ValueError("--min-learning-rate must be > 0 and <= learning-rate")
    if args.max_steps < 0 or args.max_train_samples < 0:
        raise ValueError("--max-steps and --max-train-samples cannot be negative")
    if args.num_experts_per_tok > args.num_experts:
        raise ValueError("--num-experts-per-tok cannot exceed --num-experts")
    if args.use_moe and args.num_experts < 2:
        raise ValueError("--use-moe requires at least 2 experts")
    if args.moe_intermediate_size < 0:
        raise ValueError("--moe-intermediate-size cannot be negative")
    if args.router_aux_loss_coef < 0:
        raise ValueError("--router-aux-loss-coef cannot be negative")


def resolve_amp(
    requested: str, context: DistributedContext
) -> tuple[torch.dtype, bool]:
    """决定自动混合精度类型，以及是否需要 GradScaler。

    bfloat16 的指数范围较大，通常不需要 loss scaling；float16 更容易发生
    梯度下溢，因此返回 ``needs_scaler=True``。CPU/MPS 在这里回退到 float32。
    """

    if requested == "float32" or context.device.type != "cuda":
        if requested != "float32" and context.is_main:
            print(f"Warning: {requested} autocast is disabled on {context.device.type}.")
        return torch.float32, False
    if requested == "bfloat16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
        if context.is_main:
            print("Warning: this GPU does not support bfloat16; falling back to float16.")
        return torch.float16, True
    return torch.float16, True


def autocast_context(device: torch.device, dtype: torch.dtype):
    """返回 AMP 上下文；非 CUDA 或 float32 时返回一个空上下文。"""

    if device.type == "cuda" and dtype != torch.float32:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    context: DistributedContext,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    """计算验证集的 token 加权 loss 和困惑度（perplexity）。

    不同 batch 的有效 token 数可能不同，所以不能简单平均 batch loss。
    多卡时还会把各 rank 的 loss 总和与 token 数汇总后再计算。
    """

    model.eval()
    # totals[0]：loss * 有效 token 数；totals[1]：有效 token 总数。
    totals = torch.zeros(2, device=context.device, dtype=torch.float64)
    for batch in loader:
        batch = {key: value.to(context.device, non_blocking=True) for key, value in batch.items()}
        with autocast_context(context.device, amp_dtype):
            output = model(**batch)
        valid_tokens = (batch["labels"][:, 1:] != -100).sum()
        # Perplexity 只衡量语言建模能力，不能混入 Router 负载均衡损失。
        lm_loss = output.lm_loss if output.lm_loss is not None else output.loss
        totals[0] += lm_loss.detach().double() * valid_tokens
        totals[1] += valid_tokens
    distributed_sum(totals, context)
    loss = (totals[0] / totals[1].clamp_min(1)).item()
    # 截断指数输入只为避免训练早期 loss 很大时 math.exp 溢出。
    perplexity = math.exp(min(loss, 20.0))
    model.train()
    return loss, perplexity


def make_dataloaders(
    args: argparse.Namespace,
    tokenizer: Any,
    context: DistributedContext,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    """创建训练/验证 DataLoader，并在多卡模式下为每张卡切分数据。

    ``PretrainDataset`` 负责读取 JSONL、确定性划分训练/验证样本、编码文本，
    并生成 input_ids、attention_mask 和 labels。
    """

    train_dataset: Any = PretrainDataset(
        args.data_path,
        tokenizer,
        max_seq_len=args.max_seq_len,
        split="train",
        val_ratio=args.val_ratio,
        expected_vocab_size=args.vocab_size,
    )
    validation_dataset: Any = PretrainDataset(
        args.data_path,
        tokenizer,
        max_seq_len=args.max_seq_len,
        split="validation",
        val_ratio=args.val_ratio,
        expected_vocab_size=args.vocab_size,
    )
    # 这两个上限便于先用小数据验证代码和显存配置。
    if args.max_train_samples:
        train_dataset = Subset(
            train_dataset, range(min(args.max_train_samples, len(train_dataset)))
        )
    if args.eval_samples:
        validation_dataset = Subset(
            validation_dataset, range(min(args.eval_samples, len(validation_dataset)))
        )

    # DDP 中每个进程只读取自己的数据分片，避免所有 GPU 重复训练同一批数据。
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
        )
        if context.distributed
        else None
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
    # CUDA 下 pin_memory 配合 non_blocking=True 可加速 CPU -> GPU 拷贝。
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": context.device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        # NCCL/DDP 与 DataLoader 的 Linux 默认 fork 方式组合可能死锁。
        # spawn 不会继承父进程已初始化的 CUDA/NCCL 状态，单卡和多卡
        # 使用同一套行为，也便于在不同平台上复现。
        loader_options["multiprocessing_context"] = "spawn"
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        # 丢弃最后一个不完整训练 batch，使每一步形状和有效 batch size 稳定。
        drop_last=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        sampler=validation_sampler,
        drop_last=False,
        **loader_options,
    )
    if len(train_loader) == 0:
        raise ValueError("Training DataLoader is empty; reduce --batch-size")
    if len(validation_loader) == 0:
        raise ValueError("Validation DataLoader is empty")
    return train_loader, validation_loader, train_sampler


def print_setup(
    args: argparse.Namespace,
    model: MiniLLMForCausalLM,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    context: DistributedContext,
    total_steps: int,
    amp_dtype: torch.dtype,
) -> None:
    """只由主进程打印最终生效的训练配置摘要。"""

    if not context.is_main:
        return
    raw_model = unwrap_model(model)
    parameter_count = sum(parameter.numel() for parameter in raw_model.parameters())
    active_parameter_count = parameter_count
    if raw_model.config.use_moe:
        # Expert 0 在所有层的参数量代表“一组 Expert 参数”；Top-K 只激活 K 组。
        one_expert_group = sum(
            parameter.numel()
            for name, parameter in raw_model.named_parameters()
            if ".mlp.experts.0." in name
        )
        inactive_groups = (
            raw_model.config.num_experts - raw_model.config.num_experts_per_tok
        )
        active_parameter_count -= one_expert_group * inactive_groups
    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Data files    : {len(args.data_path)}")
    for path in args.data_path:
        print(f"  - {path}")
    print(f"Tokenizer     : {args.tokenizer_path}")
    print(f"Tokenizer SHA : {args.tokenizer_fingerprint}")
    print(f"Device        : {context.device} (world_size={context.world_size})")
    print(f"Precision     : {str(amp_dtype).removeprefix('torch.')}")
    architecture = (
        f"MoE ({args.num_experts} Experts, Top-{args.num_experts_per_tok})"
        if args.use_moe
        else "Dense"
    )
    print(f"Architecture  : {architecture}")
    print(f"Parameters    : {parameter_count:,} total")
    if args.use_moe:
        print(f"Active params : {active_parameter_count:,} per token")
    print(f"Sequence      : {args.max_seq_len}")
    print(f"Train batches : {len(train_loader):,} per rank")
    print(f"Valid batches : {len(validation_loader):,} per rank")
    print(f"Update steps  : {total_steps:,}")


def main() -> None:
    """组织整个预训练生命周期。具体通用能力位于 trainer_utils.py。"""

    # Hugging Face Tokenizer 自带线程池；DataLoader 多进程下关闭它可避免警告/死锁。
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # 1. 参数与路径规范化。
    args = parse_args()
    validate_args(args)
    args.data_path = [project_path(path) for path in args.data_path]
    args.tokenizer_path = project_path(args.tokenizer_path)
    args.save_dir = project_path(args.save_dir)
    args.output_dir = project_path(args.output_dir)
    args.tracker_log_dir = project_path(args.tracker_log_dir)

    # 2. 初始化设备/DDP。非主进程不会创建监控 run 或写 checkpoint。
    context = setup_distributed(args.device)
    tracker = ExperimentTracker()
    tracker_finish_state = "crashed"
    tracker_finish_error: str | None = "Training stopped before completion."
    try:
        # 不同 rank 使用 seed + rank，既可复现，又避免多卡生成完全相同的随机序列。
        seed_everything(args.seed, context.rank)
        if context.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        names = {path.name for path in args.data_path}
        mini_names = {"pretrain_mini.jsonl"}
        full_names = {"pretrain.jsonl"}
        if context.is_main and names & full_names and names & mini_names:
            print(
                "Warning: both MiniMind mini and full datasets were provided. "
                "Their overlap is not documented, so repeated samples may be overweighted."
            )

        # 3. 加载本地 Tokenizer。预训练只使用它编码，不会继续训练其 merges/vocab。
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path,
            local_files_only=True,
            use_fast=True,
        )
        validate_tokenizer(tokenizer, args.vocab_size)
        args.tokenizer_fingerprint = tokenizer_fingerprint(args.tokenizer_path)
        train_loader, validation_loader, train_sampler = make_dataloaders(
            args, tokenizer, context
        )

        # 4. 根据参数构建模型配置，并从随机权重初始化模型。
        config = MiniLLMConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size or None,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            max_position_embeddings=args.max_position_embeddings,
            rope_theta=args.rope_theta,
            attention_dropout=args.attention_dropout,
            hidden_dropout=args.hidden_dropout,
            use_moe=args.use_moe,
            num_experts=args.num_experts,
            num_experts_per_tok=args.num_experts_per_tok,
            moe_intermediate_size=args.moe_intermediate_size or None,
            norm_topk_prob=args.norm_topk_prob,
            router_aux_loss_coef=args.router_aux_loss_coef,
            tokenizer_fingerprint=args.tokenizer_fingerprint,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            tie_word_embeddings=True,
            use_cache=False,
        )
        model: torch.nn.Module = MiniLLMForCausalLM(config).to(context.device)
        # 梯度检查点用额外计算换显存，适合显存有限或序列较长的训练。
        if args.gradient_checkpointing:
            model.model.gradient_checkpointing = True
        if args.compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("--compile requires PyTorch 2.x")
            if args.use_moe and context.is_main:
                print(
                    "Warning: dynamic Top-K routing can cause torch.compile graph breaks."
                )
            model = torch.compile(model)
        # DDP 为每张 GPU 启动一个进程，并在反向传播时同步各卡梯度。
        if context.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
            )

        # 5. 优化器与学习率调度器。
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
        # 多个 micro-batch 累积后才 optimizer.step() 一次，因此要折算更新步数。
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

        # 6. 如指定 --resume，同时恢复权重、优化器、调度器、随机数和训练位置。
        start_epoch = 0
        start_batch = 0
        global_step = 0
        checkpoint: dict[str, Any] = {}
        resume_path = resolve_resume_path(args.save_dir, args.resume)
        if resume_path is not None:
            checkpoint = load_checkpoint(
                resume_path,
                model,
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
                print(f"Resumed       : {resume_path} (step={global_step:,})")
        elif args.resume == "auto" and context.is_main:
            print(
                "Resume       : no latest.pt found; starting a new training run."
            )

        # 7. 恢复同一个 SwanLab/W&B run，避免断点续训产生两条实验记录。
        checkpoint_tracker = checkpoint.get("tracker", {})
        restored_tracker_id = (
            checkpoint_tracker.get("run_id")
            if checkpoint_tracker.get("backend") == args.tracker
            else None
        )
        use_checkpoint_tracker = bool(restored_tracker_id and not args.tracker_run_id)
        tracker_project = (
            (checkpoint_tracker.get("project") or args.tracker_project)
            if use_checkpoint_tracker
            else args.tracker_project
        )
        tracker_entity = (
            checkpoint_tracker.get("entity")
            if use_checkpoint_tracker and "entity" in checkpoint_tracker
            else args.tracker_entity
        )
        tracker_mode = (
            (checkpoint_tracker.get("mode") or args.tracker_mode)
            if use_checkpoint_tracker
            else args.tracker_mode
        )
        tracker_run_name = (
            checkpoint_tracker.get("run_name")
            if use_checkpoint_tracker
            else args.tracker_run_name
        )
        if not tracker_run_name:
            architecture_name = (
                f"MoE{args.num_experts}xTop{args.num_experts_per_tok}"
                if args.use_moe
                else "Dense"
            )
            tracker_run_name = (
                f"miniLLM-{architecture_name}-H{args.hidden_size}-"
                f"L{args.num_hidden_layers}-"
                f"Seq{args.max_seq_len}-BS{args.batch_size}x{args.accumulation_steps}-"
                f"LR{args.learning_rate:g}"
            )
        tracker_config = vars(args).copy()
        tracker_config["effective_tracker_run_name"] = tracker_run_name
        tracker = init_experiment_tracker(
            backend=args.tracker,
            context=context,
            project=tracker_project,
            run_name=tracker_run_name,
            entity=tracker_entity,
            group=args.tracker_group,
            tags=args.tracker_tags,
            mode=tracker_mode,
            log_dir=args.tracker_log_dir,
            config=tracker_config,
            run_id=args.tracker_run_id or restored_tracker_id,
            strict_resume=use_checkpoint_tracker,
        )
        if tracker.enabled and context.is_main:
            print(f"Tracker       : {args.tracker} (run_id={tracker.run_id})")

        print_setup(
            args,
            model,
            train_loader,
            validation_loader,
            context,
            total_steps,
            amp_dtype,
        )
        if global_step >= total_steps:
            raise ValueError(
                f"Checkpoint step {global_step} already reached total_steps={total_steps}"
            )

        # 8. 正式进入训练循环。
        model.train()
        optimizer.zero_grad(set_to_none=True)
        interval_loss = 0.0
        interval_lm_loss = 0.0
        interval_router_aux_loss = 0.0
        interval_micro_batches = 0
        interval_updates = 0
        interval_expert_counts = torch.zeros(
            args.num_experts, device=context.device, dtype=torch.float64
        )
        interval_router_prob_sums = torch.zeros_like(interval_expert_counts)
        interval_router_entropy_sum = torch.zeros(
            (), device=context.device, dtype=torch.float64
        )
        interval_routed_token_count = torch.zeros_like(
            interval_router_entropy_sum
        )
        interval_start = time.perf_counter()
        stop_training = False
        latest_grad_norm = 0.0

        for epoch in range(start_epoch, training_epochs):
            # DDP sampler 每轮使用不同 shuffle 顺序，同时保证所有 rank 划分一致。
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            skip_before = start_batch if epoch == start_epoch else 0

            for batch_index, batch in enumerate(train_loader):
                # 断点可能位于 epoch 中间：已完成的 batch 直接跳过。
                if batch_index < skip_before:
                    continue
                # 一个 DataLoader batch 是一个 micro-batch；累积若干个才更新一次参数。
                should_update = (
                    (batch_index + 1) % args.accumulation_steps == 0
                    or batch_index + 1 == len(train_loader)
                )
                # DDP 的 no_sync() 暂缓梯度通信，只在本次累积的最后一个
                # micro-batch 同步梯度，可明显减少多卡通信开销。
                sync_context = (
                    nullcontext()
                    if should_update or not context.distributed
                    else model.no_sync()
                )
                batch = {
                    key: value.to(context.device, non_blocking=True)
                    for key, value in batch.items()
                }
                with sync_context:
                    with autocast_context(context.device, amp_dtype):
                        output = model(**batch)
                        # 除以累积步数，使累积后的梯度近似一个大 batch 的平均梯度。
                        scaled_loss = output.loss / args.accumulation_steps
                    if needs_scaler:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                interval_loss += output.loss.detach().float().item()
                interval_lm_loss += output.lm_loss.detach().float().item()
                if args.use_moe:
                    interval_router_aux_loss += (
                        output.router_aux_loss.detach().float().item()
                    )
                    interval_expert_counts += output.expert_counts.double()
                    interval_router_prob_sums += output.router_prob_sums.double()
                    interval_router_entropy_sum += output.router_entropy_sum.double()
                    interval_routed_token_count += output.routed_token_count.double()
                interval_micro_batches += 1

                if not should_update:
                    continue
                # 梯度裁剪必须在 fp16 梯度反缩放后执行，否则范数没有实际意义。
                if needs_scaler:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                latest_grad_norm = float(grad_norm.detach().float().item())
                if needs_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                # 每次 optimizer 更新后，学习率调度器也前进一步。
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # global_step 只统计 optimizer.step()，不是 micro-batch 数。
                global_step += 1
                interval_updates += 1
                # 日志中的 loss 是本区间所有实际 micro-batch 的平均值。
                if global_step % args.log_interval == 0:
                    # loss 与 Router 统计先跨 rank 求和，再由主进程上传。
                    loss_totals = torch.tensor(
                        [
                            interval_loss,
                            interval_lm_loss,
                            interval_router_aux_loss,
                            interval_micro_batches,
                        ],
                        device=context.device,
                        dtype=torch.float64,
                    )
                    distributed_sum(loss_totals, context)
                    moe_totals = None
                    if args.use_moe:
                        moe_totals = torch.cat(
                            [
                                interval_expert_counts,
                                interval_router_prob_sums,
                                interval_router_entropy_sum.reshape(1),
                                interval_routed_token_count.reshape(1),
                            ]
                        )
                        distributed_sum(moe_totals, context)

                    if context.is_main:
                        elapsed = max(time.perf_counter() - interval_start, 1e-6)
                        global_micro_batches = max(1.0, loss_totals[3].item())
                        mean_loss = loss_totals[0].item() / global_micro_batches
                        mean_lm_loss = loss_totals[1].item() / global_micro_batches
                        mean_router_aux_loss = (
                            loss_totals[2].item() / global_micro_batches
                        )
                        tokens_per_second = (
                            args.batch_size
                            * args.max_seq_len
                            * args.accumulation_steps
                            * interval_updates
                            * context.world_size
                            / elapsed
                        )
                        memory = (
                            torch.cuda.max_memory_allocated(context.device) / 1024**3
                            if context.device.type == "cuda"
                            else 0.0
                        )
                        moe_suffix = (
                            f" lm={mean_lm_loss:.4f}"
                            f" aux={mean_router_aux_loss:.6f}"
                            if args.use_moe
                            else ""
                        )
                        print(
                            f"step={global_step:,}/{total_steps:,} "
                            f"loss={mean_loss:.4f}{moe_suffix} "
                            f"lr={scheduler.get_last_lr()[0]:.3e} "
                            f"tokens/s={tokens_per_second:,.0f} "
                            f"memory={memory:.2f}GB"
                        )
                        metrics: dict[str, float | int] = {
                            "train/loss": mean_loss,
                            "train/lm_loss": mean_lm_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/gradient_norm": latest_grad_norm,
                            "train/tokens_per_second": tokens_per_second,
                            "train/epoch": epoch
                            + (batch_index + 1) / max(1, len(train_loader)),
                            "system/gpu_memory_allocated_gb": memory,
                            "system/gpu_memory_reserved_gb": (
                                torch.cuda.memory_reserved(context.device) / 1024**3
                                if context.device.type == "cuda"
                                else 0.0
                            ),
                        }
                        if args.use_moe and moe_totals is not None:
                            count_end = args.num_experts
                            prob_end = count_end + args.num_experts
                            expert_counts = moe_totals[:count_end]
                            router_prob_sums = moe_totals[count_end:prob_end]
                            entropy_sum = moe_totals[prob_end]
                            routed_tokens = moe_totals[prob_end + 1].clamp_min(1.0)
                            expert_usage = expert_counts / expert_counts.sum().clamp_min(1.0)
                            mean_router_probs = router_prob_sums / routed_tokens
                            normalized_entropy = entropy_sum / (
                                routed_tokens * math.log(args.num_experts)
                            )
                            metrics["train/router_aux_loss"] = mean_router_aux_loss
                            metrics["moe/router_entropy_normalized"] = (
                                normalized_entropy.item()
                            )
                            metrics["moe/max_load_ratio"] = expert_usage.max().item()
                            metrics["moe/min_load_ratio"] = expert_usage.min().item()
                            for expert_index in range(args.num_experts):
                                metrics[f"moe/expert_{expert_index}_usage"] = (
                                    expert_usage[expert_index].item()
                                )
                                metrics[
                                    f"moe/expert_{expert_index}_router_probability"
                                ] = mean_router_probs[expert_index].item()
                        tracker.log(metrics, step=global_step)

                    # 所有 rank 同时清空区间统计，为下一段日志窗口重新累计。
                    interval_loss = 0.0
                    interval_lm_loss = 0.0
                    interval_router_aux_loss = 0.0
                    interval_micro_batches = 0
                    interval_updates = 0
                    interval_expert_counts.zero_()
                    interval_router_prob_sums.zero_()
                    interval_router_entropy_sum.zero_()
                    interval_routed_token_count.zero_()
                    interval_start = time.perf_counter()

                # 所有 rank 必须一起进入 evaluate，因为函数内部包含分布式汇总。
                if global_step % args.eval_interval == 0:
                    evaluation_start = time.perf_counter()
                    val_loss, perplexity = evaluate(
                        model, validation_loader, context, amp_dtype
                    )
                    # 训练吞吐量不应把验证耗时算进去。
                    interval_start += time.perf_counter() - evaluation_start
                    if context.is_main:
                        print(
                            f"validation step={global_step:,} "
                            f"loss={val_loss:.4f} ppl={perplexity:.2f}"
                        )
                        tracker.log(
                            {
                                "validation/loss": val_loss,
                                "validation/perplexity": perplexity,
                            },
                            step=global_step,
                        )

                # checkpoint 保存完整训练状态，供 --resume 精确接续。
                if global_step % args.save_interval == 0 and context.is_main:
                    save_checkpoint(
                        args.save_dir / "latest.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch=epoch,
                        batch_in_epoch=batch_index + 1,
                        global_step=global_step,
                        args=args,
                        tracker_state=tracker.state_dict(),
                    )
                    print(f"Checkpoint    : {args.save_dir / 'latest.pt'}")

                if global_step >= total_steps:
                    stop_training = True
                    break

            # 只有恢复后的第一个 epoch 需要跳过 batch；随后都从头开始。
            start_batch = 0
            if context.is_main and stop_training:
                checkpoint_epoch = epoch
                checkpoint_batch = batch_index + 1
            else:
                checkpoint_epoch = epoch + 1
                checkpoint_batch = 0
            if context.is_main:
                save_checkpoint(
                    args.save_dir / "latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=checkpoint_epoch,
                    batch_in_epoch=checkpoint_batch,
                    global_step=global_step,
                    args=args,
                    tracker_state=tracker.state_dict(),
                )
            if stop_training:
                break

        # 9. 等所有 GPU 完成后，由主进程导出 Hugging Face 格式目录。
        if context.distributed:
            torch.distributed.barrier()
        if context.is_main:
            export_pretrained(
                args.output_dir, model, tokenizer, args.tokenizer_path
            )
            print(f"Training complete at optimizer step {global_step:,}.")
            print(f"Checkpoint    : {args.save_dir / 'latest.pt'}")
            print(f"Exported model: {args.output_dir}")
        tracker_finish_state = "success"
        tracker_finish_error = None
    # 将结束状态同步给 SwanLab/W&B；无论成功失败，finally 都释放 DDP 资源。
    except KeyboardInterrupt:
        tracker_finish_state = "aborted"
        tracker_finish_error = "Training interrupted by user (KeyboardInterrupt)."
        raise
    except BaseException:
        tracker_finish_state = "crashed"
        tracker_finish_error = traceback.format_exc()
        raise
    finally:
        try:
            tracker.finish(
                state=tracker_finish_state,
                error=tracker_finish_error,
            )
        finally:
            cleanup_distributed(context)


if __name__ == "__main__":
    main()

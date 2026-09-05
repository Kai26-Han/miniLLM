#!/usr/bin/env python3
"""miniLLM 预训练、SFT、DPO 与 PPO 公共工具。

本文件不直接启动训练，而是为训练入口提供四组能力：

1. 运行环境：单卡/多卡设备初始化、随机种子；
2. 训练策略：Warmup + Cosine 学习率；
3. 状态管理：原子保存 Checkpoint、恢复优化器和随机数状态；
4. 实验监控：用统一接口接入 SwanLab 或 Weights & Biases。

把这些与主训练循环拆开，可以让 ``train_pretrain.py`` 更接近算法流程，
同时避免保存、恢复、DDP 等工程细节散落在循环内部。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from collections.abc import Mapping
from importlib import import_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


TOKENIZER_ARTIFACT = "tokenizer.json"
CHAT_TEMPLATE_ARTIFACT = "chat_template.jinja"


@dataclass(frozen=True)
class DistributedContext:
    """描述当前进程在单卡或 DDP 作业中的身份。"""

    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    distributed: bool

    @property
    def is_main(self) -> bool:
        """只有 rank 0 负责打印、保存文件和上传实验指标。"""

        return self.rank == 0


class ExperimentTracker:
    """SwanLab/W&B 的最小适配层，让训练循环不依赖具体 SDK 名称。"""

    def __init__(
        self,
        backend: str = "none",
        run: Any = None,
        project: str | None = None,
        entity: str | None = None,
        mode: str | None = None,
        run_name: str | None = None,
    ) -> None:
        self.backend = backend
        self.run = run
        self.project = project
        self.entity = entity
        self.mode = mode
        self.run_name = run_name

    @property
    def enabled(self) -> bool:
        """未启用监控或非主进程时，所有操作自动变为空操作。"""

        return self.run is not None

    @property
    def run_id(self) -> str | None:
        value = getattr(self.run, "id", None) if self.run is not None else None
        return str(value) if value else None

    def log(self, metrics: dict[str, float | int], step: int) -> None:
        """按优化器 global_step 上传一组标量指标。"""

        if self.run is not None:
            self.run.log(metrics, step=step)

    def state_dict(self) -> dict[str, str | None]:
        """保存恢复实验所需信息；该字典会跟随模型 Checkpoint 落盘。"""

        return {
            "backend": self.backend,
            "run_id": self.run_id,
            "project": self.project,
            "entity": self.entity,
            "mode": self.mode,
            "run_name": self.run_name,
        }

    def finish(self, state: str = "success", error: str | None = None) -> None:
        """刷新剩余日志，并把成功/异常/手动中断状态提交到平台。"""

        if self.run is None:
            return
        if self.backend == "wandb":
            self.run.finish(exit_code=0 if state == "success" else 1)
        else:
            self.run.finish(state=state, error=error)
        self.run = None


def _serializable_config(value: Any) -> Any:
    """递归转换 Path 等对象，避免实验平台序列化超参数时报错。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serializable_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable_config(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def init_experiment_tracker(
    backend: str,
    context: DistributedContext,
    project: str,
    run_name: str | None,
    entity: str | None,
    group: str | None,
    tags: list[str],
    mode: str,
    log_dir: Path,
    config: dict[str, Any],
    run_id: str | None = None,
    strict_resume: bool = False,
) -> ExperimentTracker:
    """仅在 rank 0 初始化一个实验记录器。

    ``strict_resume`` 对应 MiniMind 使用的 ``resume='must'``：从模型
    Checkpoint 恢复时，SwanLab 上也必须找到原 Run，防止曲线悄悄分叉。
    """

    if backend == "none" or not context.is_main:
        return ExperimentTracker()
    log_dir = log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_config = _serializable_config(config)

    # 延迟导入：--tracker none 时不要求用户安装 SwanLab/W&B。
    try:
        module = import_module(backend)
    except ImportError as exc:
        raise RuntimeError(
            f"Tracker '{backend}' is not installed. Run: pip install {backend}"
        ) from exc

    # 两个平台共享的参数先集中构造，差异参数在下面分别映射。
    common = {
        "project": project,
        "group": group,
        "tags": tags or None,
        "config": safe_config,
        "id": run_id,
        "resume": ("must" if strict_resume else "allow") if run_id else None,
        "mode": mode,
    }
    if backend == "swanlab":
        run = module.init(
            **common,
            workspace=entity,
            experiment_name=run_name,
            logdir=str(log_dir),
        )
    elif backend == "wandb":
        run = module.init(
            **common,
            entity=entity,
            name=run_name,
            dir=str(log_dir),
        )
    else:
        raise ValueError(f"Unsupported tracker: {backend}")
    return ExperimentTracker(
        backend=backend,
        run=run,
        project=project,
        entity=entity,
        mode=mode,
        run_name=run_name,
    )


def setup_distributed(device_name: str = "auto") -> DistributedContext:
    """根据 torchrun 环境变量初始化 DDP，否则自动选择单设备。"""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training currently requires CUDA")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # NCCL 是 NVIDIA GPU 多进程通信后端；每个进程绑定一张本地 GPU。
        dist.init_process_group(backend="nccl")
        return DistributedContext(
            device=torch.device("cuda", local_rank),
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            distributed=True,
        )

    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return DistributedContext(
        device=device,
        rank=0,
        local_rank=0,
        world_size=1,
        distributed=False,
    )


def cleanup_distributed(context: DistributedContext) -> None:
    """训练结束后释放进程组，防止 torchrun 等待未销毁的通信资源。"""

    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int = 0) -> None:
    """固定 Python、NumPy、PyTorch RNG；不同 rank 使用不同子种子。"""

    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)


def tokenizer_fingerprint(tokenizer_dir: str | Path) -> str:
    """Return a stable SHA-256 fingerprint of the complete tokenizer graph.

    Vocabulary size and special-token IDs are insufficient compatibility
    checks: two BPE tokenizers can share both while assigning almost every
    ordinary token to a different embedding row.  Canonicalizing the complete
    ``tokenizer.json`` also covers vocab IDs, merges, normalizer,
    pre-tokenizer, post-processor and decoder settings without depending on
    whitespace or key ordering in the JSON file.
    """

    directory = Path(tokenizer_dir).expanduser().resolve()
    artifact = directory / TOKENIZER_ARTIFACT
    if not artifact.is_file():
        raise FileNotFoundError(f"Tokenizer artifact not found: {artifact}")
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid tokenizer artifact {artifact}: {exc}") from exc
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def ensure_config_tokenizer_fingerprint(
    config: Any,
    fingerprint: str,
    *,
    label: str,
) -> None:
    """Verify a model config's recorded tokenizer and attach it when legacy.

    Older miniLLM exports do not contain ``tokenizer_fingerprint``.  Their
    colocated tokenizer is checked separately by
    :func:`ensure_tokenizer_matches_models`; after that check the fingerprint
    is attached so the next export becomes self-describing.
    """

    recorded = (
        config.get("tokenizer_fingerprint")
        if isinstance(config, Mapping)
        else getattr(config, "tokenizer_fingerprint", None)
    )
    if recorded is not None and recorded != fingerprint:
        raise ValueError(
            f"Tokenizer fingerprint mismatch for {label}: "
            f"model={recorded}, selected={fingerprint}. The tokenizer cannot "
            "be changed without retraining the model from random weights."
        )
    if not isinstance(config, Mapping):
        setattr(config, "tokenizer_fingerprint", fingerprint)


def ensure_tokenizer_matches_models(
    tokenizer_dir: str | Path,
    model_dirs: Mapping[str, str | Path],
) -> str:
    """Require one tokenizer to match every exported miniLLM model directory."""

    selected_dir = Path(tokenizer_dir).expanduser().resolve()
    selected = tokenizer_fingerprint(selected_dir)
    selected_template_path = selected_dir / CHAT_TEMPLATE_ARTIFACT
    selected_template = (
        selected_template_path.read_bytes()
        if selected_template_path.is_file()
        else None
    )
    for label, model_dir_value in model_dirs.items():
        model_dir = Path(model_dir_value).expanduser().resolve()
        embedded = tokenizer_fingerprint(model_dir)
        if embedded != selected:
            raise ValueError(
                f"Tokenizer mismatch for {label}: selected {selected_dir} "
                f"({selected}) but {model_dir / TOKENIZER_ARTIFACT} has "
                f"fingerprint {embedded}. Use the tokenizer exported in the "
                "input model directory."
            )
        model_template_path = model_dir / CHAT_TEMPLATE_ARTIFACT
        model_template = (
            model_template_path.read_bytes()
            if model_template_path.is_file()
            else None
        )
        if model_template != selected_template:
            raise ValueError(
                f"Chat template mismatch for {label}: selected "
                f"{selected_template_path} does not match "
                f"{model_template_path}. Use the complete tokenizer exported "
                "with the input model."
            )
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid model config {config_path}: {exc}") from exc
        ensure_config_tokenizer_fingerprint(config, selected, label=label)
    return selected


def ensure_checkpoint_tokenizer_fingerprint(
    checkpoint: Mapping[str, Any],
    fingerprint: str,
) -> None:
    """Reject resume checkpoints whose tokenizer identity is unknown or changed."""

    saved = checkpoint.get("tokenizer_fingerprint")
    if saved is None:
        config = checkpoint.get("config") or checkpoint.get("actor_config") or {}
        if isinstance(config, Mapping):
            saved = config.get("tokenizer_fingerprint")
    if saved is None:
        args = checkpoint.get("args") or {}
        if isinstance(args, Mapping):
            saved = args.get("tokenizer_fingerprint")
    if saved is None:
        raise ValueError(
            "Resume checkpoint does not record a tokenizer fingerprint, so its "
            "embedding-ID mapping cannot be verified. Start a fresh run with "
            "the current code instead of resuming this legacy checkpoint."
        )
    if saved != fingerprint:
        raise ValueError(
            "Resume checkpoint tokenizer mismatch: "
            f"checkpoint={saved}, selected={fingerprint}."
        )


def build_cosine_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> LambdaLR:
    """创建线性 Warmup 后接 Cosine 衰减的学习率调度器。

    ``min_lr_ratio`` 是最终学习率与初始学习率之比，例如
    5e-5 / 5e-4 = 0.1，表示训练结束时衰减到初始值的 10%。
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def unwrap_model(model: nn.Module) -> nn.Module:
    """剥离 DDP/torch.compile 包装，获得真正持有 config/state_dict 的模型。"""

    unwrapped = model
    while True:
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        elif hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
        else:
            return unwrapped


def _rng_state() -> dict[str, Any]:
    """保存所有随机数生成器状态，实现更接近原轨迹的断点续训。"""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    """恢复由 ``_rng_state`` 保存的随机状态。"""

    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """先写临时文件再原子替换，避免中断时留下半个 Checkpoint。"""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        # 磁盘写满、进程中断等情况下，PyTorch 可能留下数 GB 的不完整文件。
        # 它无法用于恢复训练，保留反而会进一步耗尽磁盘空间。
        temporary.unlink(missing_ok=True)
        raise


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: Any,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    args: Any,
    tracker_state: dict[str, Any] | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> None:
    """保存继续训练所需的完整状态，而不只是模型权重。

    optimizer/scheduler/scaler 决定下一步怎样更新参数；epoch/global_step
    决定从哪里继续；RNG 和 tracker 则分别保证随机轨迹与监控曲线延续。
    """

    raw_model = unwrap_model(model)
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "args": vars(args).copy(),
        "config": raw_model.config.to_dict(),
        "tokenizer_fingerprint": getattr(args, "tokenizer_fingerprint", None),
        "rng_state": _rng_state(),
        "tracker": tracker_state or {},
        "extra_state": dict(extra_state or {}),
    }
    atomic_torch_save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    scaler: Any = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """加载完整 Checkpoint，并按需恢复训练组件与随机状态。"""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.4
        checkpoint = torch.load(path, map_location="cpu")

    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_rng:
        _restore_rng_state(checkpoint.get("rng_state"))
    return checkpoint


def save_ppo_checkpoint(
    path: Path,
    actor_model: nn.Module,
    critic_model: nn.Module,
    actor_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    actor_scheduler: LambdaLR,
    critic_scheduler: LambdaLR,
    scaler: Any,
    epoch: int,
    batch_in_epoch: int,
    rollout_step: int,
    global_step: int,
    args: Any,
    tracker_state: dict[str, Any] | None = None,
) -> None:
    """Atomically save the complete mutable state of a PPO run.

    Reference and reward models are immutable and intentionally omitted. Their
    configured paths remain in ``args`` so resume compatibility can verify the
    same behavioral anchor and reward source are used.
    """

    raw_actor = unwrap_model(actor_model)
    raw_critic = unwrap_model(critic_model)
    payload = {
        "actor_model": raw_actor.state_dict(),
        "critic_model": raw_critic.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "actor_scheduler": actor_scheduler.state_dict(),
        "critic_scheduler": critic_scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "rollout_step": rollout_step,
        "global_step": global_step,
        "args": vars(args).copy(),
        "actor_config": raw_actor.config.to_dict(),
        "tokenizer_fingerprint": getattr(args, "tokenizer_fingerprint", None),
        "rng_state": _rng_state(),
        "tracker": tracker_state or {},
    }
    atomic_torch_save(payload, path)


def load_ppo_checkpoint(
    path: Path,
    actor_model: nn.Module,
    critic_model: nn.Module,
    actor_optimizer: Optimizer | None = None,
    critic_optimizer: Optimizer | None = None,
    actor_scheduler: LambdaLR | None = None,
    critic_scheduler: LambdaLR | None = None,
    scaler: Any = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore Actor, Critic and both optimizer/scheduler states for PPO."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.4
        checkpoint = torch.load(path, map_location="cpu")

    required = {"actor_model", "critic_model"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(
            "PPO checkpoint is missing field(s): " + ", ".join(sorted(missing))
        )
    unwrap_model(actor_model).load_state_dict(checkpoint["actor_model"], strict=True)
    unwrap_model(critic_model).load_state_dict(checkpoint["critic_model"], strict=True)
    optional_states = (
        (actor_optimizer, "actor_optimizer"),
        (critic_optimizer, "critic_optimizer"),
        (actor_scheduler, "actor_scheduler"),
        (critic_scheduler, "critic_scheduler"),
    )
    for target, key in optional_states:
        if target is not None and checkpoint.get(key) is not None:
            target.load_state_dict(checkpoint[key])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_rng:
        _restore_rng_state(checkpoint.get("rng_state"))
    return checkpoint


def resolve_resume_path(save_dir: Path, resume: str | None) -> Path | None:
    """解析断点路径；自动模式无存档时返回 ``None``。"""

    if resume is None:
        return None
    if resume == "auto":
        candidate = save_dir.expanduser().resolve() / "latest.pt"
        return candidate if candidate.is_file() else None
    else:
        candidate = Path(resume).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {candidate}")
    return candidate


def export_pretrained(
    output_dir: Path,
    model: nn.Module,
    tokenizer: Any,
    tokenizer_source_dir: str | Path | None = None,
) -> None:
    """导出 Transformers 目录，并保留训练时 tokenizer 的原始字节。

    Fast-tokenizer ``save_pretrained`` may normalize its JSON representation
    after loading.  That can change vocabulary-graph IDs or their fingerprint
    even when training used the correct source artifact.  Save the runtime
    tokenizer first, then restore every matching tokenizer artifact verbatim
    from the explicitly selected training directory.
    """

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)

    source_dir: Path | None = None
    source_value = tokenizer_source_dir or getattr(tokenizer, "name_or_path", None)
    if source_value:
        candidate = Path(source_value).expanduser().resolve()
        if (candidate / TOKENIZER_ARTIFACT).is_file():
            source_dir = candidate
    if source_dir is not None:
        source_fingerprint = tokenizer_fingerprint(source_dir)
        ensure_config_tokenizer_fingerprint(
            raw_model.config,
            source_fingerprint,
            label="model being exported",
        )

    raw_model.save_pretrained(output_dir, safe_serialization=True)
    saved_files = tokenizer.save_pretrained(output_dir)

    if source_dir is not None and source_dir != output_dir:
        artifact_names = {
            TOKENIZER_ARTIFACT,
            CHAT_TEMPLATE_ARTIFACT,
            *(Path(saved).name for saved in saved_files),
        }
        for artifact_name in artifact_names:
            source_artifact = source_dir / artifact_name
            if source_artifact.is_file():
                shutil.copy2(source_artifact, output_dir / artifact_name)

    exported_fingerprint = tokenizer_fingerprint(output_dir)
    ensure_config_tokenizer_fingerprint(
        raw_model.config,
        exported_fingerprint,
        label="exported model",
    )


def distributed_sum(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    """把各 rank 的统计量求和，主要用于得到全局验证 Loss。"""

    if context.distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value

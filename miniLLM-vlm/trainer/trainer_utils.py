"""独立的 VLM 训练工具：资源加载、身份校验、日志、调度和断点管理。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerFast, SiglipImageProcessor, SiglipVisionModel

from minillm_vlm.model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm_vlm.model.model_vlm import MiniLLMVLM, VLMConfig
from minillm_vlm.dataset.vlm_dataset import EpochBatchSampler, VLMCollator

VLM_ROOT = Path(__file__).resolve().parents[1]

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


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (VLM_ROOT / path).resolve()


def add_resource_args(parser):
    parser.add_argument("--base-model", type=project_path, default=VLM_ROOT / "model/miniLLM-base")
    parser.add_argument("--vision-model", type=project_path, default=VLM_ROOT / "model/siglips")
    parser.add_argument("--tokenizer-path", type=project_path, default=None,
                        help="Defaults to the tokenizer exported with Base.")
    parser.add_argument("--image-token", default="<|reserved_0|>")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--attention-backend", choices=["auto", "math", "eager"], default="auto")


def add_data_args(parser):
    parser.add_argument("--data-path", type=project_path, nargs="+",
                        default=[VLM_ROOT / "dataset/pretrain_i2t.parquet"])
    parser.add_argument("--cache-dir", type=project_path, default=VLM_ROOT / "cache")
    parser.add_argument("--index-path", type=project_path, default=VLM_ROOT / "cache/pretrain_index.json")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)


def resolve_device(name, dtype_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = getattr(torch, dtype_name) if device.type == "cuda" else torch.float32
    if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support BF16; select --dtype float16")
    return device, dtype


def autocast(device, dtype):
    return torch.autocast("cuda", dtype=dtype) if device.type == "cuda" and dtype != torch.float32 else nullcontext()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def directory_hash(path):
    """只哈希模型/processor/tokenizer 相关文件；不包含 README 或绝对目录。"""
    extensions = {".json", ".safetensors", ".bin", ".jinja", ".txt", ".model"}
    files = sorted(p for p in Path(path).rglob("*")
                   if p.is_file() and p.suffix in extensions and not any(part.startswith(".") for part in p.relative_to(path).parts))
    if not files:
        raise FileNotFoundError(f"No model/tokenizer artifacts found in {path}")
    return json_hash({str(p.relative_to(path)): file_hash(p) for p in files})


def data_identity(paths):
    hashes = []
    for path in paths:
        print(f"Hashing dataset: {path}", flush=True)
        hashes.append(file_hash(path))
    return json_hash(hashes)  # 顺序是样本索引语义的一部分。


def validate_loading_info(info, label, allow_tied_head=False):
    missing = set(info.get("missing_keys", []))
    if allow_tied_head:
        missing.discard("lm_head.weight")
    if missing or info.get("unexpected_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError(f"{label} checkpoint does not match its implementation: {info}")


def load_base_tokenizer(tokenizer_dir):
    """兼容新版通用 tokenizer 的类名，始终读取原始分词图和配套配置。"""
    tokenizer_dir = Path(tokenizer_dir)
    config_path = tokenizer_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    if config.get("tokenizer_class") == "TokenizersBackend":
        # Transformers 5 的通用 Rust backend 在 4.x 中名为 PreTrainedTokenizerFast。
        # 仅处理这个已知别名；不修改导出文件，也不替换未知的自定义 tokenizer 类。
        if not (tokenizer_dir / TOKENIZER_ARTIFACT).is_file():
            raise FileNotFoundError(f"TokenizersBackend requires {tokenizer_dir / TOKENIZER_ARTIFACT}")
        print("Loading TokenizersBackend with the compatible PreTrainedTokenizerFast loader "
              "from the original local tokenizer files.", flush=True)
        return PreTrainedTokenizerFast.from_pretrained(tokenizer_dir, local_files_only=True)
    return AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)


def load_resources(args):
    base, vision = args.base_model, args.vision_model
    for directory in (base, vision):
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(f"Missing {directory / 'config.json'}; use a complete local model export")
    tokenizer_dir = args.tokenizer_path or base
    fingerprint = ensure_tokenizer_matches_models(tokenizer_dir, {"Base": base})
    tokenizer = load_base_tokenizer(tokenizer_dir)
    if not tokenizer.is_fast:
        raise ValueError("Use the fast tokenizer exported by miniLLM")
    raw_config = json.loads((base / "config.json").read_text())
    if raw_config.get("model_type") != "minillm":
        raise ValueError("--base-model must be your miniLLM export, model_type=minillm")
    config = MiniLLMConfig.from_pretrained(base, local_files_only=True)
    if len(tokenizer) != config.vocab_size:
        raise ValueError("Base vocab_size and tokenizer size differ")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(tokenizer, key) != getattr(config, key):
            raise ValueError(f"Base/tokenizer mismatch: {key}")
    if args.image_token not in tokenizer.get_vocab():
        raise ValueError("Image token must already exist in the Base tokenizer")
    image_ids = tokenizer.encode(args.image_token, add_special_tokens=False)
    if len(image_ids) != 1 or image_ids[0] in (config.bos_token_id, config.eos_token_id, config.pad_token_id):
        raise ValueError("Image token must be one distinct existing token")
    config.attention_backend = args.attention_backend
    config.use_cache = False
    llm, info = MiniLLMForCausalLM.from_pretrained(
        base, config=config, local_files_only=True, torch_dtype=torch.float32,
        output_loading_info=True,
    )
    validate_loading_info(info, "Base", allow_tied_head=config.tie_word_embeddings)
    vision_config = json.loads((vision / "config.json").read_text())
    if vision_config.get("model_type") != "siglip_vision_model":
        raise ValueError("Expected a SiglipVisionModel export (siglip_vision_model), matching MiniMind-V")
    encoder, info = SiglipVisionModel.from_pretrained(
        vision, local_files_only=True, torch_dtype=torch.float32, output_loading_info=True
    )
    validate_loading_info(info, "Vision encoder")
    processor = SiglipImageProcessor.from_pretrained(vision, local_files_only=True)
    pixel = processor(images=Image.new("RGB", (256, 256)), return_tensors="pt")["pixel_values"]
    encoder.eval()
    with torch.no_grad():
        features = encoder(pixel_values=pixel).last_hidden_state
    image_size = int(encoder.config.image_size)
    if tuple(pixel.shape[-2:]) != (image_size, image_size):
        raise ValueError("Processor size and vision model image_size differ")
    vlm_config = VLMConfig(image_ids[0], args.image_token, features.shape[-1],
                           features.shape[1], config.hidden_size, image_size)
    if vlm_config.image_token_len != (image_size // encoder.config.patch_size) ** 2:
        raise ValueError("Unexpected SigLIP patch token layout")
    model = MiniLLMVLM(llm, encoder, vlm_config)
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError(f"Non-finite initial weights: {name}")
    print("Hashing Base / vision / tokenizer identities...", flush=True)
    implementation = [VLM_ROOT / "model/model_minillm.py",
                      VLM_ROOT / "model/model_vlm.py", VLM_ROOT / "dataset/vlm_dataset.py",
                      VLM_ROOT / "minillm_vlm/__init__.py",
                      Path(__file__), VLM_ROOT / "trainer/train_pretrain_vlm.py"]
    identity = {"base": directory_hash(base), "vision": directory_hash(vision),
                "tokenizer": directory_hash(tokenizer_dir), "tokenizer_fingerprint": fingerprint,
                "implementation": json_hash([file_hash(p) for p in implementation])}
    resources = {"base_model": str(base), "vision_model": str(vision),
                 "tokenizer_path": str(tokenizer_dir), "image_token": args.image_token,
                 "attention_backend": args.attention_backend}
    return model, tokenizer, processor, identity, resources


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def adapter_payload(model, identity, resources, step, metrics=None):
    return {"format_version": 1, "kind": "minillm_vlm_projector",
            "projector": {k: v.detach().cpu().clone() for k, v in model.projector.state_dict().items()},
            "vlm_config": model.config.to_dict(), "identity": identity,
            "resources": resources, "global_step": step, "metrics": metrics or {}}


def export_adapter(path, model, identity, resources, step, metrics=None):
    atomic_torch_save(adapter_payload(model, identity, resources, step, metrics), Path(path))


def read_adapter(path):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("kind") != "minillm_vlm_projector" or artifact.get("format_version") != 1:
        raise ValueError("Expected a miniLLM-vlm Projector artifact")
    return artifact


def apply_adapter(model, artifact, identity):
    if artifact["identity"] != identity or artifact["vlm_config"] != model.config.to_dict():
        raise ValueError("Adapter Base/vision/tokenizer/code identity or VLM configuration mismatch")
    model.projector.load_state_dict(artifact["projector"], strict=True)


def model_report(model):
    return {"vlm_config": model.config.to_dict(),
            "base_config": {key: getattr(model.llm.config, key) for key in
                            ("hidden_size", "num_hidden_layers", "vocab_size", "use_moe", "num_experts")},
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def make_loader(dataset, args, epoch=0, start_batch=0, training=False):
    options = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                   collate_fn=VLMCollator(dataset.encoder.tokenizer.pad_token_id),
                   generator=torch.Generator().manual_seed(args.seed + epoch + 100000))
    if args.num_workers:
        options.update(persistent_workers=False, prefetch_factor=2, multiprocessing_context="spawn")
    if training:
        options["batch_sampler"] = EpochBatchSampler(len(dataset), args.batch_size, args.seed, epoch, start_batch)
    else:
        options.update(batch_size=args.batch_size, shuffle=False)
    return DataLoader(dataset, **options)


@torch.no_grad()
def evaluate(model, loader, device, dtype, paired=False):
    was_training = model.training
    model.eval()
    ce_sum = aux_sum = tokens = samples = 0
    correct_sum = wrong_sum = paired_tokens = paired_samples = 0
    try:
        for cpu_batch in loader:
            batch = to_device(cpu_batch, device)
            n = int((batch["labels"][:, 1:] != -100).sum())
            size = batch["input_ids"].shape[0]
            with autocast(device, dtype):
                output = model(**batch)
            if not torch.isfinite(output.loss):
                raise FloatingPointError("Non-finite validation loss")
            ce_sum += output.lm_loss.item() * n
            aux_sum += output.router_aux_loss.item() * size
            tokens += n
            samples += size
            if paired and size > 1:
                wrong_pixels = batch["pixel_values"].roll(1, 0)
                distinct = (wrong_pixels != batch["pixel_values"]).flatten(1).any(1)
                if distinct.any():
                    correct = {key: value[distinct] for key, value in batch.items()}
                    wrong = dict(correct, pixel_values=wrong_pixels[distinct])
                    with autocast(device, dtype):
                        correct_output = model(**correct)
                        wrong_output = model(**wrong)
                    if not torch.isfinite(correct_output.lm_loss) or not torch.isfinite(wrong_output.lm_loss):
                        raise FloatingPointError("Non-finite paired-image evaluation loss")
                    count = int((correct["labels"][:, 1:] != -100).sum())
                    correct_sum += correct_output.lm_loss.item() * count
                    wrong_sum += wrong_output.lm_loss.item() * count
                    paired_tokens += count
                    paired_samples += int(distinct.sum())
        if not tokens:
            raise ValueError("Validation set has no supervised tokens")
        metrics = {"val/lm_loss": ce_sum / tokens, "val/router_aux_loss": aux_sum / samples,
                   "val/samples": samples, "val/tokens": tokens}
        if paired:
            metrics["paired/samples"] = paired_samples
            if paired_tokens:
                metrics.update({"paired/correct_lm_loss": correct_sum / paired_tokens,
                                "paired/wrong_lm_loss": wrong_sum / paired_tokens,
                                "paired/wrong_minus_correct": (wrong_sum - correct_sum) / paired_tokens})
        return metrics
    finally:
        model.train(was_training)


def checked_state_load(module, state, label):
    """先检查整个 state，再加载，防止形状/非有限值错误造成部分覆盖。"""
    expected = module.state_dict()
    if set(state) != set(expected):
        raise ValueError(f"{label} parameter names mismatch")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise ValueError(f"{label} shape mismatch: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{label} non-finite weight: {key}")
    module.load_state_dict(state, strict=True)


def initialize_sft_projector(model, path, identity):
    """跨阶段初始化校验资源，不将 Pretrain 代码版本当作 SFT 续训版本。"""
    artifact = read_adapter(path)
    for key in ("base", "vision", "tokenizer", "tokenizer_fingerprint"):
        if key not in artifact["identity"] or artifact["identity"][key] != identity[key]:
            raise ValueError(f"Pretrain initialization resource identity mismatch: {key}")
    if artifact["vlm_config"] != model.config.to_dict():
        raise ValueError("Pretrain initialization VLM configuration mismatch")
    checked_state_load(model.projector, artifact["projector"], "Pretrain projector")
    return {"sha256": file_hash(path), "global_step": artifact["global_step"],
            "source_identity": artifact["identity"]}


def load_sft_resources(args):
    from minillm_vlm.model.model_vlm import MiniLLMSFT
    base, tokenizer, processor, identity, resources = load_resources(args)
    model = MiniLLMSFT(base.llm, base.vision_encoder, base.config, args.freeze_llm)
    model.projector = base.projector
    identity = dict(identity, implementation=sft_implementation_identity())
    provenance = initialize_sft_projector(model, args.from_pretrain, identity)
    return model, tokenizer, processor, identity, resources, provenance


def export_sft(path, model, identity, resources, provenance, step, metrics=None):
    """保存全部更新后的 LLM 和 Projector；SigLIP 使用外部冻结资源。"""
    atomic_torch_save({"format_version": 1, "kind": "minillm_vlm_sft",
        "llm": {k: v.detach().cpu().clone() for k, v in model.llm.state_dict().items()},
        "projector": {k: v.detach().cpu().clone() for k, v in model.projector.state_dict().items()},
        "base_config": model.llm.config.to_dict(), "vlm_config": model.config.to_dict(),
        "identity": identity, "resources": resources, "pretrain": provenance,
        "freeze_llm": model.freeze_llm, "global_step": step, "metrics": metrics or {}}, Path(path))


def load_sft_for_eval(path, args):
    from minillm_vlm.model.model_vlm import MiniLLMSFT
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("kind") != "minillm_vlm_sft" or artifact.get("format_version") != 1:
        raise ValueError("Expected an SFT inference artifact (best_sft.pt or last_sft.pt)")
    tokenizer_dir = args.tokenizer_path or Path(path).parent / "tokenizer"
    vision = args.vision_model or Path(artifact["resources"]["vision_model"])
    tokenizer = load_base_tokenizer(tokenizer_dir)
    if tokenizer_fingerprint(tokenizer_dir) != artifact["identity"]["tokenizer_fingerprint"]:
        raise ValueError("SFT inference tokenizer fingerprint mismatch")
    # 导出 tokenizer 的全部文件指纹独立记录，防止模板/特殊 token 配置被换掉。
    if directory_hash(tokenizer_dir) != artifact["resources"]["exported_tokenizer_identity"]:
        raise ValueError("SFT inference tokenizer files mismatch")
    if directory_hash(vision) != artifact["identity"]["vision"]:
        raise ValueError("SFT inference vision identity mismatch")
    config = MiniLLMConfig(**artifact["base_config"])
    config.attention_backend = args.attention_backend or artifact["resources"]["attention_backend"]
    llm = MiniLLMForCausalLM(config)
    encoder, info = SiglipVisionModel.from_pretrained(vision, local_files_only=True,
        torch_dtype=torch.float32, output_loading_info=True)
    validate_loading_info(info, "SFT vision encoder")
    processor = SiglipImageProcessor.from_pretrained(vision, local_files_only=True)
    model = MiniLLMSFT(llm, encoder, VLMConfig(**artifact["vlm_config"]), artifact["freeze_llm"])
    checked_state_load(model.llm, artifact["llm"], "SFT language model")
    checked_state_load(model.projector, artifact["projector"], "SFT projector")
    if len(tokenizer) != config.vocab_size or any(getattr(tokenizer, key) != getattr(config, key)
        for key in ("bos_token_id", "eos_token_id", "pad_token_id")):
        raise ValueError("SFT tokenizer/config mismatch")
    return model, tokenizer, processor, artifact


def make_sft_loader(dataset, args, epoch=0, start_batch=0, training=False, metadata=False):
    from minillm_vlm.dataset.vlm_dataset import SFTCollator
    options = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                   collate_fn=SFTCollator(dataset.encoder.tokenizer.pad_token_id, metadata),
                   generator=torch.Generator().manual_seed(args.seed + epoch + 100000))
    if args.num_workers:
        options.update(persistent_workers=False, prefetch_factor=2, multiprocessing_context="spawn")
    if training:
        options["batch_sampler"] = EpochBatchSampler(len(dataset), args.batch_size, args.seed, epoch, start_batch)
    else:
        options.update(batch_size=args.batch_size, shuffle=False)
    return DataLoader(dataset, **options)


@torch.no_grad()
def evaluate_sft(model, loader, device, dtype):
    from minillm_vlm.dataset.vlm_dataset import SFT_TYPES
    was_training = model.training
    model.eval()
    totals = {name: [0., 0, 0] for name in SFT_TYPES}
    aux_sum = samples = paired_tokens = paired_samples = 0
    correct_sum = wrong_sum = 0.
    try:
        for cpu_batch in loader:
            types = cpu_batch.pop("sample_type")
            batch = to_device(cpu_batch, device)
            with autocast(device, dtype):
                output = model(**batch)
            if not torch.isfinite(output.loss):
                raise FloatingPointError("Non-finite SFT validation loss")
            labels = batch["labels"][:, 1:]
            losses = torch.nn.functional.cross_entropy(output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
                labels.reshape(-1), ignore_index=-100, reduction="none").reshape_as(labels)
            for i, name in enumerate(SFT_TYPES):
                selected = types.eq(i).to(device)
                totals[name][0] += losses[selected].sum().item()
                totals[name][1] += int(labels[selected].ne(-100).sum())
                totals[name][2] += int(selected.sum())
            size = len(types)
            samples += size
            aux_sum += output.router_aux_loss.item() * size
            visual = batch["has_image"]
            if int(visual.sum()) > 1:
                images = batch["pixel_values"][visual]
                wrong_pixels = images.roll(1, 0)
                distinct = (images != wrong_pixels).flatten(1).any(1)
                if distinct.any():
                    correct = {key: value[visual][distinct] for key, value in batch.items()}
                    with autocast(device, dtype):
                        good = model(**correct).lm_loss
                        bad = model(**dict(correct, pixel_values=wrong_pixels[distinct])).lm_loss
                    if not torch.isfinite(good) or not torch.isfinite(bad):
                        raise FloatingPointError("Non-finite SFT paired-image loss")
                    count = int(correct["labels"][:, 1:].ne(-100).sum())
                    correct_sum += good.item() * count
                    wrong_sum += bad.item() * count
                    paired_tokens += count
                    paired_samples += int(distinct.sum())
        token_count = sum(value[1] for value in totals.values())
        if not token_count:
            raise ValueError("SFT validation set has no supervised tokens")
        metrics = {"val/lm_loss": sum(v[0] for v in totals.values()) / token_count,
                   "val/router_aux_loss": aux_sum / samples, "val/samples": samples,
                   "val/tokens": token_count, "paired/samples": paired_samples}
        for name, (loss, tokens, count) in totals.items():
            metrics[f"val/{name}_samples"] = count
            if tokens:
                metrics[f"val/{name}_lm_loss"] = loss / tokens
        visual_totals = [totals[name] for name in SFT_TYPES if name != "text"]
        if sum(v[1] for v in visual_totals):
            metrics["val/visual_lm_loss"] = sum(v[0] for v in visual_totals) / sum(v[1] for v in visual_totals)
        # 有显式 instruction 类型时优先；没有来源元数据就报告 visual，不猜测任务。
        metrics["val/selection_loss"] = metrics.get("val/instruction_lm_loss",
            metrics.get("val/visual_lm_loss", metrics["val/lm_loss"]))
        if paired_tokens:
            metrics.update({"paired/correct_lm_loss": correct_sum / paired_tokens,
                            "paired/wrong_lm_loss": wrong_sum / paired_tokens,
                            "paired/wrong_minus_correct": (wrong_sum-correct_sum) / paired_tokens})
        return metrics
    finally:
        model.train(was_training)


def sft_implementation_identity():
    shared = [VLM_ROOT / name for name in ("model/model_minillm.py", "model/model_vlm.py",
        "dataset/vlm_dataset.py", "minillm_vlm/__init__.py", "trainer/trainer_utils.py",
        "trainer/train_pretrain_vlm.py")]
    return json_hash({"shared": json_hash([file_hash(path) for path in shared]),
                     "sft": file_hash(VLM_ROOT / "trainer/train_sft_vlm.py")})

#!/usr/bin/env python3
"""LoRA adapters for miniLLM without third-party PEFT wrappers.

The base ``nn.Linear`` parameter names are preserved after injection.  Only
``lora_A`` and ``lora_B`` are trainable, which keeps adapter checkpoints small
and lets a merged model return to the original miniLLM architecture.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F


ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
TARGET_PRESETS = {
    "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "all-linear": (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
}


@dataclass(frozen=True)
class LoRAConfig:
    """Portable metadata needed to reconstruct a LoRA adapter."""

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = TARGET_PRESETS["attention"]
    base_model_name_or_path: str | None = None
    base_model_sha256: str | None = None
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.target_modules or any(not item for item in self.target_modules):
            raise ValueError("LoRA target_modules cannot be empty")
        if self.format_version != 1:
            raise ValueError(f"Unsupported LoRA format version: {self.format_version}")

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        payload["peft_type"] = "LORA"
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LoRAConfig":
        data = dict(payload)
        data.pop("peft_type", None)
        data["target_modules"] = tuple(data["target_modules"])
        return cls(**data)


class LoRALinear(nn.Linear):
    """A Linear layer with a frozen base path and trainable low-rank update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = float(alpha) / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_A = nn.Linear(
            in_features, rank, bias=False, device=device, dtype=dtype
        )
        self.lora_B = nn.Linear(
            rank, out_features, bias=False, device=device, dtype=dtype
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "LoRALinear":
        result = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        # Reuse the exact Parameter objects. This avoids copying the base model
        # and keeps state-dict names identical to the pre-injection model.
        result.weight = module.weight
        result.weight.requires_grad_(False)
        if module.bias is not None:
            result.bias = module.bias
            result.bias.requires_grad_(False)
        result.train(module.training)
        return result

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base = F.linear(input, self.weight, self.bias)
        update = self.lora_B(self.lora_A(self.lora_dropout(input)))
        return base + update * self.scaling

    @torch.no_grad()
    def merge_into_base(self) -> None:
        update = self.lora_B.weight.float() @ self.lora_A.weight.float()
        self.weight.add_(update.to(self.weight.dtype), alpha=self.scaling)


def resolve_target_modules(
    preset: str | None = "attention",
    custom_targets: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resolve a named target preset or an explicit list of module suffixes."""

    if custom_targets:
        targets = tuple(dict.fromkeys(item.strip() for item in custom_targets))
        if any(not item for item in targets):
            raise ValueError("Custom LoRA target names cannot be blank")
        return targets
    if preset not in TARGET_PRESETS:
        raise ValueError(
            f"Unknown LoRA target preset {preset!r}; choose {sorted(TARGET_PRESETS)}"
        )
    return TARGET_PRESETS[preset]


def _unwrap_model(model: nn.Module) -> nn.Module:
    result = model
    while True:
        if hasattr(result, "module"):
            result = result.module
        elif hasattr(result, "_orig_mod"):
            result = result._orig_mod
        else:
            return result


def _matches_target(name: str, targets: Sequence[str]) -> bool:
    return any(name == target or name.endswith(f".{target}") for target in targets)


def _parent_and_attribute(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def apply_lora(model: nn.Module, config: LoRAConfig) -> tuple[str, ...]:
    """Replace selected Linear layers and freeze every non-LoRA parameter."""

    raw_model = _unwrap_model(model)
    if any(isinstance(module, LoRALinear) for module in raw_model.modules()):
        raise ValueError("LoRA is already applied to this model")
    candidates = [
        (name, module)
        for name, module in raw_model.named_modules()
        if isinstance(module, nn.Linear)
        and _matches_target(name, config.target_modules)
    ]
    matched_targets = {
        target
        for target in config.target_modules
        if any(_matches_target(name, (target,)) for name, _ in candidates)
    }
    unmatched_targets = sorted(set(config.target_modules) - matched_targets)
    if not candidates or unmatched_targets:
        raise ValueError(
            "LoRA target modules were not found in the model: "
            + ", ".join(unmatched_targets or config.target_modules)
        )
    for parameter in raw_model.parameters():
        parameter.requires_grad_(False)
    injected: list[str] = []
    for name, module in candidates:
        parent, attribute = _parent_and_attribute(raw_model, name)
        replacement = LoRALinear.from_linear(
            module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        setattr(parent, attribute, replacement)
        injected.append(name)
    return tuple(injected)


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    for module in _unwrap_model(model).modules():
        if isinstance(module, LoRALinear):
            parameters.extend((module.lora_A.weight, module.lora_B.weight))
    return parameters


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return detached CPU LoRA tensors using stable model parameter names."""

    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in _unwrap_model(model).named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    if not state:
        raise ValueError("Model has no LoRA parameters")
    return state


def load_adapter_state_dict(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    """Strictly load only adapter tensors without touching frozen base weights."""

    expected = {
        name: parameter
        for name, parameter in _unwrap_model(model).named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatches = sorted(
        name
        for name in set(expected) & set(state)
        if expected[name].shape != state[name].shape
    )
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            "Adapter state does not match the injected model: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatches={shape_mismatches[:5]}"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(state[name].to(parameter.device, parameter.dtype))


def model_directory_sha256(path: str | Path) -> str:
    """Hash model config and weight shards so adapters cannot use a wrong base."""

    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {directory}")
    files: list[Path] = []
    config_path = directory / "config.json"
    if config_path.is_file():
        files.append(config_path)
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        files.extend(sorted(directory.glob(pattern)))
    files = list(dict.fromkeys(files))
    if not files or config_path not in files:
        raise FileNotFoundError(
            f"No config.json and model weights found in base model: {directory}"
        )
    digest = hashlib.sha256()
    for file_path in files:
        digest.update(file_path.name.encode("utf-8"))
        with file_path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def save_adapter(
    model: nn.Module,
    output_dir: str | Path,
    config: LoRAConfig,
    *,
    tokenizer: Any | None = None,
    base_model_path: str | Path | None = None,
) -> LoRAConfig:
    """Save adapter-only safetensors, metadata, and optionally the tokenizer."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_base = (
        str(Path(base_model_path).expanduser().resolve())
        if base_model_path is not None
        else config.base_model_name_or_path
    )
    fingerprint = config.base_model_sha256
    if fingerprint is None and resolved_base is not None:
        fingerprint = model_directory_sha256(resolved_base)
    saved_config = LoRAConfig(
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
        target_modules=config.target_modules,
        base_model_name_or_path=resolved_base,
        base_model_sha256=fingerprint,
        format_version=config.format_version,
    )
    weights_path = output / ADAPTER_WEIGHTS_NAME
    temporary_weights = weights_path.with_suffix(weights_path.suffix + ".tmp")
    config_path = output / ADAPTER_CONFIG_NAME
    temporary_config = config_path.with_suffix(config_path.suffix + ".tmp")
    try:
        save_file(adapter_state_dict(model), str(temporary_weights))
        temporary_config.write_text(
            json.dumps(saved_config.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_weights, weights_path)
        os.replace(temporary_config, config_path)
    except BaseException:
        temporary_weights.unlink(missing_ok=True)
        temporary_config.unlink(missing_ok=True)
        raise
    if tokenizer is not None:
        tokenizer.save_pretrained(output)
    return saved_config


def read_adapter_config(adapter_dir: str | Path) -> LoRAConfig:
    path = Path(adapter_dir).expanduser().resolve() / ADAPTER_CONFIG_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Adapter config not found: {path}")
    return LoRAConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_adapter(
    model: nn.Module,
    adapter_dir: str | Path,
    *,
    base_model_path: str | Path | None = None,
    verify_base: bool = True,
) -> LoRAConfig:
    """Inject and load an adapter, optionally verifying its exact base weights."""

    directory = Path(adapter_dir).expanduser().resolve()
    config = read_adapter_config(directory)
    weights_path = directory / ADAPTER_WEIGHTS_NAME
    if not weights_path.is_file():
        raise FileNotFoundError(f"Adapter weights not found: {weights_path}")
    if verify_base and config.base_model_sha256:
        candidate = base_model_path or config.base_model_name_or_path
        if candidate is None:
            raise ValueError("Adapter requires a base model path for fingerprint check")
        actual = model_directory_sha256(candidate)
        if actual != config.base_model_sha256:
            raise ValueError(
                "Adapter base model fingerprint mismatch: "
                f"expected {config.base_model_sha256}, got {actual}"
            )
    apply_lora(model, config)
    load_adapter_state_dict(model, load_file(str(weights_path), device="cpu"))
    return config


@torch.no_grad()
def merge_lora(model: nn.Module, *, unload: bool = True) -> tuple[str, ...]:
    """Merge all low-rank updates into base weights and optionally remove LoRA."""

    raw_model = _unwrap_model(model)
    modules = [
        (name, module)
        for name, module in raw_model.named_modules()
        if isinstance(module, LoRALinear)
    ]
    if not modules:
        raise ValueError("Model has no LoRA layers to merge")
    for name, module in modules:
        module.merge_into_base()
        if unload:
            replacement = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            replacement.weight = module.weight
            if module.bias is not None:
                replacement.bias = module.bias
            replacement.train(module.training)
            parent, attribute = _parent_and_attribute(raw_model, name)
            setattr(parent, attribute, replacement)
    return tuple(name for name, _ in modules)


def count_parameters(parameters: Iterable[nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)

#!/usr/bin/env python3
"""miniLLM-vlm 自包含的 Decoder-only 语言模型实现。

本项目独立维护此文件，保持 miniLLM Base 的配置与参数名兼容；
运行时不导入其他项目的代码，支持视觉特征 inputs_embeds 输入。

学习时可以把一次前向传播理解为下面这条流水线：

    input_ids
        -> Token Embedding
        -> N 个 Decoder Layer
           （RMSNorm -> GQA + RoPE -> 残差 -> RMSNorm -> SwiGLU -> 残差）
        -> 最终 RMSNorm
        -> LM Head
        -> 每个位置预测下一个 Token

本文件将配置和模型实现集中在一起，保持小项目结构。
核心组件包括 RMSNorm、RoPE、分组查询注意力（GQA）、SwiGLU，以及兼容
``transformers.GenerationMixin`` 的因果语言模型输出层。
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput


PastKeyValue = tuple[torch.Tensor, torch.Tensor]
PastKeyValues = tuple[PastKeyValue, ...]


@dataclass
class MiniLLMModelOutput(BaseModelOutput):
    """Decoder 输出，以及训练/监控 MoE Router 所需的聚合统计。"""

    router_aux_loss: torch.FloatTensor | None = None
    expert_counts: torch.FloatTensor | None = None
    router_prob_sums: torch.FloatTensor | None = None
    router_entropy_sum: torch.FloatTensor | None = None
    routed_token_count: torch.FloatTensor | None = None
    past_key_values: PastKeyValues | None = None


@dataclass
class MiniLLMCausalLMOutput(CausalLMOutput):
    """在标准 CausalLM 输出上增加语言模型损失与 MoE 路由统计。"""

    lm_loss: torch.FloatTensor | None = None
    router_aux_loss: torch.FloatTensor | None = None
    expert_counts: torch.FloatTensor | None = None
    router_prob_sums: torch.FloatTensor | None = None
    router_entropy_sum: torch.FloatTensor | None = None
    routed_token_count: torch.FloatTensor | None = None
    past_key_values: PastKeyValues | None = None


class MiniLLMConfig(PretrainedConfig):
    """集中保存模型结构参数，并支持 Hugging Face 配置的保存与加载。"""

    model_type = "minillm"

    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_size: int = 768,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        attention_dropout: float = 0.0,
        attention_backend: str = "eager",
        hidden_dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_cache: bool = False,
        use_moe: bool = True,
        num_experts: int = 4,
        num_experts_per_tok: int = 1,
        moe_intermediate_size: int | None = None,
        norm_topk_prob: bool = True,
        router_aux_loss_coef: float = 5e-4,
        tokenizer_fingerprint: str | None = None,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        # 特殊 Token ID 和词嵌入权重共享等通用配置交由父类管理。
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        # 每个注意力头必须分到相同维度：head_dim = hidden_size / heads。
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        # GQA 要求若干个 Query 头恰好共用一个 Key/Value 头。
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if vocab_size <= max(pad_token_id, bos_token_id, eos_token_id):
            raise ValueError("vocab_size does not cover the configured special token IDs")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not 1 <= num_experts_per_tok <= num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")
        if router_aux_loss_coef < 0:
            raise ValueError("router_aux_loss_coef cannot be negative")
        if attention_backend not in {"eager", "auto", "math"}:
            raise ValueError(
                "attention_backend must be one of: eager, auto, math"
            )

        if intermediate_size is None:
            # MiniMind 风格的紧凑 SwiGLU 中间层，并向上取整到 64 的倍数，
            # 以便 GPU 矩阵计算使用更规整的形状。
            intermediate_size = math.ceil(hidden_size * math.pi / 64) * 64
        if moe_intermediate_size is None:
            moe_intermediate_size = intermediate_size
        if intermediate_size <= 0 or moe_intermediate_size <= 0:
            raise ValueError("intermediate sizes must be positive")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.attention_backend = attention_backend
        self.hidden_dropout = hidden_dropout
        self.initializer_range = initializer_range
        # MoE 默认对齐 MiniMind：4个 Expert、每个 Token 激活 Top-1。
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef
        # Embedding/output-head rows are meaningful only under this exact
        # tokenizer mapping.  New training exports persist the fingerprint so
        # every downstream stage can fail before silently using another BPE.
        self.tokenizer_fingerprint = tokenizer_fingerprint
        # 训练默认关闭缓存；部署生成时可开启，避免每步重复计算完整前缀。
        self.use_cache = use_cache


class MiniLLMRMSNorm(nn.Module):
    """RMSNorm：只按均方根缩放，不像 LayerNorm 那样减去均值。"""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """沿隐藏维归一化，输入输出形状均为 [batch, sequence, hidden]。"""

        input_dtype = hidden_states.dtype
        # 在 float32 中计算方差，避免 float16/bfloat16 精度不足。
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states.float() * torch.rsqrt(variance + self.eps)
        # 乘以 FP32 权重后再恢复输入 dtype，对齐 MiniMind 的
        # ``(weight * norm(x.float())).type_as(x)`` 语义。
        return (self.weight * hidden_states).to(input_dtype)


class MiniLLMRotaryEmbedding(nn.Module):
    """生成 RoPE 所需的不同频率正弦、余弦位置编码。"""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

    def forward(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据 [batch, sequence] 的位置编号生成 cos/sin 张量。"""

        # Do not keep inv_freq as a non-persistent buffer. During
        # `from_pretrained`, Transformers may construct the model on the meta
        # device; non-persistent RoPE buffers are not restored from the
        # checkpoint and can be materialized with uninitialized values. Build
        # this tiny deterministic vector on the real device instead.
        frequency_indices = torch.arange(
            0,
            self.head_dim,
            2,
            dtype=torch.float32,
            device=position_ids.device,
        )
        inv_freq = 1.0 / (
            self.rope_theta ** (frequency_indices / self.head_dim)
        )
        # 外积把每个 token 的位置编号映射到 head_dim / 2 个旋转频率。
        frequencies = torch.einsum(
            "bi,j->bij", position_ids.float(), inv_freq
        )
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    """完成 RoPE 中的二维旋转：把 (x1, x2) 变为 (-x2, x1)。"""

    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把位置信息旋转进 Query 和 Key；Value 不需要应用 RoPE。"""

    # 在 head 维插入 1，使 cos/sin 可广播到所有注意力头。
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return (
        query * cosine + rotate_half(query) * sine,
        key * cosine + rotate_half(key) * sine,
    )


def repeat_kv(hidden_states: torch.Tensor, repetitions: int) -> torch.Tensor:
    """将较少的 K/V 头复制到 Query 头数，这是 GQA 的共享方式。

    输入形状：[batch, kv_heads, sequence, head_dim]；
    输出形状：[batch, attention_heads, sequence, head_dim]。
    """

    if repetitions == 1:
        return hidden_states
    batch, kv_heads, sequence, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, repetitions, sequence, head_dim
    )
    return hidden_states.reshape(batch, kv_heads * repetitions, sequence, head_dim)


class MiniLLMAttention(nn.Module):
    """带 RoPE 的分组查询自注意力（Grouped Query Attention）。"""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # 例如 8 个 Q 头、4 个 KV 头时，每个 KV 头服务 2 个 Q 头。
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        self.attention_backend = config.attention_backend

        # GQA 只减少 K/V 投影头数，Q 仍保留完整的注意力头数。
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.rotary_emb = MiniLLMRotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, PastKeyValue | None]:
        """执行自注意力，输入输出均为 [batch, sequence, hidden_size]。"""

        batch, sequence, _ = hidden_states.shape
        past_length = 0
        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError("past_key_value must contain key and value tensors")
            past_key, past_value = past_key_value
            expected_prefix = (batch, self.num_key_value_heads)
            if past_key.shape[:2] != expected_prefix or past_value.shape[:2] != expected_prefix:
                raise ValueError("cached key/value batch or head count does not match input")
            if past_key.shape[-1] != self.head_dim or past_value.shape[-1] != self.head_dim:
                raise ValueError("cached key/value head dimension does not match model")
            if past_key.shape[-2] != past_value.shape[-2]:
                raise ValueError("cached key and value sequence lengths differ")
            past_length = past_key.shape[-2]
        key_value_length = past_length + sequence
        mask = (
            attention_mask.to(torch.bool)
            if attention_mask is not None
            else None
        )
        if mask is not None and mask.shape != (batch, key_value_length):
            raise ValueError(
                "attention_mask must cover cached and current tokens: "
                f"expected {(batch, key_value_length)}, got {tuple(mask.shape)}"
            )
        has_padding = (
            self.attention_backend != "eager"
            and mask is not None
            and not bool(torch.all(mask).item())
        )
        use_eager_attention = self.attention_backend == "eager" or has_padding
        disable_autocast = (
            torch.autocast(device_type="cuda", enabled=False)
            if use_eager_attention and hidden_states.device.type == "cuda"
            else nullcontext()
        )

        with disable_autocast:
            projection_input = (
                hidden_states.float()
                if use_eager_attention
                else hidden_states
            )
            # 先投影再拆分多头：[B, S, H] -> [B, heads, S, head_dim]。
            query = self.q_proj(projection_input).view(
                batch, sequence, self.num_heads, self.head_dim
            )
            key = self.k_proj(projection_input).view(
                batch, sequence, self.num_key_value_heads, self.head_dim
            )
            value = self.v_proj(projection_input).view(
                batch, sequence, self.num_key_value_heads, self.head_dim
            )
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            # 未提供位置时，默认每条序列的位置都是 0 到 sequence - 1。
            if position_ids is None:
                position_ids = torch.arange(
                    sequence, device=hidden_states.device
                )
                position_ids = position_ids.unsqueeze(0).expand(batch, -1)
            cosine, sine = self.rotary_emb(position_ids, query.dtype)
            query, key = apply_rotary_pos_emb(
                query, key, cosine, sine
            )

            # Cache unrepeated K/V heads. GQA replication is only needed for
            # the actual attention operation and would otherwise waste memory.
            if past_key_value is not None:
                past_key, past_value = past_key_value
                key = torch.cat((past_key.to(key), key), dim=-2)
                value = torch.cat((past_value.to(value), value), dim=-2)
            present_key_value = (key, value) if use_cache else None

            # 将共享的 K/V 头扩展到与 Q 头一致。
            key = repeat_kv(key, self.num_key_value_groups)
            value = repeat_kv(value, self.num_key_value_groups)
            dropout = self.attention_dropout if self.training else 0.0

            if use_eager_attention:
                # Dynamic-padding batches deliberately avoid SDPA. The complete
                # attention block stays in float32 so BF16 kernels cannot create
                # non-finite Q/K/V or scores before the stable softmax fallback.
                float_query = query.float()
                float_key = key.float()
                float_value = value.float()
                scores = torch.matmul(
                    float_query, float_key.transpose(-2, -1)
                ) / math.sqrt(self.head_dim)
                query_positions = torch.arange(
                    past_length,
                    key_value_length,
                    device=hidden_states.device,
                ).unsqueeze(-1)
                key_positions = torch.arange(
                    key_value_length, device=hidden_states.device
                ).unsqueeze(0)
                causal_mask = key_positions <= query_positions
                allowed_mask = causal_mask[None, None, :, :]
                if mask is not None:
                    allowed_mask = (
                        allowed_mask & mask[:, None, None, :]
                    )
                scores = scores.masked_fill(
                    ~allowed_mask, torch.finfo(scores.dtype).min
                )
                attention_probs = F.softmax(scores, dim=-1)
                attention_probs = F.dropout(
                    attention_probs,
                    p=dropout,
                    training=self.training,
                )
                attention_output = torch.matmul(
                    attention_probs, float_value
                )
            else:
                # For cached decoding q_len differs from kv_len, so an explicit
                # bottom-right aligned causal mask is required. SDPA's plain
                # is_causal=True mask is only unambiguous for a fresh sequence.
                if past_length == 0:
                    attention_output = F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        dropout_p=dropout,
                        is_causal=True,
                    )
                elif sequence == 1:
                    # The only query is the newest token, so every cached key
                    # is causally visible. Avoid allocating a causal mask on
                    # every decode step; retain only a padding mask if needed.
                    key_padding_mask = None
                    if mask is not None and not bool(torch.all(mask).item()):
                        key_padding_mask = mask[:, None, None, :]
                    attention_output = F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        attn_mask=key_padding_mask,
                        dropout_p=dropout,
                        is_causal=False,
                    )
                else:
                    query_positions = torch.arange(
                        past_length,
                        key_value_length,
                        device=hidden_states.device,
                    ).unsqueeze(-1)
                    key_positions = torch.arange(
                        key_value_length, device=hidden_states.device
                    ).unsqueeze(0)
                    allowed_mask = (
                        key_positions <= query_positions
                    )[None, None, :, :]
                    if mask is not None:
                        allowed_mask = allowed_mask & mask[:, None, None, :]
                    attention_output = F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        attn_mask=allowed_mask,
                        dropout_p=dropout,
                        is_causal=False,
                    )

            if mask is not None:
                query_mask = mask[:, -sequence:]
                attention_output = attention_output.masked_fill(
                    ~query_mask[:, None, :, None], 0.0
                )

            # 合并所有注意力头：[B, heads, S, head_dim] -> [B, S, H]。
            attention_output = attention_output.transpose(1, 2).contiguous()
            attention_output = attention_output.view(batch, sequence, -1)
            return self.o_proj(attention_output), present_key_value


class MiniLLMMLP(nn.Module):
    """SwiGLU 前馈网络：SiLU(gate(x)) * up(x)，再投影回隐藏维度。"""

    def __init__(
        self,
        config: MiniLLMConfig,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(
            config.hidden_size, intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """保持 [batch, sequence, hidden_size] 的输入输出形状不变。"""

        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class MiniLLMSparseMoE(nn.Module):
    """无容量上限的稀疏 MoE：Router 为每个有效 Token 选择 Top-K Expert。

    当前推荐配置为4个 Expert、Top-1。所有 Expert 参数都保存在显存中，
    但一次前向传播中每个 Token 只执行一个 Expert，因此激活计算量仍接近
    Dense 64M 模型。这里不设置 capacity factor，也不会丢弃 Token。
    """

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                MiniLLMMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """路由有效 Token，并返回输出、辅助损失和可视化统计。

        统计张量均已 detach，不会为了 SwanLab 日志保留计算图；只有
        router_aux_loss 会参与反向传播。
        """

        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, original_shape[-1])
        if token_mask is None:
            valid_mask = torch.ones(
                flat_states.shape[0], dtype=torch.bool, device=hidden_states.device
            )
        else:
            valid_mask = token_mask.reshape(-1).to(torch.bool)
        valid_indices = valid_mask.nonzero(as_tuple=False).flatten()
        valid_states = flat_states.index_select(0, valid_indices)

        # 正常训练数据至少包含 BOS/EOS；这里仍处理全 Padding 的防御性场景。
        if valid_states.shape[0] == 0:
            zero = hidden_states.new_zeros(())
            zeros = hidden_states.new_zeros(self.num_experts, dtype=torch.float32)
            # 建立零值依赖，确保 DDP 知道所有 Expert/Router 参数参与了前向图。
            parameter_link = sum(
                (parameter.sum() * 0.0 for parameter in self.parameters()),
                start=zero,
            )
            return (
                hidden_states.new_zeros(original_shape)
                + parameter_link.to(hidden_states.dtype),
                zero + parameter_link,
                zeros,
                zeros.clone(),
                zero.detach().float(),
                zero.detach().float(),
            )

        # Router 概率固定在 float32 中计算，降低混合精度下的数值风险。
        router_logits = self.gate(valid_states)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_indices = torch.topk(
            router_probs,
            k=self.num_experts_per_tok,
            dim=-1,
            sorted=False,
        )
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-20)

        routed_output = torch.zeros_like(valid_states)
        unused_parameter_link = routed_output.new_zeros(())
        for expert_index, expert in enumerate(self.experts):
            token_indices, selected_slots = torch.where(
                topk_indices == expert_index
            )
            if token_indices.numel() > 0:
                expert_output = expert(valid_states.index_select(0, token_indices))
                weights = topk_weights[token_indices, selected_slots]
                weighted_output = expert_output * weights.to(
                    expert_output.dtype
                ).unsqueeze(-1)
                # AMP 下 residual/Router 路径可能保持 float32，而 Linear Expert
                # 输出 bfloat16/float16。index_add 不会自动做类型提升，因此先转成
                # 累加缓冲区的 dtype；这也让路由结果在 float32 缓冲区中
                # 稳定累加。
                weighted_output = weighted_output.to(routed_output.dtype)
                routed_output = routed_output.index_add(
                    0, token_indices, weighted_output
                )
            elif self.training:
                # Top-1 小 batch 可能没有 Token 选择某个 Expert。零值依赖让
                # DDP 仍为这些参数生成零梯度，避免 unused parameter 报错。
                unused_parameter_link = unused_parameter_link + sum(
                    (parameter.sum() * 0.0 for parameter in expert.parameters()),
                    start=routed_output.new_zeros(()),
                )
        routed_output = routed_output + unused_parameter_link.to(routed_output.dtype)

        flat_output = torch.zeros_like(flat_states).index_copy(
            0, valid_indices, routed_output
        )

        if not self.training and self.router_aux_loss_coef == 0.0:
            # Router balancing statistics are training/monitoring signals.
            # Skip one-hot load accounting and entropy on every decode step;
            # they do not affect inference logits and are relatively expensive
            # for batch-size-one CPU serving.
            zero = router_probs.new_zeros(())
            zeros = router_probs.new_zeros(self.num_experts)
            return (
                flat_output.reshape(original_shape),
                zero,
                zeros,
                zeros.clone(),
                zero,
                router_probs.new_tensor(float(valid_states.shape[0])),
            )

        expert_mask = F.one_hot(
            topk_indices, num_classes=self.num_experts
        ).float()
        expert_load = expert_mask.mean(dim=0)
        mean_router_prob = router_probs.mean(dim=0)
        raw_aux_loss = (
            expert_load * mean_router_prob.unsqueeze(0)
        ).sum() * self.num_experts
        router_aux_loss = raw_aux_loss * self.router_aux_loss_coef

        expert_counts = expert_mask.sum(dim=(0, 1)).detach()
        router_prob_sums = router_probs.detach().sum(dim=0)
        router_entropy_sum = (
            -(router_probs * router_probs.clamp_min(1e-20).log()).sum(dim=-1)
            .detach()
            .sum()
        )
        routed_token_count = router_probs.new_tensor(
            float(valid_states.shape[0])
        ).detach()
        return (
            flat_output.reshape(original_shape),
            router_aux_loss,
            expert_counts,
            router_prob_sums,
            router_entropy_sum,
            routed_token_count,
        )


class MiniLLMDecoderLayer(nn.Module):
    """一个 Pre-Norm Transformer Decoder 层：注意力块 + MLP 块。"""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.input_layernorm = MiniLLMRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.self_attn = MiniLLMAttention(config)
        self.post_attention_layernorm = MiniLLMRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.use_moe = config.use_moe
        self.mlp = MiniLLMSparseMoE(config) if config.use_moe else MiniLLMMLP(config)
        self.hidden_dropout = config.hidden_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        PastKeyValue | None,
    ]:
        """依次执行自注意力和前馈网络，并在两个子层外加入残差连接。"""

        token_mask = (
            attention_mask[:, -hidden_states.shape[1] :]
            if attention_mask is not None
            else None
        )
        # 注意力子层：x = x + Attention(RMSNorm(x))。
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + F.dropout(
            hidden_states, p=self.hidden_dropout, training=self.training
        )
        if token_mask is not None:
            hidden_states = hidden_states.masked_fill(
                ~token_mask.to(torch.bool).unsqueeze(-1), 0.0
            )

        # MLP 子层：x = x + MLP(RMSNorm(x))。
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.use_moe:
            (
                hidden_states,
                router_aux_loss,
                expert_counts,
                router_prob_sums,
                router_entropy_sum,
                routed_token_count,
            ) = self.mlp(hidden_states, token_mask=token_mask)
        else:
            hidden_states = self.mlp(hidden_states)
            router_aux_loss = hidden_states.new_zeros(())
            expert_counts = hidden_states.new_zeros(0, dtype=torch.float32)
            router_prob_sums = hidden_states.new_zeros(0, dtype=torch.float32)
            router_entropy_sum = hidden_states.new_zeros((), dtype=torch.float32)
            routed_token_count = hidden_states.new_zeros((), dtype=torch.float32)
        hidden_states = residual + F.dropout(
            hidden_states, p=self.hidden_dropout, training=self.training
        )
        if token_mask is not None:
            hidden_states = hidden_states.masked_fill(
                ~token_mask.to(torch.bool).unsqueeze(-1), 0.0
            )
        return (
            hidden_states,
            router_aux_loss,
            expert_counts,
            router_prob_sums,
            router_entropy_sum,
            routed_token_count,
            present_key_value,
        )


class MiniLLMPreTrainedModel(PreTrainedModel):
    """miniLLM 模型公共父类，声明配置类型、初始化规则和框架能力。"""

    config_class = MiniLLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniLLMDecoderLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        """初始化线性层与词嵌入；PAD 行仅在初始化时置零。"""

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
            if module.padding_idx is not None:
                # 共享 LM Head 后，输出侧梯度仍可能更新这一行；forward
                # 必须依据 attention_mask 清零 Padding hidden state。
                module.weight.data[module.padding_idx].zero_()


class MiniLLMModel(MiniLLMPreTrainedModel):
    """不含 LM Head 的基础 Decoder，输出每个 token 的隐藏状态。"""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__(config)
        # 把离散 token id 映射到 hidden_size 维连续向量。
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        # 所有 Decoder Layer 结构相同，但每层拥有各自独立参数。
        self.layers = nn.ModuleList(
            [MiniLLMDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = MiniLLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Hugging Face 钩子：调用上面定义的 _init_weights，并处理权重共享。
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """供 Hugging Face 通用接口获取输入词嵌入层。"""

        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """供扩充词表等场景替换输入词嵌入层。"""

        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> MiniLLMModelOutput | tuple[torch.Tensor, ...]:
        """把 [batch, sequence] 的 token id 编码成上下文隐藏状态。"""

        return_dict = (
            self.config.return_dict if return_dict is None else return_dict
        )
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            if input_ids.ndim != 2:
                raise ValueError("input_ids must have shape [batch, sequence]")
            hidden_states = self.embed_tokens(input_ids)
        else:
            if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.hidden_size:
                raise ValueError("inputs_embeds must have shape [batch, sequence, hidden_size]")
            hidden_states = inputs_embeds
        batch, sequence = hidden_states.shape[:2]
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if self.gradient_checkpointing and self.training and use_cache:
            raise ValueError("KV cache is incompatible with gradient checkpointing")

        if past_key_values is None:
            layer_past: tuple[PastKeyValue | None, ...] = (None,) * len(self.layers)
            past_length = 0
        else:
            if len(past_key_values) != len(self.layers):
                raise ValueError(
                    "past_key_values must contain one key/value pair per decoder layer"
                )
            layer_past = past_key_values
            past_length = past_key_values[0][0].shape[-2]
            if any(item[0].shape[-2] != past_length for item in past_key_values):
                raise ValueError("all cached decoder layers must have the same length")

        total_length = past_length + sequence
        if total_length > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {total_length} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch, total_length),
                dtype=torch.long,
                device=hidden_states.device,
            )
        elif attention_mask.shape != (batch, total_length):
            raise ValueError(
                "attention_mask must cover cached and current tokens: "
                f"expected {(batch, total_length)}, got {tuple(attention_mask.shape)}"
            )
        if position_ids is None:
            # 用 attention_mask 累加生成位置编号，使左 padding 的生成输入也能得到
            # 正确位置；训练数据通常使用右 padding。
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            position_ids = position_ids[:, -sequence:]
        elif position_ids.shape != (batch, sequence):
            raise ValueError(
                f"position_ids must have shape {(batch, sequence)}, "
                f"got {tuple(position_ids.shape)}"
            )

        # Input/output weight tying lets LM-head gradients update the PAD row,
        # even though Embedding.padding_idx blocks input-side gradients. Never
        # assume the loaded PAD embedding is zero: mask it explicitly before
        # the first decoder layer and again after every residual block.
        token_mask = attention_mask[:, -sequence:]
        hidden_states = hidden_states.masked_fill(
            ~token_mask.to(torch.bool).unsqueeze(-1), 0.0
        )
        router_aux_losses: list[torch.Tensor] = []
        expert_counts = hidden_states.new_zeros(
            self.config.num_experts, dtype=torch.float32
        )
        router_prob_sums = hidden_states.new_zeros(
            self.config.num_experts, dtype=torch.float32
        )
        router_entropy_sum = hidden_states.new_zeros((), dtype=torch.float32)
        routed_token_count = hidden_states.new_zeros((), dtype=torch.float32)
        next_cache: list[PastKeyValue] = []
        for layer_index, decoder_layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                # 前向时不保留层内激活，反向时重新计算，以计算量换显存。
                layer_outputs = checkpoint(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=layer_past[layer_index],
                    use_cache=use_cache,
                )
            (
                hidden_states,
                layer_aux_loss,
                layer_expert_counts,
                layer_router_prob_sums,
                layer_router_entropy_sum,
                layer_routed_token_count,
                present_key_value,
            ) = layer_outputs
            if use_cache:
                if present_key_value is None:
                    raise RuntimeError("decoder layer did not return its KV cache")
                next_cache.append(present_key_value)
            if self.config.use_moe:
                router_aux_losses.append(layer_aux_loss)
                expert_counts += layer_expert_counts
                router_prob_sums += layer_router_prob_sums
                router_entropy_sum += layer_router_entropy_sum
                routed_token_count += layer_routed_token_count
        hidden_states = self.norm(hidden_states)

        router_aux_loss = (
            torch.stack(router_aux_losses).sum()
            if router_aux_losses
            else hidden_states.new_zeros(())
        )

        if not return_dict:
            output: tuple[Any, ...] = (hidden_states,)
            if use_cache:
                output += (tuple(next_cache),)
            return output
        return MiniLLMModelOutput(
            last_hidden_state=hidden_states,
            router_aux_loss=router_aux_loss,
            expert_counts=expert_counts if self.config.use_moe else None,
            router_prob_sums=(router_prob_sums if self.config.use_moe else None),
            router_entropy_sum=(
                router_entropy_sum if self.config.use_moe else None
            ),
            routed_token_count=(routed_token_count if self.config.use_moe else None),
            past_key_values=tuple(next_cache) if use_cache else None,
        )


class MiniLLMForCausalLM(MiniLLMPreTrainedModel, GenerationMixin):
    """基础 Decoder + 词表投影层，用于下一个 Token 预测和文本生成。"""

    # 输入 Embedding 与输出 LM Head 共享权重。Transformers 5.x 要求用
    # {待共享权重: 权重来源} 的映射；dict 在 Transformers 4.x 中也可按键迭代。
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__(config)
        self.model = MiniLLMModel(config)
        # 把每个位置的隐藏状态投影为 vocab_size 个未归一化分数（logits）。
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """返回 Decoder 使用的 Token Embedding。"""

        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """替换 Decoder 的 Token Embedding。"""

        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        """返回将隐藏状态映射到词表的 LM Head。"""

        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        """替换输出 LM Head。"""

        self.lm_head = value

    def set_decoder(self, decoder: MiniLLMModel) -> None:
        """替换底层 Decoder，兼容 Hugging Face GenerationMixin 接口。"""

        self.model = decoder

    def get_decoder(self) -> MiniLLMModel:
        """返回底层 Decoder。"""

        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        labels: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> MiniLLMCausalLMOutput | tuple[torch.Tensor, ...]:
        """计算词表 logits；提供 labels 时额外计算下一 Token 预测损失。"""

        return_dict = (
            self.config.return_dict if return_dict is None else return_dict
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        # Generation only needs the newest token's logits. Avoid projecting the
        # complete prompt through the vocabulary head during prefill.
        logits_hidden_states = outputs.last_hidden_state
        if labels is None:
            if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                logits_hidden_states = logits_hidden_states[:, -logits_to_keep:, :]
            elif isinstance(logits_to_keep, torch.Tensor):
                logits_hidden_states = logits_hidden_states[:, logits_to_keep, :]
        # logits 形状为 [batch, kept_sequence, vocab_size]。
        logits = self.lm_head(logits_hidden_states)

        lm_loss = None
        loss = None
        if labels is not None:
            # 因果语言模型目标：位置 t 的输出预测位置 t+1 的真实 Token。
            # 因此删除 logits 的最后一位和 labels 的第一位，再让二者对齐。
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            # labels 中的 -100 一般对应 padding；ignore_index 使其不计入 loss。
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # MiniMind 风格：总损失 = 下一个 Token 损失 + 各层 Router 辅助损失。
            loss = lm_loss + outputs.router_aux_loss

        if not return_dict:
            output = (logits,)
            if outputs.past_key_values is not None:
                output += (outputs.past_key_values,)
            return ((loss,) + output) if loss is not None else output
        return MiniLLMCausalLMOutput(
            loss=loss,
            logits=logits,
            lm_loss=lm_loss,
            router_aux_loss=outputs.router_aux_loss,
            expert_counts=outputs.expert_counts,
            router_prob_sums=outputs.router_prob_sums,
            router_entropy_sum=outputs.router_entropy_sum,
            routed_token_count=outputs.routed_token_count,
            past_key_values=outputs.past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """为 ``model.generate()`` 整理每一步需要传入 forward 的参数。"""

        del kwargs
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:]
        else:
            past_key_values = None
        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            position_ids = position_ids[:, -input_ids.shape[1] :]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": self.config.use_cache if use_cache is None else use_cache,
        }

    @staticmethod
    def _reorder_cache(
        past_key_values: PastKeyValues,
        beam_idx: torch.LongTensor,
    ) -> PastKeyValues:
        """Reorder legacy tuple caches for beam-search generation."""

        return tuple(
            tuple(state.index_select(0, beam_idx.to(state.device)) for state in layer)
            for layer in past_key_values
        )

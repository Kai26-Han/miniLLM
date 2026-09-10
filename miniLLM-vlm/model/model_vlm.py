"""MiniMind-V 风格：冻结 SigLIP / miniLLM，只训练两层 MLP。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from minillm_vlm.model.model_minillm import MiniLLMForCausalLM


@dataclass(frozen=True)
class VLMConfig:
    image_token_id: int
    image_token: str
    image_hidden_size: int
    image_token_len: int
    hidden_size: int
    image_size: int
    format_version: int = 1

    def to_dict(self):
        return asdict(self)


class VisionProjector(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim),
                         nn.GELU(), nn.Linear(out_dim, out_dim))


class MiniLLMVLM(nn.Module):
    def __init__(self, llm: MiniLLMForCausalLM, vision_encoder: nn.Module,
                 config: VLMConfig):
        super().__init__()
        self.llm = llm
        self.vision_encoder = vision_encoder
        self.config = config
        self.projector = VisionProjector(config.image_hidden_size, config.hidden_size)
        self.llm.requires_grad_(False)
        self.vision_encoder.requires_grad_(False)
        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        # 参数冻结与 eval 是两回事；固定编码器 dropout，保留 LLM 输入梯度。
        self.vision_encoder.eval()
        self.llm.eval()
        return self

    def image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        expected = (pixel_values.shape[0], self.config.image_token_len,
                    self.config.image_hidden_size)
        if tuple(features.shape) != expected:
            raise ValueError(f"Vision feature shape {tuple(features.shape)} != {expected}")
        return features

    def forward(self, input_ids: torch.Tensor, attention_mask=None, labels=None,
                pixel_values=None, past_key_values=None, use_cache=False,
                logits_to_keep=0):
        marker = input_ids.eq(self.config.image_token_id)
        prefill = past_key_values is None
        if prefill:
            if pixel_values is None:
                raise ValueError("VLM prefill requires pixel_values")
            if not torch.all(marker.sum(-1) == self.config.image_token_len):
                raise ValueError("Each sample must contain exactly one complete image block")
            for positions in marker.nonzero(as_tuple=False).split(self.config.image_token_len):
                if not torch.all(positions[1:, 1] - positions[:-1, 1] == 1):
                    raise ValueError("Image token positions must be contiguous")
            if attention_mask is not None and not torch.all(attention_mask[marker].bool()):
                raise ValueError("Image positions must have attention_mask=1")
            if labels is not None and not torch.all(labels[marker] == -100):
                raise ValueError("Image labels must be -100")
            embeddings = self.llm.get_input_embeddings()(input_ids)
            visual = self.projector(self.image_features(pixel_values))
            # masked_scatter 可微，不能 detach Projector 输出或包住 LLM no_grad。
            embeddings = embeddings.masked_scatter(
                marker.unsqueeze(-1).expand_as(embeddings), visual.to(embeddings.dtype)
            )
            return self.llm(inputs_embeds=embeddings, attention_mask=attention_mask,
                            labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, return_dict=True)
        if marker.any():
            raise ValueError("Cached decoding cannot introduce a new image block")
        return self.llm(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels, past_key_values=past_key_values,
                        use_cache=use_cache, logits_to_keep=logits_to_keep,
                        return_dict=True)

    @torch.no_grad()
    def generate_caption(self, input_ids, pixel_values, max_new_tokens=128):
        """单图贪心解码；只在 prefill 编码图片，后续使用原 miniLLM KV cache。"""
        if input_ids.shape[0] != 1 or max_new_tokens < 1:
            raise ValueError("Generation requires batch=1 and max_new_tokens >= 1")
        if input_ids.shape[1] + max_new_tokens > self.llm.config.max_position_embeddings:
            raise ValueError("Prompt + max_new_tokens exceeds model position limit")
        was_training = self.training
        self.eval()
        try:
            attention_mask = torch.ones_like(input_ids)
            output = self(input_ids, attention_mask=attention_mask,
                          pixel_values=pixel_values, use_cache=True, logits_to_keep=1)
            generated = []
            for _ in range(max_new_tokens):
                logits = output.logits[:, -1].float().clone()
                logits[:, self.config.image_token_id] = -torch.inf
                logits[:, self.llm.config.pad_token_id] = -torch.inf
                token = logits.argmax(-1, keepdim=True)
                generated.append(token)
                if token.item() == self.llm.config.eos_token_id:
                    break
                attention_mask = torch.cat((attention_mask, torch.ones_like(token)), dim=1)
                output = self(token, attention_mask=attention_mask,
                              past_key_values=output.past_key_values,
                              use_cache=True, logits_to_keep=1)
            return torch.cat(generated, dim=1)
        finally:
            self.train(was_training)


class MiniLLMSFT(MiniLLMVLM):
    """单图/纯文本 SFT；冻结 SigLIP，可选首尾语言层或全部语言层。"""
    def __init__(self, llm, vision_encoder, config, freeze_llm=1):
        super().__init__(llm, vision_encoder, config)
        if freeze_llm not in (0, 1, 2):
            raise ValueError("freeze_llm must be 0 (full), 1 (first/last), or 2 (projector)")
        self.freeze_llm = freeze_llm
        if freeze_llm == 0:
            self.llm.requires_grad_(True)
        elif freeze_llm == 1:
            self.llm.model.layers[0].requires_grad_(True)
            self.llm.model.layers[-1].requires_grad_(True)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        policy = getattr(self, "freeze_llm", 2)
        if policy == 0:
            self.llm.train(mode)
        elif policy == 1:
            self.llm.model.layers[0].train(mode)
            self.llm.model.layers[-1].train(mode)
        return self

    def forward(self, input_ids, attention_mask=None, labels=None, pixel_values=None,
                has_image=None, past_key_values=None, use_cache=False, logits_to_keep=0):
        if past_key_values is not None:
            return super().forward(input_ids, attention_mask, labels, past_key_values=past_key_values,
                                   use_cache=use_cache, logits_to_keep=logits_to_keep)
        marker = input_ids.eq(self.config.image_token_id)
        counts = marker.sum(-1)
        inferred = counts.ne(0)
        if has_image is not None and (has_image.shape != inferred.shape or
                                     not torch.equal(has_image.bool(), inferred)):
            raise ValueError("has_image disagrees with the image blocks")
        if not torch.all((counts == 0) | (counts == self.config.image_token_len)):
            raise ValueError("Each visual sample requires one complete image block")
        for row in marker[inferred]:
            positions = row.nonzero().flatten()
            if not torch.all(positions[1:] - positions[:-1] == 1):
                raise ValueError("Image token positions must be contiguous")
        if attention_mask is not None and not torch.all(attention_mask[marker].bool()):
            raise ValueError("Image attention mask must be 1")
        if labels is not None and not torch.all(labels[marker] == -100):
            raise ValueError("Image labels must be -100")
        embeddings = self.llm.get_input_embeddings()(input_ids)
        if inferred.any():
            if pixel_values is None or pixel_values.shape[0] != input_ids.shape[0]:
                raise ValueError("Visual samples require batch-aligned pixel_values")
            visual = self.projector(self.image_features(pixel_values[inferred]))
            embeddings = embeddings.masked_scatter(marker.unsqueeze(-1).expand_as(embeddings),
                                                    visual.to(embeddings.dtype))
        return self.llm(inputs_embeds=embeddings, attention_mask=attention_mask, labels=labels,
                        use_cache=use_cache, logits_to_keep=logits_to_keep, return_dict=True)

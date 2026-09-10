"""Parquet 图文数据：离线索引、按图片划分、只监督 assistant。"""
from __future__ import annotations

import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def image_bytes_from_row(row):
    value = row.get("image_bytes")
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Pretrain accepts exactly one image per sample")
        value = value[0]
    if not isinstance(value, (bytes, bytearray)) or not value:
        raise ValueError("image_bytes must contain nonempty image bytes")
    return bytes(value)


def decode_image(blob):
    with Image.open(io.BytesIO(blob)) as image:
        image.load()  # 在文件关闭前校验完整解码。
        return ImageOps.exif_transpose(image).convert("RGB")


def read_conversations(row):
    conversations = row.get("conversations")
    if isinstance(conversations, str):
        conversations = json.loads(conversations)
    if not isinstance(conversations, list) or not all(isinstance(turn, dict) for turn in conversations):
        raise ValueError("conversations must be a list or JSON list string")
    expected = ["user", "assistant"]
    if conversations and conversations[0].get("role") == "system":
        expected.insert(0, "system")
    if [turn.get("role") for turn in conversations] != expected:
        raise ValueError("Expected optional system, one user and one assistant")
    for turn in conversations:
        if not isinstance(turn.get("content"), str) or not turn["content"].strip():
            raise ValueError("Empty or non-string conversation content")
        if any(turn.get(key) for key in ("tools", "tool_calls", "functions")):
            raise ValueError("Tool conversations are outside caption pretraining")
    if sum(turn["content"].count("<image>") for turn in conversations) != 1:
        raise ValueError("Expected exactly one <image> marker")
    if conversations[-2]["content"].count("<image>") != 1:
        raise ValueError("<image> must occur in the user message")
    return conversations


class CaptionEncoder:
    def __init__(self, tokenizer, config, max_length=512):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        if not tokenizer.chat_template:
            raise ValueError("Base tokenizer must include its chat_template")
        if tokenizer.encode(config.image_token, add_special_tokens=False) != [config.image_token_id]:
            raise ValueError("Image marker is not a single existing tokenizer token")
        if (tokenizer.bos_token, tokenizer.eos_token) != ("<|im_start|>", "<|im_end|>"):
            raise ValueError("This pretrain adapter requires miniLLM ChatML BOS/EOS")

    def prompt_ids(self, messages):
        converted = []
        for turn in messages:
            content = turn["content"]
            if any(special in content for special in
                   (self.config.image_token, self.tokenizer.bos_token, self.tokenizer.eos_token)):
                raise ValueError("Conversation contains a reserved control token")
            content = content.replace("<image>", self.config.image_token * self.config.image_token_len)
            converted.append({"role": turn["role"], "content": content})
        # Base 配套 tokenizer 的 generation prompt 可能额外插入空 think 标签。
        prompt = self.tokenizer.apply_chat_template(
            converted, tokenize=False, add_generation_prompt=False
        ) + self.tokenizer.bos_token + "assistant\n"
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        positions = [i for i, token in enumerate(ids) if token == self.config.image_token_id]
        if len(positions) != self.config.image_token_len or positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError("Prompt must contain exactly one complete image block")
        return ids

    def encode(self, conversations):
        prompt = self.prompt_ids(conversations[:-1])
        answer = conversations[-1]["content"].strip()
        if any(token in answer for token in
               (self.config.image_token, self.tokenizer.bos_token, self.tokenizer.eos_token)):
            raise ValueError("Answer contains reserved control tokens")
        completion = self.tokenizer.encode(answer, add_special_tokens=False)
        if not completion:
            raise ValueError("Answer tokenization is empty")
        original_length = len(prompt) + len(completion) + 1
        room = self.max_length - len(prompt)
        if room < 1:
            raise ValueError("Prompt exceeds sequence budget; refusing to truncate image/prompt")
        # 只在真实结尾监督 EOS；截断回答时不制造虚假的结束标签。
        completion = (completion + [self.tokenizer.eos_token_id])[:room]
        ids = prompt + completion
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor([-100] * len(prompt) + completion, dtype=torch.long),
            "original_length": original_length,
            "truncated": original_length > self.max_length,
            "prompt_length": len(prompt),
        }


def load_parquet(paths, cache_dir):
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Parquet data not found: {path}")
    source = load_dataset("parquet", data_files=[str(p) for p in paths], split="train",
                          cache_dir=str(cache_dir), keep_in_memory=False)
    if not {"conversations", "image_bytes"}.issubset(source.column_names):
        raise ValueError("Parquet requires conversations and image_bytes columns")
    return source


def validation_image(image_key, seed, val_ratio):
    digest = hashlib.sha256(f"{seed}:{image_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < val_ratio


def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_index(source, encoder, path, signature, seed=42, val_ratio=0.01):
    """只扫描一次；索引中不保存图像副本。签名涵盖数据、词表、模板和长度。"""
    path = Path(path)
    spec = {"signature": signature, "seed": seed, "val_ratio": val_ratio,
            "max_length": encoder.max_length, "vlm_config": encoder.config.to_dict(),
            "index_version": 1}
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("spec") != spec:
            raise ValueError(f"Index configuration changed: use a new --index-path ({path})")
        return cached
    train, validation = [], []
    stats = Counter(total=len(source))
    errors = []
    # Arrow memory-map 随机访问；iter(batch_size) 避免一次读入全部图片。
    offset = 0
    for batch in source.iter(batch_size=128):
        for local_index in range(len(batch["image_bytes"])):
            row_index = offset + local_index
            row = {key: values[local_index] for key, values in batch.items()}
            try:
                blob = image_bytes_from_row(row)
                decode_image(blob)
                conversations = read_conversations(row)
                sample = encoder.encode(conversations)
            except (ValueError, TypeError, KeyError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
                stats["invalid"] += 1
                if len(errors) < 20:
                    errors.append({"row": row_index, "error": str(exc)})
                continue
            image_key = hashlib.sha256(blob).hexdigest()
            is_val = validation_image(image_key, seed, val_ratio)
            (validation if is_val else train).append(row_index)
            stats["valid"] += 1
            stats["truncated"] += int(sample["truncated"])
            stats["original_tokens"] += sample["original_length"]
            stats["supervised_tokens"] += int((sample["labels"] != -100).sum())
            stats["max_original_length"] = max(stats["max_original_length"], sample["original_length"])
            length = sample["original_length"]
            stats["length_0_512" if length <= 512 else "length_513_768" if length <= 768 else "length_over_768"] += 1
            answer = conversations[-1]["content"]
            stats["contains_chinese" if any("\u4e00" <= c <= "\u9fff" for c in answer) else "other_language"] += 1
        offset += len(batch["image_bytes"])
        if offset % 12800 == 0:
            print(f"Index scan: {offset}/{len(source)}, valid={stats['valid']}, invalid={stats['invalid']}", flush=True)
    result = {"spec": spec, "stats": dict(stats), "invalid_examples": errors,
              "train": train, "validation": validation}
    atomic_json(result, path)
    return result


class VLMDataset(Dataset):
    def __init__(self, source, indices, encoder, processor):
        self.source, self.indices = source, indices
        self.encoder, self.processor = encoder, processor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        row = self.source[source_index]
        sample = self.encoder.encode(read_conversations(row))
        image = decode_image(image_bytes_from_row(row))
        sample["pixel_values"] = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        sample["row_index"] = source_index
        return sample


class VLMCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, samples):
        length = max(len(sample["input_ids"]) for sample in samples)
        ids = torch.full((len(samples), length), self.pad_token_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        for i, sample in enumerate(samples):
            n = len(sample["input_ids"])
            ids[i, :n] = sample["input_ids"]
            labels[i, :n] = sample["labels"]
            mask[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask,
                "pixel_values": torch.stack([s["pixel_values"] for s in samples])}


class EpochBatchSampler:
    """随机顺序由 epoch 独立决定，直接跳过已提交的 batch，不重读图片。"""
    def __init__(self, size, batch_size, seed, epoch, start_batch=0):
        self.size, self.batch_size = size, batch_size
        self.seed, self.epoch, self.start_batch = seed, epoch, start_batch

    def __len__(self):
        return max(0, (self.size + self.batch_size - 1) // self.batch_size - self.start_batch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        for i in range(self.start_batch * self.batch_size, self.size, self.batch_size):
            yield order[i:i + self.batch_size]


SFT_TYPES = ("image", "instruction", "caption", "text")


def read_sft_conversations(row):
    """按标记/显式任务字段识别文本，不根据黑色像素猜测样本类型。"""
    turns = row.get("conversations")
    if isinstance(turns, str):
        turns = json.loads(turns)
    if not isinstance(turns, list) or not turns:
        raise ValueError("SFT conversations must be a nonempty list")
    conversations = []
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("Conversation turn must be an object")
        role = turn.get("role", turn.get("from"))
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        content = turn.get("content", turn.get("value"))
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty or non-string conversation content")
        if any(turn.get(key) for key in ("tools", "tool_calls", "functions")):
            raise ValueError("SFT tool conversations are unsupported")
        conversations.append({"role": role, "content": content})
    start = int(conversations[0]["role"] == "system")
    roles = [turn["role"] for turn in conversations[start:]]
    if not roles or len(roles) % 2 or roles != ["user", "assistant"] * (len(roles) // 2):
        raise ValueError("Expected optional system followed by complete user/assistant pairs")
    task = row.get("task_type")
    if task == "text":
        # 只接受数据明确标为 text 的占位图样本；不检查/编码占位图片。
        for turn in conversations:
            if turn["role"] == "user":
                turn["content"] = turn["content"].replace("<image>", "").strip()
                if not turn["content"]:
                    raise ValueError("Text task has an empty question after removing image placeholder")
    markers = sum(turn["content"].count("<image>") for turn in conversations)
    if markers not in (0, 1) or (markers and conversations[start]["content"].count("<image>") != 1):
        raise ValueError("Single image marker must occur in the first user turn")
    kind = (task if task in ("instruction", "caption") else "image") if markers else "text"
    if task in ("instruction", "caption") and not markers:
        raise ValueError("Explicit visual task is missing <image>")
    return conversations, kind


class SFTEncoder(CaptionEncoder):
    """完整 ChatML；所有 assistant 正文及真实 EOS 监督，保留首图上下文。"""
    def encode(self, conversations):
        text, spans, pairs = "", [], []
        pair_start = None
        for turn in conversations:
            content = turn["content"].strip() if turn["role"] == "assistant" else turn["content"]
            if any(special in content for special in
                   (self.config.image_token, self.tokenizer.bos_token, self.tokenizer.eos_token,
                    self.tokenizer.pad_token)):
                raise ValueError("Conversation contains reserved control tokens")
            content = content.replace("<image>", self.config.image_token * self.config.image_token_len)
            if turn["role"] == "user":
                pair_start = len(text)
            text += self.tokenizer.bos_token + turn["role"] + "\n"
            start = len(text)
            text += content + self.tokenizer.eos_token
            if turn["role"] == "assistant":
                spans.append((start, len(text)))
                pairs.append((pair_start, start))
            text += "\n"
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        labels = [-100] * len(ids)
        for i, (start, end) in enumerate(offsets):
            if end > start and any(start >= a and end <= b for a, b in spans):
                labels[i] = ids[i]
        limit = min(len(ids), self.max_length)
        # 若预算用在下一轮 user/header 中，整轮舍弃；不截断问题或图像。
        for pair_start, answer_start in pairs:
            pair_token = next((i for i, (_, end) in enumerate(offsets) if end > pair_start), len(ids))
            answer_token = next((i for i, (_, end) in enumerate(offsets) if end > answer_start), len(ids))
            if pair_token < limit <= answer_token:
                limit = pair_token
                break
        if not any(label != -100 for label in labels[1:limit]):
            raise ValueError("No supervised answer fits without truncating the image/question")
        kept_ids, kept_labels = ids[:limit], labels[:limit]
        markers = [i for i, token in enumerate(kept_ids) if token == self.config.image_token_id]
        expected = sum(turn["content"].count("<image>") for turn in conversations) * self.config.image_token_len
        if len(markers) != expected or (markers and markers != list(range(markers[0], markers[-1] + 1))):
            raise ValueError("Truncation would remove or split the image block")
        for i in markers:
            kept_labels[i] = -100
        return {"input_ids": torch.tensor(kept_ids), "labels": torch.tensor(kept_labels),
                "original_length": len(ids), "truncated": limit < len(ids),
                "prompt_length": next(i for i, label in enumerate(kept_labels) if label != -100)}

    def prompt_ids(self, messages):
        # 复用相同序列化；最后一个 user 后添加 assistant header，供文本/多轮推理使用。
        converted = []
        for turn in messages:
            content = turn["content"]
            if any(token in content for token in (self.config.image_token,
                                                  self.tokenizer.bos_token, self.tokenizer.eos_token)):
                raise ValueError("Prompt contains reserved control tokens")
            converted.append(self.tokenizer.bos_token + turn["role"] + "\n" +
                             content.replace("<image>", self.config.image_token * self.config.image_token_len) +
                             self.tokenizer.eos_token + "\n")
        text = "".join(converted) + self.tokenizer.bos_token + "assistant\n"
        return self.tokenizer.encode(text, add_special_tokens=False)


class SFTDataset(VLMDataset):
    def __getitem__(self, index):
        source_index = self.indices[index]
        row = self.source[source_index]
        turns, kind = read_sft_conversations(row)
        sample = self.encoder.encode(turns)
        visual = kind != "text"
        sample["pixel_values"] = (self.processor(images=decode_image(image_bytes_from_row(row)),
                                                return_tensors="pt")["pixel_values"][0] if visual else
                                  torch.zeros(3, self.encoder.config.image_size, self.encoder.config.image_size))
        sample.update(row_index=source_index, has_image=visual, sample_type=SFT_TYPES.index(kind))
        return sample


class SFTCollator(VLMCollator):
    def __init__(self, pad_token_id, metadata=False):
        super().__init__(pad_token_id)
        self.metadata = metadata

    def __call__(self, samples):
        batch = super().__call__(samples)
        batch["has_image"] = torch.tensor([sample["has_image"] for sample in samples], dtype=torch.bool)
        if self.metadata:
            batch["sample_type"] = torch.tensor([sample["sample_type"] for sample in samples])
        return batch


def prepare_sft_index(source, encoder, path, signature, seed=42, val_ratio=.01, scan_samples=0):
    """全量索引每 12800 行提交进度；短跑可固定抽样，使用独立索引。"""
    import random
    import time
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    spec = {"signature": signature, "seed": seed, "val_ratio": val_ratio,
            "max_length": encoder.max_length, "vlm_config": encoder.config.to_dict(),
            "scan_samples": scan_samples, "index_version": "sft-1"}
    state = None
    for candidate in (path, partial):
        if candidate.is_file():
            cached = json.loads(candidate.read_text(encoding="utf-8"))
            if cached.get("spec") != spec:
                raise ValueError(f"SFT index configuration changed: use a new --index-path ({path})")
            if candidate == path:
                print(f"Reusing SFT index: {path}", flush=True)
                return cached
            state = cached
    selected = sorted(random.Random(seed).sample(range(len(source)), min(scan_samples, len(source)))) if scan_samples else None
    total = len(selected) if selected is not None else len(source)
    state = state or {"spec": spec, "train": [], "validation": [], "stats": {},
                      "invalid_examples": [], "next_offset": 0}
    stats = Counter(state["stats"])
    stats["total"] = total
    offset = state["next_offset"]
    view = source.select(selected[offset:] if selected is not None else range(offset, total))
    started = time.monotonic()
    starting_offset = offset
    for batch in view.iter(batch_size=128):
        count = len(batch["conversations"])
        for j in range(count):
            row_index = selected[offset + j] if selected is not None else offset + j
            row = {key: values[j] for key, values in batch.items()}
            try:
                turns, kind = read_sft_conversations(row)
                encoded = encoder.encode(turns)
                if kind == "text":
                    key = "text:" + hashlib.sha256(json.dumps(turns, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                else:
                    blob = image_bytes_from_row(row)
                    decode_image(blob)
                    key = hashlib.sha256(blob).hexdigest()  # 与 Pretrain 完全相同的图片划分
            except (ValueError, TypeError, KeyError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
                stats["invalid"] += 1
                if len(state["invalid_examples"]) < 20:
                    state["invalid_examples"].append({"row": row_index, "error": str(exc)})
                continue
            split = "validation" if validation_image(key, seed, val_ratio) else "train"
            state[split].append(row_index)
            stats["valid"] += 1
            stats[kind] += 1
            stats[f"{split}/{kind}"] += 1
            stats["truncated"] += int(encoded["truncated"])
            stats["supervised_tokens"] += int((encoded["labels"] != -100).sum())
            stats["max_original_length"] = max(stats["max_original_length"], encoded["original_length"])
        offset += count
        state.update(next_offset=offset, stats=dict(stats))
        if offset % 12800 == 0 or offset == total:
            atomic_json(state, partial)
            rate = (offset - starting_offset) / max(time.monotonic() - started, 1e-9)
            print(f"SFT index: {offset}/{total}, valid={stats['valid']}, invalid={stats['invalid']}, "
                  f"rows/s={rate:.1f}, ETA={(total-offset)/max(rate, 1e-9)/60:.1f} min", flush=True)
    atomic_json(state, path)
    partial.unlink(missing_ok=True)
    return state

#!/usr/bin/env python3
"""加载 Projector 产物，做单图描述或验证集正确图/错配图对照。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import minillm_vlm  # noqa: F401,E402

import torch
from PIL import Image, ImageOps

from minillm_vlm.dataset.vlm_dataset import (CaptionEncoder, VLMDataset, atomic_json,
    load_parquet, prepare_index)
from minillm_vlm.trainer.trainer_utils import (add_data_args, add_resource_args, apply_adapter,
    autocast, data_identity, load_resources, project_path, read_adapter, resolve_device, evaluate, make_loader)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_resource_args(parser)
    add_data_args(parser)
    parser.set_defaults(base_model=None, vision_model=None, image_token=None, attention_backend=None)
    artifacts = parser.add_mutually_exclusive_group(required=True)
    artifacts.add_argument("--adapter", type=project_path)
    artifacts.add_argument("--sft-model", type=project_path, help="SFT best_sft.pt or last_sft.pt")
    parser.add_argument("--text", action="store_true", help="SFT pure-text generation")
    parser.add_argument("--messages", type=project_path, help="SFT JSON conversation ending in user; image marker inserted in first user")
    parser.add_argument("--image", type=project_path)
    parser.add_argument("--prompt", default="请详细描述这张图片。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--output", type=project_path, default=None)
    parser.add_argument("--scan-samples", type=int, default=0)
    parser.set_defaults(max_seq_len=None, data_path=None, index_path=None)
    args = parser.parse_args(argv)
    from minillm_vlm.trainer.trainer_utils import VLM_ROOT
    stage = "sft" if args.sft_model else "pretrain"
    args.max_seq_len = args.max_seq_len if args.max_seq_len is not None else (768 if args.sft_model else 512)
    args.data_path = args.data_path or [VLM_ROOT / f"dataset/{stage}_i2t.parquet"]
    args.index_path = args.index_path or VLM_ROOT / f"cache/{stage}_index.json"
    if args.scan_samples < 0:
        raise ValueError("scan-samples must be nonnegative")
    if args.batch_size < 1 or args.eval_samples < 1 or args.num_workers < 0:
        raise ValueError("Invalid batch/sample/worker settings")
    if args.sft_model:
        return evaluate_sft_artifact(args)
    if args.text or args.messages:
        raise ValueError("--text and --messages require --sft-model")
    artifact = read_adapter(args.adapter)
    for key in ("base_model", "vision_model", "tokenizer_path", "image_token", "attention_backend"):
        if getattr(args, key) is None:
            value = artifact["resources"][key]
            setattr(args, key, project_path(value) if key.endswith("model") or key == "tokenizer_path" else value)
    model, tokenizer, processor, identity, _ = load_resources(args)
    apply_adapter(model, artifact, identity)
    device, dtype = resolve_device(args.device, args.dtype)
    model.to(device).eval()
    encoder = CaptionEncoder(tokenizer, model.config, args.max_seq_len)
    if args.image:
        if not args.prompt.strip() or "<image>" in args.prompt:
            raise ValueError("Provide a nonempty text prompt without <image>; the marker is inserted automatically")
        ids = encoder.prompt_ids([{"role": "user", "content": "<image>\n" + args.prompt}])
        with Image.open(args.image) as image:
            pixels = processor(images=ImageOps.exif_transpose(image).convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
        with autocast(device, dtype):
            generated = model.generate_caption(torch.tensor([ids], device=device), pixels, args.max_new_tokens)
        result = {"image": str(args.image), "prompt": args.prompt,
                  "caption": tokenizer.decode(generated[0], skip_special_tokens=True),
                  "adapter_step": artifact["global_step"]}
    else:
        if args.max_seq_len < 1 or not 0 < args.val_ratio < .5:
            raise ValueError("Invalid max-seq-len or val-ratio")
        source = load_parquet(args.data_path, args.cache_dir / "arrow")
        signature = {"data": data_identity(args.data_path), "tokenizer": identity["tokenizer"],
                     "implementation": identity["implementation"]}
        index = prepare_index(source, encoder, args.index_path, signature, args.seed, args.val_ratio)
        indices = index["validation"]
        order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(args.seed + 1)).tolist()
        dataset = VLMDataset(source, [indices[i] for i in order[:args.eval_samples]], encoder, processor)
        result = evaluate(model, make_loader(dataset, args), device, dtype, paired=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        atomic_json(result, args.output)


def evaluate_sft_artifact(args):
    from minillm_vlm.dataset.vlm_dataset import (SFTEncoder, SFTDataset,
        read_sft_conversations, prepare_sft_index)
    from minillm_vlm.trainer.trainer_utils import (load_sft_for_eval, evaluate_sft,
        make_sft_loader, sft_implementation_identity)
    model, tokenizer, processor, artifact = load_sft_for_eval(args.sft_model, args)
    device, dtype = resolve_device(args.device, args.dtype)
    model.to(device).eval()
    encoder = SFTEncoder(tokenizer, model.config, args.max_seq_len)
    if args.text and args.image:
        raise ValueError("Choose --text or --image")
    if args.image or args.text:
        messages = (json.loads(args.messages.read_text(encoding="utf-8")) if args.messages else
                    [{"role": "user", "content": args.prompt}])
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "user":
            raise ValueError("--messages must be a JSON list ending in a user message")
        if args.image and not any("<image>" in turn.get("content", "") for turn in messages):
            first = next(turn for turn in messages if turn.get("role") == "user")
            first["content"] = "<image>\n" + first["content"]
        checked, kind = read_sft_conversations({"conversations": messages + [{"role": "assistant", "content": "check"}]})
        if (kind != "text") != bool(args.image):
            raise ValueError("Image markers do not match --image/--text")
        ids = encoder.prompt_ids(checked[:-1])
        pixels = None
        if args.image:
            with Image.open(args.image) as image:
                pixels = processor(images=ImageOps.exif_transpose(image).convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
        with autocast(device, dtype):
            generated = model.generate_caption(torch.tensor([ids], device=device), pixels, args.max_new_tokens)
        result = {"messages": messages, "answer": tokenizer.decode(generated[0], skip_special_tokens=True),
                  "sft_step": artifact["global_step"]}
    else:
        if args.messages:
            raise ValueError("--messages requires --image or --text")
        if args.max_seq_len < 1 or not 0 < args.val_ratio < .5:
            raise ValueError("Invalid max-seq-len or val-ratio")
        source = load_parquet(args.data_path, args.cache_dir / "arrow")
        signature = {"data": data_identity(args.data_path), "tokenizer": artifact["identity"]["tokenizer"],
                     "implementation": sft_implementation_identity()}
        index = prepare_sft_index(source, encoder, args.index_path, signature, args.seed, args.val_ratio, args.scan_samples)
        indices = index["validation"]
        order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(args.seed + 1)).tolist()
        dataset = SFTDataset(source, [indices[i] for i in order[:args.eval_samples]], encoder, processor)
        result = evaluate_sft(model, make_sft_loader(dataset, args, metadata=True), device, dtype)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        atomic_json(result, args.output)
    return result


if __name__ == "__main__":
    main()

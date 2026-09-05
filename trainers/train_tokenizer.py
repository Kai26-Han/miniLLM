#!/usr/bin/env python3
"""训练 miniLLM 的 ByteLevel-BPE Tokenizer。

学习时可以把本文件理解成一条从“原始文本”到“可被模型加载的词表”的流水线：

    收集输入文件
        -> 逐行解析 JSONL/TXT
        -> 过滤短文本并切分超长文本
        -> BPE 统计与合并
        -> 保存 tokenizer.json / vocab.json / merges.txt
        -> 包装为 Transformers Tokenizer
        -> 验证词表、特殊 Token 和编解码一致性

The defaults intentionally follow MiniMind's text-tokenizer conventions:

    0 -> <|endoftext|>   (padding / unknown fallback)
    1 -> <|im_start|>    (BOS / message start)
    2 -> <|im_end|>      (EOS / message end)

Example:
    python trainer/train_tokenizer.py \
        --input dataset/tokenizer_corpus \
        --output model/tokenizer

Supported input files: .jsonl, .jsonl.gz, .txt, and .txt.gz. Directories are
scanned recursively. JSONL rows may use ``text``/``content`` fields or common
``conversations``/``messages`` chat structures.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO

try:
    import jinja2  # noqa: F401 - required by transformers.apply_chat_template
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install them with:\n"
        "  pip install 'tokenizers>=0.20' 'transformers>=4.45' 'jinja2>=3.1'"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "tokenizer_corpus"
DEFAULT_OUTPUT = PROJECT_ROOT / "model" / "tokenizer"
SUPPORTED_SUFFIXES = (".jsonl", ".jsonl.gz", ".txt", ".txt.gz")

# BpeTrainer 会按照列表顺序分配 ID。模型开始预训练后绝不能改变顺序，
# 否则相同的数字 ID 会对应不同 Token，已有模型权重将失去意义。
CORE_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
]

# 结构标记必须被编码成单个 Token，便于后续 SFT/Tool Calling 使用；
# 但解码时仍希望看见它们，所以训练后会把 special 标志改回 false。
STRUCTURAL_TOKENS = [
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
]

# 提前预留 ID。未来增加控制标记时，可复用这些槽位，避免扩大 Embedding 矩阵。
RESERVED_TOKENS = [f"<|reserved_{index}|>" for index in range(16)]
TRAINER_TOKENS = CORE_SPECIAL_TOKENS + STRUCTURAL_TOKENS + RESERVED_TOKENS

# Chat Template 只负责把 messages 列表渲染为一段文本；Tokenizer 预训练本身
# 不依赖对话模板，但现在保存它可避免后续 SFT 阶段重新修改 Tokenizer 文件。
CHAT_TEMPLATE = r"""{%- for message in messages %}
{{- '<|im_start|>' + message['role'] + '\n' }}
{{- message['content'] + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- endif %}"""

SMOKE_TEST_TEXTS = [
    "你好，世界！miniLLM 从零开始训练。",
    "Byte-level BPE keeps English, numbers 12.5%, and punctuation.",
    "Python 示例：\ndef add(a, b):\n    return a + b",
    "中英混合：Transformer 使用 attention 处理上下文。",
    "生僻字与 emoji：𠮷野家 🙂🚀",
    "空白测试：第一行\n    第二行\tTab",
]


@dataclass
class CorpusStats:
    """记录语料读取与清洗统计，最终写入 tokenizer_metadata.json。"""

    documents_seen: int = 0
    documents_used: int = 0
    chunks_yielded: int = 0
    characters_yielded: int = 0
    empty_or_short_skipped: int = 0
    invalid_json_skipped: int = 0
    unsupported_rows_skipped: int = 0


def parse_args() -> argparse.Namespace:
    """定义命令行参数；默认值保证在项目根目录即可直接训练。"""

    parser = argparse.ArgumentParser(
        description="Train miniLLM's 8192-token ByteLevel-BPE tokenizer."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        default=[DEFAULT_INPUT],
        help=(
            "One or more corpus files/directories. Defaults to "
            f"{DEFAULT_INPUT.relative_to(PROJECT_ROOT)}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8192,
        help="Final vocabulary size including reserved/control tokens (default: 8192).",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency for a BPE pair to be merged (default: 2).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Skip documents shorter than this many characters (default: 20).",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=20_000,
        help="Split very long documents into character chunks (default: 20000).",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=0,
        help="Stop after this many usable documents; 0 means no limit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing tokenizer files already present in the output directory.",
    )
    return parser.parse_args()


def is_supported(path: Path) -> bool:
    """按完整文件名判断格式，从而同时识别 .jsonl.gz 等双后缀。"""

    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def resolve_inputs(input_paths: Sequence[Path]) -> list[Path]:
    """把文件/目录参数展开为去重、排序后的实际语料文件列表。"""

    files: list[Path] = []
    for raw_path in input_paths:
        path = raw_path.expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if path.is_file():
            if not is_supported(path):
                raise ValueError(f"Unsupported corpus file: {path}")
            files.append(path.resolve())
        elif path.is_dir():
            files.extend(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file() and is_supported(candidate)
            )
        else:
            raise FileNotFoundError(f"Corpus path does not exist: {path}")

    unique_files = sorted(set(files), key=lambda item: str(item))
    if not unique_files:
        raise FileNotFoundError(
            "No supported corpus files found. Expected .jsonl, .jsonl.gz, .txt, or .txt.gz."
        )
    return unique_files


def open_text(path: Path) -> TextIO:
    """统一打开普通文本和 gzip 文本，损坏字符使用 ignore 跳过。"""

    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def message_content(message: Any) -> str | None:
    """从单条聊天消息中提取纯文本，兼容字符串和多段 content。"""

    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts) or None
    return None


def extract_text(row: Any) -> str | None:
    """从常见预训练/SFT JSON 结构中提取可用于 Tokenizer 的文本。"""
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return None

    for key in ("text", "content", "code"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            title = row.get("title")
            if key != "text" and isinstance(title, str) and title.strip():
                return f"{title}\n{value}"
            return value

    for key in ("conversations", "messages"):
        messages = row.get(key)
        if isinstance(messages, list):
            parts = [message_content(message) for message in messages]
            text_parts = [part for part in parts if part]
            if text_parts:
                return "\n".join(text_parts)

    # 兼容 instruction/input/output 风格。训练 Tokenizer 只关心文本覆盖面，
    # 因此可以把问题和回答拼接后参与词频统计。
    parts = [
        row.get(key)
        for key in ("instruction", "input", "output", "response")
        if isinstance(row.get(key), str) and row[key].strip()
    ]
    return "\n".join(parts) if parts else None


def iter_documents(path: Path, stats: CorpusStats) -> Iterator[str]:
    """逐条流式产出文档，避免一次性把 GB 级语料读进内存。"""

    lower_name = path.name.lower()
    is_jsonl = lower_name.endswith(".jsonl") or lower_name.endswith(".jsonl.gz")

    with open_text(path) as handle:
        if not is_jsonl:
            for line in handle:
                stats.documents_seen += 1
                yield line.rstrip("\n")
            return

        for line in handle:
            stats.documents_seen += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats.invalid_json_skipped += 1
                continue

            text = extract_text(row)
            if text is None:
                stats.unsupported_rows_skipped += 1
                continue
            yield text


def iter_training_chunks(
    files: Sequence[Path],
    stats: CorpusStats,
    min_chars: int,
    chunk_chars: int,
    max_documents: int,
) -> Iterator[str]:
    """过滤短文档并切块，向 tokenizers 库持续提供训练文本。"""

    for path in files:
        for text in iter_documents(path, stats):
            if len(text.strip()) < min_chars:
                stats.empty_or_short_skipped += 1
                continue

            stats.documents_used += 1
            for start in range(0, len(text), chunk_chars):
                chunk = text[start : start + chunk_chars]
                if not chunk:
                    continue
                stats.chunks_yielded += 1
                stats.characters_yielded += len(chunk)
                yield chunk

            if max_documents and stats.documents_used >= max_documents:
                return


def set_structural_tokens_visible(tokenizer_json: Path) -> None:
    """让 think/tool 标记保持原子性，同时在普通 decode 时仍然可见。"""
    data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    structural = set(STRUCTURAL_TOKENS)
    for token_info in data.get("added_tokens", []):
        if token_info.get("content") in structural:
            token_info["special"] = False
    tokenizer_json.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_tokenizer(tokenizer: PreTrainedTokenizerFast, vocab_size: int) -> dict[str, Any]:
    """执行训练后不变量检查，防止错误 Tokenizer 流入模型预训练。"""

    # 模型 Embedding 行数必须等于 Tokenizer 的总词表大小。
    if len(tokenizer) != vocab_size:
        raise RuntimeError(
            f"Expected vocabulary size {vocab_size}, but trainer produced {len(tokenizer)}. "
            "Use more/diverse corpus data or lower --min-frequency."
        )

    expected_ids = {
        "<|endoftext|>": 0,
        "<|im_start|>": 1,
        "<|im_end|>": 2,
    }
    actual_ids = {
        token: tokenizer.convert_tokens_to_ids(token) for token in TRAINER_TOKENS
    }
    for token, expected_id in expected_ids.items():
        if actual_ids[token] != expected_id:
            raise RuntimeError(
                f"Token ID invariant failed: {token}={actual_ids[token]}, expected {expected_id}."
            )

    # 结构标记必须各自只编码成一个 ID，否则模型很难稳定学习这些边界。
    for token in STRUCTURAL_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or ids[0] != actual_ids[token]:
            raise RuntimeError(f"Structural token is not atomic: {token} -> {ids}")

    # ByteLevel-BPE 的重要性质：任意 UTF-8 文本都应当可以无损往返。
    roundtrip_results = []
    for text in SMOKE_TEST_TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        roundtrip_results.append(
            {
                "text": text,
                "tokens": len(ids),
                "characters_per_token": round(len(text) / max(len(ids), 1), 4),
                "roundtrip_exact": decoded == text,
            }
        )
        if decoded != text:
            raise RuntimeError(
                f"Encode/decode round-trip failed:\ninput={text!r}\ndecoded={decoded!r}"
            )

    rendered_chat = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "你是 miniLLM 助手。"},
            {"role": "user", "content": "什么是 BPE？"},
            {"role": "assistant", "content": "BPE 是一种子词切分算法。"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "vocab_size": len(tokenizer),
        "token_ids": actual_ids,
        "roundtrip_tests": roundtrip_results,
        "chat_template_example": rendered_chat,
    }


def sha256(path: Path) -> str:
    """计算产物哈希，方便确认服务器与本地使用的是同一套 Tokenizer。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(args: argparse.Namespace) -> None:
    """执行 Tokenizer 训练、保存、验证的完整流程。"""

    # 1. 参数检查：ByteLevel 至少要容纳 256 个字节符号和控制 Token。
    if args.vocab_size <= len(TRAINER_TOKENS) + 256:
        raise ValueError("--vocab-size is too small for ByteLevel alphabet and control tokens.")
    if args.min_frequency < 1:
        raise ValueError("--min-frequency must be at least 1.")
    if args.min_chars < 1 or args.chunk_chars < args.min_chars:
        raise ValueError("Require 1 <= --min-chars <= --chunk-chars.")
    if args.max_documents < 0:
        raise ValueError("--max-documents cannot be negative.")

    # 2. 解析输入和输出路径。所有相对路径都以项目根目录为基准。
    input_files = resolve_inputs(args.input)
    output_dir = args.output.expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()

    tokenizer_json = output_dir / "tokenizer.json"
    if tokenizer_json.exists() and not args.overwrite:
        raise FileExistsError(
            f"Tokenizer already exists at {tokenizer_json}. Pass --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Corpus files : {len(input_files)}")
    print(f"Output       : {output_dir}")
    print(f"Vocabulary   : {args.vocab_size}")
    for path in input_files:
        print(f"  - {path}")

    # 3. 构建 BPE：ByteLevel 先把文本映射到稳定的字节符号，再由 BPE
    # 学习高频符号组合，因此生僻字、emoji 和代码字符也不会真正 OOV。
    stats = CorpusStats()
    tokenizer = Tokenizer(models.BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=TRAINER_TOKENS,
        max_token_length=64,
    )

    # 4. train_from_iterator 消费生成器。Python 端不保存全部语料，但
    # tokenizers 的 BPE 统计阶段仍可能占用较多内存。
    tokenizer.train_from_iterator(
        iter_training_chunks(
            files=input_files,
            stats=stats,
            min_chars=args.min_chars,
            chunk_chars=args.chunk_chars,
            max_documents=args.max_documents,
        ),
        trainer=trainer,
    )
    tokenizer.decoder = decoders.ByteLevel()

    if stats.documents_used == 0:
        raise RuntimeError("No usable text was found in the corpus.")
    if tokenizer.get_vocab_size() != args.vocab_size:
        raise RuntimeError(
            f"Expected {args.vocab_size} tokens, got {tokenizer.get_vocab_size()}. "
            "The corpus is likely too small or insufficiently diverse."
        )

    # 5. 保存 tokenizers 原生产物。vocab.json + merges.txt 便于检查 BPE，
    # tokenizer.json 则包含完整、可直接加载的处理流水线。
    tokenizer.save(str(tokenizer_json))
    tokenizer.model.save(str(output_dir))  # Also emits vocab.json and merges.txt.
    set_structural_tokens_visible(tokenizer_json)

    # 6. 包装为 Transformers Tokenizer，并显式声明模型会使用的特殊 ID。
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        model_max_length=32_768,
        clean_up_tokenization_spaces=False,
    )
    fast_tokenizer.chat_template = CHAT_TEMPLATE
    fast_tokenizer.save_pretrained(output_dir)

    # 7. 验证完成后写入元数据和哈希，作为可复现性记录。
    validation = validate_tokenizer(fast_tokenizer, args.vocab_size)
    metadata = {
        "name": "miniLLM-tokenizer",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "ByteLevel-BPE",
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "min_chars": args.min_chars,
        "chunk_chars": args.chunk_chars,
        "max_documents": args.max_documents,
        "input_files": [str(path) for path in input_files],
        "corpus_stats": asdict(stats),
        "validation": validation,
    }
    metadata_path = output_dir / "tokenizer_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata["artifact_sha256"] = {
        path.name: sha256(path)
        for path in output_dir.iterdir()
        if path.is_file() and path.name != metadata_path.name
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nTokenizer training completed successfully.")
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
    print("Core token IDs:")
    for token in CORE_SPECIAL_TOKENS:
        print(f"  {token:18s} -> {fast_tokenizer.convert_tokens_to_ids(token)}")
    print(f"Artifacts: {output_dir}")


def main() -> int:
    """CLI 入口：把常见可预期错误转成简洁的终端提示和退出码。"""

    try:
        train(parse_args())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

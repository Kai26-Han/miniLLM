# 从零训练 miniLLM Tokenizer

> 项目：miniLLM  
> 方案版本：Tokenizer v2  
> 训练目标：使用约 3 GB 中英文与代码语料，训练一个词表大小为 8192 的 ByteLevel BPE Tokenizer。

## 1. 项目目标

Tokenizer 负责将原始文本转换为模型可以处理的 Token ID，并在模型输出后将 Token ID 还原为文本。它会直接影响：

- 相同文本需要多少 Token。
- 模型的实际上下文长度。
- 中文、英文和代码的切分效果。
- Embedding 层和输出层的参数量。
- 模型预训、微调和推理的整体效率。

miniLLM 的第一版 Tokenizer 希望同时满足：

1. 中文为主，可以较好地切分常用汉字和中文词组。
2. 支持英文、数字、URL、数学符号和中英文混合文本。
3. 保留代码中的缩进、换行、标点和常见关键词。
4. 可以编码任意 UTF-8 内容，尽量不产生未知 Token。
5. 提前固定对话、思考和工具调用所需的结构 Token。

## 2. 为什么选择 ByteLevel BPE

本项目使用 **ByteLevel BPE**（Byte-level Byte Pair Encoding）。

ByteLevel 先将文本表示为 UTF-8 字节，因此初始字母表可以覆盖所有字节。BPE 再根据语料中的出现频率，反复合并常见的相邻单元。

它对 miniLLM 的主要优点是：

- 可以覆盖生僻字、Emoji 和任意 Unicode 文本。
- 不需要为每个汉字手工建立字典。
- 能够根据语料自动学习常用中文片段、英文子词和代码片段。
- 能精确保留空格、换行和 Tab，对代码尤其重要。
- 在较小词表下仍然可以处理词表外内容。

## 3. 为什么词表大小是 8192

词表越大，常见文本通常可以用更少 Token 表示，但 Embedding 层和输出层也会随之增大。对 miniLLM 这样的小模型，过大的词表会占用不必要的参数。

8192 是本次实验的平衡点：

- 比纯字节词表具有更好的中英文压缩率。
- 可容纳常见中文片段、英文子词和编程符号。
- 对小参数量模型的 Embedding 成本较低。
- 便于完整走通从 Tokenizer 到模型预训的实验流程。

这不代表 8192 对所有模型都是最优值。如果后续明显增大模型规模、语言范围或专业领域，可以对 8192、16384 和 32768 词表做同语料对比实验。

## 4. 语料方案

项目从 Hugging Face 抽取约 **3.00 GB** 语料，当前版本直接使用全部五份 JSONL 训练：

| 类别 | Hugging Face 数据集 | 比例 | 训练大小 | 作用 |
| --- | --- | ---: | ---: | --- |
| 中文网页 | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) `cmn_Hani` | 50% | 1.50 GB | 广泛覆盖中文网页用语 |
| 中文百科 | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.zh` | 20% | 600 MB | 补充实体、书面语和百科表述 |
| 中文高质量 | [BAAI/CCI3-HQ](https://huggingface.co/datasets/BAAI/CCI3-HQ) | 15% | 450 MB | 提升中文高质量网页占比 |
| 英文通用 | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) `sample-10BT` | 10% | 300 MB | 学习常见英文单词和子词 |
| 代码 | [bigcode/the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol) | 5% | 150 MB | 学习代码关键词、符号和缩进 |
| **合计** |  | **100%** | **3.00 GB** |  |

中文占 85%，英文占 10%，代码占 5%。文件系统中 `du -sh` 可能显示约 `2.8G`，这是十进制 GB 与二进制 GiB 的换算差异，不代表语料缺失。

## 5. 语料抽取脚本

本项目为每个数据源提供了独立的流式抽取脚本：

| 脚本 | 默认输出 |
| --- | --- |
| [`extract_fineweb2_zh.py`](../scripts/extract_fineweb2_zh.py) | `dataset/tokenizer_corpus/fineweb2_zh_1_5gb.jsonl` |
| [`extract_wikipedia_zh.py`](../scripts/extract_wikipedia_zh.py) | `dataset/tokenizer_corpus/wikipedia_zh_600mb.jsonl` |
| [`extract_cci3_hq_zh.py`](../scripts/extract_cci3_hq_zh.py) | `dataset/tokenizer_corpus/cci3_hq_zh_450mb.jsonl` |
| [`extract_fineweb_en.py`](../scripts/extract_fineweb_en.py) | `dataset/tokenizer_corpus/fineweb_en_300mb.jsonl` |
| [`extract_stack_smol_code.py`](../scripts/extract_stack_smol_code.py) | `dataset/tokenizer_corpus/stack_smol_code_150mb.jsonl` |

这些脚本具有以下共同特性：

- 通过 Hugging Face `datasets` 以 streaming 模式读取数据。
- 不会先将整个数据集下载到本地。
- 根据最终 JSONL 字节数停止，保证配比可控。
- 过滤空文档、过短文档和异常过长文档。
- 使用 streaming shuffle buffer 减少只截取数据集开头的偏差。
- 按固定 seed 打乱，便于复现。
- 定期更新 `*.metadata.json`，记录数据源和进度。
- 用户中断时，已经写入的 JSONL 仍然保持逐行可解析。

### 5.1 校验 3 GB 训练语料

正式训练直接读取以下目录，不需要再执行本地采样：

```text
dataset/tokenizer_corpus/
├── fineweb2_zh_1_5gb.jsonl
├── wikipedia_zh_600mb.jsonl
├── cci3_hq_zh_450mb.jsonl
├── fineweb_en_300mb.jsonl
├── stack_smol_code_150mb.jsonl
└── 对应的 *.metadata.json
```

训练前执行：

```bash
ls -lh dataset/tokenizer_corpus/*.jsonl
wc -l dataset/tokenizer_corpus/*.jsonl
```

五份 `*.metadata.json` 的 `status` 都应为 `complete`。元数据文件不以 `.jsonl` 结尾，不会参与训练。

[`build_tokenizer_training_sample.py`](../scripts/build_tokenizer_training_sample.py) 和 `dataset/tokenizer_train_1gb/` 继续保留，只作为低内存环境下的回退方案，不属于当前3GB正式训练流程。

### 5.2 统一 JSONL 格式

普通语料统一保存为：

```json
{"text":"文档内容"}
```

Wikipedia 默认在正文前加入文章标题：

```json
{"text":"文章标题\n\n文章正文"}
```

The Stack Smol 除 `text` 外还保留许可和来源信息：

```json
{
  "text": "def add(a, b):\n    return a + b",
  "language": "python",
  "licenses": ["mit"],
  "repository_name": "example/project",
  "path": "src/example.py"
}
```

训练器只读取 `text`，会忽略其他字段。

### 5.3 代码语料配比

The Stack Smol 内部以 Python 为主：

| 语言 | 比例 |
| --- | ---: |
| Python | 60% |
| JavaScript | 10% |
| TypeScript | 5% |
| Java | 10% |
| C++ | 5% |
| C | 5% |
| Shell | 2% |
| Go | 2% |
| Rust | 1% |

如果某个语言的数据提前耗尽，容量差额会由后续语言尽量补足。

## 6. 环境准备

语料抽取可以在 MacBook CPU 上完成，但3GB全量 ByteLevel BPE 训练会在词频和 pair 统计阶段占用大量系统内存。本次正式训练使用Linux 服务器：

```text
CPU：64 核
系统内存：480 GB
GPU：4 × NVIDIA GeForce RTX 4090
```

Tokenizer 训练使用的是 CPU 和系统内存，不使用 CUDA。4张GPU不会加速这一阶段，执行期间 GPU 利用率接近0属于正常现象；多卡GPU从模型预训练阶段才开始使用。

本项目 Tokenizer 阶段的主要依赖为：

- `datasets`：流式读取 Hugging Face 数据集。
- `tokenizers`：底层 ByteLevel BPE 训练。
- `transformers`：封装为 Hugging Face `PreTrainedTokenizerFast`。
- `jinja2`：支持 Chat Template。

## 7. 训练器设计

训练脚本位于 [`trainer/train_tokenizer.py`](../trainer/train_tokenizer.py)。

默认参数如下：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--vocab-size` | 8192 | 最终词表大小，包含控制和预留 Token |
| `--min-frequency` | 2 | BPE 合并对的最低出现频率 |
| `--min-chars` | 20 | 训练时忽略过短文档 |
| `--chunk-chars` | 20000 | 过长文档按字符分块 |
| `--max-documents` | 0 | 0 表示不限制文档数 |
| `--input` | `dataset/tokenizer_corpus` | 默认值；本次直接使用完整3GB语料 |
| `--output` | `model/tokenizer` | 默认值；当前3GB正式 Tokenizer 的唯一目录 |

训练脚本会递归扫描 `.jsonl`、`.jsonl.gz`、`.txt` 和 `.txt.gz` 文件。`*.metadata.json` 不属于支持的语料后缀，因此不会参与训练。

对 JSONL，训练器支持：

- `text`、`content` 或 `code` 字段。
- `messages` 或 `conversations` 对话结构。
- `instruction` / `input` / `output` / `response` 类型数据。

当前五份 Tokenizer 训练语料均通过 `text` 字段读取。训练命令显式传入 `--input dataset/tokenizer_corpus`，五份 JSONL 会按文件名排序后依次进入同一次 BPE 统计。

## 8. 特殊 Token 与 Chat Template

核心 Token 在 BPE 训练时按固定顺序加入，从而确保 ID 稳定：

| ID | Token | 用途 |
| ---: | --- | --- |
| 0 | `<\|endoftext\|>` | Padding 及 Unknown fallback |
| 1 | `<\|im_start\|>` | BOS / 消息开始 |
| 2 | `<\|im_end\|>` | EOS / 消息结束 |

结构 Token：

```text
<think>          </think>
<tool_call>      </tool_call>
<tool_response>  </tool_response>
```

这些结构 Token 在编码时必须保持为单个原子 Token，但在解码时需要保持可见。项目还提前预留了 16 个 Token：

```text
<|reserved_0|> ... <|reserved_15|>
```

这样在后续增加少量控制语义时，可以尽量避免改变模型 Embedding 大小。

默认对话格式为：

```text
<|im_start|>system
你是 miniLLM 助手。<|im_end|>
<|im_start|>user
用户问题<|im_end|>
<|im_start|>assistant
助手回答<|im_end|>
```

## 9. 启动 Tokenizer 训练

训练前先确认五份原始语料的 metadata 均为 `complete`。当前从头训练流程将3GB正式 Tokenizer 输出到项目标准目录：

```bash
python trainer/train_tokenizer.py \
  --input dataset/tokenizer_corpus \
  --output model/tokenizer
```

全量3GB方案的主要瓶颈依然是 CPU RAM，而不是 `vocab-size=8192` 或GPU显存。

## 10. 训练产物

训练完成后，`model/tokenizer` 中应该包含以下核心文件：

```text
model/tokenizer/
├── tokenizer.json
├── vocab.json
├── merges.txt
├── tokenizer_config.json
├── special_tokens_map.json
├── chat_template.jinja
└── tokenizer_metadata.json
```

其中：

- `tokenizer.json`：完整 Tokenizer 定义，可由 Hugging Face 直接加载。
- `vocab.json`：Token 到 ID 的映射。
- `merges.txt`：BPE 合并规则。
- `tokenizer_config.json`：模型最大长度和 Chat Template 等配置。
- `special_tokens_map.json`：BOS、EOS、PAD 和 UNK 对应关系。
- `chat_template.jinja`：SFT、对话推理和后续工具调用使用的消息模板。
- `tokenizer_metadata.json`：训练参数、语料统计、冒烟测试和产物 SHA-256。

完成后训练脚本会立即检查：

- 最终词表是否恰好为 8192。
- 核心 Token 的 ID 是否为 0、1、2。
- 中文、英文、代码、Emoji、生僻字和空白符能否精确往返。
- Chat Template 能否正常生成对话文本。

## 11. 独立评估

评估脚本位于 [`eval/eval_tokenizer.py`](../eval/eval_tokenizer.py)。

```bash
python eval/eval_tokenizer.py \
  --tokenizer model/tokenizer \
  --expected-vocab-size 8192 \
  --show-examples 10 \
  --report eval/tokenizer_3gb_eval_report.json
```

### 11.1 硬性验收标准

| 检查项 | 要求 |
| --- | --- |
| 整体状态 | `PASS` |
| 词表大小 | 8192 |
| 核心 Token ID | 必须是 0、1、2 |
| 结构 Token | 每个必须是单个原子 Token |
| Chat Template | 检查通过 |
| UNK rate | 0 |
| Round-trip failures | 0 |

ByteLevel BPE 应当能够通过字节退化表示任意 UTF-8 文本，因此 UNK rate 不应高于 0。编码后再解码的文本应与原文完全一致。

### 11.2 压缩效率

评估报告还会记录：

- `characters_per_token`：平均每个 Token 承载的字符数。
- `bytes_per_token`：平均每个 Token 承载的 UTF-8 字节数。
- `tokens_per_sample.mean`：每条样本的平均 Token 数。
- `tokens_per_sample.p95`：95% 样本不超过的 Token 数。

这些指标应该在同一份测试集上比较。`characters_per_token` 越高，通常表示压缩效率越好，但不能用中文的值直接与英文或代码的值比较。

### 11.3 独立留出集

正式评估不应只使用参与训练的文本。建议从未参与 Tokenizer 训练的数据中准备：

```text
dataset/tokenizer_eval/
├── chinese.jsonl
├── english.jsonl
├── code.jsonl
└── mixed.jsonl
```

然后分别评估，避免总平均值掩盖某个类别的过度切分：

```bash
python eval/eval_tokenizer.py \
  --tokenizer model/tokenizer \
  --input dataset/tokenizer_eval/chinese.jsonl \
  --max-samples 10000 \
  --report eval/tokenizer_3gb_eval_chinese.json
```

## 12. 与后续训练衔接

从当前 Tokenizer 版本开始，Pretrain、SFT和RL等训练阶段必须始终加载同一个目录。

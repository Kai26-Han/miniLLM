# Training the miniLLM Tokenizer from Scratch

> Project: miniLLM
> Pipeline version: Tokenizer v2
> Training objective: use about 3 GB of Chinese, English, and code data to train a byte-level BPE tokenizer with a vocabulary of 8,192 tokens.

## 1. Project objectives

The tokenizer converts raw text into token IDs the model can process, then decodes generated token IDs back into text. It directly affects:

- how many tokens are required to represent a given text;
- the model's effective context length;
- token boundaries in Chinese, English, and code;
- the parameter counts of the embedding and output layers; and
- the overall efficiency of pretraining, fine-tuning, and inference.

The first miniLLM tokenizer is designed to satisfy the following requirements:

1. Prioritize Chinese while learning useful boundaries for common characters and phrases.
2. Support English, numbers, URLs, mathematical symbols, and mixed Chinese-English text.
3. Preserve indentation, newlines, punctuation, and common programming keywords.
4. Encode arbitrary UTF-8 content without producing unknown tokens whenever possible.
5. Reserve structural tokens for conversations, reasoning, and tool calls in advance.

## 2. Why choose ByteLevel BPE?

This project uses **byte-level BPE** (Byte Pair Encoding).

The byte-level pre-tokenizer first represents text as UTF-8 bytes, so the initial alphabet can cover every possible byte. BPE then repeatedly merges adjacent units that occur frequently in the corpus.

Its main advantages to miniLLM are:

- It can represent rare characters, emoji, and arbitrary Unicode text.
- There is no need to manually build a dictionary for each Chinese character.
- It can automatically learn common Chinese fragments, English subwords and code fragments according to the corpus.
- It can accurately keep spaces, line breaks and Tabs, which is especially important for the code.
- Byte fallback allows content not represented by larger vocabulary entries to remain encodable.

## 3. Why is the size of the vocabulary 8192?

The larger the vocabulary, the more common text can usually be represented by fewer tokens, but the embedding layer and the output layer will also increase accordingly. For a small model like miniLLM, too large a vocabulary will occupy unnecessary parameters.

8192 is the balance point of this experiment:

- It has a better Chinese and English compression rate than the pure byte vocabulary.
- It can accommodate common Chinese fragments, English subwords and programming symbols.
- The cost of embedding for small-parameter models is low.
- It is convenient to complete the experimental process from tokenizer to model pre-training.

This does not mean that 8,192 is optimal for every model. If the model becomes much larger or expands to more languages or specialized domains, compare vocabulary sizes such as 8,192, 16,384, and 32,768 on the same corpus.

## 4. Corpus scheme

The project extracts about **3.00 GB** corpus from Hugging Face, and the current version directly uses all five JSONL trainings:

| Category | Hugging Face Data Set | Proportion | Training Size | Function |
| --- | --- | ---: | ---: | --- |
| Chinese web page | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) `cmn_Hani` | 50% | 1.50 GB | Widely cover Chinese web terms |
| Chinese Encyclopedia | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.zh` | 20% | 600 MB | Supplementary Entity, Written Language and Encyclopedia Expression |
| Chinese high quality | [BAAI/CCI3-HQ](https://huggingface.co/datasets/BAAI/CCI3-HQ) | 15% | 450 MB | Improve the proportion of Chinese high-quality web pages |
| English General | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) `sample-10BT` | 10% | 300 MB | Learn common English words and sub-words |
| Code | [bigcode/the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol) | 5% | 150 MB | Learn code keywords, symbols and indentation |
| **total**| |**100%**|**3.00 GB** | |

Chinese accounts for 85%, English accounts for 10%, and code accounts for 5%. `du -sh` in the file system may show about `2.8G`, which is the conversion difference between decimal GB and binary GiB, and does not mean that the corpus is missing.

## 5. Corpus extraction script

This project provides an independent stream extraction script for each data source:

| Script | Default output |
| --- | --- |
| [`extract_fineweb2_zh.py`](../../scripts/extract_fineweb2_zh.py) | `dataset/tokenizer_corpus/fineweb2_zh_1_5gb.jsonl` |
| [`extract_wikipedia_zh.py`](../../scripts/extract_wikipedia_zh.py) | `dataset/tokenizer_corpus/wikipedia_zh_600mb.jsonl` |
| [`extract_cci3_hq_zh.py`](../../scripts/extract_cci3_hq_zh.py) | `dataset/tokenizer_corpus/cci3_hq_zh_450mb.jsonl` |
| [`extract_fineweb_en.py`](../../scripts/extract_fineweb_en.py) | `dataset/tokenizer_corpus/fineweb_en_300mb.jsonl` |
| [`extract_stack_smol_code.py`](../../scripts/extract_stack_smol_code.py) | `dataset/tokenizer_corpus/stack_smol_code_150mb.jsonl` |

These scripts have the following common characteristics:

- Read data in streaming mode through Hugging Face `datasets`.
- The entire data set will not be downloaded locally first.
- Stop according to the final number of JSONL bytes to ensure that the ratio can be controlled.
- Filter empty documents, too short documents and abnormally too long documents.
- Use streaming shuffle buffer to reduce the deviation of only intercepting the beginning of the data set.
- Disturb according to the fixed seed, which is convenient for reproduction.
- Update `*.metadata.json` regularly to record the data source and progress.
- When the user interrupts, the written JSONL can still be parsed line by line.

### 5.1 Check 3 GB training corpus

The formal training directly reads the following directories, and there is no need to perform local sampling:

```text
dataset/tokenizer_corpus/
├── fineweb2_zh_1_5gb.jsonl
├── wikipedia_zh_600mb.jsonl
├── cci3_hq_zh_450mb.jsonl
├── fineweb_en_300mb.jsonl
├── stack_smol_code_150mb.jsonl
└── Corresponding *.metadata.json
```

Execution before training:

```bash
ls -lh dataset/tokenizer_corpus/*.jsonl
wc -l dataset/tokenizer_corpus/*.jsonl
```

The `status` of the five `*.metadata.json` should be `complete`. The metadata file does not end with `.jsonl` and will not participate in the training.

The legacy `build_tokenizer_training_sample.py` workflow and `dataset/tokenizer_train_1gb/` are retained only as a low-memory fallback; they are not part of the current 3 GB production training pipeline.

### 5.2 Unified JSONL format

Ordinary corpus is uniformly saved as:

```json
{"text":"Document content"}
```

Wikipedia adds the title of the article before the text by default:

```json
{"text":"Article title\n\nArticle body"}
```

In addition to `text`, The Stack Smol also retains license and source information:

```json
{
  "text": "def add(a, b):\n    return a + b",
  "language": "python",
  "licenses": ["mit"],
  "repository_name": "example/project",
  "path": "src/example.py"
}
```

The trainer only reads `text` and will ignore other fields.

### 5.3 Code corpus ratio

The Stack Smol is mainly Python:

| Language | Proportion |
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

If the data of a language is exhausted in advance, the capacity difference will be made up by the subsequent language as much as possible.

## 6. Environmental preparation

Corpus extraction can be done on MacBook CPU, but 3GB full ByteLevel BPE training will occupy a large amount of system memory in the word frequency and pair statistics stages. This official training uses Linux server:

```text
CPU: 64 cores
System memory: 480 GB
GPU: 4 × NVIDIA GeForce RTX 4090
```

tokenizer training uses CPU and system memory, not CUDA. 4 GPUs will not accelerate this stage. It is normal for the GPU utilization rate to be close to 0 during execution; multi-GPU GPUs are only used from the model pre-training stage.

The main dependencies of the tokenizer stage of this project are:

- `datasets`: Stream reading of Hugging Face data set.
- `tokenizers`: Low-level ByteLevel BPE training.
- `transformers`: Encapsulated as Hugging Face `PreTrainedTokenizerFast`.
- `jinja2`: Support chat template.

## 7. Trainer design

The training script is located at [`trainer/train_tokenizer.py`](../../trainer/train_tokenizer.py).

The default parameters are as follows:

| Parameters | Default Value | Description |
| --- | ---: | --- |
| `--vocab-size` | 8192 | Final vocabulary size, including control and reserved token |
| `--min-frequency` | 2 | The lowest frequency of BPE combined pairs |
| `--min-chars` | 20 | Ignore too short documents during training |
| `--chunk-chars` | 20000 | Too long documents are divided by characters |
| `--max-documents` | 0 | 0 means that the number of documents is not limited |
| `--input` | `dataset/tokenizer_corpus` | Default value; directly use the complete 3GB corpus this time |
| `--output` | `model/tokenizer` | Default value; the only directory of the current 3GB official tokenizer |

The training script will recursively scan `.jsonl`, `.jsonl.gz`, `.txt` and `.txt.gz` files. `*.metadata.json` is not a supported corpus suffix, so it will not participate in the training.

For JSONL, the trainer supports:

- `text`, `content` or `code` field.
- `messages` or `conversations` dialogue structure.
- `instruction` / `input` / `output` / `response` type data.

At present, the five tokenizer training corpus are read through the `text` field. The training command is explicitly passed to `--input dataset/tokenizer_corpus`, and the five JSONLs will be sorted by file name and enter the same BPE statistics in turn.

## 8. Special token and chat template

Core tokens are added in a fixed order during BPE training to ensure the stability of ID:

| ID | token | Purpose |
| ---: | --- | --- |
| 0 | `<\|endoftext\|>` | Padding and Unknown fallback |
| 1 | `<\|im_start\|>` | BOS / News Start |
| 2 | `<\|im_end\|>` | EOS / End of message |

Structure token:

```text
<think>          </think>
<tool_call>      </tool_call>
<tool_response>  </tool_response>
```

These structures token must be kept as a single atomic token when encoding, but they need to be visible when decoding. The project also reserves 16 tokens in advance:

```text
<|reserved_0|> ... <|reserved_15|>
```

In this way, when adding a small amount of control semantics in the future, you can try to avoid changing the size of the model embedding.

The default dialogue format is:

```text
<|im_start|>system
You are miniLLM's assistant. <|im_end|>
<|im_start|>user
User problems <|im_end|>
<|im_start|>assistant
The assistant answered <|im_end|>
```

## 9. Start tokenizer training

Before training, make sure that the metadata of the five original corpus are all `complete`. The current training process from the beginning outputs 3GB official tokenizer to the project standard catalog:

```bash
python trainer/train_tokenizer.py \
  --input dataset/tokenizer_corpus \
  --output model/tokenizer
```

The main bottleneck of the full 3GB scheme is still CPU RAM, not `vocab-size=8192` or GPU GPU memory.

## 10. Training products

After the training is completed, the following core files should be included in `model/tokenizer`:

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

Among them:

- `tokenizer.json`: Complete tokenizer definition, which can be loaded directly by Hugging Face.
- `vocab.json`: The mapping of token to ID.
- `merges.txt`: BPE merge rules.
- `tokenizer_config.json`: The maximum length of the model and chat template and other configurations.
- `special_tokens_map.json`: CORRESPONDENCE OF BOS, EOS, PAD AND UNK.
- `chat_template.jinja`: Message templates used by SFT, dialogue reasoning and follow-up tool calls.
- `tokenizer_metadata.json`: training parameters, corpus statistics, smoke test and product SHA-256.

After completion, the training script will be checked immediately:

- Is the final vocabulary exactly 8192?
- Whether the ID of the core token is 0, 1, 2.
- Chinese, English, code, emoji, strange characters and blanks can be accurately rounded and forth.
- Can chat template generate dialogue text normally?

## 11. Independent evaluation

The evaluation script is located at [`eval/eval_tokenizer.py`](../../eval/eval_tokenizer.py).

```bash
python eval/eval_tokenizer.py \
  --tokenizer model/tokenizer \
  --expected-vocab-size 8192 \
  --show-examples 10 \
  --report eval/tokenizer_3gb_eval_report.json
```

### 11.1 Rigid acceptance standards

| Inspection items | Requirements |
| --- | --- |
| Overall status | `PASS` |
| vocabulary size | 8192 |
| Core token ID | Must be 0, 1, 2 |
| Structure token | Each must be a single atom token |
| chat template | Check and pass |
| UNK rate | 0 |
| Round-trip failures | 0 |

ByteLevel BPE should be able to represent any UTF-8 text through byte degradation, so UNK rate should not be higher than 0. The text that is encoded and then decoded should be exactly the same as the original text.

### 11.2 Compression efficiency

The evaluation report will also record:

- `characters_per_token`: The average number of characters carried by each token.
- `bytes_per_token`: The average number of UTF-8 bytes carried by each token.
- `tokens_per_sample.mean`: The average number of tokens per sample.
- `tokens_per_sample.p95`: The number of tokens that do not exceed 95% of the sample.

These indicators should be compared on the same test set. The higher the `characters_per_token`, the better the compression efficiency, but the Chinese value cannot be directly compared with the value of English or code.

### 11.3 Independent set

Formal evaluation should not only use the text that participates in the training. It is recommended to prepare in the data that has never participated in tokenizer training:

```text
dataset/tokenizer_eval/
├── chinese.jsonl
├── english.jsonl
├── code.jsonl
└── mixed.jsonl
```

Then evaluate separately to avoid the total average to cover up the excessive division of a category:

```bash
python eval/eval_tokenizer.py \
  --tokenizer model/tokenizer \
  --input dataset/tokenizer_eval/chinese.jsonl \
  --max-samples 10000 \
  --report eval/tokenizer_3gb_eval_chinese.json
```

## 12. Connect with follow-up training

Starting from the current tokenizer version, training stages such as Pretrain, SFT and RL must always load the same directory.

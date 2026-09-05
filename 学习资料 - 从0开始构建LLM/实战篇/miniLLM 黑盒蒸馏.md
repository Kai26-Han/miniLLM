# miniLLM 黑盒蒸馏

> 目标：使用 `Qwen/Qwen3-1.7B` 为现有 SFT Prompt 生成新的 Assistant
> 答案，再从 `out/pretrain` 开始进行蒸馏式 SFT。

## 1. 训练路线

```text
dataset/sft/sft.jsonl
        ↓ Qwen3-1.7B 离线生成
dataset/distill/qwen3_1_7b_sft.jsonl
        ↓ 数据校验
trainer/train_distillation.py + out/pretrain
        ↓
out/sft_distilled
```

这是序列级黑盒蒸馏。Qwen 与 miniLLM 的 Tokenizer 不同，因此不读取
Qwen logits，也不计算 token-level KL Loss。Teacher 只负责生成文本，
训练时所有文本都由 miniLLM 的 8192 Tokenizer 重新编码。

## 2. 准备环境

```bash
pip install -r requirements.txt
```

首次运行会从 Hugging Face 下载 `Qwen/Qwen3-1.7B`。如果已下载到本地，
可以把 `--model-path` 改为本地模型目录，并增加 `--local-files-only`。

## 3. 先跑小样本

不要第一次就处理整个 13GB SFT 文件。先生成 100 条：

```bash
python scripts/generate_distill_data.py \
    --input-path dataset/sft/sft.jsonl \
    --output-path dataset/distill/qwen3_1_7b_smoke.jsonl \
    --model-path Qwen/Qwen3-1.7B \
    --distill-ratio 0.7 \
    --thinking-ratio 0.15 \
    --batch-size 2 \
    --max-samples 100
```

默认行为：

- 70% 非 Tool 对话的所有 Assistant turn 由 Teacher 重新生成；
- 30% 样本保留原答案，用作 SFT Replay；
- 每个需要生成的 Assistant turn 有15% 概率开启 Thinking；
- Tool Call 轨迹默认整条保留，避免新调用与旧 Tool Result 不一致；
- 输出与输入一行对应一行，不会把相同 Prompt 拆到训练集与验证集。

中断后从已完成的 JSONL 行继续：

```bash
python scripts/generate_distill_data.py \
    --output-path dataset/distill/qwen3_1_7b_smoke.jsonl \
    --max-samples 100 \
    --resume
```

`--resume` 会检查 sidecar manifest 中的模型、比例、Token 上限和随机种子；
关键配置改变时会拒绝向旧文件继续追加。

## 4. 校验蒸馏数据

```bash
python scripts/validate_distill_data.py \
    --data-path dataset/distill/qwen3_1_7b_smoke.jsonl \
    --model-path out/pretrain \
    --max-seq-len 1024 \
    --token-sample-every 1
```

校验包括：

- JSONL 和消息结构；
- Assistant 空答案；
- `reasoning_content` 与 `<think>` 泄漏；
- Qwen Chat 特殊 Token 泄漏；
- Tool 字段的 JSON 可解析性；
- 使用 miniLLM Tokenizer 得到的序列长度与 Assistant Target Token 数。

大数据默认约抽样 1% 做精确 Token 统计，但所有行都会做结构检查。
报告默认保存为 `*.validation.json`。

## 5. 蒸馏式 SFT 冒烟训练

```bash
python trainer/train_distillation.py \
    --model-path out/pretrain \
    --data-path dataset/distill/qwen3_1_7b_smoke.jsonl \
    --save-dir checkpoints/sft_distilled_smoke \
    --output-dir out/sft_distilled_smoke \
    --max-train-samples 100 \
    --eval-samples 20 \
    --max-steps 20 \
    --batch-size 2 \
    --accumulation-steps 2 \
    --learning-rate 1e-5 \
    --tracker none
```

重点检查 Loss 是否有限、MoE Router 是否均衡、回答是否被大量截断，以及
`out/sft_distilled_smoke` 能否正常加载生成。

## 6. 生成正式数据

小样本通过后再换成正式输出路径。可以先做 1万～5万条 PoC：

```bash
python scripts/generate_distill_data.py \
    --input-path dataset/sft/sft.jsonl \
    --output-path dataset/distill/qwen3_1_7b_sft.jsonl \
    --model-path Qwen/Qwen3-1.7B \
    --distill-ratio 0.7 \
    --thinking-ratio 0.15 \
    --batch-size 4 \
    --max-samples 50000
```

验证后增大 `--max-samples` 并加 `--resume` 即可继续扩展同一份数据。
要处理全部数据时，不传 `--max-samples`。

## 7. 正式训练

```bash
python trainer/train_distillation.py \
    --model-path out/pretrain \
    --data-path dataset/distill/qwen3_1_7b_sft.jsonl \
    --save-dir checkpoints/sft_distilled \
    --output-dir out/sft_distilled \
    --batch-size 4 \
    --accumulation-steps 16 \
    --max-seq-len 1024 \
    --learning-rate 1e-5 \
    --epochs 1 \
    --attention-backend eager \
    --tracker swanlab
```

蒸馏文件本身已包含 30% 原始 SFT Replay，正式训练时不要再同时传入
`dataset/sft/sft.jsonl`，否则会改变原设定的 Teacher/Replay 比例。

`train_distillation.py` 是黑盒蒸馏的明确入口：它先检查数据生成 manifest、
Teacher 来源、Teacher/Replay 行数和生成状态，再调用已有的 SFT 训练循环。
蒸馏训练进程不会加载 Qwen，因此只占用 Student 训练显存。

## 8. 建议的对照实验

保留两个独立输出：

1. `out/sft_baseline`：只用原始 SFT 数据；
2. `out/sft_distilled`：使用同样的训练步数和 Batch，但换成蒸馏混合数据。

对比 Assistant Loss、中英文问答、代码、指令遵循、输出长度、预训练 PPL
与 MoE Expert 负载。只有在相同计算量对照下改善，才能归因于蒸馏数据。

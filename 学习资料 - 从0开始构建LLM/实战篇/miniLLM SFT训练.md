# miniLLM SFT 训练

> 阶段：从 miniLLM 预训练模型开始进行全参数监督微调

> 默认方案：`sft_mini.jsonl`、序列长度 768、学习率 1e-5、2 轮、Assistant-only Loss

---

## 1. SFT 的训练目标

预训练让模型学习“根据左侧上下文预测下一个 Token”，SFT 则进一步教会模型：

- 理解 system 和 user 指令；
- 以 assistant 身份回答；
- 进行多轮对话；
- 在需要时输出显式思考结构；
- 按约定格式发起 Tool Calling；
- 在生成回答后输出 `<|im_end|>` 并停止。

SFT 不会重新训练 Tokenizer，也不会从随机参数开始。训练入口必须加载预训练阶段导出的 Transformers 模型目录。

```text
预训练模型 out/pretrain
        +
conversation JSONL
        ↓
ChatML / Reasoning / Tool Calling 模板
        ↓
只保留 assistant 标签
        ↓
全参数 SFT + MoE Router 辅助损失
        ↓
out/sft + checkpoints/sft/latest.pt
```

## 2. 相关文件

| 文件 | 作用 |
|---|---|
| `dataset/sft_dataset.py` | 数据校验、对话渲染、长对话截断、Assistant Loss Mask、动态 Padding |
| `trainer/train_sft.py` | 单卡/DDP 训练、评估、日志、断点和模型导出 |
| `eval/eval_sft.py` | Assistant-only Loss/PPL 与对话生成测试 |
| `out/pretrain/chat_template.jinja` | 随预训练模型导出的对话、思考和工具调用模板；SFT 必须继承同目录 Tokenizer |
| `trainer/trainer_utils.py` | 复用预训练阶段的分布式、Scheduler、Checkpoint 和 Tracker 能力 |

## 3. 数据格式

每行必须包含一个 `conversations` 数组：

```json
{"conversations":[
  {"role":"system","content":"你是一个可靠的AI助手。"},
  {"role":"user","content":"解释一下什么是梯度下降。"},
  {"role":"assistant","content":"梯度下降是一种优化算法……","reasoning_content":"先说明目标，再解释更新过程。"}
]}
```

支持的角色是：

- `system`：系统指令，只作为上下文；
- `user`：用户输入，只作为上下文；
- `assistant`：模型输出，参与 Loss；
- `tool`：工具返回值，只作为上下文。

可选字段：

- `reasoning_content`：渲染到 `<think>...</think>`；
- `tools`：函数定义列表或其 JSON 字符串；
- `tool_calls`：Assistant 要生成的函数调用。

### 3.1 Assistant-only Loss

完整对话会输入模型，但只有 Assistant 输出和 `<|im_end|>` 参与交叉熵：

```text
system/user/tool/header/padding → label = -100
assistant reasoning/content/tool_call/EOS → label = token_id
```

模型自身会将 logits 和 labels 左右错开一位计算 next-token loss。

### 3.2 长对话截断

超过 `--max-seq-len` 时：

1. 保留开头的 system 消息；
2. 从左侧删除最早的完整对话轮次；
3. 尽量保留最近的 user/assistant 交互；
4. 如果最新一轮本身仍超过上限，优先舍弃可选的长 reasoning，保留最近的
   Prompt 上下文、Assistant Header 和最终回答；回答仍过长时保留回答开头，
   不伪造提前结束的 EOS；
5. 日志记录 `truncated_ratio` 和 `answer_truncated_ratio`。

### 3.3 训练集与验证集

默认使用 `--val-ratio 0.001`。数据集通过由随机种子控制的确定性全局置换
划分训练集和验证集，不需要为 510 万条数据在内存中保存索引数组，同时可让
有限数量的验证样本覆盖文件前后不同的数据来源。相同数据文件和 `--seed` 会
得到完全相同的划分。

## 4. 开始训练

先确认预训练模型已经导出：

```text
out/pretrain/
├── config.json
├── model.safetensors
└── tokenizer files...
```

默认参数已固化为这次验证效果较好的 mini 数据方案。单卡正式训练：

```bash
python trainer/train_sft.py \
    --model-path out/pretrain \
    --tracker swanlab
```

六卡训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun --standalone --nproc_per_node=6 trainer/train_sft.py \
    --model-path out/pretrain \
    --tracker swanlab
```

有效全局 Batch Size：

```text
global_batch = batch_size × accumulation_steps × GPU数量
```

默认单卡配置是 `4 × 4 × 1 = 16` 条序列/更新；六卡配置是
`4 × 4 × 6 = 96` 条序列/更新。

全量数据不再是默认方案。如需专门做全量实验，显式传入
`--data-path dataset/sft/sft.jsonl`，并使用独立的 `--save-dir` 和
`--output-dir`，避免覆盖 mini 数据方案的模型。

## 5. 先做冒烟训练

正式运行前建议先检查数据、Loss Mask、显存和保存链路：

```bash
python trainer/train_sft.py \
    --model-path out/pretrain \
    --max-train-samples 1000 \
    --eval-samples 100 \
    --max-steps 20 \
    --batch-size 2 \
    --accumulation-steps 2 \
    --eval-interval 10 \
    --save-interval 10 \
    --tracker none
```

重点观察：

- `loss` 和 `lm` 是否为有限值；
- `assistant_tok/s` 是否持续输出；
- `pad_eff` 是否合理；
- 截断比例是否过高；
- MoE 的 `aux` 是否异常增大；
- `checkpoints/sft/latest.pt` 和 `out/sft` 是否正确生成。

### 5.1 Loss 出现 NaN

训练器会在三处主动中止，而不是继续写坏模型：

1. 加载模型后检查所有预训练参数是否含 `NaN/Inf`；
2. 每次反向传播前检查总 Loss、LM Loss、Router Loss，并输出对应 JSONL
   原始行号、序列长度、监督 Token 数和 Logits 的有限值统计；
3. BF16/FP32 更新前检查梯度范数是否有限。

如果第一次日志就是 `loss=nan lm=nan aux=nan`，应立即停止且不要使用该次
SFT 的断点续训。先用单卡、单步复现：

```bash
CUDA_VISIBLE_DEVICES=0 python trainer/train_sft.py \
    --model-path out/pretrain \
    --data-path dataset/sft/sft_mini.jsonl \
    --max-train-samples 64 \
    --eval-samples 16 \
    --max-steps 1 \
    --batch-size 1 \
    --accumulation-steps 1 \
    --num-workers 0 \
    --tracker none
```

- 若启动时报告 `Pretrained model contains non-finite model state`，回退到
  Loss 仍正常的预训练 checkpoint，重新导出 `out/pretrain`；
- 默认 `float32` 基线会显式关闭 TF32，启动日志应显示
  `Precision: float32` 与 `TF32: disabled`；
- SFT 默认不启用梯度检查点，避免在基线前向中引入重算路径。
  只有在基线稳定但显存不足时，才增加 `--gradient-checkpointing`；
- 若只在某些数据上失败，根据 `source_jsonl_rows` 检查对应 JSONL 行。

动态 Padding 批次应使用默认的 Eager Attention：

```bash
python trainer/train_sft.py --attention-backend eager
```

Eager 后端显式构造因果与 Padding Mask，并在 float32 中完成 Q/K/V
投影、RoPE、Attention Score、Softmax、Value 聚合和输出投影；所有
Padding hidden state 会在 Embedding 后及每个残差块后清零。它比融合
SDPA 更占显存、速度更慢，但不会让动态 Padding 的无效位置参与跨层
数值传播。`auto` 和 `math` 只建议在无 Padding 批次或完成充分稳定性
验证后使用。

训练器会自动对第一个 forward 做模块级 NaN/Inf 检查，通过后立即
移除 hooks，不影响后续吞吐。增加 `--debug-numerics` 才会在每个 batch
保留该检查，仅适用于持续性数值问题的诊断运行。

RoPE 频率不保存为 `persistent=False` buffer，而是在真实计算设备上按配置
确定性生成。这避免 Transformers `from_pretrained` 的 meta-device 初始化路径
将未写入 checkpoint 的 RoPE buffer 实例化为未初始化数据。

## 6. 断点续训

自动恢复默认断点：

```bash
python trainer/train_sft.py --resume
```

恢复时会在 batch sampler 层直接跳过已完成批次，不会重新读取和
tokenize 已训练数据。例如日志中的
`Resume data: skipping 192,000 completed batches at sampler level` 表示恢复
位置已直接应用，不是重新扫描 192,000 个 batch。

恢复指定断点：

```bash
python trainer/train_sft.py \
    --resume checkpoints/sft/latest.pt
```

两类加载含义不同：

| 参数 | 含义 |
|---|---|
| `--model-path out/pretrain` | 只加载预训练结构和权重，开始新的 SFT |
| `--resume checkpoints/sft/latest.pt` | 恢复 SFT 模型、优化器、Scheduler、Scaler、训练位置、随机状态和 Tracker |

恢复训练时应保持数据、Batch、梯度累积和总训练步数等关键参数一致。

## 7. 评估

同时计算验证 Loss 并生成回答：

```bash
python eval/eval_sft.py \
    --model-path out/sft \
    --mode both
```

只测试生成：

```bash
python eval/eval_sft.py \
    --model-path out/sft \
    --mode generate \
    --prompt "请解释什么是注意力机制。"
```

打开显式思考提示：

```bash
python eval/eval_sft.py \
    --model-path out/sft \
    --mode generate \
    --open-thinking \
    --prompt "证明根号2是无理数。"
```

如果要测试函数调用，可准备一个工具定义文件：

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "查询天气",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }
]
```

然后运行：

```bash
python eval/eval_sft.py \
    --mode generate \
    --tools-json tools.json \
    --prompt "上海今天的天气怎么样？"
```

## 8. 默认超参数

| 参数 | 默认值 |
|---|---:|
| data_path | dataset/sft/sft_mini.jsonl |
| epochs | 2 |
| max_seq_len | 768 |
| batch_size | 4/GPU |
| accumulation_steps | 4 |
| learning_rate | 1e-5 |
| min_learning_rate | 1e-6 |
| warmup_ratio | 0.0 |
| weight_decay | 0.01 |
| AdamW betas | (0.9, 0.95) |
| grad_clip | 1.0 |
| dtype | float32 |
| attention_backend | eager |
| gradient_checkpointing | false |
| val_ratio | 0.001 |
| eval_samples | 2048 |
| empty_think_ratio | 0.2 |
| system_prompt_ratio | 0.2 |

模型结构、MoE Expert 数、Top-K、RoPE 和 Router 辅助损失系数全部从预训练模型配置继承，不在 SFT 命令中重新定义。

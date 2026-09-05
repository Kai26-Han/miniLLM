# miniLLM DPO 训练

> 阶段：在完成全参数 SFT 的 miniLLM 上进行偏好对齐

> 默认方案：对齐 MiniMind 官方 DPO 配置，标准 DPO、Beta 0.15、学习率 4e-8、无 Warmup、BF16、全量数据 1 轮、冻结 SFT 参考模型

---

## 1. DPO 训练要解决什么

SFT 使用标准答案教模型“应该怎样回答”，但同一个问题通常存在多个看似合理、
质量却不同的答案。DPO 使用成对偏好数据，直接告诉模型：

```text
对于同一个 Prompt：chosen 比 rejected 更好
```

它可以进一步调整模型的回答风格、安全性、完整性和指令遵循能力。DPO 不需要
额外训练 Reward Model，也不需要像 PPO 一样在训练循环中持续采样。其核心是让
策略模型相对参考模型提高 chosen 回答的优势，同时降低 rejected 回答的优势。

本项目的完整训练顺序是：

```text
随机初始化
    ↓ 预训练
out/pretrain
    ↓ 全参数 SFT
out/sft
    ├── 复制为可训练 Policy
    └── 复制为冻结 Reference
              ↓
      chosen / rejected 偏好对
              ↓ Standard DPO Loss
           out/dpo
```

DPO 必须从已经具备稳定对话能力的 `out/sft` 开始。直接从 `out/pretrain` 进行
DPO，偏好数据需要同时承担指令学习和偏好对齐两个目标，通常无法得到可靠结果。

## 2. DPO 的基本原理

给定同一个 Prompt `x`、偏好回答 `y_w` 和非偏好回答 `y_l`，分别计算策略模型
`πθ` 与冻结参考模型 `πref` 的回答序列 Log Probability：

```text
log π(y|x) = 所有回答 Token 的 next-token log probability 之和
```

策略模型和参考模型各自形成 chosen/rejected 的对数概率差：

```text
policy_logratio = log πθ(y_w|x)  - log πθ(y_l|x)
ref_logratio    = log πref(y_w|x) - log πref(y_l|x)
```

最终偏好间隔与 DPO Loss 为：

```text
margin   = policy_logratio - ref_logratio
dpo_loss = -(1-ε) log sigmoid(beta × margin)
           - ε log sigmoid(-beta × margin)
```

默认 `ε=0`，与 MiniMind 一致，使用标准 DPO。只有在已经确认偏好标签存在噪声、
并准备单独进行 Conservative DPO 对照实验时，才显式设置非零
`--label-smoothing`。

当前 miniLLM 是 MoE 模型，因此实际反向传播目标还包括 Router 辅助损失：

```text
total_loss = dpo_loss + router_aux_loss
```

参考模型只提供稳定的相对基准，始终保持 `eval()` 和 `requires_grad_(False)`。
它没有优化器，不计算梯度，也不会写入最终 DPO 模型。

DPO 的推导和算法细节可参考
[Direct Preference Optimization 原论文](https://arxiv.org/abs/2305.18290)，
本项目的整体训练形式参考了 MiniMind 的 DPO 实现，但针对当前工程补充了动态
Padding、验证集、严格数据检查、断点续训和更完整的偏好指标。

## 3. 相关文件

| 文件 | 作用 |
|---|---|
| [`dataset/dpo_dataset.py`](../dataset/dpo_dataset.py) | 偏好对校验、ChatML 渲染、最终回答 Mask、成对截断和动态 Padding |
| [`trainer/train_dpo.py`](../trainer/train_dpo.py) | Policy/Reference 前向、DPO Loss、DDP、验证、断点和模型导出 |
| [`eval/eval_dpo.py`](../eval/eval_dpo.py) | 在同一偏好验证集上独立评估 DPO Policy 和 SFT Reference |
| [`dataset/rl/dpo.jsonl`](../dataset/rl/dpo.jsonl) | chosen/rejected 偏好训练数据 |
| [`model/model_minillm.py`](../model/model_minillm.py) | miniLLM Causal LM 与 MoE Router 辅助损失 |
| [`trainer/trainer_utils.py`](../trainer/trainer_utils.py) | 分布式、Scheduler、Checkpoint、Tracker 和 Transformers 导出 |
| [`tests/test_dpo_dataset.py`](../tests/test_dpo_dataset.py) | 偏好数据、Mask、截断和 Collator 测试 |
| [`tests/test_dpo_objective.py`](../tests/test_dpo_objective.py) | 序列 Log Probability 和 DPO 公式测试 |

本项目使用原生 PyTorch 实现 DPO，不依赖 TRL。

## 4. 数据格式

每行必须包含 `chosen` 和 `rejected` 两个完整对话数组：

```json
{
  "chosen": [
    {"role": "user", "content": "如何养成稳定的阅读习惯？"},
    {"role": "assistant", "content": "可以从每天固定阅读十分钟开始……"}
  ],
  "rejected": [
    {"role": "user", "content": "如何养成稳定的阅读习惯？"},
    {"role": "assistant", "content": "多看书就可以了。"}
  ]
}
```

数据加载器强制要求：

1. `chosen` 和 `rejected` 都以 `assistant` 消息结束；
2. 除最后一条 Assistant 回复外，两边的历史消息必须完全一致；
3. 最终回复经过 Tokenize 后至少包含一个可监督 Token；
4. 角色、`reasoning_content`、`tools` 和 `tool_calls` 必须符合 SFT 数据规范；
5. 非法样本会报告原始 JSONL 行号并立即停止，而不会静默跳过。

多轮偏好数据也可以使用：

```text
system → user → assistant → user → chosen assistant
                              └──→ rejected assistant
```

前面的 Assistant 历史只作为公共上下文，不参与当前偏好分数。训练仅对最后一个
chosen/rejected Assistant 回复计算序列 Log Probability。

### 4.1 当前数据审计

当前 [`dataset/rl/dpo.jsonl`](../dataset/rl/dpo.jsonl) 的审计结果为：

| 项目 | 结果 |
|---|---:|
| 偏好对数量 | 17,166 |
| 公共 Prompt 完全一致 | 17,166 |
| 两边均以 Assistant 结束 | 17,166 |
| 当前对话结构 | 全部为 user → assistant |
| chosen 平均回答 Token | 358.97 |
| rejected 平均回答 Token | 332.90 |

使用当前 Tokenizer 和 `max_seq_len=1024` 全量编码时：

| 项目 | 数量 | 比例 |
|---|---:|---:|
| chosen 序列发生截断 | 1,158 | 6.75% |
| rejected 序列发生截断 | 1,158 | 6.75% |
| chosen 回答本身被截断 | 578 | 3.37% |
| rejected 回答本身被截断 | 623 | 3.63% |

如果显存允许，可以实验 `--max-seq-len 1536` 或 `2048`，但序列长度增加会显著
提高 Policy 和 Reference 两次前向的计算量与显存占用。

### 4.2 成对截断

DPO 比普通 SFT 更强调 chosen/rejected 条件的一致性。两边不能独立地随意裁剪
Prompt，否则它们不再表示同一个条件概率比较。

当前策略是：

1. 先确认两边最终回复前的 Token 前缀完全一致；
2. 短回答优先完整保留，并把剩余窗口分配给最近的公共 Prompt；
3. 回答很长时，至少为公共 Prompt 保留约 1/4 窗口，最多 256 Token；
4. chosen/rejected 使用相同的 Prompt 起始位置；
5. 回答仍超过窗口时保留回答开头，不伪造提前结束的 EOS；
6. 日志分别记录两边的序列截断率和回答截断率。

### 4.3 动态 Padding 与拼批

每个 Batch 会先找出 chosen/rejected 中的最长序列，再统一右侧 Padding 到 8 的
倍数。设原始偏好 Batch Size 为 `B`，模型实际看到：

```text
[chosen_0 ... chosen_B-1, rejected_0 ... rejected_B-1]
```

因此一次 Policy 前向和一次 Reference 前向的实际序列数都是 `2B`。日志中的
`batch_size` 和 `global_batch` 表示偏好对数量，不是模型前向的序列数量。

## 5. Policy 与 Reference

训练开始时，两者都从同一个 `out/sft` 加载：

```text
Policy    = deepcopy(SFT 权重)，全参数可训练
Reference = 同一份 SFT 权重，完全冻结
```

在第一次参数更新前，两者对任何输入的 Log Probability 应相同。因此：

```text
margin = 0
dpo_loss = -log sigmoid(0) = log(2) ≈ 0.693147
```

初始 DPO Loss 接近 `0.6931` 是正常现象，不表示训练失败。总 Loss 会因为加入
`router_aux_loss` 而略高于 `0.6931`。

为降低长回答在 BF16/FP16 下累计产生的数值误差，模型主体默认使用 BF16，
但 `log_softmax` 和回答 Token 的 Log Probability 求和固定使用 FP32。

### 5.1 显存组成

DPO 全参数训练需要同时保存：

- 一份可训练 Policy 权重；
- Policy 的梯度；
- AdamW Optimizer 状态；
- 一份冻结 Reference 权重；
- chosen/rejected 的 Policy 激活；
- 当前 Batch 的 Policy 和 Reference Logits。

Reference 不保存梯度和反向激活，但仍需要权重与前向显存。相比 SFT，DPO 的
显存和计算开销都会明显增加。MiniMind 默认每卡 Batch Size 为 4，当前工程也
采用该默认值。由于每个偏好对包含 chosen 和 rejected 两条序列，单卡实际前向
序列数为 8。若 1024 长序列触发显存不足，应先把 `--batch-size` 降到 2 或 1，
再用梯度累积恢复目标全局 Batch。

## 6. 开始训练

先确认 SFT 已成功导出：

```text
out/sft/
├── config.json
├── model.safetensors
├── tokenizer.json
└── chat_template.jinja
```

如果没有 `out/sft/config.json`，训练器会直接拒绝启动。

### 6.1 单卡训练

```bash
python trainer/train_dpo.py \
    --model-path out/sft \
    --tokenizer-path out/sft \
    --data-path dataset/rl/dpo.jsonl \
    --output-dir out/dpo \
    --save-dir checkpoints/dpo \
    --epochs 1 \
    --batch-size 4 \
    --accumulation-steps 1 \
    --max-seq-len 1024 \
    --learning-rate 4e-8 \
    --min-learning-rate 4e-9 \
    --warmup-ratio 0 \
    --beta 0.15 \
    --label-smoothing 0 \
    --dtype bfloat16 \
    --tracker swanlab
```

默认单卡有效全局 Batch Size：

```text
4 × 1 × 1 = 4 个偏好对/更新
```

但每次微批前向实际包含：

```text
4 chosen + 4 rejected = 8 条序列
```

### 6.2 六卡训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun --standalone --nproc_per_node=6 trainer/train_dpo.py \
    --model-path out/sft \
    --data-path dataset/rl/dpo.jsonl \
    --tracker swanlab
```

此时有效全局 Batch Size 为：

```text
4 × 1 × 6 = 24 个偏好对/更新
```

该配置与 MiniMind 一样不做梯度累积，学习率为 `4e-8`，采用无 Warmup 的余弦
衰减并在训练末端降到 `4e-9`。

多卡只对 Policy 使用 DDP。每个 Rank 各自持有一份完整冻结 Reference，避免
Reference 参与不必要的梯度同步。

## 7. 先做冒烟训练

正式训练前先用少量偏好对验证数据、双模型前向、反向、保存和评估链路：

```bash
python trainer/train_dpo.py \
    --model-path out/sft \
    --data-path dataset/rl/dpo.jsonl \
    --max-train-samples 64 \
    --eval-samples 16 \
    --max-steps 5 \
    --batch-size 1 \
    --accumulation-steps 1 \
    --max-seq-len 512 \
    --num-workers 0 \
    --dtype float32 \
    --log-interval 1 \
    --eval-interval 5 \
    --save-interval 5 \
    --tracker none
```

重点确认：

- Policy 与 Reference 均从 `out/sft` 加载；
- Policy 参数显示为 `trainable`，Reference 显示为 `frozen`；
- 第一次 `dpo` 接近 `0.6931`；
- `total_loss`、梯度范数和序列 Log Probability 都是有限值；
- `checkpoints/dpo/latest.pt` 成功生成；
- 训练结束后 `out/dpo/model.safetensors` 成功导出。

正式训练默认使用 `4e-8`；冒烟训练步数很少，指标变化不明显属于正常现象。
冒烟测试的目标是验证管线，不是验证最终收敛效果。

## 8. 训练指标

训练器向终端及 SwanLab/W&B 记录以下核心指标：

| 指标 | 含义 |
|---|---|
| `train/dpo_loss` | 标准 DPO 偏好分类损失 |
| `train/total_loss` | DPO Loss 与 MoE Router 辅助损失之和 |
| `train/reward_chosen` | Policy 相对 Reference 对 chosen 的隐式奖励 |
| `train/reward_rejected` | Policy 相对 Reference 对 rejected 的隐式奖励 |
| `train/reward_margin` | chosen reward 减 rejected reward |
| `train/preference_accuracy` | reward margin 大于 0 的偏好对比例 |
| `train/policy_chosen_logp` | Policy 的 chosen 序列 Log Probability |
| `train/policy_rejected_logp` | Policy 的 rejected 序列 Log Probability |
| `train/router_aux_loss` | MoE Expert 负载均衡辅助损失 |
| `data/padding_efficiency` | 有效 Token 占 Padding 后 Token 槽位的比例 |
| `data/*_truncated_ratio` | chosen/rejected 序列截断比例 |
| `data/*_answer_truncated_ratio` | chosen/rejected 回答本身被截断的比例 |

验证集会记录相同的 `validation/*` 指标。判断是否有效时重点观察：

```text
validation/reward_margin          应逐步增大
validation/preference_accuracy    应高于随机水平并趋于稳定
validation/dpo_loss               应下降，但不要求快速下降
```

不能只看训练集 `dpo_loss`。如果训练准确率持续上升、验证准确率却下降，说明模型
正在记忆偏好对，应减少训练步数或降低学习率。

## 9. 默认超参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `epochs` | 1 | 偏好数据通常只训练少量轮次 |
| `batch_size` | 4/GPU | 对齐 MiniMind；每次模型前向实际为 8 条序列/GPU |
| `accumulation_steps` | 1 | 对齐 MiniMind，不做梯度累积 |
| `learning_rate` | 4e-8 | 对齐 MiniMind DPO 初始学习率 |
| `min_learning_rate` | 4e-9 | Cosine 末端为初始学习率的 10% |
| `warmup_ratio` | 0 | 对齐 MiniMind，无 Warmup |
| `weight_decay` | 0.01 | AdamW 权重衰减 |
| `adam_beta1` | 0.9 | 对齐 MiniMind 使用的 AdamW 默认值 |
| `adam_beta2` | 0.999 | 对齐 MiniMind 使用的 AdamW 默认值 |
| `grad_clip` | 1.0 | 梯度范数裁剪 |
| `beta` | 0.15 | 对齐 MiniMind DPO 默认值 |
| `label_smoothing` | 0 | 对齐 MiniMind，使用标准 DPO |
| `max_seq_len` | 1024 | 每个 chosen/rejected 分支的上限 |
| `val_ratio` | 0.05 | 当前约 858 个验证偏好对 |
| `eval_samples` | 1000 | 当前验证集不足 1000 时自动使用全部 |
| `eval_batch_size` | 2/GPU | 验证无反向，可使用更大的微批 |
| `early_stopping_patience` | 0 | 默认关闭，不改变 MiniMind 的完整一轮训练行为 |
| `dtype` | bfloat16 | 对齐 MiniMind，适合 RTX 4090 |
| `attention_backend` | eager | 动态 Padding 的稳定默认后端 |
| `gradient_checkpointing` | false | 显存不足时再开启 |

### 9.1 Beta 怎么选择

`beta` 控制 DPO 对相对偏好间隔的缩放：

- 较小 Beta：对 Reference 的约束更弱，允许 Policy 偏离得更远，但初始梯度缩放也更小；
- 较大 Beta：对 Reference 的约束更强，同时会放大当前偏好 Logit 的梯度尺度；
- 当前默认 `0.15`，与 MiniMind 官方实现一致。

建议先固定其他参数，对 `0.05、0.10、0.15、0.20` 做小规模对照，并同时比较
偏好验证集和固定通用能力测试集。

### 9.2 学习率怎么选择

默认 `4e-8` 与 MiniMind 配置一致。如果清洗后的数据仍需要做学习率对照，建议只
在较窄范围内尝试：

```text
2e-8 → 4e-8 → 5e-8
```

每次只改变一个变量。不要同时大幅提高学习率、Beta 和 Epoch，否则无法判断是哪
个因素造成能力提升或退化。

## 10. 断点续训与输出

训练中保存：

```text
checkpoints/dpo/
├── latest.pt
└── best.pt
```

`latest.pt` 用于断点续训；`best.pt` 保存验证 DPO Loss 最低的模型。训练结束时
`out/dpo` 从 `best.pt` 导出，而不是无条件导出最后一步。新训练还会在 step 0
检查 Policy 与 Reference 的 Log Probability 是否一致，并把 SFT 基线保存为第一个
候选最佳模型。

Checkpoint 包含：

- Policy 参数；
- Optimizer、Scheduler 和 GradScaler；
- Epoch、Batch 位置和 Global Step；
- Python、NumPy、PyTorch 随机状态；
- Tracker Run 信息；
- 最佳验证 Loss、最佳 Step 和 Early Stopping 计数；
- 本次 DPO 参数与模型配置。

Reference 不保存进 Checkpoint。恢复时会重新从原始 `out/sft` 加载，因此不要在
同一次 DPO 实验中覆盖或更换该目录。

自动恢复：

```bash
python trainer/train_dpo.py --resume
```

恢复指定断点：

```bash
python trainer/train_dpo.py \
    --resume checkpoints/dpo/latest.pt
```

训练器会拒绝在恢复时修改模型、序列长度、数据划分、DPO Objective、Batch、学习率
或 Warmup 等关键参数，避免 Policy Checkpoint 与 Reference 或优化器轨迹不匹配。

最终导出：

```text
out/dpo/
├── config.json
├── model.safetensors
├── generation_config.json
└── tokenizer files...
```

`out/dpo` 是标准 Transformers 模型目录，可以直接供现有生成和后续训练代码加载。

## 11. 独立评估

训练完成后，在确定性划分的验证偏好对上比较 DPO Policy 与原始 SFT Reference：

```bash
python eval/eval_dpo.py \
    --policy-path out/dpo \
    --reference-path out/sft \
    --data-path dataset/rl/dpo.jsonl \
    --split validation \
    --val-ratio 0.05 \
    --eval-samples 1000 \
    --batch-size 2 \
    --beta 0.15 \
    --label-smoothing 0 \
    --dtype bfloat16
```

评估输出为 JSON，方便保存和比较多个实验：

```bash
python eval/eval_dpo.py \
    --policy-path out/dpo \
    --reference-path out/sft \
    > dpo_eval.json
```

也可以先把 Policy 指向 `out/sft` 检查基线：

```bash
python eval/eval_dpo.py \
    --policy-path out/sft \
    --reference-path out/sft
```

此时 `dpo_loss` 应接近 `0.693147`，`reward_margin` 应接近 0。

DPO 离线指标只表示模型更倾向数据集中的 chosen，不能完全替代生成质量评估。
还应使用一组固定 Prompt 对比 `out/sft` 和 `out/dpo`：

- 通用问答、翻译、写作和代码能力是否退化；
- 回答是否变得过长、模板化或拒答过多；
- Tool Calling 与 `<think>` 格式是否仍然稳定；
- chosen 的改进是否符合真实偏好，而不是利用数据集中的长度或措辞偏差。

## 12. 常见问题

### 12.1 Loss 一直是 0.6931

训练刚开始时这是理论正确值。默认学习率很小，终端只显示 4 位小数，前几十步
看起来不变也可能是正常的。优先观察更多小数位的 `reward_margin`、梯度范数和
验证集曲线。

如果完整一轮后所有指标仍严格不变，再检查：

- 学习率是否被错误覆盖为 0；
- Policy 参数是否确实为 `requires_grad=True`；
- 梯度范数是否持续为 0；
- chosen/rejected 是否大量重复或内容完全相同；
- 是否误把已训练的 DPO 模型同时作为 Policy 和 Reference 重新开始实验。

### 12.2 CUDA OOM

按以下顺序调整：

1. `--batch-size 4` 降到 `2` 或 `1`；
2. 用 `--accumulation-steps` 恢复所需的全局 Batch；
3. `--max-seq-len 1024` 降到 `768` 或 `512`；
4. 增加 `--gradient-checkpointing`。

降低 `max_seq_len` 后必须重新观察回答截断率。显存恢复不能以大量截掉 chosen
和 rejected 的关键差异为代价。如果 BF16 出现非有限值，保持 eager attention，
并显式使用 `--dtype float32 --batch-size 1` 做数值稳定性复查。

### 12.3 Preference Accuracy 很快接近 100%

如果训练准确率快速接近 100%，验证准确率却没有同步提高，通常是过拟合。可以：

- 减少 `max_steps` 或 Epoch；
- 降低学习率；
- 降低 Beta；
- 对训练与验证偏好对去重；
- 检查 rejected 是否存在过于明显的低质量模式。

### 12.4 DPO 后通用能力下降

DPO 优化的是静态偏好分布，不保证所有 SFT 能力都同步提升。出现退化时：

- 优先回退到更早的 DPO Checkpoint；
- 降低学习率或只训练部分步数；
- 清理错误、过度单一或长度偏置明显的偏好数据；
- 对固定 SFT 验证集继续运行 `eval/eval_sft.py`；
- 同时比较生成结果，而不是只看 DPO Accuracy。

### 12.5 NaN 或 Inf

训练器会在反向传播前检查：

- 总 Loss、DPO Loss 和 Router Loss；
- chosen/rejected reward；
- Policy 与 Reference 序列 Log Probability。

发生异常时会输出对应 `source_jsonl_rows`。先用单卡、`batch-size=1`、
`num-workers=0` 和 `dtype=float32` 复现，再检查该行回答长度与内容。

## 13. 运行测试

在已经安装 PyTorch、Transformers 和 Datasets 的环境中执行：

```bash
python -m unittest discover \
    -s tests \
    -p 'test_dpo_*.py' \
    -v
```

测试覆盖：

- Policy 与 Reference 相同时，DPO Loss 等于 `log(2)`；
- chosen 优势增大时 Loss 下降；
- next-token Shift 和回答 Mask 正确；
- 历史 Assistant 消息不会被计入当前偏好分数；
- chosen/rejected Prompt 不一致时立即报错；
- 成对截断后两边保留相同的公共 Prompt；
- Collator 按 chosen 在前、rejected 在后的顺序拼批；
- 微型 MoE 前后向中只有 Policy 产生梯度。

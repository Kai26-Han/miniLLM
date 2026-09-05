# miniLLM GRPO 训练

> 阶段：在完成 DPO 的 miniLLM 上继续进行在线强化学习

> 默认方案：每个 Prompt 采样 4 个回答、InternLM2-1.8B-Reward、Beta 0.1、Clip 0.2、学习率 1e-7

---

## 1. GRPO 训练要解决什么

DPO 只能学习固定 `chosen/rejected` 偏好对。GRPO 会让当前 Policy 在线生成回答，
再根据 Reward 调整这些回答的概率，因此可以继续探索 DPO 数据中没有覆盖的行为。

本项目的训练顺序是：

```text
out/sft
   ↓ DPO
out/dpo
   ├── Policy：全参数训练
   └── Reference：冻结，提供 KL 约束
              ↓
dataset/rl/rlaif.jsonl 中的 Prompt
              ↓ 每题生成 G 个回答
Reward Model + 规则奖励
              ↓ 组内相对 Advantage
GRPO Clip Loss + Reference KL + MoE Router Loss
              ↓
out/grpo
```

GRPO 不需要 PPO 的 Critic/Value Model，但每个 Prompt 要生成多条回答，主要成本通常
来自在线 Rollout 和 Reward Model 推理。

## 2. 相关文件

| 文件 | 作用 |
|---|---|
| [`trainer/train_grpo.py`](../trainer/train_grpo.py) | Reward、分组 Advantage、GRPO Loss、训练、验证、断点和导出 |
| [`dataset/ppo_dataset.py`](../dataset/ppo_dataset.py) | Prompt-only 数据校验、模板渲染、截断和确定性划分 |
| [`trainer/rollout_engine.py`](../trainer/rollout_engine.py) | Policy 在线生成、Completion Mask 和行为策略 Log Probability |
| [`dataset/rl/rlaif.jsonl`](../dataset/rl/rlaif.jsonl) | GRPO Prompt 数据 |
| [`tests/test_grpo.py`](../tests/test_grpo.py) | GRPO 公式、Reward、Mask 和分组顺序测试 |

本项目使用原生 PyTorch 实现 GRPO，不依赖 TRL。整体训练方式参考 MiniMind 的
`train_grpo.py`，并复用当前 miniLLM 已有的 Dataset、Rollout、DDP、Tracker 和
Checkpoint 基础设施。

## 3. GRPO 原理

对每个 Prompt 生成 `G` 个回答，并得到奖励 `R1...RG`。组内 Advantage 为：

```text
Ai = (Ri - group_mean) / (group_std + 1e-4)
```

高于组内平均 Reward 的回答得到正 Advantage，低于平均值的回答得到负 Advantage。
每个回答的 Advantage 会分配给该回答的所有有效 Completion Token。

当前 Policy 与生成回答时的行为策略之间的概率比为：

```text
ratio = exp(current_log_prob - old_log_prob)
```

GRPO Policy Loss 使用 PPO 风格裁剪：

```text
surrogate = min(
    ratio × advantage,
    clip(ratio, 1-epsilon, 1+epsilon) × advantage
)
```

Reference KL 使用 MiniMind 同类实现中的非负逐 Token 估计：

```text
log_ratio = reference_log_prob - current_log_prob
kl = exp(log_ratio) - log_ratio - 1
```

最终训练目标：

```text
total_loss = grpo_policy_loss
           + beta × reference_kl
           + router_aux_loss
```

只有 Completion Token 参与 GRPO Loss。首个 EOS 会参与训练，EOS 后用于补齐批次的
Token 和 Padding 不参与 Loss。

### 3.1 零方差组

如果同一 Prompt 的所有回答 Reward 相同：

```text
R1 = R2 = ... = RG
```

该组所有 Advantage 都是 0，不产生有效 Policy Gradient。这通常意味着：

- 问题太简单，所有回答都获得相同高分；
- 问题太难，所有回答都获得相同低分；
- Reward Model 无法区分回答；
- 采样温度过低，多个回答高度相似。

训练时应重点观察 `zero_std_group_rate`。

## 4. Reward 组成

当前奖励沿用 MiniMind 的简洁组合：

```text
total_reward = reward_model
             + length_reward
             + thinking_reward
             - repetition_penalty
```

### 4.1 Reward Model

默认使用 `InternLM2-1.8B-Reward`。模型分数被裁剪到 `[-3, 3]`，异常的
`NaN/Inf` 分数会立即终止训练。

Reward Model 只负责打分：

- 始终保持冻结和 `eval` 模式；
- CUDA 上使用 FP16；
- 不保存进 GRPO Checkpoint；
- 恢复训练时从原路径重新加载。

### 4.2 规则奖励

- 回答长度在 20～800 字符之间：`+0.5`，否则 `-0.5`；
- 出现单个 `</think>` 且思考内容长度合理：最高 `+1.25`；
- 重复 trigram：最多扣 `0.5`。

规则奖励只能用于约束明显格式问题，不能代替真实回答质量。应定期人工检查高
Reward 回答，防止模型通过堆叠长度、模板或特定词语获得虚假高分。

## 5. 数据格式

默认数据是 [`dataset/rl/rlaif.jsonl`](../dataset/rl/rlaif.jsonl)，共 19,502 条。
每行包含完整 `conversations`，最后一条必须是空 Assistant 占位符：

```json
{
  "conversations": [
    {"role": "user", "content": "请解释什么是光合作用。"},
    {"role": "assistant", "content": ""}
  ]
}
```

训练时会移除空 Assistant，由当前 Policy 重新生成回答。数据集本身没有
`chosen/rejected` 或 Ground Truth，因此不能脱离 Reward Model 直接训练。

多轮历史、system、tool 定义都可以保留。Prompt 过长时优先删除较早的完整轮次，
然后在必要时保留最近 Token 后缀。默认验证比例是 `0.02`。

## 6. 准备模型

先完成 DPO，并确认以下文件存在：

```text
out/dpo/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

再下载 `InternLM2-1.8B-Reward`，推荐与 miniLLM 放在同级目录：

```text
Project/
├── miniLLM/
└── internlm2-1_8b-reward/
    ├── config.json
    ├── model.safetensors
    └── tokenizer files...
```

安装依赖：

```bash
pip install -r requirements.txt
```

训练器只从本地加载 Policy、Reference、Tokenizer 和 Reward Model，不会在启动时
自动从网络下载缺失文件。

## 7. 先做冒烟训练

正式训练前先验证模型加载、分组 Rollout、Reward、反向和保存链路：

```bash
python trainer/train_grpo.py \
    --model-path out/dpo \
    --reward-model-path ../internlm2-1_8b-reward \
    --max-train-samples 8 \
    --eval-samples 2 \
    --max-steps 2 \
    --batch-size 1 \
    --num-generations 2 \
    --accumulation-steps 1 \
    --max-prompt-len 256 \
    --max-new-tokens 64 \
    --num-workers 0 \
    --log-interval 1 \
    --eval-interval 2 \
    --save-interval 2 \
    --tracker none
```

重点确认：

- Policy 和 Reference 都从 `out/dpo` 加载；
- Reward Model 成功加载且全部参数冻结；
- `reward`、`loss`、`kl` 和梯度范数都是有限值；
- `checkpoints/grpo/latest.pt` 成功生成；
- 结束后 `out/grpo/model.safetensors` 成功导出。

两步训练不能证明模型已经收敛，只用于验证完整管线。

## 8. 正式训练

### 8.1 单卡

```bash
python trainer/train_grpo.py \
    --model-path out/dpo \
    --reference-path out/dpo \
    --reward-model-path ../internlm2-1_8b-reward \
    --data-path dataset/rl/rlaif.jsonl \
    --output-dir out/grpo \
    --save-dir checkpoints/grpo \
    --epochs 1 \
    --batch-size 1 \
    --num-generations 4 \
    --accumulation-steps 8 \
    --max-prompt-len 512 \
    --max-new-tokens 256 \
    --learning-rate 1e-7 \
    --beta 0.1 \
    --epsilon 0.2 \
    --dtype bfloat16 \
    --tracker swanlab
```

`--reference-path` 默认等于 `--model-path`，因此可以省略。训练开始时 Policy 与
Reference 完全相同，初始 KL 应接近 0。

### 8.2 双卡

```bash
torchrun --nproc_per_node 2 trainer/train_grpo.py \
    --model-path out/dpo \
    --reward-model-path ../internlm2-1_8b-reward \
    --batch-size 1 \
    --num-generations 4 \
    --accumulation-steps 4 \
    --dtype bfloat16 \
    --tracker swanlab
```

每张 GPU 都会持有完整 Policy、Reference 和 Reward Model。DDP 只同步 Policy
梯度，不同步 Reference 和 Reward Model。

## 9. 默认参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `epochs` | 1 | 在线 RL 默认只训练少量轮次 |
| `batch_size` | 1/GPU | 每卡 Prompt 数，不是生成回答数 |
| `num_generations` | 4 | 每个 Prompt 的回答数量 |
| `accumulation_steps` | 8 | 梯度累积的 Prompt Batch 数 |
| `learning_rate` | 1e-7 | DPO 后继续训练的保守学习率 |
| `min_learning_rate` | 1e-8 | Cosine 末端学习率 |
| `beta` | 0.1 | Reference KL 权重 |
| `epsilon` | 0.2 | GRPO Ratio Clip 范围 |
| `max_prompt_len` | 512 | Prompt Token 上限 |
| `max_new_tokens` | 256 | 单个回答生成 Token 上限 |
| `temperature` | 0.8 | Rollout 采样温度 |
| `top_p` | 0.9 | Nucleus Sampling |
| `top_k` | 50 | Top-k Sampling |
| `thinking_ratio` | 0.0 | 默认不开启显式思考 |
| `val_ratio` | 0.02 | 确定性验证集比例 |
| `eval_samples` | 32 | 在线验证 Prompt 数量 |

每次前向的实际回答数为：

```text
batch_size × num_generations
```

有效 Prompt Group 数为：

```text
batch_size × accumulation_steps × GPU 数量
```

增加 `num_generations` 可以提高组内比较质量，但 Rollout、Reward 和训练前向成本
近似线性增长。

## 10. 训练指标

| 指标 | 含义 |
|---|---|
| `train/reward` | 总奖励均值 |
| `train/reward_model` | Reward Model 分数均值 |
| `train/reward_length` | 长度规则奖励 |
| `train/reward_thinking` | Thinking 格式奖励 |
| `train/repetition_penalty` | 重复惩罚，越低越好 |
| `train/policy_loss` | GRPO Policy Loss |
| `train/kl_penalty` | 非负 Reference KL 惩罚 |
| `train/router_aux_loss` | MoE Router 负载均衡 Loss |
| `train/zero_std_group_rate` | 组内 Reward 无差异的比例 |
| `train/clip_fraction` | Ratio 超出 Clip 范围的 Token 比例 |
| `train/eos_rate` | 回答正常生成 EOS 的比例 |
| `train/response_length` | 平均有效回答 Token 数 |

验证阶段还会记录 `validation/reward`、`validation/sampled_kl`、
`validation/eos_rate` 和各 Reward 分量。

不能只根据训练 Reward 选择模型。Reward 上升但重复率、输出长度或真实回答质量
恶化，通常表示 Reward Hacking。

## 11. 断点续训与输出

训练状态保存在：

```text
checkpoints/grpo/latest.pt
```

自动恢复：

```bash
python trainer/train_grpo.py \
    --model-path out/dpo \
    --reward-model-path ../internlm2-1_8b-reward \
    --resume
```

训练器会拒绝在恢复时修改 Policy、Reference、Reward Model、`num_generations`、
最大长度、Beta 或 Epsilon，防止新旧训练语义不一致。

训练完成后导出：

```text
out/grpo/
├── config.json
├── model.safetensors
├── generation_config.json
└── tokenizer files...
```

## 12. 常见问题

### 12.1 CUDA OOM

依次尝试：

1. `num_generations` 从 4 降到 2；
2. `max_new_tokens` 从 256 降到 128 或 64；
3. `max_prompt_len` 从 512 降到 384 或 256；
4. 增加 `--gradient-checkpointing`；
5. 保持 `batch-size=1`，通过累积步数维持有效 Group 数。

GRPO 同时持有 Policy、Reference、Reward Model 和在线回答，通常比 DPO 更容易
出现 OOM。

### 12.2 `zero_std_group_rate` 很高

- 增加采样温度或适当提高 `top_p`；
- 将 `num_generations` 从 2 增加到 4 或 6；
- 清理过易或超出模型能力的问题；
- 检查 Reward Model 是否把不同回答打成相同分数。

### 12.3 KL 快速增大

- 降低学习率；
- 增大 `beta`；
- 减少训练步数；
- 检查 Reward 是否存在容易利用的规则漏洞。

### 12.4 Reward 上升但回答变差

人工检查以下样本：

- Reward 最高的回答；
- 最长和重复最多的回答；
- Reward 突然跃升前后的回答；
- GRPO 与 DPO 输出差异最大的固定 Prompt。

同时继续运行现有 SFT/DPO 评测，防止通用能力、偏好对齐和 Tool Calling 退化。

## 13. 运行测试

在安装 PyTorch、Transformers 和 Datasets 的环境中运行：

```bash
python -m unittest tests.test_grpo -v
```

测试覆盖：

- Prompt 按组连续展开；
- 每组 Advantage 均值为 0；
- 零方差组 Advantage 为 0；
- 正负 Advantage 产生相反梯度；
- 正 Advantage 的过大 Ratio 被裁剪；
- Padding 不参与 GRPO Loss；
- Reward 分量与回答顺序正确；
- Thinking 和重复规则奖励正确。

## 14. 参考资料

- [MiniMind `train_grpo.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_grpo.py)
- [DeepSeekMath：Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)

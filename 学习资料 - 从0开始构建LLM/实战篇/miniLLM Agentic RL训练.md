# miniLLM Agentic RL 训练

> 阶段：在完成 PPO 的 miniLLM Actor 上继续训练多轮 Tool-Use 能力

> 默认方案：多轮工具环境、GRPO、每个 Prompt 采样 4 条轨迹、冻结 PPO Actor 作为 Reference

---

## 1. Agentic RL 要解决什么

普通 PPO 主要优化一次回答的整体质量，Agentic RL 则把模型放入一个可交互环境：

```text
用户问题
    ↓
Assistant 生成 <tool_call>
    ↓
环境校验并执行工具
    ↓
Tool observation 写回上下文
    ↓
Assistant 继续调用工具或给出最终答案
    ↓
GT Verifier 计算轨迹奖励
```

训练目标不只是“答案看起来合理”，还包括：

- 是否选择了可用工具；
- Tool Call JSON 和参数是否有效；
- 工具是否成功执行；
- 是否根据工具结果给出正确答案；
- 是否能在有限轮次内停止；
- 是否存在重复、无效或奖励投机行为。

本项目参考
[MiniMind Agentic RL](https://github.com/jingyaogong/minimind/blob/master/trainer/train_agent.py)
的集中式实现，将工具、Reward、多轮 Rollout、GRPO/CISPO 和训练循环放在一个
入口中，同时补充安全数学执行、严格 Token Mask、验证集和断点续训。

## 2. 完整训练顺序

```text
out/pretrain
    ↓ SFT
out/sft
    ↓ DPO（可选）
out/dpo
    ↓ PPO
out/ppo 或 checkpoints/ppo/latest.pt
    ├── 可训练 Agent Policy
    └── 冻结 Reference Policy
              ↓
       多轮 Tool-Use GRPO
              ↓
           out/agent
```

Agentic RL 应从已经具备基础 Tool Call 能力的 PPO Actor 开始。如果 PPO 后的模型
几乎无法输出合法 `<tool_call>`，同组轨迹可能全部失败，GRPO 将得不到有效的组内
差异。正式训练前应先运行 `--eval-only` 检查工具调用基线。

## 3. 相关文件

| 文件 | 作用 |
|---|---|
| [`trainer/train_agent.py`](../trainer/train_agent.py) | 工具环境、多轮 Rollout、Reward、GRPO/CISPO、训练、评测和导出 |
| [`dataset/ppo_dataset.py`](../dataset/ppo_dataset.py) | `AgentRLDataset`、可验证任务过滤、稳定切分和 Collator |
| [`trainer/rollout_engine.py`](../trainer/rollout_engine.py) | Tokenized 单轮生成、原始 Tool Call 文本和行为策略 Log Probability |
| [`dataset/rl/agent_rl.jsonl`](../dataset/rl/agent_rl.jsonl) | MiniMind 格式的 Agentic RL 数据 |
| [`tests/test_agent_rl.py`](../tests/test_agent_rl.py) | 数据、工具、Reward、轨迹 Mask、Objective 和 PPO Actor 加载测试 |

本项目使用原生 PyTorch，不依赖 TRL。

## 4. 数据格式

每行包含 `conversations` 和 `gt`。最后一条消息必须是空 Assistant 占位符：

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "",
      "tools": "[{\"type\":\"function\",\"function\":{\"name\":\"calculate_math\",\"parameters\":{\"type\":\"object\",\"properties\":{\"expression\":{\"type\":\"string\"}},\"required\":[\"expression\"]}}}]"
    },
    {"role": "user", "content": "计算 570*8156"},
    {"role": "assistant", "content": ""}
  ],
  "gt": ["4648920"]
}
```

数据加载时会：

1. 删除最后的空 Assistant 占位符；
2. 解析 system 消息中的工具定义；
3. 保留原始 JSONL 行号用于错误定位；
4. 只选择 tools 非空且 `gt` 非空的可验证任务；
5. 根据 `--seed` 做确定性的训练/验证切分。

### 4.1 当前数据统计

当前 `agent_rl.jsonl` 共 39,988 条：

| 类型 | 数量 | 第一阶段是否使用 |
|---|---:|---|
| 带工具且 GT 非空 | 20,000 | 是 |
| 开放问答且 GT 为空 | 19,988 | 否 |

默认 `--val-ratio 0.02`，因此可验证任务被切分为：

| Split | 数量 |
|---|---:|
| Train | 19,600 |
| Validation | 400 |

开放问答没有客观 Verifier，需要额外 Reward Model。当前阶段不将其混入工具训练，
避免模型通过长度、格式或固定话术获得虚假奖励。

### 4.2 工具类型

训练环境支持与当前数据一致的六种工具：

- `calculate_math`：基础数学表达式；
- `unit_converter`：长度、重量和温度换算；
- `get_current_weather`：确定性的模拟天气；
- `get_current_time`：确定性的模拟时间；
- `get_exchange_rate`：确定性的模拟汇率；
- `translate_text`：数据集覆盖的固定翻译。

数学工具使用 AST 白名单，只允许数字、括号和基础运算符，不执行 Python 名称、
函数、属性或索引。所有工具只接受当前任务提供的白名单名称，错误会作为结构化
observation 返回给模型，使其可以在下一轮修正。

## 5. 多轮轨迹与 Loss Mask

每条轨迹最多运行 `--max-turns` 轮。默认值为 3：

```text
Prompt                       mask=0
Assistant tool_call          mask=1
Tool observation             mask=0
Assistant tool_call          mask=1
Tool observation             mask=0
Assistant final answer       mask=1
```

只有模型实际生成的 Assistant Token 参与 GRPO/CISPO Loss。以下内容均不参与：

- system/user 历史；
- Assistant generation header；
- 训练环境插入的 Tool observation；
- 环境补充的轮次分隔符；
- padding。

每轮生成时都会保存行为策略的 old log-probability。后续将所有轮次拼成一条完整
轨迹，再与 action mask 对齐。这样 Tool observation 可以改变下一轮决策，但不会
被当成模型行为计算梯度。

## 6. Reward 设计

总奖励由以下部分组成，并裁剪到 `[-3, 3]`：

```text
total_reward
  = outcome_reward
  + tool_reward
  + format_reward
  + efficiency_reward
```

### 6.1 Outcome Reward

最终回答覆盖 GT 的比例是主要奖励：

```text
outcome_reward = 2.5 × matched_gt / total_gt
```

必须至少成功执行一次工具才会得到 Outcome Reward。模型直接猜中答案但没有调用
工具，不计为 Agent 成功。

数值 GT 使用绝对/相对容差匹配，字符串 GT 使用规范化匹配。工具 observation 中
的 GT 命中会单独记录，但完整 Outcome Reward 仍要求最终 Assistant 回答包含结果。

### 6.2 工具与格式奖励

- 成功解析并执行工具获得正奖励；
- 不在任务白名单中的工具扣分；
- 非法参数、执行失败和损坏的 JSON 扣分；
- `<tool_call>` 标签不闭合扣分；
- 达到最大轮数仍未给出最终回答扣分。

### 6.3 效率奖励

- 重复相同 Tool Call 扣分；
- 无效调用扣分；
- 最终回答出现重复三元组时扣分。

不同任务需要的工具数量不一定等于 GT 数量，因此不会机械地要求“调用数必须等于
GT 数量”。

## 7. GRPO 与 CISPO

默认使用 GRPO。对于同一个 Prompt 采样 `G` 条完整 Agent 轨迹：

```text
advantage_i = (reward_i - group_mean) / (group_std + eps)
```

随后对所有 Assistant action token 使用相同的轨迹 Advantage，并加入冻结 PPO
Reference 的 KL 约束：

```text
loss = clipped_policy_loss + beta × sampled_kl
```

如果一组轨迹奖励完全相同，组内标准差为 0，该组无法提供相对学习信号。训练器会：

- 将该组 Advantage 置零；
- 屏蔽该组的策略和 MoE Router 更新；
- 通过 `effective_group_rate` 记录有效组比例。

可以通过 `--loss-type cispo` 切换到 MiniMind 风格的 CISPO Objective。建议先建立
稳定的 GRPO 基线，再进行 CISPO 对照实验。

## 8. 加载 PPO Actor

支持两种输入方式。

### 8.1 Transformers Actor 目录

```bash
python trainer/train_agent.py \
    --model-path out/ppo \
    --data-path dataset/rl/agent_rl.jsonl
```

目录至少应包含：

```text
out/ppo/
├── config.json
└── model.safetensors
```

### 8.2 原生 PPO Checkpoint

```bash
python trainer/train_agent.py \
    --ppo-checkpoint checkpoints/ppo/latest.pt \
    --data-path dataset/rl/agent_rl.jsonl
```

Checkpoint 必须包含 PPO 训练器保存的：

- `actor_model`；
- `actor_config`。

Agentic RL 不加载 PPO Critic。训练开始时复制 PPO Actor：

```text
Policy    = PPO Actor，全参数可训练
Reference = PPO Actor 冻结副本
```

## 9. 先运行基线评测

在正式训练前，先检查 PPO Actor 的工具能力：

```bash
python trainer/train_agent.py \
    --model-path out/ppo \
    --eval-only \
    --eval-samples 128 \
    --max-turns 3 \
    --max-new-tokens 256 \
    --tracker none
```

评测使用 greedy decoding，并固定关闭 thinking，保证不同训练阶段的结果可比较。

重点观察：

- `task_success_rate`；
- `gt_match_rate`；
- `valid_tool_call_rate`；
- `tool_execution_success_rate`；
- `unfinished_rate`；
- `parse_errors_per_sample`。

如果合法工具调用率接近 0，应先补充 Tool-Call SFT，而不是直接增加 Agent RL 步数。

## 10. 冒烟训练

先用少量数据检查显存、轨迹和保存链路：

```bash
python trainer/train_agent.py \
    --model-path out/ppo \
    --max-train-samples 20 \
    --eval-samples 8 \
    --max-steps 2 \
    --batch-size 1 \
    --num-generations 2 \
    --max-new-tokens 128 \
    --eval-interval 2 \
    --save-interval 2 \
    --tracker none
```

确认以下结果：

- Loss、Reward、KL 和梯度范数都是有限值；
- `mean_action_tokens` 大于 0；
- `checkpoints/agent/latest.pt` 成功保存；
- `out/agent` 成功导出；
- observation token 没有进入 action loss。

## 11. 正式训练

单卡默认训练：

```bash
python trainer/train_agent.py \
    --model-path out/ppo \
    --data-path dataset/rl/agent_rl.jsonl \
    --loss-type grpo \
    --num-generations 4 \
    --batch-size 2 \
    --accumulation-steps 1 \
    --learning-rate 3e-7 \
    --beta 0.05 \
    --max-prompt-len 1024 \
    --max-total-len 2048 \
    --max-new-tokens 256 \
    --max-turns 3 \
    --epochs 1 \
    --tracker swanlab
```

双卡训练：

```bash
torchrun --nproc_per_node 2 trainer/train_agent.py \
    --model-path out/ppo \
    --batch-size 2 \
    --num-generations 4 \
    --learning-rate 3e-7 \
    --tracker swanlab
```

一次 Rollout Batch 实际生成的轨迹数为：

```text
trajectories = batch_size × num_generations × GPU数量
```

默认单卡是 `2 × 4 = 8` 条轨迹。Agent 训练的主要成本通常来自多轮生成，而不是
一次反向传播，因此增大 `num_generations` 会显著增加训练时间。

## 12. 默认参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--loss-type` | `grpo` | `grpo` 或 `cispo` |
| `--num-generations` | 4 | 每个 Prompt 的轨迹数 |
| `--batch-size` | 2 | 每卡 Prompt 数 |
| `--learning-rate` | `3e-7` | Policy 学习率 |
| `--beta` | `0.05` | Reference KL 系数 |
| `--epsilon` | `0.2` | GRPO Ratio Clip |
| `--epsilon-high` | `5.0` | CISPO 上界 |
| `--max-turns` | 3 | 最大 Assistant 轮数 |
| `--max-prompt-len` | 1024 | 初始 Prompt 上限 |
| `--max-total-len` | 2048 | 完整轨迹上限 |
| `--max-new-tokens` | 256 | 每轮最大生成长度 |
| `--temperature` | `0.8` | 训练 Rollout 温度 |
| `--top-p` | `0.9` | Nucleus Sampling |
| `--top-k` | 50 | Top-K Sampling |
| `--thinking-ratio` | `0.1` | 开启显式 thinking 的轨迹比例 |
| `--val-ratio` | `0.02` | 验证集比例 |

## 13. 训练指标

| 指标 | 含义 |
|---|---|
| `loss` | Policy、KL 和有效 MoE Router 辅助损失之和 |
| `policy_loss` | GRPO/CISPO 策略损失 |
| `kl` | 当前 Policy 相对冻结 PPO Reference 的 sampled KL |
| `reward` | 当前轨迹平均总奖励 |
| `task_success_rate` | 成功调用工具、覆盖全部 GT 且正常结束的比例 |
| `effective_group_rate` | Reward 有组内差异、能够提供梯度的 Prompt 比例 |
| `clip_fraction` | Importance Ratio 被裁剪的 action token 比例 |
| `mean_ratio` | 当前策略与行为策略的平均概率比 |
| `mean_action_tokens` | 每条轨迹中参与 Loss 的 Assistant Token 数 |
| `grad_norm` | 裁剪前梯度范数 |

### 13.1 如何判断训练是否健康

- `task_success_rate` 和 `gt_match_rate` 应逐步提高；
- `valid_tool_call_rate` 不应持续下降；
- `effective_group_rate` 不能长期接近 0；
- KL 应保持有限，不能快速单调爆炸；
- `unfinished_rate` 和解析错误率应下降；
- Reward 上升时要同步检查成功率，防止只学会格式奖励。

## 14. 断点续训与输出

自动恢复：

```bash
python trainer/train_agent.py \
    --model-path out/ppo \
    --resume
```

指定 Checkpoint：

```bash
python trainer/train_agent.py \
    --model-path out/ppo \
    --resume checkpoints/agent/latest.pt
```

训练状态保存到：

```text
checkpoints/agent/latest.pt
```

其中包含 Policy、Optimizer、Scheduler、AMP Scaler、训练位置、随机数状态和实验
记录信息。冻结 Reference 不重复保存，恢复时仍从相同 PPO Actor 构建。

最终 Transformers 模型导出到：

```text
out/agent/
├── config.json
├── model.safetensors
├── tokenizer.json
└── chat_template.jinja
```

## 15. 常见问题

### 15.1 `effective_group_rate` 长期接近 0

说明同一 Prompt 的所有轨迹几乎总是同奖：

- 全部失败：先检查 PPO Actor 是否具备 Tool Call 基础；
- 全部成功：这些任务已经没有继续训练的必要；
- 生成过于确定：适当提高 temperature 或增加 `num_generations`；
- Reward 区分度不足：检查各 Reward 分项和 GT verifier。

### 15.2 显存不足

按以下顺序降低开销：

1. 减小 `--batch-size`；
2. 减小 `--num-generations`；
3. 减小 `--max-new-tokens`；
4. 减小 `--max-total-len`；
5. 开启 `--gradient-checkpointing`；
6. 增加 `--accumulation-steps` 补偿有效 Batch。

### 15.3 合法调用率低

检查模型是否输出以下格式：

```text
<tool_call>
{"name": "calculate_math", "arguments": {"expression": "2+2"}}
</tool_call>
```

常见错误包括：工具名不在当前白名单、`arguments` 不是 JSON Object、标签未闭合，
或者模型在 `<tool_call>` 外输出了不完整 JSON。

### 15.4 Reward 上升但任务成功率不升

这通常表示模型只改善了格式或工具执行，却没有把 observation 写入最终回答。重点
对比 `reward`、`outcome_reward`、`gt_match_rate` 和 `task_success_rate`，不要只根据
总 Reward 选择模型。

### 15.5 项目 `.venv` 无法导入 PyTorch

先按 [`requirements.txt`](../requirements.txt) 安装依赖。CUDA 服务器应优先保留
镜像自带的 CUDA 兼容 PyTorch，或者安装与服务器 CUDA 版本对应的官方 Wheel。

## 16. 测试

运行 Agent 定向测试：

```bash
python -m unittest tests.test_agent_rl tests.test_ppo -v
```

运行完整回归测试：

```bash
python -m unittest discover -s tests -v
```

Agent 测试覆盖安全工具执行、多轮 observation、Assistant-only action mask、GT
Reward、零方差组、GRPO 梯度、PPO checkpoint 加载和现有 PPO Rollout 兼容性。

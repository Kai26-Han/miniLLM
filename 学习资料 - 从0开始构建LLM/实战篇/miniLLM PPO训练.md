# miniLLM PPO 训练

> 默认路线：`SFT Actor + 冻结 SFT Reference + SFT 初始化 Critic + 外部 Reward Model`。也可通过 `--model-path` 改用有效的 DPO 导出模型。

## 1. 训练链路

```text
out/sft
  ├─ Actor（可训练）
  ├─ Reference（冻结）
  └─ Critic Backbone（可训练）+ Value Head

dataset/rl/rlaif.jsonl
  → Actor 在线生成
  → Reward Model 打分
  → Reference KL + Critic Value + GAE
  → PPO Clip 更新 Actor/Critic
  → out/ppo
```

Actor、Reference 和 Critic Backbone 必须由同一个基础模型初始化。当前 DPO
没有优于 SFT，因此默认直接使用 `out/sft`；Reference 用于限制 PPO 相对该起点
的偏移。

## 2. 代码结构

| 文件 | 作用 |
|---|---|
| `dataset/ppo_dataset.py` | 校验 RLAIF 对话、删除空 Assistant 占位、渲染 Prompt |
| `trainer/rollout_engine.py` | PyTorch 生成、Response Mask、Old Log Probability |
| `trainer/train_ppo.py` | Critic、Reward、GAE、PPO Loss、DDP、AMP、训练循环 |
| `trainer/trainer_utils.py` | Actor/Critic 双模型断点保存与恢复 |
| `tests/test_ppo.py` | 数据、Mask、GAE、Clip、Rollout 和 Checkpoint 测试 |

PPO 采用原生 PyTorch，不依赖 TRL。

## 3. 数据格式

`dataset/rl/rlaif.jsonl` 每行应为：

```json
{
  "conversations": [
    {"role": "user", "content": "请解释蓝天的原因"},
    {"role": "assistant", "content": ""}
  ]
}
```

最后一个 Assistant 是空占位，不是训练答案。Dataset 会删除它并通过 ChatML
模板加入 Assistant generation prompt。

长对话优先删除最早的完整轮次；如果最新问题本身仍超长，才使用 Token
左截断。日志中的 `prompt_truncated_ratio` 用来审计这一情况。

## 4. Reward Model

SFT/DPO 模型本身不包含 Reward Model。正式 PPO 要求一个本地 Hugging Face Reward Model，
并实现 MiniMind 使用的接口：

```python
model.get_score(tokenizer, messages)
```

Policy 与 Reward Model 的数值精度是独立的。`--dtype` 只控制
Actor/Reference/Critic；`--reward-dtype` 只控制外部 Reward Model。
InternLM2 Reward 的官方用法和 MiniMind 都以 FP16 加载，因此后者默认为
`float16`，不会跟随 Policy 的 BF16 设置。启动时还会先检查 Reward
权重和一次固定前向，在采样与 PPO 更新前拦截 NaN/Inf。
旧 InternLM2 代码的 `rotary_emb.inv_freq` 是不写入权重文件的派生
buffer；加载后会按官方公式用 FP32 重建全部 RoPE 频率，防止新版
Transformers 的快速初始化遗留随机或非有限 buffer。

InternLM2 Reward 的远程模型代码使用旧 Cache/RoPE API，而项目的 Qwen3
蒸馏功能需要 Transformers 4.51+。`requirements.txt` 因此将共同兼容区间
固定为 `transformers>=4.51,<4.56`，避免 4.56 开始的显式 Cache 重构破坏
这份旧远程代码。

默认禁止远程下载和远程代码。只有已审核该模型仓库时，才显式加入：

```bash
--allow-remote-reward-model \
--trust-reward-remote-code
```

`--reward-backend rule` 只用于确认训练管线能跑通，不能用于正式对齐。

总奖励为：

```text
Reward Model 分数 - 重复惩罚 - 未生成 EOS 惩罚
```

每个分量会分开记录，避免只看总 Reward 无法发现 Reward Hacking。

## 5. 先跑冒烟测试

确保 `out/sft` 已存在，然后用少量样本和规则奖励检查管线：

```bash
python trainer/train_ppo.py \
  --model-path out/sft \
  --data-path dataset/rl/rlaif.jsonl \
  --reward-backend rule \
  --max-train-samples 8 \
  --eval-samples 2 \
  --eval-batches 0 \
  --max-rollout-steps 2 \
  --batch-size 2 \
  --mini-batch-size 1 \
  --ppo-update-iters 1 \
  --max-prompt-len 128 \
  --max-new-tokens 16 \
  --num-workers 0 \
  --device cuda
```

冒烟测试不用于评价模型质量。

## 6. 单卡正式训练

```bash
python trainer/train_ppo.py \
  --model-path out/sft \
  --data-path dataset/rl/rlaif.jsonl \
  --reward-backend external \
  --reward-model-path /path/to/internlm2-1_8b-reward \
  --trust-reward-remote-code \
  --output-dir out/ppo \
  --save-dir checkpoints/ppo \
  --epochs 1 \
  --batch-size 2 \
  --mini-batch-size 1 \
  --ppo-update-iters 2 \
  --actor-learning-rate 1e-7 \
  --critic-learning-rate 5e-7 \
  --max-prompt-len 512 \
  --max-new-tokens 128 \
  --dtype bfloat16 \
  --reward-dtype float16 \
  --device cuda
```

## 7. 多卡训练

```bash
torchrun --nproc_per_node 2 trainer/train_ppo.py \
  --model-path out/sft \
  --reward-model-path /path/to/internlm2-1_8b-reward \
  --trust-reward-remote-code \
  --batch-size 2 \
  --mini-batch-size 1 \
  --dtype bfloat16 \
  --reward-dtype float16
```

每张 GPU 都会持有 Actor、Reference、Critic 和 Reward Model。DDP 不会自动对这四份
权重做显存切分，因此先从 `batch-size=1~2` 开始。

## 8. 核心参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `actor-learning-rate` | `1e-7` | PPO Actor 使用较小学习率 |
| `critic-learning-rate` | `5e-7` | Critic 需要更快学习 Value |
| `dtype` | `bfloat16` | Actor/Reference/Critic 的混合精度 |
| `reward-dtype` | `float16` | 外部 Reward Model 推理精度，与 Policy 解耦 |
| `clip-epsilon` | `0.2` | PPO Policy Ratio 裁剪 |
| `value-clip` | `0.2` | Critic Value 变化裁剪 |
| `value-loss-coef` | `0.5` | Value Loss 权重 |
| `kl-coef` | `0.02` | 相对冻结基础模型 Reference 的 KL 惩罚 |
| `gamma` | `1.0` | 终局奖励任务的折扣因子 |
| `gae-lambda` | `0.95` | GAE 偏差—方差折中 |
| `ppo-update-iters` | `2` | 同一批 Rollout 的重复更新轮数 |
| `early-stop-kl` | `0.10` | New/Old KL 过大时停止当前批更新 |

## 9. 日志与诊断

重点监控：

- `train/reward` 与 `train/model_reward`；
- `train/reference_kl`；
- `train/approximate_kl`；
- `train/clip_fraction`；
- `train/value_loss` 和 `train/value_clip_fraction`；
- `train/raw_advantage_mean`；
- `train/eos_ratio` 与 `train/response_length`；
- Actor/Critic Router 辅助损失；
- Actor/Critic Gradient Norm。

常见异常：

| 现象 | 可能原因 | 处理 |
|---|---|---|
| Reward 上升但人工质量下降 | Reward Hacking | 降低学习率/KL，审核最高分样本 |
| `reference_kl` 快速增长 | KL 太弱或 Actor LR 太大 | 提高 `kl-coef` 或降低 Actor LR |
| Value Loss 爆炸 | Reward 尺度或 Critic LR 不合适 | 缩小 Reward Clip/Critic LR |
| Reward 第一次前向就是 NaN | Reward 精度与官方配置不一致或权重损坏 | InternLM2 使用 `--reward-dtype float16`；核对启动状态检查 |
| Reward 的 `rotary_emb.inv_freq` 非有限 | 旧远程代码的非持久 RoPE buffer 未正确恢复 | 确认启动时出现 `Reward RoPE: ... FP32 buffers rebuilt` |
| EOS 比例很低 | 回答长度不足或策略退化 | 检查样本、长度和 EOS 惩罚 |
| 第一轮 New/Old KL 不接近 0 | LogProb/Mask 错位 | 中止训练并运行单测 |

## 10. 断点恢复

```bash
python trainer/train_ppo.py \
  --model-path out/sft \
  --reward-model-path /path/to/internlm2-1_8b-reward \
  --trust-reward-remote-code \
  --resume
```

`--resume` 默认读取 `checkpoints/ppo/latest.pt`。恢复时不允许更改 Policy 起点、
Reward Model、序列长度、Clip 和 KL 等影响训练语义的参数。

## 11. 训练后验证

PPO 结束后 Actor 会导出到 `out/ppo`。先检查生成：

```bash
python eval/eval_sft.py \
  --mode generate \
  --model-path out/ppo
```

再检查 DPO 偏好保持：

```bash
python eval/eval_dpo.py \
  --policy-path out/ppo \
  --reference-path out/sft
```

不能按训练 Reward 最高点直接选模型。还应比较基础模型/PPO 的固定 Prompt 生成、
通用 SFT Loss、偏好准确率、安全性和回答长度。

## 12. 当前性能边界

当前 `MiniLLMForCausalLM` 没有 KV Cache，Rollout 会重复计算完整前缀。第一轮建议
`max_new_tokens=64~128` 并先用小数据验证。正式扩大训练前，应将 KV Cache
作为独立性能里程碑实现，并校验缓存前后 Logits 一致性。

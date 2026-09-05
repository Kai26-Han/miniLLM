# miniLLM 预训练

> 项目：miniLLM  
> 阶段：从随机权重开始进行 Language Model 预训练  
> 当前正式方案：4个 Expert、Top-1 MoE、约2亿总参数、约6500万激活参数、全量数据训练1轮  
> 训练监控：SwanLab  
> 默认序列长度：512 Token

---

## 1. 这次预训练要完成什么

Tokenizer 阶段解决的是“怎样把文本转换成 Token ID”。预训练阶段要让一个随机初始化的 Decoder-only Transformer，通过大量 next-token prediction 学会：

- 语言的基本规律；
- 中文和英文的词法、语法与语义关系；
- 常见知识和文本结构；
- 根据左侧上下文预测下一个 Token；
- 为后续 SFT、对话微调和领域微调提供基础模型。

整个训练链路如下：

```text
JSONL 文本
   ↓
miniLLM Tokenizer（8192词表）
   ↓
BOS + Token IDs + EOS + Padding
   ↓
8层 Decoder-only Transformer
   ├── GQA + RoPE Self-Attention
   └── 4 Expert、Top-1 Sparse MoE
   ↓
预测每个位置的下一个 Token
   ↓
Cross Entropy + Router 辅助损失
   ↓
AdamW 更新模型参数
   ↓
Checkpoint + SwanLab 指标 + Transformers 模型
```

预训练过程中不会继续修改 Tokenizer。Tokenizer 的词表及特殊 Token ID 一旦确定，就必须与模型配置保持一致。

---

## 2. 当前项目中与预训练有关的文件

```text
miniLLM/
├── dataset/
│   ├── lm_dataset.py
│   └── pretrain/
│       ├── pretrain_mini.jsonl
│       └── pretrain.jsonl
├── model/
│   ├── model_minillm.py
│   └── tokenizer/
├── trainer/
│   ├── train_pretrain.py
│   └── trainer_utils.py
├── eval/
│   └── eval_llm.py
├── checkpoints/             # 训练后生成，用于断点续训
├── out/                     # 训练后生成，用于评估、推理和后续微调
└── requirements.txt
```

各文件职责如下：

| 文件 | 作用 |
|---|---|
| [`dataset/lm_dataset.py`](../dataset/lm_dataset.py) | 加载 JSONL、划分训练/验证集、Tokenize、截断和 Padding |
| [`model/model_minillm.py`](../model/model_minillm.py) | miniLLM Dense/MoE 模型结构与 Causal LM Loss |
| [`trainer/train_pretrain.py`](../trainer/train_pretrain.py) | 主训练入口、训练循环、评估、日志和模型导出 |
| [`trainer/trainer_utils.py`](../trainer/trainer_utils.py) | DDP、学习率调度、Checkpoint、SwanLab/W&B 接入 |
| [`eval/eval_llm.py`](../eval/eval_llm.py) | 计算验证 Loss/PPL，并进行文本续写测试 |
| [`model/tokenizer/`](../model/tokenizer/) | 预训练使用的完整 Tokenizer 目录 |
| [`requirements.txt`](../requirements.txt) | PyTorch、Transformers、SwanLab 等依赖 |

---

## 3. 数据方案

### 3.1 当前数据

项目中已经有两份 MiniMind 预训练数据：

| 文件 | 本地大小 | 行数 | 推荐用途 |
|---|---:|---:|---|
| `pretrain_mini.jsonl` | 约1.2 GB | 1,270,238 | 冒烟测试、显存测试、短周期实验 |
| `pretrain.jsonl` | 约7.7 GB | 8,468,827 | 正式全量预训练 |

正式方案使用：

```text
dataset/pretrain/pretrain.jsonl
```

不要为了“数据更多”而把 `pretrain_mini.jsonl` 和 `pretrain.jsonl` 同时传入。两者的重叠关系没有明确保证，同时使用可能让重叠样本被重复训练。mini 文件应被看作快速实验数据，而不是正式数据的额外增量。

### 3.2 JSONL 格式

训练脚本要求每行都是一个合法 JSON 对象，并包含字符串字段 `text`：

```json
{"text": "这里是一段用于语言模型训练的文本。"}
```

只读取 `text` 字段。元数据文件、Tokenizer 的 `metadata.json` 以及其他 JSON 配置不能作为预训练语料传入。

### 3.3 每条数据如何变成训练样本

`PretrainDataset` 对每行文本执行：

```text
原始 text
  ↓ Tokenizer，不自动添加特殊 Token
普通 Token IDs
  ↓ 最多保留 max_seq_len - 2 个
[BOS] + Token IDs + [EOS]
  ↓ 右侧补 PAD 到固定长度
input_ids、attention_mask、labels
```

当前特殊 Token ID 必须是：

| 作用 | Token ID |
|---|---:|
| PAD | 0 |
| UNK | 0 |
| BOS | 1 |
| EOS | 2 |

Padding 位置的 label 会改为 `-100`，因此不参与 Cross Entropy。模型学习的是从左到右预测下一个 Token：

```text
输入：  BOS  今天  天气  很好  EOS
标签：  今天  天气  很好  EOS  ——
```

需要注意：当前实现没有 sequence packing。一条短文本仍会被补齐到512长度，不能把另一条文本装进剩余位置。因此“处理的512个 Token 槽位”不等于“512个有效语料 Token”。

### 3.4 训练集与验证集划分

默认 `--val-ratio 0.001`，即按照源数据行号确定性划分：

```text
第0、1000、2000……行 → Validation
其他行                  → Train
```

对全量 `pretrain.jsonl`：

- 总样本：8,468,827；
- 验证候选：约8,469；
- 训练样本：约8,460,358；
- 实际每次验证最多使用 `--eval-samples 2048` 条。

固定划分便于不同实验对比，但如果源文件按主题高度有序，按行号取样仍可能带来分布偏差。更严谨的后续版本可以建立独立、去重的验证集。

---

## 4. 模型架构方案

### 4.1 公共骨架

当前 miniLLM 是 Decoder-only Causal Language Model：

| 参数 | 当前值 |
|---|---:|
| Tokenizer 词表 | 8,192 |
| 隐藏维度 | 768 |
| Decoder 层数 | 8 |
| Query Heads | 8 |
| Key/Value Heads | 4 |
| Head Dimension | 96 |
| FFN/Expert 中间维度 | 2,432 |
| 训练序列长度 | 512 |
| 最大位置配置 | 32,768 |
| RoPE Theta | 1,000,000 |
| Attention | GQA + RoPE + Causal Mask |
| 归一化 | RMSNorm |
| FFN 激活 | SwiGLU |
| Embedding 与 LM Head | 权重共享 |

`max_position_embeddings=32768` 只是模型允许的位置编号上限，本轮实际训练长度仍是512。只在512长度上训练，并不意味着模型已经具备可靠的32K长上下文能力。

### 4.2 正式采用的 MoE 结构

正式方案开启：

```text
--use-moe
--num-experts 4
--num-experts-per-tok 1
```

训练入口与 `MiniLLMConfig` 均已默认开启这套 MoE 配置。正式命令仍显式写出 `--use-moe`，以便实验复现；需要运行 Dense 对照实验时，使用 `--no-use-moe`。

每个 Decoder Layer 中都有4个独立 Expert。Router 为每个有效 Token 计算4个概率，只把该 Token 发送给得分最高的1个 Expert：

```text
Token hidden state
       ↓ Router Softmax
[E0=0.10, E1=0.62, E2=0.18, E3=0.10]
       ↓ Top-1
只执行 Expert 1
```

因此：

- 所有4个 Expert 参数都需要保存并驻留在显存中；
- 每个 Token 每层只执行1个 Expert；
- 总参数量大幅增加；
- 单 Token 激活计算量仍接近同宽度 Dense 模型；
- MoE 节省的是激活计算，不等于节省模型权重和优化器显存。

### 4.3 参数量

根据当前源码的实际维度计算：

| 架构 | 总参数 | 每 Token 激活参数 |
|---|---:|---:|
| Dense | 65,286,912，约65.3M | 约65.3M |
| 4 Expert、Top-1 MoE | 199,791,360，约199.8M | 65,311,488，约65.3M |

这就是此前所说“约198M总参数、约64M激活参数”的当前源码精确版本。不同项目对参数分组和四舍五入方式不同，文档和程序启动时应以 `Parameters`、`Active params` 的实际输出为准。

### 4.4 Router 辅助损失

如果没有约束，Router 可能把绝大多数 Token 都送到同一个 Expert，形成 Expert Collapse。当前模型使用负载均衡辅助损失：

```text
total_loss = lm_loss + router_aux_loss
```

默认系数：

```text
--router-aux-loss-coef 5e-4
```

这个值的目标是鼓励 Expert 均衡，但不能大到压过语言模型本身的学习目标。

---

## 5. 训练超参数方案

### 5.1 当前正式配置

| 参数 | 值 | 含义 |
|---|---:|---|
| epochs | 1 | 完整遍历正式数据1轮 |
| max_seq_len | 512 | 每条样本固定512 Token 槽位 |
| batch_size | 8 | 每张 GPU、每个 micro-batch 的样本数 |
| accumulation_steps | 8 | 累积8个 micro-batch 后更新一次 |
| learning_rate | 5e-4 | 峰值学习率 |
| min_learning_rate | 5e-5 | Cosine 结束时的最低学习率 |
| warmup_ratio | 0.03 | 前3%优化器步进行线性 Warmup |
| weight_decay | 0.1 | AdamW 权重衰减 |
| grad_clip | 1.0 | 梯度范数裁剪 |
| dtype | bfloat16 | CUDA 混合精度首选 |
| val_ratio | 0.001 | 0.1%数据作为验证候选 |
| eval_samples | 2048 | 每次验证最多评估的样本数 |
| log_interval | 10 | 每10个 optimizer step 记录一次 |
| eval_interval | 500 | 每500步验证一次 |
| save_interval | 1000 | 每1000步覆盖保存 `latest.pt` |
| seed | 42 | 随机种子 |

### 5.2 有效 Batch Size

有效全局 Batch Size 为：

$$
B_{global}=B_{micro}\times N_{accumulation}\times N_{GPU}
$$

单卡默认配置：

$$
8\times8\times1=64\text{ samples/update}
$$

最大 Token 槽位：

$$
64\times512=32768\text{ token slots/update}
$$

如果显存不足，可在保持乘积近似不变的情况下调整：

| batch_size | accumulation_steps | 单卡有效 Batch |
|---:|---:|---:|
| 8 | 8 | 64 |
| 4 | 16 | 64 |
| 2 | 32 | 64 |
| 1 | 64 | 64 |

更小的 micro-batch 通常更省激活显存，但梯度累积次数增加，会提高循环开销、降低吞吐量。

### 5.3 一轮大约多少步

对全量数据，单卡默认参数大约为：

```text
Train samples       ≈ 8,460,358
Micro-batch size    = 8
Accumulation steps  = 8
Optimizer steps     ≈ 132,193
```

`global_step` 统计的是 `optimizer.step()` 次数，不是读取了多少个 DataLoader batch。

`--max-steps` 大于0时会覆盖 `--epochs` 计算出的总步数，适合短实验；正式一轮训练应让 `--max-steps` 保持默认0。

### 5.4 学习率曲线

当前调度方式：

```text
0
↓ 前3% step 线性 Warmup
5e-4
↓ 剩余 step Cosine 衰减
5e-5
```

Warmup 用于降低随机初始化阶段梯度不稳定的风险；Cosine 衰减让训练后期用更小步长收敛。

---

## 6. 训练前环境准备

以下命令都假设终端位于项目根目录：

```bash
cd /root/miniLLM
```

安装项目依赖：

```bash
pip install -r requirements.txt
```

镜像如果已经带有与 CUDA 匹配的 PyTorch，不建议随意覆盖安装其他 CUDA 版本的 PyTorch。先检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

检查训练脚本参数：

```bash
python trainer/train_pretrain.py --help
```

SwanLab 在线监控需要完成账号认证。登录后，训练命令中的参数必须是：

```text
--tracker swanlab
```

---

## 7. 不要直接开始全量训练：先做冒烟测试

建议先用少量数据运行20个 optimizer step，检查模型、显存、Checkpoint、验证和 SwanLab 是否全部正常：

```bash
python trainer/train_pretrain.py \
  --data-path dataset/pretrain/pretrain_mini.jsonl \
  --tokenizer-path model/tokenizer \
  --use-moe \
  --num-experts 4 \
  --num-experts-per-tok 1 \
  --max-seq-len 512 \
  --batch-size 8 \
  --accumulation-steps 8 \
  --max-train-samples 4096 \
  --max-steps 20 \
  --log-interval 1 \
  --eval-interval 10 \
  --save-interval 10 \
  --gradient-checkpointing \
  --tracker swanlab \
  --tracker-project miniLLM-Pretrain \
  --tracker-run-name moe-smoke-test \
  --tracker-tags moe smoke-test \
  --save-dir checkpoints/pretrain_moe_smoke \
  --output-dir out/pretrain_moe_smoke
```

冒烟测试至少确认：

1. 启动输出显示 `MoE (4 Experts, Top-1)`；
2. 总参数约199.8M、激活参数约65.3M；
3. Device 为 CUDA；
4. Precision 为 bfloat16 或自动回退后的 float16；
5. loss 可以正常下降且不是 NaN；
6. 四个 Expert 都能收到 Token；
7. SwanLab 网页出现训练曲线；
8. `checkpoints/pretrain_moe_smoke/latest.pt` 能生成；
9. `out/pretrain_moe_smoke/` 能导出模型。

冒烟测试目录要与正式训练目录分开，防止正式训练误加载测试断点。

---

## 8. 单卡正式训练命令

### 8.1 推荐的完整命令

完整写出关键参数，方便复现与检查：

```bash
python trainer/train_pretrain.py \
  --data-path dataset/pretrain/pretrain.jsonl \
  --tokenizer-path model/tokenizer \
  --use-moe \
  --num-experts 4 \
  --num-experts-per-tok 1 \
  --router-aux-loss-coef 5e-4 \
  --max-seq-len 512 \
  --batch-size 8 \
  --accumulation-steps 8 \
  --epochs 1 \
  --learning-rate 5e-4 \
  --min-learning-rate 5e-5 \
  --warmup-ratio 0.03 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --dtype bfloat16 \
  --gradient-checkpointing \
  --log-interval 10 \
  --eval-interval 500 \
  --save-interval 1000 \
  --tracker swanlab \
  --tracker-project miniLLM-Pretrain \
  --tracker-run-name pretrain-moe4-top1-full-1epoch \
  --tracker-tags moe full-data 1epoch \
  --save-dir checkpoints/pretrain_moe \
  --output-dir out/pretrain_moe
```

### 8.2 精简命令

因为绝大多数值已经是脚本默认值，所以也可以只覆盖决定实验性质的参数：

```bash
python trainer/train_pretrain.py \
  --data-path dataset/pretrain/pretrain.jsonl \
  --use-moe \
  --gradient-checkpointing \
  --tracker swanlab \
  --tracker-run-name pretrain-moe4-top1-full-1epoch \
  --save-dir checkpoints/pretrain_moe \
  --output-dir out/pretrain_moe
```

两条命令在当前代码默认值下对应同一核心方案。正式实验更推荐保留完整命令，避免未来修改默认参数后无法复现实验。

### 8.3 为什么建议开启 Gradient Checkpointing

```text
--gradient-checkpointing
```

它不会减少模型参数、梯度和 AdamW 状态占用，而是减少前向激活的保存；反向传播时会重新计算部分前向结果。因此：

- 优点：显著降低激活显存；
- 代价：训练速度下降；
- 适合：显存有限、batch较大或序列较长的训练。

当前模型参数本身以 float32 保存，`--dtype bfloat16` 控制的是 CUDA autocast 计算精度，并不会把所有长期保存的参数和 AdamW 状态都变成 bfloat16。约2亿参数的权重、梯度和两个 Adam 状态仅理论基础占用就约3.2 GB，还要叠加激活、临时张量、CUDA缓存和框架开销。

### 8.4 暂不建议开启 `--compile`

当前 MoE 使用动态 Top-1 路由和逐 Expert 分发。`torch.compile` 可能发生 graph break，第一次编译也需要额外时间和显存。应先完成不带 `--compile` 的稳定基线，再单独测试它是否真的提高吞吐。

---

## 9. 多卡训练

两张 GPU：

```bash
torchrun --nproc_per_node 2 trainer/train_pretrain.py \
  --data-path dataset/pretrain/pretrain.jsonl \
  --use-moe \
  --gradient-checkpointing \
  --tracker swanlab \
  --tracker-run-name pretrain-moe4-top1-full-1epoch-2gpu \
  --save-dir checkpoints/pretrain_moe \
  --output-dir out/pretrain_moe
```

当前多卡方式是 DDP：

- 每张 GPU 持有完整模型和优化器状态；
- 不同 GPU 读取不同数据分片；
- 反向传播时同步梯度；
- 只有 rank 0 上传 SwanLab、保存 Checkpoint 和导出模型。

`num_workers > 0` 时，DataLoader 显式使用 `spawn` 创建 Worker，避免 Linux 默认 `fork` 继承已初始化的 CUDA/NCCL 状态而导致多卡死锁。`--num-workers 0` 时不创建子进程，因此不传入 multiprocessing context。

DDP 可以提高数据吞吐，但不会像 FSDP/ZeRO 那样把模型参数和优化器状态切分到多张卡。因此它不能解决“单张卡连完整模型状态都放不下”的问题。

多卡时有效全局 Batch 会随 GPU 数量增加。如果从1卡扩到2卡且其他参数不变：

```text
8 × 8 × 2 = 128 samples/update
```

如果希望保持全局 Batch 为64，可把每卡 batch 或累积步数减半。

---

## 10. SwanLab 监控方案

正式训练使用：

```text
--tracker swanlab
--tracker-project miniLLM-Pretrain
--tracker-run-name pretrain-moe4-top1-full-1epoch
--tracker-tags moe full-data 1epoch
```

训练脚本会记录以下指标。

### 10.1 基础训练指标

| 指标 | 含义 | 主要观察方式 |
|---|---|---|
| `train/loss` | LM Loss 与 Router 辅助损失之和 | 应整体下降，短时波动正常 |
| `train/lm_loss` | 纯 next-token Cross Entropy | 判断语言建模学习效果的核心指标 |
| `train/learning_rate` | 当前学习率 | 应先 Warmup，再 Cosine 衰减 |
| `train/gradient_norm` | 裁剪前的全局梯度范数 | 长期异常尖峰、NaN 需要排查 |
| `train/tokens_per_second` | 估算的 Token 槽位吞吐 | 用于比较配置速度，不是精确有效 Token 数 |
| `train/epoch` | 当前训练进度 | 从0逐步接近1 |
| `system/gpu_memory_allocated_gb` | PyTorch 当前分配显存 | 观察稳定占用与泄漏 |
| `system/gpu_memory_reserved_gb` | PyTorch 预留显存 | 一般大于 allocated |

### 10.2 验证指标

| 指标 | 含义 |
|---|---|
| `validation/loss` | 验证集上的纯语言模型 Loss |
| `validation/perplexity` | `exp(validation_loss)`，越低越好 |

判断是否继续训练不能只看训练 Loss：

```text
train loss 下降 + validation loss 下降 → 正常学习
train loss 下降 + validation loss 上升 → 可能过拟合或训练/验证分布有差异
两者长期不下降                 → 检查学习率、数据和模型实现
loss/grad norm 出现 NaN        → 检查数值稳定性与异常数据
```

### 10.3 MoE Router 指标

| 指标 | 含义 |
|---|---|
| `train/router_aux_loss` | Router 负载均衡辅助损失 |
| `moe/expert_i_usage` | 第 i 个 Expert 实际接收 Token 的比例 |
| `moe/expert_i_router_probability` | Router 分配给第 i 个 Expert 的平均概率 |
| `moe/router_entropy_normalized` | Router 概率分布的归一化熵 |
| `moe/max_load_ratio` | 最繁忙 Expert 的 Token 占比 |
| `moe/min_load_ratio` | 最空闲 Expert 的 Token 占比 |

4个 Expert 完全均匀时，每个 usage 接近：

$$
1/4=0.25
$$

不要求每一步都严格等于0.25，但如果长时间出现某个 Expert 接近0、另一个接近1，就说明路由可能坍缩。需要结合 Router auxiliary loss、训练 Loss 和验证 Loss 一起判断，不能只为了曲线均匀而盲目增大辅助损失系数。

### 10.4 Offline 模式

服务器暂时无法访问外网时可以使用：

```text
--tracker swanlab --tracker-mode offline
```

日志会写入 `logs/pretrain`。离线数据如何同步以当前安装的 SwanLab SDK 命令为准。

---

## 11. Checkpoint 与断点续训

### 11.1 保存了什么

`checkpoints/pretrain_moe/latest.pt` 包含：

- 模型权重；
- AdamW 优化器状态；
- 学习率调度器状态；
- GradScaler 状态；
- 当前 epoch、batch 和 global step；
- 模型配置与训练参数；
- Python、NumPy、PyTorch、CUDA 随机数状态；
- SwanLab run ID 等监控状态。

它是“继续训练文件”，不是推荐发布或推理使用的最终模型。

当前实现反复覆盖一个 `latest.pt`，不会自动保留每1000步的历史版本。它节省磁盘，但无法回退到更早的最佳步骤。如需保存多个里程碑版本，需要后续扩展命名策略。

### 11.2 自动恢复

使用相同的结构参数、数据和训练计划，在原命令末尾添加：

```text
--resume
```

完整示例：

```bash
python trainer/train_pretrain.py \
  --data-path dataset/pretrain/pretrain.jsonl \
  --use-moe \
  --gradient-checkpointing \
  --tracker swanlab \
  --tracker-run-name pretrain-moe4-top1-full-1epoch \
  --save-dir checkpoints/pretrain_moe \
  --output-dir out/pretrain_moe \
  --resume
```

如果存在，它会读取：

```text
checkpoints/pretrain_moe/latest.pt
```

并尝试恢复同一个 SwanLab run，使曲线继续写在原实验中。如果文件尚未
生成（例如第一次启动训练），则输出提示并自动从头开始。

### 11.3 指定断点文件

```bash
python trainer/train_pretrain.py \
  ... \
  --resume checkpoints/pretrain_moe/latest.pt
```

断点恢复时必须保持模型结构一致，例如 `--use-moe`、Expert 数、Top-K、隐藏维度和层数不能改变；总训练步数也必须仍大于断点中的 `global_step`。

---

## 12. Checkpoint 与最终导出模型的区别

训练成功结束后会有两类产物：

```text
checkpoints/pretrain_moe/latest.pt
out/pretrain_moe/
```

| 产物 | 主要内容 | 用途 |
|---|---|---|
| `latest.pt` | 模型、优化器、调度器、训练位置、随机状态 | 断点续训 |
| `out/pretrain_moe/` | `config.json`、`model.safetensors`、Tokenizer 文件 | 评估、推理、后续 SFT、发布 |

后续预训练评估和 SFT 应优先读取 `out/pretrain_moe/`。只有训练被中断、尚未导出最终目录时，才直接用 `latest.pt` 进行临时评估。

---

## 13. 完成后的评估

### 13.1 验证 Loss、PPL 与文本续写

```bash
python eval/eval_llm.py \
  --model-path out/pretrain_moe \
  --data-path dataset/pretrain/pretrain.jsonl \
  --mode both \
  --max-seq-len 512 \
  --eval-samples 2048 \
  --batch-size 8
```

### 13.2 只评估语言模型 Loss

```bash
python eval/eval_llm.py \
  --model-path out/pretrain_moe \
  --data-path dataset/pretrain/pretrain.jsonl \
  --mode loss
```

### 13.3 使用指定 Prompt 生成

```bash
python eval/eval_llm.py \
  --model-path out/pretrain_moe \
  --mode generate \
  --prompt "人工智能的发展" \
  --prompt "def quick_sort("
```

预训练模型只完成了 next-token learning，还没有经过专门的 Chat SFT 和偏好对齐。它可能更擅长“续写文本”，不一定能稳定遵循对话指令。不能用成熟聊天模型的标准直接判断预训练是否失败。

建议至少从四个维度评估：

1. **Loss/PPL**：固定验证集上的语言建模指标；
2. **续写质量**：中英文、知识、代码多种 Prompt；
3. **记忆与泛化**：避免只复述训练样本；
4. **MoE 路由**：Expert 是否长期均衡且没有坍缩。

---

## 14. 常见问题与处理顺序

### 14.1 CUDA Out of Memory

按以下顺序处理：

1. 确认已经使用 `--gradient-checkpointing`；
2. 把 `--batch-size 8` 降到4或2；
3. 相应增加 `--accumulation-steps`，保持有效 Batch；
4. 确认没有其他进程占用 GPU；
5. 再考虑缩短 `--max-seq-len`；
6. 最后才考虑缩小模型结构。

不要期待 Top-1 MoE 自动把显存降低到65M Dense 模型水平。未激活 Expert 不做主要 MLP 计算，但其权重与优化器状态仍然存在。

### 14.2 训练 Loss 变成 NaN

检查：

- 学习率是否过大；
- `gradient_norm` 是否先出现异常尖峰；
- GPU 是否真正支持 bfloat16；
- 是否有极端或损坏文本；
- 是否错误恢复了不匹配的断点。

可以先减小学习率到 `3e-4` 做对照实验。不要只看一次尖峰，应结合前后数十步走势。

### 14.3 SwanLab 没有数据

检查启动输出是否包含：

```text
Tracker : swanlab (run_id=...)
```

然后检查：

- 参数是否确实为 `--tracker swanlab`；
- 是否完成 SwanLab 认证；
- 服务器是否可以访问外网；
- 是否误用了 `--tracker-mode offline`；
- `swanlab` 是否安装在当前 Python 环境中；
- 多卡时是否正在查看 rank 0 的日志。

### 14.4 训练中断后无法恢复

检查：

- `checkpoints/pretrain_moe/latest.pt` 是否存在；
- `--save-dir` 是否与上次完全相同；
- 是否仍然添加了 `--use-moe`；
- Expert 数、Top-K、隐藏维度和层数是否一致；
- 当前计划的 total steps 是否大于断点 global step。

### 14.5 生成速度较慢

当前紧凑版本还没有实现 KV Cache，生成新 Token 时会重复计算历史上下文。这不影响预训练 Loss 的正确性，但会降低自回归推理速度。KV Cache 属于后续推理优化任务。

---

## 15. 当前方案的边界与后续改进

当前实现适合从零学习完整预训练链路，但它不是大规模生产训练框架，主要边界包括：

1. **没有 sequence packing**：短样本 Padding 浪费算力；
2. **数据按行截断**：超过510个正文 Token 的尾部会被丢弃，不会滑窗续接；
3. **验证集来自同一文件**：适合训练监控，不等于独立基准评测；
4. **MoE 没有 capacity factor**：不会丢 Token，但路由实现不以超大规模吞吐为目标；
5. **DDP 不切分状态**：每张卡保存完整约2亿参数模型；
6. **只保留 latest checkpoint**：不能自动挑选历史最佳模型；
7. **没有 KV Cache**：生成评估速度偏慢；
8. **数据偏 Text-to-Text/指令形式**：模型能力会明显受到当前数据分布影响；
9. **只训练1轮**：是当前算力和完整数据覆盖之间的折中，不代表理论最优轮数。

推荐的改进顺序：

```text
先完成稳定的1轮基线
  ↓
固定验证集比较 Dense 与 MoE
  ↓
加入独立验证集和数据质量统计
  ↓
实现 sequence packing / 文档切块
  ↓
保存多个里程碑并选择最佳模型
  ↓
实现 KV Cache
  ↓
需要更大规模时再引入 FSDP / DeepSpeed
```

---

## 16. 推荐的完整执行流程

### 阶段一：环境检查

```bash
cd /root/miniLLM
pip install -r requirements.txt
python trainer/train_pretrain.py --help
```

### 阶段二：MoE 冒烟测试

使用 `pretrain_mini.jsonl`、`--max-train-samples 4096`、`--max-steps 20`，确认训练、验证、保存和 SwanLab 都能工作。

### 阶段三：显存与吞吐测试

逐步尝试：

```text
batch 2 × accumulation 32
batch 4 × accumulation 16
batch 8 × accumulation 8
```

选择不 OOM 且 tokens/s 最好的组合。

### 阶段四：正式全量训练

使用 `pretrain.jsonl`、4 Expert、Top-1、512长度、1轮，并将正式 Checkpoint 和 Output 写入独立目录。

### 阶段五：训练中监控

重点观察：

```text
train/lm_loss
validation/loss
train/gradient_norm
system/gpu_memory_allocated_gb
moe/expert_0_usage ... moe/expert_3_usage
moe/max_load_ratio / min_load_ratio
```

### 阶段六：训练完成后评估

用 `eval/eval_llm.py` 计算验证 Loss、PPL，并进行中文、英文、代码等多类型续写测试。

### 阶段七：进入 SFT

确认预训练模型能进行基本连贯续写、验证 Loss 合理、MoE 路由没有坍缩后，再以 `out/pretrain_moe/` 作为下一阶段初始化权重。

---

## 17. 最终方案卡片

```text
项目名称        miniLLM
训练目标        从随机权重训练 Decoder-only Base Model
Tokenizer       ByteLevel BPE，vocab=8192
训练数据        pretrain.jsonl，全量约846.9万条
训练轮数        1 epoch
模型层数        8
隐藏维度        768
Attention       GQA，8 Q Heads / 4 KV Heads
位置编码        RoPE，theta=1,000,000
FFN             Sparse MoE + SwiGLU
Experts         4
Top-K           1
总参数          199,791,360（约199.8M）
激活参数        65,311,488（约65.3M）
序列长度        512
Micro Batch     8 / GPU
梯度累积        8
单卡有效 Batch  64 samples/update
优化器          AdamW，betas=(0.9, 0.95)
学习率          5e-4 → 5e-5
Warmup          3%
精度            bfloat16 autocast
显存优化        Gradient Checkpointing
训练监控        SwanLab
断点目录        checkpoints/pretrain_moe/latest.pt
最终模型        out/pretrain_moe/
```

这套方案的核心目标，是先在可理解、可监控、可恢复的工程结构中跑通一次完整的 MoE 预训练，而不是一次性追求工业级规模。完成这一轮后，Loss、PPL、样本质量、Expert 路由和训练吞吐将成为下一轮调整数据、模型和训练预算的实际依据。

---

# 从零实现GPT：LLM与 Transformer

## ChatGPT 的核心逻辑

GPT 的核心可以概括为：把文本转换成 token，用因果自注意力让每个 token 汇聚此前上下文的信息，再训练模型预测下一个 token；推理时不断把新预测的 token 接回输入，便能自回归地生成文本。

## 关键点

1. **语言模型学习的是下一个 token 的条件概率。** 即根据已有序列估计 $P(x_t\mid x_1,\ldots,x_{t-1})$。
2. **token 是模型处理文本的基本单位。** 字符级分词最直观，子词级分词效率更高。
3. **输入和标签只相差一个位置。** 输入是一段 token，标签是这段 token 向左移动一位后的结果。
4. **一个长度为 $T$ 的序列块包含 $T$ 个训练样本。** 每个位置都在用其左侧上下文预测下一个 token。
5. **Bigram 模型是最小可运行基线。** 它只根据当前 token 预测下一个 token，不理解更长的上下文。
6. **自注意力是 token 之间的通信机制。** Query 决定“我要找什么”，Key 表示“我有什么”，Value 表示“我可以传递什么”。
7. **掩码机制禁止读取未来。** 预测当前位置时只能使用当前位置及其左侧信息。
8. **多头注意力负责通信，前馈网络负责逐 token 计算。** 二者交替构成 Transformer Block。
9. **残差连接、LayerNorm 和 Dropout 让深层网络更容易训练。** 它们不是装饰，而是稳定训练的重要结构。
10. **预训练模型并不能呈现很好的对话能力。** 预测下一个 token 得到的是基础模型，指令遵循和偏好对齐还需要后续训练。

---

## 1. 最终要实现什么？

目标是实现一个 Decoder-only Transformer。它接收 token 序列，输出序列中每个位置对“下一个 token”的概率分布。

```text
原始文本
  ↓
分词与整数编码
  ↓
Token Embedding + Position Embedding
  ↓
多层 Transformer Block
  ├─ 因果多头自注意力
  └─ 前馈网络
  ↓
LayerNorm
  ↓
词表维度的 logits
  ↓
Softmax + 采样
  ↓
下一个 token
```

训练阶段一次处理完整批次并计算交叉熵；生成阶段一次只取最后一个位置的输出，采样出一个新 token，再重复运行。

---

## 2. 文本如何变成模型能处理的数据？

### 2.1 字符级分词

最容易理解的方案，是把每个不同字符都视为一个 token：

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
```

例如：

```text
"hello" → [46, 43, 50, 50, 53]
```

具体编号没有语义，只要编码和解码互为逆运算即可。

### 2.2 字符级与子词级的取舍

| 方案 | 词表 | 序列长度 | 优点 | 缺点 |
|---|---:|---:|---|---|
| 字符级 | 小 | 长 | 简单、无未登录词、便于学习 | 计算低效，单个 token 语义弱 |
| 子词级 | 较大 | 较短 | 效率高，能表示常见词根和片段 | 分词算法更复杂 |

实际的大模型通常采用 BPE 一类子词分词。分词方案本身也是系统设计的一部分：词表越大，Embedding 和输出层越大；词表越小，同一段文本就需要更多 token。

### 2.3 张量化与数据划分

模型需要整数索引，因此将全文编码为 `torch.long`：

```python
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

训练集用于更新参数，验证集只用于评估泛化能力。若训练损失持续下降而验证损失开始上升，通常意味着过拟合。

---

## 3. 如何构造语言模型的训练批次？

设：

- `batch_size = B`：一次并行处理多少条序列；
- `block_size = T`：模型最多查看多长的上下文；
- `vocab_size = V`：词表大小。

批次生成器可以写成：

```python
def get_batch(split):
    source = train_data if split == "train" else val_data
    starts = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in starts])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
```

`x` 与 `y` 的形状均为 `(B, T)`，区别是 `y` 比 `x` 向后错开一个位置：

```text
x: [18, 47, 56,  5, 57]
y: [47, 56,  5, 57, 43]
```

这里并不只有一个训练样本，而是同时包含：

```text
[18]                 → 47
[18, 47]             → 56
[18, 47, 56]         → 5
[18, 47, 56, 5]      → 57
[18, 47, 56, 5, 57]  → 43
```

因此，一个 `(B, T)` 批次实际提供了 `B × T` 次下一个 token 预测。

---

## 4. 最小基线：Bigram 语言模型

### 4.1 核心思想

Bigram 模型只看当前 token，不看更早的历史。可以直接创建一个形状为 `(V, V)` 的 Embedding 表：

- 输入 token 的编号负责选中一行；
- 这一行的 `V` 个数，就是下一个 token 的 logits。

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, V)
        loss = None

        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
            )

        return logits, loss
```

### 4.2 为什么要展平张量？

`cross_entropy` 把最后一个维度视为类别，常用输入形状是 `(N, C)`：

```text
logits:  (B, T, V) → (B×T, V)
targets: (B, T)    → (B×T)
```

这里 `N = B×T`，`C = V`。

如果初始预测接近均匀分布，交叉熵应接近：

$$
L \approx -\log\left(\frac{1}{V}\right)=\log V
$$

这个数可以用来快速检查损失是否处于合理量级。

### 4.3 Bigram 基线的价值与局限

它能够验证整条训练链路是否正确：数据、前向传播、损失、反向传播、优化器和生成循环都可以先跑通。

但它只能学习“某个字符后面通常出现什么字符”，无法根据长距离上下文改变预测。要让 token 结合历史信息，需要引入注意力机制。

---

## 5. 注意力之前的关键数学：加权聚合

假设张量 `x` 的形状为 `(B, T, C)`：

- `B`：批次维度；
- `T`：时间或序列维度；
- `C`：每个 token 的特征维度。

如果当前位置只允许使用自己和此前 token，可以构造一个下三角矩阵：

```python
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(dim=1, keepdim=True)
out = wei @ x
```

以 `T=4` 为例：

```text
wei = [[1,   0,   0,   0],
       [1/2, 1/2, 0,   0],
       [1/3, 1/3, 1/3, 0],
       [1/4, 1/4, 1/4, 1/4]]
```

矩阵乘法后，第 `t` 个位置就得到从第 `0` 个位置到第 `t` 个位置的平均值。

更通用的写法是先把未来位置设为负无穷，再做 Softmax：

```python
wei = torch.zeros(T, T)
mask = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(mask == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
out = wei @ x
```

Softmax 同时完成两件事：

1. 把可见位置的分数变成非负权重；
2. 让每一行权重之和等于 1。

平均历史信息仍然过于粗糙，因为它对所有历史 token 一视同仁。自注意力的改进，就是让模型根据内容动态计算这些权重。

---

## 6. 自注意力：让 token 有选择地读取上下文

### 6.1 Query、Key、Value

对每个 token 的表示 $x$ 做三次线性投影：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

可以这样理解：

- **Query**：当前 token 正在寻找什么信息；
- **Key**：当前 token 能被怎样匹配；
- **Value**：匹配成功后实际传递的内容。

位置 $i$ 对位置 $j$ 的关注程度来自 Query 与 Key 的点积：

$$
s_{ij}=q_i\cdot k_j
$$

点积越大，表示二者越匹配。

### 6.2 缩放点积注意力

完整计算为：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

其中：

- $d_k$ 是单个注意力头的维度；
- $M$ 是因果遮罩，可见位置为 0，未来位置为 $-\infty$；
- Softmax 产生每个位置对历史 token 的读取权重。

除以 $\sqrt{d_k}$ 是为了控制点积的方差。维度越大，未经缩放的点积越容易出现极端值，使 Softmax 过早变得尖锐、梯度变差。

### 6.3 一个注意力头

```python
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                         # (B, T, H)
        q = self.query(x)                       # (B, T, H)

        wei = q @ k.transpose(-2, -1)           # (B, T, T)
        wei = wei * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf"),
        )
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)                       # (B, T, H)
        return wei @ v                          # (B, T, H)
```

### 6.4 自注意力的几个重要性质

- 注意力没有卷积的固定邻域限制，任意两个可见位置都能直接交互；
- 注意力只根据当前内容动态决定聚合比例；
- 矩阵乘法能并行处理所有位置，不需要逐 token 写循环；
- 自注意力本身不知道顺序，必须额外加入位置信息；
- 因果遮罩是语言生成任务的要求，不是所有注意力任务都必须使用。

---

## 7. Token Embedding 与 Position Embedding

只使用 token embedding 时，相同 token 在任何位置的初始表示都一样；而语言顺序显然重要。因此为每个位置再学习一个 embedding：

```python
tok_emb = self.token_embedding_table(idx)       # (B, T, C)
pos_emb = self.position_embedding_table(
    torch.arange(T, device=idx.device)
)                                                # (T, C)
x = tok_emb + pos_emb                            # (B, T, C)
```

这里利用广播机制，把同一组位置向量加到批次中的每条序列。

两类 embedding 分工明确：

- Token Embedding 表示“这是什么”；
- Position Embedding 表示“它在哪里”。

经过多层注意力后，二者共同形成上下文化表示：同一个 token 出现在不同位置、不同上下文中，会得到不同的最终向量。

---

## 8. 从一个注意力头到多头注意力

单个注意力头只能在一个表示子空间中建立关系。多头注意力并行运行多个头，让不同的头学习不同类型的匹配模式，再拼接结果：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embd, block_size, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))
```

若 `n_embd = C`、注意力头数为 `h`，通常令每个头的维度为 `C/h`，这样拼接后仍回到 `C` 维。

多头的含义不是提前指定某个头负责语法、某个头负责指代，而是给模型多个并行的关系建模通道，具体分工由训练自动形成。

---

## 9. 前馈网络：让每个 token 独立计算

注意力完成 token 之间的信息交换，前馈网络则对每个位置独立处理：

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

中间层扩张到 `4 × n_embd`，给每个 token 更多计算空间，再投影回残差通道的维度。

可以用一句话区分两部分：

> 注意力负责“读取哪些上下文”，前馈网络负责“如何加工读取到的信息”。

---

## 10. 残差连接、LayerNorm 与 Dropout

### 10.1 残差连接

深层网络若每层都彻底改写输入，优化会很困难。残差连接保留一条直通路径：

```python
x = x + self.sa(x)
x = x + self.ffwd(x)
```

子层只需学习“应该在原表示上增加什么”，梯度也能沿残差路径更顺畅地传播。

### 10.2 LayerNorm

LayerNorm 对每个 token 的特征维度做归一化，不跨 batch 汇总统计量。预归一化结构通常写成：

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

与 BatchNorm 相比，它不依赖批次中的其他样本，更适合长度变化的序列建模。

### 10.3 Dropout

Dropout 在训练时随机屏蔽部分通道或注意力权重，降低网络对特定路径的依赖。验证和推理时必须切换到 `eval()`，使 Dropout 停用。

---

## 11. Transformer Block

把多头自注意力和前馈网络组合起来：

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(
            n_head, n_embd, block_size, dropout
        )
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

多个 Block 串联后，token 会反复经历“跨位置通信 → 本地计算”：

```text
x
├─ LayerNorm → 多头因果自注意力 → 残差相加
└─ LayerNorm → 前馈网络         → 残差相加
        ↓
      下一层
```

层数越多，模型就能反复整合和加工上下文，但训练成本、显存占用和过拟合风险也随之增加。

---

## 12. 完整的 Decoder-only 语言模型

```python
class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        n_embd,
        n_head,
        n_layer,
        dropout,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)           # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )                                                    # (T, C)
        x = tok_emb + pos_emb                                # (B, T, C)
        x = self.blocks(x)                                   # (B, T, C)
        x = self.ln_f(x)                                     # (B, T, C)
        logits = self.lm_head(x)                             # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(B * T, -1),
                targets.reshape(B * T),
            )

        return logits, loss
```

最后的 `lm_head` 把每个 token 的 `n_embd` 维上下文表示映射到 `vocab_size` 维，从而为词表中的每个候选 token 产生一个 logit。

---

## 13. 训练循环与验证

### 13.1 基本训练循环

```python
model = GPTLanguageModel(...).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

每一步都经历：

```text
随机取批次 → 前向传播 → 计算损失 → 清空旧梯度
          → 反向传播 → 更新参数
```

### 13.2 为什么使用 AdamW？

AdamW 结合了自适应学习率与解耦权重衰减，通常比朴素 SGD 更适合 Transformer 的初始实验。不过，优化器不能弥补错误的数据、遮罩或张量维度。

### 13.3 稳定估计损失

单个随机批次的损失波动较大，评估时应对多个批次求平均：

```python
@torch.no_grad()
def estimate_loss(eval_iters):
    result = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        result[split] = losses.mean().item()

    model.train()
    return result
```

`torch.no_grad()` 避免保存反向传播所需的中间结果，节省内存并提高评估速度。

### 13.4 如何读训练曲线？

| 现象 | 可能原因 | 调整方向 |
|---|---|---|
| 训练和验证损失都高 | 欠拟合、训练不足、学习率不合适 | 增大模型、延长训练、调整学习率 |
| 训练损失低，验证损失高 | 过拟合 | 增加数据、Dropout、权重衰减或减少模型容量 |
| 损失剧烈震荡或发散 | 学习率过大、数值或实现错误 | 降低学习率，检查缩放与遮罩 |
| 损失几乎不变 | 梯度未更新、标签错误、学习率过小 | 检查 `backward()`、`step()` 和错位标签 |

---

## 14. 自回归生成

生成时只关心最后一个位置对下一 token 的预测：

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :]          # (B, V)
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

几个关键点：

1. **截断上下文**：位置 embedding 和注意力遮罩只支持 `block_size` 长度，所以只保留最后一段上下文；
2. **取最后位置**：前面位置预测的是训练序列内部的后继，生成只需要当前序列末尾的预测；
3. **采样而非固定取最大值**：从概率分布中抽样会产生多样结果；
4. **逐步追加**：新 token 加到序列末尾后，会成为下一轮预测的上下文。

模型是概率系统，因此同一个起点可能产生不同结果。输出流畅程度来自模型对条件概率分布的学习，不代表生成内容一定真实。

---

## 15. Encoder、Decoder 与 Cross-Attention

三种常见结构的区别主要在于信息可见范围和任务形式：

| 结构 | 注意力可见范围 | 典型用途 |
|---|---|---|
| Encoder-only | 每个位置可以看整个输入 | 文本理解、分类、表示学习 |
| Decoder-only | 每个位置只能看自己及左侧 | 自回归语言生成 |
| Encoder-Decoder | Decoder 既看左侧输出，也读取 Encoder 表示 | 翻译、摘要等输入到输出任务 |

自注意力中，Query、Key、Value 都来自同一序列。Cross-Attention 则通常由 Decoder 提供 Query，Encoder 输出提供 Key 和 Value。

本实现没有 Encoder，也没有 Cross-Attention；因果遮罩使它成为 Decoder-only 语言模型。

---

## 16. 预训练与对齐训练不是一回事

### 16.1 预训练

预训练使用大量文本，目标仍是预测下一个 token。模型由此学习语言结构、写作模式和数据中反复出现的知识关系。预训练完成后，它更像一个通用文本续写器。

### 16.2 后训练

如果希望模型理解指令、按问答格式回应并符合人类偏好，还需要后训练，例如：

- 使用高质量“指令—回答”样本做监督微调；
- 收集人类对多个候选回答的偏好；
- 用偏好信号继续优化模型行为。

因此，“会续写文本”和“会作为助手回答问题”是两个不同阶段的能力。小型字符模型主要展示预训练和生成机制，不应把它的输出质量与完整助手系统等同起来。

---

## 17. 关键张量维度速查

| 名称 | 形状 | 含义 |
|---|---|---|
| `idx` | `(B, T)` | token 整数索引 |
| `tok_emb` | `(B, T, C)` | token 表示 |
| `pos_emb` | `(T, C)` | 位置表示 |
| `q`, `k`, `v` | `(B, T, H)` | 单头的 Query、Key、Value |
| `q @ kᵀ` | `(B, T, T)` | 每个位置对每个位置的注意力分数 |
| 单头输出 | `(B, T, H)` | 加权聚合后的 Value |
| 多头拼接 | `(B, T, C)` | 所有头的输出合并 |
| `logits` | `(B, T, V)` | 每个位置对下一 token 的未归一化分数 |
| 展平 logits | `(B×T, V)` | 交叉熵所需的类别预测 |
| 展平 targets | `(B×T)` | 每个预测对应的正确类别 |

其中：

- `B`：batch size；
- `T`：sequence length；
- `C`：embedding dimension；
- `H`：head size；
- `V`：vocabulary size。

---

## 18. 最容易出错的地方

### 18.1 标签没有错开一位

错误做法是让模型重建当前 token；正确目标是预测下一个 token。

```text
输入：data[i     : i+T]
标签：data[i+1   : i+T+1]
```

### 18.2 忘记因果遮罩

若训练时当前位置能看到未来 token，模型会通过“偷看答案”获得虚假的低损失；生成时没有未来信息，效果会立刻崩溃。

### 18.3 Softmax 维度写错

注意力应在最后一个维度归一化，即让每个 Query 对所有 Key 的权重之和为 1：

```python
F.softmax(wei, dim=-1)
```

### 18.4 忘记缩放点积

注意力分数应乘以 `head_size ** -0.5`。维度增大后，缺少缩放容易让 Softmax 过于尖锐。

### 18.5 Embedding 维度与多头数不匹配

通常要求：

```text
n_embd % n_head == 0
```

否则无法把通道平均分给多个头并正确拼接。

### 18.6 生成时上下文超过 block size

必须使用：

```python
idx_cond = idx[:, -block_size:]
```

否则位置 embedding 索引和遮罩大小都会出错。

### 18.7 评估后忘记恢复训练模式

评估前使用 `model.eval()`，结束后恢复 `model.train()`，否则 Dropout 的行为不正确。

### 18.8 清空梯度的时机错误

PyTorch 默认累加梯度。每次反向传播前都要清空上一轮梯度：

```python
optimizer.zero_grad(set_to_none=True)
```

---

## 19. 从最小模型逐步扩展的路线

不要一开始就堆出完整 Transformer。更可靠的实现顺序是：

1. 完成字符级 `encode` 与 `decode`，验证可逆；
2. 构造 `(B, T)` 的输入与错位标签；
3. 实现 Bigram 模型，跑通损失和生成；
4. 加入训练集与验证集的损失评估；
5. 用下三角矩阵实现历史 token 的平均聚合；
6. 把固定平均权重替换为 Query-Key 动态权重；
7. 加入 Value 投影、缩放和因果遮罩；
8. 加入 token embedding 与 position embedding；
9. 从单头扩展为多头注意力；
10. 加入前馈网络、残差连接与 LayerNorm；
11. 堆叠多个 Block，加入 Dropout；
12. 扩大模型与训练步数，观察训练和验证损失。

这种顺序的意义在于：每一步都保留一个可运行系统，出现问题时容易定位是数据、维度、注意力还是训练环节造成的。

---

## 20. 可调超参数及影响

| 参数 | 增大后的主要影响 |
|---|---|
| `batch_size` | 梯度更稳定，但显存占用增加 |
| `block_size` | 可利用更长上下文，注意力计算和显存开销显著增加 |
| `n_embd` | 表示能力增强，参数量与计算量增加 |
| `n_head` | 提供更多关系建模通道，但每头维度需要合理分配 |
| `n_layer` | 增加上下文加工深度，也让优化更困难 |
| `dropout` | 正则化更强，过大时会导致欠拟合 |
| `learning_rate` | 过大可能发散，过小会使训练缓慢 |
| `max_iters` | 训练更充分，但可能开始过拟合 |

尤其要注意，自注意力分数矩阵的大小是 `(T, T)`，所以标准注意力对序列长度的时间和内存开销近似为 $O(T^2)$。

---

## 21. 复习自测

1. 为什么语言模型的输入和标签只相差一个位置？
2. 一个 `(B, T)` 批次为什么包含 `B×T` 个预测任务？
3. Bigram 模型为什么不能利用长距离上下文？
4. Query、Key、Value 分别承担什么角色？
5. 因果遮罩为什么在训练和生成之间建立了一致性？
6. 注意力分数为什么要除以 $\sqrt{d_k}$？
7. 为什么自注意力仍需要 position embedding？
8. 多头注意力输出为何通常仍保持 `n_embd` 维？
9. 残差连接和 LayerNorm 分别解决什么问题？
10. 为什么训练损失下降而验证损失上升意味着过拟合？
11. 生成时为什么只使用最后一个位置的 logits？
12. Decoder-only 与 Encoder-Decoder 的信息流有何不同？
13. 为什么完成预训练后仍需要后训练才能成为可靠的指令助手？

---

## 22. 最终心智模型

整个系统可以压缩成五层理解：

```text
第 1 层：文本被编码为 token
第 2 层：token 与位置被映射为向量
第 3 层：因果自注意力让 token 有选择地读取历史
第 4 层：多层通信与计算形成上下文化表示
第 5 层：输出层给出下一个 token 的概率并循环生成
```

GPT 并不是把句子一次性写出来，也不是从数据库中检索一段固定答案。它在每一步根据当前上下文计算下一 token 的概率分布，再把采样结果接回上下文。Transformer 的作用，是让这个“下一 token 预测”充分利用长距离、内容相关的上下文信息。

当数据、模型规模和训练计算持续扩大时，这套简单目标会逐步形成越来越强的语言建模能力；但模型规模不会改变其基本机制：**表示 token、聚合上下文、预测下一个 token、重复生成。**

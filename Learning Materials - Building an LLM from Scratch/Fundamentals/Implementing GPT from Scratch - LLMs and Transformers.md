# Implementing GPT from Scratch: LLMs and Transformers

## The Core Idea Behind GPT

GPT converts text into tokens, uses causal self-attention to let each token gather information from the preceding context, and learns to predict the next token. At inference time, each newly generated token is appended to the input and fed back into the model, producing text autoregressively.

## Key points

1. **A language model learns the conditional probability of the next token.** Given the existing sequence, it estimates $P(x_t\mid x_1,\ldots,x_{t-1})$.
2. **Tokens are the model's basic units of text.** Character-level tokenization is the easiest to understand, while subword tokenization is usually more efficient.
3. **Training inputs and labels differ by a one-token shift.** Each input position is trained to predict the token that follows it.
4. **A sequence of length $T$ supplies $T$ next-token predictions.** Every position uses its left context to predict what comes next.
5. **A bigram model is the smallest runnable baseline.** It predicts the next token from only the current token and cannot model longer context.
6. **Self-attention is the communication mechanism between tokens.** A query asks “what am I looking for?”, a key describes “what do I contain?”, and a value carries the information to retrieve.
7. **The causal mask prevents the model from reading the future.** A position can use only itself and positions to its left.
8. **Multi-head attention communicates across tokens; the feedforward network transforms each position independently.** Alternating these components produces a Transformer block.
9. **Residual connections, LayerNorm, and dropout make deep networks trainable and stable.** They are essential parts of the architecture, not decoration.
10. **A pretrained base model is not automatically a good conversationalist.** Next-token pretraining builds general capability; instruction following and preference alignment require post-training.

---

## 1. What will be achieved in the end?

The goal is to achieve a Decoder-only Transformer. It receives the token sequence and outputs the probability distribution of the "next token" for each position in the sequence.

```text
Original text
  ↓
tokenization and integer coding
  ↓
token embedding + Position embedding
  ↓
Multi-layer Transformer Block
├─ Cause and effect multi-headed self-attention
└─ Feedforward network
  ↓
LayerNorm
  ↓
Logits of the vocabulary dimension
  ↓
Softmax + sampling
  ↓
The next token
```

In the training stage, the complete batch is processed and the cross-entropy is calculated; in the generation stage, only the output of the last position is taken at a time, a new token is sampled, and then executed repeatedly.

---

## 2. How can text become data that models can handle?

### 2.1 Character-level tokenization

The easiest way to understand is to regard each different character as a token:

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
```

For example:

```text
"hello" → [46, 43, 50, 50, 53]
```

The specific number has no semantics, as long as the encoding and decoding are inverse to each other.

### 2.2 The choose between character levels and sub-word levels

| Scheme | vocabulary | Sequence Length | Advantages | Disadvantages |
|---|---:|---:|---|---|
| Character level | Small | Long | Simple, no out-of-vocabulary words, easy to learn | Inefficient calculation, weak semantics of a single token |
| Sub-word level | Large | Short | High efficiency, can represent common roots and fragments | Word segmentation algorithm is more complex |

The actual large model usually adopts sub-word tokenization such as BPE. The word segmentation scheme itself is also part of the system design: the larger the vocabulary, the larger the embedding and output layer; the smaller the vocabulary, the more tokens are needed for the same text.

### 2.3 Tensionization and data division

The model needs an integer index, so the full text is encoded as `torch.long`:

```python
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

The training set is used to update parameters, and the validation set is only used to evaluate the generalization ability. If the training loss continues to decline and the verification loss begins to rise, it usually means overfitting.

---

## 3. How to construct the training batch of the language model?

Set up:

- `batch_size = B`: How many sequences can be processed in parallel at a time;
- `block_size = T`: How long is the maximum context that the model can view;
- `vocab_size = V`: vocabulary size.

The batch generator can be written as:

```python
def get_batch(split):
    source = train_data if split == "train" else val_data
    starts = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in starts])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
```

The shape of `x` and `y` are both `(B, T)`. The difference is that `y` is staggered backwards compared with `x`:

```text
x: [18, 47, 56,  5, 57]
y: [47, 56,  5, 57, 43]
```

There is not only one training sample here, but also contains:

```text
[18]                 → 47
[18, 47]             → 56
[18, 47, 56]         → 5
[18, 47, 56, 5]      → 57
[18, 47, 56, 5, 57]  → 43
```

Therefore, a `(B, T)` batch actually provides the next token prediction of `B × T`.

---

## 4. Minimum baseline: Bigram language model

### 4.1 Core Thoughts

The Bigram model only looks at the current token, not the earlier history. You can directly create an embedding table with the shape of `(V, V)`:

- Enter the number of the token to select a line;
- The number of `V` in this line is the logits of the next token.

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.embedding(vocab_size, vocab_size)

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

### 4.2 Why do we need to flatten the tensor?

`cross_entropy` regards the last dimension as a category, and the commonly used input shape is `(N, C)`:

```text
logits:  (B, T, V) → (B×T, V)
targets: (B, T)    → (B×T)
```

Here `N = B×T`, `C = V`.

If the initial prediction is close to uniform distribution, the cross-entropy should be close to:

$$
L \approx -\log\left(\frac{1}{V}\right)=\log V
$$

This number can be used to quickly check whether the loss is at a reasonable level.

### 4.3 The value and limitations of Bigram baseline

It can verify whether the entire training pipeline is correct: data, forward propagation, loss, backpropagation, optimizer and generation loop can all run first.

However, it can only learn "what characters usually appear after a character" and cannot change the prediction according to the long-distance context. To make token combine historical information, it is necessary to introduce an attention mechanism.

---

## 5. Key Mathematics Before attention: Weighted Aggregation

Suppose the shape of the tenser `x` is `(B, T, C)`:

- `B`: batch dimension;
- `T`: time or sequence dimension;
- `C`: the characteristic dimension of each token.

If the current position only allows the use of itself and the previous token, a lower triangle matrix can be constructed:

```python
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(dim=1, keepdim=True)
out = wei @ x
```

Take `T=4` as an example:

```text
wei = [[1,   0,   0,   0],
       [1/2, 1/2, 0,   0],
       [1/3, 1/3, 1/3, 0],
       [1/4, 1/4, 1/4, 1/4]]
```

After matrix multiplication, the average from the `0` position to the `t` position is obtained.

A more general way to write is to set the future position to negative infinity first, and then do Softmax:

```python
wei = torch.zeros(T, T)
mask = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(mask == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
out = wei @ x
```

Softmax accomplishes two things at the same time:

1. Turn the score of the visible position into a non-negative weight;
2. Let the sum of the weights of each line be equal to 1.

The average historical information is still too rough, because it treates all historical tokens equally. The improvement of self-attention is to let the model calculate these weights dynamically according to the content.

---

## 6. Self-attention: Let token read the context selectively

### 6.1 Query, Key, Value

Make three linear projections for each token's representation $x$:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

It can be understood as follows:

- **Query**: What information is the current token looking for;
- **Key**: How the current token can be matched;
- **Value**: The content actually delivered after successful matching.

Position $i$ The degree of attention to position $j$ comes from the point product of Query and Key:

$$
s_{ij}=q_i\cdot k_j
$$

The larger the product, the more the two match.

### 6.2 Zoom to accumulate attention

The complete calculation is:

$$
\operatorname{attention}(Q,K,V) =\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

Among them:

- $d_k$ is the dimension of a single attention head;
- $M$ is a causal mask, the visible position is 0, and the future position is $-\infty$;
- Softmax generates the read weight of the historical token for each position.

Divided by $\sqrt{d_k}$ is to control the variance of the point product. The larger the dimension, the more prone the extreme value of the unscaled point product, which makes Softmax prematurely sharp and the gradient worse.

### 6.3 A head of attention

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

### 6.4 Several important properties of self-attention

- attention has no convolutional fixed neighborhood restrictions, and any two visible positions can directly interact;
- attention only determines the aggregation ratio according to the current content dynamics;
- Matrix multiplication can handle all positions in parallel, and there is no need to write loops one by token;
- Self-attention itself does not know the order, and additional location information must be added;
- causal masking is a requirement for language-generated tasks, and not all attention tasks must be used.

---

## 7. token embedding and Position embedding

When only token embedding is used, the initial representation of the same token in any position is the same; and the language order is obviously important. Therefore, learn another embedding for each position:

```python
tok_emb = self.token_embedding_table(idx)       # (B, T, C)
pos_emb = self.position_embedding_table(
    torch.arange(T, device=idx.device)
)                                                # (T, C)
x = tok_emb + pos_emb                            # (B, T, C)
```

Here, the broadcast mechanism is used to add the same set of position vectors to each sequence in the batch.

The division of labor of the two types of embedding is clear:

- token embedding means "what is this";
- Position embedding means "where is it".

After multiple layers of attention, the two together form an upper and lower cultural representation: the same token appears in different positions and different contexts, and different final vectors will be obtained.

---

## 8. From one head to multiple attention

A single attention head can only establish a relationship in a representation subspace. Multi-head attention runs multiple heads in parallel, so that different heads can learn different types of matching patterns, and then stitch the results:

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

If `n_embd = C` and the number of attention heads is `h`, the dimension of each head is usually `C/h`, and it will still return to the `C` dimension after splicing.

The meaning of multi-head is not to specify a header in charge of syntax and a header in charge of reference in advance, but to model multiple parallel relationship models channels for the model, and the specific division of labor is automatically formed by training.

---

## 9. Feedforward network: let each token be calculated independently

attention completes the information exchange between tokens, and the feedforward network processes each position independently:

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

The middle layer expands to `4 × n_embd`, gives each token more computing space, and then projects back the dimension of the residual channel.

Two parts can be distinguished in one sentence:

> attention is responsible for "which context to read", and the feedforward network is responsible for "how to process the information read".

---

## 10. Residual connection, LayerNorm and Dropout

### 10.1 Residual connection

If each layer of the deep network completely rewrites the input, it will be difficult to optimize. The residual connection keeps a pass-through path:

```python
x = x + self.sa(x)
x = x + self.ffwd(x)
```

The sublayer only needs to learn "what should be added to the original representation", and the gradient can also spread more smoothly along the residual path.

### 10.2 LayerNorm

LayerNorm normalizes the characteristic dimension of each token, and does not aggregate statistics across batches. Pre-normalized structures are usually written as:

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

Compared with BatchNorm, it does not rely on other samples in the batch and is more suitable for sequence modeling with length changes.

### 10.3 Dropout

Dropout randomly blocks some channels or attention weights during training to reduce the dependence of the network on specific paths. When verifying and reasoning, you must switch to `eval()` to disable Dropout.

---

## 11. Transformer Block

Combine multi-head self-attention and feedforward network:

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

After multiple Blocks are connected in series, token will repeatedly experience "cross-location communication → local computing":

```text
x
├─ LayerNorm → Multi-headed causal self-attention → residual addition
└─ LayerNorm → Feedforward Network → Residual Addition
        ↓
The next floor
```

The more layers there are, the more the model can repeatedly integrate and process the context, but the training cost, memory occupation and overfitting risks also increase.

---

## 12. Complete Decoder-only language model

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
        self.token_embedding_table = nn.embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.embedding(block_size, n_embd)
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

Finally, `lm_head` maps the `n_embd` dimension context representation of each token to the `vocab_size` dimension, thus generating a logit for each candidate token in the vocabulary.

---

## 13. Training cycle and verification

### 13.1 Basic training cycle

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

Experience every step:

```text
Random batch → forward propagation → calculate loss → empty the old gradient
→ Reverse propagation → Update parameters
```

### 13.2 Why use AdamW?

AdamW combines adaptive learning rate and decoupling weight attenuation, which is usually more suitable for the initial experiment of Transformer than simple SGD. However, the optimizer cannot make up for the wrong data, mask or tensor dimensions.

### 13.3 Stable estimated loss

The loss of a single random batch fluctuates greatly, and multiple batches should be averaged when evaluating:

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

`torch.no_grad()` avoids saving the intermediate results required for backpropagation, saving memory and improving the evaluation speed.

### 13.4 How to read the training curve?

| Phenomenon | Possible Causes | Adjust Direction |
|---|---|---|
| High loss of training and verification | Underfitting, insufficient training, inappropriate learning rate | Increase the model, extend training, adjust the learning rate |
| Low training loss, high verification loss | Overfitting | Increase data, Dropout, weight decay or reduce model capacity |
| Loss of violent oscillation or divergence | Excessive learning rate, numerical or implementation error | Reduce the learning rate, check scaling and masking |
| The loss is almost unchanged | Gradient is not updated, label error, learning rate is too small | Check `backward()`, `step()` and misaligned labels |

---

## 14. Self-regressive generation

When generating, only care about the prediction of the last position for the next token:

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

Several key points:

1. **Truncate context**: Location embedding and attention mask only support the length of `block_size`, so only the last context is retained;
2. **Taken the last position**: The previous position predicts the succession inside the training sequence, and only the prediction at the end of the current sequence is generated;
3. **Sampling instead of fixing the maximum value**: sampling from the probability distribution will produce diverse results;
4. **Step-by-step addition**: After adding a new token to the end of the sequence, it will become the context of the next round of prediction.

The model is a probability system, so the same starting point may produce different results. The fluency of the output comes from the model's learning of the conditional probability distribution, which does not mean that the generated content must be true.

---

## 15. Encoder, Decoder and Cross-attention

The difference between the three common structures mainly lies in the visible range of information and the form of task:

| Structure | attention Visible Range | Typical Use |
|---|---|---|
| Encoder-only | Each position can see the entire input | Text understanding, classification, representation learning |
| Decoder-only | Each position can only look at itself and the left | Autoregressive language generation |
| Encoder-Decoder | Decoder not only looks at the left output, but also reads Encoder representation | Translation, summary and other input to output tasks |

In self-attention, Query, Key and Value all come from the same sequence. Cross-attention is usually provided by Query by Decoder, and Key and Value are provided by Encoder output.

This implementation has no Encoder or Cross-attention; the causal mask makes it a Decoder-only language model.

---

## 16. Pre-training and alignment training are not the same thing.

### 16.1 Pre-training

Pre-training uses a large amount of text, and the goal is still to predict the next token. From this, the model learns the knowledge relationships that appear repeatedly in the language structure, writing patterns and data. After the pre-training is completed, it is more like a general-purpose text continuer.

### 16.2 Post-training

If you want the model to understand the instructions, respond according to the Q&A format and meet human preferences, you also need post-training, such as:

- Use high-quality "instruction-answer" samples for supervision and fine-tuning;
- Collect human preferences for multiple candidate answers;
- Continue to optimize the model behavior with preference signals.

Therefore, "being able to continue writing text" and "being able to answer questions as an assistant" are two different stages of ability. The small character model mainly shows the pre-training and generation mechanism, and its output quality should not be equated with the complete assistant system.

---

## 17. Quick check of key tensore dimensions

| Name | Shape | Meaning |
|---|---|---|
| `idx` | `(B, T)` | token integer index |
| `tok_emb` | `(B, T, C)` | token representation |
| `pos_emb` | `(T, C)` | Position indication |
| `q`, `k`, `v` | `(B, T, H)` | Single-headed Query, Key, Value |
| `q @ kᵀ` | `(B, T, T)` | attention score for each position |
| Single-head output | `(B, T, H)` | Value after weighted aggregation |
| Multi-head splicing | `(B, T, C)` | Output merging of all heads |
| `logits` | `(B, T, V)` | The unleveled score of each position for the next token |
| Flat logits | `(B×T, V)` | Category prediction required for cross-entropy |
| Flat targets | `(B×T)` | The correct category corresponding to each prediction |

Among them:

- `B`: batch size;
- `T`: sequence length;
- `C`: embedding dimension;
- `H`: head size;
- `V`: vocabulary size.

---

## 18. The most prone to mistakes

### 18.1 The tag is not staggered.

The wrong practice is to let the model rebuild the current token; the correct goal is to predict the next token.

```text
Input: data[i : i+T]
Label: data[i+1 : i+T+1]
```

### 18.2 Forget the mask of cause and effect

If the future token can be seen in the current position during training, the model will get a false low loss through "peeping at the answer"; if there is no future information at the time of generation, the effect will collapse immediately.

### 18.3 Softmax dimension is wrong

attention should be normalized in the last dimension, that is, the sum of the weights of each Query on all Keys should be 1:

```python
F.softmax(wei, dim=-1)
```

### 18.4 Forget to scale the point product

The attention score should be multiplied by `head_size ** -0.5`. After the dimension is increased, the lack of scaling is easy to make Softmax too sharp.

### 18.5 embedding dimension does not match the number of multiple heads

Usually required:

```text
n_embd % n_head == 0
```

Otherwise, the channel cannot be evenly divided into multiple heads and spliced correctly.

### 18.6 When generating, the context exceeds the block size

Must use:

```python
idx_cond = idx[:, -block_size:]
```

Otherwise, the location embedding index and mask size will be wrong.

### 18.7 Forget to resume training mode after evaluation

Use `model.eval()` before evaluation, and restore `model.train()` after the evaluation, otherwise the behavior of Dropout is incorrect.

### 18.8 The timing error of emptying the gradient

PyTorch defaults to the cumulative gradient. The previous round of gradients should be cleared before each reverse propagation:

```python
optimizer.zero_grad(set_to_none=True)
```

---

## 19. The route of gradual expansion from the minimum model

Don't pile up the complete Transformer at the beginning. The more reliable order of implementation is:

1. Complete the character-level `encode` and `decode`, and verify the reversible;
2. Construct the input and misaligned labels of `(B, T)`;
3. Realize the Bigram model, run through loss and generation;
4. Add the loss assessment of the training set and the validation set;
5. Use the triangular matrix to realize the average aggregation of historical tokens;
6. Replace the fixed average weight with Query-Key dynamic weight;
7. Add Value projection, zooming and causal masking;
8. Join token embedding and position embedding;
9. Expand from single-headed to multi-headed attention;
10. Join the feedforward network, residual connection and LayerNorm;
11. Stack multiple Blocks and join Dropout;
12. Expand the model and training steps, observe the training and verify the loss.

The meaning of this sequence is that an operable system is retained at each step, and it is easy to locate whether the problem is caused by data, dimension, attention or training.

---

## 20. ADJUSTABLE SUPER PARAMETERS AND EFFECTS

| Parameters | The main impact after increasing |
|---|---|
| `batch_size` | The gradient is more stable, but the memory occupation increases |
| `block_size` | Longer context can be used, and attention calculation and GPU memory overhead will increase significantly |
| `n_embd` | The representation ability is enhanced, and the parameter volume and compute are increased |
| `n_head` | Provide more relationship modeling channels, but each dimension needs to be reasonably allocated |
| `n_layer` | Increasing the depth of context processing also makes optimization more difficult |
| `dropout` | The regularization is stronger, and if it is too large, it will lead to a lack of fit |
| `learning_rate` | Too big may diverge, and too small will slow down training |
| `max_iters` | The training is more complete, but it may have started to fit |

IN PARTICULAR, IT SHOULD BE NOTED THAT THE SIZE OF THE SELF-ATTENTION SCORE MATRIX IS `(T, T)`, SO THE TIME AND MEMORY EXPENSS OF STANDARD ATTENTION TO THE SEQUENCE LENGTH ARE SIMILAR TO $O(T^2)$.

---

## 21. Review and test yourself

1. Why is there only one position difference between the input and the label of the language model?
2. Why does a `(B, T)` batch contain `B×T` prediction tasks?
3. Why can't the Bigram model use the long-distance context?
4. What roles do Query, Key and Value play respectively?
5. Why does the Causal Mask establish consistency between training and generation?
6. Why should the attention score be divided by $\sqrt{d_k}$?
7. Why does self-attention still need position embedding?
8. Why does the multi-head attention output usually maintain the `n_embd` dimension?
9. What problems do residual connection and LayerNorm solve respectively?
10. Why does the decrease in training loss and the increase in verification loss mean overfitting?
11. Why is only the logits of the last position used when generating?
12. What is the difference between Decoder-only and Encoder-Decoder information flow?
13. Why do you still need post-training to become a reliable instruction assistant after completing the pre-training?

---

## 22. The Final Mental Model

The whole system can be compressed into five layers of understanding:

```text
Layer 1: The text is encoded as token
Layer 2: token and position are mapped into vectors
Layer 3: causal self-attention allows token to read history selectively
Layer 4: Multi-layer communication and computing form up and down cultural representation
Layer 5: The output layer gives the probability of the next token and cycles it.
```

GPT is not to write sentences all at once, nor to retrieve a fixed answer from the database. It calculates the probability distribution of the next token according to the current context at each step, and then takes the sampling results back to the context. The function of Transformer is to make full use of long-distance and content-related context information for this "next token prediction".

When data, model scale and training computing continue to expand, this set of simple goals will gradually form more and more strong language modeling capabilities; but the model scale will not change its basic mechanism: **represents token, aggregate context, predicts the next token, and repeats generation.**

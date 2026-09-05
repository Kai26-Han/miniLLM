# Modern LLM Architecture: From Transformer Backbones to Multimodal and Reasoning Systems

> The core of this article is the mainstream large model architecture in the industry. miniLLM is only used as a runnable teaching mapping at the end of the article to explain how these designs fall into the code.

## 1. First, distinguish the four levels of the "big model architecture"

When discussing large model architecture, different levels are often mixed together. The more accurate disassembly method is:

1. **Sequence modeling paradigm**: Encoder-only, Decoder-only, Encoder–Decoder, and attention–SSM hybrid architecture.
2. **Transformer Block**: How to combine attention, FFN/MoE, normalization, residual and position coding.
3. **Model System**: How to enter the same model in text, visual, audio, tools, etc., and how to train and reason in parallel.
4. **Service System**: Prefill, Decode, KV Cache, continuous batch processing, quantification and distributed reasoning.

Therefore, "using MoE" cannot fully describe a model; it only shows that the FFN sublayer uses sparse experts, but does not attention, position coding, multimodal and service architecture.

## 2. Three classic Transformer paradigms

### 2.1 Encoder-only: Good at understanding and expression

Each token in Encoder can usually pay attention to the whole input in both directions and output the contextual representations. BERT models are suitable for classification, retrieval, entity identification and vector representation.

Its advantage is that the input is fully understood, and its limitation is that it cannot naturally generate continuous text in an autoregressive way.

### 2.2 Decoder-only: the backbone of the current general generation model

Decoder-only uses the causal mask, and the \(t\) position can only see token no later than \(t\):

\[
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t})
\]

The same goal can unify text completeness, dialogue, code, structured output and tool call. Llama, Mistral, Gemma, Qwen, DeepSeek and other public model families have proved the scalability of this route.

### 2.3 Encoder–Decoder: Input understanding and output generation division of labor

Encoder reads the input in both directions, and Decoder generates output through Cross-attention conditions. T5, BART and original Transformer belong to this paradigm and are still valuable in tasks with clear input and output boundaries such as translation, summary, voice recognition, etc.

| Paradigm | attention Visibility | Typical Goal | Main Scene |
|---|---|---|---|
| Encoder-only | Input two-way | MLM, comparative learning | representation, retrieval, classification |
| Decoder-only | Cause and effect one-way | Next-token Prediction | General generation, dialogue, code, Agent |
| Encoder–Decoder | Encoder Two-way, Decoder Cause and Effect | Denoising, Seq2Seq | Translation, Abstract, Cross-modal Generation |

## 3. Modern Decoder-only common skeleton

Although the modern text model has a lot of differences in details, the main trunk is highly similar:

```text
token IDs
  -> token embedding
  -> N × Decoder Block
       -> Norm -> Causal Self-attention -> Residual
       -> Norm -> Dense FFN / Sparse MoE -> Residual
  -> Final Norm
  -> LM Head
  -> Next-token logits
```

This architecture has become mainstream, not because it is theoretically optimal in every dimension, but because it has: unified training goals, regular matrix calculation, mature parallel schemes, stable expansion laws and perfect software and hardware ecology.

## 4. Input layer: tokenizer, embedding and output head

tokenizer converts text or structure tags into discrete IDs, and embedding maps IDs to \(d_{model}\) dimension vectors. The design of the vocabulary will also affect:

- Multi-language and code compression rate;
- embedding/LM Head parameter quantity;
- Sequence length and attention cost;
- The stability of special tags, tool calls and chat template.

LM Head will project the hidden state back to the vocabulary space. Input embedding and output heads can share weight (weight tying) to reduce parameters and enhance the consistency of input and output representation; oversized models may also choose not to share based on training or deployment goals.

## 5. Location coding: RoPE has become mainstream, but not all of the long context

Absolute position embedding is common in early models, and modern Decoder-only uses RoPE more often. RoPE acts position-related rotation on Query and Key, so that the point product naturally contains relative position information.

If a pair of two-dimensional components is \((x_1,x_2)\), the position \(m\) corresponds to the angle \(m\theta\), then the rotation is:

\[
\begin{bmatrix}x'_1\\x'_2\end{bmatrix} = \begin{bmatrix} \cos(m\theta)&-\sin(m\theta)\\ \sin(m\theta)&\cos(m\theta) \end{bmatrix} \begin{bmatrix}x_1\\x_2\end{bmatrix}
\]

Long context usually also requires position scaling, long sequence continuation training, matching data distribution and retrieval ability evaluation. Changing the maximum position in the configuration from 8K to 128K does not mean that the model really obtains the effective context capability of 128K.

## 6. attention: From MHA to GQA, MLA and local attention

### 6.1 MHA, MQA and GQA

- **MHA**: Each Query header has an independent K/V header, which is expressive, but the KV Cache is large.
- **MQA**: All Query headers share a set of K/V, which significantly reduces KV Cache and compresses more radically.
- **GQA**: Several Query headers share a K/V header to achieve a balance between quality and inference cost, which has become a common solution.

If the number of layers is \(L\), the context length is \(S\), the number of KV heads is \(H_{kv}\), and the head dimension is \(D_h\), the single sample KV Cache element is approximately:

\[
2LSH_{kv}D_h
\]

This shows that reducing the number of KV heads will directly reduce the long context decoding memory.

### 6.2 MLA: compressed KV representation

Multi-head Latent attention compresses K/V to low-dimensional potential representation and reconstructs the relevant components when using it. It is similar to the goal of GQA, which is to reduce the reasoning cache and bandwidth, but the structure is different from the engineering implementation. DeepSeek-V2/V3 is its representative public practice.

### 6.3 Full, Sliding-window and Hybrid attention

Global attention can connect any historical position, but the training calculation increases approximately twice with the length of the sequence. Sliding window attention only focuses on the recent \(W\) token, and the complexity has been reduced from \(O(S^2)\) to about \(O(SW)\).

The actual model often staggers the local layer and the global layer: the local layer controls the cost, and the global layer is responsible for long-distance information dissemination. Mistral uses sliding windows; Gemma 2/3 shows the public route of the local and global layers.

## 7. FFN: SwiGLU and parameter body

attention is responsible for the information mixing between tokens, and FFN is responsible for the channel transformation within each token. SwiGLU is commonly used in modern models:

\[
\operatorname{SwiGLU}(x)=W_o\big(\operatorname{SiLU}(W_gx)\odot W_ux\big)
\]

FFN usually occupies most of the parameters of Transformer, so "Dense or MoE" is mainly the choice of the FFN sublayer.

### 7.1 Dense FFN

Each token goes through the same parameters. The advantage is that training is simple, communication is low, and the delay is stable; the disadvantage is that adding parameters usually increases the compute of each token synchronously.

### 7.2 Sparse MoE

router selects a small number of Experts for each token:

\[
y=\sum_{i\in\operatorname{TopK}(r(x))}g_i(x)E_i(x)
\]

MoE can make the total parameter volume much larger than the single token activation parameter volume, but it will introduce load balancing, capacity control, All-to-All communication and deployment GPU memory problems. Modern implementations also often add Shared expert, so that general knowledge always goes through the shared path.

## 8. Reorganization, residual and Block organization

### 8.1 Pre-Norm and Post-Norm

Pre-Norm is normalized in front of the sublayer:

\[
h'=h+F(\operatorname{Norm}(h))
\]

It is usually more conducive to deep network stability training. Post-Norm is normalized after adding residuals, and some new models re-adopt or combine the two through additional skills. There is no eternal and unique answer in the industry, but stability must be verified by large-scale experiments.

### 8.2 RMSNorm

RMSNorm only scales according to the root of the mean square, and does not reduce the average value:

\[
\operatorname{RMSNorm}(x)=g\odot\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}}
\]

It is simple to calculate and has been widely used in Decoder-only models. QK-Norm is additionally normalized on Query/Key to control the scale of attention logits.

## 9. Beyond Transformer: SSM and Hybrid Architecture

State Space Model can theoretically extend the length linearly and avoid complete KV Cache by processing sequences in a recursive state. Mamba is a representative route. Pure SSM has different chooses from attention in terms of content addressing and ecological maturity, so one of the more common engineering directions is attention–SSM mixing:

- SSM/linear recursive layer is responsible for most of the long sequence propagation;
- A small number of attention layers undertake precise retrieval and global interaction;
- Convolution or gate control modules supplement local modeling.

This route is important, but the default baseline of the general model is still Transformer or a hybrid structure with Transformer as the core.

## 10. Multimodal architecture: access and native integration

Multimodal models usually include three parts: modal encoder, connection module and language model trunk.

### 10.1 Modular access

The visual encoder first outputs visual features, and then connects to LLM through Projector or Cross-attention. This method is easy to reuse mature models and has low training costs.

### 10.2 Early integration and unification token flow

Text, images or videos indicate that it is earlier to enter the unified trunk for joint pre-training. The public introduction of Llama 4 calls it native multimodal early fusion. The advantage is that the cross-modal interaction is deeper, and the cost is higher data, training stability and system complexity.

Multimodal "unity" usually represents the unification of the trunk, which does not mean that the original pixels, audio waveforms and text characters use exactly the same front-end coding algorithms.

## 11. The reasoning architecture is also a part of the model architecture.

### 11.1 Prefill and Decode

- **Prefill**: Parallel processing of complete input, calculation-intensive.
- **Decode**: Generated by token, access storage and KV Cache bandwidth often become bottlenecks.

### 11.2 KV Cache and Paged attention

KV Cache avoids each Decode recalculation history K/V. Paged attention manages the cache of requests of different lengths by paging, reduces fragments, and supports continuous batch processing.

### 11.3 Quantitative, parallel and speculative decoding

- Weight/Activation/KV quantify to reduce GPU memory and bandwidth;
- Tensor, Pipeline, expert, Sequence Parallel solve the capacity limitation of a single device;
- Speculative decoding uses the small model to propose candidates and large models for parallel verification, so as to reduce the number of serial steps without changing the target distribution.

Therefore, the same amount of model parameters does not mean the same online cost; KV structure, context length, batch strategy and hardware kernel are also crucial.

## 12. Representative combination of public model architecture

The following table is used to understand the "design combination", not for simple ranking.

| Open model route | Representative structure | Main goal |
|---|---|---|
| Llama 3 | Decoder-only, Dense, GQA, RoPE, SwiGLU | Stable expansion and open ecology |
| Mistral 7B | Decoder-only, GQA, Sliding-window attention | Control long sequence inference cost |
| DeepSeek-V3 | Decoder-only, MLA, fine-grained MoE, Shared expert, MTP | Improve parameter capacity and training/reasoning efficiency |
| Gemma 3 | GQA, Local/Glocal attention Staggered, QK-Norm, Multimodal | Long Context and End-side/Multi-Size Deployment |
| Llama 4 | Native multi-modal, MoE, sharing and routing expert | Large capacity, low activation, multi-modal integration |
| Mamba Series | Selective SSM or attention–SSM Hybrid | Linear Sequence Modeling and Low Cache |

## 13. How to choose the architecture

The structure selection should be pushed back from the bottleneck:

| Goal or bottleneck | Priority | Verification required |
|---|---|---|
| single GPU, small model, low latency | Dense + GQA | Is the Dense baseline fully trained |
| Total parameter capacity is limited by calculation | Sparse MoE | Routing equalization, communication, GPU memory |
| Long context Decode display memory high | GQA/MQA/MLA, local attention | Long context validity instead of only testing and running |
| Ultra-long sequence training is too expensive | Local/sparse attention, mixed SSM | Long-distance retrieval and global reasoning |
| Multi-modal fast landing | Encoder + Projector + LLM | Modal alignment and input length |
| Native cross-modal ability | Early fusion joint pre-training | Data scale, training stability, cost |

Correct architecture decision-making must report quality, training throughput, peak memory, initial token delay, single token delay and deployment complexity at the same time.

## 14. miniLLM: shrink mainstream components into a teaching model

The current implementation of miniLLM is a compact Decoder-only Causal LM: 8 layers, hidden dimension 768, 8 Query heads and 4 KV heads; using RoPE, RMSNorm, GQA, SwiGLU, and supporting Dense and 4 expert/Top-1 Sparse MoE switching. Input and output embedding default shared weight, and the training target is Next-token Cross-Entropy.

Its correspondence with the mainstream architecture is very direct:

```text
Industry Decoder-only trunk -> MiniLLMForCausalLM
RMSNorm + Pre-Norm       -> MiniLLMRMSNorm / MiniLLMDecoderLayer
GQA + RoPE + SDPA        -> MiniLLMAttention
SwiGLU Dense FFN         -> MiniLLMMLP
Top-1 Sparse MoE         -> MiniLLMSparseMoE
```

The positioning of miniLLM is "readable, trainable and ablationable", not production-level micro-replication. It has not yet achieved KV Cache, Paged attention, expert Parallel, Shared expert and local/global hybrid attention; the configuration support of 32768 positions does not mean that it has passed long context training and evaluation. The most valuable way to use it is to compare the quality-cost changes brought about by GQA, MoE, sequence length and reasoning cache with Dense as the baseline.

## Reference materials

- [attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [Mistral 7B](https://arxiv.org/abs/2310.06825)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Gemma explained: What’s new in Gemma 3](https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/)
- [The Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

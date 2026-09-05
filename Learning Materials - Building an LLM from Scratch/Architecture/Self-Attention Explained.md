# Self-Attention Architectures: From QKV to GQA, MLA, and Efficient Inference

> This article revolves around the common principle and main evolution routes of attention in the industry; the implementation of miniLLM only appears as the smallest case at the end of the article.

## 1. attention, what exactly does it solve?

Self-attention allows each token to dynamically read information from the context according to the current task. The connection mode of convolution is predetermined by the location, and the connection weight of attention is calculated by the content.

For the position \(i\), it can be understood as a traceable search:

1. Query means "what do you want to find in the current location";
2. Key means "in how each historical position can be matched";
3. Value means "what information is actually retrieved after matching".

## 2. Scaled Dot-Product attention

Input the hidden state \(X\in\mathbb{R}^{B\times S\times d}\), and the linear projection will get:

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
\]

The output of a single head is:

\[
\operatorname{attention}(Q,K,V)= \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V
\]

Among them, \(d_h\) is the head dimension, and \(M\) is Mask.

### 2.1 Why divide by \(\sqrt{d_h}\)

If the dimensions of Q and K are approximately independent and have similar variances, the point product variance will increase with \(d_h\). Zooming can avoid excessive logits, excessive Softmax saturation and poor gradient.

### 2.2 What did Softmax do?

Softmax changes the score of each Query pair that allows Key to be non-negative and the sum of 1, and then weights the Value. attention weight can be used to assist diagnosis, but it cannot be directly equalized with a complete cause and effect explanation.

## 3. Causal Mask, Padding Mask and Document Mask

### 3.1 Causal Mask

Decoder-only cannot read the future token:

\[
M_{ij}=\begin{cases} 0,&j\le i\\ -\infty,&j>i \end{cases}
\]

This is the premise of self-regression training not to leak the answer.

### 3.2 Padding Mask

When samples of different lengths form batch, the Padding position cannot be used as a valid Key/Value. Loss Mask only does not calculate the loss of Padding tags and cannot replace attention Mask.

### 3.3 Boundary Mask of Packed Sequence

In order to improve the utilization rate of token, industrial training often spells multiple documents into the same sequence. At this time, Block-diagonal or document boundary perception Mask is required to avoid the latter document from unintentionally reading the previous irrelevant document; whether to allow cross-document attention needs to be consistent with the training formula.

## 4. Why is Multi-Head attention effective?

Split the hidden dimension into multiple heads:

\[
\operatorname{MHA}(X)=\operatorname{Concat}(head_1,\ldots,head_H)W_O
\]

Different headers can learn different relational subspaces, such as local collocation, reference, code structure or long-distance dependence. The more heads are not necessarily better, because the head dimension, redundancy, training scale and hardware shape together determine the effect.

## 5. MHA, MQA, GQA and MLA

### 5.1 MHA: COMPLETE K/V HEAD

Query, Key and Value all have \(H\) heads. The expression freedom is high, but each historical token in each layer needs to cache \(2Hd_h\) K/V elements.

### 5.2 MQA: Share a set of K/V

All Query headers share a Key header and a Value header. KV Cache has been greatly reduced, which is suitable for models that emphasize decoding efficiency, but the quality may be lost when the compression is too strong.

### 5.3 GQA: Group Sharing K/V

Keep \(H_q\) Query Headers, Use Only Fewer \(H_{kv}\) K/V Headers, And Each Query Group Shares A Set Of K/V:

\[
\text{group size}=H_q/H_{kv}
\]

GQA is a common compromise of modern Decoder-only, and public models such as Llama 2 70B, Llama 3, Mistral and new versions of Gemma have adopted this route.

### 5.4 MLA: Cache low-dimensional potential variables

MLA not only reduces the K/V header, but also compresses the K/V representation to a low-dimensional potential space, and then restores the required content for each header. The goal is to further reduce KV Cache while retaining multi-head expressions. DeepSeek-V2/V3 is a representative public implementation.

| Structure | KV representation | Cache size | Engineering characteristics |
|---|---|---:|---|
| MHA | Each Q head is independent K/V | Maximum | Simple and fully expressed |
| GQA | A group of Q heads shared K/V | Medium | Quality and efficiency balance |
| MQA | All Q Heads Shared K/V | Small | Decode Friendly, Compressed Radical |
| MLA | Cache low-dimensional potential representation | Very small | More complex structure and kernel |

## 6. How to enter attention?

### 6.1 Absolute Position embedding

Add the position vector to token embedding, which is intuitive but weak in extrapolation beyond the training length.

### 6.2 Relative Position Bias

Add Bias to the score according to the relative distance between Query and Key. T5, ALiBi and other routes belong to this category.

### 6.3 RoPE

RoPE applies position-related rotation to the pair dimensions of Q/K. The point product after rotation depends on the relative position, so it is suitable for Decoder-only and easy to combine with KV Cache.

Common means of RoPE length extension include frequency scaling, NTK-aware Scaling, YaRN and long context continuing training. No matter which method is adopted, it should be verified on Needle Retrieval, multi-jump retrieval, long article understanding and long output respectively.

## 7. Global, local, sparse and mixed attention

### 7.1 Full attention

Each position can pay attention to the entire history, and the training score matrix scale is \(O(S^2)\). It is semanticly flexible, but the cost of long sequences is high.

### 7.2 Sliding-window attention

Each position only focuses on the recent \(W\) token, and the complexity is about \(O(SW)\). It is suitable for locally dependent text, but a single layer cannot directly connect to far away positions.

### 7.3 Local/Glogal Layer Staggered

Multi-layer local attention will expand the perception field layer by layer, and a small number of global layers will establish long-distance paths. The public design of Mistral and Gemma 2/3 shows this kind of idea.

### 7.4 Block-sparse and structured sparse

Selecting connections by window, block, global token or task structure can reduce theoretical calculations, but only matching sparse kernels and hardware layouts can be transformed into real acceleration.

## 8. QK-Norm, Soft Capping and attention Stability

After the model scale and context increase, attention logits may be abnormally amplified. Common control methods include:

- Apply RMSNorm or LayerNorm on Query/Key;
- Do Soft Capping on logits;
- Adjust initialization, learning rate and numerical accuracy;
- Monitor the attention entropy and logits extreme value of each layer.

The public description of Gemma 3 shows the specific architecture selection from Soft Capping to QK-Norm. It shows that the stability mechanism is also a defusing design, not a fixed template.

## 9. FlashAttention: IO optimization of accurate attention

Simple implementation will explicitly objectify the \(S\times S\) fraction matrix, resulting in a large number of GPU memory reading and writing. FlashAttention reduces HBM round-trip through block, online Softmax and Kernel integration.

Key conclusion:

- It calculates accurate attention, not approximate sparse attention;
- The complexity of asymptotic calculation is still \(O(S^2)\);
- What is really reduced is the intermediate tensor objectification and IO;
- Whether to go to the fastest Kernel depends on dtype, head dimension, Mask shape and hardware.

PyTorch SDPA is a unified interface, and Flash, Memory-efficient or Math Backend can be selected at runtime. Using the interface does not mean that any input will automatically hit the fastest path.

## 10. KV Cache and Decode

When self-regression generates the \(t\) token, the K/V of the historical token will not change, so it can be cached and only the Q/K/V of the new token can be calculated.

The amount of single sample cache elements is approximately:

\[
N_{KV}=2\cdot L\cdot S\cdot H_{kv}\cdot d_h
\]

This explains why GQA, MQA, MLA and KV quantification are so important for long-context services.

### 10.1 Paged attention and Continuous Batching

Paging management KV Cache can reduce the GPU memory fragments caused by different length requests. Continuous Batching dynamically joins and moves out of the sequence during the request generation process to improve service throughput.

### 10.2 Prefill is different from the bottleneck of Decode

- Prefill processes a large number of tokens at the same time, which is usually more calculating-intensive;
- Decode only processes a small number of new tokens in each step, usually more weighted and KV memory bandwidth density.

Therefore, the benefits of the same attention structure may be different in the training, Prefill and Decode stages.

## 11. How to estimate the cost of attention

When ignoring constants:

- QKV/O projection calculation is about \(O(Sd^2)\);
- Full attention's QK and AV are about \(O(S^2d)\);
- The middle of attention score is about \(O(HS^2)\);
- Decode's KV Cache grows linearly with \(L\), \(S\), \(H_{kv}\), \(d_h\).

In short sequences and large hidden dimensions, the projection layer may dominate; in ultra-long sequences, the quadratic term will become more and more important. Just memorizing "attention is square complexity" is not enough to judge the real bottleneck.

## 12. How to verify the implementation of attention

A reliable implementation requires at least:

1. **Shape test**: Different batch, length, number of heads and KV head combinations are output correctly.
2. **Causal Leakage Test**: Changing The Future token Does Not Affect The Earlier Position Output.
3. **Padding invariance**: Adding Padding on the right does not change the effective position result.
4. **Simple formula comparison**: align with handwriting Softmax attention on the small tensen.
5. **GQA Equivalence Test**: The explicit replication of K/V is consistent with the implementation of grouping.
6. **RoPE test**: Location, broadcast, dtype and cache increment are consistent.
7. **Full/Incremental Consistency**: After enabling KV Cache, the gradual Decode is consistent with the whole segment forward.
8. **Performance test**: Test Prefill throughput, Decode delay, video memory and Kernel hit respectively.

## 13. The selection logic of mainstream attention

| Objectives | Common Choices | Main Risks |
|---|---|---|
| Training baseline and highest compatibility | MHA/GQA + Full attention | Long context cost |
| Reduce Decode KV Cache | GQA, MQA, MLA | Quality and Implementation Complexity |
| Reduce the cost of long-sequence training | Sliding-window, Hybrid, Sparse | Long-distance information dissemination |
| Improve accuracy attention kernel efficiency | FlashAttention/SDPA | Input conditions affect Kernel |
| Ultra-long context extrapolation | RoPE Scaling + continuing training | Nominal length is not equal to effective ability |
| End-to-end service throughput | KV Cache + Paging + Continuous Batching | Scheduling and video memory management complex |

## 14. attention mapping of miniLLM

miniLLM uses GQA with 8 Query heads and 4 KV heads, first applies RoPE to Q/K, and then explicitly extends K/V and calls PyTorch SDPA. Use `is_causal=True` when there is no Padding; combine causal Mask and Key Padding Mask when there is Padding.

This implementation covers the core teaching path of modern Decoder attention, but it is not a complete reasoning structure: at present, there are no KV Cache, Paged attention, local/global layer mixing and QK-Norm; each layer of Padding branches will also construct a causal matrix. The most valuable step is to first make up for causal leakage, Padding invariance and simple formula comparison test, and then implement KV Cache and verify full/incremental consistency.

## Reference materials

- [attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [RoFormer: Enhanced Transformer with Rotary Position embedding](https://arxiv.org/abs/2104.09864)
- [Mistral 7B](https://arxiv.org/abs/2310.06825)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Gemma explained: What’s new in Gemma 3](https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/)

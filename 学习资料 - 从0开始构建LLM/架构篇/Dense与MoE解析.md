# 大模型主流架构解析：Dense 与 Sparse MoE

> Dense 与 MoE 的分歧主要发生在 Transformer 的 FFN 子层。本文先解析业界通用机制、训练难点和部署经济性，最后再用 miniLLM 做简短映射。

## 1. 一句话抓住区别

- **Dense**：每个 Token 使用同一套 FFN 参数，总参数与每 Token 激活参数基本同步增长。
- **Sparse MoE**：准备多组 Expert，由 Router 为每个 Token 只激活其中少数几个，使总参数容量和单 Token 计算量部分解耦。

MoE 的核心价值不是“免费计算”，而是用路由、显存与通信复杂度换取更大的参数容量。

## 2. 两种架构共享什么

Dense 和 MoE 通常共享同一条 Decoder 主干：

```text
Embedding
  -> RMSNorm
  -> Causal Attention（MHA/GQA/MLA）
  -> Residual
  -> RMSNorm
  -> Dense FFN 或 Sparse MoE
  -> Residual
```

因此 RoPE、GQA、KV Cache、FlashAttention 与 MoE 并不冲突。MoE 解决的是 Token 内部通道变换的容量问题，Attention 解决的是 Token 之间的信息交互。

## 3. Dense FFN 如何工作

以 SwiGLU 为例：

\[
\operatorname{FFN}(x)=W_o\left(\operatorname{SiLU}(W_gx)\odot W_ux\right)
\]

输入维度为 \(d\)、中间维度为 \(d_{ff}\) 时，忽略 Bias 的参数量约为：

\[
P_{dense}\approx 3dd_{ff}
\]

Dense 的优势：

- 每个 Token 更新全部 FFN 参数，样本利用稳定；
- 计算图规则，适合高效 GEMM；
- 不需要 Token 路由和 All-to-All；
- 单卡、小 Batch、低并发时通常更容易达到高利用率；
- 调试、量化和部署路径成熟。

它的代价是：想增加知识容量时，通常也会增加每 Token FLOPs 和推理延迟。

## 4. Sparse MoE 的完整数据流

假设有 \(E\) 个 Expert，每个 Token 激活 \(K\) 个。

### 4.1 Router 打分

\[
r(x)=W_rx,\qquad p(x)=\operatorname{softmax}(r(x))
\]

Router 输出 Token 对每个 Expert 的偏好。

### 4.2 Top-K 选择

\[
S(x)=\operatorname{TopK}(p(x),K)
\]

只执行集合 \(S(x)\) 中的 Expert。Top-1 路径更省计算；Top-2 或更高 K 能融合多个 Expert，但增加计算与通信。

### 4.3 Dispatch、Expert 计算与 Combine

系统按 Expert 对 Token 重排，将 Token 发送到 Expert 所在设备，分别执行 FFN，再把结果送回原位置：

\[
y=\sum_{i\in S(x)}\tilde p_i(x)E_i(x)
\]

大规模 MoE 的瓶颈往往不在公式，而在 Token 排序、容量限制、All-to-All、负载不均和小矩阵效率。

## 5. 总参数、激活参数和 FLOPs 必须分开

如果每个 Expert 与 Dense FFN 等宽，则 MoE FFN 总参数约为：

\[
P_{total}\approx E\cdot 3dd_{ff}
\]

每 Token 激活参数约为：

\[
P_{active/token}\approx K\cdot 3dd_{ff}
\]

但实际成本还包括：

- 所有 Expert 权重仍需存储或分片；
- Router、Dispatch 和 Combine 的开销；
- 跨设备 All-to-All 通信；
- Padding 到 Expert Capacity 产生的无效计算；
- Expert Batch 太小导致硬件利用率下降。

所以“37B 激活参数”描述计算路径，不等于模型只需要存储 37B 参数。

## 6. 负载均衡：MoE 最核心的训练问题

若少数 Expert 被频繁选中，会出现：

- 热门 Expert 溢出、丢 Token 或排队；
- 冷门 Expert 缺少梯度，形成路由坍塌；
- 不同设备工作量不均，吞吐被最慢设备限制。

### 6.1 辅助负载均衡损失

经典做法同时约束 Expert 接收比例和平均路由概率。直观目标是让两者接近均匀分布：

\[
L=L_{LM}+\lambda L_{balance}
\]

\(\lambda\) 太小无法均衡，太大又会干扰语言建模目标。

### 6.2 Auxiliary-loss-free Balancing

另一条路线是不向主损失加入强均衡梯度，而是在路由分数上维护按 Expert 更新的偏置，使过载 Expert 的选择概率下降。DeepSeek-V3 的公开报告采用了这类方法。

### 6.3 Capacity 与 Token Dropping

每个 Expert 常设容量上限：

\[
C\approx \left\lceil\frac{N\cdot K}{E}\cdot \text{capacity factor}\right\rceil
\]

超过容量的 Token 可以被丢弃、转给备选 Expert，或通过无丢弃调度处理。训练吞吐、模型质量和实现复杂度需要联合权衡。

## 7. Expert 结构的主流演进

### 7.1 Routed Expert 与 Shared Expert

Shared Expert 对所有 Token 激活，承担通用知识；Routed Expert 学习更具条件性的模式。这样能减少所有通用能力都被迫重复存入每个 Expert 的问题。

### 7.2 细粒度 Expert

把少量大 Expert 拆成更多小 Expert，可让组合更灵活，也能保持相近激活宽度。但 Expert 越细，路由元数据和通信调度越复杂。

### 7.3 每层 MoE 与交错 MoE

不是每个 Block 都必须使用 MoE。Dense 层与 MoE 层交错可以降低通信与部署负担；Llama 4 的公开架构就是这种组合思路之一。

### 7.4 Expert Parallel

不同设备保存不同 Expert。Token 根据路由结果跨设备交换，这是扩展总参数的关键，也让网络拓扑成为模型架构的一部分。

## 8. MoE 为什么可能学到“专家化”，但不能想当然

Router 会根据隐藏状态选择 Expert，因此 Expert 可能对语言、代码、句法或抽象特征形成偏好。但真实专家化通常不是人类预先命名的清晰分工。

判断专家化应看证据：

- 不同数据域的 Expert 使用分布；
- Token 类型与路由的互信息；
- 屏蔽或替换某个 Expert 后的能力变化；
- 跨层路由稳定性；
- Expert 表示相似度和梯度相似度。

只看某几个示例 Token 被分到哪个 Expert，不能证明语义专家化。

## 9. Dense 与 MoE 的公平比较

至少要固定一种比较口径：

| 比较口径 | 固定项 | 适合回答的问题 |
|---|---|---|
| 等训练 FLOPs | Token 数、总训练计算 | 同预算下谁的质量更高 |
| 等激活参数 | 每 Token 计算规模 | 稀疏容量是否有效 |
| 等总参数 | 模型存储规模 | 相同权重容量谁更高效 |
| 等显存 | 设备与峰值显存 | 实际硬件能训练/部署什么 |
| 等延迟/吞吐 | 推理框架与并发 | 线上服务谁更划算 |

需要同时记录语言模型损失、下游能力、Token/s、MFU、峰值显存、All-to-All 时间、Expert 负载离散度和失败率。

## 10. 训练与部署的典型失败模式

### 10.1 Router 坍塌

症状是少数 Expert 长期过载。处理方向包括均衡机制、Router 初始化、路由温度、容量与数据混合。

### 10.2 Expert 学习不足

总 Token 数不变时，单个 Expert 看到的样本可能更少。小数据、小 Batch 场景尤其容易出现“参数更多但没训透”。

### 10.3 通信淹没计算

Expert 太小或单卡/单节点划分不合理时，All-to-All 开销可能超过节省的 FFN 计算。

### 10.4 线上低并发效率差

单请求产生的 Expert Batch 很小，稀疏 GEMM 难以吃满 GPU。高吞吐服务与端侧低延迟可能得出完全不同的架构选择。

### 10.5 量化与部署碎片化

不同 Expert 的访问频率不一致，权重调度和量化校准比 Dense 更难；总权重也可能跨越更多设备。

## 11. 业界公开路线怎么看

| 路线 | 代表 | 可观察重点 |
|---|---|---|
| 大规模 Dense | Llama 3 等 | 数据规模、训练稳定、部署生态 |
| 经典 Top-K MoE | Switch Transformer、GShard | 容量控制、均衡损失、Expert Parallel |
| Shared + Routed Expert | Mixtral、DeepSeekMoE、Llama 4 | 通用路径与条件容量组合 |
| 细粒度 MoE | DeepSeek-V2/V3 | 更多小 Expert、较少激活、通信协同设计 |
| 混合 Dense/MoE 层 | 多种新一代模型 | 用交错结构折中质量、通信和延迟 |

这些路线表明：Dense 和 MoE 会长期并存。Dense 是可靠基线，MoE 是规模足够大、系统能力足够强时的重要扩展手段。

## 12. 怎么选 Dense 或 MoE

优先 Dense 的典型条件：

- 模型与数据规模较小；
- 单卡或低带宽互联；
- 追求低并发、稳定延迟或端侧部署；
- 尚未建立可信的训练和评测基线；
- 团队不准备维护专门的 MoE 内核与监控。

考虑 MoE 的典型条件：

- 希望增加总参数容量但控制每 Token FLOPs；
- 有足够 Token 训练多个 Expert；
- 有高速互联和成熟的 Expert Parallel；
- 能持续监控路由、容量、通信和专家退化；
- 线上吞吐能摊薄稀疏调度成本。

## 13. miniLLM 的 Dense/MoE 映射

miniLLM 在 Attention 主干不变的前提下，可在 `MiniLLMMLP` 和 `MiniLLMSparseMoE` 之间切换。默认 MoE 配置是 4 个 Expert、每 Token Top-1，并使用辅助负载均衡损失；Router 以 float32 计算，只对有效 Token 路由。

需要特别注意：当 Top-1 选中概率被重新归一化时，唯一权重会变成 1，因此 Router 主要通过“选中了谁”和辅助损失学习，主任务损失不会再通过连续门控权重提供同样的梯度信号。这个性质很适合做 `norm_topk_prob` 消融。

miniLLM 没有 Expert Parallel、Shared Expert、Capacity/Token Dropping，也没有生产级稀疏内核。因此它适合回答“MoE 算法是否正确、在相同训练预算下是否优于 Dense”，不适合用单卡速度推断工业级 MoE 的服务效率。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [GShard](https://arxiv.org/abs/2006.16668)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664)
- [The Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)

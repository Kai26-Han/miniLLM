# LLM 预训练阶段主流算法解析：目标、数据、优化与规模化训练

> 预训练算法不只是一条 Cross-Entropy 公式，而是数据采样、序列构造、优化器、数值精度和分布式系统共同组成的闭环。本文以业界通用方法为主，最后简述 miniLLM。

## 1. 预训练究竟在做什么

预训练让模型从大规模未标注或弱标注数据中学习语言、知识、代码和世界模式。对当前主流 Decoder-only LLM，核心目标是 Causal Language Modeling（CLM）：

\[
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})
\]

训练最小化有效 Token 上的负对数似然：

\[
L_{LM}=-\frac{1}{N}\sum_{t\in\mathcal V}\log p_\theta(x_t\mid x_{<t})
\]

\(\mathcal V\) 是参与监督的位置集合，Padding、部分控制标记或跨文档边界可按配方排除。

预训练的直接产物是 Base Model：它擅长续写并具备大量潜在能力，但未必稳定遵循指令。SFT、偏好优化和推理 RL 属于后训练阶段。

## 2. 主流预训练目标

### 2.1 Causal Language Modeling

输入 `[x0, x1, x2, x3]` 时，通常用位置 0–2 的输出分别预测 `x1, x2, x3`。同一序列中大多数 Token 都能成为监督信号，训练和生成形式一致，是通用生成模型的主流目标。

### 2.2 Masked Language Modeling

随机遮盖部分 Token，并利用左右上下文恢复。BERT 类 Encoder 模型使用这一路线，适合理解和表示，但与逐 Token 生成存在训练—推理形式差异。

### 2.3 Denoising / Span Corruption

破坏输入片段，再由 Encoder–Decoder 恢复原文。T5 使用 Span Corruption，适合统一多种 Text-to-Text 任务。

### 2.4 Prefix LM 与混合 Mask

前缀部分可双向读取，生成部分保持因果。它在特定输入输出任务中有价值，但 Mask 和 KV Cache 语义更复杂。

### 2.5 Multi-Token Prediction（MTP）

除预测下一个 Token 外，再用辅助头预测更远的若干 Token。它增加监督密度，可能改善表示或为推测解码提供结构。DeepSeek-V3 的公开训练方案包含 MTP，但它是增强目标，不取代主 CLM。

## 3. 数据算法决定训练信号上限

### 3.1 数据处理主链路

```text
采集
 -> 文档解析
 -> 语言/格式识别
 -> 精确去重与近似去重
 -> 质量、安全、隐私过滤
 -> 数据源与许可记录
 -> 域混合和重采样
 -> Tokenize / Packing
 -> Train/Validation 污染检查
```

只增加原始字节量并不保证提升；重复、模板垃圾和低质量合成内容会浪费计算，并放大记忆与评测污染。

### 3.2 去重

- 精确哈希去除完全重复；
- MinHash/LSH 或 n-gram 方法发现近重复；
- URL、文档、段落和跨数据源要分层处理；
- 与公开 Benchmark 做重叠检测。

阈值太松保留污染，太严会误删通用短语和合法引用。去重需要记录命中原因并抽样审计。

### 3.3 质量过滤

常结合规则、分类器和模型打分：语言置信度、困惑度区间、广告/导航密度、乱码比例、代码可解析性、教育价值与安全策略。

高质量不等于风格单一。过强过滤会让模型丢失口语、方言、真实错误分布和长尾知识。

### 3.4 数据混合与温度采样

设域 \(i\) 的原始占比为 \(p_i\)，可用温度 \(\alpha\) 调整：

\[
q_i=\frac{p_i^\alpha}{\sum_jp_j^\alpha}
\]

\(\alpha<1\) 会提升小数据域占比。多语言、代码、数学和高质量知识域常需重采样，最终配比必须通过小规模实验验证。

### 3.5 合成数据

合成数据可补充推理轨迹、代码测试、长尾问答和结构化样本，但要控制：

- Teacher 偏差和错误自举；
- 同质化与多样性下降；
- 与评测集泄漏；
- 生成、过滤和验证的成本。

可验证领域优先使用编译器、单元测试和数学验证器，而不只依赖 LLM 打分。

## 4. 从文档到训练序列

### 4.1 Token Stream 与 Packing

逐样本 Padding 到固定长度会浪费大量 Token。主流训练更常把多个文档组织为连续 Token Stream 或 Packed Sequence，再切为固定长度块。

有效 Token 利用率：

\[
U=\frac{N_{valid}}{N_{allocated}}
\]

它应接近 1，但需要正确处理 EOS、Position ID、文档边界 Mask 和跨文档 Loss 语义。

### 4.2 序列长度课程

短序列训练吞吐更高，长序列训练培养长上下文能力。实际配方可先用较短序列完成主体预训练，再用长文数据进行 Context Extension 或 Mid-training。

### 4.3 独立验证集

验证集应与训练源隔离并按域分层。只从同一文件按行抽样容易产生近重复泄漏，也会让混合 PPL 掩盖某个域的退化。

## 5. 前向传播与语言模型损失

典型 Decoder Layer：

\[
h'=h+\operatorname{Attention}(\operatorname{Norm}(h))
\]

\[
h''=h'+\operatorname{FFN/MoE}(\operatorname{Norm}(h'))
\]

最后：

\[
z_t=W_{vocab}\operatorname{Norm}(h_t),\qquad
p_t=\operatorname{softmax}(z_t)
\]

实现 Cross-Entropy 时通常直接传 logits，由数值稳定的 `log_softmax + NLL` 内核完成，不应手工先算概率。

### 5.1 Perplexity

\[
\operatorname{PPL}=\exp(L_{LM})
\]

PPL 只在相同 Tokenizer、数据、Mask 和计权方式下可直接比较。跨 Tokenizer 更适合报告 Bits per Byte。

## 6. Batch、梯度累积与 Token 预算

数据并行下，全局 Batch 的序列数近似为：

\[
B_{global}=B_{micro}\times A\times W
\]

其中 \(A\) 为梯度累积步数，\(W\) 为数据并行进程数。更准确的训练预算应以有效 Token 计：

\[
T_{step}=\sum\text{valid tokens across workers and micro-batches}
\]

变长序列或 Packing 下，仅报告“Batch Size”可能掩盖实际 Token 规模。

梯度累积时应将 Loss 除以累积步数，或按全局有效 Token 做精确归一化；最后不足一个完整累积窗口时也要避免缩放错误。

## 7. AdamW：大模型训练的常用基线

Adam 维护一阶和二阶矩：

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

AdamW 将 Weight Decay 与自适应梯度更新解耦：

\[
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}-\eta\lambda\theta_t
\]

Norm 和 Bias 参数常不做 Weight Decay，但是否分组应通过配方和实验确认。Adafactor、Lion、Muon 等优化器也是研究与实践分支，AdamW 仍是最常见的稳健基线之一。

## 8. Warmup、主学习率与衰减

训练早期二阶矩估计不稳定，直接使用峰值学习率容易产生 Loss Spike，因此先线性 Warmup：

\[
\eta_t=\eta_{max}\frac{t}{T_w},\quad t<T_w
\]

之后常使用 Cosine Decay：

\[
\eta_t=\eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min})
\left[1+\cos\left(\pi\frac{t-T_w}{T-T_w}\right)\right]
\]

也有模型采用 WSD（Warmup–Stable–Decay）：长时间维持稳定学习率，在训练末段再衰减，便于灵活扩展 Token 预算。

## 9. 初始化、归一化与稳定性

规模化训练需要协同控制：

- 参数初始化尺度和残差分支缩放；
- Pre-Norm/RMSNorm 或混合 Norm 结构；
- 学习率、Warmup 和 Batch Token 数；
- 梯度裁剪；
- Attention logits 与 QK-Norm；
- MoE Router 的数值精度与负载均衡；
- 数据异常、超长重复和坏 Batch 检测。

Loss Spike 不应只靠回滚掩盖。要保存数据游标、优化器状态、随机数状态，并定位是数据、数值、通信还是硬件故障。

## 10. 混合精度

### 10.1 FP16 与 BF16

- FP16 尾数更精细，但指数范围较小，常需 Dynamic Loss Scaling；
- BF16 指数范围接近 FP32，训练更稳，现代加速器上广泛使用；
- Master Weight、优化器状态和部分归一化/归约仍可能保留 FP32。

### 10.2 FP8

FP8 可显著提高吞吐并降低内存带宽，但需要按 Tensor/Block Scaling、精度敏感算子白名单和异常监控。DeepSeek-V3、Llama 4 等公开资料展示了大规模 FP8 训练实践。

低精度的目标是保持端到端收敛质量，不是让所有算子强制使用同一种 dtype。

## 11. 显存优化与并行算法

### 11.1 Activation Checkpointing

只保存部分激活，反向传播时重算其余中间值，用额外计算换显存。

### 11.2 Data Parallel / DDP

每个进程保存完整模型，处理不同数据，再 All-Reduce 梯度。它扩展吞吐，但不降低单卡模型权重和优化器状态。

### 11.3 FSDP / ZeRO

跨设备切分参数、梯度和优化器状态，使更大模型能进入集群。

### 11.4 Tensor、Pipeline、Context 与 Expert Parallel

- Tensor Parallel：切分层内矩阵；
- Pipeline Parallel：切分层；
- Context/Sequence Parallel：切分长序列；
- Expert Parallel：MoE Expert 分布到不同设备。

工业训练通常是多维并行组合。最优组合取决于模型形状、序列长度、网络拓扑和故障恢复要求。

## 12. Scaling Law 与计算预算

Dense Decoder 的训练 FLOPs 常用粗略式：

\[
C\approx 6ND
\]

\(N\) 为参与计算的参数规模，\(D\) 为训练 Token 数。它适合数量级估算，不包含 Attention 长度效应、MoE 路由、重算和系统低效。

Kaplan 等工作研究了模型、数据和计算的幂律关系；Chinchilla 强调在固定计算预算下，许多早期模型参数过大、训练 Token 不足。后续公开模型常选择更多 Token，以兼顾性能、部署摊销和数据可用性。

真正的计算最优是目标相关的：一次训练成本、长期推理成本、目标能力和数据质量可能导向不同配置。

## 13. 数据、模型和系统的联合优化

主流预训练不再把算法和系统分开：

- GQA/MLA 因 KV Cache 和推理带宽而设计；
- MoE 因参数容量、All-to-All 和 Expert Parallel 而设计；
- Packing 与 FlashAttention 同时提高 Token 与 IO 利用率；
- FP8 需要模型尺度、Kernel 和网络归约共同适配；
- 长上下文需要数据课程、位置方法和 Context Parallel 配合。

只比较理论 FLOPs 会遗漏通信、重算、Padding、故障和低利用率。

## 14. 训练监控与验收

### 14.1 学习指标

- Token-weighted Train/Validation Loss；
- 分域 PPL 或 Bits per Byte；
- 学习率、梯度范数、参数范数；
- 数据域占比与有效 Token/s；
- 固定能力集的阶段性结果。

### 14.2 系统指标

- MFU、设备吞吐和算子时间；
- 峰值显存、通信占比、重算占比；
- DataLoader 等待时间；
- NaN/Inf、Loss Spike、硬件错误和恢复耗时。

### 14.3 MoE 专项指标

- 每层 Expert Token 数和负载变异系数；
- Router 概率、熵与辅助损失；
- Capacity 溢出/Token Dropping；
- All-to-All 时间和 Expert GEMM 利用率。

## 15. 一个可靠的预训练实验顺序

1. 用小数据做过拟合测试，证明 Loss、Mask 和 Shift 正确；
2. 固定数据、Tokenizer 和 Eval，建立 Dense 小模型基线；
3. 做学习率、Batch Token 和初始化扫描；
4. 验证 Packing、混合精度和 Checkpoint 恢复；
5. 再引入 GQA、MoE、MTP 或长上下文等变量；
6. 用小模型 Scaling 实验预测大训练；
7. 冻结正式配方并进行阶段性评测；
8. 保留可复现实验清单、数据版本和失败记录。

每次实验只回答一个主要问题，比同时改十个超参数更有解释力。

## 16. miniLLM 的预训练映射

miniLLM 使用 Decoder-only CLM、Token-shifted Cross-Entropy、AdamW、线性 Warmup + Cosine Decay、梯度累积、梯度裁剪和 CUDA 混合精度；支持 DDP、Activation Checkpointing、Dense/MoE 切换、定期验证和断点恢复。默认模型为 8 层、隐藏维度 768、GQA，并可启用 4 Expert/Top-1 MoE。

它与工业配方的主要差距集中在数据与系统：当前每条样本独立截断/Padding 到 512，尚无 Packing；验证集按同一数据源的固定行号划分；DDP 不切分模型状态；MoE 没有 Expert Parallel 和容量控制。因而最优先的算法升级应是独立分域验证集与 Token Packing，其次才是扩大上下文或增加专家数量。

## 参考资料

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Megatron-LM](https://arxiv.org/abs/1909.08053)
- [ZeRO](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [FlashAttention](https://arxiv.org/abs/2205.14135)

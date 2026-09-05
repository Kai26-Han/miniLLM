# 大模型 Eval 主流方法解析：从离线基准到安全门槛与线上闭环

> Eval 的目标不是制造一个漂亮总分，而是为训练、架构、发布和运营提供可重复的决策证据。本文以业界通用评估体系与公开公司实践为主，最后简述 miniLLM。

## 1. Eval 是一个决策系统

完整 Eval 回答五类问题：

1. **能力**：模型能完成哪些任务，在哪些难度开始失败？
2. **行为**：是否遵循指令、表达不确定性并保持预期风格？
3. **安全**：哪些请求会造成有害输出、越狱、误拒答或高风险能力放大？
4. **效率**：达到该质量需要多少延迟、吞吐、显存与费用？
5. **可靠性**：结果在不同 Prompt、采样、语言、用户群和环境下是否稳定？

如果一个指标不会改变任何训练或发布决策，它可能只是展示数据，而不是有效 Eval。

## 2. 先定义评测对象

“模型分数”至少可能对应四个不同对象：

| 对象 | 包含内容 | 适合回答 |
|---|---|---|
| Base Model | 权重 + Tokenizer | 预训练知识与续写能力 |
| Chat Model | 权重 + Chat Template + System Prompt | 对话行为与指令遵循 |
| Inference Policy | 采样、思考预算、工具、搜索 | 给定计算预算下的能力 |
| Product System | 模型 + RAG + Guardrail + UI + 监控 | 用户实际体验和风险 |

模型版本相同，但 Prompt、工具、上下文、采样次数或思考预算不同，结果就不再是同一系统。

## 3. 五条核心原则

### 3.1 先定义决策，再设计指标

“是否发布”“是否替换旧模型”“是否扩大训练”需要不同证据。先写清通过阈值和失败动作，再看结果，避免事后挑指标。

### 3.2 冻结协议

至少冻结：模型与 Tokenizer、Chat Template、System Prompt、Few-shot 示例、采样参数、最大输出、工具版本、Judge 和数据集版本。

### 3.3 分层而非只看总分

一个平均数会掩盖语言、难度、安全域和高价值用户的回退。必须报告分域、分难度、分人群和最坏分位数。

### 3.4 报告不确定性

对准确率可用 Bootstrap 置信区间；模型对比宜用配对检验。0.3 分提升若落在噪声范围内，不应形成强结论。

### 3.5 质量、成本和安全联合报告

推理模型可用更多采样或更长思考换取分数。公平比较需要同时报告输出 Token、采样次数、延迟和费用。

## 4. 全生命周期 Eval

### 4.1 Tokenizer 阶段

- Round-trip、特殊 Token 和 Chat Template 正确性；
- 分语言/代码/数学的 `bytes/token`；
- 长尾字符与极端长度；
- 不同词表的下游损失和吞吐。

### 4.2 预训练阶段

- Token-weighted Validation Loss；
- 分域 PPL/Bits per Byte；
- 数据污染和记忆测试；
- 固定 Base 能力集；
- 训练稳定性、吞吐和 MFU。

### 4.3 SFT 与偏好对齐

- 指令遵循、格式和多轮一致性；
- Pairwise Preference；
- 过度拒答、迎合和风格退化；
- 基础知识与推理能力回归。

### 4.4 推理 RL

- Pass@1 与 Pass@k；
- 解题 Token、思考预算和答案校准；
- Verifier 通过率与 Reward Hacking；
- 难度分层和跨域泛化。

### 4.5 发布与线上

- 安全红队和滥用压力测试；
- TTFT、TPOT、吞吐、超时和成本；
- A/B 任务完成率、用户修正率与升级人工率；
- 线上漂移、故障聚类和持续回归。

## 5. 六类评分器

### 5.1 确定性规则

Exact Match、正则、Schema、格式和关键词。便宜可复现，但语义覆盖有限。

### 5.2 参考答案指标

F1、ROUGE、BLEU、BERTScore 等适合有参考输出的任务。开放生成中，多种正确表达会让单参考指标低估质量。

### 5.3 可执行验证器

代码单元测试、编译器、SQL 执行、数学等价检查和环境状态是高价值信号。它们更接近任务是否真正完成，但需防测试不全和环境泄漏。

### 5.4 LLM-as-a-Judge

Judge 能按 Rubric 评估开放回答、成对偏好和多轮轨迹，扩展性强。主要偏差包括：

- Position Bias；
- Verbosity Bias；
- Self-preference；
- 风格或格式偏好；
- Judge 与被测模型同源；
- Prompt Injection；
- 评分漂移。

缓解方法包括交换回答顺序、隐藏模型身份、明确 Rubric、要求结构化理由、多 Judge 仲裁，并用人工标注校准一致率。

### 5.5 人工评估

人评适合产品体验、文化语境、复杂安全和 Judge 校准。必须提供 Rubric、示例、盲测、重复样本和标注者一致性。

### 5.6 红队与对抗评估

目标是主动寻找失败边界，而非估计普通请求平均表现。自动攻击适合覆盖面，专家红队适合高影响链路，两者应与真实滥用监控结合。

## 6. 预训练指标

### 6.1 Token-weighted Cross-Entropy

\[
L=\frac{\sum_b N_bL_b}{\sum_bN_b}
\]

不能简单平均 Batch Loss，因为每个 Batch 的有效 Token 数可能不同。

### 6.2 Perplexity

\[
\operatorname{PPL}=e^L
\]

只适合同 Tokenizer、同数据、同 Mask 协议比较。

### 6.3 Bits per Byte

\[
\operatorname{BPB}=\frac{L\cdot N_{token}}{N_{byte}\ln2}
\]

它对跨 Tokenizer 比较更公平，但仍需相同原始文本和预处理。

### 6.4 分域 Loss

Web、书籍、中文、英文、代码、数学等分开报告，能发现数据配方或 Tokenizer 导致的局部退化。

## 7. 能力评估地图

### 7.1 知识与事实性

闭卷问答、时效知识、长尾知识、引用可验证性和幻觉率要分开。RAG 系统还需区分检索失败、证据不足和生成不忠实。

### 7.2 数学与推理

关注答案正确率、过程可验证性、难度曲线和采样预算。GSM8K、MATH、AIME 类任务各自覆盖不同难度，单一集合易饱和或污染。

### 7.3 代码

HumanEval/MBPP 测函数生成，SWE-bench 类任务测真实仓库问题。要固定容器、测试、超时、工具权限和重试次数。

### 7.4 多语言与中文

翻译后的英文题不能充分代表本土知识、表达和文化语境。应有原生题、跨语言一致性、低资源语言和代码混合输入。

### 7.5 指令与格式

IFEval、JSON Schema、函数参数约束等可执行验证优于主观打分。还要测试冲突指令、否定约束和长指令列表。

### 7.6 多轮对话

评估状态保持、纠错、指代、长期约束、遗忘与上下文污染。不能把独立单轮题简单串接后声称是多轮 Eval。

### 7.7 长上下文

Needle-in-a-Haystack 只测浅层检索，还需多 Needle、顺序、聚合、长文问答和真实文档任务；分别报告位置、长度和干扰强度。

### 7.8 Agent 与工具

以完整轨迹为单位：任务成功率、工具选择、参数正确率、无效调用、恢复能力、步数、成本和副作用。

## 8. 偏好与开放生成

成对比较通常比 1–10 绝对打分稳定。若 A 对 B 的胜率为 \(p\)，可用 Bradley–Terry 或 Elo 聚合，但要报告：

- 平局如何处理；
- 顺序是否随机交换；
- Prompt 分布和时间窗口；
- Judge/用户群体构成；
- 长度和风格是否形成混淆。

Arena 类评测反映特定流量与偏好，不等价于所有专业场景的能力排名。

## 9. 安全评估

### 9.1 三个概念要分开

- **有害内容遵从**：不应完成的请求是否被完成；
- **误拒答**：正常请求是否被错误拒绝；
- **高风险能力**：模型是否显著降低实施严重伤害的门槛。

只降低有害遵从率可能通过“全部拒绝”作弊，因此必须联合报告 False Refusal Rate。

### 9.2 主要安全域

网络安全、CBRN、自主行动、欺骗/操纵、隐私、未成年人、自伤、仇恨和违法活动等域需要不同专家、Threat Model 和评分标准。

### 9.3 Uplift 与系统防线

高风险评估要比较模型相对于搜索、教材或专家基线增加了多少能力，而不是只问模型能否答题。还要分别测模型能力、部署 Guardrail、访问控制和监控后的剩余风险。

## 10. 效率与部署评估

- **TTFT**：首 Token 延迟；
- **TPOT/ITL**：相邻输出 Token 延迟；
- **Throughput**：每秒请求或 Token；
- **Goodput**：满足 SLO 的有效吞吐；
- **Peak Memory**：权重、KV Cache、激活和工作区；
- **Cost**：每百万输入/输出 Token 或每任务完成成本。

性能测试要固定硬件、精度、量化、输入/输出长度、并发、Batch 策略和缓存状态。平均延迟不能替代 P95/P99。

## 11. Benchmark 最常见的失真

1. 训练数据污染或答案记忆；
2. 题目饱和，无法区分新模型；
3. Prompt/Chat Template 为某模型特殊优化；
4. 不同模型使用不同采样或思考预算；
5. 只报告最好的一次运行；
6. Judge 偏爱长度、位置或自身风格；
7. 用同一题反复调参造成 Eval 过拟合；
8. 只看均值，不看高风险失败；
9. 比较不同截止日期的知识而不做说明；
10. 把模型能力、脚手架和工具能力混为一谈。

Goodhart 定律在这里非常直接：一旦单一 Benchmark 成为目标，它就更容易失去作为测量的价值。

## 12. 主流公司的公开 Eval 实践

### 12.1 OpenAI

公开 System Card 同时报告能力、安全、红队和 Preparedness 评估；Preparedness Framework 将高风险能力阈值与部署缓解、评审和发布决策连接。SWE-bench Verified 也体现了对基准不可解题和标注质量的再审计。

### 12.2 Google DeepMind

Frontier Safety Framework 使用 Critical Capability Level、早期预警 Eval 和对应缓解计划；模型卡覆盖能力、责任和限制。公开工作强调 Holistic Evaluation 与外部评估。

### 12.3 Anthropic

公开路线包括模型生成评测集、专家红队、不同训练快照/安全状态对比和 Responsible Scaling Policy。其重点之一是用 Eval 发现训练期间逐步出现的能力，而非只测最终模型。

### 12.4 Meta

Llama 模型卡结合标准 Benchmark、人工偏好、安全测试和开放工具；CyberSec Eval、Purple Llama、Llama Guard 等项目让部分安全评测和防护可复用。

### 12.5 DeepSeek、Qwen 与 Mistral

这些公开模型报告通常覆盖知识、数学、代码、多语言和开放对话；推理模型更强调多次采样、Pass@k 和生成长度，工具模型则加入 Function Calling/Agent 评测。阅读时必须核对每个表格的 Prompt、思考模式和采样预算。

## 13. 业界共同趋势

1. 从静态 Benchmark 转向持续、私有和动态生成 Eval；
2. 从单模型转向模型 + 工具 + 检索 + Guardrail 的系统评估；
3. 从单次准确率转向质量—计算曲线；
4. 从纯自动或纯人工转向规则、Verifier、Judge 和人评混合；
5. 从最终分数转向能力阈值、风险等级和发布门槛；
6. 从平均用户转向分语言、人群、难度和最坏情况；
7. 从离线一次验收转向线上反馈驱动的回归集。

## 14. 一套可落地的 Eval 架构

```text
版本化数据集与任务定义
  -> Prompt / Tool / Sampling 配置
  -> 可复现执行 Harness
  -> Rule / Verifier / Judge / Human Scoring
  -> 分层指标 + 置信区间 + 成本
  -> 失败聚类与人工审计
  -> 发布门槛 / 训练数据回流
  -> 持续回归
```

每条样本至少保存：样本 ID、数据版本、Prompt、原始输出、结构化分数、评分器版本、Token/延迟/成本和错误标签。只保存聚合分数会让失败分析无法进行。

## 15. miniLLM 的 Eval 映射

miniLLM 已能计算 Token-weighted Validation Loss/PPL，并用固定 Prompt 做文本生成；Tokenizer Eval 覆盖特殊 ID、结构 Token、Round-trip 和分域样例。这足以做训练 Smoke Test，但还不是完整能力评估。

下一步应保持轻量：先把验证数据从训练源中独立出来，并按中文、英文、代码等分域报告；再建立几十到几百条可执行的小型 Base/SFT 回归集；最后补 TTFT、TPOT、吞吐，以及 Dense/MoE 路由指标。这样既符合小项目规模，又能让每次架构和算法改动都有可信证据。

## 参考资料

- [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110)
- [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132)
- [OpenAI Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/)
- [Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)
- [Google DeepMind Frontier Safety Framework](https://deepmind.google/frontier-safety/)
- [Anthropic Responsible Scaling Policy](https://www.anthropic.com/responsible-scaling-policy)
- [Meta Purple Llama](https://ai.meta.com/llama/purple-llama/)
- [SWE-bench](https://arxiv.org/abs/2310.06770)

# LLM SFT 阶段主流算法解析：从指令数据到可用对话模型

> Supervised Fine-Tuning（SFT，监督微调）是现代大模型后训练的起点。它用高质量示范把“会续写的 Base Model”变成“能按协议完成任务的 Assistant”。本文以业界主流算法、数据和工程取舍为核心，最后简述 miniLLM 的实现。

## 1. SFT 在大模型生命周期中的位置

一条典型训练链路是：

```text
Tokenizer
  -> Pretraining
  -> Continued Pretraining / Mid-training（可选）
  -> SFT
  -> Preference Optimization（DPO / RLHF 等）
  -> Reasoning RL / RLVR（可选）
  -> Safety、Eval 与部署
```

各阶段解决的问题不同：

| 阶段 | 主要数据 | 主要目标 |
|---|---|---|
| 预训练 | 大规模自然文本、代码、多模态数据 | 学习知识、语言和通用表示 |
| 持续预训练 | 领域原始语料、长上下文语料 | 补充知识、领域和上下文分布 |
| SFT | 指令—回答、多轮对话、工具轨迹 | 学习任务协议和目标行为 |
| 偏好优化 | Chosen/Rejected 或奖励 | 在多个可行回答中偏向更优行为 |
| 推理 RL | 可验证任务与环境反馈 | 扩展搜索、推理和 Agent 能力 |

SFT 主要负责“示范应该怎样回答”。它能激活和组织预训练能力，也能教会模型新格式和部分技能，但不适合用少量示范替代缺失的基础知识。

## 2. SFT 的条件语言建模目标

设上下文为 \(c\)，目标回答为 \(y=(y_1,\ldots,y_T)\)。SFT 最大化：

\[
p_\theta(y\mid c)=\prod_{t=1}^{T}p_\theta(y_t\mid c,y_{<t})
\]

对应的 Token-level Cross-Entropy 为：

\[
L_{SFT}=-\frac{1}{T}\sum_{t=1}^{T}
\log p_\theta(y_t\mid c,y_{<t})
\]

它与预训练使用相同的 Next-token Prediction 内核，核心差异在于数据分布、序列模板和哪些位置参与 Loss。

## 3. Loss Mask：哪些 Token 应该监督

一段对话序列可以写成：

```text
<system> ...
<user> ...
<assistant> ...
<user> ...
<assistant> ...
```

令 \(s=(s_1,\ldots,s_n)\) 是序列，\(m_t\in\{0,1\}\) 表示位置是否参与 Loss：

\[
L=-\frac{\sum_{t=1}^{n}m_t\log p_\theta(s_t\mid s_{<t})}
{\sum_{t=1}^{n}m_t}
\]

### 3.1 Full-sequence Loss

System、User、Assistant 等所有非 Padding Token 都参与预测。

优点是监督密度高，也能学习完整对话格式；缺点是模型花费容量模仿用户输入，并可能对固定 System Prompt 或模板过拟合。

### 3.2 Response-only / Assistant-only Loss

System、User、Tool Result 只作为上下文，Assistant 内容、工具调用和结束标记参与 Loss。这是对话 SFT 的常见选择，因为训练目标与部署时需要生成的区域一致。

### 3.3 所有 Assistant 轮次还是最后一轮

多轮对话可以监督所有 Assistant 消息，也可以只监督最后一轮：

- 监督所有 Assistant 轮次，Token 利用率更高；
- 只监督最后一轮，便于构造“完整历史 -> 当前回答”的样本；
- 若历史 Assistant 回答质量不稳定，把它们全部作为目标会传播错误。

没有脱离数据来源的唯一答案，关键是显式记录 Mask 策略并与 Eval 保持一致。

## 4. Chat Template 是训练协议的一部分

原始 `messages` 必须序列化为模型看到的 Token 流：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
解释什么是梯度下降。<|im_end|>
<|im_start|>assistant
梯度下降是一种优化方法……<|im_end|>
```

模板至少定义：

- Role 名称和消息边界；
- BOS/EOS 与 Assistant 结束标记；
- System Prompt 的放置；
- 多轮换行规则；
- Reasoning、Tool Call 和 Tool Result 的结构；
- 推理时是否加入 Generation Prompt；
- 哪些 Token 参与训练 Loss。

训练与推理模板不一致会造成模型看到陌生前缀、无法停止、角色串线或工具 JSON 失效。模板文件、Tokenizer 和模型权重应作为同一版本发布。

## 5. SFT 数据从哪里来

### 5.1 人工示范

专家或标注员直接编写目标回答，适合安全政策、专业领域和复杂风格。优势是意图清晰，代价是昂贵、速度慢且存在标注者偏好。

InstructGPT 的公开流程先收集标注员示范训练 SFT 模型，再用偏好比较进入 Reward Model 和 PPO 阶段。这奠定了“SFT 是后续偏好优化起点”的经典范式。

### 5.2 传统任务转成指令

把分类、抽取、翻译、摘要、问答等数据集改写成自然语言指令。FLAN 系列显示，扩大任务数、模板数、模型规模并加入 Chain-of-Thought 数据，可以提升对未见任务的泛化。

风险是模板化过强：同一任务的大量近重复指令不等于真实任务多样性。

### 5.3 人机对话与产品数据

真实请求分布能覆盖用户真正关心的问题，但必须处理：

- 隐私和个人信息；
- 用户授权与数据许可；
- 低质量、攻击性和不完整对话；
- 旧模型输出造成的自我模仿；
- 线上分布对少数群体和高频场景的偏置。

### 5.4 Teacher Distillation

用更强模型生成回答、解释或工具轨迹，再训练较小模型模仿。它可以扩大数据规模，但 Student 的上限会受 Teacher 错误、风格单一和知识截止影响。

### 5.5 Self-Instruct 与迭代合成

典型流程是：

1. 从少量 Seed Task 出发；
2. 让模型生成新指令；
3. 生成对应输入与回答；
4. 去除无效、相似和高风险样本；
5. 用保留数据训练模型；
6. 迭代扩展困难度和任务覆盖。

合成数据真正的价值来自“生成 + 过滤 + 验证”，而不是把所有生成结果直接加入训练。

## 6. 数据质量通常比无差别扩量更重要

LIMA 使用少量精心筛选的数据展示了高质量示范的强作用；这不意味着所有模型只需极少数据，而是说明重复、矛盾和低质量回答可能比数据不足更有害。

### 6.1 硬性校验

- Role 序列合法，至少存在一个 Assistant 目标；
- 特殊标记、JSON、代码和工具参数可解析；
- 回答没有被错误拼接或截断；
- 编码后仍有有效监督 Token；
- 长度、语言和数据来源字段完整。

### 6.2 去重与污染控制

应在 Prompt、Response 和完整对话多个层级做精确/近似去重，并检查与能力 Benchmark、代码测试集和发布 Eval 的重叠。

### 6.3 质量评分

可以组合：

- 规则：乱码、模板残留、重复、异常长度；
- Verifier：代码测试、数学答案、JSON Schema、引用核验；
- LLM Judge：相关性、完整性、风格和安全；
- 人工审计：复杂语境、边界政策和 Judge 校准。

### 6.4 多样性

高质量不等于所有回答都冗长、分点且风格相同。需要覆盖：短答与长答、正式与口语、不同语言、拒答与正常回答、简单任务与复杂任务。

## 7. 数据混合与采样算法

SFT 通常混合通用对话、知识、数学、代码、多语言、长上下文、工具和安全数据。若域 \(i\) 有 \(n_i\) 条样本，可用温度采样：

\[
q_i=\frac{n_i^\alpha}{\sum_j n_j^\alpha}
\]

当 \(\alpha<1\) 时，小数据域会被提升。还需要决定按“样本数”还是“目标 Token 数”平衡，因为一条长推理回答可能抵得上数十条短回答的梯度量。

### 7.1 Token-level 平均的长度偏置

标准 Token 平均会让长回答贡献更多梯度。若希望每条样本权重接近，可先计算每个样本的平均 Loss，再在 Batch 内平均：

\[
L_{example}=\frac{1}{B}\sum_{i=1}^{B}
\frac{\sum_tm_{it}\ell_{it}}{\sum_tm_{it}}
\]

两种计权方式优化的目标不同，应根据产品分布选择并在实验中说明。

### 7.2 Curriculum 与阶段配方

常见思路包括：

- 先建立通用指令遵循，再加入高难推理或工具数据；
- 先短上下文，再进行长上下文 SFT；
- 先高置信人工数据，再加入经过筛选的合成数据；
- 训练末段提高目标领域权重，但保留通用数据防止遗忘。

顺序是否优于一次混合需通过消融验证，不能只凭直觉。

## 8. 多轮对话、截断与 Packing

### 8.1 多轮样本

多轮 SFT 应覆盖指代、纠错、状态保持、约束延续和 Tool Result 回填。把彼此无关的单轮样本机械拼接，不能产生真实多轮能力。

### 8.2 截断策略

长对话超出上限时，常见策略是：

- 保留 System Prompt；
- 删除最早的完整轮次；
- 保留最近 User 请求和目标 Assistant 回答；
- 避免在被截断回答后伪造 EOS；
- 记录 Prompt/Answer 截断率。

只从右侧硬截断可能删除目标答案；只从左侧硬截断可能破坏 System Prompt 和工具定义。

### 8.3 Dynamic Padding

将 Batch Padding 到本批最长序列，而不是全局最大长度。若最长长度为 \(S_{max}\)，有效率为：

\[
U=\frac{\sum_iS_i}{B\cdot S_{max}}
\]

长度分桶能进一步提高 \(U\)。

### 8.4 SFT Packing

多个短对话可 Pack 到同一序列。需要保证：

- 样本间有清晰 EOS；
- Position ID 与 Attention Mask 符合设计；
- 不产生无意跨样本注意；
- Loss Mask 不跨边界错位；
- 统计仍能还原每域和每样本贡献。

Packing 提高吞吐，但其 Mask 复杂度高于普通动态 Padding。

## 9. Reasoning SFT

Reasoning SFT 使用带推理过程的示范，例如数学推导、代码规划或工具决策。它能教会模型展开中间步骤，但“文本更长”不等于“推理更强”。

### 9.1 数据构造

高质量 Reasoning 数据通常来自：

- 人类专家推导；
- 强 Teacher 生成多个候选；
- 用答案、单元测试或形式化验证器筛选；
- 对错误轨迹进行修复或丢弃；
- 从易到难组织任务。

### 9.2 长思考与短回答混合

如果所有样本都使用长 Chain-of-Thought，模型可能在简单问题上过度推理、延迟升高并暴露不可靠过程。业界逐渐采用混合思考配方：复杂任务提供推理轨迹，简单任务直接回答，并通过特殊标记或控制条件区分模式。

Qwen3 的公开路线包含长 CoT 冷启动、推理 RL、Thinking/Non-thinking 融合和通用 RL，说明 SFT/冷启动只负责建立初始推理格式，能力扩展还依赖后续可验证训练。

### 9.3 不要用训练 Loss 判断推理质量

模型可以很好地模仿错误推理文本而获得低 Loss。必须用最终答案、执行结果、步骤一致性和跨难度泛化评估。

## 10. Tool Calling 与 Agent 轨迹 SFT

一个工具回合通常是：

```text
System + Tool Definitions
  -> User Request
  -> Assistant Tool Call
  -> Tool Result
  -> Assistant Final Answer
```

训练中常把 Tool Definition、User 和 Tool Result 作为上下文，把 Assistant 的工具选择、参数 JSON 和最终回答作为目标。

数据要覆盖：

- 正确选择工具；
- 参数类型、必填项和 Schema；
- 不需要工具时直接回答；
- 工具报错、空结果和超时后的恢复；
- 多工具顺序与停止条件；
- 防止把 Tool Result 中的恶意文本当作高优先级指令。

只评估 JSON 是否可解析还不够，必须执行工具并判断最终任务是否完成。

## 11. 安全与行为 SFT

安全数据不应只有简单拒答，还应区分：

- 明确有害请求：拒绝或安全重定向；
- 合法但敏感的教育、医疗、新闻语境：给出受限帮助；
- 信息不足：澄清或表达不确定性；
- 正常请求：避免误拒答；
- Prompt Injection 和工具边界：保持指令优先级。

如果拒答样本比例过高或模板过于固定，模型会学会“看到敏感词就拒绝”。因此要联合评估 Harmful Compliance 和 False Refusal Rate。

SFT 可以建立基本行为边界，但生产安全还需要偏好优化、红队、部署 Guardrail、访问控制和持续监控。

## 12. 全参数微调、LoRA 与 QLoRA

### 12.1 Full Fine-Tuning

更新所有模型参数。优点是容量充分，适合通用后训练和较大分布变化；代价是需要保存完整梯度、优化器状态和可训练权重。

### 12.2 LoRA

冻结原权重 \(W\)，只学习低秩增量：

\[
W'=W+\Delta W,\qquad \Delta W=BA
\]

其中秩 \(r\ll\min(d_{in},d_{out})\)。LoRA 显著减少可训练参数和优化器显存，适合领域适配、快速实验和多租户 Adapter。

### 12.3 QLoRA

将 Base Model 以 4-bit 量化形式冻结，梯度通过量化权重传播到 LoRA Adapter。QLoRA 进一步降低显存，但训练吞吐、量化 Kernel 和合并部署需要单独评估。

### 12.4 怎么选

| 目标 | 常见选择 | 重点验证 |
|---|---|---|
| 构建通用 Instruct Model | Full FT | 能力保持、总训练成本 |
| 小数据领域适配 | LoRA | Rank、目标层、过拟合 |
| 单卡适配大模型 | QLoRA | 量化误差、Kernel 与导出 |
| 多个独立客户/风格 | LoRA Adapter | Adapter 路由和版本管理 |

PEFT 节省训练资源，不自动保证质量；数据、Rank、目标模块和基座模型仍决定结果。

## 13. 优化器与超参数

SFT 常沿用 AdamW、Warmup、Cosine/Linear Decay、混合精度和梯度裁剪，但相对预训练通常使用更小学习率和更短训练周期。

需要联合扫描的参数包括：

- Peak Learning Rate；
- 全局 Assistant Token 数/更新；
- Epoch 或最大更新步数；
- Warmup Ratio；
- Weight Decay；
- 最大序列长度；
- 数据混合权重；
- Full FT 或 LoRA Rank/Alpha/Target Modules。

不能把“1–3 个 Epoch”当成固定规律：重复数据、目标 Token 数、数据质量和模型规模不同，最佳步数也不同。应使用独立 Eval 做 Early Stopping。

NEFTune 等方法在训练 Embedding 上加入噪声，曾在部分指令微调设置中改善泛化，但它属于可选正则化手段，应与强数据基线做消融，而不是默认必需组件。

## 14. 灾难性遗忘与能力保持

SFT 可能提升指令遵循，却损害知识、代码、多语言或 Base Completion 能力。常见原因包括学习率过大、训练过久、数据域过窄和回答风格单一。

缓解方法：

1. 降低学习率并使用 Early Stopping；
2. 混入通用 SFT 或少量预训练 Replay 数据；
3. 使用多域平衡，而不是只训练目标域；
4. 对参考模型加入 KL/Anchor 约束；
5. 分层冻结或使用 LoRA；
6. 持续跑 Base、Instruct 和 Safety 回归集；
7. 保留多个中间 Checkpoint，按综合 Eval 选模型。

能力保持不是让所有旧指标完全不变，而是明确哪些提升允许交换哪些回退。

## 15. MoE 模型的 SFT 特殊问题

SFT 数据量通常远小于预训练数据，MoE 可能出现：

- Router 分布随对话数据快速漂移；
- 少数 Expert 被特定格式或语言占据；
- 小 Batch 下 Expert Token 数太少；
- 辅助均衡损失与主 SFT 目标比例失衡；
- 未选中 Expert 得不到足够更新。

需要监控每层 Expert 使用率、Router 熵、负载离散度和分域路由。还要明确辅助损失作用于哪些 Token：即使 LM Loss 只监督 Assistant，Router 均衡也可能统计整个上下文，具体取决于实现。

可选策略包括降低 Router 学习率、冻结部分 Router/Expert、混合通用数据，以及使用更大的有效 Token Batch。任何策略都应与 Dense 或冻结 Router 的基线比较。

## 16. SFT 的训练效率

### 16.1 动态 Batch 与长度分桶

以最大 Token 数而非固定样本数组织 Batch，可以在变长数据下稳定显存。分桶减少 Padding，Packing 进一步提高利用率。

### 16.2 混合精度与 Activation Checkpointing

BF16 是常见稳健选择；FP16 可能需要 Loss Scaling。Activation Checkpointing 用重算换显存，LoRA 并不能消除长序列激活占用。

### 16.3 分布式

- DDP 扩大数据吞吐，但每卡保存完整模型；
- FSDP/ZeRO 切分模型、梯度和优化器状态；
- Tensor/Pipeline Parallel 用于更大模型；
- MoE 还可能需要 Expert Parallel。

SFT 数据和序列长度分布更不规则，DataLoader、Padding 和不同 Rank 的 Token 不均衡可能成为真实瓶颈。

## 17. SFT 应该怎样评估

Assistant-only Validation Loss 只能判断对目标文本的拟合程度，不能代表回答质量。完整 Eval 至少包括：

### 17.1 指令与格式

- 指令遵循、否定约束和多条件任务；
- JSON Schema、Tool Call 和结构化输出；
- 多轮状态和角色边界；
- 停止位置与重复生成。

### 17.2 能力保持

- 知识、数学、代码、多语言；
- Base Model 与 SFT 前后对照；
- Seen、Development 和 Unseen Eval 分开；
- 分难度和长上下文结果。

### 17.3 开放回答

结合 Pairwise Judge、人工评审和明确 Rubric，控制长度偏差、位置偏差与 Judge 风格偏好。

### 17.4 安全

同时报告有害遵从、越狱成功率、误拒答和高风险能力。只看拒答率会奖励“什么都不回答”的模型。

### 17.5 系统成本

报告输入/输出 Token、TTFT、TPOT、吞吐和任务完成成本。Reasoning SFT 可能提高质量，也可能显著增加平均输出长度。

## 18. 业界公开 SFT 路线

### 18.1 InstructGPT：人工示范作为对齐起点

使用标注员编写的理想回答训练 SFT 模型，再收集模型回答偏好训练 Reward Model，并通过 PPO 优化。其关键贡献是把“示范学习”和“偏好选择”分成两个阶段。

### 18.2 FLAN：扩大任务、模板与推理示范

FLAN 把大量 NLP 任务统一转成指令格式，并研究任务数、模型规模和 Chain-of-Thought 数据的扩展。它强调 SFT 的跨任务泛化，而不是只拟合聊天风格。

### 18.3 LIMA：强调精选数据

LIMA 展示了少量高质量、多样化示范可以强烈改变大模型的交互行为。其结论应理解为“质量和覆盖优先”，而不是“任何模型都只需固定数量样本”。

### 18.4 Llama 3：合成数据、质量控制与后续偏好优化

Llama 3 的公开报告把 SFT 放在完整后训练系统中，与合成数据、拒绝采样、Reward Model 和 DPO 等方法协同。重点不是某一个算法，而是数据生成、质量筛选和多轮 Eval 的迭代闭环。

### 18.5 Tülu 3：开放的 SFT—DPO—RLVR 流程

Tülu 3 公开数据混合、训练代码、开发/未见 Eval 和去污染方法，并把 SFT 作为 DPO 与可验证奖励 RL 的基础阶段，适合理解现代开放后训练配方。

### 18.6 Qwen3：长 CoT 冷启动与混合思考

公开流程先用长 CoT 数据建立推理模式，再进入推理 RL、Thinking/Non-thinking 融合和通用 RL。它体现了当前趋势：SFT 负责冷启动和模式融合，复杂推理能力则由可验证后训练继续扩展。

## 19. 常见失败模式与定位

| 症状 | 常见原因 | 优先排查 |
|---|---|---|
| Loss 很低但不会对话 | 模板或 Generation Prompt 不一致 | 逐 Token 检查训练/推理序列 |
| 模型复述用户问题 | User Token 也参与 Loss 或数据本身复述 | Loss Mask、回答质量 |
| 不会停止 | EOS/消息结束标记未监督 | 模板、特殊 Token、截断 |
| 回答永远很长 | 长回答 Token 权重过大、Teacher 风格单一 | 长度分布、样本计权 |
| JSON 经常损坏 | 结构 Token、Schema 和转义不一致 | 可执行验证、模板一致性 |
| 通用能力下降 | LR 过大、训练过久、域太窄 | SFT 前后回归、Replay |
| 过度拒答 | 安全样本比例或对照不足 | False Refusal、语境分层 |
| 多轮串角色 | Role 边界或历史数据错误 | 原始对话与渲染结果 |
| Reasoning 看似流畅但答案错 | 未验证的合成轨迹 | Verifier、答案与过程分评 |
| MoE Expert 坍塌 | 小 Batch、路由漂移、均衡不足 | 分层路由统计和数据域 |

## 20. 一套可靠的 SFT 实验顺序

1. 冻结 Base Model、Tokenizer 和 Chat Template 版本；
2. 用极小数据过拟合，验证 Shift、Mask、EOS 和生成链路；
3. 建立人工审计过的高质量小基线；
4. 冻结独立且去污染的多域 Eval；
5. 扫描学习率、训练步数和全局 Assistant Token；
6. 再比较数据规模、数据混合与 Full FT/LoRA；
7. 加入 Reasoning、Tool 和 Safety 数据时分别做增量消融；
8. 同时报告指令能力、基础能力、安全和成本；
9. 保存数据清单、模板、超参数和中间 Checkpoint；
10. SFT 稳定后再进入偏好优化或推理 RL。

每轮实验应尽量只回答一个主要问题。数据、模板、Loss Mask 和优化器同时变化时，很难判断收益来自哪里。

## 21. miniLLM 的 SFT 映射

miniLLM 从 `out/pretrain` 加载预训练权重，进行全参数 SFT；完整 System/User/Tool/Assistant 对话进入模型，但只有 Assistant 的 Reasoning、正文、Tool Call 和 `<|im_end|>` 参与语言模型 Loss。数据管线使用 Fast Tokenizer 的 Offset Mapping 构造精确 Assistant Span，并采用动态 Padding。

长对话会保留 System 消息、优先删除最早完整轮次，并尽量保留最近问题和回答开头；若回答被截断，不会伪造提前结束标记。训练侧采用 AdamW、Warmup + Cosine、混合精度、梯度累积、默认 Activation Checkpointing 和可选 DDP，同时记录 Assistant Token 吞吐、Padding 效率、截断率以及 MoE Router 指标。

当前最值得补强的是：将验证集改为独立数据源；增加指令遵循、JSON/Tool 执行、安全与基础能力保持 Eval；比较 Assistant Token 计权和逐样本计权；最后再评估 Packing、LoRA 与 Reasoning 数据配比。这样能让 SFT 从“Loss 能下降”升级为“行为确实改善且没有不可接受的能力回退”。

## 参考资料

- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [NEFTune: Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124)
- [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)

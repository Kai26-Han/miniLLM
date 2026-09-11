<div align="center">

![miniLLM](images/minillm-banner.png)


### 学习从 0 构建一个小型语言模型

中文 | [English](README_en.md)

🚀 [**在线体验 miniLLM 模型**](https://www.modelscope.cn/studios/kayson2026/miniLLM)

👁️ [**在线体验 miniLLM-vlm 视觉模型**](https://www.modelscope.cn/studios/kayson2026/miniLLM-vlm)

</div>

---

## 🎯 项目愿景

miniLLM 是一个面向 LLM 入门、实验与原理学习的项目。

希望把语言模型从“可以调用的黑盒”变成“可以逐层理解的工程系统”：从原始语料和 Tokenizer 开始，经过预训练、指令微调、偏好对齐、工具使用和视觉扩展，逐步理解现代语言模型与多模态模型的主要生命周期。

项目不只提供训练脚本，还配套了中英文原理文章和实战资料，希望让每一个训练阶段都能被阅读、复现、比较和扩展。

## 🖥️ 单卡起步，动手学习大模型训练

**本项目使用单张 NVIDIA GeForce RTX 4090 进行训练。** 不必从多卡集群起步，也能亲手实践语言模型训练，理解数据、模型结构与参数更新之间的关系。

miniLLM 希望降低的是实操学习的算力门槛：先用小数据集和短训练跑通流程，再逐步扩大实验。各阶段的显存与耗时不同，仍需按可用资源调整批量大小、序列长度和采样规模；单卡训练不代表所有阶段都能直接使用默认配置，也不代表完整训练没有时间成本。

## ✨ 项目包含什么

- 可从零训练的 8,192 词表 ByteLevel-BPE Tokenizer
- 由 RMSNorm、RoPE、GQA 和 SwiGLU 组成的 Decoder-only Transformer
- Dense 与稀疏 MoE 两种结构，默认使用 4 Experts / Top-1 路由
- 基于 miniLLM Base 和 SigLIP 视觉编码器扩展的 miniLLM-vlm
- 预训练、全参数 SFT、LoRA 和离线黑盒蒸馏
- DPO、PPO、GRPO 以及多轮 Agentic RL
- 断点续训、混合精度、梯度累积、梯度检查点和编译优化
- SwanLab 与 Weights & Biases 实验跟踪
- Tokenizer、预训练模型、SFT、DPO 和 PPO 评估工具
- 配套的中英文 LLM 学习资料

## 🧭 先理解：模型到底在“学”什么

可以先用“图片里有什么？”这个问题区分各阶段：Tokenizer 把文字变成编号；语言预训练让模型学会续写；SFT 用示范教它回答问题；DPO 用回答对告诉它哪个更好；PPO / GRPO 让它自己生成回答、获得评分、再调整；VLM 则额外提供图片信息，让回答能够依据视觉内容。

训练时改变的是模型中的数值参数，也就是权重。一次更新通常经过以下过程：

| 环节 | 在做什么 | 初学者需要理解的词 |
| --- | --- | --- |
| 准备一批数据 | 把文本或图文整理成模型能读的输入，并指定哪些位置需要学习 | Batch：一次送入的样本；Token：文字切分后的单位 |
| 前向计算 | 模型给出下一个 Token 的概率，或对已生成的回答重新计算概率 | Forward：使用当前参数计算输出 |
| 计算训练目标 | 根据参考答案、偏好或奖励，衡量当前输出需要怎样改进 | Loss：用于优化的数值目标；Reward：对生成结果的评分 |
| 反向传播 | 计算各参数变化会怎样影响 Loss | Gradient：指示参数调整方向的导数 |
| 更新参数 | 优化器按梯度和学习率调整可训练权重 | Learning rate：更新步幅；冻结参数不参与更新 |
| 验证与保存 | 在未参与更新的数据上检查结果，并保存可继续训练的状态 | Validation：检查泛化；Checkpoint：训练状态存档 |

一个 Epoch 是遍历一轮训练集。梯度累积允许分几次读取小 Batch、再合并更新一次，所以读取数据的次数不一定等于优化器更新次数。训练 Loss 下降只说明模型更适应训练目标，还需要验证集和实际问答来判断能力。

## 🧠 模型架构

miniLLM 是一个自回归 Decoder-only Causal Language Model。模型默认采用稀疏 MoE，同时保留 Dense 对照实验能力。

| 配置 | 默认值 |
| --- | ---: |
| 词表大小 | 8,192 |
| 隐藏维度 | 768 |
| Decoder 层数 | 8 |
| Query Heads / KV Heads | 8 / 4 |
| Head Dimension | 96 |
| FFN / Expert 中间维度 | 2,432 |
| 最大位置配置 | 32,768 |
| 归一化 | RMSNorm |
| 位置编码 | RoPE |
| 注意力 | Grouped Query Attention |
| 前馈网络 | SwiGLU |
| 默认路由 | 4 Experts / Top-1 |
| Dense 参数量 | 约 65.3M |
| MoE 总参数 / 每 Token 激活参数 | 约 199.8M / 65.3M |

最大位置配置只表示模型可接受的位置索引范围。模型是否真正具备可靠的长上下文能力，仍取决于训练时的序列长度、数据分布和评测结果。

### 一段文字如何通过模型

**文本 → Tokenizer → Token Embedding → 8 层 Decoder → 最终 RMSNorm → LM Head → 下一个 Token 的概率。**

Embedding 把离散编号变成可计算的向量。每个 Decoder 层先让 Token 读取左侧上下文，再用前馈网络加工得到的表示。训练时使用因果遮罩，可同时计算多个位置的预测，但每个位置仍不能偷看后面的答案；生成时则每次产生一个新 Token，接回上下文，继续预测。

| 组件 | 在本项目中的作用 |
| --- | --- |
| RMSNorm 与残差连接 | 调整表示的尺度，并保留前一层的信息，帮助多层网络训练 |
| GQA + RoPE | GQA 让多个 Query 头共享 Key / Value 头；RoPE 将位置信息注入注意力计算 |
| SwiGLU 前馈网络 | 对注意力汇总后的信息做带门控的非线性变换 |
| Dense / MoE | Dense 每次执行同一组前馈参数；MoE 由 Router 为每个 Token 选择 Expert |
| 共享 Embedding / LM Head | 输入词嵌入与输出词表投影共享权重，减少参数量 |

默认 MoE 每层有 4 个 Expert，每个 Token 激活其中 1 个。专家负载均衡辅助项鼓励更均衡的使用。约 65.3M 激活参数描述单 Token 的参与计算范围，所有约 199.8M 参数仍需保存；MoE 不会按激活比例缩减权重和优化器内存。

## 🗺️ 完整训练路线

| 阶段 | 目标 | 主要输出 |
| --- | --- | --- |
| Tokenizer | 将文本映射为稳定的 Token ID | ByteLevel-BPE 词表与对话模板 |
| Pretraining | 通过下一 Token 预测学习语言分布 | 基础语言模型 |
| SFT / LoRA / Distillation | 学习指令遵循和对话能力 | 指令模型或 LoRA Adapter |
| VLM Pretrain / SFT | 对齐视觉与语言表示，学习图像理解与图文对话 | miniLLM-vlm 视觉语言模型 |
| DPO | 从偏好对中学习更优回答 | 偏好对齐模型 |
| PPO / GRPO | 利用 Reward 信号进行在线策略优化 | 强化学习模型 |
| Agentic RL | 在多轮环境中学习工具使用 | Agentic 策略模型 |

这些阶段是可组合的学习路线，并非必须依次完成的关卡：

- **第一次跑通语言模型：**使用已有 Tokenizer，完成预训练 → Base → 全参数 SFT → 对话评估；想研究分词时，再从 Tokenizer 训练开始。
- **领域与教师数据实验：**在 SFT 模型上做 LoRA；或者使用教师生成的数据，从 Base 进行本项目的黑盒蒸馏。
- **偏好与强化学习实验：**DPO 从 SFT 初始化；PPO 默认从 SFT 开始，也可接 DPO；GRPO 默认从 DPO 开始。PPO 和 GRPO 可以分别比较，Agentic RL 默认接已训练的 PPO Actor。
- **视觉扩展：**miniLLM Base + 视觉编码器 → VLM Pretrain → VLM SFT；无需先跑完语言模型的 DPO / PPO / GRPO。

| 阶段 | 哪些参数更新 | 哪些部分保持固定 |
| --- | --- | --- |
| Tokenizer | 学习词表与 BPE 合并规则 | 此时不训练 LLM |
| 语言预训练 / 全参数 SFT / 黑盒蒸馏 | miniLLM 全部语言模型参数 | Tokenizer；蒸馏教师不参与学生训练更新 |
| LoRA | 注入线性层的低秩 Adapter | 原语言模型权重和 Tokenizer |
| DPO | Policy，也就是待优化的 miniLLM | Reference，也就是初始化模型的冻结副本 |
| PPO | Actor 与 Critic | Reference 与外部 Reward Model |
| GRPO | Policy | Reference 与外部 Reward Model；没有独立 Critic |
| Agentic RL | Agent Policy | Reference 与工具执行、奖励规则 |
| VLM Pretrain | Projector | miniLLM Base 与 SigLIP |
| VLM SFT（默认） | Projector 与 LLM 第 0、7 层 | SigLIP、中间语言层及其他冻结参数 |

下面的“流程”说明数据和模块怎样配合；每阶段末尾附有对应实战篇的参考资料入口，便于继续学习与动手实践。

## 🔤 Tokenizer

Tokenizer 决定了文本如何进入模型。miniLLM 提供 ByteLevel-BPE 训练、语料清洗、词表验证、特殊 Token 约束和 Chat Template。仓库中已包含一份可用的 Tokenizer，也可根据新语料重新训练。

### 思路：先学会怎样切分文字

**语料 → 字节级表示与预切分 → 统计相邻片段 → 反复合并高频片段 → 固定词表与编码规则。**

BPE 可以把常见片段合并成较长的 Token，较少出现的内容则拆成更细的单位；一个 Token 不一定等于一个汉字或一个英文单词。这里学习的是分词规则，不使用神经网络反向传播。Chat Template 负责把不同角色的消息排成约定格式，也不等于模型已经学会对话。

实操时先检查中文、英文和符号能否编码后还原，再比较同一段文字需要多少 Token。产物是后续所有语言训练阶段共同使用的 Tokenizer 文件。

实战参考：[中文 · 训练 Tokenizer](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99%20-%20%E4%BB%8E0%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BALLM/%E5%AE%9E%E6%88%98%E7%AF%87/%E8%AE%AD%E7%BB%83Tokenizer.md) · [English · Training a Tokenizer](Learning%20Materials%20-%20Building%20an%20LLM%20from%20Scratch/Hands-On/Training%20a%20Tokenizer.md)。

![miniLLM Tokenizer](images/tokenizer.png)

Tokenizer 一旦进入预训练阶段，词表映射和特殊 Token ID 就应保持不变，否则模型的 Embedding 与输出层将与 Tokenizer 失去对应关系。

## 🚀 预训练

预训练从随机权重开始，通过 Causal Language Modeling 学习文本中的词法、语法、语义和知识关联。Dense 与 MoE 模型共用同一套主体架构；MoE 通过 Router 为每个 Token 选择 Expert，并用辅助损失缓解路由塌缩。

### 思路：把原文自己变成训练答案

**文本语料 → 固定 Tokenizer → 随机初始化的 miniLLM → 每个位置预测下一个 Token → 交叉熵与 MoE 辅助项 → 更新模型。**

例如，一段文本的 Token 序列是“今天 / 天气 / 很好”，前两个位置分别学习预测“天气”和“很好”。答案来自原文向后移动一位，不需要人为为每段文字编写问题。交叉熵会惩罚模型给正确 Token 分配过低的概率；Padding 只是把序列补齐，不计入语言学习目标。

这一阶段更新全部语言模型参数，产物是 Base 模型，主要学习续写能力。实际检查时关注验证集语言损失、困惑度和文本续写；困惑度只衡量语言预测，不应把 Router 辅助损失加进去。

实战参考：[中文 · miniLLM 预训练](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99%20-%20%E4%BB%8E0%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BALLM/%E5%AE%9E%E6%88%98%E7%AF%87/miniLLM%20%E9%A2%84%E8%AE%AD%E7%BB%83.md) · [English · miniLLM Pretraining](Learning%20Materials%20-%20Building%20an%20LLM%20from%20Scratch/Hands-On/miniLLM%20Pretraining.md)。

![miniLLM Pretraining](images/pretrain.png)

训练过程支持验证集评估、学习率调度、完整检查点、分布式训练和 Hugging Face 格式导出。

## 💬 指令微调

SFT 让基础模型学会理解用户输入并给出结构化回答。miniLLM 的数据流程可保留 system、user 和 tool 消息作为上下文，只对 assistant 输出部分计算语言模型损失。

### 思路：用示范教模型怎样回答

**对话数据 → ChatML 模板 → 已训练的 Base → 在 assistant 答案位置计算交叉熵 → 更新全部语言模型参数。**

同样是预测下一个 Token，SFT 改变的是数据组织方式与监督位置：模型能读取用户问题和历史消息，但只把 assistant 输出当作需要模仿的答案。训练时提供真实答案前缀来预测下一 Token；实际对话时则使用模型自己已生成的回答作为前缀，两种情况应分别检查。

产物是指令模型。实操时检查答案 Token 是否被正确标记、长对话是否截断掉关键答案，并在固定的一组新问题上比较 Base 与 SFT 的响应。验证 Loss 之外，也要看是否答非所问、重复或不能正确结束。

实战参考：[中文 · miniLLM SFT 训练](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99%20-%20%E4%BB%8E0%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BALLM/%E5%AE%9E%E6%88%98%E7%AF%87/miniLLM%20SFT%E8%AE%AD%E7%BB%83.md) · [English · miniLLM SFT Training](Learning%20Materials%20-%20Building%20an%20LLM%20from%20Scratch/Hands-On/miniLLM%20SFT%20Training.md)。

![miniLLM SFT](images/sft.png)

### LoRA：只训练附加的小矩阵

**已有 SFT 模型 + 领域对话 → 在目标线性层旁加入低秩分支 → 原输出加上缩放后的分支输出 → 只更新 Adapter。**

LoRA 将权重调整表示为两个较小矩阵的乘积。默认在注意力的 Q、K、V、O 投影层注入这些矩阵，原权重冻结，训练目标仍是 assistant 答案的交叉熵。“Rank”控制低秩分支的容量，容量与所需训练资源会随之变化。

LoRA 是参数更新方式，可以用于 SFT，并不是 SFT 之后必经的新训练阶段。产物是小型 Adapter，需要和匹配的基础模型一起加载，也可合并成完整模型。实操时比较同一个基础模型在加载前后的领域回答，以及通用回答是否退化。

实战参考：[中文 · miniLLM LoRA 训练](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99%20-%20%E4%BB%8E0%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BALLM/%E5%AE%9E%E6%88%98%E7%AF%87/miniLLM%20LoRA%E8%AE%AD%E7%BB%83.md) · [English · miniLLM LoRA Training](Learning%20Materials%20-%20Building%20an%20LLM%20from%20Scratch/Hands-On/miniLLM%20LoRA%20Training.md)。

### 黑盒蒸馏：把教师的回答变成学生教材

**问题 → 教师模型生成文本 → 整理并验证对话数据 → miniLLM 用自己的 Tokenizer 编码 → 按 SFT 目标学习。**

本项目将教师生成与学生训练分成两步：先离线生成数据，再通过 SFT 训练学生。生成脚本默认使用 Qwen3-1.7B，并支持混合保留原始回答。学生学习的是生成文本中的 Token，即“硬标签”，不会在学生训练期间调用教师反向传播，也不直接对齐两个词表的 logits。

产物是蒸馏后的 miniLLM 指令模型。实操时先抽查教师回答的正确性、语言和格式，再比较学生在未见问题上的表现；教师错误也可能被学生学到。

实战参考：[中文 · miniLLM 黑盒蒸馏](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99%20-%20%E4%BB%8E0%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BALLM/%E5%AE%9E%E6%88%98%E7%AF%87/miniLLM%20%E9%BB%91%E7%9B%92%E8%92%B8%E9%A6%8F.md) · [English · miniLLM Black-Box Distillation](Learning%20Materials%20-%20Building%20an%20LLM%20from%20Scratch/Hands-On/miniLLM%20Black-Box%20Distillation.md)。

## 🏆 偏好对齐与强化学习

### DPO：学习同一个问题的回答偏好

**同一问题及上下文 + 较优 / 较差回答 → Policy 与冻结 Reference 分别计算回答概率 → 比较相对偏好 → 只更新 Policy。**

Policy 是要训练的模型，Reference 是训练开始时模型的固定参照。DPO 希望 Policy 相对 Reference 更偏向 chosen 回答，而不是只提高两个回答的共同概率。这里使用的是预先准备好的回答对，训练时不需要外部 Reward Model 为新回答评分；只统计最后一个 assistant 回答的 Token 概率。

产物是偏好对齐的语言模型。实操时先确认两个回答之前的历史完全一致，再观察验证偏好准确率、偏好间隔和实际生成质量。DPO 中从概率变化计算的“奖励”，与 PPO 使用的外部评分模型不是同一个对象。

实战参考：[中文 · miniLLM DPO 训练](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99%20-%20%E4%BB%8E0%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BALLM/%E5%AE%9E%E6%88%98%E7%AF%87/miniLLM%20DPO%E8%AE%AD%E7%BB%83.md) · [English · miniLLM DPO Training](Learning%20Materials%20-%20Building%20an%20LLM%20from%20Scratch/Hands-On/miniLLM%20DPO%20Training.md)。

### PPO：自己作答，由评分与价值估计指导改进

**Prompt → Actor 采样回答 → Reward Model 评分，Critic 估计预期回报 → 计算 Advantage → 更新 Actor 与 Critic。**

Rollout 指模型实际生成回答的过程。Advantage（优势）衡量结果比预期好多少；这里通过 GAE 将奖励与 Critic 的价值估计结合，形成生成位置的学习信号。

| 模块 | 作用 | 是否训练 |
| --- | --- | --- |
| Actor | 生成回答的语言模型 | 是 |
| Critic | 语言模型骨干加价值头，估计当前前缀后续的回报 | 是 |
| Reward Model | 为生成答案评分；本项目再结合规则惩罚 | 否 |
| Reference | 提供固定的语言行为参照 | 否 |

PPO 保存采样时的概率，再与更新后的概率比较，通过裁剪限制单次策略变化；KL 约束控制与 Reference 的偏离。这两个参照不同：前者对应本次采样策略，后者对应固定参考模型。

![miniLLM PPO](images/PPO.png)

训练后用于对话的是 Actor，Critic 和 Reward Model 是训练辅助组件。实操时同时观察奖励、KL、价值损失和实际回答；只看奖励上升可能漏掉重复、冗长或迎合评分规则的问题。

实战参考：[中文 · miniLLM PPO 训练](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99%20-%20%E4%BB%8E0%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BALLM/%E5%AE%9E%E6%88%98%E7%AF%87/miniLLM%20PPO%E8%AE%AD%E7%BB%83.md) · [English · miniLLM PPO Training](Learning%20Materials%20-%20Building%20an%20LLM%20from%20Scratch/Hands-On/miniLLM%20PPO%20Training.md)。

### GRPO：用同组回答互相比较

**同一 Prompt → Policy 生成多个回答 → 冻结 Reward Model 与规则评分 → 组内奖励标准化 → 更新 Policy。**

GRPO 用这一组回答的平均奖励和标准差计算相对优势，省去独立 Critic。若一组回答得分为 1、2、3、2，那么较高分回答得到正优势，较低分回答得到负优势；这是同一问题内的比较，不能直接把不同问题混成一组。本项目仍使用裁剪目标与冻结 Reference 的 KL 约束。

![miniLLM GRPO](images/GRPO.png)

产物仍是可生成文本的 Policy。实操时关注每组奖励是否有差异：全组同分时，相对优势为零，无法提供偏好方向。还应检查生成长度、重复率与 KL；采样多个答案本身也需要显存和计算时间。

实战参考：[中文 · miniLLM GRPO 训练](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99%20-%20%E4%BB%8E0%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BALLM/%E5%AE%9E%E6%88%98%E7%AF%87/miniLLM%20GRPO%E8%AE%AD%E7%BB%83.md) · [English · miniLLM GRPO Training](Learning%20Materials%20-%20Building%20an%20LLM%20from%20Scratch/Hands-On/miniLLM%20GRPO%20Training.md)。

### Agentic RL：学习“调用工具—读取结果—继续回答”

**任务与工具说明 → Policy 生成工具调用 → 环境执行 → 结果写回上下文 → Policy 继续行动或回答 → 对整条轨迹评分。**

这里的一条训练样本包含多轮行动与反馈，称为轨迹。项目默认从 PPO Actor 初始化，提供计算等本地工具环境；天气、时间等示例使用内置数据。奖励结合答案与目标的匹配、工具调用是否成功、格式是否有效，以及重复或无效调用的惩罚，而不是复用普通 PPO 的外部 Reward Model。

同一任务采样多条轨迹后，默认用 GRPO 目标更新 Policy，也提供 CISPO 选项。环境返回的工具内容是观察信息，只有模型自己生成的行动和回答 Token 参与策略目标；工具本身不通过反向传播训练，Reference 保持冻结。

实操时读完整轨迹：模型是否选对工具、参数是否有效、是否根据返回结果回答。当前结果奖励要求成功执行工具，猜对答案但未执行工具不会获得该项奖励。最终产物是 Agent Policy，推理时仍需配套工具执行环境。

实战参考：[中文 · miniLLM Agentic RL 训练](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99%20-%20%E4%BB%8E0%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BALLM/%E5%AE%9E%E6%88%98%E7%AF%87/miniLLM%20Agentic%20RL%E8%AE%AD%E7%BB%83.md) · [English · miniLLM Agentic RL Training](Learning%20Materials%20-%20Building%20an%20LLM%20from%20Scratch/Hands-On/miniLLM%20Agentic%20RL%20Training.md)。

## 👁️ miniLLM-vlm 视觉语言模型

[miniLLM-vlm](miniLLM-vlm/) 是在已训练 miniLLM Base 上扩展的视觉语言模型。它使用 SigLIP 将 256 × 256 图像编码为 64 个视觉 Token，再通过 LayerNorm、Linear、GELU 和 Linear 组成的 Projector，将视觉特征映射到 miniLLM 的语言嵌入空间。这种设计复用已有视觉与语言能力，重点学习两种模态之间的连接。

项目是一个可独立运行的子项目，包含自己的模型定义、数据处理、训练器、评估入口和中英文学习资料。它面向单图理解，不扩充 Tokenizer 词表，而是复用保留 Token 作为连续的图像占位。

### 图像怎样进入语言模型

**图片 → 冻结 SigLIP → 64 个视觉向量 → Projector 映射；问题文本 → Tokenizer → 文字向量；两路向量按序组合 → miniLLM → 回答。**

256 × 256 图像按 32 × 32 Patch 形成 8 × 8 网格。视觉编码器输出的每个向量表示一个位置的视觉特征；它们经 Projector 后替换图像占位位置的 Embedding，与文本共享同一上下文。视觉 Token 占据序列长度，但不对应要预测的答案文字。

### 视觉 Pretrain

视觉 Pretrain 使用单图描述数据建立图像与文本的基础对齐。此阶段冻结 SigLIP 视觉编码器和 miniLLM Base，只训练 Projector；答案误差仍会穿过语言模型反向传播到 Projector，从而让映射后的视觉特征逐步适应语言模型。

训练目标是在图片和问题条件下预测描述文字，只监督 assistant 答案和真实结束符；图像、问题和 Padding 不计入答案交叉熵。冻结语言模型表示“不更新语言权重”，仍须保留通向 Projector 的梯度路径。

产物主要是 Projector Adapter；下一阶段还需同一套 miniLLM Base、SigLIP 与 Tokenizer。实操时确认只有 Projector 有参数梯度，并对比正确图片和错配图片，检查模型是否开始利用视觉信息。

实战参考：[中文 · miniLLM-vlm Pretrain 训练](miniLLM-vlm/%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/%E5%AE%9E%E6%88%98%E7%AF%87/miniLLM-vlm%20Pretrain%E8%AE%AD%E7%BB%83.md) · [English · miniLLM-vlm Pretraining](miniLLM-vlm/learning-materials-en/practice/miniLLM-vlm-Pretraining.md)。

![miniLLM-vlm Pretrain 训练曲线](miniLLM-vlm/images/pretrain.png)

### 视觉 SFT

视觉 SFT 从 Pretrain 产物继续训练，覆盖单图问答、多轮图文对话和纯文本指令。默认保持 SigLIP 冻结，同时训练 Projector 以及 miniLLM 第一层和最后一层 Decoder Block，在视觉指令跟随与原有语言能力之间取得平衡。

**图文 / 纯文本对话 → 加载已对齐 Projector 和 Base → 按 assistant 回答位置监督 → 更新 Projector 与选定语言层。**

相较于描述图片，视觉 SFT 进一步要求模型依据用户问题选择相关信息。默认更新的首尾 Decoder Block 包含各自的注意力、MoE Expert 和 Router；冻结的中间层仍参与计算与梯度传递。纯文本样本跳过视觉分支，监督语言层回答；有图样本同时使用视觉与文本上下文。

默认预训练总长度预算为 512，视觉 SFT 为 768，均包含 64 个视觉位置。实操时需检查长对话是否还有足够答案位置，并比较图文与纯文本两类验证结果。SFT 导出包含更新后的完整 LLM 和 Projector，推理仍需配套 Tokenizer 与原 SigLIP。

实战参考：[中文 · miniLLM-vlm SFT 训练](miniLLM-vlm/%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/%E5%AE%9E%E6%88%98%E7%AF%87/miniLLM-vlm%20SFT%E8%AE%AD%E7%BB%83.md) · [English · miniLLM-vlm SFT](miniLLM-vlm/learning-materials-en/practice/miniLLM-vlm-SFT.md)。

![miniLLM-vlm SFT 训练曲线](miniLLM-vlm/images/sft.png)

评估流程支持单图描述、视觉问答、多轮消息和纯文本推理，并会通过正确图像与错配图像的损失差异，辅助判断模型是否真正使用了视觉信息。

🚀 [**在 ModelScope 体验 miniLLM-vlm**](https://www.modelscope.cn/studios/kayson2026/miniLLM-vlm)

- [miniLLM-vlm 项目说明](miniLLM-vlm/README.md)
- [miniLLM-vlm 中文学习资料](miniLLM-vlm/%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/)
- [miniLLM-vlm English Learning Materials](miniLLM-vlm/learning-materials-en/)

## 📚 数据与学习资料

仓库不直接携带大型训练数据和模型权重。数据需按照各阶段格式放入对应的 dataset 目录：语言预训练使用文本 JSONL，SFT 使用多轮对话，DPO 使用偏好对，PPO / GRPO 使用待生成回答的 Prompt 对话；VLM 使用包含图像与对话的 Parquet 数据。

📦 **数据集下载：**

- [ModelScope · miniLLM-dataset](https://www.modelscope.cn/datasets/kayson2026/miniLLM-dataset/files)
- [Hugging Face · miniLLM-dataset](https://huggingface.co/datasets/fenglike/miniLLM-dataset/tree/main)

如果是第一次学习 LLM，建议按以下顺序阅读：

1. 先建立对 LLM 与全流程的整体认知。
2. 学习 Tokenizer、Transformer 和自注意力。
3. 理解 Dense、MoE 以及常见优化方法。
4. 进入预训练、SFT 和 RL 算法。
5. 结合实战文档完成各阶段实验。
6. 在 miniLLM Base 上进一步学习视觉语言对齐与视觉 SFT。

- [中文学习资料](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99%20-%20%E4%BB%8E0%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BALLM/)
- [English Learning Materials](Learning%20Materials%20-%20Building%20an%20LLM%20from%20Scratch/README.md)

## 🗂️ 仓库导航

| 目录 | 内容 |
| --- | --- |
| [model](model/) | miniLLM 模型、LoRA 与 Tokenizer |
| [dataset](dataset/) | 预训练、SFT、DPO 与 PPO 数据加载器 |
| [trainers](trainers/) | 各训练阶段的入口与通用训练能力 |
| [eval](eval/) | Tokenizer 和模型评估 |
| [scripts](scripts/) | 语料抽取、蒸馏数据生成、数据验证和 LoRA 合并 |
| [images](images/) | 项目和训练阶段示意图 |
| [miniLLM-vlm](miniLLM-vlm/) | 基于 miniLLM Base 的视觉语言模型、训练、评估与学习资料 |

## 🔎 如何读训练曲线

本页图片是训练过程记录。先看指标名称和所属阶段，再判断变化意味着什么；不同数据、Tokenizer、监督位置下的 Loss 数值不能直接横向排名。

| 看到的现象 | 可能说明什么 | 下一步检查 |
| --- | --- | --- |
| 训练 Loss 与验证 Loss 一起下降 | 在当前目标上学习有效 | 用固定的新问题比较生成结果 |
| 训练 Loss 下降、验证 Loss 上升 | 可能开始过拟合 | 数据重复、训练轮数与学习率 |
| MoE 使用集中于少数 Expert | 可能存在路由失衡 | 专家分配统计与辅助损失，而不只看总 Loss |
| RL 奖励上升，回答却更重复 | 评分与期望行为可能不一致 | 奖励各分项、KL 与生成样本 |
| 正确图和错配图的 Loss 接近 | 尚缺少模型利用图片的证据 | 数据对齐、投影层梯度与视觉问答 |
| 梯度出现 NaN / Inf，或 Loss 突增 | 可能有数值、数据或更新异常 | 样本、精度、学习率和梯度裁剪 |

从一个小规模实验开始，固定验证问题和随机种子，每次只改一个主要变量。这样能解释“为什么结果变了”，也更容易把 README 中的训练思路与实际观察对应起来。

## 🧪 实验建议

- 先用小数据集和少量训练步数验证整条链路，再扩大实验。
- 保存每次实验的数据版本、Tokenizer 指纹、随机种子和完整参数。
- 检查点用于精确续训，导出模型用于评估、生成和下游训练，两者用途不同。
- PPO 和 GRPO 默认需要额外的 Reward Model 资源，应根据显存调整训练规模。
- 小模型的结果对数据质量和超参数很敏感，请优先做可复现的对照实验。

## 🙏 致谢

特别感谢以下项目：

- [MiniMind](https://github.com/jingyaogong/minimind) —— 为轻量级语言模型的全流程训练、数据处理、偏好对齐与工程组织提供了重要参考。
- [MiniMind-V](https://github.com/jingyaogong/minimind-v) —— 为在轻量语言模型上接入视觉编码器、进行跨模态对齐和视觉 SFT 提供了重要参考。
- [nanoGPT](https://github.com/karpathy/nanoGPT) —— 以简洁、透明的方式展示 GPT 训练与微调，持续启发了“通过可读代码理解模型”的项目理念。

同时感谢 PyTorch、Hugging Face Transformers、Hugging Face Datasets 以及所有开源数据和研究成果的贡献者。

---

<div align="center">

如果 miniLLM 对你有帮助，欢迎点亮 ⭐，也欢迎一起让它更容易理解、复现和扩展。

</div>

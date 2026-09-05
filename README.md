<div align="center">

![miniLLM](images/minillm-banner.png)


### 学习从 0 构建一个小型语言模型

中文 | [English](README_en.md)

🚀 [**在线体验 miniLLM 模型**](https://www.modelscope.cn/studios/kayson2026/miniLLM)

</div>

---

## 🎯 项目愿景

miniLLM 是一个面向 LLM 入门、实验与原理学习的项目。

希望把语言模型从“可以调用的黑盒”变成“可以逐层理解的工程系统”：从原始语料和 Tokenizer 开始，经过预训练、指令微调、偏好对齐和工具使用，最终走完一个现代语言模型的主要生命周期。

项目不只提供训练脚本，还配套了中英文原理文章和实战资料，希望让每一个训练阶段都能被阅读、复现、比较和扩展。

## ✨ 项目包含什么

- 可从零训练的 8,192 词表 ByteLevel-BPE Tokenizer
- 由 RMSNorm、RoPE、GQA 和 SwiGLU 组成的 Decoder-only Transformer
- Dense 与稀疏 MoE 两种结构，默认使用 4 Experts / Top-1 路由
- 预训练、全参数 SFT、LoRA 和离线黑盒蒸馏
- DPO、PPO、GRPO 以及多轮 Agentic RL
- 断点续训、混合精度、梯度累积、梯度检查点和编译优化
- SwanLab 与 Weights & Biases 实验跟踪
- Tokenizer、预训练模型、SFT、DPO 和 PPO 评估工具
- 配套的中英文 LLM 学习资料

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

## 🗺️ 完整训练路线

| 阶段 | 目标 | 主要输出 |
| --- | --- | --- |
| Tokenizer | 将文本映射为稳定的 Token ID | ByteLevel-BPE 词表与对话模板 |
| Pretraining | 通过下一 Token 预测学习语言分布 | 基础语言模型 |
| SFT / LoRA / Distillation | 学习指令遵循和对话能力 | 指令模型或 LoRA Adapter |
| DPO | 从偏好对中学习更优回答 | 偏好对齐模型 |
| PPO / GRPO | 利用 Reward 信号进行在线策略优化 | 强化学习模型 |
| Agentic RL | 在多轮环境中学习工具使用 | Agentic 策略模型 |

## 🔤 Tokenizer

Tokenizer 决定了文本如何进入模型。miniLLM 提供 ByteLevel-BPE 训练、语料清洗、词表验证、特殊 Token 约束和 Chat Template。仓库中已包含一份可用的 Tokenizer，也可根据新语料重新训练。

![miniLLM Tokenizer](images/tokenizer.png)

Tokenizer 一旦进入预训练阶段，词表映射和特殊 Token ID 就应保持不变，否则模型的 Embedding 与输出层将与 Tokenizer 失去对应关系。

## 🚀 预训练

预训练从随机权重开始，通过 Causal Language Modeling 学习文本中的词法、语法、语义和知识关联。Dense 与 MoE 模型共用同一套主体架构；MoE 通过 Router 为每个 Token 选择 Expert，并用辅助损失缓解路由塌缩。

![miniLLM Pretraining](images/pretrain.png)

训练过程支持验证集评估、学习率调度、完整检查点、分布式训练和 Hugging Face 格式导出。

## 💬 指令微调

SFT 让基础模型学会理解用户输入并给出结构化回答。miniLLM 的数据流程可保留 system、user 和 tool 消息作为上下文，只对 assistant 输出部分计算语言模型损失。

![miniLLM SFT](images/sft.png)

除全参数 SFT 外，项目还支持 LoRA 领域微调与离线黑盒序列蒸馏，用于比较不同能力迁移路线。

## 🏆 偏好对齐与强化学习

DPO 通过 chosen / rejected 回答对学习人类或 AI 偏好。在此基础上，PPO 和 GRPO 支持基于外部 Reward Model 的在线生成与策略优化。

![miniLLM PPO](images/PPO.png)

PPO 路线包含 Actor、Critic、Reward、GAE 和裁剪目标，适合学习完整的 RLHF 训练链路。

![miniLLM GRPO](images/GRPO.png)

GRPO 使用同一 Prompt 的多个候选回答构造组内相对优势，不依赖独立 Critic。项目还在此基础上提供多轮 Agentic RL，用于学习工具调用与环境交互。

## 📚 数据与学习资料

仓库不直接携带大型训练数据和模型权重。数据需按照各阶段格式放入 dataset 目录：预训练使用文本 JSONL，SFT 使用多轮对话，DPO 使用偏好对，PPO / GRPO 使用待生成回答的 Prompt 对话。

📦 **数据集下载：**

- [ModelScope · miniLLM-dataset](https://www.modelscope.cn/datasets/kayson2026/miniLLM-dataset/files)
- [Hugging Face · miniLLM-dataset](https://huggingface.co/datasets/fenglike/miniLLM-dataset/tree/main)

如果是第一次学习 LLM，建议按以下顺序阅读：

1. 先建立对 LLM 与全流程的整体认知。
2. 学习 Tokenizer、Transformer 和自注意力。
3. 理解 Dense、MoE 以及常见优化方法。
4. 进入预训练、SFT 和 RL 算法。
5. 结合实战文档完成各阶段实验。

- [中文学习资料](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99%20-%20%E4%BB%8E0%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BALLM/)
- [English Learning Materials](Learning%20Materials%20-%20Building%20an%20LLM%20from%20Scratch/README.md)
- [数据集说明](dataset/dataset.md)

## 🗂️ 仓库导航

| 目录 | 内容 |
| --- | --- |
| [model](model/) | miniLLM 模型、LoRA 与 Tokenizer |
| [dataset](dataset/) | 预训练、SFT、DPO 与 PPO 数据加载器 |
| [trainers](trainers/) | 各训练阶段的入口与通用训练能力 |
| [eval](eval/) | Tokenizer 和模型评估 |
| [scripts](scripts/) | 语料抽取、蒸馏数据生成、数据验证和 LoRA 合并 |
| [images](images/) | 项目和训练阶段示意图 |

## 🧪 实验建议

- 先用小数据集和少量训练步数验证整条链路，再扩大实验。
- 保存每次实验的数据版本、Tokenizer 指纹、随机种子和完整参数。
- 检查点用于精确续训，导出模型用于评估、生成和下游训练，两者用途不同。
- PPO 和 GRPO 默认需要额外的 Reward Model 资源，应根据显存调整训练规模。
- 小模型的结果对数据质量和超参数很敏感，请优先做可复现的对照实验。

## 🤝 贡献

欢迎通过 Issue 提交问题、建议和实验结果，也欢迎通过 Pull Request 改进代码、文档与数据流程。如果变更会影响模型结构、Tokenizer 或训练行为，请同时说明复现方式和兼容性影响。

## 🙏 致谢

miniLLM 的构建离不开开源社区的知识与实践。特别感谢以下项目：

- [MiniMind](https://github.com/jingyaogong/minimind) —— 为轻量级语言模型的全流程训练、数据处理、偏好对齐与工程组织提供了重要参考。
- [nanoGPT](https://github.com/karpathy/nanoGPT) —— 以简洁、透明的方式展示 GPT 训练与微调，持续启发了“通过可读代码理解模型”的项目理念。

同时感谢 PyTorch、Hugging Face Transformers、Hugging Face Datasets 以及所有开源数据和研究成果的贡献者。

---

<div align="center">

如果 miniLLM 对你有帮助，欢迎点亮 ⭐，也欢迎一起让它更容易理解、复现和扩展。

</div>

<div align="center">

# miniLLM-vlm

### 在 miniLLM Base 上训练视觉语言模型

中文 | [English](README_en.md)

[返回 miniLLM 主项目](../README.md)

🚀 [**在线体验已训练的 miniLLM-vlm**](https://www.modelscope.cn/studios/kayson2026/miniLLM-vlm)

</div>

---

## 🎯 项目介绍

miniLLM-vlm 是 [miniLLM](../README.md) 的视觉语言扩展。项目复用已训练的 miniLLM Base 和 SigLIP 视觉编码器，通过一个轻量 Projector 对齐图像特征与语言嵌入，让原本只能处理文本的 miniLLM 具备单图理解、图像描述、视觉问答和多轮图文对话能力。

项目覆盖从资源预检、数据索引、视觉 Pretrain、视觉 SFT、断点续训到评估和部署的完整链路。模型已完成训练并部署到 ModelScope，可直接上传图片体验。

## ✨ 主要能力

- 基于 miniLLM Base 增加视觉理解能力，保留原有文本生成路径
- 使用 SigLIP 提取图像 Patch 特征
- 使用两层 MLP Projector 完成视觉与语言空间对齐
- 支持单图描述、单图问答与多轮图文对话
- 支持图文样本与纯文本指令混合 SFT
- 不扩充 Tokenizer 词表，复用保留 Token 表示图像位置
- 支持资源指纹、数据索引、完整检查点和确定性续训
- 支持 SwanLab 与 Weights & Biases 训练监控
- 支持图文损失、分组验证与错配图像对照评估

## 🧠 模型架构

miniLLM-vlm 采用模块化的输入级多模态融合方案。图像先由 SigLIP 编码，视觉特征经 Projector 转换后，直接替换文本序列中的图像占位嵌入，再与问题文本一起输入 miniLLM。

| 组件 | 配置与作用 |
| --- | --- |
| 语言模型 | miniLLM Base，负责跨模态理解和自回归文本生成 |
| 视觉编码器 | SigLIP Vision Model，在训练中保持冻结 |
| 图像分辨率 | 256 × 256 |
| Patch 大小 | 32 × 32 |
| 视觉 Token 数 | 64 |
| 视觉特征维度 | 768 |
| Projector | LayerNorm → Linear → GELU → Linear |
| 图像占位 | 复用 miniLLM Tokenizer 中的保留 Token，不改变词表 |
| 融合方式 | 将连续图像占位的 Embedding 替换为映射后的视觉特征 |

这种设计的重点不是从零重新训练视觉和语言主干，而是学习两个已有表示空间之间的稳定连接。生成时图像只在 Prefill 阶段编码一次，后续 Token 生成复用 miniLLM 的 KV Cache。

## 🗺️ 两阶段训练

| 阶段 | 训练数据 | 默认可训练参数 | 目标 |
| --- | --- | --- | --- |
| Visual Pretrain | 单图描述 | Projector | 建立图像特征与语言嵌入的基础对齐 |
| Visual SFT | 单图问答、多轮图文对话和纯文本指令 | Projector 与 miniLLM 首尾 Decoder Block | 学习视觉指令跟随和多轮对话 |

### 🚀 Visual Pretrain

Visual Pretrain 从已训练的 miniLLM Base 和 SigLIP 开始，同时冻结两个主干，只更新 Projector。虽然 miniLLM 参数不更新，答案误差仍会穿过语言模型传回 Projector，让映射后的视觉特征逐步变成 miniLLM 能够理解的输入。

默认配置使用 BF16、1 个 Epoch、最长 512 Token，Micro Batch 为 8，累积 8 次后更新，有效 Batch 为 64。训练损失由图片描述交叉熵和 miniLLM MoE Router 辅助项组成。

![miniLLM-vlm Visual Pretrain 训练曲线](images/pretrain.png)

*Visual Pretrain 训练过程中的语言模型损失、Router 辅助损失、总损失、梯度范数、监督 Token 与样本数。*

### 💬 Visual SFT

Visual SFT 继承 Pretrain 学到的 Projector，进一步使用图像、问题和理想回答学习具体任务行为。默认策略冻结 SigLIP 和 miniLLM 中间层，训练 Projector 以及语言模型第一层和最后一层完整 Decoder Block。

项目也保留全量语言模型微调和仅 Projector 训练两种对照策略。默认 SFT 使用 BF16、1 个 Epoch、最长 768 Token，Micro Batch 为 4，累积 16 次后更新，有效 Batch 同样为 64。

![miniLLM-vlm Visual SFT 训练曲线](images/sft.png)

*Visual SFT 训练过程中的语言模型损失、Router 辅助损失、总损失、整体梯度、Projector 梯度与语言模型梯度。*

## 💾 数据与监督目标

miniLLM-vlm 使用 Parquet 保存图像与对话。训练时的序列长度预算包含视觉 Token、问题、对话历史和答案。

| 字段 | 用途 |
| --- | --- |
| image_bytes | 原始图像字节，纯文本样本可不使用 |
| conversations | system、user 和 assistant 组成的单轮或多轮对话 |
| task_type | 可选任务标记，用于区分 instruction、caption 和 text |

图像与问题都是可读上下文，只有 assistant 的正文回答和真实结束 Token 参与语言模型监督。图像位置、system / user 文本和 Padding 不计入答案损失。

数据划分基于图像字节指纹，默认使用固定随机种子和约 1% 验证集。相同图像的不同语言描述不会跨越训练集和验证集，减少图像泄漏造成的虚高结果。

## 📊 评估方法

项目的评估入口同时支持 Pretrain Adapter 和 SFT 完整模型，包含以下能力：

- 单张图片描述与视觉问答
- 以 user 消息结尾的多轮图文上下文
- SFT 模型的纯文本推理
- 按有效答案 Token 加权的验证交叉熵
- 按 caption、instruction、image 和 text 分组的验证指标
- 正确图像与错配图像的成对损失比较
- 固定验证问题的周期性生成记录

错配图对照不是独立的综合视觉评测，但它能够辅助检查模型是否真正利用了图像信息，而不是只依赖文本提示生成回答。

## 🛡️ 可复现与训练安全

- 训练前预检会验证 Base、Tokenizer、SigLIP、图像特征形状和参数冻结状态
- 数据索引会记录无效图像、非法对话、长度分布和截断情况
- 检查点保存模型、优化器、调度器、混合精度状态、随机状态和训练进度
- 恢复训练时严格校验模型、Tokenizer、视觉编码器、数据、代码和关键参数的身份
- 训练产物与可续训检查点分开，避免把裸权重或不完整文件当作可恢复状态
- 所有模型资源默认从本地加载，训练期间不自动下载或替换文件

## 📦 模型产物

| 产物 | 用途 |
| --- | --- |
| Pretrain best_adapter.pt / last_adapter.pt | Projector 权重、VLM 配置和资源身份，后续使用时仍需 Base 与 SigLIP |
| SFT best_sft.pt / last_sft.pt | 完整更新后的 miniLLM 权重、Projector、配置与资源身份 |
| tokenizer 目录 | 与 SFT 模型匹配的 Tokenizer 文件 |
| latest.pt / best.pt | 包含优化器和训练进度的完整可恢复检查点 |
| metrics.jsonl | 本地训练与验证数值记录 |
| generations.jsonl | SFT 固定问题的参考答案和生成记录 |

## 🚀 在线体验

已训练并部署的 miniLLM-vlm 可在 ModelScope 创空间直接体验。可以上传一张图片并输入问题，检查模型的图像理解与回答效果。

### [打开 miniLLM-vlm 在线体验](https://www.modelscope.cn/studios/kayson2026/miniLLM-vlm)

## 📚 学习资料

仓库提供从 VLM 基础认知、模型架构到 Pretrain 和 SFT 实战的中英文学习路径。

### 中文

- [VLM 构建方法](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/%E8%AE%A4%E7%9F%A5%E7%AF%87/VLM%E6%9E%84%E5%BB%BA%E6%96%B9%E6%B3%95.md)
- [VLM 架构与训练方法](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/%E6%9E%B6%E6%9E%84%E7%AF%87/VLM%E6%9E%B6%E6%9E%84%E4%B8%8E%E8%AE%AD%E7%BB%83%E6%96%B9%E6%B3%95.md)
- [miniLLM-vlm Pretrain 训练](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/%E5%AE%9E%E6%88%98%E7%AF%87/miniLLM-vlm%20Pretrain%E8%AE%AD%E7%BB%83.md)
- [miniLLM-vlm SFT 训练](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/%E5%AE%9E%E6%88%98%E7%AF%87/miniLLM-vlm%20SFT%E8%AE%AD%E7%BB%83.md)

### English

- [Building Vision-Language Models](learning-materials-en/concepts/Building-Vision-Language-Models.md)
- [VLM Architecture and Training](learning-materials-en/architecture/VLM-Architecture-and-Training.md)
- [miniLLM-vlm Pretraining](learning-materials-en/practice/miniLLM-vlm-Pretraining.md)
- [miniLLM-vlm SFT](learning-materials-en/practice/miniLLM-vlm-SFT.md)

## 🗂️ 项目导航

| 目录或文件 | 内容 |
| --- | --- |
| [model](model/) | miniLLM Base 架构、SigLIP 资源和 VLM Projector |
| [dataset](dataset/) | Pretrain / SFT 数据编码、索引、划分与 Collator |
| [trainer](trainer/) | Visual Pretrain、Visual SFT 和通用训练能力 |
| [eval](eval/) | 图像描述、视觉问答、纯文本推理和验证集评估 |
| [images](images/) | Pretrain 和 SFT 训练曲线 |
| [中文学习资料](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/) | 中文原理与实战文章 |
| [English Learning Materials](learning-materials-en/) | English concepts, architecture, and practical guides |
| [requirements.txt](requirements.txt) | Python 依赖范围 |

## ⚙️ 运行边界

- 当前实现面向单机单卡实验，训练配置针对 AutoDL 单张 RTX 4090 进行了设计
- PyTorch 需与当前 CUDA 环境单独匹配，其他依赖见 [requirements.txt](requirements.txt)
- 当前版本不包含多进程 DDP、LoRA、量化训练和编译优化
- 项目面向单图理解，不是通用多图、视频或图像生成模型
- 训练结果依赖数据质量、图文覆盖、训练资源和评估标准，在线 Demo 不代表全面的通用视觉能力

## 🙏 致谢

特别感谢 [MiniMind-V](https://github.com/jingyaogong/minimind-v)，为在轻量语言模型上接入视觉编码器、进行跨模态对齐和视觉 SFT 提供了重要参考。

同时感谢 PyTorch、Hugging Face Transformers、Hugging Face Datasets、SigLIP 及所有开源数据与研究成果的贡献者。

---

<div align="center">

如果 miniLLM-vlm 对你有帮助，欢迎点亮 ⭐，也欢迎一起改进这个小型视觉语言模型。

</div>

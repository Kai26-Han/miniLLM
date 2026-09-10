<div align="center">

![miniLLM](images/minillm-banner.png)


### Build and train a small language model from scratch

[中文](README.md) | English

🚀 [**Try the miniLLM model online**](https://www.modelscope.cn/studios/kayson2026/miniLLM)

👁️ [**Try the miniLLM-vlm vision model online**](https://www.modelscope.cn/studios/kayson2026/miniLLM-vlm)

</div>

---

## 🎯 Vision

miniLLM is a language-model project for learning, research, and hands-on experimentation.

The project aims to turn language models from black boxes that can merely be called into engineering systems that can be understood layer by layer. It starts with raw corpora and tokenization, then progresses through pretraining, instruction tuning, preference alignment, reinforcement learning, tool use, and visual extensions, covering the main lifecycles of modern language and multimodal models.

miniLLM pairs its implementation with Chinese and English theory articles and practical guides. Every training stage is intended to be readable, reproducible, comparable, and extensible.

## ✨ What Is Included

- An 8,192-token ByteLevel-BPE tokenizer that can be trained from scratch
- A decoder-only Transformer built with RMSNorm, RoPE, GQA, and SwiGLU
- Dense and sparse MoE variants, with 4-expert Top-1 routing as the default
- miniLLM-vlm, a visual extension built from the miniLLM Base model and a SigLIP vision encoder
- Pretraining, full-parameter SFT, LoRA, and offline black-box distillation
- DPO, PPO, GRPO, and multi-turn Agentic RL
- Checkpoint resume, mixed precision, gradient accumulation, gradient checkpointing, and compilation
- SwanLab and Weights & Biases experiment tracking
- Evaluation for the tokenizer, pretrained model, SFT, DPO, and PPO stages
- Paired Chinese and English LLM learning materials

## 🧠 Model Architecture

miniLLM is an autoregressive decoder-only causal language model. Sparse MoE is enabled by default, while a dense variant is retained for controlled comparisons.

| Configuration | Default |
| --- | ---: |
| Vocabulary size | 8,192 |
| Hidden size | 768 |
| Decoder layers | 8 |
| Query heads / KV heads | 8 / 4 |
| Head dimension | 96 |
| FFN / expert intermediate size | 2,432 |
| Maximum position configuration | 32,768 |
| Normalization | RMSNorm |
| Position encoding | RoPE |
| Attention | Grouped Query Attention |
| Feed-forward network | SwiGLU |
| Default routing | 4 experts / Top-1 |
| Dense parameters | About 65.3M |
| MoE total / active parameters per token | About 199.8M / 65.3M |

The maximum position configuration only defines the position-index range accepted by the model. Reliable long-context performance still depends on the sequence lengths, data distribution, and evaluation results used during training.

## 🗺️ End-to-End Training Path

| Stage | Goal | Main output |
| --- | --- | --- |
| Tokenizer | Map text to stable token IDs | ByteLevel-BPE vocabulary and chat template |
| Pretraining | Learn language distributions through next-token prediction | Base language model |
| SFT / LoRA / Distillation | Learn instruction following and conversation | Instruction model or LoRA adapter |
| VLM Pretrain / SFT | Align visual and language representations and learn image understanding and visual dialogue | miniLLM-vlm vision-language model |
| DPO | Learn preferred responses from comparison pairs | Preference-aligned model |
| PPO / GRPO | Optimize the policy with reward signals | Reinforcement-learning model |
| Agentic RL | Learn tool use in multi-turn environments | Agentic policy model |

## 🔤 Tokenizer

The tokenizer determines how text enters the model. miniLLM provides ByteLevel-BPE training, corpus cleaning, vocabulary validation, special-token constraints, and a chat template. A ready-to-use tokenizer is included, while the complete training path remains available for custom corpora.

![miniLLM Tokenizer](images/tokenizer.png)

Once pretraining begins, the vocabulary mapping and special-token IDs must remain unchanged. Otherwise, the model's embedding and output layers will no longer correspond to the tokenizer.

## 🚀 Pretraining

Pretraining starts from random weights and uses causal language modeling to learn lexical, syntactic, semantic, and factual relationships from text. Dense and MoE models share the same core architecture. In the MoE variant, a router selects an expert for each token, while an auxiliary objective helps reduce routing collapse.

![miniLLM Pretraining](images/pretrain.png)

The training lifecycle includes validation, learning-rate scheduling, complete checkpoints, distributed execution, experiment tracking, and Hugging Face-compatible export.

## 💬 Instruction Tuning

Supervised fine-tuning teaches the base model to interpret user requests and produce structured responses. The miniLLM data pipeline keeps system, user, and tool messages as context while applying the language-model objective only to assistant output spans.

![miniLLM SFT](images/sft.png)

In addition to full-parameter SFT, the project supports domain adaptation with LoRA and offline black-box sequence distillation, making it possible to compare different capability-transfer strategies.

## 🏆 Preference Alignment and Reinforcement Learning

DPO learns human or AI preferences from chosen and rejected response pairs. Building on an aligned policy, PPO and GRPO support online generation and policy optimization with an external reward model.

![miniLLM PPO](images/PPO.png)

The PPO path includes the actor, critic, reward, GAE, and clipped objectives, providing a complete route for studying the RLHF training lifecycle.

![miniLLM GRPO](images/GRPO.png)

GRPO constructs relative advantages from multiple candidate answers to the same prompt without requiring a separate critic. The project also extends this approach to multi-turn Agentic RL for tool use and environment interaction.

## 👁️ miniLLM-vlm Vision-Language Model

[miniLLM-vlm](miniLLM-vlm/) extends the trained miniLLM Base model with visual understanding. It encodes 256 × 256 images into 64 visual tokens with SigLIP, then maps those features into the miniLLM language-embedding space through a Projector composed of LayerNorm, Linear, GELU, and Linear layers. This design reuses existing visual and language capabilities while concentrating learning on the connection between the two modalities.

The implementation is a self-contained subproject with its own model definitions, data processing, trainers, evaluation entry point, and learning materials in both Chinese and English. It targets single-image understanding and reuses reserved tokens as a contiguous image placeholder instead of expanding the tokenizer vocabulary.

### Visual Pretraining

Visual Pretraining uses single-image caption data to establish basic image-text alignment. Both the SigLIP vision encoder and miniLLM Base model remain frozen while only the Projector is trained. Answer errors still backpropagate through the language model to the Projector, gradually adapting mapped visual features to the language model's input space.

![miniLLM-vlm Pretraining curves](miniLLM-vlm/images/pretrain.png)

### Visual SFT

Visual SFT continues from the Pretraining output and covers single-image question answering, multi-turn image-text conversations, and text-only instructions. By default, SigLIP remains frozen while the Projector and the first and last miniLLM decoder blocks are trained, balancing visual instruction following with the language capabilities of the Base model.

![miniLLM-vlm SFT curves](miniLLM-vlm/images/sft.png)

The evaluation path supports image captioning, visual question answering, multi-turn messages, and text-only inference. It also compares the losses from correctly paired and mismatched images as an additional signal for whether the model is actually using visual information.

🚀 [**Try miniLLM-vlm on ModelScope**](https://www.modelscope.cn/studios/kayson2026/miniLLM-vlm)

- [miniLLM-vlm Project Guide](miniLLM-vlm/README.md)
- [miniLLM-vlm Chinese Learning Materials](miniLLM-vlm/%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/)
- [miniLLM-vlm English Learning Materials](miniLLM-vlm/learning-materials-en/)

## 📚 Data and Learning Materials

Large training datasets and model weights are not bundled with the repository. Data should be prepared under the relevant dataset directory in the format expected by each stage: text JSONL for language pretraining, multi-turn conversations for SFT, preference pairs for DPO, prompt conversations awaiting generated answers for PPO and GRPO, and Parquet image-conversation data for the VLM.

📦 **Dataset downloads:**

- [ModelScope · miniLLM-dataset](https://www.modelscope.cn/datasets/kayson2026/miniLLM-dataset/files)
- [Hugging Face · miniLLM-dataset](https://huggingface.co/datasets/fenglike/miniLLM-dataset/tree/main)

If this is your first time studying language models, the recommended order is:

1. Build a high-level understanding of LLMs and the full lifecycle.
2. Study tokenization, Transformers, and self-attention.
3. Learn the differences between dense and MoE models and their optimization methods.
4. Move on to pretraining, SFT, and reinforcement-learning algorithms.
5. Use the hands-on guides to complete experiments for each stage.
6. Continue from the miniLLM Base model to study vision-language alignment and visual SFT.

- [Chinese Learning Materials](%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99%20-%20%E4%BB%8E0%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BALLM/)
- [English Learning Materials](Learning%20Materials%20-%20Building%20an%20LLM%20from%20Scratch/README.md)

## 🗂️ Repository Guide

| Directory | Contents |
| --- | --- |
| [model](model/) | miniLLM model, LoRA, and tokenizer |
| [dataset](dataset/) | Pretraining, SFT, DPO, and PPO dataset loaders |
| [trainers](trainers/) | Entry points and shared utilities for every training stage |
| [eval](eval/) | Tokenizer and model evaluation |
| [scripts](scripts/) | Corpus extraction, distillation data, validation, and LoRA merging |
| [images](images/) | Project and training-stage illustrations |
| [miniLLM-vlm](miniLLM-vlm/) | Vision-language model, training, evaluation, and learning materials built on miniLLM Base |

## 🧪 Experiment Guidance

- Validate the complete pipeline with a small dataset and a short run before scaling up.
- Record the dataset version, tokenizer fingerprint, random seed, and full configuration for every experiment.
- Checkpoints are intended for exact resume; exported models are intended for evaluation, generation, and downstream training.
- PPO and GRPO require additional resources for an external reward model, so training scale should match available memory.
- Small-model results are highly sensitive to data quality and hyperparameters; prioritize reproducible controlled comparisons.

## 🙏 Acknowledgements

Special thanks go to:

- [MiniMind](https://github.com/jingyaogong/minimind) — an important reference for end-to-end lightweight language-model training, data processing, preference alignment, and project organization.
- [MiniMind-V](https://github.com/jingyaogong/minimind-v) — an important reference for connecting a vision encoder to a lightweight language model, cross-modal alignment, and visual SFT.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — a clear and concise demonstration of GPT training and fine-tuning that continues to inspire the idea of understanding models through readable code.

We also thank the contributors to PyTorch, Hugging Face Transformers, Hugging Face Datasets, open datasets, and the broader research community.

---

<div align="center">

If miniLLM helps you, consider leaving a ⭐ and helping make the project easier to understand, reproduce, and extend.

</div>

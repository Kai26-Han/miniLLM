# miniLLM LoRA 训练

> 阶段：在完成通用 SFT 的 miniLLM 上进行参数高效领域微调

> 默认方案：Attention LoRA、Rank 16、Alpha 32、Dropout 0.05、Assistant-only Loss

---

## 1. LoRA 训练要解决什么

全参数 SFT 负责让预训练模型学会理解指令、进行对话和按 ChatML 格式输出。
LoRA 则在不修改 SFT 基座参数的前提下，让模型适应医疗、法律、客服或特定
身份等垂直场景。

推荐训练顺序是：

```text
预训练模型 out/pretrain
        ↓ 通用全参数 SFT
通用对话模型 out/sft
        ↓ 领域 LoRA
out/lora/medical
```

不建议把医疗 LoRA 直接作为预训练模型的最终微调阶段。否则有限的 LoRA
参数需要同时学习“如何遵循指令”和“如何回答医疗问题”，更容易出现对话
格式不稳定或通用能力不足。直接使用 `out/pretrain` 只适合验证 LoRA 管线。

## 2. LoRA 的基本原理

对于原来的线性层权重 `W`，LoRA 不直接更新 `W`，而是增加两个低秩矩阵
`A` 和 `B`：

```text
y = W x + (alpha / rank) × B A x
```

训练时：

- `W` 和其他 SFT 参数全部冻结；
- 只更新 `A` 和 `B`；
- `A` 使用 Kaiming 初始化，`B` 初始化为零；
- 注入 LoRA 的初始时刻，模型输出与原始 SFT 模型完全一致；
- 不加载 LoRA 时，原始 SFT 模型不会受到任何影响。

本项目采用手写 `LoRALinear`，不依赖 PEFT。实现方式保留原有 Linear 的
权重键名，可以单独保存 Adapter，也可以把低秩增量合并回完整模型。

## 3. 当前注入方案

默认 `attention` 方案注入每层 Attention 的：

```text
q_proj、k_proj、v_proj、o_proj
```

当前 8 层 miniLLM 在 Rank 16 下新增 688,128 个可训练参数，约占约 2 亿参数
基座的 0.34%。GQA 中的 `k_proj` 和 `v_proj` 是矩形 Linear，因此不能只匹配
输入输出维度相同的层。

如果领域验证集明显欠拟合，可以改用 `all-linear`：

```text
Attention: q_proj、k_proj、v_proj、o_proj
MoE Expert: gate_proj、up_proj、down_proj
```

该方案会为每个 Expert 注入 LoRA，Rank 16 时共有 5,603,328 个 LoRA 参数。
MoE Router 的 `gate` 仍保持冻结，Router 辅助损失继续参与总 Loss 和日志。

建议先使用 `attention` 建立低成本基线，再根据验证结果决定是否扩大范围。

## 4. 相关文件

| 文件 | 作用 |
|---|---|
| [`model/model_lora.py`](../model/model_lora.py) | LoRA 注入、冻结、Adapter 保存/加载、基模指纹和合并 |
| [`trainer/train_lora.py`](../trainer/train_lora.py) | 单卡/DDP 训练、验证、断点、日志和 Adapter 导出 |
| [`dataset/sft_dataset.py`](../dataset/sft_dataset.py) | 对话渲染、Assistant Loss Mask、截断和动态 Padding |
| [`eval/eval_sft.py`](../eval/eval_sft.py) | 基模与 LoRA 的 Loss/PPL 和生成效果对比 |
| [`scripts/merge_lora.py`](../scripts/merge_lora.py) | 将基模和 Adapter 合并为标准 Transformers 模型 |
| [`tests/test_lora.py`](../tests/test_lora.py) | 注入、冻结、保存加载、MoE 目标和合并一致性测试 |

## 5. 数据格式

LoRA 复用 SFT 数据管线，每行必须包含 `conversations` 数组：

```json
{"conversations":[
  {"role":"user","content":"请解释高血压常见的危险因素。"},
  {"role":"assistant","content":"高血压的常见危险因素包括……"}
]}
```

训练时完整对话都会输入模型，但只有 Assistant 输出参与交叉熵。医疗数据位于：

```text
dataset/lora/lora_medical.jsonl
```

当前文件约有 25,276 条样本。默认划出 2% 作为验证集，比通用 SFT 的
`0.001` 更适合这一数据规模。正式使用前仍应检查：

- 是否存在重复或高度相似的问答；
- 回答是否包含错误诊断和过时知识；
- 是否泄露隐私或个人身份信息；
- 高风险问题是否给出合理的就医提示；
- 训练集和人工测试集是否存在答案泄漏。

## 6. 开始训练

先确认全参数 SFT 已经导出：

```text
out/sft/
├── config.json
├── model.safetensors
└── tokenizer files...
```

正式训练医疗 Adapter：

```bash
python trainer/train_lora.py \
    --model-path out/sft \
    --data-path dataset/lora/lora_medical.jsonl \
    --adapter-name medical \
    --target-preset attention \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --epochs 3 \
    --batch-size 4 \
    --accumulation-steps 16 \
    --learning-rate 1e-4 \
    --min-learning-rate 1e-5 \
    --max-seq-len 1024 \
    --val-ratio 0.02 \
    --dtype bfloat16 \
    --tracker swanlab
```

默认单卡有效全局 Batch Size：

```text
4 × 16 × 1 = 64 条序列/更新
```

双卡训练：

```bash
torchrun --nproc_per_node 2 trainer/train_lora.py \
    --model-path out/sft \
    --adapter-name medical \
    --batch-size 4 \
    --accumulation-steps 8 \
    --dtype bfloat16
```

如果 `attention` 验证结果欠拟合，再实验：

```bash
python trainer/train_lora.py \
    --model-path out/sft \
    --adapter-name medical_all_linear \
    --target-preset all-linear
```

不要让两个不同注入方案共用同一个 `adapter-name` 和断点目录。

## 7. 先做冒烟训练

正式训练前先用少量数据验证注入、Loss、保存和恢复链路：

```bash
python trainer/train_lora.py \
    --model-path out/sft \
    --data-path dataset/lora/lora_medical.jsonl \
    --adapter-name medical_smoke \
    --max-train-samples 64 \
    --eval-samples 16 \
    --max-steps 10 \
    --batch-size 2 \
    --accumulation-steps 2 \
    --num-workers 0 \
    --dtype float32 \
    --tracker none
```

如果 `out/sft` 尚未生成，只为了检查管线可以显式改成：

```bash
--model-path out/pretrain
```

但该结果不应作为最终医疗模型。

启动日志应确认：

- `Frozen base` 指向正确的 `out/sft`；
- `LoRA layers` 与目标方案一致；
- 默认可训练参数为 688,128；
- 只有很小比例参数参与训练；
- Loss 和梯度范数均为有限值；
- `checkpoints/lora/medical_smoke/latest.pt` 正常生成。

## 8. Adapter 和断点

训练过程不会反复保存约 763 MiB 的完整基模。最终目录是：

```text
out/lora/medical/
├── adapter_config.json
├── adapter_model.safetensors
├── training_summary.json
└── tokenizer files...

checkpoints/lora/medical/
└── latest.pt
```

`adapter_model.safetensors` 只包含 LoRA A/B。`latest.pt` 除 Adapter 外还保存
Optimizer、Scheduler、GradScaler、训练位置、随机数状态和 Tracker 信息。

自动恢复：

```bash
python trainer/train_lora.py --adapter-name medical --resume
```

恢复指定断点：

```bash
python trainer/train_lora.py \
    --adapter-name medical \
    --resume checkpoints/lora/medical/latest.pt
```

断点会校验 Rank、Alpha、Dropout、目标模块和基模 SHA-256。不能把一个
Adapter 的断点加载到不同 SFT 基模或不同 LoRA 配置上。

## 9. 对比评估

在同一验证集上依次评估原始 SFT 和 SFT + LoRA：

```bash
python eval/eval_sft.py \
    --model-path out/sft \
    --adapter-path out/lora/medical \
    --compare-base \
    --data-path dataset/lora/lora_medical.jsonl \
    --val-ratio 0.02 \
    --eval-samples 512 \
    --mode both
```

除了医疗验证 Loss，还需要建立一组固定的通用问题，比较：

- 翻译、写作、常识和代码能力是否退化；
- 普通问题是否被过度医疗化；
- 回答是否变得机械、冗长或过度自信；
- 医疗领域回答是否真的更准确，而不只是术语更多。

如果医疗能力上升但通用能力明显下降，可以减少训练轮数、降低学习率，或在
LoRA 数据中混入约 5%～15% 的高质量通用 SFT 样本。

## 10. 合并为完整模型

需要独立部署时，可以把 LoRA 增量合并回基模：

```bash
python scripts/merge_lora.py \
    --base-model-path out/sft \
    --adapter-path out/lora/medical \
    --output-dir out/lora_medical_merged \
    --dtype float16
```

脚本会：

1. 校验 Adapter 记录的基模 SHA-256；
2. 加载基模和 Adapter；
3. 计算 `W = W + (alpha / rank) × BA`；
4. 比较合并前后的 Logits；
5. 移除 LoRA 模块；
6. 导出标准 Transformers 模型目录。

合并输出必须使用新目录，不能覆盖 `out/sft` 或 Adapter。即使完成合并，也应
继续保留原始 SFT 模型和独立 Adapter，方便回滚和后续实验。

## 11. 运行测试

在已经安装正确 PyTorch/CUDA 的训练环境中执行：

```bash
python -m unittest tests.test_lora -v
```

测试覆盖：

- B 为零时注入前后 Logits 一致；
- 只有 LoRA 参数允许计算梯度；
- Adapter 保存再加载后输出一致；
- 合并前后输出一致；
- `all-linear` 覆盖所有 MoE Expert，但不修改 Router；
- 拼错目标模块时立即报错，避免静默少训练某些层。

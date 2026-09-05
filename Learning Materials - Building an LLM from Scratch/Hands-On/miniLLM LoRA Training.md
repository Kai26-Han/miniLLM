# miniLLM LoRA Training

> Stage: parameter-efficient domain adaptation of a miniLLM that has completed general-purpose SFT

> Default recipe: attention LoRA, rank 16, alpha 32, dropout 0.05, and assistant-only loss

---

## 1. What LoRA Training Is Designed to Solve

Full-parameter SFT teaches a pretrained model to understand instructions, hold conversations, and produce output in ChatML format. LoRA then adapts the model to vertical domains—such as medicine, law, customer support, or a particular persona—without changing the SFT base model's parameters.

The recommended training order is:

```text
Pretrained model: out/pretrain
        ↓ general-purpose full-parameter SFT
General chat model: out/sft
        ↓ domain LoRA
out/lora/medical
```

Using medical LoRA directly as the final fine-tuning stage of the pretrained model is not recommended. A limited set of LoRA parameters would then have to learn both how to follow instructions and how to answer medical questions, making chat formatting less stable and general capabilities weaker. Using `out/pretrain` directly is appropriate only for validating the LoRA pipeline.

## 2. LoRA Fundamentals

For an original linear-layer weight `W`, LoRA leaves `W` unchanged and introduces two low-rank matrices, `A` and `B`:

```text
y = W x + (alpha / rank) × B A x
```

During training:

- `W` and all other SFT parameters are frozen.
- Only `A` and `B` are updated.
- `A` uses Kaiming initialization, while `B` is initialized to zero.
- Immediately after LoRA is injected, the model's output is exactly the same as that of the original SFT model.
- The original SFT model is unaffected when LoRA is not loaded.

This project uses a custom `LoRALinear` implementation and does not depend on PEFT. It preserves the original linear layers' weight-key names, allowing the adapter to be saved separately or the low-rank update to be merged back into the full model.

## 3. Current Injection Strategy

The default `attention` preset injects LoRA into the following attention projections in every layer:

```text
q_proj, k_proj, v_proj, o_proj
```

For the current eight-layer miniLLM at rank 16, this adds 688,128 trainable parameters—about 0.34% of the roughly 200-million-parameter base model. In GQA, `k_proj` and `v_proj` are rectangular linear layers, so target selection must not be restricted to layers whose input and output dimensions match.

If the domain validation set is clearly underfit, try `all-linear`:

```text
attention: q_proj, k_proj, v_proj, o_proj
MoE experts: gate_proj, up_proj, down_proj
```

This preset injects LoRA into every expert and contains 5,603,328 LoRA parameters at rank 16. The MoE router's `gate` remains frozen, while the router auxiliary loss continues to contribute to the total loss and logs.

Start with `attention` as a low-cost baseline, then expand the target set only if validation results justify it.

## 4. Relevant Files

| File | Purpose |
|---|---|
| [`model/model_lora.py`](../../model/model_lora.py) | LoRA injection and freezing, adapter save/load, base-model fingerprinting, and merging |
| [`trainer/train_lora.py`](../../trainer/train_lora.py) | Single-GPU/DDP training, validation, checkpoints, logging, and adapter export |
| [`dataset/sft_dataset.py`](../../dataset/sft_dataset.py) | Conversation rendering, assistant loss masks, truncation, and dynamic padding |
| [`eval/eval_sft.py`](../../eval/eval_sft.py) | Loss/perplexity and generation comparisons between the base model and LoRA model |
| [`scripts/merge_lora.py`](../../scripts/merge_lora.py) | Merges the base model and adapter into a standard Transformers model |
| [`tests/test_lora.py`](../../tests/test_lora.py) | Tests injection, freezing, save/load, MoE targets, and merge equivalence |

## 5. Data Format

LoRA reuses the SFT data pipeline. Every line must contain a `conversations` array:

```json
{"conversations":[
  {"role":"user","content":"Please explain the common risk factors for hypertension."},
  {"role":"assistant","content":"Common risk factors for hypertension include…"}
]}
```

The full conversation is fed to the model during training, but only assistant output contributes to cross-entropy. The medical dataset is located at:

```text
dataset/lora/lora_medical.jsonl
```

The current file contains about 25,276 examples. By default, 2% is held out for validation, which is more appropriate for this dataset size than the `0.001` used for general-purpose SFT. Before production use, you should still check:

- whether duplicate or highly similar question-answer pairs exist;
- whether responses contain incorrect diagnoses or outdated knowledge;
- whether private or personally identifiable information is exposed;
- whether high-risk questions include appropriate advice to seek professional care; and
- whether answers from the training set leak into the human-curated test set.

## 6. Start Training

First confirm that full-parameter SFT has been exported:

```text
out/sft/
├── config.json
├── model.safetensors
└── tokenizer files...
```

Train the production medical adapter:

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

The default effective global batch size on one GPU is:

```text
4 × 16 × 1 = 64 sequences per update
```

For two-GPU training:

```bash
torchrun --nproc_per_node 2 trainer/train_lora.py \
    --model-path out/sft \
    --adapter-name medical \
    --batch-size 4 \
    --accumulation-steps 8 \
    --dtype bfloat16
```

If the `attention` preset underfits the validation data, then experiment with:

```bash
python trainer/train_lora.py \
    --model-path out/sft \
    --adapter-name medical_all_linear \
    --target-preset all-linear
```

Do not let two different injection presets share the same `adapter-name` and checkpoint directory.

## 7. Run a Smoke Test First

Before production training, use a small amount of data to verify injection, loss computation, saving, and resume behavior:

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

If `out/sft` has not yet been generated, you can explicitly use the following solely to check the pipeline:

```bash
--model-path out/pretrain
```

That result should not be treated as the final medical model.

The startup log should confirm that:

- `Frozen base` points to the correct `out/sft`;
- `LoRA layers` matches the intended preset;
- the default trainable-parameter count is 688,128;
- only a small fraction of parameters participates in training;
- both loss and gradient norm remain finite; and
- `checkpoints/lora/medical_smoke/latest.pt` is created successfully.

## 8. Adapters and Checkpoints

Training does not repeatedly save the complete, approximately 763 MiB base model. The final directory layout is:

```text
out/lora/medical/
├── adapter_config.json
├── adapter_model.safetensors
├── training_summary.json
└── tokenizer files...

checkpoints/lora/medical/
└── latest.pt
```

`adapter_model.safetensors` contains only LoRA A/B parameters. In addition to the adapter, `latest.pt` stores optimizer, scheduler, GradScaler, training position, random-number state, and tracker information.

Resume automatically:

```bash
python trainer/train_lora.py --adapter-name medical --resume
```

Resume from a specific checkpoint:

```bash
python trainer/train_lora.py \
    --adapter-name medical \
    --resume checkpoints/lora/medical/latest.pt
```

The checkpoint validates rank, alpha, dropout, target modules, and the base model's SHA-256 hash. An adapter checkpoint cannot be loaded onto a different SFT base or a different LoRA configuration.

## 9. Comparative Evaluation

Evaluate the original SFT model and SFT + LoRA on the same validation set:

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

In addition to medical validation loss, create a fixed set of general-purpose questions and compare:

- whether translation, writing, general knowledge, or coding abilities regress;
- whether ordinary questions become over-medicalized;
- whether responses become mechanical, verbose, or overconfident; and
- whether medical responses actually become more accurate rather than merely containing more technical terminology.

If medical performance improves but general capabilities decline substantially, reduce the number of epochs, lower the learning rate, or mix roughly 5–15% high-quality general-purpose SFT examples into the LoRA data.

## 10. Merge into a Full Model

For standalone deployment, merge the LoRA update back into the base model:

```bash
python scripts/merge_lora.py \
    --base-model-path out/sft \
    --adapter-path out/lora/medical \
    --output-dir out/lora_medical_merged \
    --dtype float16
```

The script:

1. Validates the base-model SHA-256 recorded by the adapter.
2. Loads the base model and adapter.
3. Computes `W = W + (alpha / rank) × BA`.
4. Compares logits before and after merging.
5. Removes the LoRA modules.
6. Exports a standard Transformers model directory.

The merged output must use a new directory; it must not overwrite `out/sft` or the adapter. Even after merging, retain both the original SFT model and the standalone adapter to support rollback and future experiments.

## 11. Run the Tests

In a training environment with a correct PyTorch/CUDA installation, run:

```bash
python -m unittest tests.test_lora -v
```

The tests cover:

- identical logits before and after injection when B is zero;
- gradients being enabled only for LoRA parameters;
- identical output after saving and reloading an adapter;
- identical output before and after merging;
- `all-linear` covering every MoE expert without modifying the router; and
- immediate errors for misspelled target-module names, preventing layers from being silently omitted from training.

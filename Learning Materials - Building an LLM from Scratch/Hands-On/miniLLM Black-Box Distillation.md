# miniLLM Black-Box Distillation

> Goal: use `Qwen/Qwen3-1.7B` to generate new assistant responses for the existing SFT prompts, then run distillation-style SFT starting from `out/pretrain`.

## 1. Training Pipeline

```text
dataset/sft/sft.jsonl
        ↓ offline generation with Qwen3-1.7B
dataset/distill/qwen3_1_7b_sft.jsonl
        ↓ data validation
trainer/train_distillation.py + out/pretrain
        ↓
out/sft_distilled
```

This is sequence-level black-box distillation. Qwen and miniLLM use different tokenizers, so the pipeline neither reads Qwen logits nor computes token-level KL loss. The teacher only generates text; during training, miniLLM's 8,192-token tokenizer re-encodes all text.

## 2. Set Up the Environment

```bash
pip install -r requirements.txt
```

On the first run, the script downloads `Qwen/Qwen3-1.7B` from Hugging Face. If the model is already available locally, point `--model-path` to its directory and add `--local-files-only`.

## 3. Start with a Small Sample

Do not process the entire 13 GB SFT file on your first run. Generate 100 examples first:

```bash
python scripts/generate_distill_data.py \
    --input-path dataset/sft/sft.jsonl \
    --output-path dataset/distill/qwen3_1_7b_smoke.jsonl \
    --model-path Qwen/Qwen3-1.7B \
    --distill-ratio 0.7 \
    --thinking-ratio 0.15 \
    --batch-size 2 \
    --max-samples 100
```

Default behavior:

- The teacher regenerates every assistant turn in 70% of conversations that do not contain tool calls.
- The remaining 30% retain their original responses for SFT replay.
- Each assistant turn selected for regeneration has a 15% chance of enabling thinking mode.
- Complete tool-call trajectories are retained by default, preventing a newly generated call from becoming inconsistent with an existing tool result.
- Each output line corresponds to one input line, so the same prompt cannot be split between the training and validation sets.

Resume from the completed JSONL lines after an interruption:

```bash
python scripts/generate_distill_data.py \
    --output-path dataset/distill/qwen3_1_7b_smoke.jsonl \
    --max-samples 100 \
    --resume
```

`--resume` checks the model, ratios, token limit, and random seed recorded in the sidecar manifest. It refuses to append to an old file if any critical setting has changed.

## 4. Validate the Distillation Data

```bash
python scripts/validate_distill_data.py \
    --data-path dataset/distill/qwen3_1_7b_smoke.jsonl \
    --model-path out/pretrain \
    --max-seq-len 1024 \
    --token-sample-every 1
```

Validation covers:

- JSONL syntax and message structure;
- empty assistant responses;
- leaked `reasoning_content` or `<think>` content;
- leaked Qwen chat special tokens;
- JSON parseability of tool fields; and
- sequence lengths and assistant target-token counts produced by the miniLLM tokenizer.

For large datasets, exact token statistics are sampled at roughly 1% by default, while structural checks still run on every line. The report is saved as `*.validation.json` by default.

## 5. Run a Distillation-SFT Smoke Test

```bash
python trainer/train_distillation.py \
    --model-path out/pretrain \
    --data-path dataset/distill/qwen3_1_7b_smoke.jsonl \
    --save-dir checkpoints/sft_distilled_smoke \
    --output-dir out/sft_distilled_smoke \
    --max-train-samples 100 \
    --eval-samples 20 \
    --max-steps 20 \
    --batch-size 2 \
    --accumulation-steps 2 \
    --learning-rate 1e-5 \
    --tracker none
```

In particular, verify that the loss remains finite, the MoE router is balanced, responses are not heavily truncated, and `out/sft_distilled_smoke` can be loaded for generation.

## 6. Generate the Production Dataset

Once the small run passes, switch to the production output path. You can begin with a 10,000–50,000-example proof of concept:

```bash
python scripts/generate_distill_data.py \
    --input-path dataset/sft/sft.jsonl \
    --output-path dataset/distill/qwen3_1_7b_sft.jsonl \
    --model-path Qwen/Qwen3-1.7B \
    --distill-ratio 0.7 \
    --thinking-ratio 0.15 \
    --batch-size 4 \
    --max-samples 50000
```

After validation, increase `--max-samples` and add `--resume` to continue extending the same dataset. Omit `--max-samples` to process the entire dataset.

## 7. Production Training

```bash
python trainer/train_distillation.py \
    --model-path out/pretrain \
    --data-path dataset/distill/qwen3_1_7b_sft.jsonl \
    --save-dir checkpoints/sft_distilled \
    --output-dir out/sft_distilled \
    --batch-size 4 \
    --accumulation-steps 16 \
    --max-seq-len 1024 \
    --learning-rate 1e-5 \
    --epochs 1 \
    --attention-backend eager \
    --tracker swanlab
```

The distillation file already contains 30% replay from the original SFT data. Do not also pass `dataset/sft/sft.jsonl` during production training, or you will change the intended teacher/replay ratio.

`train_distillation.py` is the explicit entry point for black-box distillation. It first checks the data-generation manifest, teacher source, teacher/replay line counts, and generation status, then invokes the existing SFT training loop. The distillation training process does not load Qwen, so GPU memory is used only for student training.

## 8. Recommended Controlled Experiment

Keep two separate outputs:

1. `out/sft_baseline`: trained only on the original SFT data.
2. `out/sft_distilled`: trained for the same number of steps and with the same batch size, but on the mixed distillation data.

Compare assistant loss, Chinese and English question answering, coding, instruction following, output length, pretraining perplexity, and MoE expert load. An improvement can be attributed to the distilled data only under a compute-matched comparison.

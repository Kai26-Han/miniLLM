# miniLLM DPO Training

> Stage: preference alignment after full-parameter SFT

> Default recipe: standard DPO following MiniMind's reference configuration, beta 0.15, learning rate 4e-8, no warmup, BF16, one epoch over the full dataset, and a frozen SFT reference model

---

## 1. What Does DPO Training Solve?

SFT teaches a model how to answer by imitating reference responses, but the same prompt may have several plausible responses of different quality. Direct Preference Optimization (DPO) uses paired preference data to tell the model which response is better:

```text
For the same prompt: chosen is better than rejected
```

DPO can further improve response style, safety, completeness, and instruction following. Unlike classic RLHF, it does not require a separately trained reward model or online sampling during every training iteration. Its core objective increases the policy's relative preference for the chosen response and decreases its relative preference for the rejected response, both measured against a frozen reference model.

The complete training sequence of this project is:

```text
Random initialization
↓ Pretraining
out/pretrain
↓ Full-parameter SFT
out/sft
├── Copy as the trainable policy
└── Copy as the frozen reference model
              ↓
Chosen/rejected preference pairs
              ↓ Standard DPO loss
           out/dpo
```

DPO should start from `out/sft`, which already has stable conversational behavior. Starting directly from `out/pretrain` forces the preference data to teach instruction following and preference alignment at the same time, which rarely produces reliable results.

## 2. The basic principle of DPO

Given the same prompt `x`, preferred answer `y_w` and non-preferred answer `y_l`, calculate the answer sequence of the policy model `πθ` and frozen reference model `πref` respectively log probability:

```text
Log π(y|x) = the sum of all next-token log probability of Answer token
```

The policy model and the reference model respectively form the logarithm probability difference of chosen/rejected:

```text
policy_logratio = log πθ(y_w|x)  - log πθ(y_l|x)
ref_logratio    = log πref(y_w|x) - log πref(y_l|x)
```

The final preference interval and DPO Loss are:

```text
margin   = policy_logratio - ref_logratio
dpo_loss = -(1-ε) log sigmoid(beta × margin)
           - ε log sigmoid(-beta × margin)
```

The default `ε=0` is consistent with MiniMind and uses standard DPO. Only when it has been confirmed that there is noise in the preference label and the Conservative DPO control experiment is ready to be carried out separately, the non-zero `--label-smoothing` is explicitly set.

At present, miniLLM is a MoE model, so the actual reverse propagation target also includes router auxiliary loss:

```text
total_loss = dpo_loss + router_aux_loss
```

The reference model only provides a stable relative benchmark, always maintaining `eval()` and `requires_grad_(False)`. It does not have an optimizer, does not calculate gradients, and does not write to the final DPO model.

The derivation and algorithm details of DPO can be referred to [Direct Preference Optimization Original Paper ](https://arxiv.org/abs/2305.18290). The overall training form of this project refers to MiniMind's DPO implementation, but dynamic Padding, validation set, strict data inspection, checkpoint training and more complete preference indicators are supplemented for the current project.

## 3. Related documents

| File | Function |
|---|---|
| [`dataset/dpo_dataset.py`](../../dataset/dpo_dataset.py) | Preference check, ChatML rendering, final answer Mask, pair truncation and dynamic Padding |
| [`trainer/train_dpo.py`](../../trainer/train_dpo.py) | Policy/Reference Forward, DPO Loss, DDP, Verification, checkpoint and Model Export |
| [`eval/eval_dpo.py`](../../eval/eval_dpo.py) | Independently evaluate DPO Policy and SFT Reference on the same preference validation set |
| [`dataset/rl/dpo.jsonl`](../../dataset/rl/dpo.jsonl) | chosen/rejected preference training data |
| [`model/model_minillm.py`](../../model/model_minillm.py) | miniLLM Causal LM and MoE router Auxiliary Loss |
| [`trainer/trainer_utils.py`](../../trainer/trainer_utils.py) | Distributed, Scheduler, Checkpoint, Tracker and Transformers Export |
| [`tests/test_dpo_dataset.py`](../../tests/test_dpo_dataset.py) | Preference data, Mask, truncated and Collator test |
| [`tests/test_dpo_objective.py`](../../tests/test_dpo_objective.py) | Sequence log probability and DPO Formula Test |

This project uses native PyTorch to implement DPO without relying on TRL.

## 4. Data format

Each line must contain two complete dialogue arrays of `chosen` and `rejected`:

```json
{
  "chosen": [
    {"role": "user", "content": "How to develop stable reading habits?"},
    {"role": "assistant", "content": "You can start with a fixed reading for ten minutes every day..."}
  ],
  "rejected": [
    {"role": "user", "content": "How to develop stable reading habits?"},
    {"role": "assistant", "content": "Just read more books."}
  ]
}
```

Mandatory requirements of data loader:

1. `chosen` and `rejected` both end with `assistant` messages;
2. Except for the last assistant reply, the historical information on both sides must be exactly the same;
3. The final reply contains at least one supervised token after Tokenize;
4. Roles, `reasoning_content`, `tools` and `tool_calls` must comply with the SFT data specifications;
5. Illegal samples will report the original JSONL line number and stop immediately without silently skipping.

Multiple rounds of preference data can also be used:

```text
system → user → assistant → user → chosen assistant
                              └──→ rejected assistant
```

The previous assistant history is only used as a common context and does not participate in the current preference score. Training only responds to the last chosen/rejected assistant to calculate the sequence log probability.

### 4.1 Current data audit

At present, the audit results of [`dataset/rl/dpo.jsonl`](../../dataset/rl/dpo.jsonl) are:

| Project | Result |
|---|---:|
| Quantity of preferred pairs | 17,166 |
| Public prompt is completely consistent | 17,166 |
| Both sides end with assistant | 17,166 |
| Current dialogue structure | All are user → assistant |
| chosen average answer token | 358.97 |
| rejected average answer token | 332.90 |

When using the current tokenizer and `max_seq_len=1024` full coding:

| Project | Quantity | Proportion |
|---|---:|---:|
| chosen sequence truncated | 1,158 | 6.75% |
| rejected sequence truncated | 1,158 | 6.75% |
| chosen The answer itself is truncated | 578 | 3.37% |
| rejected The answer itself is truncated | 623 | 3.63% |

If the GPU memory allows, `--max-seq-len 1536` or `2048` can be experimented, but the increase in sequence length will significantly increase the compute and memory occupancy of the two forwards of Policy and Reference.

### 4.2 Truncation in pairs

DPO emphasizes the consistency of chosen/rejected conditions more than ordinary SFT. Both sides cannot independently crop prompt at will, otherwise they will no longer represent the probability comparison of the same condition.

The current strategy is:

1. First, confirm that the token prefix before the final reply on both sides is exactly the same;
2. The short answer is kept intact first, and the remaining windows are assigned to the nearest public prompt;
3. When the answer is long, at least about 1/4 window is reserved for the public prompt, up to 256 token;
4. chosen/rejected use the same prompt starting position;
5. Keep the beginning of the answer when the answer still exceeds the window, and do not forge the EOS that ended early;
6. The log records the sequence truncation rate and answer truncation rate on both sides respectively.

### 4.3 Dynamic Padding and Batching

Each batch will first find out the longest sequence in chosen/rejected, and then unify the right Padding to the multiple of 8. Set the original preference batch Size as `B`, and the model actually sees:

```text
[chosen_0 ... chosen_B-1, rejected_0 ... rejected_B-1]
```

Therefore, the actual number of sequences of one Policy forward and one Reference forward are `2B`. `batch_size` and `global_batch` in the log represent the number of preferred pairs, not the number of sequences forward by the model.

## 5. Policy and Reference

At the beginning of the training, both are loaded from the same `out/sft`:

```text
Policy = deepcopy (SFT weight), all parameters can be trained
Reference = the same SFT weight, completely frozen
```

Before the first parameter update, the log probability of both should be the same for any input. Therefore:

```text
margin = 0
dpo_loss = -log sigmoid(0) = log(2) ≈ 0.693147
```

The initial DPO Loss approaching `0.6931` is a normal phenomenon, which does not mean that the training has failed. The total loss will be slightly higher than `0.6931` because of the addition of `router_aux_loss`.

In order to reduce the cumulative numerical error generated by the long answer under BF16/FP16, the model body uses BF16 by default, but the log Probability sum of `log_softmax` and answer token is fixed using FP32.

### 5.1 Composition of GPU memory

DPO full-parameter training needs to be saved at the same time:

- A Trainable Policy Weight;
- The gradient of Policy;
- AdamW Optimizer status;
- A frozen Reference weight;
- Policy activation of chosen/rejected;
- Policy and Reference Logits of the current batch.

Reference does not save gradient and reverse activation, but still needs weight and forward memory. Compared with SFT, the GPU memory and computing expenses of DPO will increase significantly. MiniMind defaults to 4 per card batch Size, and the current project also adopts this default value. Since each preference pair contains two sequences, chosen and rejected, the actual forward sequence number of a single GPU is 8. If the 1024 long sequence trigger memory is insufficient, `--batch-size` should be reduced to 2 or 1 first, and then the target global batch should be restored with gradient accumulation.

## 6. Start training

First, confirm that SFT has been successfully exported:

```text
out/sft/
├── config.json
├── model.safetensors
├── tokenizer.json
└── chat_template.jinja
```

If there is no `out/sft/config.json`, the trainer will directly refuse to start.

### 6.1 single GPU Training

```bash
python trainer/train_dpo.py \
    --model-path out/sft \
    --tokenizer-path out/sft \
    --data-path dataset/rl/dpo.jsonl \
    --output-dir out/dpo \
    --save-dir checkpoints/dpo \
    --epochs 1 \
    --batch-size 4 \
    --accumulation-steps 1 \
    --max-seq-len 1024 \
    --learning-rate 4e-8 \
    --min-learning-rate 4e-9 \
    --warmup-ratio 0 \
    --beta 0.15 \
    --label-smoothing 0 \
    --dtype bfloat16 \
    --tracker swanlab
```

Default single GPU valid global batch Size:

```text
4 × 1 × 1 = 4 preference pairs/updates
```

However, each micro-batch actually includes:

```text
4 chosen + 4 rejected = 8 sequences
```

### 6.2 six-GPU training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun --standalone --nproc_per_node=6 trainer/train_dpo.py \
    --model-path out/sft \
    --data-path dataset/rl/dpo.jsonl \
    --tracker swanlab
```

At this time, the effective global batch Size is:

```text
4 × 1 × 6 = 24 preference pairs/update
```

Like MiniMind, this configuration does not do gradient accumulation. The learning rate is `4e-8`, using cosine attenuation without Warmup and drops to `4e-9` at the end of training.

Doka only uses DDP for Policy. Each Rank holds a complete frozen Reference to avoid Reference participating in unnecessary gradient synchronization.

## 7. Do smoke training first.

Before formal training, use a small number of preferences to verify data, double-model forward, reverse, save and evaluate links:

```bash
python trainer/train_dpo.py \
    --model-path out/sft \
    --data-path dataset/rl/dpo.jsonl \
    --max-train-samples 64 \
    --eval-samples 16 \
    --max-steps 5 \
    --batch-size 1 \
    --accumulation-steps 1 \
    --max-seq-len 512 \
    --num-workers 0 \
    --dtype float32 \
    --log-interval 1 \
    --eval-interval 5 \
    --save-interval 5 \
    --tracker none
```

Key confirmation:

- Both Policy and Reference are loaded from `out/sft`;
- Policy parameters are displayed as `trainable`, and Reference is displayed as `frozen`;
- The first time `dpo` is close to `0.6931`;
- `total_loss`, gradient norm and sequence log probability are all finite values;
- `checkpoints/dpo/latest.pt` was successfully generated;
- After the training, `out/dpo/model.safetensors` was successfully exported.

Formal training defaults to `4e-8`; the number of smoke training steps is very small, and the change of indicators is not obvious, which is a normal phenomenon. The goal of the smoke test is to verify the pipeline, not to verify the final convergence effect.

## 8. Training indicators

The trainer records the following core indicators to the terminal and SwanLab/W&B:

| Indicator | Meaning |
|---|---|
| `train/dpo_loss` | Standard DPO Preference Classification Loss |
| `train/total_loss` | The sum of DPO Loss and MoE router auxiliary loss |
| `train/reward_chosen` | Policy Relative Reference Implicit Reward for chosen |
| `train/reward_rejected` | Policy Relative Reference Implicit Reward for rejected |
| `train/reward_margin` | chosen reward minus rejected reward |
| `train/preference_accuracy` | reward margin preference ratio greater than 0 |
| `train/policy_chosen_logp` | Policy's chosen sequence log probability |
| `train/policy_rejected_logp` | Policy's rejected sequence log probability |
| `train/router_aux_loss` | MoE expert Load Balancing Auxiliary Loss |
| `data/padding_efficiency` | The proportion of valid token accounts for token slots after Padding |
| `data/*_truncated_ratio` | chosen/rejected sequence truncation ratio |
| `data/*_answer_truncated_ratio` | chosen/rejected The proportion of the answer itself being truncated |

Verify the same `validation/*` indicators recorded in the assembly. Focus on observing when judging whether it is effective:

```text
Validation/reward_margin should be gradually increased
Validation/preference_accuracy should be higher than the random level and tend to be stable
Validation/dpo_loss should decline, but it is not required to decline quickly.
```

You can't just look at the training set `dpo_loss`. If the training accuracy rate continues to increase and the verification accuracy rate decreases, it means that the model is remembering preferences, and the number of training steps should be reduced or the learning rate should be reduced.

## 9. Default hyperparameter

| Parameters | Default Value | Description |
|---|---:|---|
| `epochs` | 1 | PREFERENCE DATA USUALLY ONLY TRAINS A SMALL NUMBER OF ROUNDS |
| `batch_size` | 4/GPU | Align MiniMind; Each time the model is actually 8 sequences/GPU |
| `accumulation_steps` | 1 | Align MiniMind, do not do gradient accumulation |
| `learning_rate` | 4e-8 | Align MiniMind DPO Initial Learning Rate |
| `min_learning_rate` | 4e-9 | Cosine ends at 10% of the initial learning rate |
| `warmup_ratio` | 0 | Align MiniMind, no Warmup |
| `weight_decay` | 0.01 | AdamW Weight Attenuation |
| `adam_beta1` | 0.9 | Align the AdamW default value used by MiniMind |
| `adam_beta2` | 0.999 | Align the AdamW default value used by MiniMind |
| `grad_clip` | 1.0 | Gradient Normal Cropping |
| `beta` | 0.15 | Align MiniMind DPO Default Value |
| `label_smoothing` | 0 | Align MiniMind, use standard DPO |
| `max_seq_len` | 1024 | The upper limit of each chosen/rejected branch |
| `val_ratio` | 0.05 | Currently about 858 verification preferences |
| `eval_samples` | 1000 | Automatically use all when the current validation set is less than 1000 |
| `eval_batch_size` | 2/GPU | Verification without reverse, larger micro-batch can be used |
| `early_stopping_patience` | 0 | Default off, do not change the complete round of training behavior of MiniMind |
| `dtype` | bfloat16 | Align MiniMind, suitable for RTX 4090 |
| `attention_backend` | eager | Stable default back-end of dynamic Padding |
| `gradient_checkpointing` | false | Turn it on again when the GPU memory is insufficient |

### How to choose 9.1 Beta

`beta` controls the scaling of DPO's relative preference interval:

- Smaller Beta: The constraints on Reference are weaker, allowing Policy to deviate further, but the initial gradient scale is also smaller;
- Larger Beta: Stronger constraints on Reference, and will amplify the gradient scale of the current preference Logit;
- The current default `0.15` is consistent with the official implementation of MiniMind.

It is recommended to fix other parameters first, make a small-scale comparison of `0.05, 0.10, 0.15, 0.20`, and compare the preference validation set and the fixed general ability test set at the same time.

### 9.2 How to choose the learning rate

The default `4e-8` is consistent with the configuration of MiniMind. If the cleaned data still needs to be compared with the learning rate, it is recommended to try only within a narrow range:

```text
2e-8 → 4e-8 → 5e-8
```

Only one variable is changed at a time. Do not significantly improve the learning rate, Beta and Epoch at the same time, otherwise it is impossible to judge which factor is causing the improvement or degradation of ability.

## 10. checkpoint retraining and output

Save during training:

```text
checkpoints/dpo/
├── latest.pt
└── best.pt
```

`latest.pt` is used for checkpoint continuation training; `best.pt` saves and verifies the model with the lowest DPO Loss. At the end of the training, `out/dpo` is exported from `best.pt`, instead of unconditionally exporting the last step. The new training will also check whether the Policy is consistent with the log probability of Reference at step 0, and save the SFT baseline as the first candidate best model.

Checkpoint contains:

- Policy parameters;
- Optimizer, Scheduler and GradScaler;
- Epoch, batch position and Global Step;
- Python, NumPy, PyTorch random status;
- Tracker Run information;
- The best verification Loss, the best Step and Early Stopping count;
- This DPO parameter and model configuration.

Reference is not saved to Checkpoint. When recovering, it will be reloaded from the original `out/sft`, so do not overwrite or replace the directory in the same DPO experiment.

Automatic recovery:

```bash
python trainer/train_dpo.py --resume
```

Restore the specified checkpoint:

```bash
python trainer/train_dpo.py \
    --resume checkpoints/dpo/latest.pt
```

The trainer will refuse to modify key parameters such as model, sequence length, data division, DPO Objective, batch, learning rate or Warmup during recovery to avoid the mismatch between Policy Checkpoint and Reference or optimizer trajectory.

Final export:

```text
out/dpo/
├── config.json
├── model.safetensors
├── generation_config.json
└── tokenizer files...
```

`out/dpo` is a standard Transformers model directory, which can be directly loaded with existing generation and subsequent training code.

## 11. Independent evaluation

After the training is completed, compare the DPO Policy with the original SFT Reference on the validation preference pair of deterministic division:

```bash
python eval/eval_dpo.py \
    --policy-path out/dpo \
    --reference-path out/sft \
    --data-path dataset/rl/dpo.jsonl \
    --split validation \
    --val-ratio 0.05 \
    --eval-samples 1000 \
    --batch-size 2 \
    --beta 0.15 \
    --label-smoothing 0 \
    --dtype bfloat16
```

The evaluation output is JSON, which is convenient for saving and comparing multiple experiments:

```bash
python eval/eval_dpo.py \
    --policy-path out/dpo \
    --reference-path out/sft \
    > dpo_eval.json
```

You can also point the Policy to `out/sft` to check the baseline first:

```bash
python eval/eval_dpo.py \
    --policy-path out/sft \
    --reference-path out/sft
```

At this time, `dpo_loss` should be close to `0.693147`, and `reward_margin` should be close to 0.

DPO offline indicators only indicate that the model is more inclined to the chosen in the data set, and cannot completely replace the generation quality evaluation. A set of fixed Prompts should also be used to compare `out/sft` and `out/dpo`:

- Whether the general Q&A, translation, writing and code ability have deteriorated;
- Whether the answer has become too long, templated or rejected too much;
- Whether Tool Calling and `<think>` formats are still stable;
- Whether the improvement of chosen is in line with the real preference, rather than using the length or wording deviation of the data set.

## 12. Frequently asked question

### 12.1 Loss has always been 0.6931

This is the theoretical correct value at the beginning of the training. The default learning rate is very small, and the terminal only displays 4 decimal places. It may be normal for the first few dozen steps to look unchanged. Prioritize the observation of more decimal `reward_margin`, gradient norm and validation set curves.

If all the indicators remain strictly unchanged after a complete round, check again:

- Whether the learning rate is wrongly overwritten as 0;
- Whether the Policy parameter is really `requires_grad=True`;
- Whether the gradient norm continues to be 0;
- Whether chosen/rejected is repeated in large quantities or the content is exactly the same;
- Whether to restart the experiment by mistakenly using the trained DPO model as Policy and Reference at the same time.

### 12.2 CUDA OOM

Adjust in the following order:

1. `--batch-size 4` drops to `2` or `1`;
2. Use `--accumulation-steps` to restore the required global batch;
3. `--max-seq-len 1024` drops to `768` or `512`;
4. Increase `--gradient-checkpointing`.

After lowering `max_seq_len`, the answer truncation rate must be re-observed. The recovery of video memory cannot be at the expense of cutting off a large number of key differences between chosen and rejected. If there is a non-limited value in BF16, keep eager attention and explicitly use `--dtype float32 --batch-size 1` for numerical stability review.

### 12.3 Preference Accuracy will soon be close to 100%

If the training accuracy rate quickly approaches 100%, the verification accuracy rate does not improve synchronously, which is usually overfitting. OK:

- Reduce `max_steps` or Epoch;
- Reduce the learning rate;
- Reduce Beta;
- The preference for training and verification is de-weighted;
- Check whether there is an obvious low-quality pattern in rejected.

### The general capacity declines after 12.4 DPO

DPO is optimized for static preference distribution, which does not guarantee that all SFT capabilities are upgraded synchronously. When there is degeneration:

- Priority return to the earlier DPO Checkpoint;
- Reduce the learning rate or only train part of the steps;
- Clean up the preference data with errors, excessive singleness or obvious length bias;
- Continue to run `eval/eval_sft.py` for the fixed SFT validation set;
- Compare the generated results at the same time, not just look at DPO Accuracy.

### 12.5 NaN or Inf

The trainer will check before reverse propagation:

- Total Loss, DPO Loss and router Loss;
- chosen/rejected reward;
- Policy and Reference sequence log probability.

When an abnormality occurs, the corresponding `source_jsonl_rows` will be output. First reproduce with a single GPU, `batch-size=1`, `num-workers=0` and `dtype=float32`, and then check the length and content of the answer in the line.

## 13. Run the test

Perform in an environment where PyTorch, Transformers and Datasets have been installed:

```bash
python -m unittest discover \
    -s tests \
    -p 'test_dpo_*.py' \
    -v
```

Test coverage:

- When Policy and Reference are the same, DPO Loss is equal to `log(2)`;
- Loss decreases when the chosen advantage increases;
- next-token Shift and answer Mask correctly;
- Historical assistant messages will not be included in the current preference score;
- Report an error immediately when chosen/rejected prompt is inconsistent;
- After truncation in pairs, keep the same public prompt on both sides;
- Collator is batched in the order of chosen first and rejected later;
- Only Policy generates gradients in the front and rear direction of micro MoE.

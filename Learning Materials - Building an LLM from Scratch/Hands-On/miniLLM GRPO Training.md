# miniLLM GRPO Training

> Stage: online reinforcement learning after DPO

> Default recipe: four sampled responses per prompt, InternLM2-1.8B-Reward, beta 0.1, clipping range 0.2, and learning rate 1e-7

---

## 1. What Does GRPO Training Solve?

DPO learns from fixed `chosen/rejected` pairs. GRPO instead samples responses from the current policy online, scores them, and adjusts their probabilities according to reward. This lets the model explore behaviors that are not covered by the DPO dataset.

The training order of this project is:

```text
out/sft
   ↓ DPO
out/dpo
├── Policy: full-parameter training
└── Reference model: frozen, provides the KL constraint
              ↓
prompt in dataset/rl/rlaif.jsonl
↓ Generate G responses per prompt
Reward model + rule-based rewards
↓ Group-relative advantages
GRPO clipped loss + reference KL + MoE router loss
              ↓
out/grpo
```

GRPO does not require PPO's critic/value model, but it must generate several responses per prompt. Most of its cost therefore comes from online rollouts and reward-model inference.

## 2. Related documents

| File | Function |
|---|---|
| [`trainer/train_grpo.py`](../../trainer/train_grpo.py) | Reward, Grouping Advantage, GRPO Loss, Training, Verification, checkpoint and Export |
| [`dataset/ppo_dataset.py`](../../dataset/ppo_dataset.py) | prompt-only data validation, template rendering, truncation and deterministic division |
| [`trainer/rollout_engine.py`](../../trainer/rollout_engine.py) | Policy Online Generation, Completion Mask and Behavioral Strategy log probability |
| [`dataset/rl/rlaif.jsonl`](../../dataset/rl/rlaif.jsonl) | GRPO prompt Data |
| [`tests/test_grpo.py`](../../tests/test_grpo.py) | GRPO Formula, Reward, Mask and Grouping Order Test |

This project uses native PyTorch to implement GRPO without relying on TRL. The overall training method refers to MiniMind's `train_grpo.py`, and reuses the existing Dataset, Rollout, DDP, Tracker and Checkpoint infrastructure of miniLLM.

## 3. The principle of GRPO

Generate `G` answers for each prompt and get `R1...RG` as a reward. The Advantage in the group is:

```text
Ai = (Ri - group_mean) / (group_std + 1e-4)
```

The answer higher than the average Reward in the group gets a positive Advantage, and the answer lower than the average gets a negative Advantage. The Advantage of each answer will be assigned to all valid Completion tokens for the answer.

The probability ratio between the current Policy and the behavioral strategy at the time of generating the answer is:

```text
ratio = exp(current_log_prob - old_log_prob)
```

GRPO Policy Loss uses PPO style cropping:

```text
surrogate = min(
    ratio × advantage,
    clip(ratio, 1-epsilon, 1+epsilon) × advantage
)
```

Reference KL uses non-negative token estimation in similar implementations of MiniMind:

```text
log_ratio = reference_log_prob - current_log_prob
kl = exp(log_ratio) - log_ratio - 1
```

The final training goal:

```text
total_loss = grpo_policy_loss
           + beta × reference_kl
           + router_aux_loss
```

Only Completion token participates in GRPO Loss. The first EOS will participate in the training, and the token and Padding used to complete the batch after EOS will not participate in Loss.

### 3.1 Zero variance group

If all the answers in the same prompt are the same Reward:

```text
R1 = R2 = ... = RG
```

All Advantages in this group are 0, and no valid Policy Gradient is generated. This usually means:

- The questions are too simple, and all the answers get the same high scores;
- The questions are too difficult, and all the answers get the same low scores;
- reward model can't distinguish the answers;
- The sampling temperature is too low, and the multiple answers are highly similar.

During training, you should focus on `zero_std_group_rate`.

## 4. Reward composition

The current reward follows the simple combination of MiniMind:

```text
total_reward = reward_model
             + length_reward
             + thinking_reward
             - repetition_penalty
```

### 4.1 reward model

`InternLM2-1.8B-Reward` is used by default. The model score is cropped to `[-3, 3]`, and the abnormal `NaN/Inf` score will terminate the training immediately.

reward model is only responsible for scoring:

- Always keep frozen and `eval` mode;
- Use FP16 on CUDA;
- Do not save to GRPO Checkpoint;
- Reload from the original path when resuming training.

### 4.2 Rule Reward

- The answer length is between 20 and 800 characters: `+0.5`, otherwise `-0.5`;
- A single `</think>` appears and the length of the thinking content is reasonable: the highest `+1.25`;
- Repeat trigram: maximum deduction `0.5`.

Rule rewards can only be used to constrain obvious format problems, and cannot replace the quality of real answers. High Reward answers should be manually checked regularly to prevent models from obtaining false high scores by stacking lengths, templates or specific words.

## 5. Data format

The default data is [`dataset/rl/rlaif.jsonl`](../../dataset/rl/rlaif.jsonl), a total of 19,502. Each line contains the complete `conversations`, and the last one must be an empty assistant placeholder:

```json
{
  "conversations": [
    {"role": "user", "content": "Please explain what photosynthesis is."},
    {"role": "assistant", "content": ""}
  ]
}
```

During training, the empty assistant will be removed, and the answer will be regenerated by the current Policy. The data set itself does not have `chosen/rejected` or Ground Truth, so it cannot be trained directly from the reward model.

Multiple rounds of history, system and tool definitions can be retained. When the prompt is too long, delete the earlier complete round first, and then keep the latest token suffix if necessary. The default verification ratio is `0.02`.

## 6. Prepare the model

Complete the DPO first, and confirm the existence of the following documents:

```text
out/dpo/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

Then download `InternLM2-1.8B-Reward`, and it is recommended to put it in the same directory as miniLLM:

```text
Project/
├── miniLLM/
└── internlm2-1_8b-reward/
    ├── config.json
    ├── model.safetensors
    └── tokenizer files...
```

Installation dependency:

```bash
pip install -r requirements.txt
```

The trainer only loads Policy, Reference, tokenizer and reward model locally, and will not automatically download missing files from the network at startup.

## 7. Do smoke training first.

Verify model loading, grouping Rollout, Reward, reverse and save links before formal training:

```bash
python trainer/train_grpo.py \
    --model-path out/dpo \
    --reward-model-path ../internlm2-1_8b-reward \
    --max-train-samples 8 \
    --eval-samples 2 \
    --max-steps 2 \
    --batch-size 1 \
    --num-generations 2 \
    --accumulation-steps 1 \
    --max-prompt-len 256 \
    --max-new-tokens 64 \
    --num-workers 0 \
    --log-interval 1 \
    --eval-interval 2 \
    --save-interval 2 \
    --tracker none
```

Key confirmation:

- Both Policy and Reference are loaded from `out/dpo`;
- reward model is successfully loaded and all parameters are frozen;
- `reward`, `loss`, `kl` and gradient norm are all finite values;
- `checkpoints/grpo/latest.pt` was successfully generated;
- After the end, `out/grpo/model.safetensors` was successfully exported.

Two-step training cannot prove that the model has converged, and is only used to verify the complete pipeline.

## 8. Formal training

### 8.1 single GPU

```bash
python trainer/train_grpo.py \
    --model-path out/dpo \
    --reference-path out/dpo \
    --reward-model-path ../internlm2-1_8b-reward \
    --data-path dataset/rl/rlaif.jsonl \
    --output-dir out/grpo \
    --save-dir checkpoints/grpo \
    --epochs 1 \
    --batch-size 1 \
    --num-generations 4 \
    --accumulation-steps 8 \
    --max-prompt-len 512 \
    --max-new-tokens 256 \
    --learning-rate 1e-7 \
    --beta 0.1 \
    --epsilon 0.2 \
    --dtype bfloat16 \
    --tracker swanlab
```

`--reference-path` is equal to `--model-path` by default, so it can be omitted. At the beginning of the training, Policy and Reference are exactly the same, and the initial KL should be close to 0.

### 8.2 Dual SIM

```bash
torchrun --nproc_per_node 2 trainer/train_grpo.py \
    --model-path out/dpo \
    --reward-model-path ../internlm2-1_8b-reward \
    --batch-size 1 \
    --num-generations 4 \
    --accumulation-steps 4 \
    --dtype bfloat16 \
    --tracker swanlab
```

Each GPU will hold the complete Policy, Reference and reward model. DDP only synchronizes the Policy gradient, not Reference and reward model.

## 9. Default parameters

| Parameters | Default Value | Description |
|---|---:|---|
| `epochs` | 1 | Online RL only trains a small number of rounds by default |
| `batch_size` | 1/GPU | prompt number per card, not the number of generated answers |
| `num_generations` | 4 | Number of answers per prompt |
| `accumulation_steps` | 8 | prompt batch Accumulated by Gradient |
| `learning_rate` | 1e-7 | Conservative learning rate of continuing training after DPO |
| `min_learning_rate` | 1e-8 | Cosine Terminal Learning Rate |
| `beta` | 0.1 | Reference KL Weight |
| `epsilon` | 0.2 | GRPO Ratio Clip Range |
| `max_prompt_len` | 512 | prompt token Upper Limit |
| `max_new_tokens` | 256 | Single answer generates token upper limit |
| `temperature` | 0.8 | Rollout Sampling Temperature |
| `top_p` | 0.9 | Nucleus Sampling |
| `top_k` | 50 | Top-k Sampling |
| `thinking_ratio` | 0.0 | Do not turn on explicit thinking by default |
| `val_ratio` | 0.02 | deterministic validation set Ratio |
| `eval_samples` | 32 | Online Verification prompt Quantity |

The actual number of answers in each forward direction is:

```text
batch_size × num_generations
```

The effective prompt Group number is:

```text
Batch_size × accumulation_steps × GPU quantity
```

Adding `num_generations` can improve the quality of comparison in the group, but the forward cost of Rollout, Reward and training is close to linear growth.

## 10. Training indicators

| Indicator | Meaning |
|---|---|
| `train/reward` | Total reward average |
| `train/reward_model` | reward model Score Average |
| `train/reward_length` | Length Rule Reward |
| `train/reward_thinking` | Thinking Format Reward |
| `train/repetition_penalty` | Repeat punishment, the lower the better |
| `train/policy_loss` | GRPO Policy Loss |
| `train/kl_penalty` | Non-negative Reference KL Punishment |
| `train/router_aux_loss` | MoE router Load Balancing Loss |
| `train/zero_std_group_rate` | In-group Reward No Difference Ratio |
| `train/clip_fraction` | Ratio token ratio beyond the Clip range |
| `train/eos_rate` | Answer the ratio of normally generating EOS |
| `train/response_length` | Average Effective Answer token Number |

The verification stage will also record `validation/reward`, `validation/sampled_kl`, `validation/eos_rate` and each Reward component.

You can't choose the model only based on the training Reward. Reward rises but the repetition rate, output length or real answer quality deteriorates, which usually indicates Reward Hacking.

## 11. checkpoint retraining and output

The training status is saved in:

```text
checkpoints/grpo/latest.pt
```

Automatic recovery:

```bash
python trainer/train_grpo.py \
    --model-path out/dpo \
    --reward-model-path ../internlm2-1_8b-reward \
    --resume
```

The trainer will refuse to modify Policy, Reference, reward model, `num_generations`, maximum length, Beta or Epsilon at the time of recovery to prevent inconsistency between the old and new training semantics.

Export after the training is completed:

```text
out/grpo/
├── config.json
├── model.safetensors
├── generation_config.json
└── tokenizer files...
```

## 12. Frequently asked question

### 12.1 CUDA OOM

Try in order:

1. `num_generations` decreased from 4 to 2;
2. `max_new_tokens` drops from 256 to 128 or 64;
3. `max_prompt_len` dropped from 512 to 384 or 256;
4. Add `--gradient-checkpointing`;
5. Maintain `batch-size=1`, and maintain the effective Group number through the cumulative number of steps.

GRPO holds Policy, Reference, reward model and online answer at the same time, which is usually easier to have OOM than DPO.

### 12.2 `zero_std_group_rate` is very high

- Increase the sampling temperature or appropriately increase `top_p`;
- Increase `num_generations` from 2 to 4 or 6;
- Clean up problems that are too easy or beyond the ability of the model;
- Check whether reward model scores different answers into the same score.

### 12.3 KL Rapid increase

- Reduce the learning rate;
- Increase `beta`;
- Reduce the number of training steps;
- Check whether Reward has easy-to-use rule vulnerabilities.

### 12.4 Reward rises but the answer gets worse

Manually check the following samples:

- Reward The highest answer;
- The longest and most repeated answers;
- The answer before and after the sudden jump of Reward;
- Fixed prompt with the greatest difference between GRPO and DPO output.

At the same time, continue to run the existing SFT/DPO evaluation to prevent general capability, preference alignment and Tool Calling degradation.

## 13. Run the test

Run in the environment where PyTorch, Transformers and Datasets are installed:

```bash
python -m unittest tests.test_grpo -v
```

Test coverage:

- prompt expand continuously by group;
- The average value of each group of Advantage is 0;
- Zero variance group Advantage is 0;
- Positive and negative Advantage produces opposite gradients;
- The excessive Ratio of Positive Advantage is cropped;
- Padding does not participate in GRPO Loss;
- Reward weight and answer order are correct;
- Thinking and repeating the rule reward are correct.

## 14. Reference material

- [MiniMind `train_grpo.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_grpo.py)
- [DeepSeekMath: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)

# miniLLM PPO Training

> Default route: `SFT Actor + frozen SFT Reference + SFT-initialized Critic + external reward model`. You can also use a DPO-exported model through `--model-path` when it performs well.

## 1. Training Pipeline

```text
out/sft
├─ Actor (trainable)
├─ Reference model (frozen)
└─ Critic Backbone (trainable) + Value Head

dataset/rl/rlaif.jsonl
→ Actor generates responses online
→ Reward model scores responses
  → Reference KL + Critic Value + GAE
→ PPO clipped updates for actor and critic
  → out/ppo
```

The actor, reference model, and critic backbone must be initialized from the same base model. The current DPO checkpoint does not outperform SFT, so the default starts directly from `out/sft`. The reference model constrains how far PPO can drift from that starting point.

## 2. Code structure

| File | Function |
|---|---|
| `dataset/ppo_dataset.py` | Check RLAIF dialogue, delete empty assistant position, render prompt |
| `trainer/rollout_engine.py` | PyTorch Generation, Response Mask, Old log probability |
| `trainer/train_ppo.py` | Critic, Reward, GAE, PPO Loss, DDP, AMP, Training Cycle |
| `trainer/trainer_utils.py` | Actor/Critic Dual Model checkpoint Saving and Recovery |
| `tests/test_ppo.py` | Data, Mask, GAE, Clip, Rollout and Checkpoint Test |

PPO adopts native PyTorch and does not rely on TRL.

## 3. Data format

`dataset/rl/rlaif.jsonl` Each line should be:

```json
{
  "conversations": [
    {"role": "user", "content": "Please explain the reason for the blue sky."},
    {"role": "assistant", "content": ""}
  ]
}
```

The last assistant is empty, not the training answer. Dataset will delete it and join the assistant generation prompt through the ChatML template.

Long conversations give priority to deleting the earliest complete round; if the latest problem itself is still too long, use token left truncation. `prompt_truncated_ratio` in the log is used to audit this situation.

## 4. reward model

The SFT/DPO model itself does not include the reward model. The official PPO requires a local Hugging Face reward model and implements the interface used by MiniMind:

```python
model.get_score(tokenizer, messages)
```

The numerical accuracy of Policy and reward model is independent. `--dtype` only controls Actor/Reference/Critic; `--reward-dtype` only controls the external reward model. The official usage of InternLM2 Reward and MiniMind are both loaded in FP16, so the latter defaults to `float16` and will not follow the BF16 settings of Policy. When starting, it will also check the Reward weight and a fixed forward first, and intercept NaN/Inf before sampling and PPO update. The `rotary_emb.inv_freq` of the old InternLM2 code is a derived buffer that does not write weighted files; after loading, it will rebuild all RoPE frequencies with FP32 according to the official formula to prevent the rapid initialization of the new Transformers from leaving a random or non-finite buffer.

The remote model code of InternLM2 Reward uses the old Cache/RoPE API, and the Qwen3 distillation function of the project requires Transformers 4.51+. `requirements.txt` therefore fixes the common compatibility interval as `transformers>=4.51,<4.56` to avoid the explicit Cache reconstruction starting at 4.56 from destroying this old remote code.

Remote download and remote code are prohibited by default. Only when the model warehouse has been audited will it be explicitly added:

```bash
--allow-remote-reward-model \
--trust-reward-remote-code
```

`--reward-backend rule` is only used to confirm that the training pipeline can run through, and cannot be used for formal alignment.

The total reward is:

```text
reward model Score - Repeated Punishment - No EOS Punishment
```

Each component will be recorded separately to avoid finding Reward Hacking only by looking at the total Reward.

## 5. Run the smoke test first

Make sure that `out/sft` already exists, and then check the pipeline with a small number of samples and rules:

```bash
python trainer/train_ppo.py \
  --model-path out/sft \
  --data-path dataset/rl/rlaif.jsonl \
  --reward-backend rule \
  --max-train-samples 8 \
  --eval-samples 2 \
  --eval-batches 0 \
  --max-rollout-steps 2 \
  --batch-size 2 \
  --mini-batch-size 1 \
  --ppo-update-iters 1 \
  --max-prompt-len 128 \
  --max-new-tokens 16 \
  --num-workers 0 \
  --device cuda
```

The smoke test is not used to evaluate the quality of the model.

## 6. single GPU formal training

```bash
python trainer/train_ppo.py \
  --model-path out/sft \
  --data-path dataset/rl/rlaif.jsonl \
  --reward-backend external \
  --reward-model-path /path/to/internlm2-1_8b-reward \
  --trust-reward-remote-code \
  --output-dir out/ppo \
  --save-dir checkpoints/ppo \
  --epochs 1 \
  --batch-size 2 \
  --mini-batch-size 1 \
  --ppo-update-iters 2 \
  --actor-learning-rate 1e-7 \
  --critic-learning-rate 5e-7 \
  --max-prompt-len 512 \
  --max-new-tokens 128 \
  --dtype bfloat16 \
  --reward-dtype float16 \
  --device cuda
```

## 7. multi-GPU training

```bash
torchrun --nproc_per_node 2 trainer/train_ppo.py \
  --model-path out/sft \
  --reward-model-path /path/to/internlm2-1_8b-reward \
  --trust-reward-remote-code \
  --batch-size 2 \
  --mini-batch-size 1 \
  --dtype bfloat16 \
  --reward-dtype float16
```

Each GPU will hold Actor, Reference, Critic and reward model. DDP will not automatically make a display memory cut on these four weights, so let's start with `batch-size=1~2`.

## 8. Core parameters

| Parameters | Default Value | Description |
|---|---:|---|
| `actor-learning-rate` | `1e-7` | PPO Actor uses a small learning rate |
| `critic-learning-rate` | `5e-7` | Critic needs to learn faster Value |
| `dtype` | `bfloat16` | Mixing accuracy of Actor/Reference/Critic |
| `reward-dtype` | `float16` | External reward model Reasoning Accuracy, Decoupled from Policy |
| `clip-epsilon` | `0.2` | PPO Policy Ratio Crop |
| `value-clip` | `0.2` | Critic Value Change Crop |
| `value-loss-coef` | `0.5` | Value Loss Weight |
| `kl-coef` | `0.02` | KL punishment of the relative freezing of the base model Reference |
| `gamma` | `1.0` | Discount Factor of Final Reward Mission |
| `gae-lambda` | `0.95` | GAE Deviation - variance compromise |
| `ppo-update-iters` | `2` | Repeated update rounds of the same batch of Rollout |
| `early-stop-kl` | `0.10` | New/Old KL Stop the current batch update when it is too large |

## 9. Log and diagnosis

Key monitoring:

- `train/reward` and `train/model_reward`;
- `train/reference_kl`;
- `train/approximate_kl`;
- `train/clip_fraction`;
- `train/value_loss` and `train/value_clip_fraction`;
- `train/raw_advantage_mean`;
- `train/eos_ratio` and `train/response_length`;
- Actor/Critic router auxiliary loss;
- Actor/Critic Gradient Norm.

Common abnormalities:

| Phenomenon | Possible Causes | Handling |
|---|---|---|
| Reward rises but labor quality decreases | Reward Hacking | Reduce the learning rate/KL, review the highest score sample |
| `reference_kl` Rapid growth | KL is too weak or Actor LR is too big | Increase `kl-coef` or reduce Actor LR |
| Value Loss Explosion | Reward Scale or Critic LR Inappropriat | Reduce Reward Clip/Critic LR |
| Reward's first forward is NaN | Reward accuracy inconsistent with the official configuration or weight damage | InternLM2 uses `--reward-dtype float16`; check the startup status |
| Reward's `rotary_emb.inv_freq` is not limited | The non-persisting RoPE buffer of the old remote code is not restored correctly | Confirm that `Reward RoPE: ... FP32 buffers rebuilt` appears at startup |
| EOS ratio is very low | Answer length is insufficient or strategy degradation | Check sample, length and EOS punishment |
| The first round of New/Old KL is not close to 0 | LogProb/Mask misalignment | Stop training and run single test |

## 10. checkpoint recovery

```bash
python trainer/train_ppo.py \
  --model-path out/sft \
  --reward-model-path /path/to/internlm2-1_8b-reward \
  --trust-reward-remote-code \
  --resume
```

`--resume` reads `checkpoints/ppo/latest.pt` by default. Parameters that affect the training semantics, such as Policy starting point, reward model, sequence length, Clip and KL, are not allowed to be changed during recovery.

## 11. Post-training verification

After PPO, Actor will be exported to `out/ppo`. Check the generation first:

```bash
python eval/eval_sft.py \
  --mode generate \
  --model-path out/ppo
```

Check the DPO preference maintenance again:

```bash
python eval/eval_dpo.py \
  --policy-path out/ppo \
  --reference-path out/sft
```

You can't choose the model directly according to the highest point of training Reward. The fixed prompt generation, general SFT Loss, preferred accuracy, security and answer length of the base model/PPO should also be compared.

## 12. Current performance boundary

At present, `MiniLLMForCausalLM` does not have KV Cache, and Rollout will repeatedly calculate the complete prefix. The first round recommends `max_new_tokens=64~128` and verifies it with small data first. Before the formal expansion of training, KV Cache should be implemented as an independent performance milestone, and Logits consistency before and after the cache should be checked.

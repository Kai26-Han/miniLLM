# miniLLM Pretraining

> Project: miniLLM
> Stage: language-model pretraining from random weights
> Current production recipe: four experts, Top-1 MoE, about 200M total parameters, about 65M active parameters, and one epoch over the full dataset
> Training monitoring: SwanLab
> Default sequence length: 512 tokens

---

## 1. What Pretraining Does

The tokenizer stage determines how text becomes token IDs. During pretraining, a randomly initialized decoder-only Transformer learns from a large number of next-token predictions:

- basic patterns of language;
- lexical, grammatical, and semantic relationships in Chinese and English;
- common knowledge and document structure; and
- how to predict the next token from its left context.

The result is a base model for later SFT, conversational alignment, and domain fine-tuning.

The whole training pipeline is as follows:

```text
JSONL text
   ↓
miniLLM tokenizer (8192 vocabulary)
   ↓
BOS + token IDs + EOS + Padding
   ↓
8-layer Decoder-only Transformer
   ├── GQA + RoPE Self-attention
   └── 4 expert, Top-1 Sparse MoE
   ↓
Predict the next token of each position
   ↓
cross-entropy + router Auxiliary Loss
   ↓
AdamW updates model parameters
   ↓
Checkpoint + SwanLab Indicator + Transformers Model
```

tokenizer will not continue to be modified during the pre-training process. Once the vocabulary and special token ID of tokenizer are determined, they must be consistent with the model configuration.

---

## 2. Documents related to pre-training in the current project

```text
miniLLM/
├── dataset/
│   ├── lm_dataset.py
│   └── pretrain/
│       ├── pretrain_mini.jsonl
│       └── pretrain.jsonl
├── model/
│   ├── model_minillm.py
│   └── tokenizer/
├── trainer/
│   ├── train_pretrain.py
│   └── trainer_utils.py
├── eval/
│   └── eval_llm.py
├──checkpoints/ # Generated after training, used for checkpoint training
├── out/ # Generated after training, used for evaluation, reasoning and subsequent fine-tuning
└── requirements.txt
```

The responsibilities of each document are as follows:

| File | Function |
|---|---|
| [`dataset/lm_dataset.py`](../../dataset/lm_dataset.py) | Load JSONL, divide training/validation set, Tokenize, truncated and Padding |
| [`model/model_minillm.py`](../../model/model_minillm.py) | miniLLM Dense/MoE Model Structure and Causal LM Loss |
| [`trainer/train_pretrain.py`](../../trainer/train_pretrain.py) | Main training entry point, training loop, evaluation, log and model export |
| [`trainer/trainer_utils.py`](../../trainer/trainer_utils.py) | DDP, learning rate scheduling, Checkpoint, SwanLab/W&B access |
| [`eval/eval_llm.py`](../../eval/eval_llm.py) | Calculate and verify Loss/PPL, and conduct text continuation test |
| [`model/tokenizer/`](../../model/tokenizer/) | Complete tokenizer Catalog for Pre-Training |
| [`requirements.txt`](../../requirements.txt) | PyTorch, Transformers, SwanLab and other dependencies |

---

## 3. Data scheme

### 3.1 Current data

There are already two MiniMind pre-training data in the project:

| File | Local Size | Number of Lines | Recommended Use |
|---|---:|---:|---|
| `pretrain_mini.jsonl` | About 1.2 GB | 1,270,238 | Smoke test, video memory test, short-cycle experiment |
| `pretrain.jsonl` | About 7.7 GB | 8,468,827 | Official full pre-training |

Use of the official program:

```text
dataset/pretrain/pretrain.jsonl
```

Don't pass `pretrain_mini.jsonl` and `pretrain.jsonl` at the same time for "more data". The overlapping relationship between the two is not clearly guaranteed, and the simultaneous use may allow the overlapping samples to be trained repeatedly. Mini files should be regarded as quick experimental data, not an additional increment of formal data.

### 3.2 JSONL format

The training script requires each line to be a legitimate JSON object and contain the string field `text`:

```json
{"text": "Here is a text for language model training."}
```

Only read the `text` field. Metadata files, `metadata.json` of tokenizer and other JSON configurations cannot be transmitted as pre-training corpus.

### 3.3 How can each data become a training sample?

`PretrainDataset` executes for each line of text:

```text
Original text
↓ tokenizer, do not automatically add special token
Ordinary token IDs
↓ Keep up to max_seq_len - 2
[BOS] + token IDs + [EOS]
↓ Make up the PAD on the right side to a fixed length
input_ids, attention_mask, labels
```

The current special token ID must be:

| Function | token ID |
|---|---:|
| PAD | 0 |
| UNK | 0 |
| BOS | 1 |
| EOS | 2 |

The label of the Padding position will be changed to `-100`, so it does not participate in cross-entropy. The model learns to predict the next token from left to right:

```text
Input: BOS Today's weather is very good EOS
Tag: Today's weather is very good EOS——
```

Note: There is no sequence packing in the current implementation. A short text will still be completed to the length of 512, and another text cannot be loaded into the remaining position. Therefore, "512 token slots processed" is not equal to "512 valid corpus tokens".

### 3.4 Division of training set and validation set

Default `--val-ratio 0.001`, that is, according to the certainty of the source data line number:

```text
No. 0, 1000, 2000... Line → Validation
Other lines → Train
```

For the full quantity `pretrain.jsonl`:

- Total sample: 8,468,827;
- Verification candidates: about 8,469;
- Training samples: about 8,460,358;
- In fact, `--eval-samples 2048` strips are used at most for each verification.

Fixed division is convenient for the comparison of different experiments, but if the source files are highly ordered by theme, sampling by line number may still bring distribution deviation. A more rigorous follow-up version can establish an independent and de-weighted validation set.

---

## 4. Model architecture scheme

### 4.1 Public skeleton

At present, miniLLM is Decoder-only Causal Language Model:

| Parameters | Current value |
|---|---:|
| tokenizer vocabulary | 8,192 |
| Hidden Dimension | 768 |
| Decoder Number of layers | 8 |
| Query Heads | 8 |
| Key/Value Heads | 4 |
| Head Dimension | 96 |
| FFN/expert Intermediate Dimension | 2,432 |
| Training sequence length | 512 |
| Maximum position configuration | 32,768 |
| RoPE Theta | 1,000,000 |
| attention | GQA + RoPE + Causal Mask |
| Normalization | RMSNorm |
| FFN Activation | SwiGLU |
| embedding and LM Head | Weight Sharing |

`max_position_embeddings=32768` is only the upper limit of the position number allowed by the model, and the actual training length of this round is still 512. Training only on 512 length does not mean that the model has a reliable 32K long context capability.

### 4.2 Officially adopted MoE structure

The official plan is open:

```text
--use-moe
--num-experts 4
--num-experts-per-tok 1
```

Both the training entry point and `MiniLLMConfig` have turned on this MoE configuration by default. The official command still explicitly writes `--use-moe` so that the experiment can be reproduced; when it is necessary to run the Dense control experiment, use `--no-use-moe`.

There are 4 independent Experts in each Decoder Layer. router calculates 4 probabilities for each valid token, and only sends the token to the 1 expert with the highest score:

```text
token hidden state
       ↓ router Softmax
[E0=0.10, E1=0.62, E2=0.18, E3=0.10]
       ↓ Top-1
Only execute expert 1
```

Therefore:

- All 4 expert parameters need to be saved and stored in the GPU memory;
- Each token executes only 1 expert per layer;
- The total amount of parameters has increased significantly;
- The activation calculation of a single token is still close to the same width of the Dense model;
- MoE saves activation calculation, which is not the same as saving model weight and optimizer GPU memory.

### 4.3 Parameter quantity

Calculate according to the actual dimension of the current source code:

| Architecture | Total Parameters | Each token Activation Parameters |
|---|---:|---:|
| Dense | 65,286,912, about 65.3M | about 65.3M |
| 4 expert, Top-1 MoE | 199,791,360, about 199.8M | 65,311,488, about 65.3M |

This is the current accurate version of the source code of "about 198M total parameters and about 64M activation parameters". Different projects have different parameter grouping and rounding methods. The actual output of `Parameters` and `Active params` shall prevail when starting documents and programs.

### 4.4 router Auxiliary Loss

If there is no constraint, router may send most tokens to the same expert, forming expert Collapse. The current model uses load balancing auxiliary loss:

```text
total_loss = lm_loss + router_aux_loss
```

Default coefficient:

```text
--router-aux-loss-coef 5e-4
```

The goal of this value is to encourage expert to balance, but not too big to overcome the learning goal of the language model itself.

---

## 5. Training hyperparameter scheme

### 5.1 Current official configuration

| Parameters | Values | Meaning |
|---|---:|---|
| epochs | 1 | Complete traversal official data 1 round |
| max_seq_len | 512 | Each sample fixed 512 token slot |
| batch_size | 8 | Samples of each GPU and each micro-batch |
| accumulation_steps | 8 | Update once after accumulating 8 micro-batch |
| learning_rate | 5e-4 | Peak learning rate |
| min_learning_rate | 5e-5 | The lowest learning rate at the end of Cosine |
| warmup_ratio | 0.03 | The first 3% optimizer steps to carry out linear Warmup |
| weight_decay | 0.1 | AdamW weight decay |
| grad_clip | 1.0 | Gradient standard cropping |
| dtype | bfloat16 | CUDA mixing accuracy first choice |
| val_ratio | 0.001 | 0.1% data as a verification candidate |
| eval_samples | 2048 | The number of samples with the most evaluation per verification |
| log_interval | 10 | Record once for every 10 optimizer steps |
| eval_interval | 500 | Verify once every 500 steps |
| save_interval | 1000 | Overwrite every 1000 steps to save `latest.pt` |
| seed | 42 | random seeds |

### 5.2 Valid batch Size

The effective global batch Size is:

$$
B_{global}=B_{micro}\times N_{accumulation}\times N_{GPU}
$$

Default configuration of a single GPU:

$$
8\times8\times1=64\text{ samples/update}
$$

The largest token slot:

$$
64\times512=32768\text{ token slots/update}
$$

If the display memory is insufficient, it can be adjusted under the condition that the product is approximately unchanged:

| batch_size | accumulation_steps | single GPU valid batch |
|---:|---:|---:|
| 8 | 8 | 64 |
| 4 | 16 | 64 |
| 2 | 32 | 64 |
| 1 | 64 | 64 |

Smaller micro-batch is usually more economical to activate video memory, but the increase in the cumulative number of gradients will increase the cycle overhead and reduce the throughput.

### 5.3 How many steps does a round take?

For full data, the default parameters of a single GPU are approximately:

```text
Train samples       ≈ 8,460,358
Micro-batch size    = 8
Accumulation steps  = 8
Optimizer steps     ≈ 132,193
```

`global_step` counts the number of `optimizer.step()`, not how many DataLoader batches are read.

When `--max-steps` is greater than 0, it will cover the total number of steps calculated by `--epochs`, which is suitable for short experiments; a formal round of training should keep `--max-steps` at default 0.

### 5.4 Learning rate curve

Current scheduling method:

```text
0
↓ The first 3% step linear Warmup
5e-4
↓ Remaining step Cosine attenuation
5e-5
```

Warmup is used to reduce the risk of gradient instability in the random initialization stage; Cosine attenuation allows the later stage of training to converge with a smaller step length.

---

## 6. Pre-training environmental preparation

The following commands assume that the terminal is located in the root directory of the project:

```bash
cd /root/miniLLM
```

Installation project dependency:

```bash
pip install -r requirements.txt
```

If the mirror already has a PyTorch that matches CUDA, it is not recommended to overwrite and install other CUDA versions of PyTorch at will. Check first:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Check the parameters of the training script:

```bash
python trainer/train_pretrain.py --help
```

SwanLab online monitoring requires account authentication. After logging in, the parameters in the training command must be:

```text
--tracker swanlab
```

---

## 7. Don't start full training directly: do the smoke test first.

It is recommended to run 20 optimizer steps with a small amount of data first to check whether the model, video memory, Checkpoint, verification and SwanLab are all normal:

```bash
python trainer/train_pretrain.py \
  --data-path dataset/pretrain/pretrain_mini.jsonl \
  --tokenizer-path model/tokenizer \
  --use-moe \
  --num-experts 4 \
  --num-experts-per-tok 1 \
  --max-seq-len 512 \
  --batch-size 8 \
  --accumulation-steps 8 \
  --max-train-samples 4096 \
  --max-steps 20 \
  --log-interval 1 \
  --eval-interval 10 \
  --save-interval 10 \
  --gradient-checkpointing \
  --tracker swanlab \
  --tracker-project miniLLM-Pretrain \
  --tracker-run-name moe-smoke-test \
  --tracker-tags moe smoke-test \
  --save-dir checkpoints/pretrain_moe_smoke \
  --output-dir out/pretrain_moe_smoke
```

The smoke test confirms at least:

1. Startup output display `MoE (4 Experts, Top-1)`;
2. The total parameters are about 199.8M, and the activation parameters are about 65.3M;
3. Device is CUDA;
4. Precision is bfloat16 or float16 after automatic return;
5. Loss can drop normally and is not NaN;
6. All four Experts can receive token;
7. The training curve appears on the SwanLab web page;
8. `checkpoints/pretrain_moe_smoke/latest.pt` can be generated;
9. `out/pretrain_moe_smoke/` can export the model.

The smoke test directory should be separated from the official training directory to prevent the formal training from loading the test checkpoint by mistake.

---

## 8. single GPU official training order

### 8.1 Recommended complete commands

Write down the key parameters completely, which is convenient for reproduction and inspection:

```bash
python trainer/train_pretrain.py \
  --data-path dataset/pretrain/pretrain.jsonl \
  --tokenizer-path model/tokenizer \
  --use-moe \
  --num-experts 4 \
  --num-experts-per-tok 1 \
  --router-aux-loss-coef 5e-4 \
  --max-seq-len 512 \
  --batch-size 8 \
  --accumulation-steps 8 \
  --epochs 1 \
  --learning-rate 5e-4 \
  --min-learning-rate 5e-5 \
  --warmup-ratio 0.03 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --dtype bfloat16 \
  --gradient-checkpointing \
  --log-interval 10 \
  --eval-interval 500 \
  --save-interval 1000 \
  --tracker swanlab \
  --tracker-project miniLLM-Pretrain \
  --tracker-run-name pretrain-moe4-top1-full-1epoch \
  --tracker-tags moe full-data 1epoch \
  --save-dir checkpoints/pretrain_moe \
  --output-dir out/pretrain_moe
```

### 8.2 Simple commands

Because the vast majority of values are already the default values of the script, only the parameters that determine the nature of the experiment can be overwritten:

```bash
python trainer/train_pretrain.py \
  --data-path dataset/pretrain/pretrain.jsonl \
  --use-moe \
  --gradient-checkpointing \
  --tracker swanlab \
  --tracker-run-name pretrain-moe4-top1-full-1epoch \
  --save-dir checkpoints/pretrain_moe \
  --output-dir out/pretrain_moe
```

The two commands correspond to the same core scheme under the default value of the current code. It is recommended to keep the complete command for formal experiments to avoid the inability to reproduce the experiment after modifying the default parameters in the future.

### 8.3 Why is it recommended to turn on Gradient Checkpointing?

```text
--gradient-checkpointing
```

It will not reduce the model parameters, gradients and AdamW state occupancy, but reduce the preservation of forward activation; some forward results will be recalculated when backpropagation. Therefore:

- Advantages: Significantly reduce the activation of GPU memory;
- Price: the training speed decreases;
- Suitable for: training with limited GPU memory, large batch or long sequence.

The current model parameters themselves are saved in float32. `--dtype bfloat16` controls the calculation accuracy of CUDA autocast, and does not turn all long-term parameters and AdamW status into bfloat16. The weight, gradient and two Adam states of about 200 million parameters occupy about 3.2 GB on the theoretical basis alone, and also superimpose activation, temporary tenser, CUDA cache and framework overhead.

### 8.4 It is not recommended to open `--compile` for the time being

At present, MoE uses dynamic Top-1 routing and expert distribution. `torch.compile` may have graph break, and the first compilation also requires extra time and GPU memory. You should first complete the stable baseline without `--compile`, and then test separately whether it really improves throughput.

---

## 9. multi-GPU training

Two GPUs:

```bash
torchrun --nproc_per_node 2 trainer/train_pretrain.py \
  --data-path dataset/pretrain/pretrain.jsonl \
  --use-moe \
  --gradient-checkpointing \
  --tracker swanlab \
  --tracker-run-name pretrain-moe4-top1-full-1epoch-2gpu \
  --save-dir checkpoints/pretrain_moe \
  --output-dir out/pretrain_moe
```

The current multi-GPU method is DDP:

- Each GPU holds the complete model and optimizer status;
- Different GPUs read different data fragments;
- Synchronous gradient when backpropagation;
- Only rank 0 upload SwanLab, save Checkpoint and export models.

When `num_workers > 0`, DataLoader explicitly uses `spawn` to create Worker to avoid Linux's default `fork` inheriting the initialized CUDA/NCCL state and causing multi-GPU deadlock. `--num-workers 0` does not create sub-processes, so it does not pass into the multiprocessing context.

DDP can increase data throughput, but it will not divide the model parameters and optimizer status into multiple cards like FSDP/ZeRO. Therefore, it cannot solve the problem that "a single GPU can't even release the complete model state".

The effective global batch will increase with the number of GPUs when there are multiple cards. If it expands from 1 card to 2 cards and other parameters remain unchanged:

```text
8 × 8 × 2 = 128 samples/update
```

If you want to keep the global batch at 64, you can halve the batch or cumulative steps per card.

---

## 10. SwanLab Monitoring Scheme

Use for formal training:

```text
--tracker swanlab
--tracker-project miniLLM-Pretrain
--tracker-run-name pretrain-moe4-top1-full-1epoch
--tracker-tags moe full-data 1epoch
```

The training script will record the following indicators.

### 10.1 Basic Training Indicators

| Indicators | Meaning | Main observation methods |
|---|---|---|
| `train/loss` | The sum of LM Loss and router Assisted Loss | Should decline as a whole, and short-term fluctuations are normal |
| `train/lm_loss` | Pure next-token cross-entropy | Core indicators for judging the learning effect of language modeling |
| `train/learning_rate` | Current learning rate | Should Warmup first, and then Cosine attenuation |
| `train/gradient_norm` | The global gradient norm before cutting | Long-term abnormal spikes, NaN need to be investigated |
| `train/tokens_per_second` | Estimated token slot throughput | Used to compare the configuration speed, not the exact and effective number of tokens |
| `train/epoch` | Current training progress | From 0 to 1 |
| `system/gpu_memory_allocated_gb` | PyTorch Current Allocation Of Display Storage | Observe Stable Occupancy And Leakage |
| `system/gpu_memory_reserved_gb` | PyTorch reserved video memory | generally greater than allocated |

### 10.2 Verification Indicators

| Indicator | Meaning |
|---|---|
| `validation/loss` | Pure language model on the validation set Loss |
| `validation/perplexity` | `exp(validation_loss)`, the lower the better |

To judge whether to continue training, you can't just look at training Loss:

```text
Train loss decline + validation loss decline → normal learning
Train loss decrease + validation loss increase → there may be differences in the distribution of overfitting or training/verification
Both do not decline in the long run → Check the learning rate, data and model implementation
Loss/grad norm appears NaN → Check numerical stability and abnormal data
```

### 10.3 MoE router Indicator

| Indicator | Meaning |
|---|---|
| `train/router_aux_loss` | router Load Balancing Auxiliary Loss |
| `moe/expert_i_usage` | The ratio of the ith expert actually receiving token |
| `moe/expert_i_router_probability` | The average probability of router assigned to the ith expert |
| `moe/router_entropy_normalized` | The normalized entropy of router probability distribution |
| `moe/max_load_ratio` | The proportion of token of the busiest expert |
| `moe/min_load_ratio` | The proportion of token of the most idle expert |

When 4 Experts are completely uniform, each usage is close to:

$$
1/4=0.25
$$

It is not required that each step is strictly equal to 0.25, but if one expert is close to 0 and another is close to 1 for a long time, it means that the route may collapse. It is necessary to judge in combination with router auxiliary loss, training Loss and verification Loss. You can't blindly increase the auxiliary loss coefficient just for the sake of curve uniformity.

### 10.4 Offline mode

When the server is temporarily unable to access the external network, it can be used:

```text
--tracker swanlab --tracker-mode offline
```

The log will be written to `logs/pretrain`. How to synchronize offline data is subject to the currently installed SwanLab SDK command.

---

## 11. Checkpoint and checkpoint resume training

### 11.1 What is saved?

`checkpoints/pretrain_moe/latest.pt` contains:

- Model weight;
- AdamW optimizer status;
- Learning rate dispatcher status;
- GradScaler status;
- Current epoch, batch and global step;
- Model configuration and training parameters;
- Python, NumPy, PyTorch, CUDA random number status;
- SwanLab run ID and other monitoring status.

It is a "continuing training file", not the final model recommended for publication or reasoning.

At present, the implementation of repeatedly overwriting a `latest.pt` will not automatically retain the historical version for every 1000 steps. It saves disk, but cannot go back to the earlier best step. If you need to save multiple milestone versions, you need to extend the naming strategy later.

### 11.2 Automatic recovery

Use the same structural parameters, data and training plans to add at the end of the original command:

```text
--resume
```

Complete example:

```bash
python trainer/train_pretrain.py \
  --data-path dataset/pretrain/pretrain.jsonl \
  --use-moe \
  --gradient-checkpointing \
  --tracker swanlab \
  --tracker-run-name pretrain-moe4-top1-full-1epoch \
  --save-dir checkpoints/pretrain_moe \
  --output-dir out/pretrain_moe \
  --resume
```

If it exists, it will read:

```text
checkpoints/pretrain_moe/latest.pt
```

And try to restore the same SwanLab run, so that the curve continues to be written in the original experiment. If the file has not been generated (such as the first start-up training), the prompt will be output and automatically started from the beginning.

### 11.3 Specify the checkpoint file

```bash
python trainer/train_pretrain.py \
  ... \
  --resume checkpoints/pretrain_moe/latest.pt
```

When the checkpoint is restored, the model structure must be kept consistent. For example, `--use-moe`, expert number, Top-K, hidden dimension and layer number cannot be changed; the total number of training steps must also be greater than `global_step` in the checkpoint.

---

## 12. The difference between Checkpoint and the final export model

After the successful training, there will be two types of products:

```text
checkpoints/pretrain_moe/latest.pt
out/pretrain_moe/
```

| Product | Main Content | Purpose |
|---|---|---|
| `latest.pt` | Model, optimizer, scheduler, training position, random state | checkpoint continuation training |
| `out/pretrain_moe/` | `config.json`, `model.safetensors`, tokenizer File | Evaluation, Reasoning, Follow-up SFT, Release |

Subsequent pre-training evaluation and SFT should give priority to reading `out/pretrain_moe/`. Only when the training is interrupted and the final catalog has not been exported, `latest.pt` is directly used for temporary evaluation.

---

## 13. Evaluation after completion

### 13.1 Verify Loss, PPL and text continuation

```bash
python eval/eval_llm.py \
  --model-path out/pretrain_moe \
  --data-path dataset/pretrain/pretrain.jsonl \
  --mode both \
  --max-seq-len 512 \
  --eval-samples 2048 \
  --batch-size 8
```

### 13.2 Only evaluate the language model Loss

```bash
python eval/eval_llm.py \
  --model-path out/pretrain_moe \
  --data-path dataset/pretrain/pretrain.jsonl \
  --mode loss
```

### 13.3 Use the specified prompt to generate

```bash
python eval/eval_llm.py \
  --model-path out/pretrain_moe \
  --mode generate \
  --prompt "The development of artificial intelligence" \
  --prompt "def quick_sort("
```

The pre-training model has only completed next-token learning, and has not undergone special Chat SFT and preference alignment. It may be better at "continuing the text" and may not be able to follow the dialogue instructions stably. The standard of the mature chat model cannot be used to directly judge whether the pre-training has failed.

It is recommended to evaluate from at least four dimensions:

1. **Loss/PPL**: Language modeling indicators on the fixed validation set;
2. **Continuation quality**: Chinese and English, knowledge, code and a variety of prompt;
3. **Memory and Generalization**: Avoid only retelling training samples;
4. **MoE routing**: Whether expert is balanced for a long time and without collapse.

---

## 14. Common problems and processing order

### 14.1 CUDA Out of Memory

Handle it in the following order:

1. Confirm that `--gradient-checkpointing` has been used;
2. Downgrade `--batch-size 8` to 4 or 2;
3. Add `--accumulation-steps` accordingly to maintain an effective batch;
4. Confirm that no other processes occupy the GPU;
5. Then consider shortening `--max-seq-len`;
6. Finally, consider narrowing the model structure.

Don't expect Top-1 MoE to automatically reduce the GPU memory to the 65M Dense model level. The main MLP calculation is not done without the activation of expert, but its weight and optimizer status still exist.

### 14.2 Training Loss becomes NaN

Check:

- Whether the learning rate is too high;
- Whether `gradient_norm` has an abnormal peak first;
- Whether the GPU really supports bfloat16;
- Whether there is extreme or damaged text;
- Whether the mismatched checkpoint was restored by mistake.

You can first reduce the learning rate to `3e-4` for control experiments. Don't just look at the peak once, you should combine the trend of dozens of steps before and after.

### 14.3 SwanLab No data

Check whether the startup output contains:

```text
Tracker : swanlab (run_id=...)
```

Then check:

- Are the parameters indeed `--tracker swanlab`;
- Whether to complete the SwanLab certification;
- Whether the server can access the external network;
- Whether `--tracker-mode offline` was misused;
- Whether `swanlab` is installed in the current Python environment;
- Whether the log of rank 0 is being viewed during the multi-GPU.

### 14.4 Unable to recover after training interruption

Check:

- Whether `checkpoints/pretrain_moe/latest.pt` exists;
- Is `--save-dir` exactly the same as last time;
- Whether `--use-moe` is still added;
- Whether the expert number, Top-K, hidden dimension and layer number are consistent;
- Whether the total steps of the current plan is greater than the checkpoint global step.

### 14.5 Slow generation speed

The current compact version has not yet implemented KV Cache, and the historical context will be calculated repeatedly when generating a new token. This does not affect the correctness of pre-training Loss, but it will reduce the speed of autoregressive reasoning. KV Cache belongs to the subsequent reasoning optimization task.

---

## 15. Boundaries and follow-up improvements of the current plan

The current implementation is suitable for learning complete pre-training links from scratch, but it is not a mass production training framework. The main boundaries include:

1. **No sequence packing**: Short sample Padding wastes computing power;
2. **Data is truncated by line**: The tail of more than 510 body tokens will be discarded, and the sliding window will not be continued;
3. **validation set comes from the same file**: suitable for training monitoring, not equal to independent benchmark evaluation;
4. **MoE has no capacity factor**: token will not be lost, but the routing implementation does not target ultra-large-scale throughput;
5. **DDP non-cut state**: Each card stores about 200 million complete parameter models;
6. **Only keep the latest checkpoint**: The best model in history cannot be automatically selected;
7. **No KV Cache**: The speed of generating evaluation is slow;
8. **Data bias Text-to-Text/instruction form**: The model ability will be significantly affected by the current data distribution;
9. **Only one round of training**: It is a compromise between the current computing power and complete data coverage, and does not represent the optimal number of rounds in theory.

Recommended order of improvement:

```text
Complete a stable baseline first.
  ↓
Fixed validation set compares Dense and MoE
  ↓
Join independent verification sets and data quality statistics
  ↓
Realize sequence packing / document cutting
  ↓
Save multiple milestones and choose the best model
  ↓
Realize KV Cache
  ↓
Introduce FSDP / DeepSpeed when a larger scale is needed
```

---

## 16. The recommended complete execution process

### Stage One: Environmental Inspection

```bash
cd /root/miniLLM
pip install -r requirements.txt
python trainer/train_pretrain.py --help
```

### Stage 2: MoE Smoke Test

Use `pretrain_mini.jsonl`, `--max-train-samples 4096`, `--max-steps 20` to confirm that training, verification, saving and SwanLab can all work.

### Stage 3: Display memory and throughput test

Try step by step:

```text
batch 2 × accumulation 32
batch 4 × accumulation 16
batch 8 × accumulation 8
```

Choose the best combination of non-OOM and tokens/s.

### Stage 4: Formal full training

Use `pretrain.jsonl`, 4 expert, Top-1, 512 length, 1 round, and write the official Checkpoint and Output into the independent directory.

### Stage 5: Monitoring during training

Key observation:

```text
train/lm_loss
validation/loss
train/gradient_norm
system/gpu_memory_allocated_gb
moe/expert_0_usage ... moe/expert_3_usage
moe/max_load_ratio / min_load_ratio
```

### Stage 6: Evaluation after training

Use `eval/eval_llm.py` to calculate and verify Loss and PPL, and conduct multi-type continuation tests such as Chinese, English and code.

### Stage 7: Enter SFT

After confirming that the pre-training model can carry out basic coherent continuation, verify that the Loss is reasonable, and the MoE routing does not collapse, then use `out/pretrain_moe/` as the initialization weight of the next stage.

---

## 17. Final plan card

```text
Project name miniLLM
Training Objectives Training Decoder-only base model from Random Weights
tokenizer       ByteLevel BPE, vocab=8192
Training data pretrain.jsonl, with a total volume of about 8.469 million
Number of training rounds 1 epoch
Number of model layers 8
Hidden dimension 768
attention       GQA, 8 Q Heads / 4 KV Heads
Location code RoPE, theta=1,000,000
FFN             Sparse MoE + SwiGLU
Experts         4
Top-K           1
Total parameters 199,791,360 (about 199.8M)
Activation parameters 65,311,488 (about 65.3M)
Sequence length 512
Micro batch     8 / GPU
Gradient accumulation 8
single GPU valid batch 64 samples/update
Optimizer AdamW, betas=(0.9, 0.95)
Learning rate 5e-4 → 5e-5
Warmup          3%
Accuracy bfloat16 autocast
Gradient Checkpointing for GPU memory optimization
Training and monitoring SwanLab
checkpoint directory checkpoints/pretrain_moe/latest.pt
Final model out/pretrain_moe/
```

The core goal of this program is to first run a complete MoE pre-training in an understandable, monitored and recoverable engineering structure, rather than pursuing an industrial scale at once. After completing this round, Loss, PPL, sample quality, expert routing and training throughput will become the practical basis for the next round of adjusting data, models and training budgets.

---

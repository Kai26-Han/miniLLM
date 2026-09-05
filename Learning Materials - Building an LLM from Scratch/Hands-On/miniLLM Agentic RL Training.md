# miniLLM Agentic RL Training

> Stage: train multi-turn tool use on a miniLLM actor that has completed PPO

> Default recipe: a multi-turn tool environment, GRPO, four trajectories per prompt, and a frozen copy of the PPO actor as the reference model

---

## 1. What does Agentic RL solve?

Standard PPO mainly optimizes the quality of a single response. Agentic RL instead places the model in an interactive environment:

```text
User request
    ↓
assistant generates <tool_call>
    ↓
Environment validates and executes the tool
    ↓
Tool observation is added to the context
    ↓
assistant continues to call the tool or give the final answer
    ↓
Ground-truth verifier computes the trajectory reward
```

The training goal is not only "the answer looks reasonable", but also includes:

- Whether the available tools have been selected;
- Whether tool call JSON and parameters are valid;
- Whether the tool is executed successfully;
- Whether to give the correct answer according to the tool results;
- Can it be stopped within a limited number of rounds;
- Whether there is duplication, invalidity or reward speculation.

This project refers to the centralized implementation of [MiniMind Agentic RL](https://github.com/jingyaogong/minimind/blob/master/trainer/train_agent.py), putting tools, Rewards, multiple rounds of Rollout, GRPO/CISPO and training cycles in one entrance, while supplementing security mathematical execution, strict token Mask, validation set and checkpoint continuation training.

## 2. Complete training sequence

```text
out/pretrain
    ↓ SFT
out/sft
↓ DPO (optional)
out/dpo
    ↓ PPO
Out/ppo or checkpoints/ppo/latest.pt
├── Trainable Agent Policy
└── Freeze Reference Policy
              ↓
Multi-round Tool-Use GRPO
              ↓
           out/agent
```

Agentic RL should start with PPO Actor who already has the basic tool call ability. If the model after PPO can hardly output the legal `<tool_call>`, the same group trajectory may all fail, and GRPO will not get a valid intra-group difference. Before formal training, you should run the `--eval-only` check tool to call the baseline.

## 3. Related documents

| File | Function |
|---|---|
| [`trainer/train_agent.py`](../../trainer/train_agent.py) | Tool environment, multi-round Rollout, Reward, GRPO/CISPO, training, evaluation and export |
| [`dataset/ppo_dataset.py`](../../dataset/ppo_dataset.py) | `AgentRLDataset`, verifiable task filtering, stable cutting and Collator |
| [`trainer/rollout_engine.py`](../../trainer/rollout_engine.py) | Tokenized single-round generation, original tool call text and behavior strategy log probability |
| [`dataset/rl/agent_rl.jsonl`](../../dataset/rl/agent_rl.jsonl) | Agentic RL data in MiniMind format |
| [`tests/test_agent_rl.py`](../../tests/test_agent_rl.py) | Data, Tools, Reward, Track Mask, Objective and PPO Actor Loading Test |

This project uses native PyTorch and does not rely on TRL.

## 4. Data format

Each line contains `conversations` and `gt`. The last message must be an empty assistant placeholder:

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "",
      "tools": "[{\"type\":\"function\",\"function\":{\"name\":\"calculate_math\",\"parameters\":{\"type\":\"object\",\"properties\":{\"expression\":{\"type\":\"string\"}},\"required\":[\"expression\"]}}}]"
    },
    {"role": "user", "content": "Calculation 570*8156"},
    {"role": "assistant", "content": ""}
  ],
  "gt": ["4648920"]
}
```

When the data is loaded, it will:

1. Delete the last empty assistant placeholder;
2. Parse the tool definition in the system message;
3. Keep the original JSONL line number for incorrect positioning;
4. Only select tools non-empty and `gt` non-empty verifiable tasks;
5. Do deterministic training/verification cutting according to `--seed`.

### 4.1 Current data statistics

At present, `agent_rl.jsonl` has a total of 39,988 articles:

| Type | Quantity | Whether to use in the first stage |
|---|---:|---|
| With tools and GT non-empty | 20,000 | Yes |
| Open Q&A and GT is empty | 19,988 | No |

The default is `--val-ratio 0.02`, so the verifiable task is divided into:

| Split | Quantity |
|---|---:|
| Train | 19,600 |
| Validation | 400 |

Open Q&A does not have an objective Verifier and requires an additional reward model. At this stage, it should not be mixed into the tool training to avoid the model getting false rewards through length, format or fixed rhetoric.

### 4.2 Tool Types

The training environment supports six tools consistent with the current data:

- `calculate_math`: Basic Mathematical Expressions;
- `unit_converter`: length, weight and temperature conversion;
- `get_current_weather`: Deterministic simulated weather;
- `get_current_time`: Simulation time of certainty;
- `get_exchange_rate`: Deterministic simulated exchange rate;
- `translate_text`: Fixed translation covered by the data set.

Mathematical tools use AST whitelists, and only numbers, brackets and basic operators are allowed, and Python names, functions, attributes or indexes are not executed. All tools only accept the whitelist name provided by the current task, and the error will be returned to the model as a structured observation so that it can be corrected in the next round.

## 5. Multiple Trajectories and the Loss Mask

Each trajectory runs for at most `--max-turns` turns. The default is 3:

```text
prompt                       mask=0
assistant tool_call          mask=1
Tool observation             mask=0
assistant tool_call          mask=1
Tool observation             mask=0
assistant final answer       mask=1
```

Only the assistant token actually generated by the model participates in GRPO/CISPO Loss. The following contents are not involved:

- system/user history;
- assistant generation header;
- Tool observation inserted in the training environment;
- Environmentally supplemented round separator;
- padding.

The old log-probability of the behavior policy will be saved in each round of generation. In the future, assemble all rounds into a complete trajectory, and then align them with the action mask. In this way, Tool observation can change the next round of decision-making, but it will not be regarded as a model behavior calculation gradient.

## 6. Reward Design

The total reward consists of the following parts and is cropped to `[-3, 3]`:

```text
total_reward
  = outcome_reward
  + tool_reward
  + format_reward
  + efficiency_reward
```

### 6.1 Outcome Reward

The proportion of GT covered by the final answer is the main reward:

```text
outcome_reward = 2.5 × matched_gt / total_gt
```

You must successfully execute the tool at least once to get the Outcome Reward. The model directly guessed the answer without calling the tool, which is not counted as Agent's success.

Numerical GT uses absolute/relative tolerance matching, and string GT uses normalized matching. The GT hit in the tool observation will be recorded separately, but the complete Outcome Reward still requires the final assistant answer to include the result.

### 6.2 Tools and Format Rewards

- Successfully analyze and execute the tool to get positive rewards;
- Points will be deducted for tools that are not on the task whitelist;
- Points will be deducted for illegal parameters, execution failure and damaged JSON;
- `<tool_call>` label is not closed and points are deducted;
- If the maximum number of rounds is reached, the final answer will still be deducted.

### 6.3 Efficiency Reward

- Repeat the same tool call to deduct points;
- Points will be deducted for invalid calls;
- Points will be deducted when there is a duplicate tria in the final answer.

The number of tools required for different tasks is not necessarily equal to the number of GT, so it is not mechanically required that "the number of calls must be equal to the number of GT".

## 7. GRPO and CISPO

Use GRPO by default. For the same prompt sampling `G` complete Agent trajectory:

```text
advantage_i = (reward_i - group_mean) / (group_std + eps)
```

Then use the same trajectory Advantage for all assistant action tokens, and add the KL constraint to freeze PPO Reference:

```text
loss = clipped_policy_loss + beta × sampled_kl
```

If a group of trajectory rewards is exactly the same and the standard deviation in the group is 0, the group cannot provide a relative learning signal. Trainer will:

- Set the Group Advantage To Zero;
- Block the group's policy and MoE router updates;
- Record the effective group ratio through `effective_group_rate`.

You can switch to MiniMind-style CISPO Objective through `--loss-type cispo`. It is recommended to establish a stable GRPO baseline first, and then conduct a CISPO control experiment.

## 8. Load PPO Actor

Support two input methods.

### 8.1 Transformers Actor Catalog

```bash
python trainer/train_agent.py \
    --model-path out/ppo \
    --data-path dataset/rl/agent_rl.jsonl
```

The catalog should at least contain:

```text
out/ppo/
├── config.json
└── model.safetensors
```

### 8.2 Native PPO Checkpoint

```bash
python trainer/train_agent.py \
    --ppo-checkpoint checkpoints/ppo/latest.pt \
    --data-path dataset/rl/agent_rl.jsonl
```

Checkpoint must contain the saved by the PPO trainer:

- `actor_model`;
- `actor_config`.

Agentic RL does not load PPO Critic. Copy PPO Actor at the beginning of the training:

```text
Policy = PPO Actor, all parameters can be trained
Reference = PPO Actor Freeze Copy
```

## 9. Run the baseline evaluation first

Before formal training, check the tool ability of PPO Actor:

```bash
python trainer/train_agent.py \
    --model-path out/ppo \
    --eval-only \
    --eval-samples 128 \
    --max-turns 3 \
    --max-new-tokens 256 \
    --tracker none
```

The evaluation uses greedy decoding and fixed thinking to ensure that the results of different training stages can be compared.

Key observation:

- `task_success_rate`;
- `gt_match_rate`;
- `valid_tool_call_rate`;
- `tool_execution_success_rate`;
- `unfinished_rate`;
- `parse_errors_per_sample`.

If the legal tool call rate is close to 0, Tool-Call SFT should be supplemented first, instead of directly increasing the number of Agent RL steps.

## 10. Smoking training

First, use a small amount of data to check the GPU memory, trajectory and save links:

```bash
python trainer/train_agent.py \
    --model-path out/ppo \
    --max-train-samples 20 \
    --eval-samples 8 \
    --max-steps 2 \
    --batch-size 1 \
    --num-generations 2 \
    --max-new-tokens 128 \
    --eval-interval 2 \
    --save-interval 2 \
    --tracker none
```

Confirm the following results:

- Loss, Reward, KL and gradient norms are all finite values;
- `mean_action_tokens` is greater than 0;
- `checkpoints/agent/latest.pt` was saved successfully;
- `out/agent` SUCCESSFULLY EXPORTED;
- Observation token did not enter action loss.

## 11. Formal training

single GPU default training:

```bash
python trainer/train_agent.py \
    --model-path out/ppo \
    --data-path dataset/rl/agent_rl.jsonl \
    --loss-type grpo \
    --num-generations 4 \
    --batch-size 2 \
    --accumulation-steps 1 \
    --learning-rate 3e-7 \
    --beta 0.05 \
    --max-prompt-len 1024 \
    --max-total-len 2048 \
    --max-new-tokens 256 \
    --max-turns 3 \
    --epochs 1 \
    --tracker swanlab
```

Dual card training:

```bash
torchrun --nproc_per_node 2 trainer/train_agent.py \
    --model-path out/ppo \
    --batch-size 2 \
    --num-generations 4 \
    --learning-rate 3e-7 \
    --tracker swanlab
```

The actual number of trajectories generated by a Rollout batch is:

```text
trajectories = batch_size × num_generations × number of GPUs
```

The single-GPU default is `2 × 4 = 8` trajectories. Most of the cost of agent training comes from multi-turn generation rather than the backward pass, so increasing `num_generations` substantially increases training time.

## 12. Default parameters

| Parameters | Default Value | Description |
|---|---:|---|
| `--loss-type` | `grpo` | `grpo` or `cispo` |
| `--num-generations` | 4 | number of trajectories of each prompt |
| `--batch-size` | 2 | prompt number per card |
| `--learning-rate` | `3e-7` | Policy Learning Rate |
| `--beta` | `0.05` | Reference KL coefficient |
| `--epsilon` | `0.2` | GRPO Ratio Clip |
| `--epsilon-high` | `5.0` | CISPO Upper Bound |
| `--max-turns` | 3 | Maximum assistant Number of Rounds |
| `--max-prompt-len` | 1024 | Initial prompt Upper Limit |
| `--max-total-len` | 2048 | Full trajectory upper limit |
| `--max-new-tokens` | 256 | Maximum generation length per round |
| `--temperature` | `0.8` | Training Rollout Temperature |
| `--top-p` | `0.9` | Nucleus Sampling |
| `--top-k` | 50 | Top-K Sampling |
| `--thinking-ratio` | `0.1` | Open the trajectory ratio of explicit thinking |
| `--val-ratio` | `0.02` | validation set ratio |

## 13. Training indicators

| Indicator | Meaning |
|---|---|
| `loss` | The sum of Policy, KL and effective MoE router auxiliary losses |
| `policy_loss` | GRPO/CISPO Strategy Loss |
| `kl` | Sampled KL of PPO Reference is relatively frozen in the current Policy |
| `reward` | Current trajectory average total reward |
| `task_success_rate` | The ratio of successfully calling tools, covering all GT and ending normally |
| `effective_group_rate` | Reward has intra-group differences and can provide gradient prompt ratio |
| `clip_fraction` | Importance Ratio cropped action token ratio |
| `mean_ratio` | The average probability ratio of current strategy and behavior strategy |
| `mean_action_tokens` | Number of assistant tokens involved in Loss in each trajectory |
| `grad_norm` | Gradient standard before cropping |

### 13.1 How to judge whether training is healthy

- `task_success_rate` and `gt_match_rate` should be gradually improved;
- `valid_tool_call_rate` should not continue to decline;
- `effective_group_rate` cannot be close to 0 for a long time;
- KL should be kept limited and cannot explode quickly and monotonously;
- `unfinished_rate` and the analysis error rate should be reduced;
- When Reward rises, the success rate should be checked synchronously to prevent only learning the format reward.

## 14. checkpoint retraining and output

Automatic recovery:

```bash
python trainer/train_agent.py \
    --model-path out/ppo \
    --resume
```

Specify Checkpoint:

```bash
python trainer/train_agent.py \
    --model-path out/ppo \
    --resume checkpoints/agent/latest.pt
```

The training status is saved to:

```text
checkpoints/agent/latest.pt
```

It includes Policy, Optimizer, Scheduler, AMP Scaler, training position, random number status and experimental record information. Frozen Reference is not saved twice, and it is still built from the same PPO Actor when restored.

The final Transformers model is exported to:

```text
out/agent/
├── config.json
├── model.safetensors
├── tokenizer.json
└── chat_template.jinja
```

## 15. Frequently asked question

### 15.1 `effective_group_rate` long-term close to 0

It shows that all the trajectories of the same prompt are almost always the same prize:

- All failed: first check whether PPO Actor has the basis of tool call;
- All success: these tasks are no longer necessary to continue training;
- Generation is too certain: increase the temperature appropriately or increase `num_generations`;
- Insufficient reward distinction: check each Reward sub-item and GT verifier.

### 15.2 Insufficient GPU memory

Reduce expenses in the following order:

1. Reduce `--batch-size`;
2. Reduce `--num-generations`;
3. Reduce `--max-new-tokens`;
4. Reduce `--max-total-len`;
5. Open `--gradient-checkpointing`;
6. Add `--accumulation-steps` compensation effective batch.

### 15.3 Low legal call rate

Check whether the model outputs the following format:

```text
<tool_call>
{"name": "calculate_math", "arguments": {"expression": "2+2"}}
</tool_call>
```

Common errors include: the tool name is not on the current whitelist, `arguments` is not JSON Object, the label is not closed, or the model outputs incomplete JSON outside `<tool_call>`.

### 15.4 Reward increases but the mission success rate does not increase

This usually means that the model only improves the format or tool execution, but does not write the observation into the final answer. Key points: Compare `reward`, `outcome_reward`, `gt_match_rate` and `task_success_rate`, do not only choose the model according to the total Reward.

### 15.5 Project `.venv` Cannot Import PyTorch

First, press [`requirements.txt`](../../requirements.txt) to install dependencies. The CUDA server should give priority to retaining the CUDA compatible PyTorch that comes with the mirror, or install the official Wheel corresponding to the CUDA version of the server.

## 16. Test

Run Agent directional test:

```bash
python -m unittest tests.test_agent_rl tests.test_ppo -v
```

Run the complete regression test:

```bash
python -m unittest discover -s tests -v
```

Agent test covers security tool execution, multi-round observation, assistant-only action mask, GT Reward, zero variance group, GRPO gradient, PPO checkpoint loading and existing PPO Rollout compatibility.

# Supervised Fine-Tuning for LLMs: From Instruction Data to a Capable Chat Model

> Supervised Fine-Tuning (SFT) is the starting point of modern large model post-training. It uses high-quality demonstration to turn the "Continuation base model" into "assistant who can complete tasks according to the agreement". This article takes the mainstream algorithms, data and engineering trade-offs of the industry as the core, and finally briefly describes the implementation of miniLLM.

## 1. The position of SFT in the life cycle of large models

A typical training pipeline is:

```text
tokenizer
  -> Pretraining
-> Continued Pretraining / Mid-training (optional)
  -> SFT
-> Preference Optimization (DPO / RLHF, etc.)
-> Reasoning RL / RLVR (optional)
-> Safety, Eval and Deployment
```

The problems solved at each stage are different:

| Stage | Main Data | Main Objective |
|---|---|---|
| Pre-training | Large-scale natural text, code, multimodal data | Learning knowledge, language and general representation |
| Continuous pre-training | Domain primitive corpus, long context corpus | Supplementary knowledge, domain and context distribution |
| SFT | Instruction - Answer, Multi-round Dialogue, Tool Trajectory | Learning Task Protocol and Target Behavior |
| Preference Optimization | Chosen/Rejected or Reward | Bias for better behavior in multiple feasible answers |
| Reasoning RL | Verifiable Task and Environmental Feedback | Extended Search, Reasoning and Agent Capabilities |

SFT is mainly responsible for "how to demonstrate how to answer". It can activate and organize pre-training ability, and can also teach the model new formats and some skills, but it is not suitable to replace the missing basic knowledge with a small number of demonstrations.

## 2. Conditional language modeling goals of SFT

The context is \(c\), and the target answer is \(y=(y_1,\ldots,y_T)\). SFT maximization:

\[
p_\theta(y\mid c)=\prod_{t=1}^{T}p_\theta(y_t\mid c,y_{<t})
\]

The corresponding token-level Cross-Entropy is:

\[
L_{SFT}=-\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(y_t\mid c,y_{<t})
\]

It uses the same Next-token Prediction kernel as pre-training. The core difference lies in data distribution, sequence templates and which positions participate in Loss.

## 3. Loss Mask: Which tokens should be supervised

A dialogue sequence can be written as:

```text
<system> ...
<user> ...
<assistant> ...
<user> ...
<assistant> ...
```

\(s=(s_1,\ldots,s_n)\) is a sequence, and \(m_t\in\{0,1\}\) indicates whether the position participates in Loss:

\[
L=-\frac{\sum_{t=1}^{n}m_t\log p_\theta(s_t\mid s_{<t})} {\sum_{t=1}^{n}m_t}
\]

### 3.1 Full-sequence Loss

All non-Padding tokens such as System, User, assistant, etc. participate in the prediction.

The advantage is that the supervision density is high, and the complete dialogue format can also be learned; the disadvantage is that the model spends capacity to imitate user input, and may overfit the fixed System prompt or template.

### 3.2 Response-only / assistant-only Loss

System, User, Tool Result only as context, assistant content, tool call and end tag participate in Loss. This is a common choice for dialogue SFT, because the training goal is consistent with the area that needs to be generated at the time of deployment.

### 3.3 All assistant rounds or the last round

Multiple rounds of dialogue can supervise all assistant messages, or only the last round can be supervised:

- Supervise all assistant rounds, and the token utilization rate is higher;
- Only supervise the last round, which is convenient to construct the sample of "complete history -> current answer";
- If the quality of the historical assistant's answers is unstable, using them all as targets will spread errors.

There is no single answer from the data source. The key is to explicitly record the Mask strategy and be consistent with Eval.

## 4. chat template is part of the training protocol.

The original `messages` must be serialized as the token stream seen by the model:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Explain what gradient descent is. <|im_end|>
<|im_start|>assistant
Gradient descent is an optimization method... <|im_end|>
```

The template is defined at least:

- Role name and message boundary;
- BOS/EOS and assistant end tags;
- Placement of System prompt;
- Multi-round line change rules;
- The structure of Reasoning, tool call and Tool Result;
- Whether to join Generation prompt at inference time;
- Which tokens participate in the training Loss.

Inconsistency between training and inference template will cause the model to see a strange prefix, unable to stop, character string or tool JSON to fail. Template files, tokenizer and model weights should be released as the same version.

## 5. Where does SFT data come from?

### 5.1 Manual demonstration

Experts or annotators directly write target answers, which are suitable for security policies, professional fields and complex styles. The advantage is that the intention is clear, but the price is expensive, slow and there is a labeler preference.

The public process of InstructGPT first collects the SFT model of the labeler's demonstration training, and then uses preference comparison to enter the reward model and PPO stages. This establishes the classic paradigm that "SFT is the starting point for subsequent preference optimization".

### 5.2 Traditional tasks are turned into instructions

Rewrite data sets such as classification, extraction, translation, summary, questions and answers into natural language instructions. FLAN series shows that expanding the number of tasks, templates, model scale and adding Chain-of-Thought data can improve the generalization of unseen tasks.

The risk is that the template is too strong: a large number of nearly repeated instructions for the same task are not equal to the diversity of real tasks.

### 5.3 Human-computer dialogue and product data

The real request distribution can cover the problems that users really care about, but it must deal with:

- Privacy and personal information;
- User authorization and data licensing;
- Low-quality, aggressive and incomplete dialogue;
- Self-imitation caused by the output of the old model;
- Online distribution of bias for minorities and high-frequency scenes.

### 5.4 Teacher Distillation

Use stronger models to generate answers, explanations or tool trajectories, and then train smaller models to imitate. It can expand the scale of data, but the upper limit of Student will be affected by Teacher errors, single style and knowledge cut-off.

### 5.5 Self-Instruct and iterative synthesis

The typical process is:

1. Starting from a small number of Seed Tasks;
2. Let the model generate new instructions;
3. Generate corresponding inputs and answers;
4. Remove invalid, similar and high-risk samples;
5. Train the model with retained data;
6. Iterative expansion difficulty and task coverage.

The real value of synthetic data comes from "generation + filtering + verification", rather than directly adding all the generated results to the training.

## 6. Data quality is usually more important than indiscriminate expansion.

LIMA uses a small amount of carefully screened data to demonstrate the strong effect of high-quality demonstration; this does not mean that all models only need very little data, but shows that repetition, contradiction and low-quality answers may be more harmful than insufficient data.

### 6.1 Hard verification

- The Role sequence is legal, and there is at least one assistant target;
- Special tags, JSON, code and tool parameters can be parsed;
- The answer was not wrongly stitched or truncated;
- token is still effectively supervised after coding;
- The length, language and data source fields are complete.

### 6.2 deduplication and contamination control

It should be precise/approximately de-weighted at multiple levels of prompt, Response and complete dialogue, and check for overlap with capability Benchmark, code test set and release Eval.

### 6.3 Quality score

Can be combined:

- Rules: garbled code, template residue, repetition, abnormal length;
- Verifier: code test, math answer, JSON Schema, reference verification;
- LLM Judge: relevance, integrity, style and security;
- Manual audit: complex context, boundary policy and Judge calibration.

### 6.4 Diversity

High quality does not mean that all answers are long, pointed and have the same style. Need to cover: short answers and long answers, formal and spoken, different languages, rejection and normal answers, simple tasks and complex tasks.

## 7. Data mixing and sampling algorithm

SFT usually mixes general dialogue, knowledge, mathematics, code, multilingual, long context, tools and security data. If the domain \(i\) has \(n_i\) samples, temperature sampling can be used:

\[
q_i=\frac{n_i^\alpha}{\sum_j n_j^\alpha}
\]

When \(\alpha<1\), the small data domain will be upgraded. It is also necessary to decide whether to balance according to the "sample number" or "target token number", because a long reasoning answer may be equal to the gradient of dozens of short answers.

### 7.1 token-level average length bias

The standard token average will allow the long answer to contribute more gradients. If you want the weight of each sample to be close, you can first calculate the average Loss of each sample, and then average it in the batch:

\[
L_{example}=\frac{1}{B}\sum_{i=1}^{B} \frac{\sum_tm_{it}\ell_{it}}{\sum_tm_{it}}
\]

The goals of optimization of the two weighting methods are different, which should be selected according to the product distribution and explained in the experiment.

### 7.2 Curriculum and Stage Formula

Common ideas include:

- First establish a general instruction to follow, and then add high-difficulty reasoning or tool data;
- Short context first, and then long context SFT;
- First, highly trust the artificial data, and then add the screened synthetic data;
- Improve the weight of the target area at the end of the training, but keep the general data to prevent forgetting.

Whether the order is better than a mixture needs to be verified by ablation, not just by intuition.

## 8. Multiple rounds of dialogue, truncation and Packing

### 8.1 Multiple rounds of samples

Multiple rounds of SFT should cover pointing, error correction, state maintenance, constraint continuation and Tool Result backfilling. Mechanically splicing unrelated single-wheel samples cannot produce real multi-round capabilities.

### 8.2 Truncation strategy

When a long conversation exceeds the upper limit, the common strategy is:

- Keep System prompt;
- Delete the earliest complete round;
- Keep the latest User requests and target assistant answers;
- Avoid forging EOS after being truncated;
- Record the prompt/Answer truncation rate.

Hard truncation only from the right may delete the target answer; hard truncation only from the left may damage System prompt and tool definitions.

### 8.3 Dynamic Padding

batch Padding to the longest sequence of this batch, not the global maximum length. If the longest length is \(S_{max}\), the efficiency is:

\[
U=\frac{\sum_iS_i}{B\cdot S_{max}}
\]

The length of the barrel can further improve \(U\).

### 8.4 SFT Packing

Multiple short conversations can be packed into the same sequence. Need to guarantee:

- There is a clear EOS between the samples;
- Position ID and attention Mask conform to the design;
- No unintentional cross-sample attention;
- Loss Mask does not be misaligned across borders;
- Statistics can still restore the contribution of each domain and each sample.

Packing improves throughput, but its Mask complexity is higher than that of ordinary dynamic Padding.

## 9. Reasoning SFT

Reasoning SFT uses demonstrations with reasoning processes, such as mathematical derivation, code planning or tool decision-making. It can teach the model to develop intermediate steps, but "longer text" does not equal "stronger reasoning".

### 9.1 Data construction

High-quality Reasoning data usually comes from:

- Derivation by human experts;
- Strong Teacher generates multiple candidates;
- Screening with answers, unit tests or formalized verifiers;
- Repair or discard the failed trajectory;
- Organize tasks from easy to difficult.

### 9.2 Mix of long thinking and short answers

If all samples use long Chain-of-Thought, the model may over-reason on simple problems, delay elevation and expose unreliable processes. The industry gradually adopts a mixed thinking formula: complex tasks provide reasoning trajectories, simple tasks answer directly, and distinguish patterns through special marking or control conditions.

The public route of Qwen3 includes long CoT cold start, reasoning RL, Thinking/Non-thinking fusion and general RL, indicating that SFT/cold start is only responsible for establishing the initial reasoning format, and capability expansion also depends on subsequent verifiable training.

### 9.3 Don't use training Loss to judge the quality of reasoning

The model can imitate the wrong reasoning text well to obtain low Loss. It must be evaluated with the final answer, execution results, step consistency and cross-difficulty generalization.

## 10. Tool Calling and Agent-Trajectory SFT

A tool round is usually:

```text
System + Tool Definitions
  -> User Request
  -> assistant tool call
  -> Tool Result
  -> assistant Final Answer
```

In training, Tool Definition, User and Tool Result are often taken as the context, and assistant's tool selection, parameter JSON and final answer are used as the target.

The data should be covered:

- Choose the correct tools;
- Parameter types, required items and Schema;
- Answer directly when you don't need tools;
- Recovery after tool error, empty results and timeout;
- Multi-tool sequence and stop conditions;
- Prevent malicious text in Tool Result from being treated as a high-priority instruction.

It is not enough to evaluate whether JSON can be parsed. It is necessary to execute the tool and judge whether the final task is completed.

## 11. Safety and Behavior SFT

Security data should not only have simple rejection, but also distinguish:

- Clarify harmful requests: rejection or secure redirection;
- Legal but sensitive educational, medical and news contexts: give limited help;
- Insufficient information: clarifying or expressing uncertainty;
- Normal request: avoid wrong refusal to answer;
- prompt Injection and Tool Boundary: Maintain the priority of instructions.

If the proportion of rejection samples is too high or the template is too fixed, the model will learn to "reject sensitive words". Therefore, Harmful Compliance and False Refusal Rate should be evaluated jointly.

SFT can establish basic behavior boundaries, but production security also needs preference optimization, red team, deployment Guardrail, access control and continuous monitoring.

## 12. Full-parameter fine-tuning, LoRA and QLoRA

### 12.1 Full Fine-Tuning

Update all model parameters. The advantage is that the capacity is sufficient, suitable for general post-training and large distribution changes; the cost is the need to save the complete gradient, optimizer state and trainable weight.

### 12.2 LoRA

Freeze the original weight \(W\), and only learn the low-rank increment:

\[
W'=W+\Delta W,\qquad \Delta W=BA
\]

Among them, the rank is \(r\ll\min(d_{in},d_{out})\). LoRA significantly reduces the GPU memory of trainable parameters and optimizers, which is suitable for domain adaptation, rapid experiments and multi-tenant Adapters.

### 12.3 QLoRA

Freeze the base model in the form of 4-bit quantization, and the gradient spreads to LoRA Adapter through quantization weight. QLoRA further reduces the display memory, but training throughput, quantification of Kernel and consolidated deployment need to be evaluated separately.

### 12.4 How to choose

| Target | Common Choices | Key Verification |
|---|---|---|
| Build a general Instruct Model | Full FT | Ability maintenance, total training cost |
| Small data field adaptation | LoRA | Rank, target layer, overfitting |
| single GPU adapts to large models | QLoRA | Quantitative error, Kernel and export |
| Multiple Independent Customers/Styles | LoRA Adapter | Adapter Routing and Version Management |

PEFT saves training resources and does not automatically guarantee quality; data, Rank, target module and base model still determine the result.

## 13. Optimizer and hyperparameter

SFT often follows AdamW, Warmup, Cosine/Linear Decay, mixed accuracy and gradient cropping, but relative pre-training usually uses a smaller learning rate and a shorter training cycle.

The parameters that require joint scanning include:

- Peak Learning Rate;
- Global assistant token Number/Update;
- Epoch or the maximum number of update steps;
- Warmup Ratio;
- Weight Decay;
- Maximum sequence length;
- Data mixed weight;
- Full FT or LoRA Rank/Alpha/Target Modules.

"1–3 Epoch" cannot be regarded as a fixed rule: duplicate data, target token number, data quality and model size are different, and the optimal number of steps is also different. Independent Eval should be used for Early Stopping.

Methods such as NEFTune add noise to training embedding, and have improved generalization in some instruction fine-tuning settings, but it is an optional regularization method and should be integrated with the strong data baseline, not the default required components.

## 14. Catastrophic forgetfulness and ability maintenance

SFT may improve instruction compliance, but damage knowledge, code, multilingual or Base Completion ability. Common reasons include excessive learning rate, too long training, too narrow data domain and single answer style.

Relief method:

1. Reduce the learning rate and use Early Stopping;
2. Mix in general SFT or a small amount of pre-training Replay data;
3. Use multi-domain balance instead of training only the target domain;
4. Add KL/Anchor constraints to the reference model;
5. Layered freezing or using LoRA;
6. Continuously run Base, Instruct and Safety regression sets;
7. Keep multiple intermediate Checkpoints and select the model according to the comprehensive Eval.

Ability maintenance is not to keep all the old indicators completely unchanged, but to clarify which promotions are allowed to be exchanged and which retreats are allowed.

## 15. SFT special problems of MoE model

The amount of SFT data is usually much smaller than the pre-training data, and MoE may occur:

- router distribution drifts quickly with the dialogue data;
- A few Experts are occupied by specific formats or languages;
- There are too few expert tokens under the small batch;
- The proportion of auxiliary equilibrium loss and the main SFT target is unbalanced;
- If you don't select expert, you won't get enough updates.

It is necessary to monitor each layer of expert utilization rate, router entropy, load dispersion and domain routing. It is also necessary to clarify which token the auxiliary loss affects: even if LM Loss only supervises the assistant, the router equilibrium may also count the entire context, depending on the implementation.

Optional strategies include reducing the router learning rate, blocking some router/expert, mixing general data, and using a larger valid token batch. Any strategy should be compared with the baseline of Dense or freeze router.

## 16. Training efficiency of SFT

### 16.1 Dynamic batch and Length Divided into Barrels

Organize batch with the maximum number of tokens instead of the fixed number of samples, which can stabilize the GPU memory under variable-length data. Divide the bucket to reduce Padding, and Packing further improve the utilization rate.

### 16.2 Mixing Accuracy and Activation Checkpointing

BF16 is a common stable choice; FP16 may require Loss Scaling. Activation Checkpointing uses recalculation to exchange GPU memory, and LoRA cannot eliminate the long sequence activation occupation.

### 16.3 Distributed

- DDP expands data throughput, but each card saves the complete model;
- FSDP/ZeRO cut model, gradient and optimizer state;
- Tensor/Pipeline Parallel is used for larger models;
- MoE may also need expert Parallel.

The distribution of SFT data and sequence length is more irregular, and the imbalance of DataLoader, Padding and token of different Ranks may become a real bottleneck.

## 17. How should SFT be evaluated?

assistant-only Validation Loss can only judge the degree of fit to the target text, and cannot represent the quality of the answer. The complete Eval includes at least:

### 17.1 Instructions and Formats

- Instruction compliance, negative constraints and multi-conditional tasks;
- JSON Schema, tool call and structured output;
- Multi-round status and character boundaries;
- Stop position and repeat generation.

### 17.2 Ability to maintain

- Knowledge, mathematics, code, multilingual;
- Comparison before and after base model and SFT;
- Seen, Development and Unseen Eval are separated;
- Divided by difficulty and long context results.

### 17.3 Open answer

Combine Pairwise Judge, manual review and clear Rubric to control length deviation, position deviation and Judge style preferences.

### 17.4 Security

At the same time, report harmful compliance, jailbreak success rate, wrong refusal and high-risk ability. Only looking at the rejection rate will reward the model of "nothing to answer".

### 17.5 System cost

Report input/output token, TTFT, TPOT, throughput and task completion costs. Reasoning SFT may improve the quality or significantly increase the average output length.

## 18. The industry discloses the SFT route

### 18.1 InstructGPT: Manual demonstration as the starting point for alignment

Use the ideal answer written by the annotator to train the SFT model, and then collect the model answer preference to train the reward model, and optimize it through PPO. Its key contribution is to divide "model learning" and "preference" into two stages.

### 18.2 FLAN: Expand Tasks, Templates and Reasoning Demonstration

FLAN unified a large number of NLP tasks into instruction formats, and studied the extension of the number of tasks, model scale and Chain-of-Thought data. It emphasizes the cross-task generalization of SFT, rather than just fitting the chat style.

### 18.3 LIMA: Emphasis on selected data

LIMA showed that a small number of high-quality and diversified demonstrations can strongly change the interaction behavior of large models. The conclusion should be understood as "quality and coverage first", not "any model only needs a fixed number of samples".

### 18.4 Llama 3: Synthetic data, quality control and follow-up preference optimization

The public report of Llama 3 puts SFT in a complete post-training system, in conjunction with methods such as synthetic data, rejection sampling, reward model and DPO. The focus is not on a certain algorithm, but on the iterative closed loop of data generation, quality screening and multiple rounds of Eval.

### 18.5 Tülu 3: Open SFT-DPO-RLVR process

Tülu 3 discloses data mixing, training code, development/unseen Eval and decontamination methods, and takes SFT as the basic stage of DPO and verifiable reward RL, which is suitable for understanding modern post-open training formulas.

### 18.6 Qwen3: Long CoT Cold Start and Mixed Thinking

The public process first uses long CoT data to establish a reasoning mode, and then enters the reasoning RL, Thinking/Non-thinking fusion and general RL. It reflects the current trend: SFT is responsible for cold start and mode fusion, and the complex reasoning ability continues to expand from verifiable post-training.

## 19. Common failure patterns and positioning

| Symptoms | Common Causes | Priority Troubleshoot |
|---|---|---|
| Loss is very low but can't talk | Template or Generation prompt is inconsistent | token-by-one check training/reasoning sequence |
| Model retelling user problems | User token also participates in Loss or data itself retelling | Loss Mask, answer quality |
| Will not stop | EOS/Message End Mark Unsupervised | Template, Special token, Truncation |
| The answer is always long | Long answer token weight is too large, Teacher style is single | Length distribution, sample weighting |
| JSON is often damaged | Structure token, Schema and escapancy | Executable verification, template consistency |
| General ability decline | LR too large, too long training, too narrow domain | SFT back and forth regression, Replay |
| Excessive rejection | Safety sample ratio or insufficient comparison | False Refusal, context stratification |
| Multi-round string role | Role boundary or historical data error | Original dialogue and rendering results |
| Reasoning seems to be smooth but the answer is wrong | Unverified synthetic trajectory | Verifier, answer and process evaluation |
| MoE expert Collapse | Small batch, Routing Drift, Insufficient Balance | Hierarchical Routing Statistics and Data Domain |

## 20. A set of reliable SFT experimental sequences

1. Freeze base model, tokenizer and chat template versions;
2. Use very small data to fit, verify Shift, Mask, EOS and generate links;
3. Establish a high-quality small baseline that has been manually audited;
4. Freeze independent and de-polluted multi-domain Eval;
5. Scan learning rate, training steps and global assistant token;
6. Then compare the data scale, data mixing and Full FT/LoRA;
7. When adding Reasoning, Tool and Safety data, do incremental ablation respectively;
8. At the same time, report the command capability, basic capability, safety and cost;
9. Save data list, template, hyperparameter and intermediate Checkpoint;
10. After SFT is stable, enter preference optimization or reasoning RL.

Each round of experiments should try to answer only one main question. When data, templates, Loss Mask and optimizers change at the same time, it is difficult to judge where the benefits come from.

## 21. SFT mapping of miniLLM

miniLLM loads pre-training weights from `out/pretrain` to perform full-parameter SFT; complete System/User/Tool/assistant dialogue enters the model, but only assistant's Reasoning, body, tool call and `<|im_end|>` participate in the language model Loss. The data pipeline uses Fast tokenizer's Offset Mapping to construct precise assistant Span and adopts dynamic Padding.

The long dialogue will keep the System message, give priority to deleting the earliest complete round, and try to keep the beginning of the recent question and answer; if the answer is truncated, the early end mark will not be forged. The training side adopts AdamW, Warmup + Cosine, hybrid accuracy, gradient accumulation, default Activation Checkpointing and optional DDP, while recording assistant token throughput, Padding efficiency, truncation rate and MoE router indicators.

At present, the most important thing to strengthen is: change the validation set to an independent data source; increase instruction compliance, JSON/Tool execution, security and basic ability maintenance Eval; compare assistant token weighting and sample-by-sample weighting; and finally evaluate the data ratio of Packing, LoRA and Reasoning. This can upgrade SFT from "Loss can decline" to "behavior does improve and there is no unacceptable ability to fall back".

## Reference materials

- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [NEFTune: Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124)
- [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)

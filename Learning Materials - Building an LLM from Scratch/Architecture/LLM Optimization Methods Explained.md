# Optimizing LLMs Across the Lifecycle: Data, Training, Inference, and Product Feedback

> Large model optimization is not a competition of a single Benchmark, parameter quantity or token/s, but a Pareto frontier between quality, cost, delay, security and maintainability. This article mainly focuses on the public method of the industry, and finally gives a brief mapping of miniLLM.

## 1. Optimization goal: define the system first, and then talk about local indicators

A practical goal can be written as constraint optimization:

\[
\max \; Q(M,D,A,S)
\]

The constraints are:

\[
C_{train}\le B_{train},\quad C_{serve}\le B_{serve},\quad L_{p95}\le L_{target},\quad R_{safety}\le R_{max}
\]

Among them, \(M\) is the model architecture, \(D\) is the data, \(A\) is the training algorithm, and \(S\) is the service system. Local optimization may harm the whole: for example, a larger vocabulary shortens the sequence but expands the output head; MoE reduces activation parameters, but increases communication and weight memory.

## 2. The four scales must be separated.

### 2.1 Total parameters and activation parameters

The two are close in Dense; the total parameters in Sparse MoE determine the capacity and storage, and the activation parameters are closer to the calculation scale of each token.

### 2.2 Data Bytes and Training token

The file size is affected by coding, compression and formatting. The training budget should report the number of de-duplication documents, the proportion of domains, the effective token and the repetition rate.

### 2.3 Theoretical FLOPs and Effective Throughput

Dense Decoder often uses \(C\approx6ND\) for quantitative estimates, but the real cost also includes Padding, recomputing, communication, routing, failure and low utilization.

### 2.4 Nominal context and effective context

The model "can receive" 128K does not mean that it can use 128K stably. The effective context should be evaluated through retrieval, combined reasoning, long output and real workload.

## 3. Data optimization: usually the highest leverage

### 3.1 deduplication and contamination control

- Documents, paragraphs, n-gram multi-layer deduplication;
- Cross-data source near-repetition detection;
- Check overlap with the ability evaluation set, test warehouse and question bank;
- Save data sources, permissions, processing logs and hashs.

deduplication also affects quality, memory, privacy and effective token budget.

### 3.2 Quality filtering

Rules are suitable for dealing with garbled codes, templates, advertisements and format anomalies; classifiers are suitable for scale semantic screening; LLM Judge can make complex quality judgments, but the cost is higher and biased. The mature process will combine the three and carry out manual sampling.

### 3.3 Data mixing

The proportion of Web, books, papers, code, mathematics, multilingual and synthetic data determines the shape of the ability. Temperature sampling can increase the proportion of small languages or scarce fields, and dynamic formulas can enhance high-quality, long-term context or reasoning data after training.

### 3.4 Synthetic data and distillation

The strong model can generate interpretation, problem, tool trajectory and rejection samples; Verifier, actuator and de-weighter are responsible for filtering. The high-value mode is "generate-verify-retain failure information", rather than unconditionally copying Teacher output.

## 4. tokenizer and sequence organization optimization

### 4.1 vocabulary size

The large vocabulary reduces the average sequence length, but increases the cost of embedding/Softmax. `bytes/token`, model throughput and downstream quality should be compared according to language, code, mathematics and long-tail input.

### 4.2 Packing

Fill in multiple short documents into a fixed-length sequence to improve:

\[
U=\frac{N_{valid}}{N_{allocated}}
\]

However, EOS, document boundaries, Position ID and Loss Mask must be handled correctly.

### 4.3 Length split bucket and dynamic batch

Reduce Padding by length group batch; control batch with the maximum number of tokens instead of the number of samples, which can make the GPU memory more stable.

## 5. Architecture optimization: make the calculation more valuable

### 5.1 Dense and MoE

Dense is simple and the delay is stable; MoE exchanges routing and system complexity for total parameter capacity. Whether To Adopt MoE Should Be Decided By Data Scale, Interconnected Bandwidth, expert batch And Service Concurrency.

### 5.2 GQA, MQA and MLA

The three reduce KV Cache from different angles. The optimization focus is usually on long context Decode, not just reducing attention projection parameters.

### 5.3 Local/Gocal attention and Mixed SSM

The sliding window controls the secondary cost of the long sequence, and the periodic global layer transmits remote information; the attention-SSM hybrid undertakes part of the sequence modeling with the recursive layer. The real benefit depends on the long-distance mission and the hardware kernel.

### 5.4 Long context

The complete scheme includes location extension, long text data, length course, attention structure, Context Parallel, KV management and long context Eval. Single modification of RoPE parameters is not a complete optimization.

### 5.5 Multimodal

Modular encoder + Projector is suitable for low-cost expansion; early fusion joint pre-training pursues deeper cross-modal capabilities. Image token number, visual resolution and modal sampling will directly affect training and service costs.

## 6. Training algorithm optimization

### 6.1 Scaling Experiment

Before formal training, use multiple sets of small models and token budgets to fit the Loss/ability trend, and find a reasonable ratio of model scale, data scale and training time. The conclusion of the small model is to be alert to architectural inflection points and data quality changes.

### 6.2 Optimizer and learning rate

AdamW + Warmup + Cosine/WSD is a common baseline. The key is not the name, but whether the peak learning rate, global batch token, initialization, gradient cropping and attenuation end segment are coordinated.

### 6.3 Mixing accuracy

BF16 is a stable common scheme; FP8 requires block scaling, sensitive oper to retain high accuracy and end-to-end convergence verification. Low-precision returns must be based on the premise that there is no significant regression in quality.

### 6.4 Auxiliary target and distillation

MTP increases the supervision density, and distillation migrates the distribution or reasoning trajectory of the large model to the small model. Both should be compared with the pure CLM baseline and equal FLOPs.

### 6.5 Stability Engineering

Automatically detect NaN, Loss Spike, bad data batch, hardware error and communication timeout; Checkpoint must contain optimizer, Scheduler, data cursor and RNG status. Restorability itself is training efficiency.

## 7. Distributed and kernel optimization

| Dimension | Problem solving | Main cost |
|---|---|---|
| Data Parallel | Expand throughput | Each card still saves the model, gradient synchronization |
| FSDP/ZeRO | Cut Model Status | Gather/Reduce Communication |
| Tensor Parallel | Intra-layer matrix | High-frequency set communication |
| Pipeline Parallel | Cut by layer | Pipeline Bubble, scheduling complex |
| Context Parallel | Cut long sequence | attention Communication |
| expert Parallel | Cutting MoE expert | All-to-All and Load Balancing |

FlashAttention, Fused RMSNorm, Fused MLP, communication computing overlap and topological perception placement are responsible for converting theoretical FLOPs into real throughput.

MFU is more explanatory than "GPU Utilization": the latter may also count communication or invalid Kernel as busy, and MFU pays more attention to the proportion of effective model calculation to hardware peaks.

## 8. Post-training optimization: from continuation to behavior

### 8.1 SFT

Use high-quality instructions - answer data learning format, tasks and style. Data deduplication, difficulty, domain and error analysis are usually more important than unconstrained expansion.

### 8.2 Preference Optimization

- RLHF/RLAIF: Train the reward model, and then use RL to optimize the strategy;
- DPO/IPO/ORPO, etc.: directly use the preference pair, and the process is simplified;
- Constitutional AI: Generate feedback and revision with explicit principles.

The preference algorithm improves the behavior defined by the data, and cannot fix the gap in pre-training knowledge out of thin air.

### 8.3 Reasoning RL and Verifiable Rewards

Mathematics, code and some Agent tasks can be rewarded with answers, tests or environmental status. The mainstream trend is to expand the RL calculation on verifiable tasks, and then integrate it with general instruction data to control the reasoning budget and generalized degradation.

### 8.4 Agent and tool training

It is necessary to train tool selection, parameter generation, state reading, error recovery and stop conditions at the same time. The evaluation unit should be upgraded from a single answer to a complete trajectory.

## 9. Reasoning and deployment optimization

### 9.1 KV Cache and Paging Management

Cache history K/V avoids duplicate calculations; Paged attention, Continuous Batching and Prefix Caching improve the utilization of multi-request video memory.

### 9.2 Quantification

- Weight-only quantitatively reduce the weight bandwidth;
- W8A8/FP8 compresses weight and activates at the same time;
- KV Cache quantification directly affects long context concurrency.

It must be evaluated separately by task, long-tail input and hardware Kernel, and the weight file size cannot be reported only.

### 9.3 Speculative decoding

Draft Model proposes multiple candidates at once, and Target Model verifies them in parallel. The return depends on the acceptance rate, verification Kernel, batch and model size ratio.

### 9.4 Calculate expansion at inference time

Improve the quality of problems through longer thinking, parallel sampling, searching or Verifier. It transfers part of the capacity expansion from training to reasoning, but it will increase the delay and token cost, and the budget should be dynamically allocated according to the task.

### 9.5 Routing and model cascade

Simple requests are given to small models, and difficult or high-risk requests are upgraded to large models/reasoning models. The misjudgment cost, cache hit and security policy of the router need to enter the end-to-end Eval.

## 10. Evaluation and product feedback closed loop

Optimized closed-loop should cover:

1. Training indicators: Loss, PPL, domain data and stability;
2. Ability indicators: knowledge, reasoning, code, multilingual, long context;
3. Behavior and security: instruction compliance, refusal to answer, jailbreak, bias and high-risk ability;
4. System indicators: TTFT, TPOT, throughput, GPU memory, cost and failure rate;
5. Product indicators: task completion rate, user correction rate, retention and manual upgrade rate.

Offline Benchmark, automated judge, manual review, red team and online A/B each only cover a part. The release threshold must freeze the evaluation version, prompt, sampling budget and statistical methods.

## 11. The public optimization route of mainstream companies

Public information can only reflect the part that the company is willing to disclose. The following is used for refining methods, not for guessing undisclosed details.

### 11.1 OpenAI: Pre-training Scaling + Reasoning Scaling + System Eval

The public route is calculated when extending from large-scale pre-training to o-series reasoning, and connecting capabilities, security and deployment decisions with System Card, Preparedness Framework and product feedback.

### 11.2 Google DeepMind: Optimal Computing, Multimodal and Efficient attention

Chinchilla emphasizes the optimization of model and data calculation; Gemini/Gemma route covers native multimodal, distillation, long context, local/gobal attention and end-side dimensions; TPU/JAX system reflects software and hardware coordination.

### 11.3 Meta: Strong Dense baseline to MoE and early integration evolution

Llama 3 shows Dense, GQA, large-scale data and complete post-training; Llama 4 openly turns to MoE, sharing/routing experts and native multi-modal early integration, and emphasizes FP8 and long-context Mid-training.

### 11.4 Anthropic: Extensible supervision, security and interpretability

Constitutional AI, RLAIF, model behavior evaluation, red team and Mechanistic Interpretability constitute its public features. The core is to connect target specifications, training feedback and deployment security.

### 11.5 DeepSeek: Architecture-Algorithm-System Joint Design

DeepSeek-V3's public scheme combination MLA, fine-grained MoE, Shared expert, unassisted loss equilibrium, MTP, FP8 and communication optimization shows that the cost advantage comes from multi-layer collaboration, not a single "secret".

### 11.6 Qwen: Dense/MoE, multilingual and mixed thinking

The Qwen series openly covers multi-size Dense/MoE, code, multi-language, Agent and vision; Qwen3 further shows the post-training route of Thinking/Non-thinking integration and controllable thinking budget.

### 11.7 Mistral: Simple structure and deployment efficiency

Mistral 7B's GQA + Sliding-window attention and Mixtral's Sparse MoE represent the pursuit of parameter efficiency and open source deployment availability with a small number of clear structural changes.

## 12. The common trend of these routes

1. The importance of data quality, deduplication and formula is higher than that of non-differential expansion;
2. Dense and MoE coexist, and the architecture selection is bound to the system scale;
3. KV Cache, long context and reasoning bandwidth enter the model design center;
4. Multimodal evolution from plug-in encoder to earlier integration;
5. Post-training extends from general preferences to verifiable reasoning and Agent trajectory;
6. at inference time, the calculation becomes a new telescopic axis;
7. Eval changed from a static score to a continuous release threshold;
8. The combined optimization of quality-cost-safety replaces a single total score.

## 13. How to determine the optimization priority

| Symptoms | Prioritize troubleshooting | What should not be done first |
|---|---|---|
| Loss does not decrease or fluctuates violently | Data, Shift/Mask, LR, numerical accuracy | Directly expand the model |
| Low GPU utilization | Padding, DataLoader, Kernel, Communication | Only increase the nominal value of batch |
| Good verification, poor ability | Data coverage, contamination, tokenizer, post-training | Only follow PPL |
| Long-context memory explosion | GQA/MLA, local attention, KV management | Only RoPE |
| Decode slow | KV Cache, quantification, batch, speculative decoding | Only look at training throughput |
| MoE No Profit | Data Volume, Load, expert batch, Communication | Continue to Increase expert |
| Security capabilities retreat from each other | Layered Eval, data formula, reward conflict | Only optimize a single Judge score |

The most reliable way is to establish a stable baseline, introduce only one major variable per round, and report the confidence interval and quality-cost curve.

## 14. Optimized mapping of miniLLM

miniLLM has covered Decoder-only, GQA, RoPE, RMSNorm, SwiGLU, switchable Dense/MoE, AdamW, mixed accuracy, DDP, checkpoint recovery and basic PPL/generation evaluation, which is a complete teaching closed loop.

From the perspective of the whole life cycle, it is most worth investing in at present: first establish an independent subdomain Eval and Dense baseline, and then use Packing to improve the effective token utilization rate; then supplement KV Cache and real reasoning performance test; and finally do MoE, long context and larger-scale parallel ablation. This order allows each optimization to answer "what exactly improves quality, cost or stability".

## Reference materials

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)
- [The Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Gemma explained: What’s new in Gemma 3](https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)

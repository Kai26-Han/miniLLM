# Fine-Tuning LLMs: A Guide to Mainstream Approaches

> The value of Fine-tuning is not "backing documents into the model", but changing the stable behavior of the model with domain data: making it more reliable to follow the task protocol, output fixed formats, use professional expressions, call tools, or migrate the capabilities of large models to smaller and cheaper models.

## Key points

1. **First, judge whether the problem is knowledge, behavior or ability.** Dynamic knowledge gives priority to RAG, process actions give priority to tools, and stable behavior and task ability are the advantages of Fine-tuning.
2. **The most common in business is SFT + LoRA/QLoRA.** SFT defines learning goals, and LoRA/QLoRA decides how many parameters to update at a lower cost; the two are not mutually exclusive.
3. **full-parameter fine-tuning is not the default answer.** It has a large capacity and direct deployment, but the cost of training, storage, rollback and multi-tenant is higher.
4. **Continuous pre-training and SFT solve different problems.** The former learns domain language and distribution, while the latter learns "how to output when receiving input".
5. **Data quality and evaluation closed loop are usually more important than training skills.** Error demonstration, label contradiction and evaluation leakage will be steadily learned by the model.
6. **Fine-tuning cannot replace external fact sources.** High-frequency changes in prices, inventory, policies and account status should be read at inference time, rather than solidified in the weight.
7. **Establish an untrained baseline first, and then start the fine-tuning.** Without the prompt/RAG baseline, it is impossible to judge whether the training benefits are worth sustaining the cost of the model life cycle.
8. **What is online is a version system.** Base model, tokenizer, chat template, Adapter, data, code, evaluation set and inference settings must be managed together.

---

## 1. What exactly has Fine-tuning changed?

The pre-training model has learned common language, knowledge and patterns. Fine-tuning continues to train from this checkpoint, updating all or part of the parameters through smaller and more purposeful data sets, so that the condition distribution is from:

\[
p_{\theta_0}(y\mid x)
\]

Become more in line with business goals:

\[
P_{\theta}(y\mid x,\text{business specification})
\]

It is best at changing four types of stability attributes:

- **Task protocol**: classification, extraction, rewriting, summary, question and answer, tool call;
- **Output structure**: JSON Schema, fixed field, SQL, DSL, code patch;
- **Expression habits**: terminology, tone, length, rejection boundary and brand style;
- **Efficiency structure**: Let the small model imitate the large model to handle narrow tasks with lower latency and cost.

Fine-tuning is not database writing. The knowledge in the weight cannot be accurately addressed, difficult to delete, without source citation, and cannot be guaranteed to be updated on time. Therefore, "let the model remember the latest product price" is usually the wrong goal, and "let the model learn to generate compliance responses based on the retrieved product data" is the appropriate goal.

## 2. Let's do the problem attribution first: prompt, RAG, tools or Fine-tuning

Business needs are often generally described as "the model does not understand us enough". In fact, it should be disassembled into the following problems:

| Demand | Preferred Plan | Reason |
|---|---|---|
| Fixed role, tone, a small number of rules | System prompt / Few-shot | Fast modification, no training and model maintenance costs |
| Query internal documents, the latest policies, commodity information | RAG | Content can be updated, referenced, and permission control can be done |
| Query inventory, place orders, calculate, call the internal system | Tool Calling / Workflow | The result comes from determining the system, not the model memory |
| Stable output JSON, classification label or domain format | SFT | A large number of examples can solidify behavior into model habits |
| Learn industry language, code base distribution or rare terms | Continuous pre-training + SFT | Make up for the field representation first, and then learn specific tasks |
| Learning preferences in multiple feasible answers | DPO / RLHF | The goal is relatively good or bad, not the only standard answer |
| Replace high-quality large models with low-cost small models | Distillation + SFT | Use Teacher to generate supervisory data migration capabilities |
| Mathematics, code and other tasks that can automatically judge success or failure | Strong base + SFT + enhanced fine-tuning | Actuators or rules can be used to provide intensive feedback |

These technologies can be combined:

```text
User request
↓ prompt Define roles and boundaries
The retrieval system provides current facts.
  ↓
Fine-tuning model reasoning and organizing answers according to the business format
↓ Tool calls the real business system
Rules, permissions and security layer verification output
```

A practical principle is:

> Information that needs to be updated frequently, accurately referenced or isolated by permissions is placed outside the model; the behavior that requires high-frequency reproduction and stable generalization is written into the model parameters.

## 3. What mainstream training stages does "fine-tuning" include?

### 3.1 Continuous pre-training: adapt the model to the domain distribution

Continued Pretraining is also often called Domain-Adaptive Pretraining (DAPT) or Mid-training. It uses a large number of unlabeled domain corpus to continue to optimize Next-token Prediction:

\[
L_{CPT}=-\sum_t\log p_\theta(x_t\mid x_{<t})
\]

Suitable for:

- The distribution of legal, medical, financial, scientific research and other professional texts is obviously different from that of the general Internet;
- There are a large number of unlabeled corpus, but there are few high-quality Q&A pairs;
- It is necessary to supplement industry terms, fixed style, code base or minor language representations;
- New context length, tokenizer extension or special data format need to be trained.

The limitation is that it will not automatically teach the model dialogue, rejection or task format. Usually, SFT is also required; if the domain data is single, it is also mixed with general data to reduce catastrophic forgetting.

### 3.2 SFT: Let the model imitate the target answer

Supervised Fine-Tuning uses input-target output pairs. Set the context to \(x\) and answer to \(y=(y_1,\ldots,y_T)\):

\[
L_{SFT}=-\sum_{t=1}^{T}\log p_\theta(y_t\mid x,y_{<t})
\]

Dialogue training usually only allows the assistant area to participate in Loss, and System, User and Tool Result only as conditions. This is the most commonly used training target for enterprise customization.

SFT is suitable for the scenario of "can demonstrate the correct answer": if the expert can write the ideal output, the model can directly imitate it; if the expert can only say that A is better than B, but it is difficult to write the only standard answer, preference optimization is more appropriate.

### 3.3 Preference and Strengthening Stage: Make the model optimization "better"

After SFT, you can also use the chosen/rejected preference pair, reward model or verifiable Grader to continue training. Strictly speaking, these belong to Preference Optimization or Reinforcement Fine-tuning. For details, please refer to RLHF: Analysis of Business Mainstream Schemes in the same catalog.

### 3.4 Distillation: Migrate the ability to a smaller model

The common business processes of distillation are:

1. Collect real requests or construct prompt to cover the business distribution;
2. Let the more capable Teacher generate answers, reasoning or tool trajectories;
3. Filter with rules, actuators, LLM Judge and manual sampling;
4. Conduct SFT for smaller students, and do preference optimization if necessary;
5. Compare quality, delay, throughput and single request cost.

Distillation is not simply copying all Teacher outputs. Unverified data will migrate hallucinations, lengthy and single expressions to Student.

## 4. How to update parameters: full parameters, LoRA and QLoRA

The training target and the parameter update method are two dimensions. SFT, DPO and other goals can be matched with full-parameter training or PEFT.

### 4.1 Full parameters Fine-tuning

Update all parameters of the model:

\[
\theta\leftarrow\theta-\eta\nabla_\theta L
\]

Advantages:

- The maximum capacity can be adjusted, which is suitable for tasks with sufficient data and large distribution changes;
- The final weight is complete, and there is no need to mount Adapter when deploying;
- It is usually easier to reach the upper limit of ability under large-scale training.

Price:

- Gradient and optimizer status significantly increase the GPU memory;
- Each business version should save the complete model;
- Small data is more likely to overfit or destroy the general ability;
- Multi-tenant, frequent iteration and fast rollback are expensive.

It is more suitable for model providers, single high-value models, large-scale high-quality data, or scenarios where the LoRA capacity has been experimentally proven insufficient.

### 4.2 LoRA: Efficient fine-tuning of mainstream parameters

LoRA freezes the original weight \(W_0\), and only learns a low-rank increment:

\[
W=W_0+\Delta W,\qquad \Delta W=BA
\]

Among them, \(A\in\mathbb{R}^{r\times d}\), \(B\in\mathbb{R}^{k\times r}\), and \(r\ll\min(d,k)\). The training parameters were reduced from \(kd\) to about \(r(d+k)\).

The business advantages of LoRA include:

- Adapter files are small, which is convenient for version management and customer isolation;
- You can share a dock and switch different Adapters between requests;
- Less optimizer status and gradient required for training;
- The Adapter can be merged back to the base before deployment to avoid additional reasoning operators.

Common hyperparameters:

| Parameters | Meaning | Main Impact |
|---|---|---|
| `r` | Low-rank dimension | The higher the large capacity, the more parameters and GPU memory |
| `lora_alpha` | Incremental Scaling | Control Adapter Update Intensity |
| `target_modules` | Injection layer | Determine the adjustable model subspace |
| `lora_dropout` | Adapter Dropout | Small data helps to regularize |

In engineering, the baseline should be established with ordinary LoRA first, and then it is necessary to judge whether it is necessary to improve Rank, cover all Linear layers or use LoRA variants through ablation. More variants do not automatically mean that the business effect is better.

### 4.3 QLoRA: LoRA training on the quantitative base

QLoRA loads the frozen base in a low-precision form such as 4-bit, and the gradient passes through the quantitative weight, and only the LoRA parameters are updated. It greatly reduces the weight occupation of the base, allowing larger models to be trained on fewer GPUs.

It should be noted that:

- "4-bit loading" mainly saves dock weighted GPU memory, activation, LoRA gradient and optimizer still exist;
- Quantitative error may affect logarithm probability, fine-grained generation and training stability;
- After the training is completed, it should be clearly deployed whether the quantified base + Adapter, or the weights to be merged and re-quantified;
- QLoRA is a strong baseline when resources are limited, and not all quality targets are equivalent to full-parameter training.

### 4.4 Other PEFT

Adapter, Prefix Tuning, prompt Tuning, IA3 and other methods can also reduce training parameters, but in the general LLM business, LoRA ecology, tool chain and deployment support are usually more mature. Unless there are existing platform constraints or experiments that have clearly benefited, priority is given to starting with LoRA.

### 4.5 Business Selection Comparison

| Scheme | Training Display Storage | Single Task Product | Quality Capacity | Multi-tenant | Typical Choice |
|---|---:|---:|---:|---:|---|
| full-parameter SFT | High | Complete model | High | Poor | Big data, single core model |
| LoRA SFT | Medium and Low | Small Adapter | Medium and High | Good | Enterprise Customization Default Starting Point |
| QLoRA SFT | Low | Small Adapter | Medium High | Good | GPU Restricted, Quick Verification |
| prompt/Prefix Tuning | Low | Very Small Parameters | Relatively Limited | Good | Specific Platform or Simple Task |
| Continuous pre-training + SFT | Very high | Complete model or Adapter | High | Average | Strong field migration |
| Distillation + SFT | Teacher Reasoning + Training | Small Model | Depends on Teacher and Filtering | Good | Reduce Online Costs and Delays |

## 5. Six types of common business plans

### 5.1 Structured extraction, classification and routing

Example: work order classification, contract field extraction, risk label, intention routing.

Recommendation:

```text
Strong prompt baseline
  -> SFT + LoRA
-> JSON Schema / Enumeration Constraint
-> Rule check and fail to retry
```

The key indicators are field-level Precision/Recall/F1, Scheme legality rate, rejection rate and long-tail category recall, not just looking at the generation of Loss.

### 5.2 Brand customer service and corporate writing

Example: fixed tone, reply length, brand words, policy refusal.

It is recommended to use high-quality manual demonstration to do SFT, and then use preferences to correct the answer "all correct but different styles". Dynamic policies and account information are still provided by RAG/Tool.

### 5.3 Domain Knowledge assistant

Examples: legal retrieval, medical literature, equipment operation and maintenance, enterprise knowledge base.

Recommended combination:

```text
RAG is responsible for facts, sources and updates
SFT is responsible for terminology, citation format and answer organization
If necessary, continue to pre-train the language distribution in the responsible field.
```

It is usually difficult to track the source by injecting documents by SFT alone, and it is also inconvenient to perform deletion, permission and time limit control.

### 5.4 Tool Calling and Agent

The training sample should include: when to call the tool, which tool to choose, how to generate parameters, how to read Tool Result, when to stop, and the recovery path after the tool fails.

Business evaluation should be run in a sandbox environment, and at least record:

- Accuracy of tool selection;
- Parameter Schema legality rate;
- The success rate of the complete task;
- Average number of calls and end-to-end delay;
- Over-the-power call, repeated call and failure recovery rate.

Only evaluating the text similarity of the tool JSON will omit the two situations of "different format but correct execution" and "correct format but business error".

### 5.5 Code, SQL and Enterprise DSL

If there is a code warehouse, query log and test environment, you can use continuous pre-training learning distribution, use SFT learning task format, compiler, unit test, SQL sandbox or static analysis as evaluation and filter.

Do not directly execute the high-permission code or SQL generated by the model. Training improves the probability of success, which will not cancel the sandbox, minimum permissions and audit requirements.

### 5.6 Large model distilled to small model

When online requests are concentrated in narrow tasks, the common route is "strong model generation + small model fine-tuning". Business income comes from:

- Lower initial token delay and higher throughput;
- Shorter prompt, reduce duplicate rules token;
- Privatized deployment and edge deployment;
- Serve more requests with fixed computing power.

It should be evaluated by total cost of ownership, including Teacher data generation, manual audit, Student training, independent deployment, continuous regression and retraining, rather than just comparing the price of single reasoning.

## 6. Data engineering determines most of the upper limits.

### 6.1 Data Source

| Source | Advantages | Main Risks |
|---|---|---|
| expert demonstration | Accurate, clear boundaries | Expensive, small scale, style may be single |
| Historical business data | Close to real distribution | Privacy, old process, label noise |
| User-customer service dialogue | Cover real questions | Low-quality answers, authorization and PII |
| Teacher Synthesis | Fast expansion, can cover long tail | Hallucination, templateization, self-replication |
| Rule or program generation | Stable and verifiable format | Distribution is too idealized |

The correct formula for synthetic data is not "generation is training", but:

```text
Seed Task -> Diversified Generation -> Hard Rule Verification -> Verifier/LLM Judge
-> deduplication and difficulty stratification -> Manual sampling -> Enter the training set
```

### 6.2 Data Schema

It is recommended that each sample keep metadata except for the message:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "task": "contract_extract",
  "source": "expert_v2",
  "language": "zh-CN",
  "quality": 0.94,
  "policy_version": "2026-08",
  "document_id": "..."
}
```

Metadata is used for cutting, resampling, problem tracking and deletion requests, and is generally not directly spelled into the model input.

### 6.3 Quality inspection

- The order of Chat Role is legal with the template;
- assistant's target is not empty, and Loss Mask is in good position;
- JSON, tool parameters, code and references can be parsed;
- Remove PII, key, access token and unauthorized content;
- Make precise/approximate de-weight on prompt, Response and complete dialogues;
- Troubleshoot and the overlap of test sets, public Benchmark and future evaluation samples;
- Statistical language, task, length, rejection, source and difficulty distribution;
- Manually calibrate the automated judge, instead of treating the Judge score as a fact.

### 6.4 Data cutting

Random row-by-line cutting often causes leakage. More reliable cutting keys include customer, document, session, time, template family and task family.

Keep at least three types of collections:

1. **Train**: used for parameter update;
2. **Validation**: Select hyperparameters and stop points;
3. **Test/Golden Set**: Only evaluate after the candidate model is frozen.

Additional maintenance of long-tail sets, security sets, confrontation sets and online playback sets to prevent the critical failure of the average score.

## 7. Key engineering problems in the training formula

### 7.1 base model Selection

First, select the minimum model with zero samples/less samples that is close to the target, the license meets the requirements, and is supported by the deployment stack. Fine-tuning is usually good at amplifying existing capabilities, and not good at creating complex capabilities with a small number of samples from extremely weak bases.

Candidate bases should be compared on the same business set: quality, language, context, tool capabilities, throughput, video memory, license and community support.

### 7.2 chat template must be consistent

training and inference must use the same:

- Role name and message boundary;
- BOS/EOS and assistant end mark;
- System prompt placement method;
- tool call / Tool Result Structure;
- Generation prompt;
- Whether the Reasoning content is visible and whether it participates in Loss.

Template misalignment will be manifested as inability to stop, character stringing, rereading, tool JSON failure or sudden decline in overall ability.

### 7.3 Learning Rate, Epoch and Effective batch

A larger Epoch does not equal a stronger model. Small and repetitive data can easily cause training Loss to continue to decline and business generalization to deteriorate. It is recommended to save multiple Checkpoints and evaluate the points by business.

The valid batch Size is:

\[
B_{global}=B_{device}\times N_{device}\times N_{accumulation}
\]

When comparing experiments, the number of target tokens, data sampling weights, learning rate plans, accuracy, random seeds and actual update steps should be recorded at the same time.

### 7.4 Padding, Packing and truncation

- Length bucket and Dynamic Padding can reduce ineffective calculations;
- Packing can put multiple short samples into the same window, but it is necessary to block cross-sample attention and correctly handle Loss Mask;
- System, recent requests and target answers should be retained for long dialogue truncation;
- Record the prompt truncation rate and Answer truncation rate to avoid quietly training the incomplete target.

### 7.5 Catastrophic forgetfulness

Typical relief methods:

- Reduce the learning rate, number of training rounds or LoRA Rank;
- Mix in general instruction data and security data;
- Use PEFT to limit the update range;
- Continuous running general ability return in training;
- Quarantine conflict tasks with routing or multiple Adapters.

If two business domains require conflicting formats or strategies, forcibly merging them into one Adapter may not be as good as explicit routing.

## 8. Evaluation: Training Loss is not business KPI

The evaluation should form a four-layer structure:

| Level | Focus | Example |
|---|---|---|
| Format layer | Can the output be consumed by the system | JSON legality rate, Scheme pass rate |
| Task layer | Is the task itself correct | F1, Exact Match, Pass@k, execution success rate |
| Behavior layer | Whether it conforms to human preferences | Blind test of win rate, integrity, tone, safety |
| System layer | Is it worth going online | P95 delay, throughput, single request cost, failure rate |

Recommended process:

```text
Fixed baseline
-> Offline automatic indicators
-> Manual blind test / LLM Judge after calibration
-> Security and confrontation test
-> Small flow Shadow / A/B
-> Monitoring, rollback and data backflow
```

LLM Judge should randomly disrupt the candidate order, control the length preference, and test the consistency with manual samples. Key business cannot only rely on a single Judge.

## 9. Deploy the mainstream scheme

### 9.1 Hosting Fine-tuning API

The advantage is that the infrastructure is simple and the training and inference interface are unified; the limitation is that the model, hyperparameter, data residence, version life and export ability are determined by the platform.

It is suitable for scenarios with limited resources for quick verification and team engineering. Confirm when selecting the model:

- Whether the data is used for platform training, how long it is saved, and where it is located;
- Whether to support SFT, Preference or Reinforcement;
- Whether the model can be exported, and how to migrate when the base is offline;
- How to charge for training, storage, exclusive throughput and reasoning respectively;
- Whether to support VPC, KMS, audit, deletion and permission isolation.

### 9.2 Open source model self-hosting

Common tool chains include Transformers, PEFT, TRL, DeepSpeed, FSDP, vLLM, etc. The advantage is that training and deployment can be controlled and weight can be owned; the cost is that it needs to handle distributed training, kernel compatibility, monitoring, disaster tolerance and security.

### 9.3 Adapter Serving

Multiple business shared bases, each business mounted independent LoRA:

```text
base model
├─ Adapter: Customer Service
├─ Adapter: Contract extraction
  ├─ Adapter: SQL
└─ Adapter: Customer A Exclusive Style
```

It is suitable for multi-tenants, but it needs to evaluate the complexity of Adapter hot loading, batch fragments, KV Cache, routing errors and version combination. High-flow fixed tasks can combine Adapter into independent weights in exchange for simpler reasoning paths.

## 10. Cost model

The total cost of Fine-tuning can be roughly split into:

\[
C_{total}=C_{data}+C_{train}+C_{eval}+C_{serve}+C_{maintain}+C_{risk}
\]

- \(C_{data}\): expert, annotation, synthetic reasoning, cleaning and privacy processing;
- \(C_{train}\): GPU, failure rerun, experiment and Checkpoint storage;
- \(C_{eval}\): manual blind test, Judge, red team and online experiment;
- \(C_{serve}\): reasoning instance, Adapter management, delay and throughput;
- \(C_{maintain}\): base upgrade, retraining, regression and data deletion;
- \(C_{risk}\): error output, compliance events and supplier lock.

The benefits of fine-tuning usually come from quality improvement, prompt shortening, using smaller models, reducing retry and manual review. The ROI should be calculated with the unit task cost of the complete link, rather than just looking at the training bill.

## 11. Recommended minimum business closed loop

```text
1. Define a single and measurable business goal
2. Fixed prompt / RAG / Tool Baseline
3. Establish an independent Golden Set and a safe regression set
4. Collect and clean high-quality demonstrations
5. Use LoRA SFT to train the first candidate model
6. Compare quality, delay and cost with the baseline
7. Analyze the type of failure, instead of blindly adding data
8. For high-value failure to supplement data or enter preference optimization
9. Small traffic goes online and retains the ability to route back to the old model.
10. Only stream the reviewed online samples back to the next version.
```

The default starting point can be summarized as:

> Choose the smallest base that is already strong, do LoRA SFT with high-quality data, and select Checkpoint with real business Golden Set; only when the evidence shows that the capacity is insufficient, upgrade to a larger base, continuous pre-training or full-parameter training.

## 12. Common misunderstandings

1. **Using fine-tuning as a replacement for RAG.** Fine-tuned knowledge is hard to cite or update, and memorization does not guarantee factual accuracy.
2. **Training before defining metrics.** Without a fixed baseline and test set, successful optimization does not prove business value.
3. **Treating old model responses as ground truth.** The student will inherit the previous system's hallucinations and process errors.
4. **Using different templates for training and inference.** This is one of the most common and least visible engineering failures.
5. **Looking only at the average score.** Averages can hide regressions for key customers, safety-critical long-tail cases, and tool permissions.
6. **Assuming more data is always better.** Duplicated, contradictory, or low-quality examples steadily degrade the model.
7. **Assuming LoRA always matches full-parameter fine-tuning.** LoRA is a strong baseline, but ablations must confirm whether it has enough capacity.
8. **Ignoring the cost of a base-model upgrade.** Changes to the tokenizer, templates, or behavior may require rebuilding both data and evaluations.
9. **Treating an automated judge as ground truth.** Judges have biases and position effects and can themselves be attacked.
10. **Shipping without a rollback path.** Model, data, and inference-configuration versions must be reversible together.

## 13. One-page selection conclusion

| If your main problem is... | Try it first |
|---|---|
| prompt is too long and the format is often unstable | SFT + LoRA |
| Need to answer the latest private information | RAG, add SFT if necessary |
| Need to call the internal system stably | Tool Calling SFT + perform verification |
| Only a large number of unlabeled domain text | Continuous pre-training, and then do SFT |
| There are few GPUs but want to train larger models | QLoRA |
| Large data scale and requires the upper limit of capacity | Full parameters Fine-tuning |
| The same dock serves multiple customers/tasks | Multiple LoRA + routing |
| The effect of the large model is good, but it is too expensive online | Teacher distilled to the small model |
| Can't write the only answer, only good or bad | DPO / RLHF |
| Output can be automatically verified by the program | SFT + enhanced fine-tuning / RLVR |

---

## Extended reading

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Hugging Face PEFT Document ](https://huggingface.co/docs/peft/index)
- [Hugging Face LoRA Guide ](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
- [Microsoft: Selection of RAG and Fine-tuning ](https://learn.microsoft.com/azure/developer/ai/augment-llm-rag-fine-tuning)
- [Google Cloud: Overview of Model Tuning Methods ](https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-models)
- [Amazon Bedrock: Model Customization Method ](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html)

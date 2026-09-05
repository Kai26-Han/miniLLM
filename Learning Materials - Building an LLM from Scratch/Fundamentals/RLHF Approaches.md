# RLHF and Preference Alignment: A Guide to Mainstream Approaches

> RLHF (Reinforcement Learning from Human Feedback) defines "what answer is better" with human preferences, and then turns the feedback into a training signal that can be optimized by the model. Today's business practices usually put DPO, RLAIF, rule rewards and verifiable rewards into the generalized RLHF/alignment system; they share a closed loop of feedback, but not all of them use reinforcement learning.

## Key points

1. **SFT learns "like standard answers", and RLHF learns "which of the multiple feasible answers is better".** The two are usually connected back and forth, rather than replacing each other.
2. **The classic RLHF is reward model + PPO.** It can explore the answer space of the current model online, but the system is complex, training is expensive, and it is easy to exploit reward loopholes.
3. **DPO is the most common low-threshold preference alignment baseline.** It directly uses chosen/rejected data training, does not fit the reward model alone, nor does it do online Rollout; strictly speaking, it is not RL.
4. **Subjective tasks depend on high-quality preferences, and objective tasks give priority to verifiable rewards.** Code tests, mathematical answers, and tool execution results are usually more stable than purely manual scoring.
5. **RLAIF can expand the scale of feedback, but it cannot eliminate human governance.** Human beings still need to define principles, calibrate Judge, deal with conflicts and audit high-risk boundaries.
6. **Reward definition is the product definition.** If the indicators are incomplete, the model will optimize the proxy indicators instead of automatically understanding the real business goals.
7. **A high offline score does not mean online safety.** Reward Hacking, length preference, catering, excessive rejection, ability regression and distribution drift must be monitored.
8. **Most enterprises should not start with PPO.** The common and stable route is SFT → DPO; only when there is a reliable Grader, sufficient Rollout computing power and a mature training team, enter the online strengthening stage such as PPO/GRPO.

---

## 1. Where is RLHF located in the model life cycle?

The typical post-training pipeline is:

```text
base model
↓ High-quality demonstration
SFT Model
↓ Human / AI / Rules / Actuator Feedback
Preference Optimization or Reinforcement Learning
  ↓
Aligned Model
↓ Red team, system guardrail and online monitoring
Production System
```

RLHF cannot replace pre-training or SFT:

- The base model provides knowledge and general capabilities;
- SFT provides basic dialogue protocols, task formats and readable output;
- The preference/intensification stage improves the success rate of helpful, authentic, safe, style and reasoning;
- RAG, permissions, tool sandbox and content security layers are responsible for dynamic facts and systematic constraints at inference time.

If the SFT model is even unstable in the task format, doing RL directly will use a large amount of computing power to explore invalid output, and reward model may also learn error shortcuts in low-quality candidates.

## 2. Why is SFT not enough?

SFT assumes that there is a target answer \(y^*\), and maximizes its probability:

\[
L_{SFT}=-\log \pi_\theta(y^*\mid x)
\]

But open business tasks usually have multiple correct answers. It is easy for the annotator to judge:

```text
Answer A is more accurate, complete and safer than answer B.
```

But it is difficult to write a perfect answer from the blank. Preference learning changes the supervision signal from "reproducing this string of token" to "incrove the probability of better answers and relatively poor answers".

It is especially suitable for optimization:

- Helpfulness, integrity, conciseness and tone;
- Refuse to answer the border and security policy;
- The success rate of multi-step reasoning, code and tool execution;
- Business preferences between different and correct schemes;
- Combined goals such as user satisfaction and task completion rate.

It is not good at supplementing the knowledge that the base does not have out of thin air, nor can it make vague policies clear automatically.

## 3. First, clarify "narrow RLHF" and "broad alignment"

| Method | Feedback Data | Whether to train reward model | Whether to sample online | Whether to belong to RL |
|---|---|---:|---:|---:|
| SFT | Standard Answer | No | No | No |
| DPO / IPO, etc. | chosen vs rejected | No | Usually no | No |
| KTO | desirable / undesirable | No | Usually no | No |
| ORPO | Standard Answer + Negative Answer | No | No | No |
| Classic RLHF (PPO) | Preference pair → Reward | Yes | Yes | Yes |
| RLAIF | AI preference or AI Reward | Optional | Optional | Depending on the optimizer |
| RLVR / RFT | Program or Rule Score | Not Necessarily | Yes | Yes |
| GRPO | Sara Reward | No Need for Value Model | Yes | Yes |

In business communication, these are often collectively referred to as RLHF, but specific data, rewards and optimizers must be specified when reviewing the architecture, otherwise the cost and risk cannot be estimated.

## 4. What are the feedback signals?

### 4.1 Pair preferences

The same prompt generates two or more candidates, and the annotator chooses the better one:

```json
{
  "prompt": "Explain why the order was canceled",
  "chosen": "The order was automatically canceled due to the failure of inventory verification...",
  "rejected": "The system has canceled, just place a new order."
}
```

The advantage is that it is easier for humans to make relative judgments; the disadvantage is that it only expresses the local order, does not directly indicate how big the gap is, and may also be affected by the length, position and wording.

### 4.2 Binary feedback

Only mark the answer as desirable/undesirable, like/tamp or pass/fail. The collection is simple and suitable for KTO targets, but the exposure deviation needs to be controlled: the answers seen by users are not random samples, and silence is not the same as satisfaction.

### 4.3 Scalar score

For example, 1 to 5 points, task success rate, labor quality points. The scaler information is rich, but different annotators have inconsistent understanding of the ruler, and it is prone to 3-point and 4-point boundary drift.

### 4.4 Verifiable reward

Directly judged by the program:

- The final answer of mathematics;
- Unit testing, compilation and static analysis;
- SQL execution results in the sandbox;
- JSON Schema and business rules;
- Whether tool call completes the goal;
- Game, planning or simulator score.

Verifiable is not equal to the completeness of the target. For example, "test pass" may still contain low-quality code, and "order creation success" may also use the wrong permission. Therefore, it is usually combined with rewards such as format, security and cost.

### 4.5 AI, rules and constitutional feedback

The strong model compares candidates, scores, criticizes and rewrites them according to Rubric or a set of principles. It can quickly cover the long tail, but it will inherit the Judge deviation, and may also be injected into the attack by the prompts in the candidate answer.

### 4.6 Implicit Product Feedback

Clicking, copying, retrying, manual takeover, task completion, complaints and other signals are close to the business, but there are serious mixing factors. Interface location, user group, delay, candidate exposure and downstream process will all affect the results, and cannot be directly regarded as the preferred true value without causal analysis.

## 5. Scheme 1: Classic reward model + PPO

The classic InstructGPT route includes three steps:

```text
Manual demonstration -> SFT
SFT generates multiple candidates -> Manual sorting -> reward model
Current Policy Online Generation -> Reward Soring -> PPO Update
```

### 5.1 reward model

For the same prompt \(x\) preference answer \(y_w\) and non-preference answer \(y_l\), reward model output scalar \(r_\phi(x,y)\), using Bradley–Terry form training:

\[
P(y_w\succ y_l\mid x) =\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)
\]

\[
L_{RM}=-\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)
\]

What reward model has learned is the proxy function on the preference data, not the complete definition of "good" in the real world.

### 5.2 PPO Optimization Policy

The policy model generates the answer, the reward model scores, and uses KL punishment to limit its deviation from the reference model:

\[
\max_\theta\ \mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)} \left[r_\phi(x,y)-\beta D_{KL}\left(\pi_\theta\|\pi_{ref}\right)\right]
\]

PPO then limits the single update by cropping the target to reduce the risk of sudden collapse of the strategy.

### 5.3 System composition

Training often involves:

- Can train Policy;
- Freeze Reference Policy;
- reward model;
- Value/Critic Model;
- Rollout reasoning engine;
- PPO Trainer, experience cache and distributed communication.

This explains why the classic RLHF is obviously more complex than SFT/DPO: it puts high-throughput reasoning and distributed reverse training into the same online loop.

### 5.4 When is it worth using?

- It is necessary to continuously explore the distribution of the current Policy;
- The preference target is complex, and offline data cannot cover new behaviors;
- Have mature reward model, GPU cluster and online training capabilities;
- The quality benefits of the model are enough to cover several times the cost of engineering and computing power.

For most enterprises' customized tasks, PPO should not be the first version of the scheme.

## 6. Scheme 2: DPO and offline preference optimization

DPO directly uses preferences for training Policy, does not fit the reward model separately, and does not generate online in the training ring.

Given Policy \(\pi_\theta\), freeze reference model \(\pi_{ref}\), preferred answer \(y_w\) and non-preference answer \(y_l\), definition:

\[
z=\beta\left[ \log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)} - \log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)} \right]
\]

\[
L_{DPO}=-\log\sigma(z)
\]

DPO makes Policy more biased towards chosen relative to Reference, and at the same time uses Reference to implicitly constrain the distribution drift.

### 6.1 Business Advantages

- The process is close to ordinary supervision training, and it is easy to access the existing Trainer;
- No Need For Reward/Value Model And Online Rollout;
- Stable training and short experimental cycle;
- Suitable for teams with high-quality historical preferences.

### 6.2 Limitations

- The behavior that can only be fully utilized by offline data;
- If the preference pair is generated by the old model, there will be a distribution difference with the new Policy;
- chosen/rejected length, template or style leakage will be regarded as preferred features;
- Policy and Reference still need to be loaded at the same time, and the GPU memory is usually higher than SFT;
- \(\beta\), sequence probability aggregation and data quality will significantly affect the results.

### 6.3 Suitable business tasks

- Customer service tone, integrity and quality of rejection;
- Abstract, writing and brand expression;
- tool call scheme sorting;
- Experts can compare candidates, but cannot continue to write standard answers;
- Trusted chosen/rejected data has been accumulated from the production log.

## 7. Scheme 3: KTO, ORPO and other light preference targets

| Method | Data Requirements | Main Features | Application |
|---|---|---|---|
| DPO | Paired chosen/rejected | Ecologically mature, strong baseline | Reliable preference pair can be constructed for the same prompt |
| KTO | Independent good/bad sample | No strict pairing is required | Only like/tamp or audit pass/reject log |
| ORPO | chosen + rejected | Merge SFT with preference punishment, no Reference | Hope to reduce the training stage and model copy |
| IPO, etc. | Pair preferences | Modify the statistical nature of preference targets | Have the ability to research and find specific defects of DPO |

These methods are not the "upgrade chain" sorted by new and old papers. Data form, training stability, maturity and business evaluation jointly determine the choice. Default priority DPO; only when the existing data cannot be naturally formed into a pair of preferences, or when the Reference cost becomes a clear bottleneck, try to replace the target.

## 8. Scheme 4: RLAIF and Rule Reward

Reinforcement Learning from AI Feedback Use AI to replace or assist humans in providing feedback. The typical idea of Constitutional AI is:

1. Human beings define a set of principles or "constitution";
2. The model criticizes and revises its own answers based on principles;
3. Use the revised results for supervision and training;
4. AI Judge compares candidates according to the principle;
5. Use AI preferences for Preference Optimization or RL.

Business value:

- Extend to long-tail scenes that are difficult for humans to cover one by one;
- Quickly turn natural language policies into data generation and audit rules;
- Reduce the initial screening cost and allow humans to focus on dealing with conflict and high-risk samples.

Main risks:

- Judge shares bias or blind spots with the trained model;
- Complex policies are still ambiguous in natural language;
- Candidate text may prompt Injection to Judge;
- The same Judge is both generated and evaluated, and it is easy to form self-strengthening;
- AI's high consistency does not mean that it is consistent with real user preferences.

Therefore, the more stable structure is "AI expansion + program verification + manual calibration + independent red team", rather than completely canceling humans.

## 9. Scheme 5: RLVR / Reinforcement Fine-tuning

RLVR (Reinforcement Learning with Verifiable Rewards) directly trains Policy with verifiable results. It is especially suitable for mathematics, code, tool use and Agent with simulation environment.

```text
prompt
↓ Policy generates multiple candidates at once
Verifier / Grader scores each candidate
↓ Estimate Advantage
PPO / GRPO and other updates Policy
```

### 9.1 PPO and GRPO

PPO usually relies on Value Model to estimate Advantage. GRPO answers \(y_1,\ldots,y_G\) to the same prompt sampling group, and uses in-group rewards for relative standardization:

\[
A_i=\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)} {\operatorname{std}(r_1,\ldots,r_G)+\epsilon}
\]

Then update the model with strategic ratio, cropping and KL constraints. It eliminates a single Value Model, but still requires multiple candidate Rollout, and the generation cost may become the main bottleneck.

### 9.2 Which business is suitable for

- The code can run tests;
- Mathematics and logic questions have clear answers or proof checkers;
- SQL, search and planning can calculate the results in the sandbox;
- Agent's mission success, number of steps, cost and security constraints are observable;
- Models are needed to discover new strategies that are better than demonstration through exploration.

### 9.3 Which business is not suitable for direct work?

- Highly subjective goals such as "more creative copywriting" and "warmer reply";
- It takes several weeks for the results to be observed, and the attribution is unclear;
- Grader is easily deceived by the output text;
- High-risk actions cannot be performed in an isolated simulation environment;
- The basic success rate of the task is close to zero, and the sampling is almost not rewarded.

When the reward is extremely sparse, first use SFT to bring the model to the "occasional success" area, which is usually more effective than direct RL.

## 10. Mainstream scheme selection matrix

| Scene | Recommended starting point | Why | Main cost |
|---|---|---|---|
| Standard answer demonstration | SFT | Direct target, simple training | High-quality answers are required |
| Paired preference data | DPO | Stable, no need for online RL | Restricted by offline distribution |
| Only like/tread | KTO or first construct preference pair | Use non-paired feedback | Exposure and selection deviation |
| Want to merge SFT and preference stage | ORPO | No Reference required | Ecology and experience are less than DPO |
| Subjective open quality and sufficient resources | RM + PPO | Online exploration | Complex system, high cost |
| Mathematics/Code/Tools Have Verifier | GRPO/PPO + Verifiable Reward | Objective and Expandable Reward | Rollout Expensive, Easy to Drill Rule Loopholes |
| Safety specifications can be written as principles | RLAIF + manual calibration | Quick coverage long tail | Judge deviation and injection risk |
| Online models are constantly changing | Online preferences/RL | Reduce offline distribution misalignment | Complex data, training and release governance |

## 11. How to produce human preference data

### 11.1 Write Rubric first

Don't just tell the annotator to "choose a better one". Rubric should split the target into a judgmable dimension, such as:

1. The facts are correct;
2. Complete the user's explicit request;
3. Non-fictional source or system status;
4. Follow the security and permission policy;
5. The information is complete but not too long;
6. The tool selection and parameters are correct;
7. If the information is insufficient, explain it clearly and take the correct next step.

It is also necessary to stipulate the priority when there is a dimensional conflict. For example, when facts conflict with fluency, facts must take precedence.

### 11.2 Candidate Generation

There should be enough differences in the candidates, but not too different:

- Sampling from the current production model, candidate model and strong Teacher;
- Use multiple Temperature and random seeds;
- Mix in with historical real failure and confrontation samples;
- Remove the characteristics of leakage sources such as model names and templates;
- Randomize the candidate order to prevent position deviation.

If chosen is always longer and always generated by the same model, the Preference Model may only learn length or style, but not quality.

### 11.3 Label Quality Control

- Labeler training and qualification questions;
- Repeated annotation to estimate consistency;
- Insert the Gold sample of the known answer;
- Allow Tie, Both Bad and unjudgeable;
- Submit low-consistency samples to experts for arbitration;
- Record the annotator, Rubric version, time and reason;
- Use qualified experts in high-risk areas.

Differences are not necessarily noise. It may expose unclear product policies, conflicting preferences of different user groups, or prompt lacks the necessary context.

## 12. Reward and Grader Project

### 12.1 Multi-target reward

Business rewards are usually multiple components:

\[
r=w_1r_{correct}+w_2r_{helpful}+w_3r_{safe} -w_4c_{latency}-w_5c_{tool}-w_6p_{violation}
\]

Simple weighting is easy to cover up hard constraints. For example, security violations cannot be offset by "more helpful". The safer way to do it is:

- First, use hard rules to eliminate illegal output;
- Then optimize the quality in the legal collection;
- Rewards for different tasks are normalized separately;
- Monitor each component, not just look at the total Reward.

### 12.2 Grader Type

| Grader | Advantages | Risk |
|---|---|---|
| String/Schema | Fast, Confirm | Only check the surface form |
| Unit test/executor | Close to the success or failure of the task | The test is not complete, and the security sandbox is required |
| Rule Engine | Interpretable, Auditable | High maintenance cost and rigid boundaries |
| reward model | Fast, bulk | Distortion outside the distribution, will be used |
| LLM Judge | Flexible and Easy to Express Rubric | Deviation, Cost, Injection and Drift |
| Artificial expert | High context understanding | Slow, expensive, limited consistency |

Key business should combine multiple Graders and keep evaluators independent of training rewards to prevent "being both a coach and a referee".

### 12.3 Prevent Reward Hacking

- Do a confrontation test on Grader before training;
- Separate hidden tests from public tests;
- Restrict answer access to Grader tips, tests and reference answers;
- Monitor the length, repetition, rejection rate and abnormal token mode;
- Regular manual inspection of high Reward samples;
- Add newly discovered counterexamples to the training;
- Do not allow the model to directly modify the reward function, test or execution environment;
- Use minimum permissions and isolated sandbox running tools.

## 13. Online training closed loop

The mature preference system is not a one-time training, but a controlled iteration:

```text
Production/Task prompt Pool
↓ De-privacy, layered sampling
Current Policy generates candidates
  ↓
Human / AI / Verifier Feedback
↓ Quality control, deduplication, versioning
SFT / DPO / PPO / GRPO
  ↓
Offline evaluation, red team, return
  ↓
Shadow / Canary / A/B
  ↓
Monitor and reflow new failed samples
```

Each round should record data generation Policy. When using the preferences generated by the old model to train stronger models, insufficient offline coverage will gradually become a bottleneck, and it is necessary to re-sampl the candidates of the current Policy.

## 14. Evaluation cannot reuse training rewards.

At least establish the following evaluation surface:

| Evaluation surface | Indicator example |
|---|---|
| Preferred Quality | Manual Blind Test Win Rate, Pairwise Win Rate, Tie Rate |
| Task Correctness | Exact Match, F1, Pass@k, Execution Success Rate |
| Calibration and Honesty | Rejection, citation correctness rate, hallucination rate when uncertain |
| Security | Harmful response rate, excessive rejection rate, over-tool call rate |
| General regression | Knowledge, code, mathematics, multilingual and instruction following |
| System Efficiency | P50/P95 Delay, Output Length, Number of Tools, Unit Success Cost |

Offline Judge should satisfy:

- Different from training Judge or at least use hidden Rubric;
- The candidate order is random, and the exchange order is re-evaluated if necessary;
- Calibrate length preferences, identity preferences and self-preferences;
- Regularly compare the consistency with manual experts;
- The evaluation set scrolls by time, and a set of permanent regression sets is retained.

Use small-traffic Canary when going online, and set automatic rollback thresholds for quality, safety, cost and delay.

## 15. The most common failure pattern

### 15.1 Reward Hacking

Reward rises but real quality declines. The model may learn to stack keywords, output longer content, forge proofs, skip difficult steps or attack Judge.

### 15.2 Over-optimization and ability collapse

Policy deviates too far from Reference, and language quality, diversity or general ability decline. It can be alleviated by KL, lower learning rate, early stop, SFT data mixing and multidimensional regression monitoring.

### 15.3 Length preference

The annotator and Judge often misjudge the longer answer as more complete, so the model is constantly verbose. Equal-length comparison should be constructed, the length should be recorded separately, and "necessary and sufficient" should be made clear in Rubric.

### 15.4 Sycophancy

The model caters to the wrong views of users in order to prefer scores. Rubric should put facts and evidence above flattering, and specifically join the confrontation set of "users insist on the wrong premise".

### 15.5 Excessive refusal to answer

When the security reward is too strong, the model also rejects the normal request. The security evaluation must contain adjacent boundary samples that should be rejected and answered normally.

### 15.6 Distribution Drift

reward model performs well on the old Policy sample, but the strange output of the new Policy is not reliable. Online sampling, periodic relabeling and uncertainty detection can reduce risks.

### 15.7 Single average preference

The preferences of different customers, regions and roles may conflict. Pressing all preferences into a single Reward will smooth out the difference. Conditional strategies, multiple Adapters, explicit user settings or scenario-based routing can be considered, instead of assuming that there is a unique "human preference".

## 16. Security and governance

RLHF training must be incorporated into complete data and model governance:

- Clarify whether the user feedback has been authorized for training;
- Remove PII, key, internal identification and restricted data;
- Retain data sources, permissions, deletions and audit links;
- Restrict sensitive information seen by annotators;
- Access control for reward model, Judge prompt and test set;
- High-risk Tool is only Rollout in the sandbox or simulator;
- Carry out safety review, red team and rollback drills before release;
- Do not regard training alignment as the only safety boundary.

The behavioral constraints in the model weight are probabilistic. Permission, transaction confirmation, content filtering and compliance verification still need to be carried out by the determinative system.

## 17. Recommended business landing order

### The first stage: no need RL

1. Define business KPI, failure classification and security boundaries;
2. Establish the prompt/RAG/Tool baseline;
3. Use expert demonstration to do SFT;
4. Fixed Golden Set, regression set and online monitoring.

### The second stage: offline preference optimization

1. Construct high-quality preference pairs from real failure and model candidates;
2. Do the DPO baseline first;
3. Use manual blind test to verify whether it is really better than SFT;
4. Analyze the benefits from facts, style, safety or length;
5. Only when the data form is clearly not suitable for DPO, try KTO/ORPO and other targets again.

### The third stage: verifiable enhancement

1. Build a Verifier that cannot be easily used;
2. Generate multiple Rollouts in the isolated environment;
3. Run small-scale PPO/GRPO experiments first;
4. At the same time, monitor Reward, real task success rate and general regression;
5. Manually check the highest reward and abnormal strategy;
6. After the income is stable, the scale of prompt, model and computing power will be expanded.

### The fourth stage: online closed loop

Only after data compliance, labeling quality, training stability, release access control and rollback mechanisms are mature, will continuous online preference collection and iterative training be introduced.

## 18. One-page selection conclusion

```text
Can you write a clear ideal answer?
├─ Can -> Do SFT first
└─ Can't
      ↓
Can you stably compare the good and bad of the two answers?
├─ Can -> Do DPO first
└─ Only good/bad can be marked -> KTO or reconstruct data

Can the output be verified by the program or environment?
├─ Can -> Try RLVR (PPO/GRPO) after SFT
└─ No -> Human preference / RLAIF + manual calibration

Do you need the current Policy to continue to explore new answers?
├─ Yes, and the resources and governance are mature -> Online RM+PPO or verifiable RL
└─ No -> Offline preference optimization is usually more cost-effective
```

The default suggestion is:

> Most businesses start with "strong base + high-quality SFT + DPO + complete evaluation"; when the task has reliable and difficult-to-use verifiable rewards, and the offline method has reached the bottleneck, PPO/GRPO and other online intensive training are then invested.

---

## Extended reading

- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- [ORPO: Monolithic Preference Optimization without reference model](https://arxiv.org/abs/2403.07691)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [DeepSeekMath: GRPO](https://arxiv.org/abs/2402.03300)
- [OpenAI: Rule-Based Rewards for Language Model Safety](https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/)
- [NVIDIA NeMo RL: DPO](https://docs.nvidia.com/nemo/rl/latest/guides/dpo.html)
- [Amazon Bedrock: Reinforcement Fine-tuning](https://docs.aws.amazon.com/bedrock/latest/userguide/reinforcement-fine-tuning.html)
- [This project: miniLLM DPO Training](../Hands-On/miniLLM%20DPO%20Training.md)
- [This project: LLM SFT Algorithms Explained](../Algorithms/LLM%20SFT%20Algorithms%20Explained.md)

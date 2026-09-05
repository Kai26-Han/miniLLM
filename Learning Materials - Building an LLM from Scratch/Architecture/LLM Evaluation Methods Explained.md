# Evaluating LLMs: From Offline Benchmarks to Safety Gates and Online Feedback Loops

> Eval's goal is not to create a good total score, but to provide repeatable decision-making evidence for training, architecture, release and operation. This article focuses on the general evaluation system of the industry and the practice of open companies, and finally briefly describes miniLLM.

## 1. Eval is a decision-making system.

Complete Eval answers five types of questions:

1. **Capability**: What tasks can the model complete and at what difficulty does it start to fail?
2. **Behavior**: Do you follow the instructions, express uncertainty and maintain the expected style?
3. **Security**: Which requests will cause harmful output, jailbreak, false refusal or high-risk capacity amplification?
4. **Efficiency**: How much delay, throughput, GPU memory and cost do you need to achieve this quality?
5. **Reliability**: Are the results stable in different Prompts, sampling, languages, user groups and environments?

If an indicator does not change any training or release decision, it may only display data, not effective Eval.

## 2. Define the evaluation object first

The "model score" may correspond to at least four different objects:

| Object | Contains content | Suitable for answer |
|---|---|---|
| base model | Weight + tokenizer | Pre-training knowledge and continuation ability |
| Chat Model | Weight + chat template + System prompt | Dialogue Behavior and instruction following |
| Inference Policy | Sampling, Thinking Budget, Tools, Search | The Ability To Calculate The Budget Given |
| Product System | Model + RAG + Guardrail + UI + Monitoring | User Actual Experience and Risk |

The model version is the same, but the prompt, tool, context, sampling number or thinking budget are different, and the result is no longer the same system.

## 3. Five core principles

### 3.1 Define decision-making first, and then design indicators

"Whether to release", "whether to replace the old model" and "whether to expand the training" require different evidence. First, write down the passing threshold and the failure action, and then look at the results to avoid picking indicators after the fact.

### 3.2 Freeze Agreement

At least freeze: model and tokenizer, chat template, System prompt, Few-shot example, sampling parameters, maximum output, tool version, Judge and data set version.

### 3.3 Layering instead of just looking at the total score

An average number will cover up the regression of language, difficulty, security domain and high-value users. It is necessary to report the sub-domain, sub-difficulty, sub-group and the worst-performing quantile.

### 3.4 Report uncertainty

Bootstrap confidence intervals can be used for accuracy; pairing tests can be used for model comparison. If the 0.3 point increase falls within the noise range, it should not form a strong conclusion.

### 3.5 Joint report on quality, cost and safety

The reasoning model can exchange more sampling or longer thinking in exchange for scores. Fair comparison requires reporting the output token, sampling number, delay and cost at the same time.

## 4. The whole life cycle Eval

### 4.1 tokenizer Stage

- Round-trip, special token and chat template correctness;
- `bytes/token` of sub-language/code/mathematics;
- Long-tail characters and extreme lengths;
- Downstream loss and throughput of different vocabularies.

### 4.2 Pre-training stage

- token-weighted Validation Loss;
- Subdomain PPL/Bits per Byte;
- Data contamination and memory test;
- Fixed Base ability set;
- Training stability, throughput and MFU.

### 4.3 SFT aligns with preferences

- Instruction compliance, format and multi-round consistency;
- Pairwise Preference;
- Excessive refusal to answer, catering and style degradation;
- Basic knowledge and reasoning ability return.

### 4.4 Reasoning RL

- Pass@1 and Pass@k;
- Solve the problem token, think about the budget and calibrate the answer;
- Verifier pass rate and Reward Hacking;
- Difficulty stratification and cross-domain generalization.

### 4.5 Release and Online

- Safety red team and abuse stress test;
- TTFT, TPOT, throughput, timeout and cost;
- A/B task completion rate, user correction rate and upgrade labor rate;
- Online drift, fault clustering and continuous regression.

## 5. Six types of scorers

### 5.1 deterministic rules

Exact Match, regularity, Scheme, format and keywords. Cheap can be reproduced, but semantic coverage is limited.

### 5.2 Reference Answer Indicators

F1, ROUGE, BLEU, BERTScore, etc. are suitable for tasks with reference output. In open generation, multiple correct expressions will make single reference indicators underestimate the quality.

### 5.3 Executable Validator

Code unit testing, compiler, SQL execution, mathematical equivalence checking and environment state are high-value signals. They are closer to whether the task is really completed, but they need to prevent incomplete testing and environmental leakage.

### 5.4 LLM-as-a-Judge

Judge can evaluate open answers, pair preferences and multi-round trajectories according to the Rubric evaluation, which is highly extensive. The main deviations include:

- Position Bias;
- Verbosity Bias;
- Self-preference;
- Style or format preference;
- Judge is of the same origin as the tested model;
- prompt Injection;
- The score drifts.

Mitigation methods include exchanging answer orders, hiding model identity, clarifying Rubric, requiring structured reasons, multi-judjuse arbitration, and using manual labeling to calibrate the consistency rate.

### 5.5 Manual evaluation

Human evaluation is suitable for product experience, cultural context, complex security and Judge calibration. Rubric, sample, blind test, repeated sample and labeler consistency must be provided.

### 5.6 Red Team and Confrontation Assessment

The goal is to actively find the boundary of failure, not to estimate the average performance of ordinary requests. Automatic attack is suitable for coverage, and expert red team is suitable for high-impact links. The two should be combined with real abuse monitoring.

## 6. Pre-training indicators

### 6.1 token-weighted Cross-Entropy

\[
L=\frac{\sum_b N_bL_b}{\sum_bN_b}
\]

batch Loss cannot be simply averaged, because the number of valid tokens for each batch may be different.

### 6.2 Perplexity

\[
\operatorname{PPL}=e^L
\]

It is only suitable for comparison with tokenizer, data and Mask protocol.

### 6.3 Bits per Byte

\[
\operatorname{BPB}=\frac{L\cdot N_{token}}{N_{byte}\ln2}
\]

It is fairer for cross-tokenizer, but it still requires the same original text and preprocessing.

### 6.4 Subdomain Loss

Separate reports on Web, books, Chinese, English, code, mathematics, etc. can find local degradation caused by data formulas or tokenizer.

## 7. Ability assessment map

### 7.1 Knowledge and facts

Closed-volume questions and answers, time-effectiveness knowledge, long-tail knowledge, citation veriability and hallucination rate should be separated. The RAG system also needs to distinguish between retrieval failure, insufficient evidence and unfaithful generation.

### 7.2 Mathematics and Reasoning

Pay attention to the correctness of the answer, process verifiability, difficulty curve and sampling budget. GSM8K, MATH and AIME tasks each cover different difficulties, and a single collection is prone to saturation or contamination.

### 7.3 Code

HumanEval/MBPP test function generation, SWE-bench class task test real warehouse problems. To fix the container, test, timeout, tool permissions and the number of retries.

### 7.4 Multilingual and Chinese

Translated English questions cannot fully represent local knowledge, expression and cultural context. There should be native questions, cross-language consistency, low-resource language and code mixed input.

### 7.5 Instructions and Formats

IFEval, JSON Schema, function parameter constraints and other executable verifications are better than subjective scoring. Conflict instructions, negative constraints and long instructions lists should also be tested.

### 7.6 more than a round of dialogue

Evaluate state maintenance, error correction, reference, long-term constraint, forgetting and context contamination. You can't simply string independent single-round questions and claim to be multi-round Eval.

### 7.7 Long context

Needle-in-a-Haystack only tests shallow retrieval, and also requires more Needle, sequence, aggregation, long-text questions and answers and real document tasks; report the location, length and interference intensity respectively.

### 7.8 Agent and Tools

Based on the complete trajectory: task success rate, tool selection, parameter correctness, invalid call, recovery ability, number of steps, cost and side effects.

## 8. Preference and open generation

Pair comparisons are usually more stable than 1–10. If the win rate of A to B is \(p\), Bradley–Terry or Elo aggregation can be used, but report:

- How to deal with a draw;
- Whether the order is exchanged randomly;
- prompt distribution and time window;
- Judge/User group composition;
- Whether the length and style are confused.

Arena evaluation reflects specific traffic and preferences, and is not equivalent to the ability ranking of all professional scenarios.

## 9. Safety assessment

### 9.1 The three concepts should be separated

- **Harmful content compliance**: Whether the request that should not be completed has been completed;
- **Rejected answer by mistake**: Whether the normal request was wrongly rejected;
- **High-risk ability**: Whether the model significantly lowers the threshold for serious injury.

Only reducing the harmful compliance rate may be cheating through "all rejection", so the False Refusal Rate must be jointly reported.

### 9.2 Main security domains

Different experts, Threat Model and scoring standards are required in the fields of network security, CBRN, autonomous action, deception/manipulation, privacy, minors, self-harm, hatred and illegal activities.

### 9.3 Uplift and System Defense

High-risk assessment should compare how much ability the model has added relative to the search, textbook or expert baseline, rather than just asking whether the model can answer the question. It is also necessary to measure the model capability, deploy Guardrail, access control and the remaining risks after monitoring.

## 10. Efficiency and deployment evaluation

- **TTFT**: The first token delay;
- **TPOT/ITL**: Adjacent output token delay;
- **Throughput**: request or token per second;
- **Goodput**: Meet the effective throughput of SLO;
- **Peak Memory**: weight, KV Cache, activation and workspace;
- **Cost**: Cost per million input/output token or cost per task completion.

Performance testing requires fixed hardware, accuracy, quantification, input/output length, concurrency, batch strategy and cache status. The average delay cannot replace P95/P99.

## 11. Benchmark The most common distortion

1. Training data contamination or answer memory;
2. The topic is saturated, and the new model cannot be distinguished;
3. prompt/chat template is specially optimized for a model;
4. Different models use different sampling or thinking budgets;
5. Only report the best run;
6. Judge prefers length, position or his own style;
7. Repeatedly Adjusting References With The Same Question Causes Eval Overfitting;
8. Only look at the average value, not the high-risk failure;
9. Compare the knowledge of different deadlines without explanation;
10. Confusing model ability, scaffolding and tool ability.

Goodhart's law is very direct here: once a single Benchmark becomes a target, it is more likely to lose its value as a measure.

## 12. Public Eval practice of mainstream companies

### 12.1 OpenAI

Public System Card reports capability, security, red team and preparedness assessment at the same time; the Preparedness Framework connects high-risk capability thresholds with deployment mitigation, review and release decision-making. SWE-bench Verified also reflects the re-audit of the benchmark unsollable questions and labeling quality.

### 12.2 Google DeepMind

Frontier Safety Framework uses Critical Capability Level, early warning Eval and corresponding mitigation plans; model card coverage, responsibilities and limitations. Public work emphasizes Holistic Evaluation and external evaluation.

### 12.3 Anthropic

Public routes include model generation evaluation set, expert red team, comparison of different training snapshots/security status and Responsible Scaling Policy. One of the key points is to use Eval to discover the gradual ability during training, rather than just measuring the final model.

### 12.4 Meta

The Llama model card combines standard Benchmark, manual preferences, security testing and open tools; CyberSec Eval, Purple Llama, Llama Guard and other projects make some security evaluations and protections reusable.

### 12.5 DeepSeek, Qwen and Mistral

These public model reports usually cover knowledge, mathematics, code, multilingual and open dialogue; the reasoning model emphasizes multiple sampling, Pass@k and generation length, and the tool model adds Function Calling/Agent evaluation. When reading, you must check the prompt, thinking mode and sampling budget of each table.

## 13. Common trends in the industry

1. From static Benchmark to continuous, private and dynamic generation of Eval;
2. System evaluation from single model to model + tool + retrieval + Guardrail;
3. From single accuracy to mass-computation curve;
4. From purely automatic or purely manual to the mix of rules, Verifier, Judge and human evaluation;
5. From the final score to the ability threshold, risk level and release threshold;
6. From average users to sub-language, group, difficulty and worst-case;
7. From offline primary acceptance to online feedback-driven regression set.

## 14. A set of implementable Eval architecture

```text
Versioning data set and task definition
-> prompt / Tool / Sampling Configuration
-> Harness can be re-executed
  -> Rule / Verifier / Judge / Human Scoring
-> Hierarchical indicators + confidence interval + cost
-> Failed clustering and manual audit
-> Release threshold / Training data backflow
-> Continuous return
```

Each sample is saved at least: sample ID, data version, prompt, original output, structured score, scorer version, token/delay/cost and error labels. Saving only the aggregate score will make the failure analysis impossible.

## 15. Eval mapping of miniLLM

miniLLM has been able to calculate token-weighted Validation Loss/PPL and use fixed prompt for text generation; tokenizer Eval covers special ID, structure token, Round-trip and domain samples. This is enough to do the training Smoke Test, but it is not a complete ability assessment.

The next step should be kept light: first, separate the verification data from the training source and report it in Chinese, English, code and other domains; then establish dozens to hundreds of executable small Base/SFT regression sets; and finally supplement TTFT, TPOT, throughput, and Dense/MoE routing indicators. This is not only in line with the scale of the small project, but also allows reliable evidence for each architecture and algorithm change.

## Reference materials

- [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110)
- [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132)
- [OpenAI Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/)
- [Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)
- [Google DeepMind Frontier Safety Framework](https://deepmind.google/frontier-safety/)
- [Anthropic Responsible Scaling Policy](https://www.anthropic.com/responsible-scaling-policy)
- [Meta Purple Llama](https://ai.meta.com/llama/purple-llama/)
- [SWE-bench](https://arxiv.org/abs/2310.06770)

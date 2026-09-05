# LLM Pretraining Algorithms: Objectives, Data, Optimization, and Scaling

> The pre-training algorithm is not just a Cross-Entropy formula, but a closed loop composed of data sampling, sequence construction, optimizer, numerical accuracy and distributed system. This article focuses on the general methods in the industry, and finally briefly describes miniLLM.

## 1. What exactly is the pre-training doing?

Pre-training allows models to learn language, knowledge, code and world patterns from large-scale unlabeled or weakly labeled data. For the current mainstream Decoder-only LLM, the core goal is Causal Language Modeling (CLM):

\[
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})
\]

Training minimizes the negative logarithm on the effective token:

\[
L_{LM}=-\frac{1}{N}\sum_{t\in\mathcal V}\log p_\theta(x_t\mid x_{<t})
\]

\(\mathcal V\) is a collection of locations involved in supervision. Padding, partial control marking or cross-document boundaries can be excluded according to the formula.

The direct product of pre-training is base model: it is good at continuing and has a lot of potential abilities, but it may not follow the instructions steadily. SFT, preference optimization and reasoning RL belong to the post-training stage.

## 2. Mainstream pre-training goals

### 2.1 Causal Language Modeling

When inputting `[x0, x1, x2, x3]`, the output of position 0–2 is usually used to predict `x1, x2, x3` respectively. Most tokens in the same sequence can become monitoring signals, and the training and generation forms are consistent, which is the mainstream target of the general generation model.

### 2.2 Masked Language Modeling

Randomly cover part of the token, and use the left and right context to restore. The BERT-class Encoder model uses this route, which is suitable for understanding and representation, but there is a difference in the training-inference form from the token-by-token generation.

### 2.3 Denoising / Span Corruption

Destroy the input fragment, and then restore the original text by Encoder–Decoder. T5 uses Span Corruption, which is suitable for unifying a variety of Text-to-Text tasks.

### 2.4 Prefix LM and Mixed Mask

The prefix part can be read in both directions, and the generated part maintains cause and effect. It is valuable in specific input and output tasks, but the semantics of Mask and KV Cache are more complex.

### 2.5 Multi-token Prediction (MTP)

In addition to predicting the next token, use the auxiliary header to predict several further tokens. It increases the density of supervision and may improve representation or provide structure for speculative decoding. The open training program of DeepSeek-V3 includes MTP, but it is an enhancement target and does not replace the main CLM.

## 3. The data algorithm determines the upper limit of the training signal.

### 3.1 Data processing main link

```text
Collect
-> Document analysis
-> Language/format recognition
-> Precise deduplication and approximate deduplication
-> Quality, security and privacy filtering
-> Data source and license record
-> Domain mixing and resampling
 -> Tokenize / Packing
-> Train/Validation contamination Inspection
```

Only increasing the original byte volume does not guarantee improvement; duplication, template garbage and low-quality synthetic content will waste calculations and amplify memory and evaluation contamination.

### 3.2 Remove weight

- Precise hash removal and complete repetition;
- MinHash/LSH or n-gram method finds near-duplication;
- URL, documents, paragraphs and cross-data sources should be processed in layers;
- Do overlap testing with public Benchmark.

The threshold is too loose to retain contamination, and too strict will mistakenly delete common phrases and legal references. deduplication requires recording the reason for the hit and sampling and auditing.

### 3.3 Quality Filtering

Often combined with rules, classifiers and models to score: language confidence, confusion interval, advertising/navigation density, garbled proportion, code resolution, educational value and security strategy.

High quality is not equal to a single style. Excessive filtering will cause the model to lose spoken language, dialect, real error distribution and long-tail knowledge.

### 3.4 Data mixing and temperature sampling

The original proportion of \(i\) in the domain is \(p_i\), which can be adjusted by temperature \(\alpha\):

\[
q_i=\frac{p_i^\alpha}{\sum_jp_j^\alpha}
\]

\(\alpha<1\) will increase the proportion of small data domains. Multilingual, code, mathematics and high-quality knowledge domains often need to be resampled, and the final ratio must be verified by small-scale experiments.

### 3.5 Synthetic data

Synthetic data can supplement reasoning trajectories, code tests, long-tail questions and answers, and structured samples, but control:

- Teacher's deviations and mistakes;
- The decline of homogenization and diversity;
- Leakage with evaluation set;
- The cost of generation, filtering and verification.

Validable domains give priority to compilers, unit tests and mathematical validators, rather than relying only on LLM scoring.

## 4. From documents to training sequences

### 4.1 token Stream and Packing

Sample-by-sample Padding to a fixed length will waste a large amount of token. Mainstream training more often organizes multiple documents into continuous token Stream or Packed Sequence, and then cut into fixed-length blocks.

Effective token utilization rate:

\[
U=\frac{N_{valid}}{N_{allocated}}
\]

It should be close to 1, but it needs to correctly handle EOS, Position ID, document boundary Mask and cross-document Loss semantics.

### 4.2 Sequence Length Course

Short sequence training has higher throughput, and long sequence training cultivates long context ability. The actual formula can first complete the subject pre-training with a shorter sequence, and then use long data for Context Extension or Mid-training.

### 4.3 Independent validation set

The validation set should be isolated from the training source and layered by domain. Only row-by-line sampling from the same file is prone to near-repeating leaks, which will also allow mixed PPL to cover up the degradation of a domain.

## 5. Forward communication and language model loss

Typical Decoder Layer:

\[
h'=h+\operatorname{attention}(\operatorname{Norm}(h))
\]

\[
h''=h'+\operatorname{FFN/MoE}(\operatorname{Norm}(h'))
\]

Finally:

\[
z_t=W_{vocab}\operatorname{Norm}(h_t),\qquad p_t=\operatorname{softmax}(z_t)
\]

When implementing Cross-Entropy, logits are usually transmitted directly, which is completed by the numerically stable `log_softmax + NLL` kernel, and the probability should not be calculated manually.

### 5.1 Perplexity

\[
\operatorname{PPL}=\exp(L_{LM})
\]

PPL can only be directly compared under the same tokenizer, data, Mask and weighting methods. Cross tokenizer is more suitable for reporting Bits per Byte.

## 6. batch, gradient accumulation and token budget

Under the parallel data, the number of sequences of the global batch is approximately:

\[
B_{global}=B_{micro}\times A\times W
\]

Among them, \(A\) is the number of gradient cumulative steps, and \(W\) is the number of data parallel processes. A more accurate training budget should be calculated with valid token:

\[
T_{step}=\sum\text{valid tokens across workers and micro-batches}
\]

Under the variable length sequence or Packing, reporting only "batch Size" may cover up the actual token size.

When the gradient is accumulated, Loss should be divided by the cumulative number of steps, or it should be accurately normalized according to the global valid token; when there is less than a complete accumulation window in the end, scaling errors should also be avoided.

## 7. AdamW: Common baseline for large model training

Adam maintains the first-order and second-order moments:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

AdamW decouples Weight Decay from Adaptive Gradient Update:

\[
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}-\eta\lambda\theta_t
\]

Norm and Bias parameters are often not Weight Decay, but whether they are grouped should be confirmed by formulas and experiments. Adafactor, Lion, Muon and other optimizers are also branches of research and practice, and AdamW is still one of the most common stable baselines.

## 8. Warmup, main learning rate and attenuation

The second-order moment estimate in the early stage of training is unstable, and the direct use of the peak learning rate is easy to produce Loss Spike, so Warmup is linear first:

\[
\eta_t=\eta_{max}\frac{t}{T_w},\quad t<T_w
\]

After that, Cosine Decay is often used:

\[
\eta_t=\eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min}) \left[1+\cos\left(\pi\frac{t-T_w}{T-T_w}\right)\right]
\]

There are also models that adopt WSD (Warmup–Stable–Decay): maintain a stable learning rate for a long time and attenuate at the end of training, which facilitates the flexible expansion of the token budget.

## 9. Initialization, normalization and stability

Large-scale training requires collaborative control:

- Parameter initialization scale and residual branch scaling;
- Pre-Norm/RMSNorm or mixed Norm structure;
- Learning rate, Warmup and batch token number;
- Gradient cropping;
- attention logits and QK-Norm;
- Numerical accuracy and load balancing of MoE router;
- Data abnormality, ultra-long repetition and bad batch detection.

Loss Spike should not be covered up by rolling back. To save the data cursor, optimizer status, random number status, and locate whether it is data, numerical value, communication or hardware failure.

## 10. Mixing accuracy

### 10.1 FP16 and BF16

- The tail number of FP16 is more refined, but the index range is smaller, and Dynamic Loss Scaling is often required;
- The BF16 index range is close to FP32, the training is more stable, and it is widely used in modern accelerators;
- Master Weight, optimizer status and partial normalization/reduction may still retain FP32.

### 10.2 FP8

FP8 can significantly improve throughput and reduce memory bandwidth, but it needs to be monitored by Tensor/Block Scaling, precision-sensitive operator whitelist and abnormality. DeepSeek-V3, Llama 4 and other public materials show large-scale FP8 training practices.

The goal of low accuracy is to maintain end-to-end convergence quality, not to force all operators to use the same dtype.

## 11. GM memory optimization and parallel algorithm

### 11.1 Activation Checkpointing

Only save part of the activation, recalculate the remaining intermediate values during reverse propagation, and exchange the GPU memory with additional calculations.

### 11.2 Data Parallel / DDP

Each process saves a complete model, processes different data, and then All-Reduce gradients. It expands the throughput, but does not reduce the weight of the single GPU model and the optimizer state.

### 11.3 FSDP / ZeRO

Cross-device cutting parameters, gradients and optimizer states, so that larger models can enter the cluster.

### 11.4 Tensor, Pipeline, Context and expert Parallel

- Tensor Parallel: Tensor Parallel;
- Pipeline Parallel: cutting and layering;
- Context/Sequence Parallel: cut long sequences;
- expert Parallel: MoE expert is distributed to different devices.

Industrial training is usually a multi-dimensional parallel combination. The optimal combination depends on the model shape, sequence length, network topology and failure recovery requirements.

## 12. Scaling Law and Budget Calculation

The training FLOPs of Dense Decoder are commonly used in rough terms:

\[
C\approx 6ND
\]

\(N\) is the parameter size involved in the calculation, and \(D\) is the number of training tokens. It is suitable for order-order estimation and does not include attention length effect, MoE routing, recalculation and system inefficiency.

Kaplan and other work studied the power law relationship between models, data and calculations; Chinchilla emphasized that under a fixed calculation budget, many early model parameters were too large and the training token was insufficient. Subsequent public models often choose more tokens to take into account performance, deployment amortization and data availability.

The real computational optimization is related to goals: one-time training cost, long-term inference cost, target ability and data quality may lead to different configurations.

## 13. Joint optimization of data, model and system

Mainstream pre-training no longer separates algorithms from systems:

- GQA/MLA is designed for KV Cache and reasoning bandwidth;
- MoE is designed for parameter capacity, All-to-All and expert Parallel;
- Packing and FlashAttention improve the utilization rate of token and IO at the same time;
- FP8 requires the joint adaptation of model scale, kernel and network attribution;
- Long context requires the cooperation of data courses, location methods and Context Parallel.

Only comparing theoretical FLOPs will omit communication, recalculation, Padding, failure and low utilization.

## 14. Training monitoring and acceptance

### 14.1 Learning Indicators

- token-weighted Train/Validation Loss;
- Domain PPL or Bits per Byte;
- Learning rate, gradient norm, parameter norm;
- Data domain proportion and valid token/s;
- Stage results of fixed ability sets.

### 14.2 System Indicators

- MFU, device throughput and omput time;
- Peak memory, communication ratio, recalculation ratio;
- DataLoader waiting time;
- NaN/Inf, Loss Spike, hardware errors and time-consuming recovery.

### 14.3 MoE Special Indicators

- The number of expert token and the load variation coefficient of each layer;
- router probability, entropy and auxiliary loss;
- Capacity Overflow/token Dropping;
- All-to-All time and expert GEMM utilization rate.

## 15. A reliable pre-training experimental sequence

1. A fit test has been done with small data to prove that Loss, Mask and Shift are correct;
2. Fixed data, tokenizer and Eval, and establish a Dense small model baseline;
3. Do learning rate, batch token and initialization scanning;
4. Verify Packing, mixing accuracy and Checkpoint recovery;
5. Then introduce variables such as GQA, MoE, MTP or long context;
6. Use the small model Scaling experiment to predict the large training;
7. Freeze the official formula and conduct a phased evaluation;
8. Retain the list of repeatable experiments, data versions and failure records.

Each experiment only answers one main question, which is more explanatory than changing ten superparameters at the same time.

## 16. Pre-training mapping of miniLLM

miniLLM uses Decoder-only CLM, token-shifted Cross-Entropy, AdamW, linear Warmup + Cosine Decay, gradient accumulation, gradient cropping and CUDA hybrid accuracy; supports DDP, Activation Checkpointing, Dense/MoE switching, regular verification and checkpoint recovery. The default model is layer 8, hidden dimension 768, GQA, and 4 expert/Top-1 MoE can be enabled.

The main gap between it and the industrial formula is concentrated in data and system: at present, each sample is independently truncated/Padding to 512, and there is no Packing; the validation set is divided according to the fixed line number of the same data source; DDP does not cut the model state; MoE does not have expert Parallel and capacity control. Therefore, the most priority algorithm upgrade should be the independent sub-domain validation set and token Packing, followed by expanding the context or increasing the number of experts.

## Reference materials

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Megatron-LM](https://arxiv.org/abs/1909.08053)
- [ZeRO](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [FlashAttention](https://arxiv.org/abs/2205.14135)

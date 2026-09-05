# LLM RL 阶段主流算法解析：从 RLHF、PPO 到 GRPO、DAPO 与 GSPO

> 大模型 RL 不是“换一个 Loss 继续训练”这么简单，而是策略采样、奖励构造、优势估计、策略约束和分布式 Rollout 共同组成的在线闭环。本文系统解释 PPO、REINFORCE/RLOO、ReMax、GRPO、Dr.GRPO、DAPO 与 GSPO，并厘清 RLHF、RLAIF、RLVR 和 DPO 的边界，最后给出 miniLLM 的演进建议。

## 1. RL 在大模型训练链路中的位置

一条常见的现代大模型训练链路是：

```text
Tokenizer
  -> Pretraining / Mid-training
  -> SFT / Reasoning Cold Start
  -> Offline Preference Optimization（DPO 等，可选）
  -> Online RL（PPO / RLOO / GRPO 系等，可选）
  -> Safety、Eval、蒸馏与部署
```

不同阶段提供的监督信号不同：

| 阶段 | 数据或反馈 | 模型学习的核心内容 |
|---|---|---|
| SFT | 人类或 Teacher 的目标回答 | 模仿“一个好回答长什么样” |
| DPO 等离线偏好优化 | 固定 chosen/rejected 对 | 提高 chosen 相对 rejected 的概率 |
| RLHF / RLAIF | 人类或 AI 偏好训练出的 Reward Model | 在当前策略产生的新回答上最大化偏好奖励 |
| RLVR | 答案、测试、编译器、形式系统等 Verifier | 通过可验证结果发现新的高奖励轨迹 |
| Agent RL | 环境状态、工具执行结果、任务成败 | 学习多步决策、恢复和停止 |

SFT 的训练目标来自固定答案；在线 RL 的关键区别是：**训练数据会由正在变化的策略持续生成**。因此探索、奖励漏洞、策略陈旧和训练系统吞吐都会直接改变最终能力。

## 2. 先厘清三组经常混用的概念

### 2.1 RLHF、RLAIF、RLVR：奖励从哪里来

- RLHF（Reinforcement Learning from Human Feedback）：奖励模型主要学习人类偏好；
- RLAIF（Reinforcement Learning from AI Feedback）：偏好或评分主要由另一模型依据规则生成；
- RLVR（Reinforcement Learning with Verifiable Rewards）：奖励来自可执行、可判定的验证器；
- Agent RL：奖励来自真实或模拟环境中的任务完成情况，也可能混合人类、AI 和规则反馈。

这些名称描述的是**反馈源和训练范式**，不是某个固定优化器。RLHF 可以用 PPO，也可以用 RLOO；RLVR 可以用 GRPO，也可以用 PPO。

### 2.2 PPO、RLOO、GRPO、GSPO：怎样估计并更新梯度

这些才是策略优化算法。它们主要在以下问题上不同：

- 是否训练 Value/Critic；
- Advantage 如何估计；
- 每个 Prompt 采样多少回答；
- Importance Ratio 按 Token 还是按 Sequence 定义；
- 是否 Clip、如何做 KL 约束；
- Loss 按 Token、Sequence 还是 Prompt 聚合。

### 2.3 Reward Model、Verifier、Judge：怎样给回答打分

- Reward Model：学习得到的标量模型，通常输入 Prompt 和回答后输出分数；
- Verifier：根据标准答案、代码测试或形式规则判定，通常更客观；
- LLM Judge：用大模型按 Rubric 评分，本质上仍是可能被利用的学习型评估器；
- Process Reward Model（PRM）：对中间步骤评分，而不是只看最终结果。

优化器再稳定，也不能弥补错误奖励。大模型 RL 的上限往往先由“奖励是否代表真实目标”决定。

## 3. 把语言生成写成强化学习问题

给定 Prompt (x)，模型生成回答 (y=(y_1,\ldots,y_T))。可以把生成过程视为 Token-level MDP：

```text
状态 s_t = Prompt + 已生成 Token y_<t
动作 a_t = 下一个 Token y_t
转移     = 把 y_t 追加到上下文
终止     = 生成 EOS 或达到长度上限
奖励     = 最终结果奖励，也可以包含中间步骤奖励
```

策略就是语言模型：

\[
\pi_\theta(y\mid x)=\prod_{t=1}^{T}
\pi_\theta(y_t\mid x,y_{<t})
\]

最简单的目标是最大化期望奖励：

\[
J(\theta)=\mathbb E_{x\sim D,\,y\sim\pi_\theta(\cdot\mid x)}
[R(x,y)]
\]

通用对齐场景还常加入相对冻结参考策略 \(\pi_{ref}\) 的 KL 约束：

\[
J(\theta)=\mathbb E[R(x,y)]-
\beta\mathbb E\left[
\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}
\right]
\]

KL 的作用不是让策略完全不变，而是限制策略离开参考分布的速度，降低 Reward Model 在分布外被利用的风险。

## 4. 三个模型不要混淆：Policy、Old Policy、Reference

在线 RL 实现中经常同时出现：

| 名称 | 是否变化 | 作用 |
|---|---|---|
| \(\pi_\theta\) Policy / Actor | 每次优化更新 | 当前要训练的生成模型 |
| \(\pi_{old}\) Old / Behavior Policy | 每轮 Rollout 后同步 | 记录样本由哪个版本生成，用于 Importance Ratio |
| \(\pi_{ref}\) Reference Policy | 通常冻结 | 提供 KL 锚点，防止策略漂移过远 |

\(\pi_{old}\) 和 \(\pi_{ref}\) 不是一回事。前者解决 On-policy 数据经过多步更新后变旧的问题；后者定义“不要偏离太远”的行为基准。某些 RLVR 配方会移除 \(\pi_{ref}\)，但仍需要记录或复现 \(\pi_{old}\) 的采样 Log Probability。

## 5. Reward：RL 系统真正优化的目标

### 5.1 偏好 Reward Model

给同一个 Prompt 的偏好回答 \(y_w\) 和非偏好回答 \(y_l\)，经典 Bradley–Terry 形式为：

\[
P(y_w\succ y_l\mid x)=
\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)
\]

Reward Model 最小化：

\[
L_{RM}=-\log\sigma\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right)
\]

训练完成后，Policy 生成新回答，Reward Model 输出标量分数。需要特别防范：

- 长度、格式、语气和引用数量等表面偏差；
- Reward Model 只在旧策略分布上可靠；
- 标注者分歧被压缩成单一标量；
- Policy 找到高奖励但低真实质量的捷径；
- Reward 分数持续上升，但人工胜率反而下降。

### 5.2 RLAIF 与 LLM-as-a-Judge

RLAIF 用原则、Rubric 或宪法让强模型生成偏好标签，再训练 Reward Model 或直接把 Judge 分数作为奖励。优点是便于扩量和覆盖复杂规则；风险是 Judge 的位置偏差、长度偏差、自我偏好和可被 Prompt Injection 操纵。

AI Feedback 不等于自动正确。高风险场景仍需人工校准、对抗样本和独立评估器。

### 5.3 RLVR

RLVR 适合有客观结果的任务：

- 数学：规范化后比较最终答案；
- 代码：编译、单元测试、静态检查和资源限制；
- 形式证明：由 Proof Assistant 检查；
- 结构化输出：JSON Schema、SQL 执行结果；
- 工具任务：环境状态是否达到目标。

最简单的二元奖励是：

\[
R(x,y)=\mathbb 1[\operatorname{verify}(x,y)=\text{pass}]
\]

Verifier 通常比学习型 Reward Model 更难欺骗，但解析器和测试本身仍可能有漏洞。只验证最终字符串，也可能奖励猜答案、硬编码或利用测试缺口。

### 5.4 Outcome Reward 与 Process Reward

Outcome Reward 只评价最终回答，便宜但信用分配粗糙；Process Reward 给中间步骤反馈，信号更密集，但标注或自动验证更困难。

PRM 的常见用途包括：

- RL 中给每一步增量奖励；
- 推理时对多条轨迹进行搜索或重排；
- 过滤 Reasoning SFT 数据；
- 发现答案正确但过程错误的轨迹。

过程监督也可能把某种书写风格误当成推理质量，必须用最终任务成功率校验。

### 5.5 多目标奖励

生产系统往往组合多个目标：

\[
R=w_{task}R_{task}+w_{safe}R_{safe}+w_{style}R_{style}
-w_{cost}C_{tokens}-w_{viol}C_{violation}
\]

固定线性加权简单，但会让一个高分维度补偿不可接受的安全失败。硬约束、安全 Gate 或分阶段训练有时比单一加权和更合适。

## 6. Policy Gradient：所有主流在线算法的共同底座

REINFORCE 的梯度估计为：

\[
\nabla_\theta J(\theta)
=\mathbb E\left[
(R-b)\sum_{t=1}^{T}
\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})
\right]
\]

其中 \(b\) 是不依赖当前动作的 Baseline，减去它不改变期望梯度，却能降低方差。定义 Advantage：

\[
A=R-b
\]

直观上：

- \(A>0\)：提高这条采样轨迹中动作的概率；
- \(A<0\)：降低它们的概率；
- \(|A|\) 越大：更新越强。

当奖励只在回答末尾出现时，同一个 Sequence 的所有 Token 往往共享同一个 Outcome Advantage。这种估计简单，却难以判断究竟是哪一步造成成功或失败。

## 7. PPO：经典 RLHF 的主力基线

### 7.1 核心目标

样本由 \(\pi_{old}\) 生成，当前 Policy 与旧 Policy 的 Token-level Importance Ratio 为：

\[
\rho_t(\theta)=
\frac{\pi_\theta(y_t\mid x,y_{<t})}
{\pi_{old}(y_t\mid x,y_{<t})}
\]

PPO-Clip 最大化：

\[
J_{PPO}=\mathbb E_t\left[
\min\left(
\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]
\]

Clip 限制一次更新把新策略推离采样策略太远，使一批 Rollout 可以安全地进行若干轮 Mini-batch 更新。

### 7.2 Critic 与 GAE

经典 PPO 训练 Value Model \(V_\psi(s_t)\) 预测从当前状态开始的回报。Temporal Difference 残差为：

\[
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
\]

Generalized Advantage Estimation（GAE）为：

\[
\hat A_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}
\]

完整 Loss 通常包含：

\[
L=-J_{PPO}+c_vL_{value}-c_eH(\pi_\theta)
\]

再配合 Reward 中的 Reference KL 惩罚。

### 7.3 经典 PPO-RLHF 需要哪些组件

```text
Prompt
  -> Actor 采样回答
  -> Reward Model 打分
  -> Reference Model 计算 KL
  -> Critic 估计每个 Token 的 Value
  -> GAE 计算 Advantage
  -> Actor / Critic 分别更新
```

因此一个经典系统可能同时维护 Actor、Critic、Reward、Reference 四个模型。即使部分模型共享 Backbone，其显存、通信和调度仍然昂贵。

### 7.4 PPO 的优势与局限

优势：

- 成熟、通用，适合有中间奖励的多步环境；
- Critic 可以为不同 Token 或状态分配不同 Advantage；
- 对连续多轮 Agent 轨迹比单一 Outcome Baseline 更自然；
- Clip、KL、Value Loss 和 Entropy 提供多种稳定控制手段。

局限：

- Critic 占用大量显存，还可能比 Policy 更难训练；
- 超参数相互耦合，Value 失准会污染 Actor；
- 同一批 Rollout 多轮更新会产生 Off-policy 偏差；
- 实现中 Token Mask、KL、GAE 和 Padding 很容易错位；
- 对只有终局标量奖励的单轮 LLM 任务，系统可能过于复杂。

PPO 仍是通用 RLHF 和 Agent RL 的重要基线，但已不再是所有 LLM RL 场景的默认唯一选择。

## 8. REINFORCE 家族：去掉 Critic 的简单路线

### 8.1 Vanilla REINFORCE

直接使用完整回报减去 Batch Mean、移动平均或其他 Baseline。它显存低、实现清晰，但单样本方差很大，对 Reward Scale 和 Batch Size 敏感。

### 8.2 RLOO：REINFORCE Leave-One-Out

对同一个 Prompt 采样 \(G\) 个回答。第 \(i\) 条回答用其余回答的平均奖励作为 Baseline：

\[
b_i=\frac{1}{G-1}\sum_{j\ne i}R_j,
\qquad A_i=R_i-b_i
\]

它有几个实用特点：

- 不训练 Critic；
- 同一个 Prompt 内相对比较，能消除部分题目难度差异；
- Leave-one-out 避免把样本自己的 Reward 放进其 Baseline；
- 算法比 PPO 简单，但 Rollout 成本随 \(G\) 增加。

RLOO 与 GRPO 都使用同 Prompt 多样本做相对优势，区别是 RLOO 更接近直接 REINFORCE，通常不依赖组内标准差归一化和 PPO 式 Token Clip。

### 8.3 ReMax

ReMax 为每个 Prompt 用 Greedy Decoding 生成一条回答，以其 Reward 作为 Baseline：

\[
A=R(y_{sample})-R(y_{greedy})
\]

它同样不需要 Critic，Baseline 与当前 Prompt 强相关。代价是每个 Prompt 额外做一次 Greedy Rollout，而且 Greedy Baseline 的质量和稳定性会影响方差。

### 8.4 REINFORCE++

REINFORCE++ 是把 Advantage 归一化、PPO 风格 Clip、KL 和训练稳定技巧组合到 Critic-free REINFORCE 中的一类方案。它的意义更接近“简化 PPO 的工程配方”，而不是改变 Policy Gradient 的基本原理。

## 9. GRPO：推理 RL 中最有代表性的 Critic-free 算法

Group Relative Policy Optimization（GRPO）由 DeepSeekMath 系统化提出。对每个 Prompt 采样 \(G\) 个回答，组内归一化奖励：

\[
\hat A_i=
\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}
{\operatorname{std}(R_1,\ldots,R_G)+\varepsilon}
\]

然后将同一个回答的 Advantage 分配给该回答的 Token，并使用 PPO 风格的 Importance Ratio 和 Clip：

\[
J_{GRPO}=\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}
\sum_t
\min\left(
\rho_{i,t}\hat A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i
\right)
\]

原始 GRPO 还可以加入对 Reference Policy 的 KL 惩罚。

### 9.1 为什么 GRPO 适合数学与代码

- Verifier 能给每条完整回答一个稳定 Outcome Reward；
- 同题多采样天然形成“谁比同伴更好”的 Baseline；
- 不需要单独训练与 Policy 同量级的 Critic；
- 多样本探索可以发现 SFT 数据中没有的新解法；
- 组内相对奖励对不同题目的绝对分数尺度较稳健。

### 9.2 GRPO 的关键代价

省掉 Critic 不等于省掉计算。若每题采样 \(G=8\) 或 \(16\) 条长 CoT，Rollout Token 往往成为最大成本。训练效率主要由：

\[
\text{Prompt 数}\times G\times\text{平均生成长度}
\]

决定，而不是只看训练 Batch Size。

### 9.3 零方差组

二元正确性奖励下，若一组回答全错或全对：

\[
R_1=R_2=\cdots=R_G
\]

则中心化 Advantage 全为 0，这个 Prompt 不产生有效 Policy Gradient。全错通常表示题太难或探索不足；全对表示题太简单。应持续记录：

- `all_zero_group_ratio`；
- `all_one_group_ratio`；
- `nonzero_advantage_ratio`；
- 每个难度桶的有效梯度率。

### 9.4 GRPO 不是“让模型自动学会推理”的魔法

RL 只能提高已能采样到的高奖励轨迹概率。如果 Base Policy 在给定采样预算下从不产生正确答案，二元 Reward 没有正样本可放大。可训练性依赖：

- Base Model 的领域能力；
- Prompt Template 是否能激活回答行为；
- 温度、Top-p 和最大长度是否允许探索；
- 题目难度是否处于当前策略的学习边界；
- Verifier 是否能识别真实成功。

## 10. Dr.GRPO：修正长度和难度计权偏差

Dr.GRPO 的分析指出，常见 GRPO 目标可能包含两种隐含重加权：

1. 每条回答除以自身长度 \(|y_i|\)，使不同长度回答的总梯度计权发生变化；
2. 每题 Advantage 除以组内 Reward 标准差，使不同 Reward 方差的题目权重不同。

它提出：

- Advantage 只做组内中心化，不除以组内标准差；
- Token Loss 的分母使用固定生成预算等常数，不按每条回答的实际长度变化。

\[
\hat A_i=R_i-\operatorname{mean}(R_1,\ldots,R_G)
\]

其目标是避免短正确回答被异常放大、长错误回答被相对少罚，以及不同难度题目被 Reward 标准差重新加权。

这里没有脱离任务的唯一 Reduction：按样本平均、按有效 Token 平均和按固定长度归一化，对应不同优化目标。工程上必须显式写出分母、做长度分桶消融，不能只记录一个 `loss.mean()`。

## 11. DAPO：面向长 CoT 的 GRPO 系统配方

DAPO 全称 Decoupled Clip and Dynamic Sampling Policy Optimization。它建立在 Group-based、PPO-Clip 风格优化上，公开配方包含四个关键技巧。

### 11.1 Clip-Higher

将对称 Clip 改为不对称范围：

\[
\operatorname{clip}(\rho,1-\epsilon_{low},1+\epsilon_{high}),
\qquad \epsilon_{high}>\epsilon_{low}
\]

较高的上界给原本低概率、但获得正 Advantage 的探索 Token 更大提升空间，用于缓解 Entropy 过早坍塌。

### 11.2 Dynamic Sampling

过滤全对和全错的零梯度组，继续采样，直到得到足够多具有 Reward 差异的 Prompt Group：

\[
0<\#\{i:R_i=1\}<G
\]

它能减少无效更新，但会把训练分布动态推向“当前模型恰好有时会、有时不会”的题目。必须记录原始 Prompt 分布和重采样后的分布，避免困难或简单子域被无意丢失。

### 11.3 Token-level Policy Gradient Loss

DAPO 将一个 Batch 内所有有效生成 Token 汇总后统一平均：

\[
J=\frac{1}{\sum_i|y_i|}
\sum_i\sum_t L_{i,t}
\]

而不是先对每条回答按长度平均、再对回答平均。这样长 CoT 中每个 Token 不会因为 Sequence 较长而被自动降权。代价是长回答在 Batch 中自然贡献更多梯度，因此仍需单独监控长度偏置。

### 11.4 Overlong Reward Shaping

超过生成上限的回答可能只是被截断，直接给硬负奖励会把“尚未写完的正确思路”和“无意义重复”混为一谈。DAPO 使用：

- 过滤被截断样本的 Loss；或
- 在接近上限的缓冲区内逐渐增加 Soft Penalty。

这比突然从正常 Reward 跳到惩罚更平滑，也要求日志区分自然 EOS 和 Max-length Truncation。

### 11.5 怎样理解 DAPO

DAPO 不是与 Policy Gradient 完全不同的新理论底座，更适合看成一套针对大规模长 CoT RL 的算法—系统联合配方：它同时修改了 Clip、采样分布、Loss Reduction 和截断奖励。

## 12. GSPO：从 Token-level Clip 转向 Sequence-level Clip

Group Sequence Policy Optimization（GSPO）认为：奖励授予整条 Sequence 时，Off-policy Correction 和 Clip 也应以 Sequence 为单位。

它定义长度归一化的 Sequence-level Importance Ratio：

\[
s_i(\theta)=
\left(
\frac{\pi_\theta(y_i\mid x)}
{\pi_{old}(y_i\mid x)}
\right)^{1/|y_i|}
=\exp\left(
\frac{1}{|y_i|}\sum_t
\log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}
{\pi_{old}(y_{i,t}\mid x,y_{i,<t})}
\right)
\]

再对整条回答做 Clip：

\[
J_{GSPO}=\frac{1}{G}\sum_i
\min\left(
s_i\hat A_i,
\operatorname{clip}(s_i,1-\epsilon,1+\epsilon)\hat A_i
\right)
\]

### 12.1 GSPO 解决什么问题

- Reward 和优化单元都落在 Sequence 级；
- 避免少数 Token 的 Ratio 异常决定整条轨迹的 Clip 状态；
- 对 Rollout Engine 与 Training Engine 的 Log Probability 数值差异更稳健；
- 公开结果强调其对 MoE RL 训练稳定性的帮助。

### 12.2 使用时的注意事项

- GSPO 的 \(\epsilon\) 与 Token-level PPO/GRPO 不在同一数值尺度，不能照搬；
- Sequence Ratio 做了长度归一化，仍需检查不同长度的实际梯度权重；
- 只有终局 Sequence Reward 时最自然；多轮、分步奖励需要更细的 GSPO-token 或其他 Credit Assignment；
- 它是较新的公开方案，应与成熟 GRPO/PPO 基线在自己的模型和系统上比较。

## 13. 主流算法横向比较

| 算法 | Advantage / Baseline | Critic | 每题多采样 | Clip 单元 | 主要优势 | 主要代价 |
|---|---|---:|---:|---|---|---|
| PPO | Value + GAE | 是 | 可选 | Token | 通用、适合分步奖励和 Agent | 四模型系统复杂，显存高 |
| REINFORCE | Batch/移动平均 Baseline | 否 | 非必需 | 通常无 | 最简单、低显存 | 方差大、数据效率敏感 |
| RLOO | Leave-one-out 组内均值 | 否 | 是 | 通常无 | Critic-free，估计清晰 | 多 Rollout 成本 |
| ReMax | Greedy 回答 Reward | 否 | 采样 + Greedy | 通常无 | Prompt-specific Baseline | 额外 Greedy 生成 |
| GRPO | 组内均值/标准差 | 否 | 是 | Token | 推理 RL 成熟、实现较普及 | 零梯度组、长度/难度计权争议 |
| Dr.GRPO | 组内中心化 | 否 | 是 | Token | 修正特定归一化偏差 | 需重新校准梯度尺度 |
| DAPO | GRPO 风格 | 否 | 是 | Token、不对称 | 长 CoT 公开系统配方完整 | 采样与 Reduction 改动耦合 |
| GSPO | 组内相对 Advantage | 否 | 是 | Sequence | Reward/Clip 单元一致，利于 MoE 稳定 | 较新，超参尺度不同 |

这张表是典型实现，不是强制定义。KL、Entropy、Reference Model 和多轮 Reward 可以按配方组合。

## 14. DPO 和 RL 的关系：相邻，但不等价

DPO 使用固定偏好对 \((x,y_w,y_l)\)，定义：

\[
L_{DPO}=-\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\right]
\right)
\]

DPO 与 KL-Regularized RL 有紧密数学联系，但典型 DPO 训练：

- 不让当前 Policy 在线采样；
- 不在训练环中运行环境；
- 不单独训练标量 Reward Model；
- 只学习固定数据覆盖到的 chosen/rejected 区域。

因此 DPO 更准确的名称是 **Offline Direct Preference Optimization**，而不是严格意义上的 Online RL。IPO、KTO、ORPO 等也通常归入直接偏好优化家族。

### 14.1 什么时候先用 DPO

- 已有高质量偏好对，但没有稳定在线 Rollout 系统；
- 主目标是风格、安全、帮助性等行为对齐；
- 计算资源有限；
- 希望先建立一个稳定基线，再评估在线 RL 的增量。

### 14.2 什么时候在线 RL 更有价值

- 可以用 Verifier 低成本评价大量新样本；
- 需要当前 Policy 主动探索新的正确轨迹；
- 固定偏好数据与当前模型分布不匹配；
- 任务是多步工具交互，必须根据环境结果继续决策。

## 15. On-policy、Off-policy 与策略陈旧

严格 On-policy 要求样本来自当前策略。但实际系统中：

```text
Rollout Workers 用版本 v 生成
 -> Reward Workers 验证
 -> Trainer 已经更新到 v+k
 -> 旧样本才进入训练
```

这会产生 Policy Staleness。常见控制手段包括：

- 每轮冻结 \(\pi_{old}\)，限制每批数据的更新 Epoch；
- 用 Importance Ratio 和 Clip 限制偏差；
- 丢弃超过版本差阈值的 Rollout；
- 设置 Replay Buffer 最大年龄；
- 同步或流水线系统中记录每条样本的 Policy Version；
- 监控 \(\pi_\theta\) 与采样 Log Probability 的差值。

“用了 Importance Sampling”不代表任意旧数据都可安全复用。长 Sequence 的分布比逐 Token Ratio 显示的更快漂移。

## 16. 一个标准在线 RL 训练循环

```text
初始化 Policy（Base / SFT / DPO Checkpoint）
初始化可选 Reference、Reward、Critic

for rollout_step:
    1. 从 Prompt Pool 采样一批任务
    2. 用冻结的 Old Policy 按温度生成 G 条回答
    3. 保存 Token、Mask、Old LogProb、Policy Version
    4. 运行 Reward Model / Verifier / Environment
    5. 组合 Reward，并构造 Advantage
    6. 对有效轨迹做若干 Mini-batch Policy Update
    7. 更新或同步 Old Policy
    8. 跑独立 Eval，检查 Reward Hacking 和能力回退
```

每条 Rollout 至少应保留：

- 原始 Prompt、Chat Template 版本；
- 生成 Token ID，而不只保存重新 Tokenize 的文本；
- Prompt/Response Mask、EOS/Truncation 状态；
- Sampling 参数和随机种子；
- Old Policy Token Log Probability；
- 各 Reward 分量、Verifier 版本和失败原因；
- Policy Checkpoint/Version；
- 工具调用、环境观察和终止原因。

## 17. Rollout 分布决定模型能学到什么

### 17.1 温度不是普通推理参数

温度过低时，同题样本近乎相同，组内 Reward 无差异；过高时，大量乱码和无意义轨迹消耗 Verifier 与 Token。Top-p、Top-k、Min-p 也会改变 Policy 实际探索空间。

训练时必须保存 Sampling Config，并确保 Old LogProb 对应**真正参与采样的分布**。如果 Rollout 做了 Top-k 截断，Trainer 却按未截断 Softmax 计算 LogProb，Importance Ratio 语义已经不一致。

### 17.2 Prompt Curriculum

有效训练题通常位于策略的能力边界：既不是全对，也不是全错。可按成功率动态组织课程：

- 过易题降低采样频率，但保留少量防遗忘；
- 边界题提高权重；
- 过难题先经过 SFT、分解或更短子任务；
- 持续加入未见模板和对抗变体，防止记忆答案格式。

动态课程会改变目标分布，最终 Eval 必须按固定真实分布报告。

### 17.3 Group Size

更大的 \(G\) 能更可靠地估计相对优势并提高找到正确轨迹的概率，但增加生成成本。应联合比较：

- 固定总生成 Token 下的 Prompt 数与 Group Size；
- `pass@G` 与训练后 `pass@1`；
- 每个非零梯度 Group 的实际计算成本；
- 大 Group 是否只是重复同一模式。

## 18. KL、Entropy 与探索—稳定平衡

### 18.1 KL 的两种常见实现

可以把 Sample-based KL 估计作为 Reward Penalty：

\[
r_t^{KL}=-\beta\left(
\log\pi_\theta(y_t\mid s_t)-
\log\pi_{ref}(y_t\mid s_t)
\right)
\]

也可以作为训练 Loss 的独立正则项。两者的梯度路径和缩放并不完全相同，配置文件必须说明具体实现。

### 18.2 什么时候可以减弱或移除 KL

规则 Verifier 不像 Reward Model 那样因 Policy 分布漂移而直接失准，因此部分 Reasoning RL 配方移除了 Reference KL，节省一次 Reference Forward。但仍需评估：

- 通用语言和指令能力是否遗忘；
- 输出是否越来越模板化或冗长；
- 安全行为是否退化；
- Reward 是否存在可利用漏洞；
- 模型是否只对狭窄 Benchmark 过拟合。

“Reward 可验证”不自动意味着“无需任何行为锚点”。

### 18.3 Entropy

Entropy 太低意味着采样快速确定化、探索枯竭；太高则可能出现乱码、重复和随机试错。应联合观察：

- Policy Entropy；
- Unique Response / n-gram Diversity；
- Reward、准确率和输出长度；
- Up/Down Clip Fraction；
- 低概率 Token 获得正 Advantage 的比例。

只追求 Entropy 增长或下降都不合理。

## 19. 长度偏置、截断和 Loss Reduction

长 CoT RL 中，很多“算法收益”其实来自不同计权方式。

### 19.1 三种常见 Reduction

按回答等权：

\[
L_{seq}=\frac{1}{B}\sum_i\frac{1}{|y_i|}\sum_t\ell_{i,t}
\]

按有效 Token 等权：

\[
L_{token}=\frac{\sum_i\sum_t\ell_{i,t}}{\sum_i|y_i|}
\]

按固定生成预算归一化：

\[
L_{fixed}=\frac{1}{BC}\sum_i\sum_t\ell_{i,t}
\]

它们分别让“每条回答”“每个 Token”或“固定预算中的位置”权重更接近一致。没有记录 Reduction，就无法复现实验。

### 19.2 长答案不等于强推理

长度增长可能来自：

- 学会展开有用推导；
- 反复自我检查；
- 错误轨迹受到较弱惩罚；
- 终止 Token 概率下降；
- Reward 或 Parser 偏好包含更多候选答案的文本；
- Max-length 样本处理错误。

至少应报告正确/错误回答各自的长度分布，以及 Accuracy per Token，而不只报告平均长度。

### 19.3 截断语义

`EOS` 和 `max_tokens` 是两种完全不同的终止：

- 自然 EOS：Policy 主动结束；
- Max-length：系统强制截断，最终答案可能尚未出现；
- 环境终止：工具成功、失败或超时；
- Parser 终止：得到合法最终结构。

训练数据必须单独记录这些状态。不要给被截断回答伪造 EOS，也不要默认把所有截断都当逻辑错误。

## 20. Reward Hacking：训练成功、目标失败

常见 Reward Hacking 包括：

- 数学 Parser 只读取最后一个数字，模型输出多个互相矛盾答案；
- 代码只通过公开样例，没有覆盖隐藏边界；
- Judge 偏好长、礼貌、带标题的回答；
- Tool Agent 伪造“执行成功”文本而未真正调用工具；
- 模型把 Reference Answer 或测试信息从 Prompt 泄漏中复制出来；
- 用特殊字符、NaN 或异常 JSON 绕过评分器；
- Reward Model 分数上升但真实人工偏好下降。

防护手段：

1. Verifier 在隔离环境中执行，模型文本不能直接决定成功状态；
2. 隐藏测试、随机测试和对抗测试组合；
3. Reward 分量和真实 Eval 指标分开记录；
4. 定期人工阅读最高 Reward、最大 Reward 增长和最长样本；
5. 使用多个独立 Judge，并用人工样本校准；
6. 对 Parser 做 Fuzz Test；
7. 保留未见任务与未见模板 Eval；
8. Reward 发生异常跃升时先审计轨迹，而不是继续扩训练。

## 21. Cold Start、SFT 和纯 RL

### 21.1 从 Base Model 直接 RL

R1-Zero 类训练表明，在合适 Base Model、模板、任务和采样预算下，可以直接通过 RL 提升可验证推理。但它要求 Base Policy 已能偶尔探索到正确轨迹，否则没有学习信号。

### 21.2 Reasoning Cold Start

先用少量高质量长 CoT、格式和工具轨迹做 SFT，通常可以：

- 建立稳定答案格式；
- 减少语言混杂和不可读输出；
- 提高初始正 Reward 率；
- 缩短 RL 的无效探索阶段。

代价是可能限制探索风格，或让模型过度模仿 Teacher。Cold Start 数据应尽量多样，并用 RL 验证能否超越示范分布。

### 21.3 多阶段配方

一个稳健路线常是：

```text
Base / Mid-training
 -> 通用 SFT
 -> Reasoning Cold Start
 -> RLVR 扩展推理能力
 -> 通用偏好 / 安全 RL
 -> 混合能力恢复与蒸馏
```

每进入下一阶段，都要保留前一阶段的固定回归集，防止“数学变强、对话变坏”被平均分掩盖。

## 22. Reasoning RL 真正可能提升什么

Outcome RL 不直接告诉模型正确推理步骤，而是通过重复采样和强化成功轨迹，改变生成分布。它可能提升：

- 在多个候选思路间搜索；
- 分解问题和回溯；
- 自检后改正答案；
- 为困难题分配更多 Test-time Compute；
- 选择更容易被 Verifier 确认的表达。

也可能只学到：

- 更长的固定推理模板；
- Benchmark 特定答案格式；
- 对训练题型的模式记忆；
- 无效的“等等，我再检查”文本。

所以应通过难度外推、反事实题、扰动题、过程检查和 Accuracy-vs-Tokens 曲线判断是真能力还是表面行为。

## 23. Agent 与 Tool-use RL

Agent 轨迹不再是单次回答：

```text
User
 -> Assistant Tool Call
 -> Environment Observation
 -> Assistant Tool Call
 -> ...
 -> Final Answer / Success / Failure
```

### 23.1 状态与动作

状态包含对话历史、工具结果、文件或网页状态；动作可能是自然语言 Token，也可能是结构化 Tool Call。Environment Transition 不再完全由字符串拼接决定。

### 23.2 Reward 设计

可以组合：

- 最终任务是否完成；
- 工具参数是否合法；
- 是否选择了正确工具；
- 调用次数、Token 和时间成本；
- 是否泄漏数据或越权；
- 中间状态是否接近目标。

不要只奖励“最终回答声称完成”，必须从环境状态判定真实完成。

### 23.3 Credit Assignment

终局 Reward 直接广播给所有 Tool Call，方差很高。可选方法包括：

- PPO + Critic/GAE；
- 每轮或每个工具步骤的 Process Reward；
- Outcome-based Group Sampling；
- 对失败轨迹定位第一个不可恢复错误；
- 将规划、工具选择和最终回答拆成不同 Head 或阶段。

### 23.4 环境必须可复现

真实网页、搜索或数据库会变化。训练系统应记录环境快照、工具版本、时间和返回值；对写操作使用 Sandbox；对不可逆外部动作使用模拟器或显式审批。

## 24. 大规模 RL 的系统架构

在线 RL 至少有四类工作负载：

```text
Prompt/Data Workers
       ↓
Rollout Workers（高吞吐自回归生成）
       ↓
Reward/Verifier/Environment Workers
       ↓
Trainer Workers（前向、反向、优化）
       ↓
Weight Sync -> Rollout Workers
```

### 24.1 Rollout 与 Training 的计算形态不同

- Rollout 重视 KV Cache、Continuous Batching 和 Decode 吞吐；
- Training 重视大 Token Batch、Activation Checkpointing 和分布式反向；
- Reward Model 是另一套推理服务；
- 代码/工具 Verifier 可能受 CPU、容器和 I/O 限制。

因此 RL 吞吐通常不是“Trainer 每秒多少 Token”，而是完整闭环每小时产生多少**有效、有 Reward 差异的训练 Token**。

### 24.2 同步与异步

同步系统样本更新鲜、算法更清晰，但会被最长回答和最慢 Verifier 拖住；异步系统设备利用率高，却会增加 Policy Staleness 和调试难度。

### 24.3 权重同步

训练框架与推理引擎可能使用不同并行切分和权重格式。同步时要验证：

- 所有 Layer 都更新，LoRA/MoE 权重未遗漏；
- Tokenizer、Chat Template 和特殊 Token 一致；
- 权重版本与 Rollout Metadata 对应；
- 同一固定输入的 LogProb 差异处于容忍范围；
- BF16/FP8、量化和 Kernel 差异没有改变采样语义。

## 25. MoE 模型做 RL 的特殊问题

MoE RL 比 Dense 更容易不稳定：

- Policy 更新改变 Token 分布，也改变 Expert Routing；
- Rollout 与 Training Kernel 的微小 Logit 差异可能触发不同 Top-k Expert；
- 少数高 Reward 模式可能集中到少数 Expert；
- 小 Batch 下每个 Expert 的有效 Token 更少；
- Router Auxiliary Loss 可能与 Policy Gradient 尺度不匹配。

应监控每层：

- Expert Token Count、负载离散度和最大/最小比；
- Router Entropy、Dropped Token；
- 不同 Reward/领域/语言的路由分布；
- Router 与 Expert 的 Gradient Norm；
- Rollout/Training LogProb Difference；
- Policy Clip Fraction 与路由切换的相关性。

可比较冻结 Router、降低 Router LR、增大有效 Token Batch、Sequence-level GSPO，以及保留通用数据 Replay。不能只看总 Reward 判断 Router 是否健康。

## 26. 训练稳定性指标

### 26.1 所有在线 RL 都应记录

- 各 Reward 分量的 Mean/Std/Quantile；
- 独立真实 Eval，而不只是训练 Reward；
- Prompt、Response、正确/错误回答长度；
- EOS、Truncation、Invalid Format 比例；
- Policy Entropy、KL to Reference；
- Old/New LogProb Difference 和 Importance Ratio 分布；
- Clip Fraction，最好区分 Upper/Lower；
- Advantage Mean/Std 和非零比例；
- Gradient Norm、参数更新范数和 NaN/Inf；
- Rollout Tokens/s、Train Tokens/s、Verifier Latency；
- 每个有效梯度 Group 的 Token 成本。

### 26.2 PPO 额外记录

- Value Loss；
- Explained Variance；
- Return 与 Value 的尺度；
- GAE Mean/Std；
- Actor/Critic 学习率和更新比；
- Value Clip Fraction。

### 26.3 Group-based 算法额外记录

- Group Reward Std；
- All-correct / All-wrong Group Ratio；
- 每个 Prompt 的有效样本数；
- Dynamic Sampling 重试次数；
- 不同 Group Size 的 Reward Diversity；
- 组内重复率和答案模式多样性。

## 27. RL 应该怎样评估

### 27.1 能力指标

- 数学/代码：pass@1、pass@k、严格准确率和编译/执行成功率；
- 通用对话：Pairwise Human/Judge Win Rate，并校准位置与长度偏差；
- Agent：真实任务成功率、平均步骤、恢复率和成本；
- 安全：Harmful Compliance、Jailbreak Success、False Refusal；
- 格式：Schema Validity、Tool Call Accuracy 和停止正确率。

### 27.2 推理成本

Reasoning Model 的准确率必须和输出 Token 一起报告：

\[
\text{Token Efficiency}=
\frac{\text{Solved Tasks}}{\text{Generated Tokens}}
\]

还应画出不同最大 Token、Temperature 和采样数下的 Accuracy–Cost 曲线。训练后靠输出数倍 Token 换来的小幅提升，未必适合产品部署。

### 27.3 泛化与污染

- 开发集和最终测试集来源隔离；
- 与 RL Prompt Pool 做精确和近似去重；
- 使用未见题型、模板和语言；
- 训练 Verifier 与评估 Verifier 分离；
- 代码任务使用隐藏测试；
- 对公开 Benchmark 检查答案泄漏和模板记忆。

### 27.4 Checkpoint 选择

不要按训练 Reward 最高点自动选模型。应按预先定义的多目标 Scorecard，在能力、安全、通用保持和成本之间选择，并保留 Pareto 前沿 Checkpoint。

## 28. 常见失败模式与定位

| 症状 | 常见原因 | 优先排查 |
|---|---|---|
| Reward 上升、真实准确率不升 | Reward Hacking 或训练/评估 Parser 不同 | 最高 Reward 轨迹、隐藏测试 |
| GRPO Loss 长期接近 0 | Group 全对/全错或 Advantage Mask 错 | Group Reward Std、有效梯度率 |
| 输出越来越长 | Reduction 偏置、EOS 受罚、截断策略错误 | 正误样本分开看长度、EOS Logit |
| Entropy 快速坍塌 | Clip 上界过紧、采样太低温、题目过窄 | Upper Clip、Diversity、温度 |
| Entropy 过高且出现乱码 | Reward 太稀疏、过度探索、KL 太弱 | Reward 密度、KL、采样分布 |
| PPO Value Loss 爆炸 | Reward Scale 漂移、Mask/Bootstrap 错 | Return、GAE、Terminal Mask |
| KL 突然跳升 | LR/更新 Epoch 过大、陈旧 Rollout | Policy Version、Ratio、Clip Fraction |
| Actor 学不会但 Verifier 正常 | Base Policy 从不采到正样本 | pass@G、课程、Cold Start |
| 训练快、效果不可复现 | Sampling Seed/Kernel/环境不确定 | Rollout Metadata、环境快照 |
| Tool Agent 声称成功但任务没完成 | Reward 读取模型文本而非环境状态 | Success 判定链路 |
| MoE 少数 Expert 过载 | Router 漂移或高 Reward 模式单一 | 分层 Expert/Reward 路由统计 |
| 线上行为回退 | 只优化狭窄 RLVR，无通用/安全锚点 | 综合回归集、Replay/KL |

## 29. 怎样选择算法

### 29.1 通用帮助性、安全和风格对齐

优先建立强 SFT + DPO 基线。若有可靠 Reward Model 且需要在线探索，可比较 RLOO 与 PPO：

- 单轮终局 Reward、资源敏感：先试 RLOO；
- 有中间奖励、长多轮决策：PPO/Critic 更有价值；
- Reward Model 容易分布外失准：保留 Reference KL 和人工回归。

### 29.2 数学、代码和形式推理

优先使用 RLVR：

- 需要成熟、易获得实现：GRPO；
- 关注长度/难度归一化偏差：加入 Dr.GRPO 消融；
- 长 CoT、大规模训练：评估 DAPO 的四项配方；
- MoE 或 Rollout/Training LogProb 不一致明显：比较 GSPO。

不要同时改变 Base Model、数据、Verifier、Clip 和 Reduction 后宣称某个算法获益。

### 29.3 Tool-use 和 Agent

- 短、可重置、只有终局成败：Group-based 方法可先建立基线；
- 长时序、中间反馈明确：PPO + Critic/GAE 更自然；
- 环境昂贵：先做离线轨迹 SFT/DPO，再用少量在线 RL；
- 环境会变化：优先解决可复现与安全执行，再讨论优化器。

### 29.4 小模型或有限算力

建议顺序：

1. 高质量 SFT；
2. 固定偏好数据上的 DPO；
3. 小规模、严格可验证任务上的 RLOO/GRPO；
4. 确认 Rollout 确实带来增益后再扩大 Group、长度和系统复杂度；
5. 最后才考虑 Critic、多 Reward Model 和全异步集群。

## 30. 一套可靠的 RL 实验顺序

1. 冻结 Base/SFT Checkpoint、Tokenizer、Template 和 Eval；
2. 单元测试 Reward/Verifier，先用人工构造的正确与错误答案；
3. 对 Base Policy 跑 pass@G，确认能采到正 Reward；
4. 用极小 Prompt 集过拟合，检查 Reward、Mask、LogProb 和梯度方向；
5. 建立 Vanilla REINFORCE/RLOO 或 GRPO 最小基线；
6. 固定总 Rollout Token，比较 Group Size、温度和题目课程；
7. 分别消融 KL、Clip、Advantage Normalization 和 Loss Reduction；
8. 再加入 Dynamic Sampling、Overlong Shaping 等组合技巧；
9. 同时报告训练 Reward、独立能力、安全和 Token 成本；
10. 人工审计高 Reward、低 Reward、最长和 Reward 跃升样本；
11. 做多随机种子和中间 Checkpoint 评估；
12. 算法稳定后再扩大模型、最大长度和异步并行规模。

最重要的原则是：每轮实验尽量只回答一个问题。RL 闭环会放大任何数据、奖励和实现差异。

## 31. 业界公开路线怎样演进

### 31.1 InstructGPT：SFT + Reward Model + PPO

InstructGPT 奠定了经典三阶段 RLHF：人工示范做 SFT、成对偏好训练 Reward Model、PPO 在 KL 约束下优化 Policy。它仍是理解通用对齐 RL 的基础范式。

### 31.2 Constitutional AI：RLAIF

Constitutional AI 用一组原则指导模型生成批评、修订和 AI 偏好，再进行监督学习和 RL。它展示了反馈可以由 AI 扩展，但人类仍需定义原则并评估偏差。

### 31.3 DeepSeekMath：GRPO

DeepSeekMath 用 Group Relative Advantage 替代独立 Critic，降低 PPO 的内存和系统复杂度，并把 GRPO 推向数学推理训练。

### 31.4 Tülu 3：SFT + DPO + RLVR

Tülu 3 把 SFT、DPO 和可验证奖励 RL 放在同一开放后训练路线中，说明离线偏好优化与在线 RLVR 可以是连续阶段，而不是互斥方案。

### 31.5 DeepSeek-R1：大规模推理 RL 与多阶段训练

DeepSeek-R1-Zero 展示从 Base Model 直接做大规模 RL；DeepSeek-R1 则加入 Cold Start 和多阶段训练，改善可读性、语言混杂和通用行为。这两条路线说明“纯 RL 可行”和“生产配方适合多阶段”可以同时成立。

### 31.6 DAPO：公开长 CoT RL 系统细节

DAPO 把不对称 Clip、动态采样、Token-level Reduction 和超长样本处理组合成可复现配方，强调推理 RL 的效果来自算法与系统细节共同作用。

### 31.7 GSPO：Sequence-level 优化与 MoE 稳定性

GSPO 将 Importance Ratio、Clip 和 Reward 对齐到 Sequence 粒度，并公开强调其在 MoE RL 训练中的稳定性。它代表了从“照搬通用 Token-level PPO”转向“按语言模型生成结构重新设计优化单元”的趋势。

## 32. miniLLM 当前实现与 RL 演进建议

### 32.1 当前已经具备什么

miniLLM 当前实现了标准 DPO：

- [`trainer/train_dpo.py`](../trainer/train_dpo.py) 同时运行可训练 Policy 和冻结 Reference；
- [`dataset/dpo_dataset.py`](../dataset/dpo_dataset.py) 构造 chosen/rejected Pair，并只对最终 Assistant 回答计分；
- [`eval/eval_dpo.py`](../eval/eval_dpo.py) 比较 DPO Policy 与 SFT Reference；
- [`dataset/rl/dpo.jsonl`](../dataset/rl/dpo.jsonl) 提供固定偏好对。

这属于离线偏好优化，还不是本文定义的在线 RL：当前没有 Policy Rollout、Reward/Verifier、Advantage 估计、Old Policy 版本或环境闭环。

### 32.2 现有 RL 数据文件怎样理解

按当前文件样例：

- [`dataset/rl/rlaif.jsonl`](../dataset/rl/rlaif.jsonl) 主要是多轮 `conversations`，最后 Assistant 为空；文件名本身不能让它成为 RLAIF，仍需明确 AI Feedback、候选回答和评分字段；
- [`dataset/rl/agent_rl.jsonl`](../dataset/rl/agent_rl.jsonl) 包含 `conversations` 与 `gt`，可以作为部分任务 Prompt/Verifier 原料，但需要把 Ground Truth、工具执行和成功判定转换成严格环境接口。

在数据语义没有逐类审计前，不应仅根据文件名直接送入 RL Trainer。

## 33. 最终结论

大模型 RL 的主线可以压缩成四层：

```text
反馈层：Human / AI / Verifier / Environment
奖励层：Outcome / Process / Multi-objective Reward
优化层：PPO / RLOO / GRPO / DAPO / GSPO
系统层：Rollout / Reward Service / Trainer / Weight Sync
```

算法选择的核心不是追逐最新缩写，而是匹配任务结构：

- 主观偏好需要可靠 Reward Model、KL 和人工校准；
- 可验证推理适合 Critic-free Group-based RL；
- 长 CoT 必须审计 Reduction、长度和截断；
- 多轮 Agent 需要真实环境状态和更精细的 Credit Assignment；
- MoE 还要处理路由漂移和推理—训练 LogProb 不一致；
- 任何训练 Reward 都必须由独立真实 Eval 约束。

PPO 提供了通用 Actor–Critic 基础，RLOO 和 GRPO 降低了 Critic 成本，Dr.GRPO 与 DAPO 暴露了 Loss Reduction 和采样细节的重要性，GSPO 进一步把优化单元提升到 Sequence 级。真正可靠的 RL 系统，最终取决于奖励正确、数据可探索、实现可验证和评估不自欺。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models](https://arxiv.org/abs/2310.10505)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740)
- [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models](https://arxiv.org/abs/2501.03262)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
- [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)

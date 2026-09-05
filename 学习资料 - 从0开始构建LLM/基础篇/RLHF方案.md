# RLHF：主流方案解析

> RLHF（Reinforcement Learning from Human Feedback）用人类偏好定义“什么回答更好”，再把反馈转成模型可优化的训练信号。今天的业务实践通常把 DPO、RLAIF、规则奖励和可验证奖励也放进广义 RLHF/对齐体系；它们共享反馈闭环，但并不都使用强化学习。

## 关键点

1. **SFT 学习“像标准答案”，RLHF 学习“多个可行答案中哪个更好”。** 二者通常前后衔接，而不是相互替代。
2. **经典 RLHF 是 Reward Model + PPO。** 它能在线探索当前模型的回答空间，但系统复杂、训练昂贵，也容易利用奖励漏洞。
3. **DPO 是最常见的低门槛偏好对齐基线。** 它直接使用 chosen/rejected 数据训练，不单独拟合 Reward Model，也不做在线 Rollout；严格说不是 RL。
4. **主观任务依赖高质量偏好，客观任务优先使用可验证奖励。** 代码测试、数学答案、工具执行结果通常比纯人工打分更稳定。
5. **RLAIF 能扩大反馈规模，但不能消除人类治理。** 人类仍要定义原则、校准 Judge、处理冲突并审核高风险边界。
6. **奖励定义就是产品定义。** 指标写得不完整，模型会优化代理指标，而不是自动理解真实业务目标。
7. **离线高分不代表线上安全。** 必须监控 Reward Hacking、长度偏好、迎合、过度拒答、能力回退和分布漂移。
8. **大多数企业不应从 PPO 起步。** 常见稳妥路线是 SFT → DPO；只有存在可靠 Grader、足够 Rollout 算力和成熟训练团队时，再进入 PPO/GRPO 等在线强化阶段。

---

## 1. RLHF 位于模型生命周期的哪里

典型后训练链路为：

```text
Base Model
  ↓ 高质量示范
SFT Model
  ↓ 人类 / AI / 规则 / 执行器反馈
Preference Optimization 或 Reinforcement Learning
  ↓
Aligned Model
  ↓ 红队、系统护栏与线上监控
Production System
```

RLHF 不能替代预训练或 SFT：

- 基座模型提供知识和通用能力；
- SFT 提供基本对话协议、任务格式和可读输出；
- 偏好/强化阶段提高有帮助、真实、安全、风格和推理成功率；
- 推理时的 RAG、权限、工具沙箱和内容安全层负责动态事实与系统性约束。

如果 SFT 模型连任务格式都不稳定，直接做 RL 会把大量算力用于探索无效输出，Reward Model 也可能在低质量候选中学习错误捷径。

## 2. 为什么只有 SFT 还不够

SFT 假设存在一个目标回答 \(y^*\)，并最大化它的概率：

\[
L_{SFT}=-\log \pi_\theta(y^*\mid x)
\]

但开放式业务任务通常有多个正确答案。标注员很容易判断：

```text
回答 A 比回答 B 更准确、更完整、更安全
```

却很难从空白开始写出完美答案。偏好学习将监督信号从“复现这一串 Token”变成“提高更优回答相对较差回答的概率”。

它特别适合优化：

- 帮助性、完整性、简洁度和语气；
- 拒答边界与安全政策；
- 多步推理、代码与工具执行成功率；
- 不同正确方案之间的业务偏好；
- 用户满意度、任务完成率等组合目标。

它不擅长凭空补充基座没有的知识，也不能让含糊政策自动变清晰。

## 3. 先澄清“狭义 RLHF”与“广义对齐”

| 方法 | 反馈数据 | 是否训练 Reward Model | 是否在线采样 | 是否属于 RL |
|---|---|---:|---:|---:|
| SFT | 标准回答 | 否 | 否 | 否 |
| DPO / IPO 等 | chosen vs rejected | 否 | 通常否 | 否 |
| KTO | desirable / undesirable | 否 | 通常否 | 否 |
| ORPO | 标准回答 + 负回答 | 否 | 否 | 否 |
| 经典 RLHF（PPO） | 偏好对 → Reward | 是 | 是 | 是 |
| RLAIF | AI 偏好或 AI Reward | 可选 | 可选 | 取决于优化器 |
| RLVR / RFT | 程序或规则分数 | 不一定 | 是 | 是 |
| GRPO | 标量 Reward | 不需要 Value Model | 是 | 是 |

业务沟通中经常把这些统称为 RLHF，但架构评审时必须说清具体数据、奖励和优化器，否则无法估算成本与风险。

## 4. 反馈信号有哪些

### 4.1 成对偏好

同一 Prompt 生成两个或多个候选，由标注者选择更好者：

```json
{
  "prompt": "解释为什么订单被取消",
  "chosen": "订单因库存校验失败而自动取消……",
  "rejected": "系统取消了，重新下单即可。"
}
```

优点是人类更容易做相对判断；缺点是只表达局部顺序，不直接说明差距多大，也可能受长度、位置和措辞影响。

### 4.2 二元反馈

只标记回答为 desirable/undesirable、点赞/点踩或通过/失败。采集简单，适合 KTO 类目标，但需要控制曝光偏差：用户看到的回答不是随机样本，沉默也不等于满意。

### 4.3 标量评分

例如 1～5 分、任务成功率、人工质量分。标量信息丰富，但不同标注员对量尺理解不一致，容易出现 3 分和 4 分边界漂移。

### 4.4 可验证奖励

由程序直接判断：

- 数学最终答案；
- 单元测试、编译和静态分析；
- SQL 在沙箱中的执行结果；
- JSON Schema 与业务规则；
- Tool Call 是否完成目标；
- 游戏、规划或模拟器得分。

可验证不等于目标完整。例如“测试通过”可能仍包含低质量代码，“订单创建成功”也可能用了错误权限。因此通常要与格式、安全、成本等奖励组合。

### 4.5 AI、规则与宪法反馈

强模型根据 Rubric 或一组原则比较候选、打分、批评并改写。它可以快速覆盖长尾，但会继承 Judge 偏差，也可能被候选回答中的提示注入攻击。

### 4.6 隐式产品反馈

点击、复制、重试、人工接管、任务完成、投诉等信号贴近业务，但存在严重混杂因素。界面位置、用户群、延迟、候选曝光和下游流程都会影响结果，不能未经因果分析就直接当偏好真值。

## 5. 方案一：经典 Reward Model + PPO

经典 InstructGPT 路线包含三步：

```text
人工示范 -> SFT
SFT 生成多个候选 -> 人工排序 -> Reward Model
当前 Policy 在线生成 -> Reward 打分 -> PPO 更新
```

### 5.1 Reward Model

对同一 Prompt \(x\) 的偏好回答 \(y_w\) 和非偏好回答 \(y_l\)，Reward Model 输出标量 \(r_\phi(x,y)\)，使用 Bradley–Terry 形式训练：

\[
P(y_w\succ y_l\mid x)
=\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)
\]

\[
L_{RM}=-\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)
\]

Reward Model 学到的是偏好数据上的代理函数，不是真实世界中“好”的完整定义。

### 5.2 PPO 优化 Policy

策略模型生成回答，Reward Model 打分，同时用 KL 惩罚限制它偏离参考模型：

\[
\max_\theta\ \mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}
\left[r_\phi(x,y)-\beta D_{KL}\left(\pi_\theta\|\pi_{ref}\right)\right]
\]

PPO 再通过裁剪目标限制单次更新幅度，降低策略突然崩坏的风险。

### 5.3 系统组成

训练时常同时涉及：

- 可训练 Policy；
- 冻结 Reference Policy；
- Reward Model；
- Value/Critic Model；
- Rollout 推理引擎；
- PPO Trainer、经验缓存与分布式通信。

这解释了经典 RLHF 为什么明显比 SFT/DPO 复杂：它把高吞吐推理和分布式反向训练放进同一个在线循环。

### 5.4 何时值得用

- 需要针对当前 Policy 的分布持续探索；
- 偏好目标复杂，离线数据无法覆盖新行为；
- 有成熟 Reward Model、GPU 集群和在线训练能力；
- 模型质量收益足以覆盖数倍工程与算力成本。

对多数企业定制任务，PPO 不应是第一版方案。

## 6. 方案二：DPO 与离线偏好优化

DPO 直接用偏好对训练 Policy，不单独拟合 Reward Model，也不在训练环中在线生成。

给定 Policy \(\pi_\theta\)、冻结参考模型 \(\pi_{ref}\)、偏好回答 \(y_w\) 和非偏好回答 \(y_l\)，定义：

\[
z=\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\right]
\]

\[
L_{DPO}=-\log\sigma(z)
\]

DPO 让 Policy 相对 Reference 更偏向 chosen，同时用 Reference 隐式约束分布漂移。

### 6.1 业务优势

- 流程接近普通监督训练，容易接入现有 Trainer；
- 不需要 Reward/Value Model 和在线 Rollout；
- 训练稳定、实验周期短；
- 适合已有高质量历史偏好对的团队。

### 6.2 局限

- 只能充分利用离线数据覆盖到的行为；
- 偏好对若由旧模型生成，会与新 Policy 存在分布差异；
- chosen/rejected 长度、模板或风格泄漏会被当成偏好特征；
- 仍需同时加载 Policy 与 Reference，显存通常高于 SFT；
- \(\beta\)、序列概率聚合和数据质量会明显影响结果。

### 6.3 适合的业务任务

- 客服语气、完整性和拒答质量；
- 摘要、写作和品牌表达；
- Tool Call 方案排序；
- 专家可以比较候选，但无法持续写标准答案；
- 从生产日志积累了可信 chosen/rejected 数据。

## 7. 方案三：KTO、ORPO 与其他轻量偏好目标

| 方法 | 数据要求 | 主要特点 | 适用情况 |
|---|---|---|---|
| DPO | 成对 chosen/rejected | 生态成熟、强基线 | 能为同一 Prompt 构造可靠偏好对 |
| KTO | 独立好/坏样本 | 不要求严格成对 | 只有点赞/点踩或审核通过/拒绝日志 |
| ORPO | chosen + rejected | 把 SFT 与偏好惩罚合并，不需 Reference | 希望减少训练阶段和模型副本 |
| IPO 等 | 成对偏好 | 修改偏好目标的统计性质 | 有研究能力并发现 DPO 特定缺陷 |

这些方法不是按论文新旧排序的“升级链”。数据形态、训练稳定性、实现成熟度和业务评测共同决定选择。默认优先 DPO；只有现有数据无法自然组成成对偏好，或 Reference 成本成为明确瓶颈时，再尝试替代目标。

## 8. 方案四：RLAIF 与规则奖励

Reinforcement Learning from AI Feedback 使用 AI 代替或辅助人类提供反馈。Constitutional AI 的典型思路是：

1. 人类定义一组原则或“宪法”；
2. 模型依据原则批评并修订自己的回答；
3. 用修订结果做监督训练；
4. AI Judge 根据原则比较候选；
5. 用 AI 偏好进行 Preference Optimization 或 RL。

业务价值：

- 扩展到人类难以逐条覆盖的长尾场景；
- 把自然语言政策快速转成数据生成与审核规则；
- 降低初筛成本，让人类集中处理冲突和高风险样本。

主要风险：

- Judge 与被训练模型共享偏见或盲点；
- 复杂政策在自然语言中仍有歧义；
- 候选文本可能对 Judge 进行 Prompt Injection；
- 同一 Judge 既生成又评价，容易形成自我强化；
- AI 高一致性不代表与真实用户偏好一致。

因此更稳妥的结构是“AI 扩量 + 程序校验 + 人工校准 + 独立红队”，而不是完全取消人类。

## 9. 方案五：RLVR / Reinforcement Fine-tuning

RLVR（Reinforcement Learning with Verifiable Rewards）直接用可验证结果训练 Policy。它尤其适合数学、代码、工具使用和有模拟环境的 Agent。

```text
Prompt
  ↓ Policy 一次生成多个候选
Verifier / Grader 对每个候选打分
  ↓ 估计 Advantage
PPO / GRPO 等更新 Policy
```

### 9.1 PPO 与 GRPO

PPO 通常依赖 Value Model 估计 Advantage。GRPO 对同一 Prompt 采样一组回答 \(y_1,\ldots,y_G\)，用组内奖励做相对标准化：

\[
A_i=\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)+\epsilon}
\]

再用策略比率、裁剪和 KL 约束更新模型。它省去单独的 Value Model，但仍需要多候选 Rollout，生成成本可能成为主要瓶颈。

### 9.2 哪些业务适合

- 代码能运行测试；
- 数学、逻辑题有明确答案或证明检查器；
- SQL、搜索、规划可在沙箱中计算结果；
- Agent 的任务成功、步骤数、费用和安全约束可观测；
- 需要模型通过探索发现优于示范的新策略。

### 9.3 哪些业务不适合直接做

- “文案更有创意”“回复更温暖”等高度主观目标；
- 结果需要数周后才能观测，且归因不清；
- Grader 很容易被输出文本欺骗；
- 高风险动作无法在隔离模拟环境中执行；
- 任务基础成功率接近零，采样几乎得不到正奖励。

当奖励极稀疏时，先用 SFT 把模型带到“偶尔成功”的区域，通常比直接 RL 更有效。

## 10. 主流方案选型矩阵

| 场景 | 推荐起点 | 为什么 | 主要代价 |
|---|---|---|---|
| 有标准答案示范 | SFT | 目标直接、训练简单 | 需要高质量答案 |
| 有成对偏好数据 | DPO | 稳定、无需在线 RL | 受离线分布限制 |
| 只有点赞/点踩 | KTO 或先构造偏好对 | 利用非成对反馈 | 曝光与选择偏差 |
| 想合并 SFT 与偏好阶段 | ORPO | 不需 Reference | 生态和经验少于 DPO |
| 主观开放式质量且资源充足 | RM + PPO | 可在线探索 | 系统复杂、成本高 |
| 数学/代码/工具有 Verifier | GRPO/PPO + 可验证奖励 | 奖励客观、可扩量 | Rollout 昂贵、易钻规则漏洞 |
| 安全规范可写成原则 | RLAIF + 人工校准 | 快速覆盖长尾 | Judge 偏差与注入风险 |
| 线上模型不断变化 | 在线偏好/RL | 减少离线分布错位 | 数据、训练和发布治理复杂 |

## 11. 人类偏好数据怎么生产

### 11.1 先写 Rubric

不要只告诉标注员“选更好的”。Rubric 应把目标拆成可判断维度，例如：

1. 事实正确；
2. 完成用户明确请求；
3. 不虚构来源或系统状态；
4. 遵循安全与权限政策；
5. 信息完整但不过度冗长；
6. 工具选择和参数正确；
7. 若信息不足，明确说明并采取正确下一步。

还要规定维度冲突时的优先级。例如事实与流畅度冲突时，事实必须优先。

### 11.2 候选生成

候选应有足够差异但不能过于悬殊：

- 从当前生产模型、候选模型和强 Teacher 采样；
- 使用多个 Temperature 与随机种子；
- 混入历史真实失败和对抗样本；
- 去除模型名称、模板等泄漏来源的特征；
- 对候选顺序随机化，防止位置偏差。

若 chosen 永远更长、总是由同一模型生成，Preference Model 可能只学到长度或文风，而没有学到质量。

### 11.3 标注质量控制

- 标注员培训与资格题；
- 重复标注估计一致性；
- 插入已知答案的 Gold 样本；
- 允许 Tie、Both Bad 和无法判断；
- 对低一致性样本交给专家仲裁；
- 记录标注员、Rubric 版本、时间和理由；
- 高风险领域使用具备资质的专家。

分歧不一定是噪声。它可能暴露产品政策不清、不同用户群偏好冲突，或 Prompt 缺少必要上下文。

## 12. Reward 与 Grader 工程

### 12.1 多目标奖励

业务奖励通常是多个分量：

\[
r=w_1r_{correct}+w_2r_{helpful}+w_3r_{safe}
-w_4c_{latency}-w_5c_{tool}-w_6p_{violation}
\]

简单加权容易掩盖硬约束。例如安全违规不能靠“更有帮助”抵消。更稳妥的做法是：

- 先用硬规则淘汰非法输出；
- 再在合法集合中优化质量；
- 对不同任务分别归一化奖励；
- 监控每个分量，而不是只看总 Reward。

### 12.2 Grader 类型

| Grader | 优点 | 风险 |
|---|---|---|
| 字符串/Schema | 快、确定 | 只检查表面形式 |
| 单元测试/执行器 | 与任务成败接近 | 测试不完备、需安全沙箱 |
| 规则引擎 | 可解释、可审计 | 维护成本高，边界僵硬 |
| Reward Model | 快、可批量 | 分布外失真、会被利用 |
| LLM Judge | 灵活、易表达 Rubric | 偏差、成本、注入和漂移 |
| 人工专家 | 高语境理解 | 慢、贵、一致性有限 |

关键业务应组合多个 Grader，并保留与训练奖励独立的评测器，防止“既当教练又当裁判”。

### 12.3 防止 Reward Hacking

- 训练前对 Grader 做对抗测试；
- 把隐藏测试与公开测试分开；
- 限制答案访问 Grader 提示、测试和参考答案；
- 监控长度、重复、拒答率和异常 Token 模式；
- 定期人工检查高 Reward 样本；
- 在训练中加入新发现的反例；
- 不让模型直接修改奖励函数、测试或执行环境；
- 使用最小权限和隔离沙箱运行工具。

## 13. 在线训练闭环

成熟的偏好系统不是一次性训练，而是受控迭代：

```text
生产/任务 Prompt 池
  ↓ 去隐私、分层采样
当前 Policy 生成候选
  ↓
人类 / AI / Verifier 反馈
  ↓ 质量控制、去重、版本化
SFT / DPO / PPO / GRPO
  ↓
离线评测、红队、回归
  ↓
Shadow / Canary / A/B
  ↓
监控并回流新的失败样本
```

每一轮要记录数据生成 Policy。用旧模型产生的偏好对训练更强模型时，离线覆盖不足会逐渐成为瓶颈，需要重新采样当前 Policy 的候选。

## 14. 评测不能复用训练奖励

至少建立以下评测面：

| 评测面 | 指标示例 |
|---|---|
| 偏好质量 | 人工盲测胜率、Pairwise Win Rate、Tie Rate |
| 任务正确性 | Exact Match、F1、Pass@k、执行成功率 |
| 校准与诚实 | 不确定时拒识、引用正确率、幻觉率 |
| 安全 | 有害响应率、过度拒答率、越权工具调用率 |
| 通用回归 | 知识、代码、数学、多语言和指令遵循 |
| 系统效率 | P50/P95 延迟、输出长度、工具次数、单位成功成本 |

离线 Judge 最好满足：

- 与训练 Judge 不同或至少使用隐藏 Rubric；
- 候选顺序随机，必要时交换顺序复评；
- 对长度偏好、身份偏好和自我偏好做校准；
- 定期与人工专家的一致性比较；
- 评测集按时间滚动，保留一套永久回归集。

上线时采用小流量 Canary，并设置质量、安全、成本和延迟的自动回滚阈值。

## 15. 最常见的失败模式

### 15.1 Reward Hacking

Reward 上升但真实质量下降。模型可能学会堆叠关键词、输出更长内容、伪造证明、跳过困难步骤或攻击 Judge。

### 15.2 过度优化与能力坍缩

Policy 偏离 Reference 太远，语言质量、多样性或通用能力下降。可通过 KL、较小学习率、早停、SFT 数据混合和多维回归监控缓解。

### 15.3 长度偏好

标注员和 Judge 常把更长回答误判为更完整，模型因此不断变啰嗦。应构造等长对照、单独记录长度、在 Rubric 中明确“必要且充分”。

### 15.4 Sycophancy

模型为了偏好分数迎合用户错误观点。Rubric 应把事实和证据置于讨好之上，并专门加入“用户坚持错误前提”的对抗集。

### 15.5 过度拒答

安全奖励过强时，模型对正常请求也拒绝。安全评测必须同时包含应拒绝与应正常回答的相邻边界样本。

### 15.6 分布漂移

Reward Model 在旧 Policy 样本上表现良好，对新 Policy 产生的奇怪输出却不可靠。在线采样、周期性重标注和不确定性检测可降低风险。

### 15.7 单一平均偏好

不同客户、地区和角色的偏好可能冲突。将所有偏好压成单一 Reward 会抹平差异。可考虑条件化策略、多 Adapter、显式用户设置或按场景路由，而不是假设存在唯一“人类偏好”。

## 16. 安全与治理

RLHF 训练必须纳入完整数据与模型治理：

- 明确用户反馈是否获得训练授权；
- 去除 PII、密钥、内部标识和受限数据；
- 保留数据来源、许可、删除和审计链路；
- 限制标注员看到的敏感信息；
- 对 Reward Model、Judge Prompt 和测试集做访问控制；
- 高风险 Tool 只在沙箱或模拟器中 Rollout；
- 发布前执行安全评审、红队与回滚演练；
- 不把训练对齐当成唯一安全边界。

模型权重中的行为约束是概率性的。权限、交易确认、内容过滤和合规校验仍需由确定性系统执行。

## 17. 推荐的业务落地顺序

### 第一阶段：不用 RL

1. 定义业务 KPI、失败分类和安全边界；
2. 建立 Prompt/RAG/Tool 基线；
3. 用专家示范做 SFT；
4. 固定 Golden Set、回归集与线上监控。

### 第二阶段：离线偏好优化

1. 从真实失败和模型候选构造高质量偏好对；
2. 先做 DPO 基线；
3. 用人工盲测验证它是否真的优于 SFT；
4. 分析收益来自事实、风格、安全还是长度；
5. 只有数据形态明确不适合 DPO 时，再试 KTO/ORPO 等目标。

### 第三阶段：可验证强化

1. 构建不可被轻易利用的 Verifier；
2. 在隔离环境中生成多个 Rollout；
3. 先跑小规模 PPO/GRPO 实验；
4. 同时监控 Reward、真实任务成功率和通用回归；
5. 人工检查最高奖励与异常策略；
6. 收益稳定后再扩大 Prompt、模型和算力规模。

### 第四阶段：在线闭环

只有在数据合规、标注质量、训练稳定性、发布门禁和回滚机制成熟后，才引入持续在线偏好采集与迭代训练。

## 18. 一页选型结论

```text
能否写出明确的理想答案？
  ├─ 能 -> 先做 SFT
  └─ 不能
      ↓
能否稳定比较两个回答的好坏？
  ├─ 能 -> 先做 DPO
  └─ 只能标好/坏 -> KTO 或重构数据

输出能否由程序或环境验证？
  ├─ 能 -> SFT 后尝试 RLVR（PPO/GRPO）
  └─ 不能 -> 人类偏好 / RLAIF + 人工校准

是否需要当前 Policy 持续探索新回答？
  ├─ 是，且资源与治理成熟 -> 在线 RM+PPO 或可验证 RL
  └─ 否 -> 离线偏好优化通常更划算
```

默认建议是：

> 大多数业务从“强基座 + 高质量 SFT + DPO + 完整评测”开始；当任务拥有可靠、难以被利用的可验证奖励，且离线方法已达到瓶颈，再投入 PPO/GRPO 等在线强化训练。

---

## 延伸阅读

- [Training language models to follow instructions with human feedback（InstructGPT）](https://arxiv.org/abs/2203.02155)
- [Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [DeepSeekMath：GRPO](https://arxiv.org/abs/2402.03300)
- [OpenAI：Rule-Based Rewards for Language Model Safety](https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/)
- [NVIDIA NeMo RL：DPO](https://docs.nvidia.com/nemo/rl/latest/guides/dpo.html)
- [Amazon Bedrock：Reinforcement Fine-tuning](https://docs.aws.amazon.com/bedrock/latest/userguide/reinforcement-fine-tuning.html)
- [本项目：miniLLM DPO 训练](../实战篇/miniLLM%20DPO训练.md)
- [本项目：LLM SFT 阶段主流算法解析](../算法篇/LLM%20SFT阶段算法解析.md)

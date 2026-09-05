# Tokenizer 主流算法解析：从文本规范化到 BPE、Unigram 与工程评估

> Tokenizer 不是简单的“切词工具”，而是语言、数据和模型矩阵之间的离散接口。本文以业界主流算法与工程决策为核心，最后简述 miniLLM 的实现。

## 1. Tokenizer 在优化什么

Tokenizer 要把任意文本 \(x\) 映射为整数序列：

\[
\operatorname{encode}(x)=[t_1,t_2,\ldots,t_n]
\]

理想接口需要同时满足：

- **覆盖性**：中文、英文、代码、数字、Emoji、生僻字符都可编码；
- **可逆性**：解码尽量恢复原始文本；
- **压缩效率**：常见内容用更少 Token；
- **泛化性**：未登录词不会崩溃；
- **稳定性**：模型训练后 Token ID 语义不再改变；
- **结构性**：对话边界、工具调用和控制标记具有明确语义；
- **计算经济性**：词表大小与序列长度共同决定训练和推理成本。

Tokenizer 的目标不是单独让 Token 数最少。过大的词表会扩大 Embedding/LM Head，过度合并还可能损伤形态共享、数字规律和跨语言迁移。

## 2. 一条完整的 Tokenizer 流水线

```text
原始文本
  -> Unicode Normalization
  -> Pre-tokenization
  -> Subword Model
  -> Special-token Processing
  -> Token IDs
  -> Decoder
```

### 2.1 Unicode 规范化

NFC/NFKC、大小写、全半角和空白处理会改变后续统计。NFKC 能合并兼容字符，但也可能丢失对代码、数学或特定语言有意义的差异。

原则是：先明确产品是否要求字节级还原，再决定规范化强度；训练和推理必须使用完全相同的规则。

### 2.2 Pre-tokenization

预切分先确定哪些边界允许子词算法跨越。常见依据有空白、标点、正则模式或字节映射。它不是最终分词，却会显著限制最终词表形态。

### 2.3 Subword Model

子词模型在字符/字节和完整词之间寻找折中：高频片段合并为一个 Token，低频内容拆成更小单位。

### 2.4 Post-processing 与特殊 Token

BOS、EOS、PAD、消息边界、工具标记等在编码后按模板插入。它们的字符串、ID 和是否跳过解码必须成为模型协议的一部分。

## 3. 四种基本粒度

| 粒度 | 优点 | 局限 |
|---|---|---|
| Word | 序列短、语义直观 | 词表爆炸、未登录词严重 |
| Character | 覆盖稳定 | 英文和代码序列较长 |
| Byte | 固定 256 基础符号、无 OOV | 人类字符常被拆成多字节 |
| Subword | 压缩与开放词表折中 | 依赖语料、训练算法和规范化 |

现代 LLM 通常使用子词算法，并以字符或字节作为兜底。

## 4. BPE：从频繁相邻对逐步合并

### 4.1 核心算法

1. 将语料表示为初始符号序列；
2. 统计所有相邻符号对频率；
3. 合并最高频符号对；
4. 更新语料表示和统计；
5. 重复直到达到目标词表或停止条件。

若语料中频繁出现：

```text
l o w
l o w e r
n e w e s t
```

算法可能依次学习 `l+o -> lo`、`lo+w -> low`。最终编码必须按已学习 Merge 规则执行，而不是对新文本重新统计。

### 4.2 BPE 的性质

- 算法简单、训练与编码高效；
- 高频片段获得短编码；
- 合并是贪心构造，未直接优化下游语言模型损失；
- 早期合并会影响后续候选，语料偏差会固化进词表。

## 5. Byte-level BPE：字节兜底与合并压缩

Byte-level BPE 先把 UTF-8 字节映射到基础符号，再学习 BPE Merge。任何输入最终都可回退到字节，因此不需要普通 `<unk>` 才能覆盖未知字符。

优势：

- 任意 Unicode、代码和混合文本都能编码；
- 可实现严格字节级可逆；
- 词表不需要包含所有 Unicode 字符。

代价：

- 罕见非拉丁字符可能拆成多个 Token；
- 空白与字节映射后的 Token 不总是人类可读；
- 中英文比例不合理时，跨语言压缩差异会很大。

GPT-2 路线推动了 Byte-level BPE 在生成模型中的普及，许多后续模型和开源工具延续了类似思想。

## 6. WordPiece：按模型收益选择合并

WordPiece 与 BPE 都构建子词词表，但经典描述中，WordPiece 更强调某个候选加入词表后对训练数据语言模型似然的改善，而不是只看相邻对绝对频率。

它常与 `##` 一类词内边界表示关联，并因 BERT 生态广为人知。实际工具对训练细节可能不同，不能仅凭输出格式反推完整算法。

## 7. Unigram Language Model：从候选大词表向下剪枝

Unigram 假设一个分词 \(z=(z_1,\ldots,z_m)\) 的概率为：

\[
p(z)=\prod_{i=1}^{m}p(z_i)
\]

文本概率对所有可能分词求和。训练通常从大候选词表开始，反复估计概率并移除对似然贡献较小的 Token，直到达到目标词表。

与 BPE 的主要差异：

- BPE 从小词表向上合并；
- Unigram 从大候选集向下剪枝；
- Unigram 自然保留多个分词候选，可用于 Subword Regularization；
- 训练和动态规划编码相对更复杂。

## 8. SentencePiece 是框架，不是单一算法

SentencePiece 可直接在原始句子上训练，常用 BPE 或 Unigram 作为底层算法，并把空白显式表示为符号。这对无天然空格边界的语言和多语言训练很实用。

所以“SentencePiece vs BPE”不是严格同层比较；更准确的问题是“使用 SentencePiece 框架下的 BPE 还是 Unigram”。

## 9. 词表大小如何权衡

词表为 \(V\)、隐藏维度为 \(d\) 时，Embedding 参数约为 \(Vd\)；若输出头不共享，还要再增加约 \(Vd\)。

词表增大通常会：

- 降低常见文本的 Token 数；
- 增大 Embedding 和输出 Softmax；
- 降低稀有 Token 的训练频次；
- 增加 Tokenizer 训练所需覆盖数据；
- 改变多语言和代码之间的容量分配。

因此应做完整曲线，而不是只测一个词表：例如 8K、16K、32K、64K，在相同评测集上比较压缩、参数、吞吐和下游能力。

## 10. 多语言、代码、数字和空白

### 10.1 多语言平衡

高资源语言会占据更多 Merge。常见方法包括按语言重采样、限制单域占比、扩大低资源语言数据或显式加入字符覆盖。

### 10.2 代码与空白

缩进、换行和标点承载语法。过强 Unicode/空白规范化可能破坏程序；Tokenizer 评估必须包含多语言代码、路径、URL 和长标识符。

### 10.3 数字

将常见长数字整体合并可缩短序列，却可能削弱逐位运算和泛化。应单独测试日期、小数、科学计数法、货币和随机长整数。

### 10.4 Byte Fallback

不论主算法是 BPE 还是 Unigram，都可引入 Byte Fallback，保证未覆盖字符有确定编码。要验证 `decode(encode(x)) == x`，而不只是“没有抛异常”。

## 11. 特殊 Token 与 Chat Template 是协议

需要冻结的内容包括：

- PAD/BOS/EOS/UNK 的字符串与 ID；
- System/User/Assistant 消息边界；
- Reasoning、Tool Call、Tool Result 的结构标记；
- 是否自动加入 BOS/EOS；
- 训练时哪些 Token 参与 Loss；
- 解码时哪些 Token 被跳过。

修改特殊 Token 顺序会改变 ID；模型训练后再改词表，等价于改变 Embedding 行的含义。新增 Token 若确有必要，必须同步扩展并初始化模型权重。

## 12. Tokenizer 训练数据算法

一个可靠管线通常包含：

1. 语料发现与格式解析；
2. 文档级去重和近重复检测；
3. 语言、领域、质量与安全过滤；
4. 按目标分布采样，而不是简单拼接；
5. 超长文档合理切分，并保留换行等结构；
6. 冻结独立评测集；
7. 保存语料清单、哈希、算法参数和随机种子。

Tokenizer 数据应覆盖模型未来会遇到的输入，而不应直接用评测题答案去优化词表。

## 13. 怎样评估 Tokenizer

### 13.1 正确性

- 任意有效输入可编码；
- Round-trip 可逆；
- 特殊 Token 原子化且 ID 固定；
- 保存后重载结果一致；
- 快慢实现或多语言绑定行为一致。

### 13.2 压缩效率

建议分域报告：

\[
\text{tokens/character}=\frac{N_{token}}{N_{character}}
\]

\[
\text{bytes/token}=\frac{N_{UTF8\ byte}}{N_{token}}
\]

`bytes/token` 更适合跨语言比较，因为不同 Unicode 字符的字节数不同。还应报告 P50/P95/P99，防止平均值掩盖极端输入。

### 13.3 结构与公平性

- 不同语言的压缩差距；
- 代码缩进、数学和 Emoji 是否可逆；
- 名称、方言、低资源语言是否被异常拉长；
- Chat Template 是否精确产生预期边界。

### 13.4 下游效果

最终标准是在相同模型参数、训练 Token/FLOPs 和数据预算下，比较验证损失、能力评测和实际吞吐。只看压缩率不能决定最佳 Tokenizer。

## 14. 算法选择建议

| 场景 | 常见选择 | 重点验证 |
|---|---|---|
| 通用生成、代码、任意字符 | Byte-level BPE | 非拉丁语言压缩率 |
| 多语言与分词采样 | SentencePiece Unigram | 训练复杂度、可逆性 |
| BERT 兼容生态 | WordPiece | 规范化和 OOV 行为 |
| 大规模开放模型 | BPE/Unigram + Byte Fallback | 数据配比、词表规模、特殊协议 |

选择应由目标语料和下游实验决定，不应只因为某个知名模型采用了某算法。

## 15. miniLLM 的 Tokenizer 映射

miniLLM 使用 8192 词表的 ByteLevel-BPE，从 JSONL/TXT 语料流式提取文本，以字节保证开放字符覆盖，并固定 `<|endoftext|>`、`<|im_start|>`、`<|im_end|>` 的 ID。它还预留 Reasoning/Tool 结构标记和 Chat Template，训练后进行词表大小、特殊 ID、结构 Token 原子性与编解码检查。

这是一条适合小模型和教学项目的稳健基线。当前最值得补强的不是继续堆训练语料，而是冻结独立、分域的 Tokenizer Eval 集，比较不同词表大小和语料配比的 `bytes/token`、长尾分位数与下游预训练损失。

## 参考资料

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [SentencePiece](https://arxiv.org/abs/1808.06226)
- [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/)
- [Google SentencePiece](https://github.com/google/sentencepiece)

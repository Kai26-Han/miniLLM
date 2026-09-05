# Tokenizer Algorithms: From Text Normalization to BPE, Unigram, and Evaluation

> tokenizer is not a simple "tokenization tool", but a discrete interface between language, data and model matrix. This article takes the mainstream algorithms and engineering decision-making in the industry as the core, and finally briefly describes the implementation of miniLLM.

## 1. What is tokenizer optimizing?

tokenizer wants to map any text \(x\) into an integer sequence:

\[
\operatorname{encode}(x)=[t_1,t_2,\ldots,t_n]
\]

The ideal interface needs to meet at the same time:

- **Coverage**: Chinese, English, code, numbers, Emoji, rare characters can all be encoded;
- **Reversibility**: Decoding to restore the original text as much as possible;
- **Compression efficiency**: Less token is used for common content;
- **Generalization**: out-of-vocabulary words will not collapse;
- **Stability**: After model training, the semantics of token ID will no longer change;
- **STRUCTURAL**: DIALOGUE BOUNDARIES, TOOL CALLS AND CONTROL MARKS HAVE CLEAR SEMATICS;
- **Computational economy**: The size of the vocabulary and the length of the sequence together determine the cost of training and inference.

The goal of tokenizer is not to make the number of tokens the least. Oversized vocabularies will expand embedding/LM Head, and excessive merging may also damage morphological sharing, digital laws and cross-language migration.

## 2. A complete tokenizer pipeline

```text
Original text
  -> Unicode Normalization
  -> Pre-tokenization
  -> Subword Model
  -> Special-token Processing
  -> token IDs
  -> Decoder
```

### 2.1 Unicode Normalization

NFC/NFKC, uppercase and lowercase, full half-width and blank processing will change the subsequent statistics. NFKC can combine compatible characters, but may also lose the difference in meaning to code, mathematics or specific language.

The principle is: first clarify whether the product requires byte-level reduction, and then determine the standardization intensity; training and inference must use exactly the same rules.

### 2.2 Pre-tokenization

Pre-cutting first determines which boundaries allow the subword algorithm to cross. Common bases include blanks, punctuation, regular patterns or byte mapping. It is not the final tokenization, but it will significantly limit the form of the final vocabulary.

### 2.3 Subword Model

The sub-word model seeks a compromise between characters/bytes and complete words: high-frequency fragments are combined into a token, and low-frequency content is broken down into smaller units.

### 2.4 Post-processing and Special token

BOS, EOS, PAD, message boundaries, tool marks, etc. are inserted according to the template after encoding. Their strings, IDs and whether to skip decoding must be part of the model protocol.

## 3. Four basic particle sizes

| Particle size | Advantages | Limitations |
|---|---|---|
| Word | Short sequence, semantic intuition | vocabulary explosion, serious out-of-vocabulary words |
| Character | Stable coverage | Long English and code sequences |
| Byte | Fixed 256 basic symbols, no OOV | Human characters are often disassembled into multiple bytes |
| Subword | Compression and open vocabulary compromise | Rely on corpus, training algorithm and standardization |

Modern LLM usually uses sub-word algorithms and uses characters or bytes as the bottom.

## 4. BPE: gradual merging from frequent adjacent pairs

### 4.1 Core Algorithm

1. Express the corpus as the initial symbol sequence;
2. Count the frequency of all adjacent symbol pairs;
3. Merge the highest frequency symbol pair;
4. Update corpus representation and statistics;
5. Repeat until the target vocabulary or stop condition is reached.

If it appears frequently in the corpus:

```text
l o w
l o w e r
n e w e s t
```

The algorithm may learn `l+o -> lo` and `lo+w -> low` in turn. The final coding must be executed according to the learned Merge rules, instead of re-counting the new text.

### 4.2 The nature of BPE

- Simple algorithm, efficient training and coding;
- High-frequency fragments obtain short coding;
- Merging is a greedy construction, which does not directly optimize the loss of the downstream language model;
- Early mergers will affect subsequent candidates, and corpus deviations will solidify the vocabulary.

## 5. Byte-level BPE: Byte bottoming and merging compression

Byte-level BPE first maps UTF-8 bytes to the basic symbol, and then learns BPE Merge. Any input can eventually be returned to bytes, so there is no need for ordinary `<unk>` to cover unknown characters.

Advantages:

- Any Unicode, code and mixed text can be encoded;
- Strict byte-level reversible can be realized;
- The vocabulary does not need to contain all Unicode characters.

Price:

- Rare non-Latin characters may be disassembled into multiple tokens;
- token after mapping blanks and bytes is not always human-readable;
- When the ratio of Chinese and English is unreasonable, the difference in cross-language compression will be large.

The GPT-2 route has promoted the popularity of Byte-level BPE in the generation model, and many follow-up models and open source tools continue similar ideas.

## 6. WordPiece: Select and merge according to model utility gain

Both WordPiece and BPE construct sub-vocabularies, but in the classic description, WordPiece emphasizes the likelihood improvement of the training data language model after a candidate is added to the vocabulary, rather than just looking at the absolute frequency of adjacent pairs.

It is often associated with the internal boundary of words such as `##`, and is widely known for its BERT ecology. The actual tool may have different training details, and the complete algorithm cannot be reversed by the output format alone.

## 7. Unigram Language Model: Pruning down from the list of candidate big words

Unigram assumes that the probability of a tokenization \(z=(z_1,\ldots,z_m)\) is:

\[
p(z)=\prod_{i=1}^{m}p(z_i)
\]

Text probability sums all possible tokenization. Training usually starts with a large candidate list, repeatedly estimates the probability and removes tokens that contribute less to likelihood until the target list is reached.

The main differences with BPE:

- BPE merges from the list of small words upwards;
- Unigram prunes down from the big candidate set;
- Unigram naturally retains multiple tokenization candidates, which can be used for Subword Regularization;
- Training and dynamic planning coding are relatively more complex.

## 8. SentencePiece is a framework, not a single algorithm.

SentencePiece can be trained directly on the original sentence, often using BPE or Unigram as the underlying algorithm, and explicitly representing blanks as symbols. This is very practical for language and multilingual training without natural space boundaries.

Therefore, "SentencePiece vs BPE" is not a strict comparison of the same layer; the more accurate question is "use BPE or Unigram under the framework of SentencePiece".

## 9. How to weigh the size of the vocabulary

When the vocabulary is \(V\) and the hidden dimension is \(d\), the embedding parameter is about \(Vd\); if the output head is not shared, about \(Vd\) must be added.

The increase in the list of words usually:

- Reduce the number of tokens for common texts;
- Increase embedding and output Softmax;
- Reduce the training frequency of rare tokens;
- Increase the coverage data required for tokenizer training;
- Change the capacity allocation between multiple languages and codes.

Therefore, a complete curve should be made, instead of measuring only one vocabulary: for example, 8K, 16K, 32K, 64K, and compare compression, parameters, throughput and downstream capacity on the same evaluation set.

## 10. Multi-language, code, number and blank

### 10.1 Multilingual balance

High-resource languages will occupy more Merge. Common methods include resampling by language, limiting the proportion of single domains, expanding low-resource language data, or explicitly adding character coverage.

### 10.2 Code and Blank

Indentation, newline and ponctuation carry syntax. Excessive Unicode/blank normalization may damage the program; tokenizer evaluation must contain multilingual code, path, URL and long identifier.

### 10.3 numbers

Merging common long numbers as a whole can shorten the sequence, but it may weaken bit-by-bit operations and generalization. Date, decimal, scientific counting method, currency and random length integers should be tested separately.

### 10.4 Byte Fallback

Whether the main algorithm is BPE or Unigram, Byte Fallback can be introduced to ensure that unoverwritten characters have definite encoding. It is necessary to verify `decode(encode(x)) == x`, not just "no throwing exception".

## 11. Special token and chat template are protocols.

The contents that need to be frozen include:

- The string and ID of PAD/BOS/EOS/UNK;
- System/User/assistant message boundary;
- Structural tags of Reasoning, tool call, Tool Result;
- Whether to automatically join BOS/EOS;
- Which tokens participate in Loss during training;
- Which tokens are skipped when decoding.

Modifying the special token order will change the ID; changing the vocabulary after model training is equivalent to changing the meaning of the embedding line. If it is really necessary to add token, it must be synchronously expanded and initialized model weights.

## 12. tokenizer Training Data Algorithm

A reliable pipeline usually includes:

1. Corpus discovery and format analysis;
2. Document-level de-duplication and near-repetition detection;
3. Language, domain, quality and security filtering;
4. Sampling by target distribution, not simple splicing;
5. Extra-long documents are reasonably cut, and structures such as line newlines are retained;
6. Freeze the independent evaluation set;
7. Save the corpus list, hash, algorithm parameters and random seeds.

tokenizer data should cover the input that the model will encounter in the future, and should not directly use the answers to the evaluation questions to optimize the vocabulary.

## 13. How to evaluate tokenizer

### 13.1 Correctness

- Any effective input can be encoded;
- Round-trip is reversible;
- Special token is atomized and ID is fixed;
- After saving, the overloading results are consistent;
- Fast and slow implementation or consistent multi-language binding behavior.

### 13.2 Compression efficiency

Suggested sub-domain report:

\[
\text{tokens/character}=\frac{N_{token}}{N_{character}}
\]

\[
\text{bytes/token}=\frac{N_{UTF8\ byte}}{N_{token}}
\]

`bytes/token` is more suitable for cross-language comparison, because the number of bytes of different Unicode characters is different. P50/P95/P99 should also be reported to prevent the average from covering up extreme input.

### 13.3 Structure and Fairness

- The compression gap between different languages;
- Whether code indentation, mathematics and emoji are reversible;
- Whether the names, dialects and low-resource languages are abnormally lengthened;
- Whether chat template accurately generates the expected boundary.

### 13.4 Downstream effect

The final standard is to compare the verification loss, capability evaluation and actual throughput under the same model parameters, training token/FLOPS and data budget. Only the compression rate cannot determine the best tokenizer.

## 14. Suggestions for algorithm selection

| Scene | Common Choices | Key Verification |
|---|---|---|
| Common generation, code, arbitrary characters | Byte-level BPE | Non-Latin language compression rate |
| Multilingual and tokenization sampling | SentencePiece Unigram | Training complexity, reversibility |
| BERT Compatible Ecology | WordPiece | Normalization and OOV Behavior |
| Large-scale open model | BPE/Unigram + Byte Fallback | Data ratio, vocabulary scale, special protocol |

The selection should be decided by the target corpus and downstream experiments, not just because a well-known model adopts an algorithm.

## 15. tokenizer mapping of miniLLM

miniLLM uses the ByteLevel-BPE of the 8192 vocabulary to extract text from the JSONL/TXT corpus stream, guarantee open character coverage in bytes, and fix the IDs of `<|endoftext|>`, `<|im_start|>` and `<|im_end|>`. It also reserves Reasoning/Tool structure tags and chat template, and checks the vocabulary size, special ID, structure token atomicity and encoding and decoding after training.

This is a stable baseline suitable for small models and teaching projects. At present, the most worthwhile thing is not to continue to stack the training corpus, but to freeze the independent, sub-domain tokenizer Eval set, and compare the `bytes/token`, long-tail quentile and downstream pre-training losses with different vocabulary sizes and corpus ratios.

## Reference materials

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [SentencePiece](https://arxiv.org/abs/1808.06226)
- [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/)
- [Google SentencePiece](https://github.com/google/sentencepiece)

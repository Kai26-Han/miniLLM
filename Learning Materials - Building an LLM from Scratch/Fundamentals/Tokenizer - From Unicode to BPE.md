# Tokenizers: From Unicode to BPE

## What is a tokenizer?

The tokenizer is the translation layer between text and a language model. It encodes Unicode text as UTF-8 bytes, then uses Byte Pair Encoding (BPE) to merge frequently adjacent byte sequences into larger tokens. The design balances vocabulary size, sequence length, compute cost, and model capability.

## Key points

1. **Language models process token IDs, not strings.** Text must pass through `encode()` before entering the model, and generated IDs must pass through `decode()` to become readable text.
2. **The tokenizer and language model are separate systems.** They can use different training datasets and algorithms and can be trained at different stages.
3. **BPE shortens sequences by repeatedly merging the most frequent adjacent token pair.** Every merge adds one entry to the vocabulary.
4. **A BPE vocabulary is not merely a set of unrelated strings.** Its merge rules form a forest of binary trees in which later tokens depend on earlier merges.
5. **A larger vocabulary usually produces shorter sequences but makes the embedding, output projection, and softmax more expensive.** Vocabulary size is therefore an important hyperparameter.
6. **The tokenizer-training corpus determines token density across data types.** Languages and programming constructs that appear more often are more likely to form longer tokens.
7. **Many surprising model behaviors originate in tokenization.** Token boundaries affect spelling, string reversal, integer arithmetic, multilingual text, indentation, and trailing whitespace.
8. **Special tokens are not ordinary BPE merges.** Separate logic recognizes them as document boundaries, message boundaries, or control signals.
9. **Replacing or extending a tokenizer changes the model interface.** At minimum, the input embeddings and language-model output head must be updated together.

---

## 1. Where is the tokenizer located in the system?

The complete information flow is as follows:

```text
Original text
  ↓ encode
token ID sequence
  ↓ embedding
Vector sequence
  ↓ Transformer
Logits of the next token
↓ Sampling
New token ID
  ↓ decode
Readable text
```

The tokenizer needs to provide two basic functions:

```python
ids = tokenizer.encode(text)  # str -> list[int]
text = tokenizer.decode(ids)  # list[int] -> str
```

Ideally, for any valid string, it should meet:

```python
tokenizer.decode(tokenizer.encode(text)) == text
```

But the opposite direction is not necessarily true. Any integer sequence may not correspond to a valid UTF-8 byte stream, so:

```python
tokenizer.encode(tokenizer.decode(ids)) == ids
```

Not all `ids` are guaranteed.

---

## 2. Why is there not enough character-level tokenization?

The simplest scheme is to collect all the characters in the training text and assign an integer to each character:

```python
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
```

This method is suitable for teaching, but not for open world text:

- Characters that do not appear in the training set cannot be encoded;
- There are a large number of words, symbols and expressions in the world;
- The Unicode standard is still adding new code points;
- The character sets of different languages vary greatly;
- The character-by-character sequence is long, which wastes the context window.

It is not ideal to directly use each Unicode code point as a token: the vocabulary is very large, the standard will evolve, and many code points rarely appear in the training data, and it is difficult to fully train the corresponding vector.

A more secure starting point is bytes, because any Unicode text can be stably encoded into a byte stream.

---

## 3. Unicode, code point and UTF-8

### 3.1 Unicode code points

Unicode assigns integer code points to characters. In Python, you can view the code points of a single character:

```python
ord("A")       # 65
ord("é")       # 233
chr(233)       # 'é'
```

The code point is just a character number, not its actual byte representation in the file or network.

### 3.2 UTF-8 encoding

UTF-8 converts each code point into 1 to 4 bytes:

```python
raw = "Hello, café 🙂".encode("utf-8")
ids = list(raw)
```

Characters within the ASCII range usually only account for one byte, and other characters may account for multiple bytes. The advantages of UTF-8 include:

- It can represent the entire Unicode character set;
- Compatible with ASCII;
- Wide range of Internet ecological support;
- The byte range is fixed to `0-255`;
- There will be no real unknown characters, as long as the byte vocabulary is retained, it can be encoded.

### 3.3 Why not use UTF-8 bytes directly?

If a byte is a token, the vocabulary has only 256 items, and the input and output layers are very small; but the text will be expanded into a very long sequence.

The standard attention expenditure of Transformer is approximately:

$$
O(T^2)
$$

Among them, $T$ is the token quantity. After the sequence is expanded by bytes, the attention cost and context consumption will increase rapidly.

Therefore, the goal becomes:

> Retain the completeness of UTF-8, while compressing common byte combinations into larger tokens.

BPE is the algorithm that accomplishes this work.

---

## 4. The core idea of BPE

Assume that the initial sequence is:

```text
A A B C A A B D A A B
```

The initial vocabulary is only:

```text
{A, B, C, D}
```

If `AA` is the adjacent pair that appears the most times, create a new token `Z = AA`:

```text
Z B C Z B D Z B
```

The list of words increased from 4 to 5, and the sequence became shorter. Next, continue to count the new adjacent pairs, repeat:

```text
Statistics adjacent pairs
  ↓
Choose a pair with the most appearances
  ↓
Create a new token
  ↓
Replace all non-overlapping matches
  ↓
Until the size of the target vocabulary is reached
```

When applied to byte-level BPE:

- The initial token ID is `0-255`, representing a single byte respectively;
- The first merger to generate token `256`;
- The second merger generates token `257`;
- Continue to merge until the target vocabulary size is reached.

Each new token is made up of two existing tokens, so the later merge can refer to the early merge results.

---

## 5. Step 1: Count the adjacent token pair

```python
def get_stats(ids, counts=None):
    counts = {} if counts is None else counts

    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts
```

Input:

```python
ids = [1, 2, 1, 2, 3]
```

Output:

```python
{
    (1, 2): 2,
    (2, 1): 1,
    (2, 3): 1,
}
```

The adjacent pairs that appear the most times can be obtained as follows:

```python
pair = max(stats, key=stats.get)
```

Note: BPE counts **adjacent tokens**, not the number of common occurrences of any two tokens.

---

## 6. Step 2: Perform a merger

```python
def merge(ids, pair, new_id):
    out = []
    i = 0

    while i < len(ids):
        can_merge = (
            i < len(ids) - 1
            and ids[i] == pair[0]
            and ids[i + 1] == pair[1]
        )

        if can_merge:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1

    return out
```

For example:

```python
merge([5, 6, 6, 7, 9, 1], (6, 7), 99)
# [5, 6, 99, 9, 1]
```

Special attention should be paid when realizing:

- Before checking `i + 1`, you must prevent crossing the boundary;
- After the match is successful, the pointer moves forward 2;
- Matching is a non-overlapping adjacent pair;
- It should not be modified while traversing on the original list.

---

## 7. Step 3: Train the complete BPE vocabulary

```python
def train_bpe(text, vocab_size):
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256")

    ids = list(text.encode("utf-8"))
    merges = {}
    num_merges = vocab_size - 256

    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break

        pair = max(stats, key=stats.get)
        new_id = 256 + i
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id

    return merges
```

The key parameters of the training results are:

```python
merges: dict[tuple[int, int], int]
```

For example:

```python
{
    (101, 32): 256,
    (116, 104): 257,
    (257, 101): 258,
}
```

This means:

```text
token 256 = byte 101 + byte 32
token 257 = byte 116 + byte 104
token 258 = token 257 + byte 101
```

The third rule depends on the second rule, so the merge order is part of the model parameters and cannot be disrupted.

### 7.1 compression rate

The following indicators can be used to measure the degree of compression of a text by the separator:

$$
\Text{compression ratio} =\frac{\text{original bytes}}{\text{BPE token}}
$$

The higher the compression rate, the more original text can usually be loaded in the same context window. But this does not mean that the vocabulary should be infinitely increased, because a larger vocabulary will bring other costs.

---

## 8. BPE forms a forest of binary trees.

Each original byte is a leaf node, and a parent node is created each time it is merged:

```text
        token 260
        /       \
   token 257   byte 101
    /    \
byte 116 byte 104
```

There will be many unconnected or partially connected trees in the vocabulary, so the more accurate structure is "forest".

This mental model explains two things:

1. When decoding, it is necessary to restore the bytes corresponding to each token recursively or in the order of training;
2. When coding, you must first complete the low-level merge before it can trigger the high-level merge that depends on it.

---

## 9. Construct a vocabulary and realize decoding

### 9.1 Restore the vocabulary from the merge rules

```python
def build_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}

    for (left, right), new_id in merges.items():
        vocab[new_id] = vocab[left] + vocab[right]

    return vocab
```

Here, the dictionary is required to merge and create order iterations. Modern Python will retain the dictionary insertion order, but a more secure serialization format should clearly save merge rank.

### 9.2 Decoding

```python
def decode(ids, vocab):
    raw = b"".join(vocab[token_id] for token_id in ids)
    return raw.decode("utf-8", errors="replace")
```

The process is:

```text
token ID
↓ vocab check the table
The corresponding bytes of each token
↓ Stitching
Complete byte stream
↓ UTF-8 decoding
Python string
```

### 9.3 Why use `errors="replace"`?

Any token sequence does not necessarily constitute a valid UTF-8. For example, a multi-byte character may only generate a middle byte. Strict decoding will throw an anomaly:

```python
raw.decode("utf-8", errors="strict")
```

Fault-tolerant decoding will mark invalid fragments with the replacement character `�`:

```python
raw.decode("utf-8", errors="replace")
```

If `�` appears in the generated text, it usually means that the model outputs an incomplete or illegal UTF-8 token combination.

---

## 10. Implement coding: execute according to the consolidation priority

During training, the earlier the merge rules are created, the higher the priority. When encoding new text, the available rules with the smallest rank must be applied first:

```python
def encode(text, merges):
    ids = list(text.encode("utf-8"))

    while len(ids) >= 2:
        stats = get_stats(ids)

        pair = min(
            stats,
            key=lambda p: merges.get(p, float("inf")),
        )

        if pair not in merges:
            break

        ids = merge(ids, pair, merges[pair])

    return ids
```

The key is not how many times a pair appears in the coded text, but:

> Among all the current mergeable adjacent pairs, which pair was created first in the training stage?

If all adjacent pairs are not in `merges`, the minimum rank will degenerate to infinity, and the loop must be exited at this time.

### 10.1 Empty string and single token boundary

There are no adjacent pairs of the following inputs:

```python
""
"a"
```

Therefore, the coding loop should be based on `len(ids) >= 2` to avoid calling `min()` for empty dictionaries.

### 10.2 Simple performance

The above version recounts the entire sequence in each round, which is convenient for learning and verification, but it is less efficient when processing large-scale corpus. Production realization usually:

- Use more efficient data structures to maintain adjacent relationships;
- Process independent text blocks in parallel;
- Use Rust, C++, etc. to implement hotspot paths;
- Cache common results;
- Avoid merging the complete scan sequence each time.

---

## 11. Separator training and model training must be understood separately.

### 11.1 Partiator Training

The input is a representative text, and the goal is to get:

- Merge rules and their rank;
- The vocabulary of token ID to byte;
- Regular pre-blocking rules;
- Special token definition;
- Normalization and byte retrowth strategy.

It does not use backpropagation, nor does it train Transformer parameters.

### 11.2 Language Model Training

After the tokenizer is fixed, it is used to convert a large amount of raw text into token ID. The language model reads these IDs and learns to predict the next token.

```text
tokenizer training corpus - BPE -> Fixed tokenizer
                           ↓
Model training corpus ──encode──> token data set ── reverse propagation ──> model parameters
```

Once the model starts to be trained, the tokenizer is usually frozen. Changing the token ID semantics in the middle of training will invalidate the embedding and output weights that have been learned.

### 11.3 Why do the two sets of training corpus need to be coordinated?

The composition of the sub-tokenizer corpus determines which patterns will obtain independent tokens:

- The high proportion of English will form more English word fragments;
- The proportion of Chinese, Korean, etc. is low, and the text may be disassembled into more tokens;
- The high proportion of code will form common keywords, indentation and symbol patterns;
- A structured format accounts for a high proportion, which will increase the token density of the format.

If a token is very common in the tokenizer corpus, but rarely appears in the model training corpus, then although it enters the vocabulary, it may be basically not trained for embedding. Triggering such tokens at inference time may produce abnormal output.

---

## 12. Why do we still need regular pre-blocking?

Simple BPE allows any adjacent byte to merge across borders, which may form an unsatisfactory ultra-long token. The actual segmentator usually cuts the text into fragments with regular expressions first, and then executes BPE independently within each fragment:

```text
Original text
↓ Regular pre-blocking
Abbreviation | Alphabet string | Number string | Pontonation | Blank
↓ Each block executes BPE independently
token ID sequence
```

Pre-block rules can be controlled:

- Whether to put the leading space in the same part as the word;
- How to deal with the abbreviated form;
- Whether the capitalization shares similar rules;
- Numbers can be grouped by a maximum of several people;
- Can punctuation be merged with the alphabet across the border;
- Whether consecutive spaces can be merged.

It is essentially the a priori of adding "which boundaries can never be crossed" on top of BPE.

### 12.1 Why does the leading space often belong to token?

Many byte-level separators will combine spaces with the following word fragments:

```text
"Hello" and "hello"
```

Both may get a completely different token ID. This can efficiently represent the common "space + word" in natural language, but it will also cause:

- The words at the beginning of the sentence and the words in the sentence are different;
- Follow the space to change the subsequent tokenization;
- When truncated inside the token, the replenishment behavior becomes unstable.

### 12.2 Digital blocks

If the numbers are completely combined by statistics, the integers may be randomly split into 1-bit, 2-bit, 3-bit or longer fragments, which is not conducive to bit-by-bit operations. Modern rules often limit the length of digital blocks, making the numerical decomposition more regular.

### 12.3 Code and continuous space

The old rules may make each indented space an independent token, which wastes a lot of context. More reasonable rules allow multiple consecutive spaces to be merged, thus improving the code density.

---

## 13. Special token

Special tokens are used to express structures other than ordinary text, such as:

- The end of the document;
- The beginning and end of the sequence;
- The beginning and end of the message;
- User, assistant and tool roles;
- Fill the position;
- Fill the prefix, suffix and middle boundary of the task in the middle.

They usually do not go through the ordinary BPE, but are identified in advance by the encoder and directly replaced with reserved IDs:

```text
Ordinary text ──BPE──> Ordinary token
Special mark - direct mapping -> special token ID
```

### 13.1 Security boundary of special token

When the code cannot be entered untrusted, it must be clearly decided:

- Which special tokens are allowed to be parsed;
- Which strings can only be used as ordinary text;
- Whether the user input may forge the system boundary;
- Will it be interpreted by other components again after decoding?

If the user's controllable text is mistakenly identified as a control token, it may destroy the message structure, terminate the generation in advance or form an injection surface.

### 13.2 Adding special tokens requires model synchronous expansion

Assuming that the vocabulary is extended from $V$ to $V+k$, at least modified:

```text
Input embedding: (V, C) → (V+k, C)
Output LM Head: (C, V) → (C, V+k)
Output bias (if any): (V,) → (V+k,)
```

New rows are usually initialized randomly, so it is necessary to continue to train with the data containing these tokens, otherwise the model does not know their semantics and usage.

---

## 14. Two types of common implementation routes

### 14.1 Byte-level regular BPE

Typical process:

```text
Unicode string
  ↓ UTF-8
Byte
↓ Regular pre-blocking
Independent byte block
  ↓ BPE
token ID
```

Features:

- The original vocabulary is fixed to contain 256 bytes;
- Any text can be encoded;
- Logic is relatively unified;
- High-performance libraries are usually good at reasoning coding, but may not provide vocabulary training functions.

### 14.2 Code-level tokenization plus byte back

Another common route is to train BPE/Unigram directly on Unicode code points or text fragments, and provide:

- Character coverage control;
- Unknown token;
- Byte back;
- Virtual leading space;
- Rule of normalization;
- Integration of training and inference.

If you haven't seen a rare code point during training:

- When there is no byte backll, it will become a unified unknown token and lose the original information;
- After turning on byte back, it will be converted to UTF-8 byte token, and it can still be restored without loss.

Byte backto is usually more suitable for open world text, but the configuration items may affect each other and must be verified item by item.

### 14.3 Virtual leading space

Some implementations will automatically add a space at the beginning of each text, making it more likely that the first word and the word in the sentence use the same cut:

```text
"world"       → " world"
"Hello world" → keep "world"
```

This can reduce the word division caused by position, but it also changes the input specification, and the encoding and decoding must be handled consistently.

---

## 15. How to choose the size of the vocabulary?

The size of the vocabulary is not the bigger the better, but needs to be balanced.

### 15.1 Advantages of the big vocabulary

- Fewer tokens are required for the same text;
- The fixed context window can cover more original characters;
- Common words, phrases and code fragments can be directly represented;
- The number of reasoning steps may be reduced.

### 15.2 The cost of the big vocabulary

- token embedding parameter increase;
- The output LM Head parameter is increased;
- More categories of logits should be calculated at each step;
- The frequency of each token is reduced, and it is easier to under-train;
- Too long fragments are pressed into a token, and the character-level structure is more difficult to access;
- Increase the size of the model file and the memory usage.

Assuming that the embedding dimension is $C$, without considering the weight binding, the relevant parameters of the vocabulary are about:

$$
V\times C + C\times V = 2VC
$$

If the input embedding is shared with the output weight, some parameters can be reduced, but the number of categories of the output Softmax is still $V$.

### 15.3 The advantages and costs of the small vocabulary

The small vocabulary allows each token to get more training samples, and the input and output layers are lighter; but the sequence is longer, the attention cost is higher, and the context window will be exhausted faster.

### 15.4 Actual selection method

The size of the vocabulary is usually determined by experiment, and should be observed at the same time:

- Multilingual token/character ratio;
- Code token/character ratio;
- Structured data token/character ratio;
- Training throughput and GPU memory;
- Downstream task quality;
- The frequency of rare tokens;
- The ratio of embedding and output layer parameters.

---

## 16. How does the tokenization affect the ability of the model?

### 16.1 Spelling and character operation

If a long word as a whole is a single token, the model directly sees an integer, not an internal character. Therefore, the following tasks will become difficult:

- Count the number of occurrences of a letter;
- Reverse the string;
- Delete the characters in the specified position;
- Sort by characters.

Expressly disassembling strings into single characters with spaces can often improve performance, because each character is more likely to enter the context independently.

### 16.2 Integer operation

If the number is not divided irregularly:

```text
127 → a token
677 → Two tokens
804 → Another kind of two-segment combination
```

The bit-by-bit addition and subtraction method needs to cross the inconsistent token boundaries, and the model needs to learn a large number of additional combinations. More regular digital pre-blocks are usually more conducive to arithmetic.

### 16.3 Multilingual differences

If the tokenization training set is biased towards English, the same semantics may need several times token in other languages:

```text
Longer token sequence
  ↓
Exhaust the context window faster
  ↓
Pay attention to calculate more
  ↓
The cost of unit text is higher.
```

Therefore, multilingual ability depends not only on the model training data, but also on the tokenization training data.

### 16.4 Code Capability

The code contains a large number of:

- Continuous spaces;
- Line ninget and indenting;
- Punctuation combination;
- Common keywords;
- Repeat the grammatical structure.

If the indentation is split by space, the short program will also occupy a large number of tokens. Specially optimizing the word division rules of blanks and code patterns can significantly improve the effective context density.

### 16.5 Structured format

Different formats expressing the same data may produce different tokens. When choosing a format, in addition to readability and ecological compatibility, the token density should also be measured, not just comparing the number of characters.

---

## 17. Tailing space and unstable token

Many word splitters include the leading space in the following token:

```text
"Word" → a common token
```

If you put a space in advance at the end of the prompt:

```text
"... word "
```

This space may be encoded separately, and it is more common for it to form a token with the next word during model training. As a result, the input falls into a rare distribution, and the quality of the supplement may decrease.

More generally, if the string happens to be truncated inside a common token, continuing to complete cannot be simply understood as "adding the next token after the existing token sequence". There will be a boundary misalignment between character-level complementation and token-level complementation.

attention should be paid to in engineering:

- Avoid unintentional trailing spaces;
- Do not truncate UTF-8 or token at will in stream input;
- The complete interface needs to be specially handled with unstable prefixes;
- The cache key must be subject to the actual token sequence;
- Test all kinds of spaces, line breaks, tabs and some word prefixes.

---

## 18. The tokenizer is also a safe boundary.

tokenization errors not only affect the quality, but also the safety:

### 18.1 Control token injection

If the user text can be interpreted as a message boundary or an end mark, it may change the upper protocol structure.

### 18.2 Untrained token

If some tokens in the vocabulary never appear in model training, their embedding is almost random. Entering these tokens is equivalent to injecting untrained vectors into the network, and the behavior is unpredictable.

### 18.3 Normalization Differences

Visually similar characters, different Unicode combinations or invisible characters may get completely different token sequences. If security filtering is executed at the string layer, and the model works on another normalization result, it may cause bypass.

### 18.4 Inconsistency in encoding and decoding

If different components use different versions of vocabularies, regular rules or special token configurations, the same text will be interpreted as different IDs, causing protocol confusion.

Therefore, the production system should version the following content and release it as a whole:

```text
vocabulary + merge ranks + regular rules + special token + normalized rules
```

---

## 19. Add token and model surgery

When adding token to the trained model, you can:

1. Extended input embedding;
2. Extended output language model header;
3. Initialize new parameters;
4. Freeze or partially freeze the original model;
5. Use data containing new tokens to train new parameters;
6. Check whether the old token behavior returns.

In addition to structural control tokens, you can also learn a small number of "compressed tokens": freeze the main model, only optimize the embedding of new tokens, and make a few new tokens approximately replace a long fixed prompt. This shows that token can not only represent text fragments, but also serve as a learnable soft interface.

The same idea can also be extended to other modes: first compress images, audio or continuous signals into discrete token or continuous patch representations, and then hand them over to a unified sequence model for processing. The core model does not necessarily need to be changed. The changes mainly occur at the encoding and decoding ends of each modal.

---

## 20. The minimum available tokenizer class

```python
class BasicTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size):
        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break

            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id

        self.vocab = build_vocab(self.merges)

    def encode(self, text):
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(
                stats,
                key=lambda p: self.merges.get(p, float("inf")),
            )

            if pair not in self.merges:
                break

            ids = merge(ids, pair, self.merges[pair])

        return ids

    def decode(self, ids):
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")
```

This is the minimum realization for teaching. The production version also needs:

- Regular pre-blocking;
- Special token;
- File saving and loading;
- Strict version inspection;
- Faster training and coding algorithms;
- Paraly and cache;
- Complete error handling;
- Multilingual and malicious input testing.

---

## 21. Must do the test

### 21.1 Round-trip consistency

```python
assert tok.decode(tok.encode("hello")) == "hello"
assert tok.decode(tok.encode("Hello, café 🙂")) == "Hello, café 🙂"
assert tok.decode(tok.encode("")) == ""
```

### 21.2 External text of the training set

Not only test the training corpus, but also cover:

- Unseen language;
- Rare Unicode;
- Expressions and combined characters;
- Code;
- Long blank;
- Tables and line breaks;
- Binary control characters;
- Empty strings and single characters.

### 21.3 Merge Priority

Construct an example of a variety of mergeable paths, and confirm that the code always chooses the rule of the smallest rank.

### 21.4 Illegal token sequence

Confirm that the invalid decoding of UTF-8 will not cause the service to crash, and clearly adopt a strict error or replacement strategy.

### 21.5 Save and load

Reload after serialization, and you must get exactly the same:

```text
Encode results
Decode results
merge rank
Special token ID
```

### 21.6 Align with the goal

If you want to reuse the existing model, you can not only guarantee that "the text can be restored without loss", but also be exactly the same token by token. As long as the same text produces a different ID, the embedding of the original model cannot be used directly.

---

## 22. Common Errors

### 22.1 Treat Unicode code points as UTF-8 bytes

The code point may be much greater than 255, and the byte must be `0-255`.

### 22.2 Use different rules for training and coding

If there is regular blocks during training, none when coding, or if the special token configuration is inconsistent, it will produce wrong results.

### 22.3 Select the highest frequency pair when encoding

The highest frequency is only used in the training stage. Reasoning coding should choose the available pair with the smallest training rank.

### 22.4 Disturb the merges order

Later mergers may depend on early tokens, and changes in the order will change the meaning of the vocabulary.

### 22.5 Ignore empty input

Empty string and single byte input without pair, must be returned in advance.

### 22.6 Assuming that any token can be decoded independently

A token may only be a part of a multi-byte character, and decoding alone will fail.

### 22.7 Measure the compression rate only in English

Good performance in English does not mean that Chinese, code, mathematics and structured data are also efficient.

### 22.8 Don't train the model after adding token

The extended matrix can only match dimensions, and randomly initialized token vectors do not automatically have semantics.

### 22.9 Allows users to trigger special tokens at will

This may destroy the role boundary, document boundary or termination logic.

---

## 23. Do-it-yourself to realize the route

It is recommended to advance in the following order:

1. Convert the string to a UTF-8 byte list;
2. Realize `get_stats()`;
3. Realize single `merge()`;
4. Cycle training the vocabulary of specified size;
5. Build vocab according to merges;
6. Realize `decode()`;
7. Implement `encode()` according to merge rank;
8. Add empty strings and illegal UTF-8 tests;
9. Save and load merges and vocab;
10. Add regular pre-separation blocks;
11. Join the special token;
12. Compare with the target tokenizer by ID;
13. Measure the token density of multilingual, code and structured formats;
14. Then consider performance optimization.

Each step should be kept runnable and testable, and algorithms, performance, special tokens and file formats should be avoided at the same time.

---

## 24. Review and test yourself

1. What is the difference between Unicode code points and UTF-8 bytes?
2. Why does the original byte vocabulary only have 256 tokens, but it is still not suitable for direct training standard Transformer?
3. What three things did a training iteration of BPE do?
4. Why can the token created later rely on the token created first?
5. Why choose the highest frequency pair when training BPE, but choose the smallest rank pair when coding?
6. Why should `decode(encode(text))` remain unchanged, but the opposite direction is not necessarily true?
7. How does regular pre-block limit the merged boundary of BPE?
8. Why may the leading space become a part of the token?
9. Why can the increase in the list of words not only expand the effective context, but also increase the cost of the model?
10. Why does the word division training corpus affect multilingual ability and code ability?
11. Why does special token need to be processed independently of ordinary BPE?
12. When adding token to the trained model, which parameter matrix must be extended?
13. Under what circumstances will there be almost no trained token embedding?
14. Why does the tailing space may make the complement fall into a rare distribution?
15. What components need to be versionized for a reusable production separator?

---

## 25. The ultimate mental model

The GPT tokenizer can be compressed into the following five layers:

```text
Level 1: Unicode defines the characters seen by humans
Layer 2: UTF-8 converts characters into bytes stably
Layer 3: BPE compresses high-frequency adjacent bytes into token
Layer 4: Regular and special tokens add structural rules to the compression process
Layer 5: The model only sees the final token ID and learns the relationship between them.
```

The tokenization is not the cleaning of irrelevant data, but the "basic atom" that determines the model. It affects how much calculation a piece of text needs, how far the model can see, which patterns are easy to learn, which character tasks are difficult, and which inputs may be safe boundaries.

A reliable tokenizer needs to meet four goals at the same time:

```text
Complete: Any text can be coded
Reversible: Normal text can be restored without loss
High efficiency: use as few tokens as possible for common data
Consistency: Training, reasoning and all components use exactly the same rules
```

The core links that really need to be mastered are: **Unicode → UTF-8 bytes → BPE merges → token ID → embedding**. After understanding this link, many strange behaviors that seem to come from the model itself can be explained from the word boundaries, vocabulary statistics and training distribution.

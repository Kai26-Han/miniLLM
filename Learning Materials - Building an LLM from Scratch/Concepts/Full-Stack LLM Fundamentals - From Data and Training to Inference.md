# Full-Stack LLM Fundamentals: From Data and Training to Inference

## Core Idea

An LLM first converts large amounts of text into tokens and learns patterns of language and the world through next-token prediction. Supervised fine-tuning then teaches it to respond as an assistant, while reinforcement learning on verifiable tasks can help it discover more effective reasoning strategies. At heart, however, it remains a probabilistic generation system: its capabilities are broad but uneven, its parameterized knowledge is approximate, and reliable applications must combine the model with context, tools, verification, and human oversight.

## Key points

1. **LLM training starts with curated data, not with “intelligence.”** Data sources, language mix, deduplication, and quality filters directly shape the model's capabilities.
2. **The model processes tokens rather than raw text.** Its boundaries between characters, words, and symbols may differ from the boundaries a person perceives.
3. **Pretraining centers on one objective: predicting the next token.** This simple objective forces the model to compress patterns in language, facts, and the world.
4. **A base model is an Internet-document simulator, not a natural chat assistant.** Its default behavior is to continue text that resembles its training corpus.
5. **Supervised fine-tuning programs behavior through examples.** Ideal answers, behavioral policies, and refusal patterns together shape the assistant's apparent “personality.”
6. **Knowledge in parameters resembles fuzzy long-term memory; the context window resembles working memory.** Put information that must be handled precisely directly into the context.
7. **Hallucinations arise from both the generation mechanism and the training data.** If demonstrations always sound confident, the model may imitate that confidence even when it does not know the answer.
8. **Models use generated tokens as computational workspace.** Each token permits only a limited amount of computation, so intermediate steps function like scratch paper.
9. **LLM intelligence is jagged.** A model may solve a difficult technical problem yet fail at counting, spelling, or comparing decimals.
10. **Reinforcement learning moves the model from imitation toward practice.** The model generates candidate solutions, verifies outcomes, and increases the probability of successful trajectories.
11. **Verifiable rewards differ from human-preference rewards.** Verifiers are generally harder to game, while learned reward models can be exploited by the optimization process.
12. **A reliable LLM application is more than “ask a question and copy the answer.”** It supplies context, lets the model use tools, verifies the results, and keeps a human accountable for consequential decisions.

---

## Overview of the Full-Stack Pipeline

```text
Original Internet data
      │
      ▼
Crawling, text extraction, quality filtering, language screening, deduplication, privacy cleaning
      │
      ▼
tokenizer: Text → One-dimensional token sequence
      │
      ▼
Pre-training: Predict the next token
      │
      ▼
base model: Internet Document Simulator
      │
      ▼
Supervision Fine-tuning (SFT): Learning dialogue, instruction compliance and code of conduct
      │
      ▼
assistant model: can answer questions, reject requests, and call tools
      │
├── Preference optimization / RLHF: more in line with human preferences
      │
└── Verifiable reinforcement learning: practice math, code and other tasks
                     │
                     ▼
Reasoning / Thinking Model
```

---

## Part 1: Pre-training - Where does knowledge come from?

### 1. The original data must be processed by the system.

The Internet crawls not a clean knowledge base, but a raw material full of HTML, navigation, advertising, junk content, duplicate pages and privacy information. Typical data pipelines include:

1. **Web page crawling**: Start from the seed page and continue to index the Internet along the link.
2. **URL and domain name filtering**: exclude low-value or high-risk sources such as malicious, junk, adult, marketing, etc.
3. **Text extraction**: Remove navigation, styles and page templates from HTML, and only keep meaningful text.
4. **Language recognition**: Judge the document language and adjust the proportion of different languages according to the target ability.
5. **Quality filtering**: Remove garbled code, templated content, low information density pages and abnormal text.
6. **deduplication**: Delete duplicate documents and highly similar fragments to avoid the model repeatedly learning a few contents.
7. **Privacy Cleanup**: Identify and remove personally identifiable information such as address and identity number as much as possible.

What the model finally learns depends not only on the amount of data, but also on the filtering criteria. The language ratio in the training data will affect multilingual ability. Source deviations will enter the model, and duplicate data may amplify memory and overfitting.

### 2. The size of the data is not equal to the volume of the disk.

After radical filtering of high-quality text, the disk volume may only be tens of TB, but after conversion, it will still form a training sequence of trillions of tokens. The core of the training cost is not "whether the file can be put on the hard disk", but how many times these tokens need to be calculated by the neural network, and how many parameters of the model need to be updated.

### 3. Tokenization: the words in the eyes of the model

Neural networks require a one-dimensional sequence composed of a finite set of symbols, so natural language must first be converted into token.

The bottom can start from binary bits or UTF-8 bytes, but the sequence is too long. The actual system often uses algorithms similar to Byte Pair Encoding (BPE):

1. Start with bytes or basic symbols;
2. Find out the symbolic pairs that often appear continuously in the data;
3. Merge the high-frequency combination into a new token;
4. Repeatedly merge until the size of the target vocabulary is reached.

There is a basic trade-off here:

| Scheme | vocabulary size | Sequence length | Features |
|---|---:|---:|---|
| Byte-level | Small | Long | General, but high computing cost |
| Sub-word level | Medium | Medium | Current common compromise plan |
| Large word block | Large | Shorter | The vocabulary and output layer are larger, and the low-frequency combination utilization rate is low |

token is not a strict word: it may be a character, a root, a complete word, a space plus a word, or even a punctuation mark. Capitalization, leading spaces and spelling changes may produce completely different token sequences.

### 4. Why is token so important?

Tokenization is not just an input format, it will change the cognitive difficulty of the model:

- Common words may only account for one token, and rare words may be disassembled into multiple tokens;
- What the model sees is the token ID, not the characters seen by the human eye;
- The compression efficiency of different languages may be different, and the number of tokens required for the same content is also different;
- Character-level spelling, word-by-word counting and string processing are often unnatural to the model;
- The capacity of Context window is calculated by token, not by the number of words or pages.

Therefore, token is the first key to understanding LLM costs, contextual limitations and cognitive defects.

---

## Part 2: How does the neural network learn the next token?

### 1. Input and output

Intercept a token from the training corpus as a context, and the neural network outputs the probability that each token in the entire vocabulary becomes the next item.

Assuming that the vocabulary has 100,000 tokens, the network will output about 100,000 fractions. Because the real next token in the training data is known, it can:

1. Improve the probability of the correct token;
2. Reduce the relative probability of other tokens;
3. Fine-tune all parameters through reverse propagation and gradient descent;
4. Repeat this process on massive windows and tokens.

The result of the training is that the probability distribution given by the model is gradually close to the real statistical law in the training text.

### 2. What is Transformer doing inside?

Transformer is usually composed of multi-layer repeated modules, the core of which includes:

- **embedding**: Map the discrete token ID into a continuous vector;
- **attention**: Let the current location read the relevant token information in the context;
- **MLP**: Nonlinear transformation of the representation of each position;
- **Residual connection and normalization**: Maintain the training stability of the deep network;
- **Output layer**: The probability of mapping the internal representation to the next token.

The mathematical operation of each component is certain, but how billions of parameters work together to form facts, language styles and reasoning strategies cannot be fully explained like traditional programs.

### 3. The difference between training and inference

| Stage | Behavior | Whether to update parameters |
|---|---|---|
| Training | Read the real sequence, calculate the prediction error, and adjust the weight | Yes |
| Reasoning | Read the existing context, predict and sample the next token | No |

The cycle of reasoning is:

```text
Existing token → neural network → probability distribution of the next token
                         ↓
Select or sample a token
                         ↓
Add to the context and continue the loop
```

### 4. Why do the same question get different answers?

The model output is the probability distribution, not the only answer. If sampled from the distribution, each generation may take a different path.

- Lower temperature makes the distribution more concentrated and the output more stable;
- Higher temperature increases diversity and also increases the chance of deviating from high-probability answers;
- Once the early token is different, the subsequent context is also different, and the whole output will gradually fork.

This shows that the model is naturally a random system. Certainty can be enhanced by decoding strategies, but probability models cannot be turned into fact databases.

### 5. What exactly is the base model?

The base model obtained after the pre-training is completed can be regarded as an "Internet document simulator". Give it the beginning of a web page, code, forum or news style, and it will continue to generate content that conforms to the statistical rules of this type.

It has a lot of knowledge, but it does not have a natural user-assistant interaction goal. Directly give the base model a question, which may answer, or continue to write more questions, generate web page formats or simulate a dialogue. It accomplishes the continuation, not "helping users".

---

## Part 3: Post-training - How to turn the base model into an assistant?

### 1. The dialogue must also become a one-dimensional token sequence.

The dialogues seen by human beings are structured objects: systems, users, assistants, and multiple rounds of messages. The model finally sees a one-dimensional token sequence, so special tokens need to be added to mark:

- Where does a round of news start and end;
- The current content comes from the system, the user or the assistant;
- Where are the tool calls and tool return values;
- When should the model stop being generated?

The dialogue protocol of different models may be different, but the essence is to encode the structure into a linear sequence, and then continue to carry out next-token prediction.

### 2. Supervision fine-tuning (SFT) is "programming with examples"

Supervision fine-tuning does not need to change the core training algorithm, just change the training data from Internet documents to high-quality dialogues:

```text
User: Ask questions or tasks
assistant: Give an ideal answer
```

After the data covers enough tasks, the model gradually learns:

- Follow the instructions;
- Use a clear answer format;
- Maintain multiple rounds of context;
- Refuse in the appropriate scene;
- Show a "helpful, real and harmless" behavior style;
- Call the tool according to the specific protocol.

This is a kind of implicit programming. The rules are not completely written in the traditional code, but enter the parameters statistically through samples.

### 3. Where does the assistant's "personality" come from?

The training team will formulate detailed labeling specifications to explain what the ideal answer is. The annotators then create or revise dialogues according to these specifications. The model learns the statistical patterns of these answers.

Therefore, the ordinary assistant model can be understood as:

> A system for rapid neural network simulation of "professional labelers who follow the code of conduct".

This mental model can dispel the sense of mystery. The output contains not only the extensive knowledge obtained before training, but also the expression, value orientation and behavior boundaries given by the post-training data.

### 4. From artificial data to synthetic data

Early dialogue data was mainly written by people from scratch, and modern data pipelines use a large number of models to assist:

- Model generation candidate answers, human revision;
- Strong model generation training samples, smaller model learning;
- Multiple models check, rewrite or score each other;
- Humans are responsible for setting standards, sampling, correcting errors and covering key boundary scenarios.

Data has changed from "pure manual production" to "model generation + human planning and supervision". However, the final capability is still subject to data design and quality control.

---

## Part 4: Cognitive characteristics and sharp edges of LLM

### 1. Hallucination is not an accidental failure, but a natural result of the generation mechanism.

If there is almost always a confident and complete answer to the question of "who is someone" in the training data, the model will learn this form. When the user asks a person who does not exist or the model does not know, it may still generate the most statistically answer-like content.

There may already be "uncertain" signals inside the model, but if the training data does not connect the uncertainty with the output of "I don't know", it will still give the most likely guess.

#### Two paths to reduce hallucinations

The first article is **the expression of the calibration model's own knowledge boundaries**:

1. Generate factual questions and correct answers from known information;
2. Multiple sampling model answers to the same question;
3. Judge whether the model is stable and know the answer;
4. For questions that you don't know, take "I don't know" as the correct training sample;
5. Let the model learn to refuse to guess when the internal uncertainty is high.

The second article is **allow model retrieval**:

1. The model generates a search request or tool call;
2. The outer program suspends generation and performs search;
3. Write the search results into the context window;
4. The model is based on the information just obtained and the source is cited.

The first method makes the model more honest, and the second method adds facts to the model.

### 2. Parameter memory and working memory

| Information Location | Analogy | Characteristics |
|---|---|---|
| Model parameters | Vague long-term memory | Large capacity, fast call, but not accurate, with cut-off time |
| Context window | Current working memory | Direct access, more reliable processing, but limited capacity |
| External tools and databases | Queryable database | More accurate and updatable, but requires retrieval and permission control |

In practice, if the task depends on an article, contract, data table or code, it is best to provide the relevant content directly to the model or use the retrieval system, instead of requiring the model to process it "by memory".

### 3. The model does not have a natural and persistent "self"

A reasoning is usually: loading parameters, processing the current token, generating output, and then ending. The next session starts again with the parameters and the new context.

The description of "who am I" in the model usually comes from:

- Samples of identity questions and answers in supervision fine-tuning;
- The name, developer and capability boundary written in the system message;
- The most common assistant identity mode in Internet data;
- The hint of the current dialogue context.

If there is no clear configuration, the model may make statistical guesses about its own identity. Therefore, the self-statement of the model cannot be automatically regarded as a systematic fact.

### 4. The model needs token to think.

Transformer performs a forward calculation with a finite depth for each new token. It cannot complete any complex operation in a token.

If the answer requires the final number at the beginning, the model must press all the reasoning into a forward calculation that produces the number; the following explanation is only to rationalize the answer that has appeared after the fact.

On the contrary, Mr. Cheng's intermediate step can spread the calculation to multiple tokens:

```text
Disassemble the problem
  ↓
Generate intermediate results
  ↓
Keep the intermediate results in the context
  ↓
Continue the next small step calculation
  ↓
Get the final answer and review it
```

These intermediate tokens are mainly the calculation drafts of the model, not just an explanation for people to see.

### 5. The reasoning process is still not as reliable as the tool.

It is usually better to let the model write the mental calculation process step by step than to give the answer immediately, but the neural network may still miscalculate in any step. Precise tasks should give priority to calling deterministic tools:

- Arithmetic, Statistics: Calculator or Python;
- Data filtering: SQL or data processing program;
- The latest facts: search and database;
- Code correctness: running test, type checking, static analysis;
- Character processing: string functions or regular expressions.

The best model is not to "let the model guess harder", but to let the model be responsible for planning and interpretation, and let the reliable tool be responsible for calculation and verification.

### 6. Tokenization leads to spelling and counting defects

The model does not see the characters directly, so tasks such as "a word has several letters", "take every three characters" and "inverte string" are not as intuitive as for humans.

The difficulty comes from two layers:

1. Words are compressed into a few tokens, and the character boundaries cannot be directly accessed;
2. The count itself needs to be completed in a limited single-step calculation.

The solution is still instrumental: let the model hand over the string to the code for execution, instead of relying on neural network mental calculation.

### 7. Serrated Intelligence / Swiss Cheese Capability Model

The ability of LLM does not increase smoothly from simple to difficult. It may:

- Answer professional scientific questions;
- Complete the complex code design;
- Solve competitive math problems;
- At the same time, it fails on simple decimal comparison, character counting or abnormal formatting.

This ability distribution is like Swiss cheese: the whole is large, but there are some random holes. It is difficult for users to judge that the model must be reliable just by the task "looks simple".

Therefore, LLM must be regarded as a random tool, not an omniscient system.

---

## Part 5: reinforcement learning - From Imitation expert to Independent Practice

### 1. The analogy between three-stage training and school education

| Model Training | School Learning | What to Learn |
|---|---|---|
| Pre-training | Read the text of the textbook | Background knowledge and language laws |
| Supervise fine-tuning | Reading expert problem-solving process | How to organize answers and imitate expert methods |
| reinforcement learning | Complete the exercises independently and check the answers | Find the solution that suits you through trial and error |

Pre-training makes the model "know a lot", SFT makes the model "look like it can do it", and reinforcement learning allows the model to improve the success rate of real problem solving in a large number of exercises.

### 2. Why is it not enough to imitate human solutions?

The cognitive structures of human beings and models are different:

- A simple step for people may require the model to complete too many calculations in a single token;
- The explanations written by human beings may not make full use of the knowledge already mastered by the model;
- Some necessary steps for people may only be redundant for the model;
- The annotator cannot know which token path is the most suitable for the model.

SFT can initialize the model to the area of "close to the correct solution", but the better reasoning trajectory needs the model to find it by itself through trial and error.

### 3. The basic process of reinforcement learning can be verified

```text
Given questions and verifiable answers
        │
        ▼
The model randomly generates a large number of candidate solutions (rollouts)
        │
        ▼
Automatically check the final result of each solution
        │
├── Wrong path: reduce the probability of similar behavior
└── Path to Success: Increase the probability of similar behaviors
        │
        ▼
Repeatedly update the model on a large number of problems
```

The key point is that the success trajectory comes from the model itself, not the manually written expert answers. Therefore, the model can find internal strategies that are different from humans but statistically more reliable.

### 4. How did the reasoning strategy emerge?

When the reward only focuses on the final correctness, the model may gradually learn:

- Check the previous steps;
- Retreat after finding the contradiction;
- Resolve from another representation;
- Try multiple plans;
- Cross-verification before submitting the answer;
- Use more tokens in exchange for higher accuracy.

These behaviors of "wait a minute", "recheck" and "change the method" do not need to be written into the training sample one by one, but may appear naturally from the optimization process to improve the success rate.

This is the important difference between the reasoning model and the ordinary assistant model: the ordinary assistant mainly simulates the answers written by experts, and the reasoning model also optimizes the calculation process before generating the answer after a large number of verifiable exercises.

### 5. Why may reinforcement learning exceed imitation?

The upper limit of imitation learning is limited by the model. Reinforcement learning only cares about what behaviors can get the right results, so we can leave the common human strategies and explore new paths.

In a closed environment with clear rules and automatic determination of results, this is the most obvious: the system can repeatedly practice itself without being limited by the size of manual data, and find low-probability but effective strategies.

Extending this idea to the language model requires the establishment of a large number of "practice environments": math problems, code tasks, formal proofs, executable workflows, and other tasks that can automatically determine success.

### 6. Why is the verifiable field so important?

The ideal reward function should:

- Cheap;
- Automation;
- Objective;
- Difficult to deceive;
- Able to make high-frequency calls.

The final answer of mathematics, code test results, game victory or defeat, etc. meet these requirements. The system can be optimized for a long time, because "correct is correct", and it is difficult to deceive the judge through wording.

---

## Part 6: RLHF - Preferred Learning in Unverifiable Tasks

### 1. There is no single correct answer for creative tasks.

Jokes, poetry, abstracts, writing style and general help cannot be judged by simple programs. It is too expensive for humans to evaluate each reinforcement learning sampling, so a human preference agent that can be called repeatedly is needed.

### 2. How does the reward model arise?

The typical process is:

1. Generate multiple candidate answers for the same prompt;
2. Let humans sort the candidate answers from good to bad;
3. Train an independent reward model to make its scoring order close to human ranking;
4. Use the reward model to replace humans and quickly score a large number of new answers;
5. Let the language model improve the reward score through reinforcement learning.

The reward model is not an assistant to generate text, but a scorer that receives "tips + candidate answers" and outputs a single preference score.

### 3. Advantages of RLHF

- Extend reinforcement learning to tasks without standard answers;
- Humans only need to compare, and don't have to write the ideal answer by themselves;
- It is usually easier to judge "which one is better" than to create from scratch;
- It can improve the answer style, helpfulness and overall preference performance.

### 4. The fundamental limitations of the reward model

The reward model is just a destructive simulation of human judgment, not a real human being, nor a truth function. Large-scale optimization may find loopholes in the reward model: some seemingly meaningless outputs are given high scores by the error of the reward model.

This is called reward hacking or confrontational use. The common rules are:

```text
Optimization in the early stage: the quality of answers improves with rewards
        ↓
Over-optimization: the model began to look for the scorer loophole
        ↓
The reward continues to rise, but the real quality declines.
```

The discovered vulnerabilities can be added to the training data for repair, but complex neural networks usually hide more confrontation samples. Therefore, reinforcement learning based on the reward model cannot run indefinitely like games with clear rules.

The conclusion is that RLHF is very useful, but it is more like a limited preference fine-tuning, which is not equivalent to long-term reinforcement learning in a verifiable environment.

---

## Part 7: In which direction will the ability develop in the future?

### 1. Native multimodal

Text, sound and images can be encoded into token streams:

- Audio can extract discrete representations from spectrum fragments;
- Images can be split into patches and encoded;
- Text, image and audio token can be staggered into the same context.

The unified token interface enables the model to read and write text, listen to voice, understand and generate images at the same time without having to completely cut off each mode.

### 2. From a single answer to a long-term Agent

Ordinary dialogues usually give a clear task directly to the model. Further systems need:

- Disassemble the target by yourself;
- Perform multiple steps continuously;
- Call browsers, codes and external applications;
- Check the results and recover from errors;
- Manage the status of long-term operation;
- Report to people and request decision-making at key nodes.

As the duration of the task increases, the role of people will shift from step-by-step operators to supervisors. But the zigzaged ability of the model means that long tasks cannot be simply unattended.

### 3. AI will enter the existing tools more invisibly.

LLM does not always appear as a separate chat box. It can become an intention understanding layer in editors, browsers, office software, terminals and business systems, and further operate keyboards, mice, APIs and enterprise processes.

The closer the ability is to real operation, the more important permission control, audit, confirmation and error recovery are.

### 4. Context window is not long-term learning

After the model is deployed, the parameters are usually fixed. The "learning" that occurs in a session is mainly to put new information into the context window; after the session, the information will not be automatically written back to the parameters.

Simply extending context window cannot solve long-term tasks indefinitely:

- Context is a limited and expensive resource;
- Long-term multimodal tasks will generate a large token flow;
- Useful information will be drowned out by noise;
- The model requires a more lasting memory, compression, retrieval or testing learning mechanism.

Therefore, the future system needs not only "longer context", but also effective long-term memory and online adaptation methods.

---

## Part 8: How to use LLM more reliably?

### 1. First, judge where the information should come from.

| Task | Preferred source of information |
|---|---|
| Common sense, wording, creative draft | Model parameter memory |
| Specify articles, contracts, code libraries | Directly provide context or RAG |
| Latest facts, prices, regulations | Search, database, authoritative sources |
| Mathematics, Statistics, Data Processing | Calculator, Python, SQL |
| Executable conclusion | Actual operation, testing or manual verification |

### 2. Provide sufficient computing space for the model

Don't force the model to "only give answers, not explanations" for complex tasks. The better requirements are:

- Disassemble the problem first;
- List the necessary assumptions;
- Generate and check the intermediate results;
- Use tools when necessary;
- Finally, give a concise conclusion separately.

The user interface can only display the summary, but the model should be allowed to perform sufficient calculation and verification inside the system.

### 3. Clearly require the use of tools

When the task involves precise calculation, character processing, real-time information or a large amount of data, it is usually more effective to directly require the model to use the corresponding tools than to let it "think carefully again".

### 4. Don't judge the correctness by tone.

The confident tone of the model comes from the training style, not the confidence meter. Reliability should come from:

- Traceable source;
- Repeatable calculation;
- Test results;
- Multi-path cross-verification;
- Professional review.

### 5. Choose the model mode suitable for the task

| Task Type | Suitable Mode |
|---|---|
| Simple Q&A, Rewriting, Summary | Quick assistant Model |
| Complex mathematics, code, planning | Reasoning model or higher reasoning budget |
| Private and scale-controllable tasks | Local open weight model |
| Need the latest information | Models with search or data connection capabilities |

The reasoning budget should be matched with the difficulty of the task. Simple facts do not need to be thought about for a long time, and complex questions should not be answered instantly.

### 6. The ultimate responsibility is still with people.

Use the model for:

- Provide inspiration;
- Generate the first version;
- Analyze the candidates;
- Check for omissions and fill in the gaps;
- Accelerate the duplication of work.

But don't take the unchecked output directly as the final product. The model can significantly improve efficiency, but it cannot replace the person responsible for the results.

---

## Common misunderstandings

### Misconception 1: The model stores the original text of the Internet completely into the parameters

Parameters are more like lossy compression and statistical reconstruction. The model may remember some high-frequency text, but it is not a document database that can be read accurately as a whole.

### Misconception 2: The model can be explained fluently, so it must be understood and calculated correctly.

Fluency comes from language modeling, and interpretation may be the post-rationalization of wrong conclusions. The correctness requires external verification.

### Misconception 3: The simpler the task, the more unlikely the model is to make mistakes.

The ability of LLM is jagged. Simple character, counting and abnormal format tasks for people may be located in the ability hole.

### Misconception 4: A longer answer must represent stronger reasoning.

It is useful to improve the accuracy of search, retretback and verification, not simply to increase the length of the text.

### Misconception 5: The boundaries of identity and ability said by the assistant come from real self-awareness

This information is often configured by training samples and system messages, not the natural and lasting self of the model.

### Misconception 6: RLHF can infinitely improve the model

RLHF optimizes the reward model that can be deceived. Excessive optimization will lead to an increase in rewards and a decrease in real quality.

### Misconception 7: As long as the model is strong enough, there is no need for tools.

The tool is not a temporary patch when the model capability is insufficient, but the core composition of a reliable system. Search, code and database provide accuracy and timeliness that parameter memory cannot guarantee.

---

## Key Conceptual Relationship Chart

```text
Training data determines what the model has seen.
          │
          ▼
tokenizer decides how the model reads the text.
          │
          ▼
Pre-training determines the basic knowledge and generation ability.
          │
          ▼
SFT determines the assistant's behavior, format and personality
          │
          ├──────────────┐
          ▼              ▼
Verifiable RL Preference Learning / RLHF
Practice the correct solution and learn human preference style
          │              │
          ▼              ▼
Reasoning strategies emerge, and reward models may be used.
          └──────┬───────┘
                 ▼
Deployed LLM system
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
Context External Tools Manual Supervision
Working memory, precise execution, ultimate responsibility
```

---

## Review questions

1. Why can't Internet crawling data be directly used for pre-training?
2. What trade-off did tokenizer do between the size of the vocabulary and the length of the sequence?
3. How can Next-token prediction be transformed into language and knowledge ability?
4. Why is the base model not a natural chat assistant?
5. How can the dialogue be coded into a form that the model can handle?
6. Why is SFT "programmed with examples"?
7. What does hallucination have to do with the answer style of post-training data?
8. What is the difference between parameter knowledge and the information in context window?
9. Why does the model need an intermediate token to complete complex reasoning?
10. Why does Tokenization cause spelling and counting problems?
11. What does "jagged intelligence" reveal to the evaluation and usage?
12. What role do SFT and reinforcement learning play in imitation and practice respectively?
13. Why is the verifiable reward more suitable for long-term reinforcement learning than the reward model?
14. Why does reward hacking occur in RLHF?
15. Why can't long context window be directly equivalent to long-term memory?

---

## Final Cognitive Model

When understanding an LLM system, you can ask six questions in a row:

1. **Where did its data come from and how was it filtered?**
2. **How does it convert input into token?**
3. **What knowledge and abilities does the pre-training make it gain?**
4. **What kind of assistant did the post-training make it?**
5. **Does it have verified reinforcement learning, and can it invest more in reasoning calculations for difficult tasks?**
6. **What tools can it call, how to verify the results, and who is responsible in the end?**

These six layers together determine the ability, style, cost, reliability and risk of the model. If we understand them separately, we will not mistake fluent language for absolute knowledge, nor will we ignore its great value as a cognitive tool because of the flaws in the model.

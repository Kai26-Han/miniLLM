# Introduction to LLMs: From a "Compressed Internet" to a New Operating System

## What an LLM Really Is

At its core, a large language model is a neural network trained to predict the next token. Pretraining compresses patterns from vast amounts of Internet text into the model's parameters, while fine-tuning shapes the model into an assistant. Once connected to search, calculators, code execution, image models, and speech systems, an LLM begins to resemble a new kind of operating system with natural language as its interface. That expanded capability also creates new safety risks, including jailbreaks, prompt injection, and data poisoning.

## Key points

1. **A runnable LLM can be reduced conceptually to two components: parameters and inference code.** The code defines the network, while the parameters carry the capabilities and knowledge acquired during training.
2. **The central pretraining task is next-token prediction.** Although the objective looks simple, succeeding at it requires the model to learn grammar, facts, styles, and many patterns about the world.
3. **Model parameters act like a lossy compression of the Internet.** They store statistical patterns and approximate knowledge, not a database that can be queried exactly.
4. **A base model is not yet a chat assistant.** It behaves more like an Internet-document simulator until fine-tuning on high-quality conversations teaches it to answer as an assistant.
5. **Fine-tuning primarily changes behavior and presentation.** It does not turn a model into a perfectly reliable factual database, and assistant-like behavior alone cannot eliminate hallucinations.
6. **An LLM system's capabilities come from both the model and its tools.** Search, calculators, Python, retrieval systems, and image models can supplement memory, arithmetic, and execution.
7. **A useful long-term analogy is an “LLM operating system.”** The LLM is the kernel, the context window is working memory, external data is disk or network storage, and tools behave like peripherals and system calls.
8. **Greater capability creates a larger attack surface.** Multilingual and multimodal inputs, network access, and file access increase the risks of jailbreaks, prompt injection, privacy leaks, and data poisoning.

---

## Part 1: What Exactly Is an LLM?

### 1. Two files: parameters and running code

Using Llama 2 70B as an example, we can simplify an LLM into:

- **Parameter file**: 70 billion parameters occupy about 140 GB when stored as 16-bit floating-point values.
- **Inference code**: the program that performs the Transformer's forward pass. Conceptually, a minimal implementation can fit in a few hundred lines of C.

Put the two on a strong enough local computer. Even if you are not connected to the Internet, you can enter the text and let the model continue to generate. There are two things to distinguish here:

| Stage | What to do | Main features |
|---|---|---|
| Training | Find parameters from a large amount of data | Expensive, time-consuming, relying on large-scale GPU clusters |
| Inference | Generate content one by token with existing parameters | Relatively cheap, can be run locally or on the server |

The network structure and mathematical operations of the model are not mysterious. The real cohesion ability is the huge number of parameters obtained by training.

### 2. Pre-training: compress the Internet into parameters

Take Llama 2 70B as an example, its pre-training is roughly: collecting about 10 TB of text, using about 6,000 GPUs for about 12 days, costing about $2 million, and finally getting about 140 GB of parameters. This set of data is only used to help understand the scale of training, and does not represent the scale and cost of the current cutting-edge model.

The result can be thought of as an "Internet compressed package", but it is fundamentally different from ordinary ZIP:

- ZIP is lossless compression, which can restore the original file;
- LLM parameters are loss-compressed, which only retains the statistical structure, association and overall form in the data;
- Therefore, the model will remember some content, summarize some rules, and mix reasonable forms with wrong details.

This metaphor explains why LLM can not only show extensive knowledge, but also make up facts seriously.

### 3. Why can "predict the next token" produce intelligence?

During training, the model sees a token sequence, and then predicts the probability distribution of the next item. After selecting a token, put it back into the input and continue to predict the next item, so as to form a complete text.

To accurately predict the next token, the model cannot only learn the word frequency. For example, to continue writing a character introduction, it needs to implicitly grasp the characters, eras, events, language habits and context relationships. With the growth of training data and parameters, these knowledge and patterns are distributed and coded into neural networks.

It can be understood by the following causal chain:

```text
Predict the next token more accurately
        ↓
It is necessary to identify the rules and context in the text.
        ↓
Language, facts and world patterns must be compressed.
        ↓
Migratable knowledge and ability appear in the parameters
```

### 4. Is the model "remembering" or "dreaming"?

The base model is sampled from the distribution of Internet documents, so it is very similar to "dreaming of web pages": it may generate code, product pages or encyclopedia articles. The format is usually very realistic, but the title, author, ISBN or specific facts may not exist at all.

Some outputs come from the memory of training data, some are generalizations of knowledge, and some are just statistically reasonable complements. Users usually can't judge which one it belongs to from the wording itself, so:

> Fluency is not equal to correctness, and self-confidence is not evidence of reliability.

### 5. We know the structure, but we don't fully understand the internal mechanism.

Each step of Transformer's mathematical operation is known, and the training algorithm can also continuously reduce the prediction error; however, it is still difficult to fully explain how billions of parameters work together to form specific knowledge, reasoning and behavior.

LLM can be regarded as an "empirical product" mainly based on experimental knowledge: we give it input, observe the output, statistical success rate, and then judge whether it is suitable for a task through evaluation. Mechanical interpretability research is trying to open this black box, but at present, it cannot fully explain the function of each internal component as it does to check the traditional program.

This means that the actual application cannot only look at a few demonstrations, but must establish an evaluation set covering the real scene.

---

## Part 2: How can the base model become a chat assistant?

### 1. Pre-training is responsible for "knowledge", and fine-tuning is responsible for "behavior"

After pre-training, the base model is obtained. It learns to continue writing Internet documents, but it does not naturally know that users want it to answer questions. To turn it into an assistant, you need to replace the training data with high-quality dialogue:

```text
User: Ask questions or tasks
assistant: Give an ideal answer that meets the norms
```

The training goal is still to predict the next token, and the change is the data distribution. From this, the model learns to follow instructions, use the question-and-answer format, and be helpful.

The two stages can be distinguished as follows:

| Dimension | Pre-training | Supervision Fine-tuning (SFT) |
|---|---|---|
| Data | Massive Internet text | A small amount but high-quality manual or machine-assisted dialogue |
| Objectives | Obtain general knowledge and language ability | Learn to follow instructions, answer styles and behavior boundaries |
| Cost | Extremely high, slow iteration | Relatively low, can be iterated frequently |
| Product | base model | assistant Model |

Fine-tuning is not to load a reliable database into the model, but to guide the original "web dream" into a "helpful assistant dream". Therefore, it may still hallucinate.

### 2. Reverse improvement from failure cases

After the assistant is deployed, it is necessary to continuously collect failure cases:

1. Find the bad or wrong answers of the model;
2. Let the labeler write down the ideal answer;
3. Add the new sample to the fine-tuning data;
4. Re-training, re-evaluation, re-deployment.

This is a continuous data closed loop. Many product differences come not only from the size of the base model, but also from label specifications, data quality, evaluation system and iteration speed.

### 3. RLHF: It is usually easier to compare answers than to write answers from scratch.

In addition to letting people write the ideal answer directly, you can also ask the model to make multiple candidate answers, and then let the annotator sort it. For many tasks, it is easier for humans to judge "which one is better" than to write the best answer in person.

These preference comparisons can be used to enhance learning human feedback (RLHF) to further enhance the usefulness, authenticity and harmlessness of the model. Data production is also shifting from pure manual to human-computer collaboration: models are first drafted, checked or compared, and humans are responsible for screening, revision and supervision.

---

## Part 3: Why will LLM continue to get stronger?

### 1. Scaling Laws: Scale brings predictable benefits

From the empirical law, the predictive performance of the model mainly changes smoothly with two variables:

- `N`: Number of model parameters;
- `D`: Training data volume.

Increasing the model scale and training data will usually steadily improve the loss of next-token prediction; and this improvement is related to a number of downstream capabilities. This predictability has promoted the "gold rush" of computing power, data and GPU clusters.

It should be noted that scale is an important path, but it does not unconditionally guarantee that all real tasks will be improved proportionally. Data quality, training methods, reasoning resources, tool systems and evaluation design are equally key.

### 2. Tool call: Don't just let the model "calculate in the mind"

The parameter memory of LLM is fuzzy, the mathematical calculation is unstable, and the knowledge is still deadline. The solution is to let it select tools by task:

| Limitations | Callable Tools | Function |
|---|---|---|
| Don't know the latest facts | Browser, search, database | Retrieve the latest and citable information |
| Arithmetic is not reliable | Calculator | Get accurate results |
| Complex data analysis and drawing | Python, code interpreter | Perform calculations and generate charts |
| Internal information is required | RAG, file retrieval | Put relevant information into the context |
| Visual content is required | Image generation model | Turn natural language intentions into images |

The model can output a special token or structured call, which is identified and executed by the outer program, and then sent the result back to the context, and the model continues to be generated accordingly.

A truly powerful system is not "an omniscious model", but "a coordinator that will judge when to call which tools".

### 3. Multimodal: from reading and writing to reading, listening, speaking and generating

The input and output of LLM are expanding from plain text to images and audio. The main directions include:

- Understand hand-drawn web sketches and generate HTML/JavaScript;
- Generate pictures according to the text;
- Understand the content in the picture;
- Conduct a natural two-way conversation through voice.

Multimodalization makes natural language a more common human-computer interface, but images and audio will also become new attack carriers.

### 4. From System 1 to System 2

You can borrow the concepts in "Thinking, Fast and Slow" to understand two reasoning modes:

- **System 1**: fast, intuitive, automatic response;
- **System 2**: Slow, deliberate, will unfold multiple possibilities, check and retreat.

At that time, LLM was mainly generated at token at the same speed, which was closer to System 1. The ideal future system should be able to convert more reasoning time into higher accuracy: proposing candidates, searching the idea tree, reflecting, verifying, reverting, and finally answering.

The key perception here is not "the longer the output, the smarter it is", but whether the system really uses additional computing for search and verification.

### 5. Self-improvement: Looking for AlphaGo closed loop in the language world

AlphaGo first imitates the master's chess score, and then surpasses humans through self-game and clear victory or defeat. LLM can also imitate high-quality human answers, but common language tasks lack cheap, objective and automatic reward functions, so it is difficult to directly copy AlphaGo's self-improvement path.

In narrow areas where answers can be automatically verified, such as some mathematics, code and formalized tasks, it is easier to establish:

```text
Generate multiple schemes → Automatic verification → Keep the successful trajectory → Continue training
```

How to define reliable rewards for general tasks is still a core problem.

### 6. Customization: from a general assistant to a large number of expert models

Different organizations and industries require different knowledge, processes and behavioral boundaries. The means of customizing models include:

- System instructions and workflow rules;
- Upload the file and retrieve it through RAG;
- Connect business tools and databases;
- Use domain data fine-tuning.

In the future, it is more likely that there will be an ecology in which multiple specialized models or intelligent bodies collaborate, rather than one model will do everything.

---

## Part 4: LLM OS - a new computing stack

If you only regard LLM as a chatbot or text generator, you will underestimate it. A more appropriate analogy is that **LLM is becoming a kernel process of a new operating system, which is responsible for understanding natural language intentions, coordinating information and calling tools.**

| Traditional computer | Correspondence in LLM system |
|---|---|
| CPU / Kernel Process | base model and Reasoning Process |
| RAM | Context window (limited working memory) |
| Hard disk / network storage | local files, knowledge base, Internet |
| Peripheral / System Tools | Browser, Calculator, Python, Image and Audio Model |
| Application | Customized assistant or intelligent body for specific tasks |
| GUI / Command Line | Natural Language, Multimodal Interactive Interface |

Context window is a limited and valuable resource. The system needs to constantly decide which data should be loaded, retained, compressed or moved out, just like the operating system manages memory.

This metaphor also implies two ecological directions: one is a proprietary platform with closed source and unified experience; the other is a customizable ecology formed around the open weighted model, similar to the macOS/Windows and Linux genealogy in the desktop operating system.

---

## Part 5: The new computing paradigm also has new security issues

### 1. Jailbreak (Jailbreak)

The goal of jailbreak is to bypass the original security behavior of the model. Attackers may use role-playing, different languages or coding, automatically optimized meaningless suffixes, and even image noise that is almost invisible to the human eye.

The difficulty is that the rejection behavior learned by the model often only covers the expression in the training data, and the input space is almost unlimited. Repairing a specific hint does not guarantee the failure of similar variants.

### 2. prompt Injection

prompt injection does not directly require the model to violate the rules, but hides malicious instructions in the external content read by the model, such as:

- Hidden text in the web page;
- Light-colored instructions in the picture;
- Shared documents;
- The contaminated content in the RAG knowledge base.

When the model cannot reliably distinguish between "data" and "instructions", external content may hijack tasks. The more networking, reading files and calling tools, the greater the risk. In serious cases, the attacker may induce the model to leak private data or perform wrong operations.

### 3. Data Poisoning / Backdoor (Data Poisoning and Backdoor)

The attacker pollutes the training or fine-tunes the data, so that the model behaves abnormally when it sees a specific trigger word. Usually, the model may be completely normal, and the attacker's preset behavior will only be performed when trigger conditions appear, so it is not easy to detect through ordinary tests.

### 4. Security is a continuous offensive and defensive cycle.

Specific attacks can be repaired, but new variants will continue to appear. LLM security, like traditional system security, is a continuous game, not a one-time problem.

Enlightenment to the actual system:

- Do not automatically regard the text in web pages, documents, emails or pictures as trusted instructions;
- Strictly limit the tools and data permissions that the model can call;
- Procedure checks and manual confirmation to increase certainty for high-risk actions;
- Quarantine private data and untrustworthy content;
- Record the tool call chain and do the red team test;
- Take "the model may be manipulated" as a basic threat model, not a marginal situation.

---

## Key Conceptual Relationship Chart

```text
Internet text
   │
│ Pre-training: Predict the next token
   ▼
base model (knowledge and language pattern)
   │
│ SFT / Preferred Learning / RLHF
   ▼
assistant model (behavior and interaction form)
   │
├── Search / RAG ────── Supplement external knowledge
├── Calculator / Python ─ Accurate calculation and execution
├── Image / Audio ───── Multi-modal input and output
└── Professional tools ──────── Complete the real task
   │
   ▼
LLM OS / Intelligent Body System
   │
└── Must be prevented: jailbreak, prompt injection, data poisoning, permission abuse
```

---

## Practical suggestions for personal use of LLM

1. **Judge the task type first.** Creative writing can tolerate sampling; facts, mathematics, code and decision-making tasks must be verified.
2. **The model is required to use the source.** Priority search or retrieval of time-effective facts, without relying on parameter memory.
3. **Disassemble the complex tasks.** First make a plan, then implement it step by step, and check the intermediate results at the key nodes.
4. **Let the tools do what tools is good at.** Mathematics is entrusted to the calculator, data processing is entrusted to the code, and the latest facts are entrusted to the retrieval.
5. **Provide the necessary context.** The working memory of the model is limited, and the relevant information, constraints and evaluation standards should be clearly given.
6. **Take the output as a draft.** Important content needs to be cross-checked, tested or reviewed by field experts.
7. **Be careful with connection permissions.** When the model can read private files, send messages or execute commands, it should use the minimum permission and operation confirmation.

---

## Review questions

1. Why can LLM be simplified to "parameter file + running code"?
2. Why can next-token prediction force models to learn world knowledge?
3. How does "lossy compression of the Internet" explain hallucinations?
4. What is the difference between the training data of the base model and the assistant model?
5. Why can't fine-tuning fundamentally eliminate hallucinations?
6. Why does RLHF often use candidate answer comparison instead of just asking people to write the answer directly?
7. What inherent limitations of LLM have been solved by tool calls?
8. Why can Context window be analogiced to RAM?
9. What does System 2 reasoning need to convert the extra time into?
10. Why is prompt injection particularly dangerous in networking and RAG scenarios?

---

## The final cognitive model

When understanding LLM, don't just ask "will it answer this question", but also ask four things at the same time:

1. **What has the model itself learned?**——Fuzzy knowledge and patterns from pre-training parameters.
2. **How is it trained to act?**——From fine-tuning, preference data and code of conduct.
3. **What can it call?**——Retrieval, code, multimodal and other external tools.
4. **How could it fail or be attacked?**——hallucinations, prompt injection, data poisoning and permission abuse.

By separating these four layers, the capability boundaries, reliability and risk of an LLM system can be judged more accurately.

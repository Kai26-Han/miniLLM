# Building Vision-Language Models

> This introductory article explains what a VLM is, what it adds to a text-only language model, and how training develops vision-language capabilities. Here, “building” specifically means training a model.
>
> Markdown diagrams and examples provide an overall picture without requiring prior knowledge of algorithms, equations, or training code. Compiled on 2026-09-08.

## Key Points

1. **VLM stands for Vision-Language Model.** It learns relationships between visual content and language. This article focuses on VLMs that generate text answers from images and questions.
2. **Vision adds a source of information; language capabilities remain essential.** Many generative VLMs still use a language model to compose answers, with the current image providing additional evidence.
3. **Adding an image input does not mean a model has learned to see.** Images must be encoded, and visual representations must be adapted to the language network. This cooperation usually requires training.
4. **Building a VLM does not require training every component from scratch.** You can train a visual interface for an existing LLM, continue training an existing VLM, or pretrain from random initialization.
5. **Answering image questions, reading small text, understanding video, and generating images are different capabilities.** Supporting one does not automatically provide the others.
6. **Visual evidence does not eliminate errors.** A model may miss details, associate information incorrectly, reason incorrectly, or invent unsupported content. Reliability must be tested on specific tasks.

## Reading Guide

1. [What Is a VLM?](#what)
2. [How Does a VLM Differ from an LLM?](#difference)
3. [How Does a VLM Turn an Image into an Answer?](#mechanism)
4. [VLMs, OCR, Image Generation, and Agents](#neighbors)
5. [Three Paths for Training a VLM](#building)
6. [What Does Training Actually Teach?](#training)
7. [What Else Do You Need to Prepare?](#system)
8. [How Can You Tell Whether It Understands the Image?](#evaluation)
9. [Common Misconceptions and Further Reading](#reading)
10. [Where This Project Fits](#project)

<a id="what"></a>

## 1. What Is a VLM?

### 1.1 From Answering Text to Answering from Visual Evidence

VLM is short for Vision-Language Model.

Consider a simple example: you want to know how many cups are on a table. A text-only model receiving just that question has no information about your actual table; you must first describe it. A VLM with the relevant capabilities can receive a photo of the table and connect the objects in the image to the question about the number of cups.

```text
Text-only model:
User describes the table → Model answers from the description

Vision-language model:
Photo of the table + User question → Model answers using the image
```

The difference goes beyond input format: **the answer can use information in the current image**. Object colors, spatial relationships, page layouts, and chart trends do not all have to be transcribed by a person first.

### 1.2 What Counts as Visual Input?

Visual input can include photos, screenshots, document pages, charts, and frames extracted from video. These inputs place different demands on a model.

| Input | Possible tasks | Additional capabilities needed |
|---|---|---|
| Everyday photos | Describe scenes and answer questions about object attributes | Recognize objects and relationships |
| Documents or receipts | Read fields and understand tables | Recognize text and retain layout |
| Charts | Compare trends and explain values | Connect legends, axes, and data |
| Software screenshots | Explain interface content and locate buttons | Identify controls and their positions |
| Multiple images | Compare differences and match objects | Track which image an object belongs to and resolve references |
| Video | Describe actions and locate events | Connect frame order with time |

These are potential VLM tasks, not a claim that every VLM can perform all of them reliably. Actual capabilities depend on input design, training content, and evaluation results.

### 1.3 Not Every VLM Is a Chat Model

The broader VLM category includes different kinds of image-text models. CLIP, for example, learns whether images and text match. It is useful for image-text retrieval or as a source of visual representations; its dual-encoder architecture does not itself generate a conversational answer token by token. [Original CLIP paper](https://arxiv.org/abs/2103.00020)

```text
Image-text matching:
Image + Candidate descriptions → Determine which descriptions match best

Image-conditioned generation:
Image + Question → Generate a text answer
```

Everyday references to a “language model that can see” usually mean the second type. The discussion of building and training models below also focuses on generative VLMs.

<a id="difference"></a>

## 2. How Does a VLM Differ from an LLM?

### 2.1 Define the Comparison First

LLM stands for Large Language Model. In practice, some multimodal models are also called LLMs, so these names do not describe strictly exclusive categories.

For clarity, this article compares **VLMs with LLMs that accept only text**. Many VLMs themselves contain an LLM, with visual capabilities built around it.

| Dimension | Text-only LLM | Generative VLM |
|---|---|---|
| Input information | Text and material already converted to text | Text and encoded visual content |
| Sources of current facts | User text, retrieved material, tool outputs, and so on | Also the current image |
| Common processing path | Text representations → Language model | Visual and text representations → Image-text fusion → Language generation |
| Training data | Text, conversations, code, and so on | Also image-text relationships and visual tasks |
| Additional challenges | Language understanding, knowledge, and reasoning | Perceiving details, spatial association, and cross-modal cooperation |
| Common errors | Misunderstanding text, incorrect knowledge, and unsupported completion | Also misidentifying objects, misreading text, and confusing positions |
| Computational cost | Depends on model size and text length | Also depends on resolution, image count, and video frame count |

### 2.2 What Do They Share?

Many generative VLMs retain the language model's basic output process: predict the next token, then continue generating from the content so far. A token is roughly a piece of text processed by the model; it does not necessarily correspond to a complete word or Chinese character.

Language knowledge, expression, instruction following, and reasoning remain important after adding vision. Images supply additional evidence, and the language network must connect that evidence to the question and organize an answer.

```text
Existing language capabilities: understand questions, express ideas, use knowledge
                                      +
Visual input capabilities: obtain content and relationships from the current image
                                      +
Cross-modal cooperation: connect the question to relevant visual evidence
                                      ↓
                         Ability to answer image questions
```

A VLM therefore adds a learnable information-processing path, rather than merely adding an upload button around an LLM.

### 2.3 Seeing a New Image Does Not Mean Permanently Learning It

Sending an image to a VLM normally supplies context for the current inference. It does not automatically modify the model's parameters.

Keep the two processes separate:

- **Inference:** use existing capabilities to process the current input and produce an answer.
- **Training:** use data and feedback to adjust parameters, changing how the model handles later tasks.

Uploading many images in succession does not amount to fine-tuning. An application storing earlier images also does not mean that their contents have entered the model's parameters.

<a id="mechanism"></a>

## 3. How Does a VLM Turn an Image into an Answer?

### 3.1 A Common Three-Component Pipeline

```text
Image    → Visual encoder → Connector ──┐
                                       ├→ Language model → Answer
Question → Text processing ────────────┘
```

This is a common modular design, not the only possible VLM architecture. LLaVA is a representative example: it connects a visual encoder to an LLM and performs visual instruction tuning. [Original LLaVA paper](https://arxiv.org/abs/2304.08485)

| Component | Responsibility | What this does not mean |
|---|---|---|
| Visual encoder | Convert pixels into useful visual features | It does not directly produce the final answer |
| Connector | Adapt or organize visual features for the language network | It does not necessarily translate them into a text description first |
| Language model | Relate the question to visual information and generate text | The components do not cooperate automatically just because they are connected |

### 3.2 Visual Features Are Not a Hidden Written Description

A visual encoder outputs numerical representations. They can carry information about objects, colors, positions, and text, but usually cannot be read directly as human language.

A “visual token” can be understood as a visual representation occupying one processing position. Saying that an image uses a certain number of visual tokens does not mean it has become that many words.

If a system first generates a caption and passes it to a text-only model, that model can use only the information retained in the caption. A VLM that receives visual features directly has an opportunity to use richer image information in response to the question.

### 3.3 Why Can't Two Pretrained Models Simply Be Joined?

Visual and language components are trained separately and may organize their internal representations differently. Even equal output widths do not mean the language model can correctly interpret the visual module's numbers.

Connectors and cross-modal training address this mismatch. BLIP-2 demonstrates a route that reuses pretrained visual and language backbones while learning an intermediate bridging module. It shows that a VLM can be built from established components. [Original BLIP-2 paper](https://arxiv.org/abs/2301.12597)

However, a bridge can only use the information available to it. It cannot be expected to reliably reconstruct small text already lost during image preprocessing.

### 3.4 Why Can One Image Support Different Questions?

Given a restaurant photo, you might ask how many tables there are or which side the exit is on. The model must connect the question to different visual information, rather than assign the image a single fixed category.

This ability to use an image according to a question must be learned. A model trained only on broad descriptions may not yet know how to count precisely, read text, or answer complex spatial questions.

<a id="neighbors"></a>

## 4. VLMs, OCR, Image Generation, and Agents

### 4.1 OCR Plus an LLM Provides Another Information Path

OCR stands for Optical Character Recognition. Its main role is to turn text in images into machine-readable text. A system can run OCR first and then let an LLM read the result.

```text
OCR + Text-only LLM:
Document image → Extract text and optional layout → LLM → Answer

Generative VLM:
Document image → Visual representations → Fuse with question → Answer
```

If a task only requires accurate extraction of a few text fields, OCR combined with rules or an LLM may be sufficient. If it also depends on chart shapes, object appearance, or layout relationships, a text transcription alone may fall short.

The approaches can also be combined: a VLM can use visual information while consulting OCR results. The choice should depend on results on the target data, cost, and maintenance requirements.

### 4.2 Understanding Images Does Not Imply Generating Images

Answering image questions primarily maps visual input to text output. Generating images additionally requires a suitable visual output representation, an image decoding path, and corresponding training.

```text
Visual understanding: Image + Question → Text
Image generation:     Text or other conditions → Image
```

One system can support both, but the capabilities do not come together automatically. A model being called “multimodal” does not establish support for every combination of inputs and outputs.

### 4.3 Seeing an Interface Does Not Imply Operating a Computer

A VLM can contribute to interface understanding, such as recognizing buttons in a screenshot. Clicking, typing, and observing the results require external tools and a control process.

```text
Screenshot → Recognize state → Decide action → Tool execution
                   ↑                                ↓
                   └────── New screenshot or feedback ┘
```

This is one way a visual agent can work. A VLM may participate in perception and decision-making, but a model weights file alone does not contain the complete operating system, tool interfaces, or execution logic.

<a id="building"></a>

## 5. Three Paths for Training a VLM

The training path depends first on the starting point: separate visual and language capabilities, an existing ability to combine images and language, or no pretrained foundation. These starting points determine what training needs to add.

### 5.1 Path One: Train Visual Capabilities on Top of an Existing LLM

If you already have a text-only Base model and want to reuse it, you need to choose a visual encoder, a connector, and an input format, then train image-text cooperation.

```text
Existing visual encoder ──┐
                          ├→ Design connection → Image-text alignment → Visual instruction tuning
Existing language model ──┘
```

This path extends language capabilities into vision-language capabilities. The key work includes:

1. Choose visual inputs and an encoder suited to the target task.
2. Feed visual representations into the language model while keeping text processing consistent.
3. Prepare image-text data and first verify that visual input actually affects the output.
4. Train question-answering and instruction-following behavior using task data.
5. Evaluate new images and check whether existing language capabilities are retained.

Public work such as LLaVA and BLIP-2 offers different connection and training approaches. These demonstrate that reusing existing modules is viable, without implying that any two models will work when simply joined. [LLaVA](https://arxiv.org/abs/2304.08485), [BLIP-2](https://arxiv.org/abs/2301.12597)

### 5.2 Path Two: Continue Training an Existing VLM

This path starts from weights that already support image-text cooperation. New data and parameter updates develop capabilities for a particular domain or task.

```text
Existing VLM weights + Domain image-text data or visual instruction data
                                      ↓
                  Continued pretraining or supervised fine-tuning
                                      ↓
                         VLM adapted to the target task
```

For example, a model may describe receipts but fail to extract specified fields consistently. Training can use suitable images, questions, and ideal answers. Updates may affect adapters, part of a backbone, or a broader set of model parameters.

Compared with the first path, the starting model already has a visual interface and image-text alignment. Training focuses on domain adaptation, task expansion, or better behavior. New task performance and possible regression in existing capabilities still need evaluation.

### 5.3 Path Three: Pretrain from Random Initialization

Training the entire model from scratch, in the strict sense, means learning foundational visual and language capabilities from random parameters and establishing cross-modal cooperation. The components can be pretrained separately before image-text training, or the training process can introduce multimodal fusion earlier.

```text
Randomly initialized visual and language components
                           ↓
Images, text, and image-text data teach foundational capabilities
                  and cross-modal relationships
                           ↓
Visual instruction tuning and subsequent capability improvements
```

This path provides broader control over the model and data design, while requiring resources for foundational learning, training stability, and computational scale.

“Joint training” means updating multiple components together; it does not mean they necessarily start from random parameters. The first two paths can also use joint training. Therefore, describe **the source of initialization** and **the scope of parameter updates** separately.

### 5.4 Comparing the Three Training Paths

| Path | Starting point | Main training tasks |
|---|---|---|
| Add visual capabilities to an LLM | Separately pretrained language and visual components | Establish cross-modal alignment, then train visual task behavior |
| Continue training a VLM | Weights that already support image-text cooperation | Domain continued pretraining, instruction tuning, and capability expansion |
| Pretrain from scratch | Randomly initialized parameters | Learn foundational representations, language capabilities, and cross-modal relationships |

All three update parameters through data and feedback. They differ in how much existing capability is reused and where learning begins. Choose a training plan around the target capabilities, available data, and compute budget.

<a id="training"></a>

## 6. What Does Training Actually Teach?

### 6.1 Start with a Map of the Stages

```text
Acquire foundational visual and language capabilities
                           ↓
Image-text alignment: enable the components to cooperate
                           ↓
Expand multimodal training as needed: cover more inputs and tasks
                           ↓
Visual SFT: learn to answer according to questions and instructions
                           ↓
Optional feedback optimization: improve factuality and task performance
```

This is a map of training purposes, not a fixed recipe that every project must follow step by step. Some stages can be combined, and some initializations are strong enough to omit a separate interface warm-up.

### 6.2 “Pretrain” Can Refer to Different Work

| What the name refers to | What is mainly learned |
|---|---|
| Visual encoder pretraining | How to extract useful visual features |
| Language model pretraining | Language patterns, knowledge, and generation |
| Connector image-text alignment | How the language network can use visual representations |
| Multimodal continued pretraining | How to process images and text together more broadly |

A script named `pretrain` does not by itself mean that the entire model is being trained again. Look at its data, the parameters being updated, and the objective.

### 6.3 Alignment Versus SFT

Alignment mainly addresses whether the language network can use image information effectively. One common approach predicts captions from images, allowing the connector to learn an initial adaptation.

SFT stands for Supervised Fine-Tuning. It then uses images, questions, and ideal answers to train specific task behavior.

```text
Caption training:
Image → “There is a cup and a book on the table.”

Instruction tuning:
Image + “Which side of the cup is the book on?” → “On the left.”
Image + “List only the object names.”          → “Cup, book.”
```

Both can use similar text prediction methods. The larger difference is what the model is asked to do. Summarizing an image does not imply an ability to select evidence and control output under many different instructions.

### 6.4 Freezing, Fine-Tuning, and LoRA Describe Different Dimensions

“Freezing” specifies which parameters are temporarily not updated. “SFT” specifies the training task format. “LoRA” describes an adaptation method that reduces the number of trainable parameters. They can be used together.

When only the connector is trained, the visual and language backbones still participate in computation. Although the language model's parameters do not change, it must still propagate answer errors back to the connector. Fewer trainable parameters do not make the backbones disappear from memory or computation.

For an introduction, focus on what the update scope means: fewer updates reduce resource pressure but restrict adaptation; broader updates can change more behaviors but require stronger data and validation.

### 6.5 What Image Input, Training, and Inference Change

| Action | Main effect |
|---|---|
| Supply a different image | Change the visual evidence available for this answer |
| Rewrite the question or prompt | Change the current task and expression requirements |
| Fine-tune the model | Change parameters, affecting later task behavior |
| Add a tool workflow | Change how the system obtains information, acts, and verifies results |

If the output format is unclear, improving the prompt may be sufficient. If small text is unreadable in the input, inspect image processing first. If the model consistently struggles with a particular task, investigate training data and parameter adaptation.

<a id="system"></a>

## 7. What Else Do You Need to Prepare?

### 7.1 Define the Problem First

“Train a model that can see” is not specific enough. Specify the input, the required output, and how correctness will be judged.

For example, these three objectives call for different plans:

- Describe the main content of everyday photos.
- Extract an invoice number, date, and amount from scanned invoices.
- Determine how the same object has changed across several images.

The objective affects resolution, multi-image support, annotations, output format, and evaluation.

### 7.2 A Complete Training Workflow

```text
Define the task and acceptance criteria
                  ↓
Collect representative examples and establish an independent evaluation set
                  ↓
Choose the starting point, initialization weights, and update scope
                  ↓
Prepare training data; verify image processing and model inputs
                  ↓
Run a small trial; check training feedback, saving, and recovery
                  ↓
Run formal training with periodic validation and model saves
                  ↓
Evaluate new images; use errors to revise the data or training plan
```

Plan the evaluation set before training. Otherwise, repeated adjustments based on training examples can conceal whether the model works on new images.

### 7.3 Data Quality Goes Beyond Quantity

One image paired with several questions can provide several training samples, but it does not become several independent images. Also consider clarity, scene coverage, languages, task types, and answer quality.

A particularly important check is whether the image supports the answer. If an image does not show a manufacturing date but the label supplies a definite date, the model is being taught to answer confidently about invisible information.

Training and validation should also avoid sharing identical or near-duplicate images. Split documents and videos by source as well, so repeated content is not mistaken for generalization.

### 7.4 Image Count and Resolution Affect Cost

Visual input must be encoded, and subsequent networks must read the retained visual positions. More images, higher resolution, and more video frames usually increase compute and memory requirements.

Do not estimate VLM runtime cost solely from the language backbone's parameter count. Nor should larger images be assumed to be better: extra input must match the formats supported by the model and provide a real benefit for the task.

<a id="evaluation"></a>

## 8. How Can You Tell Whether It Understands the Image?

### 8.1 One Fluent Description Is Not Proof of Capability

A model may describe the overall scene correctly yet misread small text. It may also guess a common answer without using the current image. Set up controlled comparisons for the task.

```text
Fixed question: “What color is the cup?”

Image A: Red cup       → Ideal answer: Red
Image B: Blue cup      → Ideal answer: Blue
Image C: Cup obscured  → Ideal answer: Cannot determine
```

This is an evaluation design, not a measured result. The key is whether outputs change appropriately with visual evidence and acknowledge limitations when evidence is insufficient.

### 8.2 Three Levels of Checks

| Level | What to check | Example |
|---|---|---|
| Perception | Whether the input content is read correctly | Correct text, colors, and counts |
| Association and reasoning | Whether the right objects and question are connected | Whether a price belongs to the correct product |
| Expression and behavior | Whether the answer is faithful and follows instructions | Outputting only required fields without fabrication |

This avoids attributing every error to a model being “too small.” Many problems arise more directly from input clarity, task data, or association capabilities.

### 8.3 VLMs Still Have Limits

Images may be blurry, occluded, or incomplete, and models may rely too heavily on common knowledge. Higher resolution, more data, and stronger language capabilities can help some tasks, but do not guarantee that every error disappears.

Distinguish what the image shows from what the model infers. Seeing someone running is not enough to establish their identity, occupation, or true motivation. Similar-looking objects do not necessarily reveal their internal state.

Reliable applications require evaluation on real tasks, rather than relying only on model names, a few demonstrations, or general caption quality.

<a id="reading"></a>

## 9. Common Misconceptions and Further Reading

| Misconception | More accurate understanding |
|---|---|
| A VLM is a completely different model that replaces the LLM | Many generative VLMs contain an LLM and extend it with visual input |
| Image upload support means the model can understand images | Effective visual encoding, adaptation, and training are also needed |
| Visual tokens are words converted from an image | In common implementations, they are continuous visual representations |
| Showing a model many images trains it | Ordinary inference does not automatically update parameters |
| Building a VLM requires training all weights from scratch | Existing modules can be reused, with only new components trained or further adapted |
| A model that captions photos must also support OCR, video, and image generation | These require corresponding input/output designs and training |
| Training only a small subset of parameters makes the entire model small | Frozen backbones still need to be loaded and used in computation |

This article establishes the conceptual framework. Continue in this order:

1. [VLM Architecture and Training Methods](../architecture/VLM-Architecture-and-Training.md): visual encoding, connectors, fusion, and different training approaches.
2. [miniLLM-vlm Pretraining](../practice/miniLLM-vlm-Pretraining.md): how these concepts map to this project's training process.

For original sources, start with [CLIP](https://arxiv.org/abs/2103.00020), [LLaVA](https://arxiv.org/abs/2304.08485), and [BLIP-2](https://arxiv.org/abs/2301.12597). They introduce image-text representations, visual instruction learning, and cross-modal adaptation using pretrained components, respectively.

<a id="project"></a>

## 10. Where This Project Fits

miniLLM-vlm follows the path of adding visual capabilities to an existing LLM. It reuses the language Base trained by miniLLM, connects a pretrained SigLIP2 visual encoder, and adapts visual representations through an intermediate MLP.

```text
SigLIP2 visual encoder [frozen]
                 ↓
         MLP connector [trained]
                 ↓
miniLLM language Base [frozen] → Image-conditioned text answer
```

The current Pretrain stage mainly establishes the image-text interface, focusing training on the connector. It does not retrain the visual and language backbones from scratch, and it does not complete all the instruction tuning and evaluation required for a general visual assistant.

miniLLM-vlm uses its own independent code to load the Base and other model files. It does not depend on the parent miniLLM project's source code at runtime. Future capability improvements should be determined by the target tasks, training data, and validation results.

# VLM Architecture and Training Methods

> This article addresses two questions: how images become information a language model can use, and how training enables effective cooperation between visual and language components.
>
> It explains the principles through Markdown diagrams, workflows, and comparison tables, without algorithmic derivations. The main discussion covers general designs; the final section briefly maps them to this project. Compiled on 2026-09-08.

## Reading Guide

1. [What Does a VLM Add to a Language Model?](#overview)
2. [How Do the Three Core Components Divide the Work?](#modules)
3. [How Is Visual Information Fused with Text?](#fusion)
4. [How Do High Resolution, Multiple Images, and Video Change the Architecture?](#visual-input)
5. [Differences Between Representative Architectures](#architectures)
6. [What Problems Do the Main Training Stages Solve?](#stages)
7. [How Are Common Training Recipes Assembled?](#recipes)
8. [How Do Data and Evaluation Support Training?](#data-eval)
9. [Original Sources and Reading Order](#references)
10. [The Architecture Used in This Project](#project)

<a id="overview"></a>

## 1. What Does a VLM Add to a Language Model?

### 1.1 From Text Generation to Generation Grounded in Visual Evidence

A text-only language model generates continuations from textual context. A generative vision-language model, or VLM, adds image information to this process, so both the image and the question influence the answer.

```text
Text-only language model:
Text question → Text representations → Language model → Answer

Generative VLM:
Image         → Visual encoder → Connector ─┐
                                           ├→ Language model → Answer
Text question → Text representations ──────┘
```

For example, if the question is “How much are the beef noodles?”, a language model could guess from general knowledge. A VLM should instead read the price on the current menu. Replacing it with a menu showing a different price should change the answer accordingly.

The key is therefore not merely accepting images, but **using evidence from the image to generate the answer**. The system must handle perception, information adaptation, image-text association, and language expression.

### 1.2 VLM Does Not Only Mean Image Chat

| Type | Inputs and outputs | Main uses |
|---|---|---|
| Image-text representation model | Images and text → Representations and matching scores | Image-text retrieval, classification, visual features |
| Generative understanding model | Image and question → Text answer | Captioning, visual question answering, document understanding |
| Unified understanding and generation model | Image-text inputs → Text or images | Both visual understanding and image generation |

CLIP and SigLIP mainly belong to the first category. They learn relationships between images and text; their dual encoders do not themselves generate answers token by token. This article focuses on the second category: models that generate text conditioned on visual input. [CLIP](https://arxiv.org/abs/2103.00020), [SigLIP](https://arxiv.org/abs/2303.15343)

<a id="modules"></a>

## 2. How Do the Three Core Components Divide the Work?

### 2.1 The Visual Encoder: Turning Pixels into Useful Representations

An image starts as pixel values, which a language model cannot directly interpret as text. The visual encoder extracts information suitable for subsequent processing.

A common Vision Transformer, or ViT, can be summarized as follows:

```text
Original image
      ↓
Preprocessing, such as resizing and normalization
      ↓
Divide the image into patches
      ↓
Convert each patch into a numerical representation; add positional information
      ↓
Multiple visual Transformer layers: exchange information across positions
      ↓
A set of visual features
```

A patch is a small image region, not a complete object. An object may span several patches, and a patch may contain parts of several objects. After processing, the feature at each position can also contain information from its surroundings or even the whole image. [ViT](https://arxiv.org/abs/2010.11929)

A “feature” here is a set of numbers, not a predefined text label such as “this is a cat.” Information about colors, shapes, spatial relationships, and meaning may be distributed across many positions and values.

**Why retain multiple positions?** Summarizing the whole image and answering a local question require different amounts of information. Recognizing that an image is a menu may rely on overall features; reading the price at the bottom right requires local detail. Compressing the entire image into one overall representation too early may restrict detailed question answering.

### 2.2 If the Visual Encoder Is Pretrained, Why Train the VLM?

Visual components such as CLIP and SigLIP learn semantically meaningful features through image-text learning. SigLIP 2 incorporates additional objectives to improve representations. Other visual models learn mainly from patterns within images themselves. [SigLIP 2](https://arxiv.org/abs/2502.14786), [DINOv2](https://arxiv.org/abs/2304.07193)

However, visual and language models are usually trained separately. An encoder's ability to extract image information does not mean the language model already knows how to use it.

```text
The visual encoder has learned: how to represent image content
The language model has learned: how to process language and generate text

Still missing: how the language model should read those visual representations
```

Reusing established modules saves foundational training costs, but cross-modal adaptation still needs to be addressed.

### 2.3 The Connector: Bringing Visual Representations into the Language Model

The connector usually sits between the visual encoder and language model. It processes extracted features to suit the language model's input interface.

A connector may be needed even when the two models have the same hidden width. Representing each position with 768 numbers does not mean those numbers have the same organization, scale, or semantic role.

Separate the connector's two possible functions:

```text
Function 1: Feature adaptation
N visual positions → Projection / MLP → N adapted visual positions
The number of positions can remain unchanged; the representation changes.

Function 2: Information compression
N visual positions → Merging / Resampling → Fewer visual positions
This reduces downstream reading costs while introducing an information bottleneck.
```

| Connection method | Main role | Trade-off to understand |
|---|---|---|
| Linear projection | Apply a simple mapping at each position | Small structure, relatively limited adaptation |
| Two-layer MLP | Apply nonlinear adaptation at each position | Simple, usually preserves the number of positions |
| Merging or pooling followed by projection | Reduce positions, then adapt their representations | Lower cost, with possible loss of detail |
| Q-Former / Resampler | Learn to extract a shorter representation | Controlled length, with a more complex structure and training process |

The original LLaVA uses a linear projection; LLaVA-1.5 replaces it with a two-layer MLP. BLIP-2 uses a Q-Former, and Flamingo uses a Perceiver Resampler. Their differences involve not just complexity but also choices about how much visual information to retain. [LLaVA](https://arxiv.org/abs/2304.08485), [LLaVA-1.5](https://arxiv.org/abs/2310.03744), [BLIP-2](https://arxiv.org/abs/2301.12597), [Flamingo](https://arxiv.org/abs/2204.14198)

A Q-Former-like module can be understood as a set of learnable query positions that extract information from a longer visual representation. These queries are not manually specified object categories, nor are they necessarily derived directly from the user's question; that depends on the design.

### 2.4 The Language Model: Connecting Visual Information to the Question

The language model continues to process sequences and generate text, but now its readable context also includes visual information.

Different questions about the same image require different information:

| Question | Association the model must make |
|---|---|
| What kind of scene is this? | Combine overall visual information |
| What color is the person on the left wearing? | Connect the object, position, and color |
| What is the price? | Connect text and numbers to the item they belong to |
| What changed between the two images? | Connect objects and differences across images |

Attention helps the model read relevant context. Subsequent network operations continue processing the representations and eventually produce an answer. Image-text relationships emerge through multiple layers working together; a single attention head should not simply be interpreted as a fixed human skill.

**The three components need compatible capabilities.** The connector cannot reliably restore details that the visual side failed to retain. Even retained information may go unused if adaptation is inadequate. The language model's own expression and reasoning capabilities also affect the result.

<a id="fusion"></a>

## 3. How Is Visual Information Fused with Text?

The connector describes how visual features are processed. The fusion mechanism describes where the language model reads them. This is one of the most important distinctions in architecture analysis.

### 3.1 Input-Level Fusion: Building a Single Sequence

This approach converts images and text into compatible representations and passes them to the same language model backbone.

```text
Image    → Visual encoder → Connector → Visual representations ─┐
                                                               ├→ Combined input sequence
Question → Tokenizer → Text embeddings ────────────────────────┘
                                                                            ↓
                                                           Multiple language Transformer layers
                                                                            ↓
                                                                    Generated answer

Sequence example:
[Prompt text] [Visual positions 1 ... N] [Question] [Answer prefix]
```

The advantage is straightforward reuse of an existing language model. Visual and text positions enter the same downstream processing, allowing the answer to attend to preceding image information.

The cost is that visual positions occupy shared sequence space. More images, higher resolution, or more retained features usually increase language-side computation and cache use. The LLaVA family is a representative example. [LLaVA-1.5](https://arxiv.org/abs/2310.03744)

### 3.2 What Does the Image Placeholder Do?

`<image>` is often just a data-level marker telling the program where to place an image. Some implementations expand it into multiple placeholder positions, then replace the lookup embeddings with visual vectors.

```text
Text marker:         <image> Describe the image
                         ↓
Reserved positions:  [Placeholder] [Placeholder] ... [Placeholder] [Question]
                         ↓ Replace with actual image features
Actual input:        [Visual] [Visual] ... [Visual] [Question]
```

A “visual token” here usually means a continuous numerical representation occupying one sequence position. It is not a character and need not correspond to a word in the vocabulary.

The image is therefore not first translated into a hidden caption for the LLM to read. Generating a text description and then giving it to another model is a different system design, constrained by the information preserved in that description.

### 3.3 Cross-Attention Fusion: Reading Separate Visual Information Inside Language Layers

Another approach keeps a separate visual branch and adds modules inside the language model to read it.

```text
Image → Visual encoder → Visual resampling → Separate visual representations
                                                        │
                                                        ↓
Text → Language layer ─────────────────────→ Cross-attention → Later language layers → Answer
```

At the relevant layers, the language network uses its current hidden states to read visual features. Flamingo uses gated cross-attention to introduce visual information into an existing language network. [Flamingo](https://arxiv.org/abs/2204.14198)

This does not require placing every visual feature in the text input sequence. It does require additional network layers and management of visual-branch computation and access. Reading visual information still has a cost.

Input-level fusion also creates attention relationships between images and text, but that does not mean it adds a structurally separate cross-attention layer. **Information interaction and the use of a dedicated cross-attention module describe different aspects of the system.**

### 3.4 Encoder–Decoder Designs and Visual Injection at Multiple Layers

Two other variations are useful to know:

- **Image encoder–text decoder:** the encoder forms visual representations, and the decoder reads them while generating text. Pix2Struct is one example, focusing on understanding screenshots, documents, and interfaces.
- **Visual injection at multiple layers:** features from different visual depths enter several language layers instead of entering only once at the input. Qwen3-VL's DeepStack illustrates this design, but its residual injection should not be confused with Flamingo-style cross-attention.

Sources: [Pix2Struct](https://arxiv.org/abs/2210.03347), [Qwen3-VL](https://arxiv.org/abs/2511.21631).

All these architectures can be analyzed with the same questions: where is the image encoded, how much information is retained, where does fusion happen, and how does the language model generate the output?

<a id="visual-input"></a>

## 4. How Do High Resolution, Multiple Images, and Video Change the Architecture?

### 4.1 Resolution Determines Which Information Can Reach the Model

Understanding an image and reading its details impose different requirements. After a whole document page is reduced in size, the model may still recognize it as an invoice while being unable to read the amount.

```text
Fixed size:
Original image → Resize the whole image → Fixed number of visual features

Multiple crops:
Original image → Global thumbnail + Local crops from the original → More visual features

Dynamic resolution:
Original image → Variable grid based on shape and budget → Variable-length visual features
```

High-resolution designs must address three issues together: preserving details, telling the model where those details belong, and controlling additional computation. Simply increasing input dimensions does not establish reliable high-resolution capabilities in an existing encoder.

LLaVA-OneVision and Qwen2.5-VL offer different visual representation designs addressing these issues, including dynamic inputs, local processing, and feature compression. [LLaVA-OneVision](https://arxiv.org/abs/2408.03326), [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)

### 4.2 Multiple Images Require Tracking Information Ownership

Recognizing two images separately does not imply being able to compare them correctly. The model must also know which image each object, price, or piece of text comes from.

Architecture and data must jointly represent image boundaries, order, and references. Training also needs comparisons, correspondences, and questions spanning images to turn single-image capabilities into multi-image cooperation.

### 4.3 Video Requires Temporal Relationships

```text
Video → Sample frames → Extract visual information from each frame
                                      +
                            Frame order and timing
                                      ↓
                      Connect changes over time and understand events
```

“The cup is on the table” and “the cup is being picked up” describe states at different times. Understanding video also requires event order, timing, and object identity across frames.

More frames can preserve more clues, but they also increase reading costs. Modern video VLMs therefore consider frame sampling, features per frame, and temporal representations together. [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), [Qwen3-VL](https://arxiv.org/abs/2511.21631)

<a id="architectures"></a>

## 5. Differences Between Representative Architectures

| Representative model | Architectural emphasis | Main design intent |
|---|---|---|
| LLaVA / LLaVA-1.5 | Visual encoder + Linear projection or MLP + LLM | Reuse existing visual and language capabilities through a simple interface |
| BLIP-2 | Visual encoder + Q-Former + LLM | Extract shorter visual representations suitable for the language model |
| Flamingo | Visual resampler + Cross-attention inside language layers | Support image-text interaction through a separate visual branch |
| LLaVA-OneVision | Retain a simple backbone while organizing different visual settings | Coordinate representations and training for single images, multiple images, and video |
| Qwen2.5-VL / Qwen3-VL | Dynamic visual inputs, positional representations, and fusion design | Improve detail, temporal understanding, and long-input processing |
| InternVL3.5 | ViT–MLP–LLM with broad joint training | Expand capabilities through data, pretraining, and post-training |

References: [LLaVA-1.5](https://arxiv.org/abs/2310.03744), [BLIP-2](https://arxiv.org/abs/2301.12597), [Flamingo](https://arxiv.org/abs/2204.14198), [OneVision](https://arxiv.org/abs/2408.03326), [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), [Qwen3-VL](https://arxiv.org/abs/2511.21631), [InternVL3.5](https://arxiv.org/abs/2508.18265).

There is no universally “best connector” across these approaches. Results depend on visual input, backbone capabilities, training data, and compute budget working together. Even models using the same MLP structure can differ substantially because of resolution and training content.

The text output path of an understanding-oriented VLM also does not automatically support image generation. Unified understanding and generation models such as Chameleon and Janus-Pro involve visual output representations and corresponding training, introducing another set of architectural trade-offs. [Chameleon](https://arxiv.org/abs/2405.09818), [Janus-Pro](https://arxiv.org/abs/2501.17811)

<a id="stages"></a>

## 6. What Problems Do the Main Training Stages Solve?

### 6.1 Distinguish Four Activities Often Called “Training”

| Activity | Core question | Typical learning content |
|---|---|---|
| Component pretraining | What foundational capabilities do the visual and language components have? | Image representations, language patterns, and text generation |
| Multimodal alignment and continued pretraining | How can these components process images and text together? | Adapt visual inputs and learn cross-modal relationships |
| Visual instruction tuning, or SFT | How should the model answer the user's task? | Select evidence, follow instructions, and control output format |
| Preference and feedback optimization | How can the model choose better candidate behaviors? | Factuality, quality, and verifiable task performance |

A script called `Pretrain` might train only the connector or the whole multimodal system. Identical names do not imply identical objectives, costs, or update scopes.

### 6.2 Component Pretraining: Acquiring Reusable Foundations

The visual component can learn representations through image-text matching or visual self-supervision. The language component learns prediction and expression from large amounts of text. VLM construction often starts directly from these existing weights.

```text
Large image or image-text datasets → Pretrained visual weights ──┐
                                                               ├→ Assemble the VLM
Large text datasets ──────────────→ Pretrained language weights ─┘
```

Image-text representation learning should also be distinguished from generative alignment. CLIP and SigLIP mainly learn whether images and text match. Subsequent connector training can instead adapt the interface by generating captions from images; it need not reuse the same matching objective. [CLIP](https://arxiv.org/abs/2103.00020), [SigLIP](https://arxiv.org/abs/2303.15343)

Choosing a Base or Instruct language model also affects later work. A Base model mainly learns language continuation, whereas an Instruct model has already received instruction tuning. After adding vision, its handling of visual tasks and question-answer formats still needs to be checked.

### 6.3 Image-Text Alignment: Establishing a Working Interface

A typical approach uses images and captions, asking the model to generate relevant text from the image. With a randomly initialized connector, it is common to freeze the visual and language backbones initially and update only the connector.

```text
Image → Visual encoder [frozen] → Connector [trained] → LLM [frozen]
                                                            ↓
                                                    Predict caption text
                                                            ↓
                                                Compare with reference caption
                                                            ↓
                                                    Adjust the connector
```

The purpose is to make visual information influence language output usefully and reduce the mismatch between independently trained components. This does not retrain a visual encoder, nor does it establish general image-chat capabilities by itself.

Alignment need not make a visual vector exactly equal to a particular word vector. Image-conditioned text prediction can indirectly discover visual representations that the language model can use.

**How can learning happen when backbones are frozen?** Freezing means their parameters do not change; the backbones still participate in computation. Answer errors must pass through the language model back to the connector, indicating how to adjust the input. If that feedback path is cut, the connector cannot learn from the final answer. [PyTorch autograd documentation](https://docs.pytorch.org/docs/2.14/notes/autograd.html#locally-disabling-gradient-computation)

### 6.4 Multimodal Continued Pretraining: Expanding Cooperation

After connector adaptation, handling documents, charts, complex spatial relationships, multiple images, or video usually requires broader data and consideration of a wider trainable scope.

```text
Captions + Image-text documents + OCR + Charts + Video + Text-only data, etc.
                                      ↓
                       Multimodal continued pretraining
                                      ↓
         Update the connector and language model according to the plan;
                   update the visual component if needed
```

“Joint” means that relevant components adapt together under a multimodal task objective. Whether to unfreeze the visual encoder depends on the adequacy of its existing features, data volume, and compute budget.

Mixing in text-only data can help preserve language capabilities. High-resolution or long-input training helps the model adapt to new visual and context conditions. However, data proportions, input lengths, and freezing strategies require experiments; another model's recipe cannot simply be copied unchanged.

Qwen3-VL's published training includes connector warm-up followed by stages that update all parameters. InternVL3.5 also reports extensive joint image-text pretraining. Modern VLM training does not always stop at the connector. [Qwen3-VL](https://arxiv.org/abs/2511.21631), [InternVL3.5](https://arxiv.org/abs/2508.18265)

### 6.5 Visual SFT: Turning Image-Text Capabilities into Task Behavior

Captioning answers “What is in this image?” Visual SFT additionally asks the model to understand the current instruction: read a specified field, compare two images, output only a number, or explain that the evidence is insufficient.

| Training example format | Main capability learned |
|---|---|
| Image + General description | Organize the overall content |
| Image + Specific question + Answer | Select visual evidence according to the question |
| Document + Field extraction instruction + Structured result | Read text and follow an output format |
| Multiple images + Comparison question + Answer | Track image ownership and connect differences |
| Image + Unanswerable question + Appropriate response | Distinguish visible information from unsupported inference |

SFT commonly supervises the assistant's answer: the image and question are readable context, while the ideal answer is used to evaluate predictions. **Excluding image positions from answer loss does not mean preventing the model from reading them.**

Alignment and SFT can use similar text generation objectives. Their main differences are data content, task format, and parameter update scope; they do not necessarily require completely different algorithms.

### 6.6 Preference Optimization and Reinforcement Learning: Improving Through Comparisons or Outcomes

Once basic image-text capabilities and task behavior are established, feedback can further adjust answers.

```text
Same image + Same question
              ↓
     Generate candidate answers
              ↓
Compare their quality, or check whether the task result is correct
              ↓
Increase the probability of more reliable behavior
```

- **Preference optimization:** use comparisons between better and worse answers; DPO is a common method.
- **Reinforcement learning:** assign rewards to sampled answers and update the policy accordingly; PPO and GRPO are available optimization methods.
- **Human or AI feedback:** describes the feedback source, often referred to as RLHF or RLAIF. This is a different dimension from the optimization algorithm.

Sources: [DPO](https://arxiv.org/abs/2305.18290), [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300), [RLAIF-V](https://arxiv.org/abs/2405.17220).

Visual feedback must actually check the image. Rewarding only fluent, detailed wording may improve expression while encouraging fabrication. Verifiable tasks can use reading, calculation, or localization results; open-ended descriptions require more reliable evaluation and filtering.

These methods are not mandatory stages for every VLM and cannot replace foundational perception or image-text alignment.

<a id="recipes"></a>

## 7. How Are Common Training Recipes Assembled?

The previous section explained training objectives. This section shows how they combine into practical plans. Not every model follows the same fixed pipeline.

### 7.1 Recipe One: Alignment Followed by Visual Instruction Tuning

```text
Existing visual encoder + Existing language model + New connector
                               ↓
Stage 1: Adapt the connector using image-caption data
                               ↓
Stage 2: Train task behavior using visual instruction data
                               ↓
             Evaluate and make targeted improvements
```

The classic LLaVA approach follows this staged design: alignment focuses on the projection module, then instruction tuning also updates the language side while the visual backbone stays frozen. The structure is direct and the stage objectives are clear, making it useful for understanding basic generative VLM training. [LLaVA](https://arxiv.org/abs/2304.08485)

This recipe reuses the capabilities of two established backbones. Its limits still depend on visual representations, the language base, and instruction data. Completing two stages alone does not establish mastery of complex tasks.

### 7.2 Recipe Two: Freeze the Backbones and Train a More Capable Bridge

These approaches place more adaptation capacity in newly added modules. The backbones can remain frozen while the bridge performs more complex information extraction and interaction.

BLIP-2 first trains a Q-Former to learn text-related visual representations, then connects it to a frozen LLM for generative adaptation. The earlier stage includes tasks such as image-text matching, not just ordinary caption training. [BLIP-2](https://arxiv.org/abs/2301.12597)

Flamingo trains its visual resampler and new cross-attention modules while freezing the original visual and language backbones. Its “new components only” training therefore covers more structure than a two-layer MLP alone. [Flamingo](https://arxiv.org/abs/2204.14198)

When a recipe says the backbones are frozen, also check exactly which new components are being trained.

### 7.3 Recipe Three: Joint Multimodal Training Followed by Post-Training

```text
Initialize visual and language components
                    ↓
Optional connector warm-up
                    ↓
Large-scale multimodal continued pretraining
                    ↓
Visual instruction SFT
                    ↓
Optional preference optimization, distillation, or reinforcement learning
```

These approaches often aim to cover a broader set of tasks, jointly designing visual inputs, data mixtures, long contexts, and parameter update scope. The published Qwen3-VL and InternVL3.5 recipes reflect this broader approach, although their stages and details differ. [Qwen3-VL](https://arxiv.org/abs/2511.21631), [InternVL3.5](https://arxiv.org/abs/2508.18265)

They offer greater adaptation freedom but require stronger data, more compute, and evaluation for capability regression. Updating all parameters is not a quality guarantee; it only permits wider changes.

### 7.4 LoRA and Partial Unfreezing Are Update Methods, Usable Across Recipes

| Update method | What changes | Main characteristics |
|---|---|---|
| Connector only | Cross-modal interface | Few new parameters, limited adaptation scope |
| Partial unfreezing | Connector and selected backbone layers | Trade-off between resources and adaptability |
| LoRA | Low-rank adaptation parameters in selected layers | Reduces the number of parameters to update |
| Full-parameter fine-tuning | All parameters of the selected backbone | More freedom, higher resource and data requirements |

LoRA is not a training stage alongside SFT. A model can use LoRA for visual SFT or use adapters in other stages. QLoRA also stores the backbone at low precision to reduce associated memory costs, but does not eliminate all costs of visual inputs and long sequences. [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314)

Similarly, a teacher model can generate training answers, filter synthetic data, or provide feedback at multiple stages. Synthetic content still needs visual fact-checking: teachers can also make visual errors.

### 7.5 Compare at Least Four Aspects

| Aspect | Questions to ask |
|---|---|
| Initialization | What capabilities do the visual and language backbones already have? |
| Update scope | Interface only, partial unfreezing, or joint training? |
| Data and objectives | Captions, specific questions, long image-text inputs, or preference feedback? |
| Inputs and evaluation | Do resolution, visual positions, and multi-image formats match the target task? |

This separates what the network looks like, which capabilities it is learning, and which parameters may change. A model name or script name alone can otherwise be misleading.

<a id="data-eval"></a>

## 8. How Do Data and Evaluation Support Training?

### 8.1 Data Quality Affects the Entire Information Path

One image can correspond to several questions, so sample rows do not equal independent images. Training data should also be checked for task diversity, language distribution, image clarity, and whether answers are supported by the image.

Large quantities of broad captions cannot replace training for detailed reading. Text that becomes unreadable after resizing is also unsuitable as supervision demanding accurate reading. Check data and input processing together.

### 8.2 Learning Training Answers Does Not Guarantee Reliable Answers to New Questions

During training, the model can usually use the preceding part of a reference answer to practice predicting what follows. In actual use there is no reference answer; generation must continue from the model's own output. Lower training error therefore still needs to be accompanied by checks of free generation.

Validation should also avoid images that are identical or highly similar to training images. Group documents and videos by source so that validation does not merely revisit familiar content.

### 8.3 Use Visual Controls to Check Whether Images Matter

```text
Fixed question: “How much are the beef noodles?”

Menu A shows 39 yuan → Ideal answer: 39 yuan
Menu B shows 49 yuan → Ideal answer: 49 yuan
Menu C has a blurry price → Ideal answer: Cannot determine
```

This illustrates an evaluation design, not a measured model result. The key is whether the answer changes appropriately with visual evidence.

Comparing prediction errors under correct and mismatched images can also diagnose whether the visual input has an effect. That difference is not question-answering accuracy and cannot alone establish fine-grained understanding.

### 8.4 Use Errors to Locate Bottlenecks

| Symptom | What to inspect first |
|---|---|
| Recognizes broad categories but cannot read small text | Resolution, visual features, and OCR data |
| Reads content correctly but associates it with the wrong object | Spatial relationships, question understanding, and task supervision |
| Answers barely change when the image changes | Image integration, alignment, and reliance on language priors |
| Confidently answers questions without supporting evidence | Labels, instruction tuning, and feedback criteria |
| Single-image results are fine but multiple images become confused | Image ownership and multi-image training |
| New tasks improve while existing capabilities decline | Update scope, data mixture, and forgetting |

Architecture supplies the information path, training develops the ability to use it, and evaluation checks reliability on new inputs. Analyze all three together.

<a id="references"></a>

## 9. Original Sources and Reading Order

Start with ViT and LLaVA to understand image representations and a minimal generation pipeline. Then read BLIP-2 and Flamingo to compare bridging and fusion. Finally, study modern VLM input designs and training stages.

| Topic | Original sources |
|---|---|
| Visual representations and image-text foundations | [ViT](https://arxiv.org/abs/2010.11929), [CLIP](https://arxiv.org/abs/2103.00020), [SigLIP](https://arxiv.org/abs/2303.15343), [SigLIP 2](https://arxiv.org/abs/2502.14786) |
| Simple fusion and two-stage training | [LLaVA](https://arxiv.org/abs/2304.08485), [LLaVA-1.5](https://arxiv.org/abs/2310.03744) |
| Learned bridges and cross-attention | [BLIP-2](https://arxiv.org/abs/2301.12597), [Flamingo](https://arxiv.org/abs/2204.14198), [Pix2Struct](https://arxiv.org/abs/2210.03347) |
| Multiple visual settings and expanded training | [OneVision](https://arxiv.org/abs/2408.03326), [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), [Qwen3-VL](https://arxiv.org/abs/2511.21631), [InternVL3.5](https://arxiv.org/abs/2508.18265) |
| Parameter adaptation and feedback optimization | [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314), [DPO](https://arxiv.org/abs/2305.18290), [RLAIF-V](https://arxiv.org/abs/2405.17220) |

These recipes classify published research. They do not imply that all models use the same process, and they are not a leaderboard comparison.

<a id="project"></a>

## 10. The Architecture Used in This Project

miniLLM-vlm uses input-level fusion with existing visual and language weights. Only the connector is trained for the current Pretrain alignment stage.

```text
Image: 256 × 256
          ↓
SigLIP2 visual encoder [frozen]
          ↓
64 visual positions, each with width 768
          ↓
LayerNorm → Linear → GELU → Linear [trainable]
          ↓
Replace 64 image placeholder embeddings in the input sequence
          +
Text embeddings of the question
          ↓
miniLLM Base [frozen]
          ↓
Generate a text answer; propagate training feedback to the connector
```

| Component | Current choice |
|---|---|
| Visual module | Compatible export of fixed-resolution SigLIP2 visual weights; 32×32 patches form an 8×8 grid |
| Connector | Two-layer MLP with normalization, approximately 1.18 million trainable parameters; preserves 64 visual positions |
| Language module | miniLLM Base; the recorded run summary reports 8 layers, width 768, and an MoE with 4 experts |
| Image integration | Reuse an existing placeholder token and replace its embeddings, without expanding the vocabulary |
| Current training | Freeze both backbones and train only the connector; adapt through assistant-answer prediction while retaining the Base model's MoE auxiliary term |

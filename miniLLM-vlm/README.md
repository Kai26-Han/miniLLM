# miniLLM-vlm：Pretrain 与 SFT

在已训练的 miniLLM Base 上接入 SigLIP 视觉编码器。参考
[MiniMind-V](https://github.com/jingyaogong/minimind-v)，冻结语言模型和视觉编码器，
只训练 `LayerNorm → Linear → GELU → Linear` 投影层。

Pretrain 使用单图描述数据；SFT 在其产物上训练单图问答、多轮对话和纯文本指令。
实现针对 AutoDL 单张 RTX 4090，
25 核 CPU、90 GB 内存；多进程 DDP、LoRA、量化和 torch.compile 不在这一版范围内。

## 1. 资源目录

本项目可独立放在 AutoDL 的 `/root/miniLLM-vlm`，只需上传本文件夹内的代码。
本地位于 `miniLLM/miniLLM-vlm` 只是存放位置；运行、测试和代码指纹均不读取上级 miniLLM 源码。

```text
miniLLM-vlm/
├── model/
│   ├── model_minillm.py       # 本项目独立维护的 Base 架构实现
│   ├── model_vlm.py           # SigLIP、Projector 与语言模型组合
│   ├── miniLLM-base/          # 完整 Base 权重、配置与配套 tokenizer
│   └── siglips/               # 已下载的 SiglipVisionModel
├── dataset/
│   ├── vlm_dataset.py
│   └── pretrain_i2t.parquet
├── trainer/
│   ├── train_pretrain_vlm.py  # Pretrain 预检与训练入口
│   ├── train_sft_vlm.py       # SFT 预检与训练入口
│   └── trainer_utils.py      # 本项目自包含的资源、日志、调度和断点工具
├── minillm_vlm/__init__.py    # 只映射本项目内部目录的 Python 包入口
├── eval/eval_vlm.py
├── tests/test_pretrain_vlm.py
├── tests/test_sft_vlm.py
├── requirements.txt
├── README.md
├── .gitignore
├── cache/                    # 运行时创建：Arrow 缓存和数据索引
├── checkpoints/pretrain/
└── out/pretrain/
```

`model/model_minillm.py` 保留 Base 的配置字段和权重参数名，并在本项目内支持
`inputs_embeds`。训练工具也由本项目独立维护。没有跨项目导入、软链接或父项目路径配置；
不需要安装、上传或修改 miniLLM 项目的任何源码。
`minillm_vlm/__init__.py` 仍是本项目自身的必要文件，用于统一包名和多进程导入。

`miniLLM-base` 使用已有 miniLLM Base 的 Hugging Face 导出目录，至少包含：

- `config.json`，其中 `model_type` 是 `minillm`；
- `model.safetensors`，或完整权重分片与索引；
- 配套的 `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`；
- 导出时附带的其他 tokenizer 文件保持原样。

不要将其他模型的 tokenizer 拼入 Base，不要把训练断点 `.pt/.pth` 改名冒充导出权重。
本入口直接读取完整导出目录，不读取任意来源的裸 state_dict。

`siglips` 应包含 `config.json`、权重和 `preprocessor_config.json`。
本地已核对的配置是 `SiglipVisionModel`、P32、256×256、hidden_size=768，输出 64 个 patch token。
运行时仍会用真实 processor 和图像前向核验形状。视觉权重与 Base 都是独立依赖。

所有命令行的**相对路径均相对 miniLLM-vlm 根目录**解析，不受启动时工作目录影响。
可用 `--base-model`、`--vision-model`、`--tokenizer-path` 和 `--data-path` 指向其他绝对路径。
默认 tokenizer 就在 Base 目录中。代码只加载本地模型文件，不自动下载或替换资源。

## 2. 环境

PyTorch 由 AutoDL 的 GPU 环境单独管理，核心接口要求 `>=2.3,<3.0`。
进入 `/root/miniLLM-vlm`，先检查当前解释器中的版本和 CUDA 可用性：

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
nvidia-smi
```

版本满足范围且 `CUDA available: True` 时，保留现有 PyTorch，再安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 不再包含 PyTorch 安装条目，避免常规依赖安装选择新版 torch 并连带下载
整套 CUDA/cuDNN/NCCL。若 PyTorch 缺失、版本不在支持范围或 CUDA 不可用，需根据
当前 Python、操作系统和 NVIDIA 驱动单独选择匹配的 GPU 版本，再执行预检。
目前 AutoDL 上的 `2.3.0+cu121` 已报告 CUDA 可用，并已通过此前 GPU 预检；
该版本已有训练入口使用的 [torch.amp.GradScaler 接口](https://github.com/pytorch/pytorch/blob/v2.3.0/torch/amp/grad_scaler.py)。
该 AutoDL 环境随后已完成 Pretrain 100 步短跑、验证和断点保存。SFT 改变了训练参数范围，
仍需单独执行 SFT 预检和短跑；本地自动测试使用的 PyTorch 是 2.14.0。
本项目支持的 Transformers 范围是 `>=4.51,<4.56`，按本项目的 `requirements.txt` 安装。

如果已通过 GPU 预检，只是要开启 SwanLab 监控，安装并登录 SwanLab 即可：

```bash
python -m pip install "swanlab>=0.9.5,<1.0"
swanlab login
```

如果 Base 的 `tokenizer_config.json` 声明 `tokenizer_class: TokenizersBackend`，
Transformers 4.x 的 AutoTokenizer 可能报类不存在。本项目会为这个已知类名使用
`PreTrainedTokenizerFast.from_pretrained` 读取原始 `tokenizer.json`、特殊 token 和聊天模板。
这一命名变化见 [Hugging Face 的 tokenizer 后端说明](https://huggingface.co/docs/transformers/fast_tokenizers)。
加载器不修改或重新保存 Base/tokenizer 文件，也不替换未知自定义 tokenizer；资源指纹校验保持有效。
底层库可能仍打印这两个类名不同的提示，之后会继续执行预检；词表、特殊 token 或模板不匹配仍会报错停止。

## 3. 先做资源预检

```bash
python trainer/train_pretrain_vlm.py --check-only --device cuda --samples 8
```

预检会核对完整权重加载、词表/模板身份、图像特征形状，并用前 8 条数据真实前向和反向。
它要求只有 Projector 获得梯度，冻结模块不获得参数梯度；不会执行优化器更新。
结果默认保存到 `out/preflight.json`。`--check-only` 模式不会初始化 tracker、优化器或训练断点；
可在已有训练输出目录存在时单独执行，检查的是原 Base 与新初始化的 Projector。
`--prepare-index` 只与 `--check-only` 组合使用；正式训练会自动准备或复用索引。

再建立可复用的全量数据索引：

```bash
python trainer/train_pretrain_vlm.py --check-only --device cuda --prepare-index
```

这一遍会读取、解码、tokenize 全部样本，报告坏图、非法对话、长度分布和截断数量。
数据量大时需要时间，也会生成磁盘上的 Arrow 缓存，但不会把全部图片载入内存。
图片保持在 Parquet/Arrow 中，索引只保存有效行号、划分和统计信息。

索引按图片字节 SHA-256 和固定 seed 划分约 1% 验证图片。同一图片二进制的中英文描述不会跨集合；
不同编码但视觉相同的图片不保证自动识别，需要数据源级去重。后续 SFT 必须沿用图片隔离规则。

过滤的异常样本计数和前 20 个原因写入索引。已确认有效的样本若在训练期间读取失败，则报错停止，
不会偷偷换成另一条数据。改变数据内容、顺序、序列长度、tokenizer、划分 seed 或代码后，
应指定新的 `--index-path`，不能复用旧索引。

小数据可能没有验证图片；应增加唯一图片数量或提高 `--val-ratio`，不从训练集复制验证样本。

## 4. 数据与目标

支持一个或多个 Parquet 文件：

```bash
python trainer/train_pretrain_vlm.py --check-only --device cuda \
  --data-path dataset/part-000.parquet dataset/part-001.parquet \
  --prepare-index --index-path cache/shards_index.json
```

必需字段为 `image_bytes` 和 `conversations`。图像可以是 bytes 或只有一个元素的 bytes 列表。
对话可为 JSON 字符串或结构化列表，格式为可选 system、一个 user、一个 assistant：

```json
[
  {"role": "user", "content": "<image>\n请描述这张图片。"},
  {"role": "assistant", "content": "一只棕色小狗正在草地上奔跑。"}
]
```

保留原始中英文描述。使用原 miniLLM ChatML 模板，不随机添加 system 或空 think 标签。
唯一的 `<image>` 必须在 user 内容中；内部转为已有 `<|reserved_0|>` 的连续占位位置。
不新增词表条目，不改变 tokenizer 文件。

输入由图像 token、用户问题、assistant 前缀和答案组成。只有答案及真实 EOS 被监督，
其他 label 为 `-100`。图像 attention mask 为 1，padding 为 0。每个 batch 动态右侧 padding。
最长 512 token **包含图像和文本全部位置**；只截断答案，不截断图像或问题，截断处不补假 EOS。

损失为 caption 交叉熵加 Base 原有 Router 辅助项，辅助项只加一次。
`--router-aux-weight 0` 可做只优化 CE 的对照；默认 1 保留 Base 的系数。
语言模型参数冻结，经过语言模型的输入梯度仍保留，从而更新前面的 Projector。

## 5. 短跑与正式训练

先用 1 万条随机训练样本做短跑。使用独立输出目录，不与正式训练覆盖：

```bash
python trainer/train_pretrain_vlm.py --device cuda \
  --max-train-samples 10000 --max-steps 100 \
  --eval-interval 50 --save-interval 50 \
  --save-dir checkpoints/smoke --output-dir out/smoke
```

正式训练：

```bash
python trainer/train_pretrain_vlm.py --device cuda \
  --epochs 1 --batch-size 8 --accumulation-steps 8 \
  --learning-rate 4e-4 --min-learning-rate 4e-5 \
  --warmup-ratio 0.03 --max-seq-len 512 \
  --num-workers 4 --tracker swanlab \
  --tracker-run-name minillm-vlm-pretrain
```

默认 AdamW，矩阵权重衰减 0.01，bias/LayerNorm 不衰减，梯度裁剪 1.0，BF16。
单卡有效 batch=8×8=64；显存和吞吐实测后可在**新实验**改成 16×4。
默认不启用外部 tracker；上面正式命令显式启用 SwanLab。
无网络时可用 `--tracker swanlab --tracker-mode offline`，或 `--tracker none`。

步数均指优化器更新次数。每次更新按本组有效答案 token 数归一化 CE，尾部不足的累积组正常更新。
日志间隔 10，评估与保存间隔 500，epoch 结束及到达停止步数时也执行。
训练日志含 CE、辅助项、梯度范数、专家占比、学习率、样本吞吐和 CUDA 显存峰值。

为保证可对照，训练启动会对模型和数据做内容哈希。哈希和全量数据预处理耗时不计入训练吞吐。
第一次运行比后续运行慢；吞吐与显存需要在真实 AutoDL 数据上测量。

## 6. 断点恢复

使用与原实验相同的参数，再加 `--resume`：

```bash
python trainer/train_pretrain_vlm.py --device cuda \
  --epochs 1 --batch-size 8 --accumulation-steps 8 \
  --learning-rate 4e-4 --min-learning-rate 4e-5 \
  --warmup-ratio 0.03 --max-seq-len 512 \
  --num-workers 4 --tracker swanlab \
  --tracker-run-name minillm-vlm-pretrain --resume
```

默认恢复 `checkpoints/pretrain/latest.pt`；也可以 `--resume /绝对路径/latest.pt`。
断点保存 Projector、优化器、调度器、Scaler、随机状态、epoch、下一 batch 位置及 tracker ID。
Base/视觉编码器不重复保存，但恢复时严格校验其权重、tokenizer、代码、数据与训练配置身份。
重新启动后通过确定性 sampler 直接跳过已提交的 batch。

`--max-steps` 定义固定学习率调度终点，恢复时不能改大。
需要测试暂停/恢复时，用 `--stop-after-steps 50` 在第 50 次更新后安全暂停；恢复时去掉该参数，
其余参数保持一致。改变 epoch、batch、累积、学习率或数据属于新实验，不是等价续训。

Ctrl+C、掉电或异常后从上次**原子保存成功**的断点恢复，最多重做一个保存间隔的更新。
不会把只完成部分反向或部分更新的状态保存为有效断点。仅加载自己生成的可信训练 checkpoint。

### 独立项目版本与旧产物

本版将 Base 实现和必要训练工具全部纳入 VLM，代码指纹只覆盖本项目源码。
模型参数结构和训练目标保持一致，可以直接使用已有 Base 权重与配套 tokenizer。
因为代码指纹包含源文件内容，**重构前生成的索引、断点和 Projector 产物不会直接通过新版身份校验**。
旧实验请继续使用生成它的那版代码；本版新实验使用新的 `--index-path`、`--save-dir` 和 `--output-dir`。
无需重新下载 Base、视觉编码器和数据；不手动覆盖指纹，也不绕过资源校验。

## 7. 评估与产物

每次验证报告按监督 token 加权的 caption CE，并在 batch 内置换图片，比较相同文本目标的正确图/错配图 CE。
相同预处理图片的对照会排除；batch=1 时没有错配比较。`paired/wrong_minus_correct` 越明显为正，
越能支持模型确实使用图片的判断，但它不是独立的视觉能力综合评分。

训练产物：

| 文件 | 用途 |
|---|---|
| `checkpoints/pretrain/latest.pt` | 最近一次可恢复断点 |
| `checkpoints/pretrain/best.pt` | 最佳验证 CE 对应的可恢复断点 |
| `out/pretrain/best_adapter.pt` | 最佳 Projector 权重、VLM 配置和资源身份 |
| `out/pretrain/last_adapter.pt` | 最新导出的 Projector |
| `out/pretrain/run_config.json` | 实际参数、数据统计、资源与恢复契约 |
| `out/pretrain/metrics.jsonl` | 本地训练/验证指标 |

Projector 产物不包含冻结的 Base 或 SigLIP，推理与后续 SFT 仍需要它们。
单图描述（路径相对 VLM 根目录）：

```bash
python eval/eval_vlm.py --device cuda \
  --adapter out/pretrain/best_adapter.pt \
  --image images/example.jpg --prompt '请详细描述这张图片。' \
  --max-new-tokens 128
```

验证集与错配图评估：

```bash
python eval/eval_vlm.py --device cuda \
  --adapter out/pretrain/best_adapter.pt \
  --eval-samples 2000 --output out/pretrain/evaluation.json
```

若训练改过数据、长度、划分或索引路径，评估时传入相同数据参数。
迁移到其他机器时显式传入新的 `--base-model`、`--vision-model` 和 `--tokenizer-path`；
内容哈希相同即可更换资源绝对路径。解码使用贪心搜索，图像只在 prefill 编码一次。

## 8. 本地测试

在已安装依赖的环境执行：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

测试使用随机初始化的小型语言模型/SigLIP、合成图片和现场生成的测试 tokenizer，
不读取其他项目文件、不需要真实 Base，也不下载模型或 tokenizer。
覆盖输入 embedding 与原文本模型一致性、KV cache、冻结和 Projector 梯度、图像依赖、标签与截断、
重复图片划分、梯度累积，以及完整训练与暂停恢复的参数一致性。

独立版本验证记录（2026-09-08）：

- CPU 隔离环境：Python 3.12.9、PyTorch 2.14.0、Transformers 4.55.4、datasets 4.8.5。
- 9 项测试全部通过，覆盖 embedding/KV cache、冻结和梯度、标签、索引、梯度累积、预检、训练恢复和评估。
- 其中一项只复制本项目的 8 个 Python 文件到独立临时目录，清除外部 Python 路径配置，关闭 Hugging Face 网络访问，拦截外部 `model` / `trainer` / `dataset` 导入，再运行全部 8 项功能测试。
- 新增 TokenizersBackend 类名兼容回归：复现 4.x AutoTokenizer 报错，验证兼容加载后的词表、分词结果、特殊 token、聊天模板及源文件字节均保持一致；未知自定义类继续拒绝加载。训练/恢复/评估集成测试也使用该类名导出的 tokenizer。
- 隔离运行覆盖 spawn DataLoader 和从其他工作目录启动评估；测试 tokenizer 现场生成。连续训练与跨 epoch 断点恢复后的 Projector 参数逐项完全一致。
- 本次修改仅位于 `miniLLM-vlm`，改动前后校验确认原 miniLLM 项目的 Python 源码未变。迁入的模型定义、必要工具函数和原训练更新逻辑经 AST 对照保持一致。
- 小模型测试只验证代码链路，不表示真实视觉能力达标。尚未在 AutoDL 的真实 Base、全量数据及 CUDA BF16 环境执行训练；按上述预检和短跑顺序验证。


## 9. SFT：继承正式 Pretrain 产物

本次新增 `trainer/train_sft_vlm.py` 和 `tests/test_sft_vlm.py`，其余能力合并进现有文件：

| 文件 | SFT 职责 |
|---|---|
| `trainer/train_sft_vlm.py` | 阶段参数、预检、训练循环、完整断点、固定问题生成记录 |
| `model/model_vlm.py` | `MiniLLMSFT`：首尾语言层解冻、混合图文/纯文本前向 |
| `dataset/vlm_dataset.py` | 多轮 ChatML 标签、混合 batch、SFT 索引与扫描恢复 |
| `trainer/trainer_utils.py` | 跨阶段权重校验、SFT 分组验证、完整模型导出和加载 |
| `trainer/train_pretrain_vlm.py` | 共享参数解析器和 token 加权更新；梯度裁剪覆盖所有可训练参数 |
| `eval/eval_vlm.py` | 同时支持 Pretrain Projector 和 SFT 完整模型的推理/评估 |
| `tests/test_sft_vlm.py` | 离线合成数据、首尾层梯度、恢复一致性与无父项目运行测试 |

训练策略参考 [MiniMind-V SFT 入口](https://github.com/jingyaogong/minimind-v/blob/master/trainer/train_sft_vlm.py)
及其 [冻结策略实现](https://github.com/jingyaogong/minimind-v/blob/master/trainer/trainer_utils.py)：

- 默认 `--freeze-llm 1`：训练 Projector，以及 LLM 第一层和最后一层完整 decoder block。
  你的 8 层模型对应第 0、7 层，包含其中的 MoE 专家和 Router；中间层、embedding、末尾 norm、输出头冻结。
- `--freeze-llm 0`：训练全部 LLM 与 Projector；显存占用需重新预检。
- `--freeze-llm 2`：仅训练 Projector，适用于只含图像的数据，不用于混合纯文本的 SFT。
- SigLIP 始终冻结并保持 eval；冻结中间语言层保留反向计算路径。
- 默认 1 epoch、BF16、batch=4、累积=16（有效 batch=64）、最长 768 token、4 个 workers。
- AdamW，学习率 5e-6、最低 5e-7、warmup 3%、cosine、weight decay 0.01、grad clip 1。
  CE 按有效答案 token 归一化，Router 辅助项只加一次。

### 数据规则

使用已下载的 SFT Parquet，不再额外拼接 Pretrain 数据。列仍为 `conversations` 和 `image_bytes`，
可选 `task_type` 为 `instruction`、`caption`、`text`。支持 role/content 或 human/gpt 的 from/value 格式。
对话必须是可选 system 加完整 user/assistant 对；单图 `<image>` 放在第一个 user 中。
所有 assistant 正文及真实 EOS 参与监督，system/user、角色头、图片及 padding 不监督。
不随机插入 system/think，不扩充词表。长度预算包含 64 个视觉 token。

无 `<image>` 的样本按纯文本处理，不读取其图片字段。明确标记 `task_type=text` 的样本会移除 user
中的图片占位标记并跳过图片；不会仅凭黑色像素猜测纯文本，避免误删真实黑图。
如果你的下载版本给纯文本加了 `<image>`，却没有 text 元数据，需先提供可靠来源标记；
当前实现会将这些带标记行当作图文样本。预检报告实际视觉/文本数量，索引也分别统计。
没有 `task_type` 时，图文统称 `image`，不会从问题内容猜测 caption/instruction。

截断保留对话前缀和第一张图：后续 user 问题放不下时舍弃该轮及后续轮次；assistant 回答可在
预算处截断，截断末尾不补 EOS。首轮问题后没有空间容纳任何监督答案的样本计为无效。
这版不展开滑动窗口；被舍弃的较晚对话不会自动生成新训练样本。

沿用 Pretrain 的图片字节哈希、seed=42、val-ratio=0.01，防止同一图片的两阶段划分相互冲突。
如果 Pretrain 改过 seed/val-ratio，SFT 必须使用对应值。纯文本按规范化完整对话哈希划分，
不按所有样本共用的黑图划分。不同编码的重复图片仍需数据源级去重。
SFT 建立新索引，不复用 Pretrain 行号。全量扫描每 12800 行写入 `.partial` 文件，
同参数重启会从已提交位置继续，最多重扫尚未提交的部分。

### 在 AutoDL 先预检

先等正式 Pretrain 完成，再同步本版完整源码并保留旧版代码备份。
共享文件升级会改变代码指纹：旧 Pretrain 续训/评估仍使用旧版本；
SFT 的跨阶段加载器能读取旧 `best_adapter.pt`，严格核对 Base、SigLIP、tokenizer、VLM 配置、
Projector 参数名/形状/有限值，并记录来源文件哈希与步数。它不会恢复 Pretrain 的优化器。
SFT 自身的 `--resume` 仍严格校验代码、数据、初始化来源及训练配置。

下面假设正式产物是 `out/pretrain_swanlab/best_adapter.pt`，数据文件名是 `sft_i2t.parquet`；
如果实际名称不同，修改对应两个参数，不能用 smoke 产物代替正式产物。

```bash
cd /root/miniLLM-vlm
python trainer/train_sft_vlm.py \
  --check-only --device cuda --samples 8 \
  --from-pretrain out/pretrain_swanlab/best_adapter.pt \
  --data-path /root/autodl-tmp/miniLLM-vlm/dataset/sft_i2t.parquet
```

预检不创建优化器、tracker 或训练断点；报告 `out/sft_preflight.json`。
它对实际选中样本执行前向/反向，核验冻结状态和梯度；这些样本未包含的类型由离线测试覆盖，
仍应结合全量索引中的类型统计检查真实数据格式。

### 100 步短跑（单独的抽样索引）

```bash
python trainer/train_sft_vlm.py \
  --device cuda --from-pretrain out/pretrain_swanlab/best_adapter.pt \
  --data-path /root/autodl-tmp/miniLLM-vlm/dataset/sft_i2t.parquet \
  --scan-samples 20000 --index-path cache/sft_smoke_index.json \
  --max-train-samples 10000 --max-steps 100 --eval-samples 200 \
  --eval-interval 50 --save-interval 50 \
  --tracker swanlab --tracker-run-name sft-smoke-4090 \
  --save-dir checkpoints/sft_smoke --output-dir out/sft_smoke
```

`--scan-samples` 固定抽样后仅验证这些行，可减少图像解码和分词；首次 Parquet→Arrow 缓存
及文件内容哈希仍可能读取全文件。`--max-train-samples` 只限制索引建成后的训练集合。
短跑结束确认 loss/梯度有限、首尾层与 Projector 有梯度、验证和保存成功，再开始正式实验。
Pretrain 的显存结果不能代替 SFT 的实测显存。

### 正式训练并在 SwanLab 监控

已经安装并登录 SwanLab 的环境无需重新安装或下载 PyTorch。

```bash
python trainer/train_sft_vlm.py \
  --device cuda --dtype bfloat16 \
  --from-pretrain out/pretrain_swanlab/best_adapter.pt \
  --data-path /root/autodl-tmp/miniLLM-vlm/dataset/sft_i2t.parquet \
  --freeze-llm 1 --epochs 1 --max-seq-len 768 \
  --batch-size 4 --accumulation-steps 16 \
  --learning-rate 5e-6 --min-learning-rate 5e-7 --warmup-ratio 0.03 \
  --num-workers 4 --index-path cache/sft_index.json \
  --log-interval 10 --eval-interval 500 --save-interval 500 \
  --tracker swanlab --tracker-project miniLLM-VLM-SFT \
  --tracker-run-name sft-4090 \
  --save-dir checkpoints/sft --output-dir out/sft
```

正式实验不传 `--scan-samples`、`--max-train-samples` 或 `--max-steps`，从正式 Pretrain 初始化，
不继续 smoke 的优化器轨迹。更大 microbatch 可在短跑确认显存后用于新正式实验。

续训使用**相同正式命令加 `--resume`**。保留同一个 Pretrain 产物作为来源校验，
实际训练参数由 SFT checkpoint 中的 LLM/Projector 覆盖恢复。测试暂停可加
`--stop-after-steps 50`，恢复时移除此参数再加 `--resume`；不能改变调度终点。

SwanLab 项目 `miniLLM-VLM-SFT` 记录训练 CE、Router 辅助项、专家占比、LLM/Projector 梯度范数、
学习率、吞吐、显存以及分组验证损失。初始化发生在索引完成之后。
`val/selection_loss` 优先用显式 instruction 组，其次所有图文组，无图时才用总体损失；
分组计数同时记录。错配图比较只针对真实视觉行，并排除相同预处理图片，纯文本不进入该比较。
固定验证样本的生成结果在 step 0 和每次验证时追加到本地 `generations.jsonl`；
默认 4 条、最多生成 64 token，方便对比幻觉、重复和语言质量，文本内容暂不上传 SwanLab。

### SFT 产物与推理

| 文件 | 内容 |
|---|---|
| `checkpoints/sft/latest.pt` / `best.pt` | 完整 LLM、Projector、优化器、scheduler、Scaler、RNG、进度与 tracker ID |
| `out/sft/best_sft.pt` / `last_sft.pt` | 完整更新后的 LLM、Projector、配置、来源与资源身份；用于推理 |
| `out/sft/tokenizer/` | 从同一 tokenizer 导出的配套文件，不修改原 Base 文件 |
| `out/sft/run_config.json` | 实际参数、可训练参数清单、索引统计与续训契约 |
| `out/sft/metrics.jsonl` | 数值指标 |
| `out/sft/generations.jsonl` | 固定验证问题、参考答案与生成结果 |

推理不需要原 Base 权重，但需要 `best_sft.pt`、同目录 `tokenizer/` 与原 SigLIP 文件夹。
语言模型权重已经全部包含在 SFT 产物中。迁移时保留这组资源；可通过 `--vision-model`、
`--tokenizer-path` 指定新位置，加载器核对文件身份。

```bash
python eval/eval_vlm.py --device cuda \
  --sft-model out/sft/best_sft.pt \
  --image images/example.jpg --prompt '图片中有什么？' --max-new-tokens 128

python eval/eval_vlm.py --device cuda \
  --sft-model out/sft/best_sft.pt --text --prompt '请用中文介绍一下你自己。'
```

多轮推理可传 `--messages messages.json`，内容为以 user 结尾的 ChatML 消息列表。
与 `--image` 同用时若没有图片标记，会在第一个 user 中插入标记；与 `--text` 同用则不插入。

验证集评估：

```bash
python eval/eval_vlm.py --device cuda \
  --sft-model out/sft/best_sft.pt \
  --data-path /root/autodl-tmp/miniLLM-vlm/dataset/sft_i2t.parquet \
  --max-seq-len 768 --index-path cache/sft_index.json \
  --output out/sft/evaluation.json
```

评估 smoke 索引时同步传入 `--scan-samples 20000` 及 `cache/sft_smoke_index.json`；
数据、长度、seed、val-ratio 必须与对应实验一致。


SFT 本地验证记录（2026-09-08）：完整套件 **15 项测试通过**（54.6 秒）。
包含原有 9 项 Pretrain 测试和 6 项 SFT 测试；SFT 隔离测试还在没有父项目的临时目录内重跑
5 项功能测试。验证了非零 dropout 下跨 epoch 暂停恢复的 LLM/Projector 参数完全一致、
冻结中间层不变、混合样本梯度累积、索引扫描恢复、spawn 加载、旧代码 Pretrain 产物初始化，
以及移走原 Base 后的 SFT 权重加载和命令行纯文本推理。真实 AutoDL GPU SFT 仍需执行上面的预检和短跑。

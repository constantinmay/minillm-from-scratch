# 从零实现 MiniLLM

[English](README.md) | [简体中文](README_zh.md)

> **第一次学习语言模型？** 请从[中文初学者教程](docs/tutorial/README.md)开始。七章 notebook 将说明、公式推导、源码链接和可运行验证交错排列；教学实验只需要 CPU。

这是一个面向初学者和有限算力的语言模型实践项目：不只阅读公式或调用现成 API，而是从零实现并亲手跑通 tokenizer、decoder-only Transformer、Base 预训练、指令 SFT、偏好对齐、文本生成与评测。参考模型只有 17.23M 参数，使用单张 8GB RTX 4060 Laptop GPU 和原生 PyTorch 训练，不依赖 Hugging Face Transformers。

## 为什么做这个项目？

今天的大语言模型通常依赖普通学习者接触不到的大规模 GPU 集群，而消费级显卡往往连十亿参数模型都难以从头训练。这个项目想回答一个更朴素的问题：**只有一张普通显卡的初学者，能不能仍然亲手经历语言模型训练的每个关键阶段，看见模型发生变化，并知道怎样判断训练是否有效？**

目标不是训练通用助手，也不是追求 SOTA，而是把完整 pipeline 缩小到一个人可以读懂、运行、修改和排查的尺度。

```text
TinyStories → BPE tokenizer → Base 预训练 → Instruction SFT
            → DPO / RSFT → 生成、指标评测与错误分析
```

这次 RTX 4060 本地训练中，最令人惊喜的节点是：大约半小时后，Base 已开始生成具有可辨认英语语法的短文本。这是本次运行的近似观察，不是对所有硬件的速度承诺；项目保留的 loss 记录和最终生成样例才是可复查证据。

## 训练过程中会看到什么？

1. **Base 预训练：** 输出从随机 token 逐渐变成 TinyStories 风格英文，模型学会 next-token prediction。
2. **Instruction SFT：** 模型学习续写、关键词故事、精确句数和抽取式 QA 四种有限任务格式。
3. **DPO 与 RSFT：** 观察偏好数据怎样改变行为，也会看到代理偏好分数提高但实际任务指标下降的失败案例。
4. **评测：** 分开检查语言建模、指令完成、偏好排序与生成质量，不用一个总分掩盖问题。

## 模型

| 组件 | 设置 |
|---|---|
| 参数量 | 17,232,768 |
| 上下文 | 256 tokens |
| 词表 | 8,000-token BPE |
| Transformer | 6 层、6 heads、隐藏维度 384 |
| 位置编码 | RoPE |
| 归一化 | Pre-RMSNorm |
| MLP | SwiGLU，宽度 1,536 |
| 注意力 | PyTorch causal SDPA |
| 其他 | 无 bias 线性层、embedding/LM head 权重绑定 |

## 仓库结构

```text
configs/       当前模型与训练配置
data/          处理后的 TinyStories 与指令/对齐数据
docs/          双语教程、理论速查与实验报告
eval/          统一评测器和指标
model/         decoder-only Transformer 实现
scripts/       数据准备与导出脚本
tests/         单元测试和回归测试
tokenizer/     BPE 训练与运行封装
train/         预训练、SFT 和 DPO 训练器
papers/        论文目录；本地 PDF 不进入 Git
```

checkpoint、日志、原始大数据、论文 PDF 和生成评测产物默认不进入 Git。

## 环境

```bash
conda create -n dl_1 python=3.10
conda activate dl_1
pip install -r requirements.txt
```

训练环境为 8GB RTX 4060 Laptop GPU。测试、教程和小规模推理支持 CPU。

## 复现训练流水线

### 1. Tokenizer 与 Base

将 TinyStories 文本放入 `data/raw/`：

```bash
python tokenizer/train_tokenizer.py
python scripts/prepare_pretrain_data.py
python train/pretrain.py --config configs/train_pretrain.yaml
```

当前实验从 `checkpoints/base.pt` 初始化。

### 2. Instruction SFT 与纯续写对照

```bash
python scripts/build_instruction_sft.py
python train/sft.py --config configs/train_sft.yaml

python scripts/build_instruction_sft.py \
  --output-dir data/instruction_sft_continuation \
  --task-mix continuation_only
python train/sft.py --config configs/train_sft_continuation.yaml
```

两组数据均为 20,000/1,000/1,000 个训练/验证/测试样本，训练预算相同，来源故事不跨 split。主要 SFT loss 只监督 response，并加入小权重全序列 LM loss 缓解遗忘。

### 3. 共享候选池、DPO-v1 与 RSFT

```bash
python scripts/build_instruction_alignment.py
python train/dpo.py --config configs/train_instruction_dpo.yaml
python train/sft.py --config configs/train_instruction_rsft.yaml
```

DPO 与 RSFT 来自同一个候选池；候选构建支持 `--resume`。

### 4. 严格 DPO-v2

```bash
python scripts/build_strict_dpo.py
python train/dpo.py --config configs/train_instruction_dpo_v2.yaml
```

DPO-v2 排除缺乏可靠语义判据的自由续写，只保留 QA、句数和关键词任务中 chosen 硬通过、rejected 硬失败的平衡偏好对。最终选择 step 200，而不是机械使用最后一步。

## 怎样评测模型？

不存在一个能够说明全部能力的“LLM 总分”。本项目让每类指标只回答自己的问题：

| 想判断什么 | 指标 | 怎样理解 |
|---|---|---|
| Base 是否学会 next-token prediction？ | validation NLL / PPL | 越低越好 |
| 模型是否拟合未见过的任务回答？ | response NLL | 越低越好 |
| 是否遵守可自动验证的指令？ | 句数精确率、QA EM、关键词覆盖率/全通过率 | 越高越好 |
| 是否把 chosen 排在 rejected 前面？ | preference accuracy、margin、DPO loss | 只反映该偏好集，不能替代外部任务指标 |
| 生成文本在形式上是否健康？ | 重复率、distinct-n、句末率、长度 | 需分别查看，不能证明语义质量 |
| 人实际更喜欢哪个输出？ | 固定 prompts、原始样例、随机盲评对 | 属于定性证据，最好独立盲评 |

```bash
python eval/comprehensive_eval.py \
  --model Base=checkpoints/base.pt \
  --model InstructionSFT=checkpoints/instruction_sft/sft.pt \
  --model DPOv2=checkpoints/instruction_dpo_v2/dpo_step_200.pt \
  --model RSFT=checkpoints/instruction_rsft/rsft.pt \
  --seeds 42,123 --temperature 0.01 --top-k 1 \
  --output results/final_evaluation
```

评测分别报告 TinyStories PPL、response NLL、偏好排序、精确句数、QA EM、关键词覆盖/全通过、重复率、句末率、多样性、吞吐与盲评样本，不构造含义不清的单一总分。

### 核心结果

| 模型 | LM PPL ↓ | 句数精确率 ↑ | QA EM ↑ | 关键词全通过 ↑ |
|---|---:|---:|---:|---:|
| Base | 5.36 | 0.370 | 0.000 | 0.000 |
| ContinuationSFT | 5.70 | 0.345 | 0.000 | 0.024 |
| InstructionSFT | 5.80 | 0.615 | **0.850** | **0.048** |
| DPOv1 | 6.34 | 0.590 | 0.800 | 0.044 |
| DPOv2 (step 200) | 5.82 | **0.715** | 0.845 | 0.044 |
| RSFT | 5.88 | 0.605 | 0.845 | **0.048** |

主要实践观察是：缺少广泛世界知识的小模型，只要任务和答案经过仔细构造，仍能学会**有限、结构化的指令协议**。这是实验结果，不代表通用指令遵循。DPO-v2 在 PPL 几乎不变的情况下改善句数控制；DPO-v1 则说明偏好排序提高可能与外部任务退化同时出现。

完整分析见[实验报告](docs/experiment_report.md)。

## 对比演示

```bash
python demo_compare.py --task qa \
  --input "Tim did not listen to his mom. Tim played all day." \
  --question "Who did not listen?"
```

默认比较 InstructionSFT、DPO-v2 和 RSFT，也支持 `continuation`、`keywords` 和 `sentence_count`。

## 测试

```bash
pytest tests -q
```

测试覆盖因果 mask、张量形状、移位标签、prompt mask、生成、任务规则、无泄漏数据构建、严格 DPO 导出、统一评测器以及中英文教程 notebook。

## 文档

- [中文初学者教程](docs/tutorial/README.md)
- [English beginner tutorial](docs/tutorial/README_en.md)
- [文档索引](docs/README.md)
- [理论与公式](docs/theory.md)
- [实验报告](docs/experiment_report.md)
- [论文与实现映射](papers/references_and_analysis.md)

## 边界

本仓库是一个面向有限资源的教学与实践项目，展示如何在消费级显卡上真正跑通并观察完整训练流程。它不声称具备通用知识、可靠的开放故事语义判断或 SOTA benchmark 表现；窄域指令学习只是这条实践流水线中的一个受控实验结果。

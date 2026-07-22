# 从零实现 MiniLLM

[English](README.md) | [简体中文](README_zh.md)

> **第一次学习语言模型？** 请从[中文初学者教程](docs/tutorial/README.md)开始。七章 notebook 将说明、公式推导、源码链接和可运行验证交错排列；教学实验只需要 CPU。

这是一个可复现的 17.23M 参数语言模型项目，用单张 RTX 4060 Laptop GPU 研究窄域指令遵循与偏好对齐。Transformer、训练损失、数据构建、生成和评测均使用 PyTorch 实现，不依赖 Hugging Face Transformers。

## 研究问题

缺少通用世界知识的小型 TinyStories base 模型，能否学会有限、可审计的指令空间？项目依次比较：

1. TinyStories base 预训练；
2. 相同预算的纯续写 SFT 对照；
3. 四任务 Instruction SFT；
4. 从 Instruction SFT 初始化的 DPO 与 RSFT；
5. 只使用“硬成功 vs 硬失败”偏好对的保守 DPO-v2。

四类任务是故事续写、关键词故事、精确句数控制和抽取式 QA。这是一个受控研究实验，不是通用聊天机器人。

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

## 正式评测

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

主要正结果来自任务化 Instruction SFT。DPO-v2 在 PPL 几乎不变的情况下改善句数控制；DPO-v1 说明偏好排序提高可能与外部任务退化同时出现。关键词严格成功率仍低，报告中将其保留为未解决问题。

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

本仓库展示的是消费级算力下完整、可审计的研究闭环，不声称具备通用知识、可靠的开放故事语义判断或 SOTA benchmark 表现。最可靠的结论只适用于受控 TinyStories 域中的窄指令遵循。

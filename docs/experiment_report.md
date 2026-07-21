# MiniLLM 指令对齐实验报告

## 1. 摘要

本项目研究一个受算力约束的问题：17.23M 参数的 TinyStories base 模型能否在 RTX 4060 8GB 上学会窄而可审计的指令协议，以及偏好优化能否在不明显破坏语言建模的前提下进一步改善任务表现。

结果显示，Instruction SFT 将 QA exact match 从 0 提高到 0.85，并将句数精确率从 0.37 提高到 0.615；相同数据量和训练预算的 Continuation-SFT 在两项指标上分别只有 0 和 0.345，说明收益主要来自任务化监督，而非简单增加故事训练。严格 DPO-v2 在基本保持 PPL 和 QA 的同时把句数精确率提高到 0.715。DPO-v1 的偏好准确率达到 0.917，但 PPL、response NLL 和主要任务指标回退，构成一次明确的奖励过度优化案例。

关键词任务仍未解决：所有模型的 all-keyword success 都不超过 0.048。该负结果说明 17M 模型容量、当前训练量或任务构造仍不足，不能用平均覆盖率代替严格成功率。

## 2. 实验设置

### 模型与硬件

| 项目 | 设置 |
|---|---|
| 参数量 | 17,232,768 |
| 结构 | 6-layer decoder-only Transformer |
| 隐藏维度 / heads | 384 / 6 |
| 上下文 / 词表 | 256 / 8,000 BPE |
| 组件 | RoPE、Pre-RMSNorm、SwiGLU、causal SDPA、weight tying |
| 训练设备 | NVIDIA RTX 4060 Laptop GPU，8GB |
| 框架 | PyTorch，FP16 mixed precision |

### 数据

- Base：TinyStories next-token prediction。
- Instruction SFT：20,000 / 1,000 / 1,000 个 train/valid/test 样本，任务比例为续写 35%、关键词故事 25%、句数控制 20%、抽取式 QA 20%。
- Continuation-SFT：20,000 / 1,000 / 1,000 个纯续写样本，作为同预算对照。
- DPO-v1：共享候选池产生的 534 train / 56 valid 偏好对，包含自由续写及软奖励。
- RSFT：从同一候选池按规则选择的 715 train / 73 valid 响应。
- DPO-v2：108 train / 12 valid 严格偏好对；QA、句数和关键词任务各 36/4 对。chosen 必须硬通过、rejected 必须硬失败，排除自由续写。

SFT 数据按来源故事分组切分，train/valid/test 来源交集均为 0。DPO-v2 在可用严格配对中按最小任务数量下采样，以避免句数任务支配梯度。

### 对照模型

| 名称 | 初始化与训练 |
|---|---|
| Base | TinyStories 预训练 |
| ContinuationSFT | Base + 纯续写 SFT |
| InstructionSFT | Base + 四任务 SFT |
| DPOv1 | InstructionSFT + 第一版偏好数据 |
| DPOv2 | InstructionSFT + 严格平衡偏好数据，选择 step 200 |
| RSFT | InstructionSFT + reward-selected responses |

DPO-v2 checkpoint 选择在独立 200-prompt sweep 上完成。100/200/300/400 micro-step 中，step 200 的句数精确率最高（0.78），QA 保持 0.82；因此最终评测不使用最后一步。

## 3. 正式评测协议

- 测试 prompt：1,000 个，四任务分布与 Instruction SFT test 一致。
- 生成 seed：42、123，共 2,000 次/模型；相同 prompt 和 seed 对所有模型复用。
- 解码：`temperature=0.01`、`top_k=1`、`max_new_tokens=80`，实际等价于确定性 top-1 选择。
- LM 评测：TinyStories validation 上 100 batches。
- 偏好评测：DPO-v2 的 12 对严格 valid pairs。表中 accuracy 与 margin 是模型自身的 chosen/rejected 排序，不依赖 reference；原始 JSON 的附加 `dpo_loss_vs_reference` 诊断固定使用 InstructionSFT reference。
- 原始产物：`results/final_evaluation/evaluation_results.json`、CSV、逐模型样例和盲评对。

由于贪心解码对 seed 不敏感，两个 seed 不应被误解为两次独立训练。它们保持了评测接口与采样实验的一致性，但不能用于估计训练方差。

## 4. 结果

箭头只表示单个指标的方向，不表示存在可合并的总分。

| Model | LM PPL ↓ | Response NLL ↓ | Pref. acc. ↑ | Pref. margin ↑ | Sentence exact ↑ | QA EM ↑ | Keyword cov. ↑ | Keyword all ↑ | Repeat-3 ↓ | End rate ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 5.36 | 2.1819 | 0.2500 | -0.1607 | 0.3700 | 0.0000 | 0.0540 | 0.0000 | 0.0509 | 0.1670 |
| ContinuationSFT | 5.70 | 1.9106 | 0.5833 | 0.0027 | 0.3450 | 0.0000 | 0.1240 | 0.0240 | 0.0421 | 0.8050 |
| InstructionSFT | 5.80 | 1.8112 | 0.5833 | 0.3756 | 0.6150 | 0.8500 | 0.2200 | 0.0480 | 0.0281 | 0.8810 |
| DPOv1 | 6.34 | 1.9557 | **0.9167** | **1.1436** | 0.5900 | 0.8000 | **0.2560** | 0.0440 | **0.0191** | **0.9990** |
| DPOv2 | 5.82 | 1.8147 | 0.5833 | 0.4275 | **0.7150** | 0.8450 | 0.2320 | 0.0440 | 0.0343 | 0.8920 |
| RSFT | 5.88 | 1.8295 | 0.7500 | 0.5226 | 0.6050 | 0.8450 | 0.2080 | **0.0480** | 0.0246 | 0.9070 |

吞吐量为 193–216 token/s，模型间差异较小，可能包含运行时噪声；这些模型结构相同，因此不将吞吐差异解释为算法收益。

## 5. 分析

### 5.1 SFT 学到的是任务协议

Continuation-SFT 让句末率从 0.167 提高到 0.805，却没有获得 QA 能力，句数精确率还从 0.370 降到 0.345。InstructionSFT 在相同样本量下达到 QA 0.85、句数 0.615。这一对照支持“窄任务分类与协议学习有效”，而不是“额外训练自然会产生指令能力”。

### 5.2 DPO-v1 展示 reward overoptimization

DPO-v1 的偏好准确率比 SFT 高 0.3334，margin 高 0.7680；与此同时 PPL 从 5.80 恶化到 6.34，response NLL 从 1.8112 恶化到 1.9557，QA 与句数均下降。偏好代理变好而外部任务变差，说明仅依据偏好 loss 或 preference accuracy 选择 checkpoint 会得到错误结论。

### 5.3 DPO-v2 是保守但局部的改进

DPO-v2 相对 SFT：PPL 只增加 0.02，response NLL 增加 0.0035，QA 仅下降 0.005；句数精确率提高 0.10。它没有提高严格偏好准确率，也没有改善 all-keyword success。因此合理结论是“严格配对对句数控制有效且副作用较小”，而不是“DPO-v2 全面优于 SFT”。

### 5.4 RSFT 稳定但没有主要任务增益

RSFT 保持 QA 0.845 和较低重复率，但句数和关键词指标均未超过 SFT。它可以作为低成本稳定基线，当前数据下没有证据证明它优于直接 SFT。

## 6. 有效性威胁与限制

- 只有一个训练 seed，无法报告训练方差或置信区间。
- 严格 DPO valid 只有 12 对；偏好准确率每个样本占 8.33 个百分点，统计分辨率很低。
- 所有任务来自 TinyStories 域，不能外推到通用知识、数学、代码或开放对话。
- QA 与硬约束可自动评测；自由续写只使用表面指标，尚未完成独立人工盲评。
- 关键词 all-success 极低，表明这个子任务尚未学会。
- 两个生成 seed 配合贪心解码并不提供独立随机重复。
- checkpoint sweep 使用一组固定 prompt；反复试验仍可能对该集合产生选择偏差。

## 7. 结论与下一步

在单张 4060 上，17M base 足以支撑有限的故事域指令协议，前提是任务可构造、可验证且不依赖广泛世界知识。Instruction SFT 是最重要的收益来源；严格 DPO-v2 提供了一个局部改进和一个比 DPO-v1 更稳健的训练案例。

如果继续投入算力，优先顺序应是：多个训练 seed；扩大且重新设计关键词监督；人工盲评自由续写；再做 45M 容量对照。直接扩大到通用 benchmark 或声称通用助手能力并不受当前证据支持。

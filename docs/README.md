# 文档索引

本目录属于 `minillm-from-scratch` 项目。项目的 Git 根目录是本文档的上一级目录。

## 项目文档

- [`../README.md`](../README.md)：环境、数据构建、训练、评测和复现实验命令。
- [`tutorial/README.md`](tutorial/README.md)：面向初学者的完整学习路线。
- [`tutorial/01_tokenizer_and_lm.md`](tutorial/01_tokenizer_and_lm.md)：BPE、next-token prediction、交叉熵和困惑度。
- [`tutorial/02_transformer.md`](tutorial/02_transformer.md)：注意力、因果 mask、RoPE、RMSNorm、SwiGLU 与参数量。
- [`tutorial/03_pretraining.md`](tutorial/03_pretraining.md)：数据管线、AdamW、学习率、混合精度与训练代码。
- [`tutorial/04_sft.md`](tutorial/04_sft.md)：窄任务设计、response mask、SFT 对照实验。
- [`tutorial/05_alignment.md`](tutorial/05_alignment.md)：DPO 推导、严格偏好数据、RSFT 与过度优化。
- [`tutorial/06_evaluation.md`](tutorial/06_evaluation.md)：指标公式、实验设计、统计边界和失败模式。
- [`tutorial/07_reproduce.md`](tutorial/07_reproduce.md)：从环境到最终演示的完整复现手册。
- [`tutorial/notebooks/`](tutorial/notebooks/)：四个可运行的概念验证 notebook。
- [`theory.md`](theory.md)：核心理论速查；系统学习建议从教程开始。
- [`experiment_report.md`](experiment_report.md)：最终实验设置、实测结果、结论和限制。

## 论文资料

- [`../papers/references_and_analysis.md`](../papers/references_and_analysis.md)：论文目录、设计决策映射和 BibTeX。
- `../papers/*.pdf`：本地论文原文，仅作离线参考；该目录已被 Git 忽略，不随代码仓库提交。

## 实验产物

- `../results/`：统一评测报告、JSON/CSV 指标和生成样本。
- `../logs/`：训练及评测日志。
- `../checkpoints/`：模型权重。

以上实验产物和论文 PDF 默认不进入 Git；代码、配置和可复现命令由 Git 管理。

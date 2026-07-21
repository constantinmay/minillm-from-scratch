# 文档索引

本目录属于 `minillm-from-scratch` 项目。项目的 Git 根目录是本文档的上一级目录。

## 项目文档

- [`../README.md`](../README.md)：环境、数据构建、训练、评测和复现实验命令。
- [`theory.md`](theory.md)：预训练、SFT、DPO、RSFT、RoPE 与 SwiGLU 的理论说明。
- [`theory.ipynb`](theory.ipynb)：可运行的理论推导与验证笔记。

## 论文资料

- [`../papers/references_and_analysis.md`](../papers/references_and_analysis.md)：论文目录、设计决策映射和 BibTeX。
- `../papers/*.pdf`：本地论文原文，仅作离线参考；该目录已被 Git 忽略，不随代码仓库提交。

## 实验产物

- `../results/`：统一评测报告、JSON/CSV 指标和生成样本。
- `../logs/`：训练及评测日志。
- `../checkpoints/`：模型权重。

以上实验产物和论文 PDF 默认不进入 Git；代码、配置和可复现命令由 Git 管理。

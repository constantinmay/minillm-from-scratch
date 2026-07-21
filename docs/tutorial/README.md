# MiniLLM 从零学习路线

这套教程面向第一次系统学习大语言模型训练的读者。它不是脱离项目的概念摘抄，而是把公式、最小实现和本仓库的真实代码对应起来。

## 建议顺序

1. [01：从文本到 next-token prediction](01_tokenizer_and_lm.md)
2. [02：手写 decoder-only Transformer](02_transformer.md)
3. [03：预训练与优化](03_pretraining.md)
4. [04：指令数据与 SFT](04_sft.md)
5. [05：偏好优化 DPO 与 RSFT](05_alignment.md)
6. [06：如何正确评测小语言模型](06_evaluation.md)
7. [07：复现实验与读结果](07_reproduce.md)

配套 notebook：

- [01_tokenizer_and_loss.ipynb](notebooks/01_tokenizer_and_loss.ipynb)：tokenizer、移位标签、交叉熵和困惑度。
- [02_transformer_forward.ipynb](notebooks/02_transformer_forward.ipynb)：模型结构、参数量和因果性验证。
- [03_sft_and_dpo.ipynb](notebooks/03_sft_and_dpo.ipynb)：response mask、SFT loss 和 DPO loss。
- [04_evaluation_metrics.ipynb](notebooks/04_evaluation_metrics.ipynb)：生成指标与任务指标。

## 使用环境

在项目根目录运行：

```bash
conda activate dl_1
pip install -r requirements.txt -r requirements-docs.txt
jupyter lab
```

notebook 首个代码单元会把项目根目录加入 `sys.path`，因此应从项目根目录启动 Jupyter。

## 两种运行尺度

- 教学验证：只做一次前向传播或几个优化步骤，CPU 也能运行。
- 正式实验：使用 `configs/` 中的配置和 `train/` 中的训练入口，需要 GPU；README 中给出完整命令。

不要把 notebook 中的微型随机模型结果当作实验结论。项目结论以 `results/final_evaluation/` 和 [实验报告](../experiment_report.md)为准。

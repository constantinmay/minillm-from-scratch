# MiniLLM 从零学习路线

教程采用七个自包含 notebook。每章把概念说明、公式推导、项目源码对应、正式训练命令和可运行验证放在同一个文件中，不需要在 Markdown 教材与代码附件之间来回切换。

## 学习顺序

1. [01：Tokenizer 与语言模型目标](notebooks/01_tokenizer_and_lm.ipynb)
2. [02：Decoder-only Transformer](notebooks/02_transformer.ipynb)
3. [03：预训练与优化](notebooks/03_pretraining.ipynb)
4. [04：指令数据与 SFT](notebooks/04_sft.ipynb)
5. [05：DPO、严格偏好数据与 RSFT](notebooks/05_alignment.ipynb)
6. [06：评测理论、指标与实验设计](notebooks/06_evaluation.ipynb)
7. [07：完整复现与结果解读](notebooks/07_reproduce.ipynb)

GitHub 可以直接渲染 notebook 中的 Markdown、LaTeX 公式和代码。若要交互运行，请在项目根目录执行：

```bash
conda activate dl_1
pip install -r requirements.txt -r requirements-docs.txt
jupyter lab
```

## Notebook 约定

- Markdown 单元给出推导、解释、正式配置和常见错误。
- 代码单元只放能够从上到下执行的验证，不会自动启动长时间训练。
- 正式训练命令保留在 Markdown 代码块中，避免误触发数小时任务。
- 微型随机模型和教学 loss 只验证原理，不能作为实验结论。
- 项目结论以[正式实验报告](../experiment_report.md)和本地 `results/final_evaluation/` 为准。

所有代码单元都由 `tests/test_tutorial_notebooks.py` 自动执行，文档链接和 notebook 结构由 `tests/test_tutorials.py` 检查。

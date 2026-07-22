# 从零学习 MiniLLM

[简体中文](README.md) | [English](README_en.md)

这是一套以 notebook 为主体的七章教程，目标是在有限资源上亲手走完 MiniLLM 的完整训练 pipeline。大型模型训练通常离初学者很远，但缩小模型与数据后，我们仍然可以观察“随机输出怎样开始具有语法”“Base 与 SFT 分别学到了什么”“偏好指标为什么可能误导”等关键现象。

每个知识点遵循同一节奏：

> 直觉说明 → 公式推导 → 符号解释 → 对应项目代码 → 可运行验证

教程实验不需要 GPU。耗时训练命令只放在 Markdown 代码块中，不会因为“全部运行”而意外启动。

完成教程后，你应该能够达到三个里程碑：

1. 从公式和代码两侧解释 next-token prediction；
2. 独立训练 Base，并用 loss、PPL 和生成样例判断它是否开始学会基本语法；
3. 解释 SFT 为什么能教授有限任务协议，以及为什么 DPO/RSFT 必须同时看外部评测。

第 01–04 章是主线；第 05 章的 DPO/RSFT 属于进阶扩展，不影响先跑通 Base 与 SFT。

## 开始前需要什么基础？

只要能读懂 Python 函数、列表、字典和类，就可以开始。以下知识有帮助，但可以边学边补：

- 线性代数基础：向量、矩阵、点积和张量形状；
- 概率基础：条件概率、对数和平均值；
- PyTorch 基础：tensor、`nn.Module`、`loss.backward()` 和 optimizer；
- 命令行基础：切换目录、运行 `python file.py`；
- Git 对管理实验有帮助，但学习前六章不要求会用。

不需要提前学习强化学习、RLHF、CUDA kernel、分布式训练或 Hugging Face Transformers。

如果 tensor 和自动求导完全陌生，可以先看 PyTorch 官方的 [Tensor 入门](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)和[自动求导入门](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)。

## 十分钟环境检查

在项目根目录运行：

```bash
conda activate dl_1
pip install -r requirements.txt -r requirements-docs.txt
python -m pytest tests/test_model_shapes.py tests/test_tokenizer.py -q
jupyter lab
```

打开第一章，从上到下运行。教学单元在 CPU 上应当几秒内结束。如果某个单元直接启动长时间训练，那属于教程错误；只有明确准备训练时才复制 Markdown 中的正式命令。

## 七章路线

| 章节 | 要回答的问题 | 建议时间 | 中文 | English |
|---|---|---:|---|---|
| 01 | 文本、token、移位标签、交叉熵和 PPL 如何连接？ | 30–45 分钟 | [打开](notebooks/01_tokenizer_and_lm.ipynb) | [Open](notebooks_en/01_tokenizer_and_lm.ipynb) |
| 02 | Decoder-only Transformer 内部发生了什么？ | 60–90 分钟 | [打开](notebooks/02_transformer.ipynb) | [Open](notebooks_en/02_transformer.ipynb) |
| 03 | 稳定的预训练循环如何工作？ | 45–60 分钟 | [打开](notebooks/03_pretraining.ipynb) | [Open](notebooks_en/03_pretraining.ipynb) |
| 04 | Response-only SFT 如何教会小模型有限任务协议？ | 45–60 分钟 | [打开](notebooks/04_sft.ipynb) | [Open](notebooks_en/04_sft.ipynb) |
| 05 | DPO 如何推导，偏好数据和 RSFT 如何构建？ | 60–90 分钟 | [打开](notebooks/05_alignment.ipynb) | [Open](notebooks_en/05_alignment.ipynb) |
| 06 | 不同评测指标分别回答什么实践问题？ | 45–60 分钟 | [打开](notebooks/06_evaluation.ipynb) | [Open](notebooks_en/06_evaluation.ipynb) |
| 07 | 如何复现、解释并扩展整个训练流程？ | 30–45 分钟 | [打开](notebooks/07_reproduce.ipynb) | [Open](notebooks_en/07_reproduce.ipynb) |

推荐顺序是 `01 → 02 → 03 → 04 → 05 → 06 → 07`。已经熟悉 Transformer 的读者可以从第 04 章开始，但仍建议先运行第 01 章的标签移位验证。

## 两条学习路径

### 原理路径：不需要 checkpoint

阅读 01–06 章并运行小实验。无需下载 TinyStories 原始数据，也无需训练模型，就能理解数学目标与项目代码。

### 复现路径：建议使用 GPU

完成 01–06 章后，用第 07 章配合以下资料：

- [项目中文 README](../../README_zh.md)：完整训练和评测命令；
- [训练与评测报告](../experiment_report.md)：协议、结果和限制；
- [论文与代码映射](../../papers/references_and_analysis.md)：原始论文和实现对应关系。

## 每章应该怎么学？

1. 运行代码前先猜 tensor shape。
2. 读完公式后，用自然语言说出每个符号的含义。
3. 立即运行相邻代码，比较输出和自己的预测。
4. 通过章末链接阅读对应源码和测试。
5. 每次只改一个数，先预测变化，再重新运行。

不要把随机微型模型的输出当作研究结论。正式结论来自[实验报告](../experiment_report.md)和本地 `results/final_evaluation/`。

## 初学者常见卡点

- **`ModuleNotFoundError`：** 必须从项目根目录启动 Jupyter，而不是从 `docs/tutorial/` 启动。
- **CUDA 显存不足：** 教程验证使用 CPU；正式训练才需要配置好的 GPU 环境。
- **loss 数字和文档不同：** 随机教学例子不是训练 checkpoint，重点是趋势和等价关系。
- **公式读起来太快：** 先写出每个张量的 shape；大多数困惑来自混淆 batch、时间、head 和词表维度。
- **自定义问题回答错误：** 这是 17M TinyStories 受控模型，不是通用助手。

## 教程维护保证

中英文 notebook 使用完全相同的代码单元。测试会执行所有代码，并检查双语对应、内部链接、公式覆盖以及“解释必须紧邻代码”的结构。

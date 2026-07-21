# 04：指令数据与 SFT

## 1. 为什么 base 模型不能自动理解所有指令

预训练学习的是文本分布，不保证学会仓库自定义的 prompt 协议。对 17M 小模型而言，也不能期待用少量 SFT 注入广泛世界知识。本项目采用窄而可验证的任务空间：续写、抽取式 QA、关键词故事和句数控制。

这使 SFT 的目标从“掌握世界上一切任务”变为“识别有限任务协议，并调用 base 已有的故事语言能力”。

## 2. 数据格式

统一格式是：

```text
Instruction: {instruction}
Input: {input}
Response:
```

response 单独保存在 JSONL 的 `response` 字段中，不拼进 `prompt`。项目不新增模板专用 token，而是只使用 tokenizer 已知的普通文本。构建脚本负责模板化和划分：

```bash
python scripts/build_instruction_sft.py
```

数据划分必须按来源故事隔离，否则同一故事的改写进入训练集和测试集会造成泄漏。

## 3. response-only loss

prompt 是条件，不应要求模型复述 prompt。设 response token 位置集合为 $R$：

$$
\mathcal L_{SFT}=-\frac1{|R|}\sum_{t\in R}
\log p_\theta(y_t\mid x,y_{<t}).
$$

实现方法是把 prompt 对应的 target 改为 `-100`：

```python
labels = full_ids[1:]
labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
```

这里容易出现 off-by-one：模型在 input 位置 $t$ 预测 `full_ids[t+1]`，所以 mask 的对象必须是移位后的 labels。

## 4. 保留预训练能力

项目可加入小权重的全 token LM loss：

$$
\mathcal L=\mathcal L_{SFT}+\alpha\mathcal L_{LM},
\qquad \alpha=0.05.
$$

这只能缓解遗忘，不保证消除遗忘，所以仍需比较 TinyStories PPL 和开放续写指标。

## 5. 正式训练

```bash
python train/sft.py --config configs/train_sft.yaml
```

公平对照使用相同规模的纯续写 SFT：

```bash
python train/sft.py --config configs/train_sft_continuation.yaml
```

这个对照回答了一个重要问题：收益来自“又看了一遍故事”，还是来自“任务化指令数据”。

## 6. SFT 应如何评测

- QA：正式生成评测使用答案规范化后的 exact match；token F1 可作为奖励构建时的连续信号。
- 句数：exact sentence count。
- 关键词：coverage 和 all-keywords pass。
- 续写：PPL、重复率、句末率，但不要把表面指标等同于语义质量。

运行 [notebook 03](notebooks/03_sft_and_dpo.ipynb)查看 response mask。

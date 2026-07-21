# 01：从文本到 next-token prediction

## 1. 语言模型在学习什么

把文本编码为 token 序列 $x_{1:T}$ 后，自回归语言模型分解联合概率：

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

取负对数得到训练目标：

$$
\mathcal L_{\mathrm{LM}}
=-\frac{1}{T-1}\sum_{t=1}^{T-1}
\log p_\theta(x_{t+1}\mid x_{\le t}).
$$

因此输入和标签必须错开一位。若 token 是 `[BOS, A, B, EOS]`：

```text
input  = [BOS, A, B]
target = [A,   B, EOS]
```

项目的预训练数据集和 SFT/DPO 数据集都遵守这一对齐方式。

## 2. BPE tokenizer

字符词表序列很长，整词词表又无法处理新词。Byte Pair Encoding 从较小符号集合开始，反复合并频率最高的相邻符号。一次合并可写成：

$$
(a,b)^*=\arg\max_{(a,b)}\operatorname{count}(a,b).
$$

本项目词表大小为 8000，并保留 BOS、EOS、PAD 等特殊 token。训练入口：

```bash
python tokenizer/train_tokenizer.py
```

调用方式：

```python
from tokenizer.tokenizer_utils import MiniLLMTokenizer

tok = MiniLLMTokenizer("tokenizer/minillm_tokenizer.json")
ids = tok.encode("Once upon a time", add_special_tokens=True)
text = tok.decode(ids, skip_special_tokens=True)
```

## 3. 交叉熵为什么等于负对数似然

模型输出 logits $z\in\mathbb R^V$。softmax 给出：

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

真实 token 的 one-hot 分布为 $q$，交叉熵为：

$$
H(q,p)=-\sum_iq_i\log p_i=-\log p_y.
$$

批量代码不需要显式构造 one-hot：

```python
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1),
    ignore_index=-100,
)
```

`-100` 用于 SFT/DPO 中不参与监督的位置。

## 4. 困惑度

若平均 token NLL 为 $L$，困惑度是：

$$
\operatorname{PPL}=e^L.
$$

PPL 可理解为模型在每一步面对的“等效候选数量”，越低通常越好，但它只衡量给定文本的似然，不能单独证明指令遵循或事实正确。

## 5. 继续学习

运行 [notebook 01](notebooks/01_tokenizer_and_loss.ipynb)，亲自检查编码、移位和 loss。正式数据准备见 `scripts/prepare_pretrain_data.py`。

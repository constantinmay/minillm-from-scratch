# 02：手写 decoder-only Transformer

## 1. 整体数据流

MiniLLM 是 decoder-only Transformer：

```text
token ids -> embedding -> 6 x TransformerBlock -> RMSNorm -> LM head
                          | attention + RoPE
                          | SwiGLU MLP
```

当前配置为 $V=8000$、$d=384$、6 层、6 个注意力头、上下文长度 256，共 17,232,768 个参数。实现位于 `model/`。

## 2. 因果自注意力

线性投影得到：

$$Q=XW_Q,\quad K=XW_K,\quad V=XW_V.$$

缩放点积注意力：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V,
$$

其中因果 mask 满足：

$$
M_{ij}=\begin{cases}0,&j\le i\\-\infty,&j>i.\end{cases}
$$

于是位置 $i$ 无法读取未来 token。项目使用 PyTorch SDPA：

```python
attn_output = F.scaled_dot_product_attention(
    q, k, v, is_causal=True,
    dropout_p=dropout if self.training else 0.0,
)
```

## 3. 多头注意力

把 $d$ 维通道分成 $h$ 组，每组 $d_h=d/h$：

$$
\operatorname{MHA}(X)=\operatorname{Concat}(head_1,\ldots,head_h)W_O.
$$

不同头可以学习不同关系。本项目 $d_h=384/6=64$。

## 4. RoPE 推导

对二维向量使用位置相关旋转：

$$
R_m=\begin{bmatrix}
\cos(m\omega)&-\sin(m\omega)\\
\sin(m\omega)&\cos(m\omega)
\end{bmatrix}.
$$

查询和键分别旋转后：

$$
(R_mq)^\top(R_nk)=q^\top R_{n-m}k,
$$

点积只依赖相对位置 $n-m$。不同通道对使用不同频率：

$$
\omega_i=\theta^{-2i/d_h},\qquad \theta=10000.
$$

项目在初始化时预计算 sin/cos buffer，然后只对 Q、K 应用旋转。

## 5. RMSNorm 与 Pre-Norm

RMSNorm 不减均值：

$$
\operatorname{RMSNorm}(x)=g\odot
\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}.
$$

每个 block 使用 Pre-Norm 残差：

$$
x'=x+\operatorname{Attn}(\operatorname{Norm}(x)),
$$

$$
x''=x'+\operatorname{MLP}(\operatorname{Norm}(x')).
$$

## 6. SwiGLU

$$
\operatorname{SwiGLU}(x)=W_d\left[
\operatorname{SiLU}(W_gx)\odot(W_ux)
\right].
$$

门分支控制上投影分支的信息流。对应代码在 `model/block.py`。

## 7. 参数量近似

忽略 norm 后，每层主要参数约为：

$$
4d^2+3dd_{ff}.
$$

前者来自 Q/K/V/O，后者来自 SwiGLU 的 gate/up/down。总量还要加词嵌入 $Vd$；本项目绑定 embedding 与 LM head，所以只计一份 $Vd$。

运行 [notebook 02](notebooks/02_transformer_forward.ipynb)可验证 shape、参数量和因果 mask。

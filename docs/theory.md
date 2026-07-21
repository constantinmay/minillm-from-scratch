# 理论基础与公式推导

本文档解释项目中使用的核心算法和数学原理。

---

## 1. 预训练：Next-Token Prediction

### 目标

给定一段文本 $x_1, x_2, ..., x_T$，模型学习预测每个位置的下一个 token：

$$P(x_t | x_1, ..., x_{t-1}; \theta)$$

### 损失函数：交叉熵

$$\mathcal{L}_{pretrain} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)$$

具体实现中，`logits` 经过 `log_softmax` 后，对目标 token 取对数概率：

```python
# 等价于 F.cross_entropy(logits, targets)
log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)
loss = -log_probs.gather(dim=-1, index=targets).mean()
```

其中 $V$ 是词表大小（我们的项目是 8000）。

### 直觉

模型每次预测下一个词，猜对了奖励（log prob 高），猜错了惩罚。训练几万步后，模型学会了英语的语法和故事结构。

---

## 2. SFT：监督微调

### 目标

让模型学会在给定 prompt 的条件下生成 response：

$$P(\text{response} | \text{prompt}; \theta)$$

### 损失函数

和预训练相同（交叉熵），但只在 response token 上计算：

$$\mathcal{L}_{SFT} = -\frac{1}{|R|} \sum_{t \in R} \log P(y_t | \text{prompt}, y_{<t}; \theta)$$

其中 $R$ 是 response 的 token 位置集合，prompt 位置的 label 设为 -100（忽略）。

### 预训练信号保护

为防止灾难性遗忘，混合预训练损失：

$$\mathcal{L}_{total} = \mathcal{L}_{SFT} + \alpha \cdot \mathcal{L}_{pretrain}$$

$\alpha$ 通常取 0.1-0.2。

### 为什么小模型会崩？

预训练用尽了 17M 参数的全部容量来编码语言分布。SFT 的梯度虽然只改 response 部分，但反向传播会修改所有层的权重。对于小模型：

$$\text{容量} \approx \text{预训练知识} \quad \Rightarrow \quad \text{SFT梯度} \neq \text{覆盖预训练知识}$$

---

## 3. 强化学习基础

### 什么是强化学习（RL）？

强化学习中，一个**智能体**（agent）在**环境**（environment）中采取**动作**（action），获得**奖励**（reward）。目标是最大化累积奖励。

在 LLM 场景中：
- **智能体**：语言模型
- **动作**：生成的 token
- **奖励**：文本质量分数（来自奖励模型或规则函数）

### 策略梯度

最基本的想法：如果某个动作获得了高奖励，就增加该动作的概率。

$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot R]$$

其中 $\pi_\theta$ 是策略（即 LLM），$R$ 是奖励。

---

## 4. PPO：Proximal Policy Optimization

### 背景

PPO 是 OpenAI 在 2017 年提出的强化学习算法，也是早期 RLHF 系统常用的方法之一。公开资料不足以据此断言某个当前闭源产品的完整训练方案。

### 核心思想

每次更新策略时，**不要走太远**——新旧策略的差异要有上限，防止训练崩溃。

### PPO Loss

$$\mathcal{L}_{PPO} = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是新旧策略的概率比
- $\hat{A}_t$ 是优势函数（advantage），衡量"这个动作比平均水平好多少"
- $\epsilon$ 通常取 0.1-0.2，限制每次更新的幅度

### 为什么 RLHF 用 PPO？

1. **稳定**：clip 机制防止策略突变
2. **样本高效**：比原始策略梯度更高效
3. **不需要二阶梯度**：比 TRPO 简单

### 缺点

- 需要训练 3 个模型：策略模型、奖励模型、reference 模型
- 训练不稳定，超参数敏感
- 对小模型来说太重了

---

## 5. DPO：Direct Preference Optimization

### 核心创新

DPO（2023 年提出）直接用偏好数据训练，**跳过了奖励模型和 PPO**。

### 从 RLHF 到 DPO 的推导

RLHF 的目标是最大化奖励，同时不偏离 reference 模型太远：

$$\max_{\pi_\theta} \mathbb{E}_{\pi_\theta}[r(x,y)] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

这个优化问题的最优解是：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

从中可以反推出隐式奖励：

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

代入 Bradley-Terry 偏好模型，$Z(x)$ 被消掉，得到 DPO loss：

### DPO Loss

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma\left( \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right) \right]$$

其中：
- $y_w$（win）= chosen（被偏好的回答）
- $y_l$（lose）= rejected（被拒绝的回答）
- $\beta$ 控制偏好强度
- $\sigma$ 是 sigmoid 函数

### 直觉解读

$$\underbrace{\log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)}_{\text{策略模型的偏好}} - \underbrace{\log \pi_{ref}(y_w|x) - \log \pi_{ref}(y_l|x)}_{\text{reference 的偏好}}$$

- 如果策略模型比 reference **更偏好** chosen → loss 下降
- 如果策略模型比 reference **更偏好** rejected → loss 上升，梯度修正

### 代码实现

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    ref_log_ratio = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_log_ratio - ref_log_ratio)
    return -F.logsigmoid(logits).mean()
```

### 数据质量为什么关键

DPO 学到的信号 = chosen 和 rejected 之间的**质量差异**。如果 rejected 是垃圾，模型学到的只是"别输出垃圾"，这是无用的信号，但梯度仍然在破坏预训练权重。

好的数据：rejected 是模型生成的（合理但有缺陷），chosen 是真实文本。模型学到的是"输出更接近真实文本质量"。

---

## 6. RSFT：Reward-Selected Fine-Tuning

### 思路

RSFT 是 RLHF 的简化版，不需要奖励模型和 PPO：

1. 给模型一个 prompt
2. 模型生成 $k$ 个候选回答
3. 用奖励函数给每个候选打分
4. **选分数最高的那个**
5. 用选出的数据做 SFT

### 和 RLHF/DPO 的区别

| | RLHF | DPO | RSFT |
|---|---|---|---|
| 需要奖励模型 | 是（神经网络） | 否 | 否（用规则函数） |
| 训练方法 | PPO | 直接偏好优化 | 直接 SFT |
| 需要的模型数 | 3 | 2 | 1 |
| 复杂度 | 高 | 中 | 低 |
| 数据要求 | 人类偏好对 | chosen+rejected 对 | 只需 prompt |

### 为什么对小模型最安全

RSFT 的训练数据来自模型自己的候选分布，通常比外部答案更接近当前策略。但筛选规则仍可能带来偏置，因此必须在独立测试集上检查任务收益和语言建模遗忘。

### 当前任务化奖励

- QA：答案 exact match 为硬通过，token F1 提供连续奖励。
- 句数：实际句数等于要求为硬通过，句数距离提供连续奖励。
- 关键词：全部必需词出现为硬通过，覆盖率提供连续奖励。
- 普通续写：只能检查长度、句末和重复率等表面性质。

DPO-v2 排除缺少可靠语义奖励的普通续写，只使用前三类可审计任务，并要求 chosen 硬通过、rejected 硬失败。

### 历史奖励草案（已弃用）

```
Layer 1（硬约束）：必须通过才计算奖励
  - 词数 >= 10
  - 3-gram 重复率 < 30%
  - 连续词重复少

Layer 2（质量信号）：
  - 关键词覆盖率 +0.3
  - 长度合理 +0.2
  - 句子完整性 +0.2
  - 连贯性（无连续重复）+0.3

Layer 3（渐进惩罚）：
  - 重复惩罚 -max(0, ratio-0.4) * 2
  - 模板崩坏 -0.3
  - 过度简化 -0.5
```

---

## 7. RoPE：旋转位置编码

### 问题

Transformer 的注意力机制本身没有位置信息，需要额外编码位置。

### 思路

用旋转矩阵在复数空间编码位置。对于位置 $m$ 的 token，其 query/key 向量乘以旋转矩阵：

$$\begin{pmatrix} q_0' \\ q_1' \end{pmatrix} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}$$

### 为什么好

两个位置 $m$ 和 $n$ 的点积只依赖于相对位置 $m - n$：

$$q_m^T k_n = |q||k| \cos((m-n)\theta + \phi)$$

这意味着模型可以自然地学习相对位置关系。

### 预计算

$\theta_i = 10000^{-2i/d}$，预先计算所有位置的 cos/sin 并注册为 buffer。

---

## 8. SwiGLU 激活函数

### 公式

$$\text{SwiGLU}(x) = (W_{down})^T \left( \text{silu}(x W_{gate}) \odot (x W_{up}) \right)$$

其中 $\text{silu}(x) = x \cdot \sigma(x)$（也叫 Swish 激活）。

### 对比 ReLU

- ReLU：$\max(0, x)$，简单但信息损失大
- SwiGLU：带门控机制，已被 LLaMA 等公开架构采用；GPT-4 的完整结构并未公开

### 为什么需要 3 个线性层

```
gate_proj: x → silu → ┐
up_proj:   x ────────→ ⊙ → down_proj → output
```

比标准 FFN（2 个线性层）多一个，但效果更好。intermediate_size = 4 × n_embd = 1536。

---

## 9. 验证实验

以下是可以用小代码验证的关键概念：

### 验证因果 mask

```python
# 改变未来 token 不应影响过去位置的 logits
x = torch.randn(1, 8, 384)
out1 = model(x)
x_modified = x.clone()
x_modified[0, 5:] = 0  # 修改位置 5 之后
out2 = model(x_modified)
assert torch.allclose(out1[0, :4], out2[0, :4])  # 前 4 个位置不变
```

### 验证 DPO loss

```python
# chosen 的 logprob 比 rejected 大 → loss 应该小
good = torch.tensor([0.5])  # policy chosen logprob - ref chosen logprob = 0.5
bad  = torch.tensor([-0.5])  # policy rejected logprob - ref rejected logprob = -0.5
loss = dpo_loss(good, bad, torch.zeros(1), torch.zeros(1))
print(f"Loss: {loss.item():.4f}")  # 应该是一个较小的值
```

# 05：偏好优化 DPO 与 RSFT

## 1. 从 KL 约束 RL 到 DPO

常见对齐目标是在提高奖励的同时不远离参考策略：

$$
\max_\pi\;\mathbb E_{y\sim\pi}[r(x,y)]
-\beta D_{KL}(\pi(\cdot|x)\Vert\pi_{ref}(\cdot|x)).
$$

对每个 $x$ 求最优策略，可得：

$$
\pi^*(y|x)=\frac1{Z(x)}\pi_{ref}(y|x)
\exp\left(\frac{r(x,y)}{\beta}\right).
$$

移项：

$$
r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}
+\beta\log Z(x).
$$

代入 Bradley–Terry 偏好概率
$p(y_w\succ y_l|x)=\sigma(r_w-r_l)$，与 response 无关的 $Z(x)$ 抵消，得到：

$$
\mathcal L_{DPO}=-\mathbb E\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]\right).
$$

项目使用 response token 的平均 log-prob，减少长度差对分数的直接影响。它是工程选择，复现实验时必须保持一致。

最小实现直接对应上式：

```python
import torch.nn.functional as F

policy_ratio = policy_chosen_logps - policy_rejected_logps
reference_ratio = ref_chosen_logps - ref_rejected_logps
relative_margin = beta * (policy_ratio - reference_ratio)
loss = -F.logsigmoid(relative_margin).mean()
```

reference 模型必须冻结；只有 policy 参与反向传播：

```python
policy_chosen = compute_logprobs(policy, chosen_ids, chosen_labels)
policy_rejected = compute_logprobs(policy, rejected_ids, rejected_labels)
with torch.no_grad():
    ref_chosen = compute_logprobs(reference, chosen_ids, chosen_labels)
    ref_rejected = compute_logprobs(reference, rejected_ids, rejected_labels)
loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
loss.backward()
```

## 2. 偏好数据比 loss 更重要

若 chosen/rejected 只在长度或格式上不同，DPO 会优化捷径，而不是语义质量。本项目 DPO-v2 采用可审计的严格配对：

- 只保留 QA、句数和关键词三类有硬判据的任务；
- chosen 必须通过硬判据；
- rejected 必须失败；
- 三类任务平衡采样；
- 排除缺乏可靠语义判断器的自由续写。

```bash
python scripts/build_strict_dpo.py
python train/dpo.py --config configs/train_instruction_dpo_v2.yaml
```

当前正式模型选择 step 200，而不是机械选择最后一步。这是基于独立 checkpoint sweep，避免偏好指标持续上升而基础语言质量下降。

## 3. RSFT

Reward-Selected Fine-Tuning 的过程是：对每个 prompt 采样多个候选，用规则或奖励模型打分，保留较优候选，再做 response-only SFT：

$$
y^*=\arg\max_{y\in\{y_1,\dots,y_K\}}R(x,y).
$$

它计算成本较低，但会继承候选分布和奖励函数的偏差。规则无法判断语义时，RSFT 也不能凭空获得可靠监督。

教学版筛选逻辑是：

```python
selected = []
for prompt, candidates in candidate_pool:
    scored = [(reward(prompt, text), text) for text in candidates]
    score, best = max(scored, key=lambda pair: pair[0])
    if hard_constraint_passes(prompt, best):
        selected.append({"prompt": prompt, "response": best, "reward": score})
# 然后把 selected 交给与 SFT 相同的 response-only 训练循环
```

必须在筛选前固定规则、记录所有候选和分数，并保留未参与筛选的测试集，否则容易产生只对奖励函数好看的结果。

```bash
python train/sft.py --config configs/train_instruction_rsft.yaml
```

本仓库的 RSFT 数据已经由候选构建阶段生成，当前不包含在线采样循环。

## 4. 过度优化如何出现

若只报告偏好准确率，会产生错误结论：模型可能极端压低 rejected 的概率，使 preference margin 上升，同时 PPL、SFT NLL 或任务泛化变差。因此对齐实验必须同时画出：

$$
\text{preference gain}\quad\text{vs.}\quad
\text{language/task regression}.
$$

DPO-v1 与 DPO-v2 的对照正是本项目的核心消融。

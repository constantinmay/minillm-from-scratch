# 06：如何正确评测小语言模型

## 1. 没有单一“LLM 分数”

评测应与研究问题对应。本项目关心三层能力：

1. 语言建模是否保留；
2. 有明确答案的指令是否完成；
3. 偏好优化是否真的改善任务，而非只拟合奖励。

## 2. 似然指标

### LM NLL / PPL

在独立故事验证集上计算：

$$
\operatorname{NLL}_{LM}=-\frac1N\sum_t\log p(x_t|x_{<t}),
\qquad PPL=e^{\operatorname{NLL}_{LM}}.
$$

它主要检测灾难性遗忘。

### response NLL

只在参考 response token 上计算，用于衡量指令条件下参考答案的似然。它仍不等价于生成任务成功率。

## 3. 任务指标

### QA exact match

规范化大小写、空白和标点后：

$$EM=\frac1N\sum_i\mathbf1[\hat y_i=y_i].$$

它严格、可解释，但同义答案可能被误判。因此大任务通常同时报告 token F1；本项目窄 QA 答案适合 EM。

```python
import re

def normalize_answer(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

exact_match = float(normalize_answer(prediction) == normalize_answer(reference))
```

### 关键词约束

$$Coverage_i=\frac{|K_i\cap tokens(\hat y_i)|}{|K_i|},$$

$$AllKeyword=\frac1N\sum_i\mathbf1[Coverage_i=1].$$

```python
present = [keyword.lower() in prediction.lower() for keyword in required]
coverage = sum(present) / len(present)
all_keywords = float(all(present))
```

### 句数控制

$$SentenceExact=\frac1N\sum_i
\mathbf1[count(\hat y_i)=c_i].$$

## 4. 偏好指标

定义模型自身的 response 排序 margin：

$$
m=\log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x).
$$

偏好准确率是 $\Pr(m>0)$。DPO loss 另行使用 `--preference-reference` 指定的冻结参考模型。valid pair 必须来源隔离；12 对严格验证样本很小，因此只能作为项目内诊断，不能声称普适结论。

## 5. 生成质量代理指标

- Distinct-$n$：唯一 n-gram / 全部 n-gram；
- repetition-3：重复 trigram 比例；
- sentence-end rate：以 `. ! ?` 结束的比例；
- tokens/s：推理吞吐。

这些都是表面代理。Distinct 高不代表连贯，句末率高不代表回答正确。自由文本语义评测通常还需人工盲评、可靠 judge 或外部 benchmark；对 17M TinyStories 模型直接跑通用知识 benchmark 并不匹配其训练域。

## 6. 实验设计基础

- test set 只用于最终报告，checkpoint 选择使用 validation/sweep 集；
- 固定 prompt 集，并报告多个生成 seed；
- 贪心解码适合测确定性指令遵循，采样解码适合测分布与多样性；
- 与 Base、Continuation-SFT、Instruction-SFT 对比；
- 报告数据量、参数量、硬件、解码参数和失败案例；
- 不把不同训练数据、不同 prompt 或不同采样参数下的数字直接比较。

正式入口：

```bash
python eval/comprehensive_eval.py \
  --seeds 42,123 --temperature 0.01 --top-k 1 \
  --output results/final_evaluation
```

运行 [notebook 04](notebooks/04_evaluation_metrics.ipynb)验证每个指标的行为。

# 07：复现实验与读结果

## 1. 项目主问题

在 RTX 4060 8GB 和 17.23M 参数约束下，能否通过窄任务设计让 TinyStories base 学会可测的指令协议？严格 DPO 是否比噪声偏好数据更稳健？

## 2. 完整顺序

```bash
# 1. 安装
pip install -r requirements.txt

# 2. tokenizer 与预训练数据（已有产物时可跳过）
python tokenizer/train_tokenizer.py
python scripts/prepare_pretrain_data.py

# 3. base
python train/pretrain.py --config configs/train_pretrain.yaml

# 4. 指令数据与 SFT
python scripts/build_instruction_sft.py
python train/sft.py --config configs/train_sft.yaml

# 5. 严格 DPO-v2
python scripts/build_strict_dpo.py
python train/dpo.py --config configs/train_instruction_dpo_v2.yaml

# 6. RSFT
python train/sft.py --config configs/train_instruction_rsft.yaml

# 7. 正式评测与演示
python eval/comprehensive_eval.py --seeds 42,123 --temperature 0.01 --top-k 1 --output results/final_evaluation
python demo_compare.py --task qa --input "Tim did not listen to his mom. Tim played all day." --question "Who did not listen?"
```

## 3. 先做快速检查

一次 optimizer update 处理的有效 token 数可用下面的式子核对，避免把 micro-step 和 update 混为一谈：

$$
N_{tokens/update}=B_{micro}\times N_{accum}\times T.
$$

完整训练前建议：

```bash
python -m pytest tests -q
python -m py_compile model/*.py train/*.py eval/*.py scripts/*.py demo_compare.py
```

还应把配置中的 `max_steps` 暂时降到很小，验证 data → forward → backward → checkpoint 全链路。快速检查通过不代表模型已经收敛。

## 4. 如何读对比表

不要寻找“所有列都最高”的模型，而要围绕问题读表：

- Base → Continuation-SFT：额外故事训练本身带来多少收益？
- Continuation-SFT → Instruction-SFT：任务化数据带来多少额外收益？
- SFT → DPO-v1：偏好拟合是否伴随 PPL/NLL 回退？
- SFT → DPO-v2：严格小数据是否获得更保守的变化？
- SFT → RSFT：筛选后再监督微调是否稳定？

## 5. 可复现边界

仓库跟踪代码、配置和小型 JSONL 数据，不跟踪大 checkpoint、日志、原始语料和论文 PDF。GitHub 用户可复现流程，但若不提供 base checkpoint，必须先重新预训练。最终数字见 [实验报告](../experiment_report.md)，原始机器可读结果见 `results/final_evaluation/evaluation_results.json`（本地实验产物，默认被 Git 忽略）。

## 6. 可以继续扩展的实验

- 对 strict DPO 数据量和 $\beta$ 做消融；
- 用 3–5 个训练 seed 给出均值与置信区间；
- 增加人工盲评并报告评审一致性；
- 用更强 teacher 生成候选，但继续用硬约束过滤；
- 在 45M 模型上复现实验，区分容量效应和数据设计效应。

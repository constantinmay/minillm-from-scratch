# MiniLLM-from-Scratch：论文参考文献与分析

本文档整理了 MiniLLM-from-Scratch 项目所涉及的关键论文，按训练阶段分类，分析每篇论文对项目实现决策的影响。

---

## 目录

1. [模型架构基础](#1-模型架构基础)
2. [分词器](#2-分词器)
3. [预训练数据与小型模型能力](#3-预训练数据与小型模型能力)
4. [监督微调（SFT）](#4-监督微调sft)
5. [直接偏好优化（DPO）](#5-直接偏好优化dpo)
6. [RLHF 与基于奖励的训练](#6-rlhf-与基于奖励的训练)
7. [奖励塑形与知识引导（Reward Shaping）](#7-奖励塑形与知识引导reward-shaping)
8. [缩放定律与训练动力学](#8-缩放定律与训练动力学)
9. [小型模型训练实践](#9-小型模型训练实践)
10. [论文与项目阶段交叉索引](#10-论文与项目阶段交叉索引)
11. [关键公式汇总](#11-关键公式汇总)
12. [BibTeX 引用列表](#12-bibtex-引用列表)
13. [可行性评估总结](#13-可行性评估总结)
14. [模型评测方法](#14-模型评测方法)

---

## 1. 模型架构基础

### 1.1 Attention Is All You Need（Transformer 原始论文）

- **论文**：Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). *Attention Is All You Need*.
- **arXiv**：https://arxiv.org/abs/1706.03762
- **本地文件**：`1706.03762_Attention_Is_All_You_Need.pdf`

**核心贡献：**
- 提出了 Transformer 架构，用纯自注意力机制替代 RNN/CNN
- 提出缩放点积注意力：`Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`
- 多头注意力：将多个注意力头的输出拼接，每个头使用不同的投影矩阵
- 使用正弦函数的位置编码（我们改用可学习位置编码）
- 残差连接 + LayerNorm 包裹每个子层（原始为 Post-LN；现代实践更偏好 Pre-LN）

**对本项目的意义：**
- 我们的模型是 LLaMA 风格的 decoder-only Transformer，但不是逐项复刻
- 实现内容：多头因果自注意力 + RoPE、RMSNorm、SwiGLU MLP、带残差连接
- 关键现代化改进：Pre-RMSNorm（替代 Post-LN）、SwiGLU（替代 ReLU/GELU）、RoPE（替代正弦/可学习位置编码）
- 我们的模型配置为 6 层、6 头、384 维，共 17,232,768 参数

### 1.2 GPT-1：通过生成式预训练提升语言理解能力

- **论文**：Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). *Improving Language Understanding by Generative Pre-Training*.
- **链接**：https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

**核心贡献：**
- 确立了 decoder-only Transformer 的"预训练-微调"范式
- 证明了在大规模文本语料上的无监督预训练能产生可迁移的表示
- 在下游 NLP 任务上，仅做极少架构修改即可获得显著提升

**对本项目的意义：**
- 直接启发了我们的两阶段方案：先在 TinyStories 上做因果语言模型预训练，再做指令数据的 SFT
- 我们的预训练目标（下一 token 预测 + 交叉熵损失）遵循 GPT-1 的设计

### 1.3 GPT-2：语言模型是无监督多任务学习器

- **论文**：Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*.
- **链接**：https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

**核心贡献：**
- 将 decoder-only 语言模型扩展到 1.5B 参数
- 证明了无需微调即可实现零样本任务性能
- 引入 WebText 数据集，证明数据多样性很重要

**对本项目的意义：**
- 验证了 decoder-only 架构在大规模下的有效性
- 我们的权重共享（embedding 权重 = LM head 权重）遵循 GPT-2 的设计选择
- GPT-2 的字节级 BPE 分词方案启发了我们的 BPE 分词器设计

---

## 2. 分词器

### 2.1 用子词单元解决神经机器翻译中的罕见词问题（BPE 论文）

- **论文**：Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. ACL 2016.
- **arXiv**：https://arxiv.org/abs/1508.07909
- **本地文件**：`1508.07909_BPE.pdf`

**核心贡献：**
- 将字节对编码（Byte Pair Encoding, BPE）压缩算法适配为子词分词方法
- 通过将罕见词和未见词编码为高频子词 token 的序列，实现开放词汇翻译
- BPE 迭代地合并词表中频率最高的符号对

**对本项目的意义：**
- 我们使用 `tokenizers` 库（HuggingFace）在 TinyStories 上训练 BPE 分词器
- vocab_size=8000 对 TinyStories 的有限词汇量是合理的
- 特殊 token：`<pad>`、`<unk>`、`<bos>`、`<eos>` 遵循标准做法
- BPE 在词表大小和序列长度之间为小模型提供了良好的平衡

**设计决策理由：**
- TinyStories 面向 3-4 岁儿童，使用简单词汇，8000 词足够覆盖
- BPE 比词级分词更好地处理 OOV（词汇表外词）问题
- `tokenizers` 库提供快速、优化的 BPE 实现

---

## 3. 预训练数据与小型模型能力

### 3.1 TinyStories：语言模型多小还能说出连贯的英语？

- **论文**：Eldan, R. & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*
- **arXiv**：https://arxiv.org/abs/2305.07759
- **本地文件**：`2305.07759_TinyStories.pdf`

**核心贡献：**
- 研究了语言模型产生连贯文本所需的最小模型尺寸
- 使用 GPT-3.5/4 生成合成的短篇儿童故事
- 发现小至 **1M-33M 参数** 的模型在简单数据上即可生成语法正确的故事
- 在 TinyStories 上训练的 10M 参数模型可与 GPT-Neo 125M 媲美
- 引入了基于 GPT-4 的评估框架，评估语法、创造性和一致性

**对本项目的意义——这是我们的核心数据论文：**
- 直接支撑了我们的模型规模选择：实际 17.23M 参数处于 TinyStories 论文验证的小模型能力范围内
- TinyStories 数据集是我们的预训练语料（我们同时从中派生 SFT/DPO 数据）
- 证实了降低数据复杂度（简单词汇、短故事、限定主题）能让微型模型达到惊人的效果
- 我们的评估指标（语法、连贯性、关键词覆盖率）与其评估框架一致

**核心洞见：**
> "降低数据复杂度（简单词汇、短故事、限定主题）能让微型模型达到惊人的能力水平。"

这验证了我们的核心前提：在 TinyStories 上训练的 17.23M 模型具备学习连贯英文故事的合理容量。

---

## 4. 监督微调（SFT）

### 4.1 InstructGPT：通过人类反馈训练语言模型遵循指令

- **论文**：Ouyang, L., Wu, J., Jiang, X., et al. (2022). *Training language models to follow instructions with human feedback*. NeurIPS 2022.
- **arXiv**：https://arxiv.org/abs/2203.02155
- **本地文件**：`2203.02155_InstructGPT.pdf`

**核心贡献：**
- 提出三阶段对齐流水线：SFT -> 奖励模型训练 -> PPO 强化学习
- 1.3B 参数的 InstructGPT 被人类标注者认为优于 175B 的 GPT-3
- 证明在高质量示范数据上做 SFT 能大幅改善指令跟随能力
- 表明 RLHF 在 SFT 基础上能进一步提升对齐效果

**对本项目 SFT 阶段的意义：**
- 我们的 SFT 阶段对应 InstructGPT 的步骤 1：在示范数据上做监督微调
- 使用 instruction-response JSONL 格式配合 prompt 模板，遵循 InstructGPT 的方法
- 标签掩码（prompt 部分的 token 标签设为 -100，只在 response 部分计算损失）是标准 SFT 做法
- 关键经验：即使是小规模的 SFT 数据集，也能显著改变模型行为使其趋向指令跟随

**影响我们设计决策的要点：**
- SFT 使用 tokenizer 已知的纯文本字段：`Instruction: ...\nInput: ...\nResponse:`，避免新增特殊模板 token
- 只在 response token 上计算损失，防止模型记忆 prompt 模式
- 使用更低的学习率（1e-4 vs 预训练的 3e-4），保留预训练知识同时适配行为

---

## 5. 直接偏好优化（DPO）

### 5.1 直接偏好优化：你的语言模型（暗地里）就是一个奖励模型

- **论文**：Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). *Direct Preference Optimization: Your Language Model is (Secretly) a Reward Model*. NeurIPS 2023.
- **arXiv**：https://arxiv.org/abs/2305.18290
- **本地文件**：`2305.18290_DPO.pdf`

**核心贡献：**
- 消除了偏好对齐中对独立奖励模型和强化学习循环的需求
- 将奖励函数重新参数化为：`r(x,y) = beta * log(pi_theta(y|x) / pi_ref(y|x))`
- DPO 损失函数：`L = -log sigmoid(beta * (log_pi(y_w|x) - log_pi(y_l|x)) - (log_ref(y_w|x) - log_ref(y_l|x)))`
- 仅需策略模型 + 冻结的参考模型（无需奖励模型、价值函数或在线采样）
- 在奖励-KL 前沿上优于 PPO，且实现更简单

**对本项目的意义——这是我们的核心 DPO 论文：**
- 我们严格按照此论文实现 DPO
- 我们的损失函数：`-log sigmoid(beta * ((logp_policy_chosen - logp_policy_rejected) - (logp_ref_chosen - logp_ref_rejected)))`
- 策略模型和参考模型均从 `sft.pt` 初始化（标准 DPO 做法）
- 使用 response token 的平均 log 概率来降低长度偏置

**论文提供的实现要点：**
1. **beta=0.1**：论文建议的合理默认值，控制与参考模型的 KL 散度
2. **参考模型冻结**：不对 pi_ref 计算梯度，节省显存
3. **无需在线采样**：与 PPO 不同，DPO 在预先收集的偏好对上训练
4. **梯度行为**：损失函数会根据当前对偏好对的错误排序程度自适应地调整权重

**为什么选择 DPO 而非 PPO：**
- PPO 需要 4 个模型（策略模型、参考模型、奖励模型、评论家）——RTX 4060 显存不够
- DPO 只需 2 个模型（策略模型 + 冻结参考模型）——8GB 显存足够
- DPO 更稳定、更容易调参（只有 beta 一个关键超参数）
- 手写 DPO 损失大约只需 20 行 PyTorch 代码

---

## 6. RLHF 与基于奖励的训练

### 6.1 从人类偏好微调语言模型

- **论文**：Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., & Irving, G. (2019). *Fine-Tuning Language Models from Human Preferences*.
- **arXiv**：https://arxiv.org/abs/1909.08593
- **本地文件**：`1909.08593_Finetuning_LM_from_Human_Preferences.pdf`

**核心贡献：**
- 首次将人类偏好的强化学习应用于语言模型微调
- 在故事续写和摘要任务上验证
- 在人类偏好判断上训练奖励模型，然后用 PPO 优化策略
- 证明即使 1.5B 的模型也能通过偏好反馈获得显著提升

**对本项目的意义：**
- 整个 RLHF 流水线的基础性工作
- 他们的故事生成任务（从 prompt 续写故事）与我们的 TinyStories 场景相似
- 验证了基于奖励的优化在小模型规模下也能奏效

### 6.2 Constitutional AI：从 AI 反馈实现无害性

- **论文**：Bai, Y., Kadavath, S., Kundu, S., et al. (2022). *Constitutional AI: Harmlessness from AI Feedback*.
- **arXiv**：https://arxiv.org/abs/2212.08073
- **本地文件**：`2212.08073_Constitutional_AI.pdf`

**核心贡献：**
- 提出 RLAIF（从 AI 反馈进行强化学习）——用 AI 生成的偏好替代人类标注
- 两阶段流程：有监督的自我批评与修正 + RLAIF
- **拒绝采样微调（Rejection Sampling Fine-tuning）**：生成多个候选回答，通过模型评估筛选，在最佳回答上微调

**对本项目的意义——直接启发我们的 RSFT 阶段：**
- 我们的 RSFT 方法（采样 k 个候选 -> 奖励打分 -> 选择最优 -> SFT）遵循 Constitutional AI 的拒绝采样范式
- 使用基于规则的奖励代替学习到的奖励模型，类比于他们的 AI 评估方式
- 这验证了拒绝采样是替代 PPO 进行策略改进的可行方案

**为什么选择 RSFT：**
- RSFT 比 PPO 更稳定——训练过程中不会出现奖励作弊
- 实现和调试更简单（生成 -> 打分 -> 筛选 -> 重训练）
- 明确展示了"采样 -> 奖励打分 -> 选择更好的回答 -> 策略改进"这一核心循环
- 我们将潜在的 reward hacking 作为分析点记录

---

## 7. 奖励塑形与知识引导（Reward Shaping & Knowledge-Guided Rewards）

本章节是我们 RSFT 奖励函数设计的核心理论支撑。用户的想法——"用规则引导模型生成，让强化学习带有监督学习的影子"——在学术界有深厚的理论基础。

### 7.1 势函数奖励塑形理论（经典理论）

- **论文**：Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*. ICML 1999.
- **链接**：https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- **本地文件**：`1999_NgHaradaRussell_Policy_Invariance_Reward_Shaping.pdf`

**核心贡献：**
- 证明了**势函数奖励塑形（Potential-Based Reward Shaping）**不改变最优策略，但能显著加速学习
- 数学形式：如果额外奖励 `F(s,s') = gamma * Phi(s') - Phi(s)`（其中 Phi 是势函数），则最优策略不变
- 这是奖励塑形的**理论基础**——我们可以自由地往奖励函数里加领域知识，不会破坏模型的能力上限

**对本项目的意义：**
- **直接支撑了我们的设计**：把"好故事"的先验知识编码进奖励函数，不会让模型变得更差，只会让它更快学到正确行为
- 本项目的任务规则奖励并不满足 $F(s,s')=\gamma\Phi(s')-\Phi(s)$ 的形式，因此不能用势函数定理保证最优策略不变；该论文只提供设计奖励时的理论对照
- 这就是用户说的"监督学习的影子"的理论基础：规则奖励 = 软监督信号

### 7.2 Text2Reward：用语言模型进行奖励塑形

- **论文**：Xie, T., Zhao, S., Wu, C. H., Liu, Y., Luo, Q., Zhong, V., Yang, Y., & Yu, T. (2024). *Text2Reward: Reward Shaping with Language Models for Reinforcement Learning*. ICLR 2024.
- **arXiv**：https://arxiv.org/abs/2309.11489
- **本地文件**：`2309.11489_Text2Reward_Reward_Shaping_with_LMs.pdf`

**核心贡献：**
- 提出 Text2Reward 框架：用大语言模型自动生成密集的奖励函数
- 无需训练数据即可自动化奖励塑形
- 验证了**语言描述可以转化为有效的奖励信号**

**对本项目的意义：**
- 我们的奖励函数是用代码编写的规则（而非学到的模型），这与 Text2Reward 的思想一致——**领域知识直接编码为奖励**
- 我们的"关键词覆盖"、"句子完整性"、"长度合理性"等规则本质上就是用代码表达的领域知识
- 启示：奖励函数不需要是学到的模型，精心设计的规则奖励同样有效甚至更稳定

### 7.3 受约束的 RLHF：对抗奖励模型过度优化

- **论文**：Moskovitz, T., Singh, A. K., Strouse, D., Sandholm, T., Salakhutdinov, R., Dragan, A. D., & McAleer, S. (2024). *Confronting Reward Model Overoptimization with Constrained RLHF*. ICLR 2024 (Spotlight).
- **arXiv**：https://arxiv.org/abs/2310.04373
- **本地文件**：`2310.04373_Constrained_RLHF_Reward_Overoptimization.pdf`

**核心贡献：**
- 系统分析了 RLHF 中的奖励过度优化（Reward Overoptimization）问题
- 发现 PPO 训练中存在"奖励阈值现象"：模型会在某个点开始作弊（reward hacking）
- 提出受约束的 RLHF 方法：通过约束 KL 散度来防止模型偏离参考模型过远
- **关键发现**：当奖励信号过于容易满足时，模型会找到"捷径"而非真正改善质量

**对本项目的意义——这是设计奖励函数时必须警惕的问题：**
- 我们的 RSFT 使用规则奖励，更容易被模型"钻空子"（比如重复输出高分关键词）
- 这直接启发我们设计**渐进式惩罚**而非简单阈值：
  - 重复惩罚用连续函数 `-1.0 * max(0, repeat_ratio - 0.15)` 而非二值判断
  - 加入**模板崩坏检测**防止模型学会固定套路
  - 多维度奖励让模型难以通过单一维度作弊
- 论文建议的 KL 约束在我们的 SFT 阶段自然实现（从 sft.pt 初始化不会偏离太远）

### 7.4 奖励塑形缓解 RLHF 中的奖励作弊

- **论文**：(2025). *Reward Shaping to Mitigate Reward Hacking in RLHF*.
- **arXiv**：https://arxiv.org/abs/2502.18770
- **本地文件**：`2502.18770_Reward_Shaping_Mitigate_Reward_Hacking_RLHF.pdf`

**核心贡献：**
- 系统研究了奖励塑形方法在 PPO-based RLHF 中的作用
- 揭示了 PPO 训练中的奖励阈值现象（Reward Threshold Phenomenon）
- 提出通过精心设计的奖励塑形来缓解 reward hacking

**对本项目的意义：**
- 我们的规则奖励天然具有"可解释性"——可以清楚知道哪个维度被 hack 了
- 建议 RSFT 中记录每个维度的得分，方便分析是否存在 reward hacking
- 当某个维度的得分异常高而整体质量未提升时，说明该维度被利用了

### 7.5 进度奖励模型：基于大语言模型的强化学习

- **论文**：Zhang, X., Gao, N., Jiang, X., Chen, Y., Pan, Y., Zhang, M., & Deng, Y. (2025). *Progress Reward Model for Reinforcement Learning via Large Language Models*. NeurIPS 2025 (Poster).
- **本地文件**：`2025_NeurIPS_Progress_Reward_Model.pdf`

**核心贡献：**
- 受势函数奖励塑形启发，构建了**进度奖励模型（Progress Reward Model）**
- 将奖励设计为"从当前状态到目标的进度"信号，而非绝对的终点奖励
- 用大语言模型评估生成文本的质量进度

**对本项目的意义：**
- 启发我们设计**进度感知的奖励**：不仅看最终输出质量，还要看生成过程中是否"在变好"
- 在 RSFT 中可以追踪多轮迭代时的奖励变化趋势，评估是否真正在进步
- 论文的"进度"概念对应我们的多层奖励设计：硬约束是"能否参与"的门槛，质量奖励是"有多好"的进度信号

### 奖励塑形论文总结：对项目奖励函数设计的影响

| 论文 | 核心启示 | 对奖励函数的具体影响 |
|------|---------|-------------------|
| Ng et al. 1999 | 严格的势函数塑形不改变最优策略 | 当前规则不满足定理前提，必须用外部指标检查 reward hacking |
| Text2Reward 2024 | 领域知识可直接编码为奖励 | 用代码规则而非学习模型做奖励是可行的 |
| Constrained RLHF 2024 | 奖励过度优化导致 reward hacking | 需要多维度奖励 + 渐进式惩罚 |
| Reward Hacking 2025 | 奖励阈值现象 | 记录每个维度得分，监控异常 |
| Progress Reward 2025 | 进度信号比绝对分数更有指导意义 | 多轮 RSFT 中追踪奖励趋势 |

---

## 8. 缩放定律与训练动力学

### 8.1 神经语言模型的缩放定律

- **论文**：Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). *Scaling Laws for Neural Language Models*.
- **arXiv**：https://arxiv.org/abs/2001.08361
- **本地文件**：`2001.08361_Scaling_Laws.pdf`

**核心贡献：**
- 建立了语言模型性能与模型规模（N）、数据集规模（D）、计算量（C）之间的幂律关系
- 损失 L(N) ~ N^(-0.076)，L(D) ~ D^(-0.095)
- 更大的模型具有更高的样本效率
- 给定计算预算下的近最优模型规模遵循可预测的曲线

**对本项目的意义：**
- 指导我们对 17.23M 模型能实现什么效果建立合理预期
- 解释了为什么后训练技术（SFT、DPO、RSFT）对小模型特别有价值：它们从有限的参数量中提取更多能力
- 默认预算约为 `30k × 16 × 256 ≈ 1.23 亿` token 位置；梯度累积只改变优化器更新频率

**对项目的实际指导：**
- 当前默认配置约处理 1.23 亿 token 位置，约为每参数 7.1 个 token，不能声称达到 Chinchilla 最优
- 除了原始规模之外，对齐技术（SFT/DPO/RSFT）对能力有乘数效应
- 我们的 4 阶段对比（Base -> SFT -> DPO -> RSFT）将展示每个阶段如何从相同模型容量中提取更多性能

---

## 9. 小型模型训练实践

### 9.1 LLaMA：开放高效的基础语言模型

- **论文**：Touvron, H., Lavril, T., Izacard, G., et al. (2023). *LLaMA: Open and Efficient Foundation Language Models*.
- **arXiv**：https://arxiv.org/abs/2302.13971
- **本地文件**：`2302.13971_LLaMA.pdf`

**本项目采用的 LLaMA 风格组件：**
- **Pre-RMSNorm**：在 attention/MLP 之前使用 RMSNorm（非 LayerNorm），计算更高效，去掉均值中心化步骤
- **SwiGLU 激活函数**：`SwiGLU(x) = (xW1 · sigmoid(xW1)) ⊙ (xW2)`，比标准 GELU 效果更好，MLP 中使用 3 个线性层（gate/up/down）
- **旋转位置编码 RoPE**：在注意力计算中将位置信息融入 Q/K，支持相对位置推理，不使用可学习位置编码
- **无偏置**：所有线性层（Q/K/V/O 投影、MLP、lm_head）均不使用 bias（`bias: false`）
- **权重共享**：token embedding 与 lm_head 共享是本项目为小模型节省参数的选择，并非 LLaMA 的逐项复刻

**对本项目的意义：**
- 通过实现相同的核心组件理解现代 decoder-only 架构，但不声称能力或权重可以直接迁移到前沿大模型
- SwiGLU、RoPE、RMSNorm 是 LLaMA/Mistral/Qwen 等主流模型的标配
- 证明了在更多数据上训练的更小模型可以匹敌更大的模型

### 9.2 Phi-1：教科书就是你所需的一切

- **论文**：Gunasekar, S., Zhang, Y., Aneja, J., et al. (2023). *Textbooks Are All You Need*.
- **arXiv**：https://arxiv.org/abs/2306.11644
- **本地文件**：`2306.11644_Phi1_Textbooks_Are_All_You_Need.pdf`

**核心洞见：**
> 对于小模型而言，数据质量和组成远比原始数据量重要得多。

**对本项目的意义：**
- 支撑了我们对 TinyStories（高质量、简单、一致的数据）的选择，而非多样但嘈杂的网络文本
- "教科书质量"原则同样适用：TinyStories 相当于基础英文生成的"教科书"
- Phi-1 说明高质量数据能够提高训练效率，但其规模和代码数据与本项目不同，不能直接按比例外推

### 9.3 TinyLlama：开源小型语言模型

- **论文**：Zhang, P., Zeng, G., Wang, T., & Lu, W. (2024). *TinyLlama: An Open-Source Small Language Model*.
- **arXiv**：https://arxiv.org/abs/2401.02385
- **本地文件**：`2401.02385_TinyLlama.pdf`

**对本项目的意义：**
- 展示了在有限硬件上用社区工具训练 1.1B 模型（1 万亿 token）的方法
- 表明 FlashAttention 和训练优化可以高效地训练小型模型
- 本项目借鉴小模型和高质量数据的思想，实际规模为 17.23M 参数

### 9.4 MiniCPM：揭示小型语言模型的潜力

- **论文**：Hu, S., Tu, Y., Han, X., et al. (2024). *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies*.
- **arXiv**：https://arxiv.org/abs/2404.06395
- **本地文件**：`2404.06395_MiniCPM.pdf`

**核心贡献：**
- 1.2B/2.4B 参数模型匹配 7B-13B 模型的性能
- 提出 Warmup-Stable-Decay（WSD）学习率调度器
- 计算最优的数据-模型比例高于 Chinchilla 最优
- MiniCPM-DPO 变体在小模型上验证了 DPO 的有效性

**对本项目的意义：**
- 验证了 DPO 在小模型上的可行性（MiniCPM-DPO）
- 他们的 WSD 调度器为我们的余弦调度提供了潜在改进方向
- 证实了小模型从超出缩放定律建议的更多训练数据中获益

---

## 10. 论文与项目阶段交叉索引

| 项目阶段 | 主要参考论文 | 辅助参考论文 |
|---|---|---|
| **Stage 0：分词器** | BPE（Sennrich 2016） | — |
| **Stage 1：预训练** | TinyStories（Eldan 2023） | Transformer（Vaswani 2017）、GPT-1/2（Radford 2018/2019）、缩放定律（Kaplan 2020） |
| **Stage 2：SFT** | InstructGPT（Ouyang 2022） | GPT-1（Radford 2018） |
| **Stage 3：DPO** | DPO（Rafailov 2023） | Ziegler 2019、MiniCPM-DPO（Hu 2024） |
| **Stage 4：RSFT** | Constitutional AI（Bai 2022） | Ziegler 2019、InstructGPT（Ouyang 2022） |
| **奖励函数设计** | Ng et al. 1999（势函数理论） | Text2Reward 2024、Constrained RLHF 2024、Reward Hacking 2025、Progress Reward 2025 |
| **架构设计** | Transformer（Vaswani 2017） | LLaMA（Touvron 2023）、GPT-2（Radford 2019） |
| **小模型哲学** | TinyStories（Eldan 2023） | Phi-1（Gunasekar 2023）、TinyLlama（Zhang 2024）、MiniCPM（Hu 2024） |

---

## 11. 关键公式汇总

### 11.1 RMSNorm（替代 LayerNorm）
```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
```
比 LayerNorm 更高效，省去均值中心化步骤，LLaMA/Mistral/Qwen 均采用。

### 11.2 RoPE 旋转位置编码（替代可学习位置编码）
```
将 Q 和 K 在每两个维度上做旋转变换：
q_m = q * cos(m*theta) + rotate_half(q) * sin(m*theta)
k_m = k * cos(m*theta) + rotate_half(k) * sin(m*theta)

其中 m 是位置索引，theta = 10000^(-2i/d)
```
RoPE 在注意力计算中自然编码相对位置关系，支持长度外推。

### 11.3 SwiGLU 激活函数（替代 ReLU/GELU）
```
SwiGLU(x, W_gate, W_up, W_down) = (silu(x @ W_gate) * (x @ W_up)) @ W_down
其中 silu(x) = x * sigmoid(x)
```
MLP 需要 3 个线性层（gate_proj, up_proj, down_proj），中间维度通常为 4 * n_embd。

### 11.4 因果自注意力 + RoPE
```
Q = RMSNorm(x) @ W_q, K = RMSNorm(x) @ W_k, V = RMSNorm(x) @ W_v
Q_rope, K_rope = apply_rope(Q, K, positions)
Attention = softmax(Q_rope @ K_rope^T / sqrt(d_k) + causal_mask) @ V
```

### 11.5 DPO 损失函数（源自 DPO 论文，公式 7）
```
L_DPO = -E[log sigmoid(beta * (log(pi_w/pi_ref_w) - log(pi_l/pi_ref_l)))]

其中：
- pi_w = pi_theta(y_chosen | x)，pi_l = pi_theta(y_rejected | x)
- pi_ref_w = pi_ref(y_chosen | x)，pi_ref_l = pi_ref(y_rejected | x)
- beta = 0.1（KL 约束权重）
- log 概率仅在 response token 上取平均
```

### 11.6 RSFT 奖励函数（知识引导的三层设计，灵感来自 Ng 1999 + Constitutional AI + Constrained RLHF）

**第一层：硬约束（不满足直接淘汰）**
```
hard_pass = task_specific_constraint(response)
# QA: exact answer; sentence task: exact count; keyword task: all words present
# continuation: length/ending/repetition surface checks（仅用于 RSFT，不用于 DPO-v2）
```

**第二层：知识引导的质量奖励（"监督学习的影子"）**
```
reward  = +0.5 * num_required_words_present          # 关键词覆盖
       + +1.5 * (all_required_words_present)           # 全覆盖 bonus
       + +0.3 * required_words_in_first_half           # 结构引导
       + +1.0 * ends_with_sentence_punctuation          # 句子完整性
       + +0.5 * (sentence_count >= 2)                   # 至少 2 个完整句
       + +1.0 * (40 <= word_count <= 120)               # 最佳长度区间
       + +0.5 * (20 <= word_count < 40 or 120 < word_count <= 150)  # 可接受区间
       + +0.3 * has_transition_words                    # 连贯性信号
       + +0.2 * has_proper_ending                       # 结尾感
```

**第三层：负向惩罚（渐进式，防止 reward hacking）**
```
penalty = -1.0 * max(0, repeat_3gram_ratio - 0.15)    # 渐进重复惩罚
       + -0.5 * (consecutive_same_pattern >= 3)         # 模板崩坏
       + -0.5 * (same_word_count >= 5 and not_stopword) # 高频词重复
       + -0.3 * (avg_sentence_length < 4)               # 过度简化
```

### 11.7 预训练损失（标准因果语言模型）
```
L = CrossEntropyLoss(logits.view(-1, V), targets.view(-1))
其中：logits = model(input_ids)，targets = input_ids 右移 1 位
```

---

## 12. BibTeX 引用列表

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}

@article{radford2018improving,
  title={Improving language understanding by generative pre-training},
  author={Radford, Alec and Narasimhan, Karthik and Salimans, Tim and Sutskever, Ilya},
  year={2018}
}

@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI blog},
  volume={1},
  number={8},
  pages={9},
  year={2019}
}

@inproceedings{sennrich2016neural,
  title={Neural machine translation of rare words with subword units},
  author={Sennrich, Rico and Haddow, Barry and Birch, Alexandra},
  booktitle={Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics},
  pages={1715--1725},
  year={2016}
}

@article{eldan2023tinystories,
  title={TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  author={Eldan, Ronen and Li, Yuanzhi},
  journal={arXiv preprint arXiv:2305.07759},
  year={2023}
}

@article{ouyang2022training,
  title={Training language models to follow instructions with human feedback},
  author={Ouyang, Long and Wu, Jeff and Jiang, Xu and others},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={27730--27744},
  year={2022}
}

@article{rafailov2023direct,
  title={Direct preference optimization: Your language model is (secretly) a reward model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Ermon, Stefano and Manning, Christopher D and Finn, Chelsea},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}

@article{ziegler2019fine,
  title={Fine-tuning language models from human preferences},
  author={Ziegler, Daniel M and Stiennon, Nisan and Wu, Jeffrey and Brown, Tom B and Radford, Alec and Amodei, Dario and Christiano, Paul and Irving, Geoffrey},
  journal={arXiv preprint arXiv:1909.08593},
  year={2019}
}

@article{bai2022constitutional,
  title={Constitutional AI: Harmlessness from AI Feedback},
  author={Bai, Yuntao and Kadavath, Saurav and Kundu, Sandipan and others},
  journal={arXiv preprint arXiv:2212.08073},
  year={2022}
}

@article{kaplan2020scaling,
  title={Scaling laws for neural language models},
  author={Kaplan, Jared and McCandlish, Sam and Henighan, Tom and others},
  journal={arXiv preprint arXiv:2001.08361},
  year={2020}
}

@article{touvron2023llama,
  title={LLaMA: Open and efficient foundation language models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and others},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}

@article{gunasekar2023textbooks,
  title={Textbooks are all you need},
  author={Gunasekar, Suriya and Zhang, Yi and Aneja, Jyoti and others},
  journal={arXiv preprint arXiv:2306.11644},
  year={2023}
}

@article{zhang2024tinyllama,
  title={TinyLlama: An Open-Source Small Language Model},
  author={Zhang, Peiyuan and Zeng, Guangtao and Wang, Tianduo and Lu, Wei},
  journal={arXiv preprint arXiv:2401.02385},
  year={2024}
}

@article{hu2024minicpm,
  title={MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies},
  author={Hu, Shengding and Tu, Yuge and Han, Xu and others},
  journal={arXiv preprint arXiv:2404.06395},
  year={2024}
}

@inproceedings{ng1999policy,
  title={Policy invariance under reward transformations: Theory and application to reward shaping},
  author={Ng, Andrew Y and Harada, Daishi and Russell, Stuart},
  booktitle={Proceedings of the 16th International Conference on Machine Learning},
  pages={278--287},
  year={1999}
}

@inproceedings{xie2024text2reward,
  title={Text2Reward: Reward Shaping with Language Models for Reinforcement Learning},
  author={Xie, Tianbao and Zhao, Siheng and Wu, Chen Henry and Liu, Yitao and Luo, Qian and Zhong, Victor and Yang, Yanchao and Yu, Tao},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}

@inproceedings{moskovitz2024constrained,
  title={Confronting Reward Model Overoptimization with Constrained RLHF},
  author={Moskovitz, Ted and Singh, Aaditya K and Strouse, DJ and Sandholm, Tuomas and Salakhutdinov, Ruslan and Dragan, Anca D and McAleer, Stephen},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}

@article{anonymous2025reward,
  title={Reward Shaping to Mitigate Reward Hacking in RLHF},
  journal={arXiv preprint arXiv:2502.18770},
  year={2025}
}

@inproceedings{zhang2025progress,
  title={Progress Reward Model for Reinforcement Learning via Large Language Models},
  author={Zhang, Xiuhui and Gao, Ning and Jiang, Xingyu and Chen, Yihui and Pan, Yuheng and Zhang, Mohan and Deng, Yue},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

---

## 13. 可行性评估总结

| 关切点 | 参考论文 | 结论 |
|---|---|---|
| 17.23M 模型能否生成连贯文本？ | TinyStories | 可以，10M+ 在简单数据上已有实验证据 |
| BPE 是否适合小词汇量英文？ | BPE 论文 | 适合，能很好地处理 OOV 问题 |
| SFT 对微型模型是否有效？ | InstructGPT | 有效，SFT 在 1.3B 规模即可显著改变行为 |
| DPO 是否能在 RTX 4060 上运行？ | DPO 论文 | 可以，只需 2 个模型（非 PPO 的 4 个） |
| RSFT 是否是 PPO 的有效替代？ | Constitutional AI | 是，拒绝采样是成熟的实践方法 |
| 规则奖励是否安全可靠？ | Ng et al. 1999、reward hacking 文献 | 不能先验保证；当前规则不是严格势函数，需要独立任务指标和失败案例验证 |
| 奖励函数会被模型钻空子吗？ | Constrained RLHF 2024 | 有风险，需多维度奖励 + 渐进式惩罚缓解 |
| 当前训练量是否达到缩放最优？ | 缩放定律、Phi-1 | 未达到；本项目优先研究消费级算力下的完整训练链路 |
| 架构设计是否需要担心？ | LLaMA | RoPE + SwiGLU + RMSNorm + 无偏置在前沿模型上已验证 |

**总体评估**：文献支持在 TinyStories 上训练 17.23M 模型的可行性，但不保证后训练必然带来收益。本项目的规则奖励并非严格的势函数奖励塑形，不能直接套用 Ng et al. (1999) 的策略不变性结论；DPO-v1 的奖励过优化结果也说明必须用外部任务指标验证奖励设计。

---

## 14. 模型评测方法

本项目的目标是训练一个 TinyStories 领域的微型故事语言模型，而不是通用知识或代码模型。因此，MMLU、GSM8K、HumanEval 等通用榜单只能用于描述能力边界，不能作为主要成功标准。评测采用 HELM 所倡导的多指标思想，将同域语言建模、任务能力、开放生成质量、偏好对齐和效率分开报告，不把不同含义的 loss 合成为单一总分。

### 14.1 HELM：语言模型的整体评测

- **论文**：Liang, P., Bommasani, R., Lee, T., et al. (2022). *Holistic Evaluation of Language Models*.
- **arXiv**：https://arxiv.org/abs/2211.09110
- **本地文件**：`2211.09110_HELM.pdf`

**核心贡献：**
- 指出只报告 accuracy 或少量 benchmark 会掩盖模型能力之间的权衡
- 从准确性、校准、鲁棒性、公平性、偏见、毒性和效率等维度评测模型
- 强调统一场景、统一提示词、统一推理参数以及公开原始输出

**对本项目的意义：**
- Base、SFT、DPO、RSFT 必须在完全相同的数据、prompt、随机种子和解码参数下比较
- TinyStories PPL、SFT response NLL 和 DPO loss 含义不同，不能直接比较数值大小
- 报告模型参数量、生成吞吐量与完整原始样本，避免只展示有利案例

### 14.2 MT-Bench：LLM-as-a-Judge 与成对盲评

- **论文**：Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*.
- **arXiv**：https://arxiv.org/abs/2306.05685
- **本地文件**：`2306.05685_MT_Bench_LLM_Judge.pdf`

**核心贡献：**
- 使用强语言模型评价开放式生成，弥补 exact match 等指标无法判断表达质量的问题
- 以模型对战和 win/tie/loss 衡量人类偏好
- 系统分析位置偏差、长度偏好和自我增强偏差

**对本项目的意义：**
- 对相同 prompt 的 Base 与 SFT/DPO/RSFT 输出进行随机 A/B 盲评
- 评价 grammar、coherence、consistency、prompt relevance、creativity 和 age appropriateness
- A/B 顺序随机化，答案键与待评数据分离；抽样进行人工复核

### 14.3 MAUVE：机器文本与真实文本的分布差异

- **论文**：Pillutla, K., Swayamdipta, S., Zellers, R., et al. (2021). *MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers*.
- **arXiv**：https://arxiv.org/abs/2102.01454
- **本地文件**：`2102.01454_MAUVE.pdf`

**核心贡献：**
- 从分布层面比较模型生成文本与人类文本，而不是要求每个生成样本匹配唯一参考答案
- 同时反映生成质量与多样性之间的退化
- 可识别模式坍缩、低质量长尾等仅靠 PPL 或 Distinct-n 难以发现的问题

**对本项目的意义：**
- TinyStories 是开放式续写任务，同一 prompt 存在许多合理答案，BLEU/ROUGE 不适合作为核心指标
- MAUVE 可作为可选的分布评价；基础评测仍保留无需大型外部编码器的 PPL、Distinct-n、重复率和盲评

### 14.4 本项目的统一评测协议

| 评测层 | 数据 | 指标 | 解释 |
|---|---|---|---|
| 同域语言能力 | TinyStories `valid.bin` | NLL、PPL | 检测语言建模能力与微调遗忘 |
| SFT 续写能力 | SFT valid | response NLL/PPL | 仅评价被预测的 response token |
| 偏好能力 | DPO valid | preference accuracy、chosen-rejected margin、DPO loss | 检测模型是否更偏好 chosen |
| 开放生成 | 固定 prompt × 固定 seeds | Distinct-1/2/3、Repeat-3、EOS/句末率、关键词覆盖 | 检测多样性、循环和约束遵循 |
| 主观质量 | 随机 A/B 对 | 六维 rubric、win/tie/loss | 检测语法、连贯性、一致性和创造性 |
| 效率 | 相同硬件 | 参数量、token/s、耗时 | 描述消费级硬件可行性 |

统一评测入口为 `eval/comprehensive_eval.py`。结果保存为 JSON、CSV、Markdown，生成样本和盲评数据一并保留，以支持复现与误差分析。

### 14.5 新增参考文献 BibTeX

```bibtex
@article{liang2022helm,
  title={Holistic Evaluation of Language Models},
  author={Liang, Percy and Bommasani, Rishi and Lee, Tony and others},
  journal={arXiv preprint arXiv:2211.09110},
  year={2022}
}

@article{zheng2023judging,
  title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},
  author={Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and others},
  journal={arXiv preprint arXiv:2306.05685},
  year={2023}
}

@inproceedings{pillutla2021mauve,
  title={MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers},
  author={Pillutla, Krishna and Swayamdipta, Swabha and Zellers, Rowan and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

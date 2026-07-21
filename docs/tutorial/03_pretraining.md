# 03：预训练与优化

## 1. 数据管线

预训练不是把整篇文档逐条送入模型，而是：文本清洗 → tokenizer → token 流 → 固定长度窗口。项目把 token 保存为二进制数组，训练时读取长度为 `block_size + 1` 的片段，再拆成移位后的 input/target。

```bash
python scripts/prepare_pretrain_data.py
python train/pretrain.py --config configs/train_pretrain.yaml
```

正式配置：batch 16、梯度累积 8、序列长 256，单次 optimizer update 的有效 token 数约为：

$$16\times8\times256=32768.$$

注意项目进度条中的 `step` 是 micro-step；只有每 8 个 micro-step 更新一次参数。

## 2. AdamW

一阶、二阶动量为：

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,$$

$$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.$$

AdamW 将权重衰减与梯度更新解耦：

$$
\theta_{t+1}=(1-\eta_t\lambda)\theta_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

bias 和 norm 参数通常不衰减，具体分组见 `train/common.py`。

## 3. warmup 与余弦退火

训练开始时梯度统计不稳定，先线性 warmup：

$$\eta_t=\eta_{max}\frac{t}{T_w},\quad t<T_w.$$

之后余弦下降到最小学习率：

$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})
\left[1+\cos(\pi r_t)\right].
$$

## 4. 混合精度、梯度累积与裁剪

- FP16 前向/反向降低显存，`GradScaler` 避免小梯度下溢。
- 累积 $K$ 次时，每个 micro-batch loss 必须除以 $K$。
- 全局梯度范数裁剪控制突发大梯度：

$$g\leftarrow g\cdot\min\left(1,\frac{c}{\lVert g\rVert_2}\right).$$

核心训练骨架：

```python
with autocast(device_type="cuda", dtype=torch.float16):
    loss = model(input_ids, targets=targets)["loss"] / accum_steps
scaler.scale(loss).backward()

if (step + 1) % accum_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
```

## 5. 判断预训练是否有效

至少同时检查：

- train/valid NLL 是否下降且间距不过度扩大；
- PPL 与随机样例；
- 重复率、句末率和生成长度；
- checkpoint 是否可恢复；
- 固定 prompt、固定 seed 的纵向变化。

一个低 loss 模型仍可能复读；一个看似流畅的样例也可能只是挑出来的幸运结果。

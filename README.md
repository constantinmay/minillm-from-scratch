# Training LLM from Scratch

A complete LLM training pipeline implemented from scratch on a consumer-grade laptop (RTX 4060 8GB). No HuggingFace Transformers — every component is hand-written.

## What You'll Get

- **Understand how LLMs actually work** by building every layer yourself: RoPE, SwiGLU, RMSNorm, attention, generation
- **Experience the full training pipeline**: BPE Tokenizer → Pretraining → SFT → DPO → RSFT
- **Run on consumer hardware**: 17M parameters, 8GB GPU, complete training in hours
- **Learn from real experiments**: what works, what breaks, and why — with data to prove it

## Architecture

Following the LLaMA architecture strictly:

| Component | Implementation |
|-----------|---------------|
| Attention | Causal self-attention with RoPE (Rotary Position Embedding) |
| FFN | SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))` |
| Norm | RMSNorm (no bias, no mean centering) |
| Weight Tying | `lm_head.weight = token_embedding.weight` |
| All Linear | `bias=False` |

**Model config**: 6 layers, 6 heads, 384 embedding dim, ~17M parameters

## Project Structure

```
minillm-from-scratch/
├── model/                   # LLaMA architecture (attention, block, gpt, generation)
├── tokenizer/               # BPE tokenizer (vocab_size=8000)
├── train/                   # Training scripts (pretrain, sft, dpo, rsft)
├── eval/                    # Evaluation tools (metrics, loss, generation)
├── scripts/                 # Data preparation and comparison scripts
├── configs/                 # YAML configs for each training stage
├── tests/                   # Unit tests
├── docs/                    # Theory documentation with formulas
├── data/                    # Small training data samples
│   ├── sft_v2/             # SFT data (story continuations)
│   ├── dpo_v2/             # DPO data (real chosen vs model rejected)
│   └── rsft/               # RSFT data (model best-of-N selected)
├── generate_interactive.py  # Interactive text generation
└── requirements.txt
```

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA
- GPU with 8GB+ VRAM
- Basic knowledge of Python, PyTorch, and Transformer concepts

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/minillm-from-scratch.git
cd minillm-from-scratch
pip install -r requirements.txt
```

### 5-Minute Demo

If you have a pretrained checkpoint:

```bash
python generate_interactive.py checkpoints/base.pt
```

Type any prompt like `Once upon a time` and watch the model generate stories!

## Full Pipeline

### Step 1: Prepare Data

Download TinyStories dataset (or use your own text data):

```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")
# Save to data/raw/TinyStories-train.txt and data/raw/TinyStories-valid.txt
```

### Step 2: Train Tokenizer

```bash
python tokenizer/train_tokenizer.py
```

Trains a BPE tokenizer with vocab_size=8000 and special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`).

### Step 3: Preprocess Data

```bash
python scripts/prepare_pretrain_data.py
```

Tokenizes raw text and packs into binary format for efficient training.

### Step 4: Pretrain

```bash
python train/pretrain.py --config configs/train_pretrain.yaml
```

Trains the base model with next-token prediction. Expected: eval loss drops from ~8.0 to ~1.6 over 30k-50k steps (~2-4 hours on RTX 4060).

### Step 5: SFT (Supervised Fine-Tuning)

```bash
# Build SFT data (story continuation pairs)
python scripts/build_sft_data.py --input data/raw/TinyStories-valid.txt --output data/sft_v2/train.jsonl --num_samples 500

# Train
python train/sft.py --config configs/train_sft.yaml
```

### Step 6: DPO (Direct Preference Optimization)

```bash
# Build DPO data (real text as chosen, model output as rejected)
python scripts/build_dpo_data.py --input data/raw/TinyStories-valid.txt --output data/dpo_v2/train.jsonl --ckpt checkpoints/base.pt --num_samples 500

# Train
python train/dpo.py --config configs/train_dpo.yaml
```

### Step 7: RSFT (Reward-Selected Fine-Tuning)

```bash
# Generate and select best candidates
python train/rsft_generate.py --config configs/rsft_generate.yaml

# Split into train/valid
python scripts/split_rsft_data.py

# Train on selected data
python train/sft.py --config configs/train_rsft.yaml
```

## Experimental Findings

### Training Collapse on Small Models

During our experiments, we encountered a critical issue: **fine-tuning (SFT/DPO) degrades generation quality on small models**.

| Method | Result | Why |
|--------|--------|-----|
| Pretrain (50k steps) | Eval loss 1.63, fluent generation | All capacity used for language modeling |
| SFT | Degrades after ~100 steps | SFT gradient overwrites pretrain knowledge |
| DPO (bad data) | Severe degradation | Rejected samples were artificial garbage |
| DPO (good data) | Improved generation | Natural quality gap between chosen/rejected |
| RSFT | Stable, no degradation | Training data from model's own distribution |

### Data Quality is Decisive for DPO

The most important finding: **the same DPO algorithm on the same model produced completely different results depending on data quality**.

**Bad DPO data** (artificial rejected):
```
Chosen:   "She went to the park and played with her friends."
Rejected: "Ben Ben Ben Ben Ben Ben Ben"  ← artificial garbage
→ Model learned nothing useful, generation collapsed
```

**Good DPO data** (model-generated rejected):
```
Chosen:   Real TinyStories continuation (ground truth quality)
Rejected: Model-generated continuation (naturally worse but realistic)
→ Model learned meaningful preferences, generation improved
```

This demonstrates that for DPO, **both chosen and rejected must be realistic outputs** with a natural quality gap.

### Key Insight

17M-parameter models have just enough capacity for pretraining. Any fine-tuning that introduces out-of-distribution signals will overwrite the pretrained knowledge. The solution:

1. **RSFT**: safest — only uses model's own outputs
2. **DPO with good data**: effective — teaches meaningful preferences
3. **SFT**: riskiest — even with pretrain loss mixing, degradation occurs

For detailed theory and formulas, see [docs/theory.md](docs/theory.md).

## Running Tests

```bash
pytest tests/ -v
```

Tests cover: model shapes, causal masking, DPO loss, reward scoring, SFT labels, tokenizer.

## References

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)

## License

MIT

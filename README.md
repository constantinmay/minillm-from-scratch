# MiniLLM from Scratch

[English](README.md) | [简体中文](README_zh.md)

> **New to language models?** Start with the [English beginner tutorial](docs/tutorial/README_en.md) or the [中文入门教程](docs/tutorial/README.md). The seven notebooks interleave explanations, equations, source-code links, and runnable checks; no GPU is required for the tutorial exercises.

A reproducible 17.23M-parameter language-model project for studying instruction
following and preference alignment on one RTX 4060 Laptop GPU. The Transformer,
training losses, data builders, generation, and evaluation are implemented in
PyTorch without Hugging Face Transformers.

## Research question

Can a very small TinyStories base model learn a narrow, auditable instruction
space without the world knowledge of a general assistant? The project compares:

1. TinyStories base pretraining;
2. continuation-only SFT as a same-budget control;
3. four-task Instruction SFT;
4. DPO and reward-selected SFT initialized from Instruction SFT;
5. conservative DPO-v2 using only hard-success versus hard-failure pairs.

The four instruction tasks are story continuation, required-keyword story,
exact sentence count, and extractive question answering. This is a controlled
domain experiment, not a general-purpose chatbot.

## Model

| Component | Setting |
|---|---|
| Parameters | 17,232,768 |
| Context | 256 tokens |
| Vocabulary | 8,000-token BPE |
| Transformer | 6 layers, 6 heads, 384 hidden dimensions |
| Position | RoPE |
| Normalization | Pre-RMSNorm |
| MLP | SwiGLU, width 1,536 |
| Attention | causal PyTorch SDPA |
| Other | bias-free linear layers, tied embedding/LM head |

## Repository layout

```text
configs/       current model and training configurations
data/          processed TinyStories and generated instruction/alignment data
docs/          theory, experiment report, and documentation index
eval/          unified reproducible evaluator and metrics
model/         decoder-only Transformer implementation
scripts/       current data preparation/export scripts
tests/         unit and regression tests
tokenizer/     BPE training and runtime wrapper
train/         pretraining, SFT, and DPO trainers
papers/        tracked reference catalog; local PDFs are ignored by Git
```

Checkpoints, logs, raw data, local papers, and generated evaluation artifacts
are intentionally ignored by Git.

## Setup

```bash
conda create -n dl_1 python=3.10
conda activate dl_1
pip install -r requirements.txt
```

The tested local environment uses PyTorch with CUDA on an 8GB RTX 4060 Laptop
GPU. CPU execution is supported for tests and small inference runs.

## Reproduce the pipeline

### 1. Tokenizer and base model

Place TinyStories text files under `data/raw/`, then run:

```bash
python tokenizer/train_tokenizer.py
python scripts/prepare_pretrain_data.py
python train/pretrain.py --config configs/train_pretrain.yaml
```

The current experiments initialize from `checkpoints/base.pt`.

### 2. Instruction SFT and fair continuation control

```bash
python scripts/build_instruction_sft.py
python train/sft.py --config configs/train_sft.yaml

python scripts/build_instruction_sft.py \
  --output-dir data/instruction_sft_continuation \
  --task-mix continuation_only
python train/sft.py --config configs/train_sft_continuation.yaml
```

Both datasets contain 20,000/1,000/1,000 train/validation/test examples and use
the same optimization budget. Source groups do not cross dataset splits. Only
response tokens contribute to the primary SFT objective; a small full-sequence
pretraining loss reduces forgetting.

### 3. Shared candidate pool and alignment baselines

```bash
python scripts/build_instruction_alignment.py
python train/dpo.py --config configs/train_instruction_dpo.yaml
python train/sft.py --config configs/train_instruction_rsft.yaml
```

The candidate builder is append-only and supports `--resume`. DPO and RSFT are
derived from the same generated candidate pool.

### 4. Conservative DPO-v2

```bash
python scripts/build_strict_dpo.py
python train/dpo.py --config configs/train_instruction_dpo_v2.yaml
```

DPO-v2 excludes unconstrained continuation rewards. It retains only QA,
sentence-count, and keyword pairs where `chosen` passes the hard constraint and
`rejected` fails it, then balances the three tasks. Checkpoints at 100/200/300/400
microsteps permit external-metric early stopping; the current selected point is
200 microsteps.

## Evaluation

The evaluator reports each objective separately: TinyStories PPL, held-out
response NLL, strict-pair preference accuracy/margin, exact sentence count, QA
exact match, keyword coverage/all-success, repetition, ending rate, diversity,
throughput, raw generations, and randomized blind A/B pairs.

```bash
python eval/comprehensive_eval.py \
  --model Base=checkpoints/base.pt \
  --model InstructionSFT=checkpoints/instruction_sft/sft.pt \
  --model DPOv2=checkpoints/instruction_dpo_v2/dpo_step_200.pt \
  --model RSFT=checkpoints/instruction_rsft/rsft.pt \
  --seeds 42,123 \
  --temperature 0.01 \
  --top-k 1 \
  --output results/final_evaluation
```

No single aggregate score is used. In particular, preference accuracy can rise
while task success or language modeling degrades; the first DPO experiment is
retained as a documented reward-overoptimization result.

See [docs/experiment_report.md](docs/experiment_report.md) for the selected
results and limitations.

### Final result snapshot

The formal run uses 1,000 held-out prompts, two listed generation seeds, and
greedy top-1 decoding. These are separate metrics, not a combined score.

| Model | LM PPL ↓ | Sentence exact ↑ | QA EM ↑ | Keyword all ↑ |
|---|---:|---:|---:|---:|
| Base | 5.36 | 0.370 | 0.000 | 0.000 |
| ContinuationSFT | 5.70 | 0.345 | 0.000 | 0.024 |
| InstructionSFT | 5.80 | 0.615 | **0.850** | **0.048** |
| DPOv1 | 6.34 | 0.590 | 0.800 | 0.044 |
| DPOv2 (step 200) | 5.82 | **0.715** | 0.845 | 0.044 |
| RSFT | 5.88 | 0.605 | 0.845 | **0.048** |

The main positive result is narrow instruction learning from task-structured
SFT. DPO-v2 improves exact sentence control with little PPL regression, while
DPO-v1 demonstrates that stronger preference ranking can coincide with worse
external metrics. Keyword-all success remains low and is reported as an open
failure, not hidden behind average coverage.

## Side-by-side demo

```bash
python demo_compare.py --task qa \
  --input "Tim did not listen to his mom. Tim played all day." \
  --question "Who did not listen?"
```

The demo compares Instruction SFT, DPO-v2, and RSFT by default. It also supports
`continuation`, `keywords`, and `sentence_count` tasks.

## Tests

```bash
pytest tests -q
```

Tests cover causal masking, tensor shapes, shifted labels, prompt masking,
generation, task rewards, leakage-safe data construction, strict DPO export,
and the comprehensive evaluator.

## Documentation

- [Documentation index](docs/README_en.md)
- [Beginner tutorial — English](docs/tutorial/README_en.md)
- [初学者教程 — 中文](docs/tutorial/README.md)
- [Theory and equations](docs/theory.md)
- [Experiment report](docs/experiment_report.md)
- [References and implementation mapping](papers/references_and_analysis.md)

## Scope

This repository demonstrates a complete, resource-constrained research loop.
It does not claim general knowledge, robust semantic story judging, or
state-of-the-art benchmark performance. The strongest conclusion concerns
narrow instruction following under controlled data and compute.

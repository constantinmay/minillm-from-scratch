# MiniLLM from Scratch

[English](README.md) | [简体中文](README_zh.md)

[![CPU tests](https://github.com/constantinmay/minillm-from-scratch/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/constantinmay/minillm-from-scratch/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
[![Stable version: v1.1.0](https://img.shields.io/badge/stable-v1.1.0-green.svg)](https://github.com/constantinmay/minillm-from-scratch/tree/v1.1.0)

> **New to language models?** Start with the [English beginner tutorial](docs/tutorial/README_en.md) or the [中文入门教程](docs/tutorial/README.md). The seven notebooks interleave explanations, equations, source-code links, and runnable checks; no GPU is required for the tutorial exercises.

MiniLLM from Scratch is a hands-on project for building and training a small
language model on limited hardware. Instead of stopping at equations or calling
a ready-made model API, you implement and run the complete pipeline: tokenizer,
decoder-only Transformer, base pretraining, instruction SFT, preference
alignment, generation, and evaluation. The reference model has 17.23M
parameters and was trained on one 8GB RTX 4060 Laptop GPU using plain PyTorch,
without Hugging Face Transformers.

## Choose your path

- **Learn the concepts:** Run the bilingual tutorials. No GPU or pretrained
  checkpoint is required.
- **Try the trained models:** Download the tokenizer and inference checkpoints
  from a GitHub Release, then run `demo_compare.py`. **Release assets will be
  published with the next version; they are not published yet.**
- **Reproduce the experiments:** Prepare TinyStories and run tokenizer
  training, pretraining, SFT, and alignment stages.

## Why this project?

Modern frontier language models are trained on clusters that most learners
cannot access, while even billion-parameter training is often out of reach on a
consumer GPU. This repository asks a more practical question: **can a beginner
with one modest GPU still experience every important stage of language-model
training, observe the model change, and measure what worked?**

The goal is not to build a general assistant or claim state-of-the-art results.
It is to turn the whole training pipeline into something small enough to read,
run, modify, and debug yourself.

```text
TinyStories → BPE tokenizer → Base pretraining → Instruction SFT
            → DPO / RSFT → generation, metrics, and error analysis
```

In one local RTX 4060 run, the most memorable milestone was seeing the base
model begin to produce short text with recognizable English grammar after
roughly half an hour. This is an approximate observation from that run, not a
hardware-independent speed guarantee; the loss history and final samples kept
with the local run are the evidence behind this observation.

## What changes during training?

1. **Base pretraining** turns random token sequences into TinyStories-style
   English and teaches next-token prediction.
2. **Instruction SFT** teaches four limited task formats: continuation,
   required-keyword stories, exact sentence count, and extractive QA.
3. **DPO and RSFT** show how preference data can alter behavior—and how a proxy
   preference score can improve while useful task metrics get worse.
4. **Evaluation** separates language fluency, instruction success, preference
   ranking, and generation quality instead of hiding them in one score.

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

## Reproducibility and reliability

- Formal training configurations set `seed` explicitly. Python, NumPy,
  PyTorch, CUDA, and DataLoader shuffling use the same configured seed.
- SFT, DPO, and pretraining resume fail fast when a required checkpoint is
  missing instead of silently starting from random weights.
- Checkpoint v2 stores model, optimizer, scheduler, GradScaler, RNG, training
  configuration, and DataLoader progress. Legacy checkpoints remain loadable.
- Single-process pretraining resume restores the next shuffled batch as well as
  the training state; a tiny CPU regression checks the next batch, loss, and
  model parameters.
- GitHub Actions runs the complete test suite with CPU-only PyTorch.

A fixed seed improves repeatability on a given setup but does not promise
bitwise equality across hardware or PyTorch/CUDA versions. Multi-GPU/DDP and
cross-hardware bitwise reproducibility are not currently guaranteed.

### DPO response log-probability reduction

The reported DPO experiments use the token-wise `mean` reduction. `sum` is
available only as an interface for future ablations; neither reduction is
claimed to be universally better. Select it explicitly in a DPO config:

```yaml
logprob_reduction: mean  # or: sum
```

Changing this option does not rewrite the existing experiment report or its
results.

### Bootstrap confidence intervals

The comprehensive evaluator can optionally resample held-out prompts:

```bash
python eval/comprehensive_eval.py \
  --model Base=checkpoints/base.pt \
  --bootstrap-samples 1000 \
  --bootstrap-seed 42
```

This sample-level bootstrap describes sampling uncertainty on the current test
set; it is not a guarantee of cross-dataset or cross-domain generalization.
Multiple labels reported for one greedily decoded prompt are not treated as
independent statistical samples.

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

## How evaluation works

There is no single trustworthy “LLM score.” Each metric answers a different
question:

| What we want to know | Metric | Interpretation |
|---|---|---|
| Did Base learn next-token prediction? | validation NLL / PPL | lower is better |
| Does a model fit held-out task responses? | response NLL | lower is better |
| Does it obey verifiable instructions? | sentence exact, QA EM, keyword coverage/all | higher is better |
| Does it rank the chosen answer above the rejected one? | preference accuracy and margin, DPO loss | higher accuracy/margin and lower loss are better, but only for that preference set |
| Is generated text mechanically healthy? | repetition, distinct-n, ending rate, length | inspect separately; none proves semantic quality |
| Does one output look better to a reader? | fixed prompts, raw samples, randomized blind A/B | qualitative evidence, ideally judged independently |

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

The main practical observation is that a small model without broad world
knowledge can still learn a **limited, structured instruction protocol** when
the tasks and answers are carefully constructed. This is a result of the
experiment, not a claim of general instruction following. DPO-v2 improves exact
sentence control with little PPL regression, while DPO-v1 demonstrates that
stronger preference ranking can coincide with worse external metrics.

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

The current suite contains 80 tests covering causal masking, tensor shapes,
shifted labels, prompt masking, generation, task rewards, leakage-safe data
construction, strict DPO export, checkpoint recovery, inference export, the
comparison demo, and the comprehensive evaluator.

## Documentation

- [Documentation index](docs/README_en.md)
- [Beginner tutorial — English](docs/tutorial/README_en.md)
- [初学者教程 — 中文](docs/tutorial/README.md)
- [Theory and equations](docs/theory.md)
- [Experiment report](docs/experiment_report.md)
- [Model card](MODEL_CARD.md)
- [Release guide](docs/release.md)
- [Next-version release notes](RELEASE_NOTES.md)
- [References and implementation mapping](papers/references_and_analysis.md)

## Scope

This is an educational, resource-constrained implementation project. It does
not claim general knowledge, robust semantic story judging, or state-of-the-art
benchmark performance. Its strongest result is that the complete pipeline can
be made observable on consumer hardware, with narrow instruction learning as a
useful controlled experiment.

## License

This project is available under the [MIT License](LICENSE).

# Learn MiniLLM from Scratch

[English](README_en.md) | [简体中文](README.md)

This is a seven-chapter, notebook-first course for completing the MiniLLM
training pipeline on limited hardware. Frontier-scale training is far beyond
most beginners' reach, but a deliberately small model still lets us observe how
random output acquires grammar, what Base and SFT learn, and why preference
metrics can be misleading.

Each concept follows the same rhythm:

> intuition → equation → what the symbols mean → corresponding project code → a runnable check

You do not need a GPU for the course exercises. Long training commands appear as Markdown examples and never run automatically.

By the end, you should be able to reach three practical milestones:

1. explain next-token prediction from both its equation and implementation;
2. train Base and use loss, PPL, and samples to judge whether it has begun to
   learn basic grammar;
3. explain how SFT teaches a limited task protocol and why DPO/RSFT must be
   checked with external metrics.

Chapters 01–04 are the main path. DPO and RSFT in Chapter 05 are optional
advanced extensions; they are not required to complete Base and SFT first.

## What should I know first?

You can start if you can read basic Python functions, lists, dictionaries, and classes. The following knowledge is helpful but can be learned as you go:

- basic linear algebra: vectors, matrices, dot products, and shapes;
- basic probability: conditional probability, logarithms, and averages;
- basic PyTorch: tensors, `nn.Module`, `loss.backward()`, and optimizers;
- command-line basics: changing directories and running `python file.py`;
- Git basics are useful for experiments but not required for Chapters 1–6.

You do **not** need prior knowledge of reinforcement learning, RLHF, CUDA kernels, distributed training, or Hugging Face Transformers.

If tensors and autograd are completely new, first read the official [PyTorch tensor tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) and [autograd tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html).

## Ten-minute setup check

Run from the repository root:

```bash
conda activate dl_1
pip install -r requirements.txt -r requirements-docs.txt
python -m pytest tests/test_model_shapes.py tests/test_tokenizer.py -q
jupyter lab
```

Open Chapter 1 and run cells from top to bottom. A tutorial cell should finish in seconds on CPU. If a cell starts a long training job, that is a bug—please use the Markdown command only when you intentionally want to train.

## Chapters

| Chapter | Main question | Time | English | 中文 |
|---|---|---:|---|---|
| 01 | How do text, tokens, next-token labels, cross-entropy, and PPL connect? | 30–45 min | [Open](notebooks_en/01_tokenizer_and_lm.ipynb) | [打开](notebooks/01_tokenizer_and_lm.ipynb) |
| 02 | What happens inside a decoder-only Transformer? | 60–90 min | [Open](notebooks_en/02_transformer.ipynb) | [打开](notebooks/02_transformer.ipynb) |
| 03 | How does a stable pretraining loop work? | 45–60 min | [Open](notebooks_en/03_pretraining.ipynb) | [打开](notebooks/03_pretraining.ipynb) |
| 04 | How does response-only SFT teach a narrow instruction protocol? | 45–60 min | [Open](notebooks_en/04_sft.ipynb) | [打开](notebooks/04_sft.ipynb) |
| 05 | Where does DPO come from, and how do preference data and RSFT work? | 60–90 min | [Open](notebooks_en/05_alignment.ipynb) | [打开](notebooks/05_alignment.ipynb) |
| 06 | Which metrics answer which practical questions? | 45–60 min | [Open](notebooks_en/06_evaluation.ipynb) | [打开](notebooks/06_evaluation.ipynb) |
| 07 | How do I reproduce, interpret, and extend the full pipeline? | 30–45 min | [Open](notebooks_en/07_reproduce.ipynb) | [打开](notebooks/07_reproduce.ipynb) |

Recommended order: `01 → 02 → 03 → 04 → 05 → 06 → 07`. If you already understand Transformers, start at Chapter 04 but still run the shifted-label check in Chapter 01.

## Two learning paths

### Concept path — no checkpoint required

Read Chapters 01–06 and run the small checks. This path explains the mathematics and code without downloading raw TinyStories data or training a model.

### Reproduction path — GPU recommended

Finish Chapters 01–06, then use Chapter 07 together with:

- [root README](../../README.md) for exact commands;
- [training and evaluation report](../experiment_report.md) for the protocol and results;
- [paper-to-code map](../../papers/references_and_analysis.md) for primary sources.

## How to read a chapter

1. Predict tensor shapes before running a cell.
2. Read the equation and name every symbol in plain language.
3. Run the adjacent code cell and compare the output with your prediction.
4. Follow the source links at the end of the chapter.
5. Change one small value and explain what should happen before rerunning.

Do not treat the random tiny-model output as a research result. Formal conclusions come from the [experiment report](../experiment_report.md) and local `results/final_evaluation/` artifacts.

## Common beginner problems

- **`ModuleNotFoundError`:** launch Jupyter from the repository root, not from `docs/tutorial/`.
- **CUDA out of memory:** tutorial checks run on CPU; formal training needs the configured GPU environment.
- **A loss value looks different:** random teaching examples are not trained checkpoints.
- **A formula feels too fast:** write down tensor shapes first; most confusion comes from mixing batch, time, head, and vocabulary dimensions.
- **The model fails a custom prompt:** this 17M model is a controlled TinyStories model, not a general assistant.

## Maintenance guarantee

Chinese and English notebooks share identical code cells. Tests execute every cell and verify language parity, local links, equation coverage, and the explanation-before-code structure.

# MiniLLM Model Card

## Model summary

MiniLLM is a decoder-only Transformer with **17,232,768 parameters**, counted
from the current `configs/model_config.yaml` and `MiniLLM.num_params()`.
The architecture uses RoPE, Pre-RMSNorm, SwiGLU, and causal self-attention via
PyTorch SDPA. It has 6 layers, 6 attention heads, a 384-dimensional hidden
state, a 1,536-dimensional MLP, a 256-token context, and an 8,000-token BPE
vocabulary.

The project prioritizes readable teaching code and small controlled experiments,
not production inference performance. Training uses plain PyTorch and does not
depend on the Hugging Face Transformers training framework.

## Training stages

The repository implements the following stages:

1. BPE tokenizer training.
2. TinyStories base pretraining.
3. Continuation-only SFT as a fair control.
4. Multi-task instruction SFT.
5. Direct Preference Optimization (DPO), including a conservative DPO-v2 set.
6. Rejection-sampling fine-tuning (RSFT).

## Training data

The primary language-modeling domain is TinyStories. Instruction and preference
examples are constructed from that narrow domain for continuation, required-word
stories, exact sentence counts, and extractive QA. Consequently, the model does
not provide broad knowledge coverage, and these results should not be
extrapolated to general-purpose LLMs.

## Intended uses

Appropriate uses include:

- learning Transformer and language-model training internals;
- studying small-model pretraining and alignment pipelines;
- classroom demonstrations and runnable tutorials;
- controlled experiments on the repository's limited tasks;
- testing checkpoint, evaluation, and reproducibility workflows.

## Out-of-scope uses

MiniLLM is not intended for:

- use as a general chat assistant;
- medical, legal, financial, or other high-risk advice;
- factual knowledge retrieval;
- code generation or mathematical reasoning;
- safety-critical systems;
- deployment in a real production service.

## Evaluation

Metrics are kept separate because they answer different questions:

- validation NLL/PPL measures next-token fit on held-out TinyStories;
- response NLL measures fit to held-out task responses;
- sentence exactness, QA exact match, and keyword metrics measure narrow,
  mechanically verifiable instruction success;
- preference accuracy, margin, and DPO loss measure ranking on a particular
  preference set, not general capability;
- repetition, distinct-n, ending rate, and length describe mechanical generation
  properties and do not establish semantic quality.

The selected results already reported by the project are reproduced here
without recomputation:

| Model | LM PPL ↓ | Sentence exact ↑ | QA EM ↑ | Keyword all ↑ |
|---|---:|---:|---:|---:|
| Base | 5.36 | 0.370 | 0.000 | 0.000 |
| ContinuationSFT | 5.70 | 0.345 | 0.000 | 0.024 |
| InstructionSFT | 5.80 | 0.615 | **0.850** | **0.048** |
| DPOv1 | 6.34 | 0.590 | 0.800 | 0.044 |
| DPOv2 (step 200) | 5.82 | **0.715** | 0.845 | 0.044 |
| RSFT | 5.88 | 0.605 | 0.845 | **0.048** |

DPO-v1 is intentionally retained as a negative result: preference ranking can
improve while external task metrics and language modeling degrade, an example
of reward overoptimization. Higher preference accuracy therefore does not
guarantee stronger external-task capability. The project does not create a
theoretically unsupported aggregate score. See the
[experiment report](docs/experiment_report.md) for the original analysis.

## Limitations

- The model is small and its training domain is narrow.
- Experiments cover a limited number of random seeds, and some validation sets
  are small.
- Inference currently has no KV cache.
- Multi-GPU/DDP training and restart reproducibility have not been validated.
- Cross-hardware bitwise reproducibility is not claimed.
- This is not a general-purpose large language model.

## Checkpoint formats

- **Training checkpoint v2** contains model, optimizer, scheduler, GradScaler,
  RNG, training configuration, and DataLoader progress for training recovery.
- **Legacy training checkpoints** remain loadable when they contain compatible
  model weights and configuration.
- **Inference checkpoint v1** contains only model weights, model configuration,
  training-stage provenance, safe release metadata, and source hashes. It omits
  optimizer, scheduler, GradScaler, RNG, DataLoader state, data, and local paths.
- The tokenizer is distributed as a separate JSON file. Release assets should
  always be verified against their generated SHA256 checksum files.

See the [release guide](docs/release.md) for export and validation commands.

## License

Code and documentation are provided under the [MIT License](LICENSE).

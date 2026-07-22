# v1.2.0 release notes (draft)

These notes prepare the next release. **v1.2.0 and its model assets have not yet
been published.**

## Highlights

- Unified random seeding for Python, NumPy, PyTorch, CUDA, and DataLoader
  shuffling.
- Fail-fast validation for required SFT, DPO, and resume checkpoints.
- Checkpoint v2 with optimizer, scheduler, scaler, RNG, configuration, and
  DataLoader progress.
- Exact single-process pretraining resume to the next shuffled batch.
- Compatibility with legacy training checkpoints.
- Configurable DPO `mean`/`sum` response log-probability reduction; reported
  experiments retain `mean`.
- Task-level evaluation and optional prompt-level bootstrap confidence
  intervals.
- CPU-only continuous integration and regression tests.
- Verified inference checkpoint export with SHA256 metadata.
- A model card and manual release guide.

## Compatibility

- The model architecture is unchanged.
- Existing loss functions and optimizer definitions are unchanged.
- Formal DPO configurations continue to use `logprob_reduction: mean`.
- Existing experiment reports and numerical results are unchanged.
- Legacy checkpoints remain loadable.

## Limitations

- Full GPU training was not rerun for this release preparation.
- Multi-GPU/DDP has not been validated.
- Cross-hardware bitwise reproducibility is not claimed.
- KV cache is not implemented.
- Release assets must be limited to checkpoints that actually exist locally and
  pass export, checksum, reload, logits, and privacy checks.

No GitHub Release or model asset has been published by this preparation work.

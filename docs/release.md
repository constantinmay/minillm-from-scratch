# Release guide

This guide prepares a release; it does not indicate that model assets have
already been published. The current latest tag is `v1.1.0`, so the next planned
minor version is `v1.2.0`.

## 1. Pre-release checks

Run from a clean checkout after the release-preparation PR is merged:

```powershell
git status --short
git diff --check
python -m compileall model train eval tokenizer scripts
python -m pytest tests -q --basetemp .pytest-tmp
Remove-Item -Recurse -Force .pytest-tmp
```

Confirm that the `master` GitHub Actions run succeeds before creating a tag or
Release. Do not publish while CI is failing, and never overwrite an existing
tag.

## 2. Export local assets

Only export checkpoints that really exist locally. For example:

```powershell
python scripts/export_inference_checkpoint.py `
  --input checkpoints/instruction_sft/sft.pt `
  --output release_assets/instruction_sft_inference.pt `
  --stage instruction_sft `
  --tokenizer tokenizer/minillm_tokenizer.json
```

Repeat for DPO-v2 or RSFT only when the selected checkpoint exists and has been
verified. Never create placeholders or claim a missing asset is included.
Candidate assets are:

```text
minillm_tokenizer.json
instruction_sft_inference.pt
dpo_v2_inference.pt
rsft_inference.pt
SHA256SUMS (or per-checkpoint .sha256 files)
MODEL_CARD.md
```

`release_assets/` is ignored by Git. Exported `.pt` files must not be committed.

## 3. Verify assets

The export command reloads the inference checkpoint, checks parameter names and
shapes, runs a fixed tiny input through both source and exported models, and
requires matching logits within floating-point tolerance. It also emits a
SHA256 sidecar calculated from the actual output.

Before uploading, additionally:

1. Recompute each SHA256 and compare it with the sidecar.
2. Record file sizes.
3. Load each checkpoint on CPU and run a minimal `demo_compare.py` command.
4. Confirm the payload has no optimizer, scheduler, scaler, RNG, DataLoader
   state, environment variables, credentials, usernames, or absolute paths.
5. Copy the exact tokenizer file referenced by its checksum metadata.

Example demo:

```powershell
python demo_compare.py `
  --model InstructionSFT=release_assets/instruction_sft_inference.pt `
  --task raw `
  --input "Once upon a time" `
  --device cpu `
  --max-new-tokens 8
```

## 4. Create the GitHub Release manually

First merge the release-preparation PR, wait for the resulting `master` CI run,
then create the tag and Release. Do not run these commands as part of release
preparation itself:

```powershell
gh release create v1.2.0 `
  --title "v1.2.0 — Reproducible training and evaluation" `
  --notes-file RELEASE_NOTES.md `
  release_assets/*
```

Review the upload list before confirmation. Do not overwrite an existing tag or
publish unverified or nonexistent assets.

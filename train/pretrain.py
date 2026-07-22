"""Pretraining script for MiniLLM."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from model.config import MiniLLMConfig
from model.gpt import MiniLLM
from model.generation import generate
from tokenizer.tokenizer_utils import MiniLLMTokenizer
from train.common import (
    get_device,
    load_config,
    create_optimizer,
    create_scheduler,
    save_checkpoint,
    load_checkpoint,
    checkpoint_model_state,
    make_dataloader_generator,
    require_file,
    restore_rng_state,
    set_seed,
    validate_checkpoint_model_config,
    validate_training_config,
    write_run_manifest,
    AvgMetric,
)


class PretrainDataset(Dataset):
    """Memory-mapped dataset for pretraining.

    Reads uint16 tokens from a binary file, serves (input_ids, targets) pairs
    of length block_size. Targets are shifted by one position relative to inputs.
    """

    def __init__(self, data_path: str, block_size: int):
        require_file(data_path, "pretraining token data")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        if len(self.data) <= block_size:
            raise ValueError(
                f"Pretraining dataset must contain more than {block_size} tokens: "
                f"{os.path.abspath(data_path)}"
            )

    def __len__(self):
        # Number of full sequences we can extract
        return max(0, (len(self.data) - 1) // self.block_size)

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        input_ids = torch.from_numpy(self.data[start:end].astype(np.int64))
        # Targets are shifted by 1: the next token at each position
        targets = torch.from_numpy(self.data[start + 1 : end + 1].astype(np.int64))
        return input_ids, targets


@torch.no_grad()
def evaluate_loss(model, dataloader, device, max_batches=None):
    """Evaluate loss on a dataloader. Returns average loss."""
    model.eval()
    losses = []
    for i, (input_ids, targets) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        result = model(input_ids, targets=targets)
        losses.append(result["loss"].item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


@torch.no_grad()
def generate_samples(model, tokenizer, device, max_new_tokens=128, num_samples=3):
    """Generate sample text for inspection during training."""
    model.eval()
    samples = []
    prompts = ["The", "In", "Once upon a time"]
    prompts = prompts[:num_samples]
    for prompt in prompts:
        input_ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long, device=device
        )
        output_ids = generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=40,
            eos_token_id=tokenizer.eos_id(),
        )
        text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        samples.append(f"Prompt: {prompt!r}\nGenerated: {text}")
    model.train()
    return samples


def train_pretrain(config_path: str, resume_from: str = None):
    """Main pretraining loop.

    1. Load config, create model, create dataset/dataloader
    2. FP16 mixed precision with torch.amp.autocast + GradScaler
    3. Gradient accumulation
    4. Periodic: eval loss, save checkpoint, generate sample text
    5. Log with tqdm
    """
    # Load config
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    # Model config
    model_config = MiniLLMConfig.from_yaml(cfg["model_config"])

    validate_training_config(cfg, model_config.block_size)
    require_file(cfg["tokenizer_path"], "tokenizer")
    require_file(cfg["data_path"], "pretraining dataset")
    require_file(cfg["eval_data_path"], "pretraining validation dataset")

    # Tokenizer
    tokenizer = MiniLLMTokenizer(cfg["tokenizer_path"])

    # Create model
    model = MiniLLM(model_config).to(device)
    print(f"Model parameters: {model.num_params():,}")

    # Datasets
    block_size = model_config.block_size
    train_dataset = PretrainDataset(cfg["data_path"], block_size)
    eval_dataset = PretrainDataset(cfg["eval_data_path"], block_size)

    batch_size = cfg["batch_size"]
    grad_accum_steps = cfg["gradient_accumulation_steps"]

    train_generator = make_dataloader_generator(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=train_generator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Optimizer and scheduler
    max_steps = cfg["max_steps"]
    warmup_steps = cfg["warmup_steps"]
    lr = cfg["learning_rate"]
    min_lr = cfg["min_lr"]
    weight_decay = cfg["weight_decay"]

    optimizer = create_optimizer(model, lr, weight_decay)
    # scheduler.step() is called every grad_accum_steps, so divide by it
    scheduler = create_scheduler(
        optimizer, max_steps // grad_accum_steps,
        warmup_steps // grad_accum_steps, lr, min_lr,
    )

    # AMP scaler
    dtype_str = cfg.get("dtype", "float16")
    if dtype_str == "float16":
        dtype = torch.float16
    elif dtype_str == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    use_amp = dtype in (torch.float16, torch.bfloat16) and device == "cuda"
    scaler = GradScaler("cuda", enabled=(dtype == torch.float16 and use_amp))

    # Training loop
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    save_every = cfg.get("save_every", 5000)
    eval_every = cfg.get("eval_every", 1000)
    log_every = cfg.get("log_every", 100)
    generate_every = cfg.get("generate_every", 2000)
    num_generate_samples = cfg.get("num_generate_samples", 3)
    generate_max_new_tokens = cfg.get("generate_max_new_tokens", 128)
    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")

    os.makedirs(checkpoint_dir, exist_ok=True)
    manifest_checkpoints = [resume_from] if resume_from else []
    write_run_manifest(
        checkpoint_dir,
        config_path=config_path,
        config=cfg,
        seed=seed,
        device=device,
        checkpoint_paths=manifest_checkpoints,
    )

    # Resume from checkpoint if provided
    step = 0
    resume_dataloader_state = None
    if resume_from is not None:
        ckpt = load_checkpoint(resume_from, device)
        validate_checkpoint_model_config(ckpt, model_config, "pretraining checkpoint")
        model.load_state_dict(checkpoint_model_state(ckpt))
        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        step = ckpt.get("step", 0)
        restore_rng_state(ckpt.get("rng_state"))
        resume_dataloader_state = ckpt.get("dataloader_state")
        print(f"Resumed from {resume_from} at step {step}")

    loss_metric = AvgMetric()
    best_eval_loss = float("inf")
    model.train()

    # Training log
    log_records = []  # list of dicts: {"step": ..., "train_loss": ..., "eval_loss": ..., "lr": ...}

    pbar = tqdm(total=max_steps, desc="Pretraining")
    batches_consumed = 0
    if resume_dataloader_state:
        epoch_generator_state = resume_dataloader_state.get("epoch_generator_state")
        batches_consumed = int(resume_dataloader_state.get("batches_consumed", 0))
        if epoch_generator_state is not None:
            train_generator.set_state(epoch_generator_state)
    epoch_generator_state = train_generator.get_state()
    data_iter = iter(train_loader)
    for _ in range(batches_consumed):
        try:
            next(data_iter)
        except StopIteration as exc:
            raise ValueError(
                "Checkpoint dataloader cursor exceeds the current dataset"
            ) from exc

    while step < max_steps:
        try:
            input_ids, targets = next(data_iter)
        except StopIteration:
            epoch_generator_state = train_generator.get_state()
            data_iter = iter(train_loader)
            batches_consumed = 0
            input_ids, targets = next(data_iter)
        batches_consumed += 1

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        # Forward with AMP
        with autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            result = model(input_ids, targets=targets)
            loss = result["loss"] / grad_accum_steps

        # Backward
        scaler.scale(loss).backward()

        loss_metric.update(loss.item() * grad_accum_steps)

        # Gradient accumulation
        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss_metric.avg:.4f}", "step": step})

        # Logging
        if step % log_every == 0:
            current_lr = scheduler.get_last_lr()[0]
            train_loss = loss_metric.avg
            pbar.write(
                f"Step {step} | Loss: {train_loss:.4f} | LR: {current_lr:.2e}"
            )
            log_records.append({
                "step": step, "train_loss": train_loss,
                "eval_loss": None, "lr": current_lr,
            })
            loss_metric.reset()

        # Evaluation
        if step % eval_every == 0:
            eval_loss = evaluate_loss(model, eval_loader, device, max_batches=50)
            current_lr = scheduler.get_last_lr()[0]
            pbar.write(f"Step {step} | Eval Loss: {eval_loss:.4f}")
            log_records.append({
                "step": step, "train_loss": None,
                "eval_loss": eval_loss, "lr": current_lr,
            })
            model.train()

            # Save best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_path = os.path.join(checkpoint_dir, "base.pt")
                save_checkpoint(
                    model, optimizer, scheduler, step, best_path, model_config,
                    scaler=scaler, dataloader_state={
                        "epoch_generator_state": epoch_generator_state,
                        "batches_consumed": batches_consumed,
                    },
                    training_config=cfg,
                    checkpoint_type="pretrain_training",
                )
                pbar.write(f"New best model! Eval Loss: {eval_loss:.4f} -> {best_path}")

        # Generate samples
        if step % generate_every == 0:
            samples = generate_samples(
                model, tokenizer, device, generate_max_new_tokens, num_generate_samples
            )
            pbar.write(f"--- Step {step} Samples ---")
            for sample in samples:
                pbar.write(sample)
            pbar.write("---")
            model.train()

        # Save periodic checkpoint
        if step % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"pretrain_step_{step}.pt")
            save_checkpoint(
                model, optimizer, scheduler, step, ckpt_path, model_config,
                scaler=scaler, dataloader_state={
                    "epoch_generator_state": epoch_generator_state,
                    "batches_consumed": batches_consumed,
                },
                training_config=cfg,
                checkpoint_type="pretrain_training",
            )
            pbar.write(f"Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(checkpoint_dir, "pretrain_final.pt")
    save_checkpoint(
        model, optimizer, scheduler, step, final_path, model_config,
        scaler=scaler, dataloader_state={
            "epoch_generator_state": epoch_generator_state,
            "batches_consumed": batches_consumed,
        },
        training_config=cfg,
        checkpoint_type="pretrain_training",
    )
    pbar.write(f"Final checkpoint saved: {final_path}")
    pbar.close()

    # Save training log
    log_path = os.path.join(checkpoint_dir, "pretrain_log.json")
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_records, f, indent=2)
    print(f"Training log saved to {log_path}")

    # Plot training curves
    train_steps = [r["step"] for r in log_records if r["train_loss"] is not None]
    train_losses = [r["train_loss"] for r in log_records if r["train_loss"] is not None]
    eval_steps = [r["step"] for r in log_records if r["eval_loss"] is not None]
    eval_losses = [r["eval_loss"] for r in log_records if r["eval_loss"] is not None]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Train loss
    if train_steps:
        axes[0].plot(train_steps, train_losses, alpha=0.6, linewidth=0.5)
        # Smoothed line
        if len(train_losses) > 10:
            window = min(50, len(train_losses) // 5)
            smoothed = np.convolve(train_losses, np.ones(window)/window, mode="valid")
            axes[0].plot(train_steps[:len(smoothed)], smoothed, color="red", linewidth=2)
        axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")

    # Eval loss
    if eval_steps:
        axes[1].plot(eval_steps, eval_losses, "o-", color="green", linewidth=2)
        axes[1].set_title("Eval Loss")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Loss")

    # Learning rate
    lr_steps = [r["step"] for r in log_records if r.get("lr") is not None]
    lr_values = [r["lr"] for r in log_records if r.get("lr") is not None]
    if lr_steps:
        axes[2].plot(lr_steps, lr_values, color="orange", linewidth=2)
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("LR")

    plt.tight_layout()
    curve_path = os.path.join(checkpoint_dir, "pretrain_loss.png")
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {curve_path}")

    print("Pretraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain MiniLLM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_pretrain.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g. checkpoints/pretrain_final.pt)",
    )
    args = parser.parse_args()
    train_pretrain(args.config, resume_from=args.resume)

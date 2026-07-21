"""Supervised Fine-Tuning (SFT) script for MiniLLM."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from model.config import MiniLLMConfig
from model.gpt import MiniLLM
from tokenizer.tokenizer_utils import MiniLLMTokenizer
from train.common import (
    get_device,
    load_config,
    create_optimizer,
    create_scheduler,
    save_checkpoint,
    load_checkpoint,
    AvgMetric,
)


SFT_TEMPLATE = "{prompt} {response}"  # No instruction prefix - pure story continuation


def build_pretrain_targets(input_ids, response_labels):
    """Restore next-token targets for prompt tokens while keeping padding masked.

    ``response_labels`` already contains shifted response/EOS targets and uses
    -100 both for the prompt prefix and batch padding.  Only the masked prefix
    before the first response target should be restored for the auxiliary
    language-modeling loss; trailing padding must remain ignored.
    """
    targets = response_labels.clone()
    for row in range(targets.size(0)):
        valid = (response_labels[row] != -100).nonzero(as_tuple=False)
        if valid.numel() == 0:
            continue
        first_response_pos = int(valid[0].item())
        if first_response_pos > 0:
            targets[row, :first_response_pos] = input_ids[
                row, 1 : first_response_pos + 1
            ]
    return targets


class SFTDataset(Dataset):
    """Supervised Fine-Tuning dataset.

    Reads JSONL files with prompt/response fields.
    Uses a minimal format: "{prompt} {response}" - no special template tokens.
    The prompt portion is masked with -100 so loss is only on the response.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: MiniLLMTokenizer,
        max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = item["prompt"].strip()
        response = item["response"].strip()

        bos_id = self.tokenizer.bos_id()
        eos_id = self.tokenizer.eos_id()

        prompt_ids = self.tokenizer.encode(prompt + " ", add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)

        # Reserve one target position for the response (or EOS) even when the
        # prompt is longer than the context window.
        prompt_ids = prompt_ids[: max(0, self.max_length - 1)]

        # Build a sequence with one extra token, then shift it into causal-LM
        # inputs and targets: input[t] predicts full_ids[t + 1].
        full_ids = [bos_id] + prompt_ids + response_ids + [eos_id]
        full_ids = full_ids[: self.max_length + 1]
        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # The first response token is predicted after the final prompt token.
        # Therefore exactly len(prompt_ids) target positions belong to prompt.
        labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_sft(batch):
    """Collate function that pads sequences to the same length in a batch."""
    input_ids, labels = zip(*batch)
    max_len = max(len(seq) for seq in input_ids)

    padded_inputs = []
    padded_labels = []
    for ids, labs in zip(input_ids, labels):
        pad_len = max_len - len(ids)
        padded_inputs.append(torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)]))
        padded_labels.append(torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)]))

    return torch.stack(padded_inputs), torch.stack(padded_labels)


@torch.no_grad()
def evaluate_loss(model, dataloader, device, max_batches=None):
    """Evaluate loss on a dataloader. Returns average loss."""
    model.eval()
    losses = []
    for i, (input_ids, labels) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        result = model(input_ids, targets=labels)
        losses.append(result["loss"].item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


def train_sft(config_path: str):
    """SFT training loop.

    Load base model, train on instruction-response data.
    """
    # Load config
    cfg = load_config(config_path)
    device = get_device()
    print(f"Using device: {device}")

    # Model config
    model_config = MiniLLMConfig.from_yaml(cfg["model_config"])

    # Tokenizer
    tokenizer = MiniLLMTokenizer(cfg["tokenizer_path"])

    # Load base model
    model = MiniLLM(model_config).to(device)
    base_model_path = cfg.get("base_model_path")
    if base_model_path and os.path.exists(base_model_path):
        checkpoint = load_checkpoint(base_model_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded base model from {base_model_path}")
    else:
        print("Training from scratch (no base model found).")

    print(f"Model parameters: {model.num_params():,}")

    # Datasets
    max_length = model_config.block_size
    train_dataset = SFTDataset(cfg["data_path"], tokenizer, max_length)
    eval_dataset = SFTDataset(cfg["eval_data_path"], tokenizer, max_length)

    batch_size = cfg["batch_size"]
    grad_accum_steps = cfg["gradient_accumulation_steps"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_sft,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_sft,
    )

    # Optimizer and scheduler
    max_steps = cfg["max_steps"]
    warmup_steps = cfg["warmup_steps"]
    lr = cfg["learning_rate"]
    min_lr = cfg["min_lr"]
    weight_decay = cfg["weight_decay"]

    optimizer = create_optimizer(model, lr, weight_decay)
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

    # Training config
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    save_every = cfg.get("save_every", 500)
    eval_every = cfg.get("eval_every", 200)
    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")
    checkpoint_name = cfg.get("checkpoint_name", "sft.pt")
    checkpoint_stem = os.path.splitext(checkpoint_name)[0]

    os.makedirs(checkpoint_dir, exist_ok=True)

    step = 0
    loss_metric = AvgMetric()
    model.train()

    pbar = tqdm(total=max_steps, desc="SFT Training")
    data_iter = iter(train_loader)

    while step < max_steps:
        try:
            input_ids, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            input_ids, labels = next(data_iter)

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward with AMP - compute both SFT loss and pretrain loss
        pretrain_weight = cfg.get("pretrain_loss_weight", 0.0)
        with autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            # SFT loss: only on response tokens (labels with -100 masked)
            result = model(input_ids, targets=labels)
            sft_loss = result["loss"]

            # Pretrain loss: on ALL tokens (no masking)
            if pretrain_weight > 0:
                pretrain_targets = build_pretrain_targets(input_ids, labels)
                pretrain_result = model(input_ids, targets=pretrain_targets)
                pretrain_loss = pretrain_result["loss"]
                loss = (sft_loss + pretrain_weight * pretrain_loss) / grad_accum_steps
            else:
                loss = sft_loss / grad_accum_steps

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

        # Evaluation
        if step % eval_every == 0:
            eval_loss = evaluate_loss(model, eval_loader, device, max_batches=50)
            current_lr = scheduler.get_last_lr()[0]
            pbar.write(
                f"Step {step} | Train Loss: {loss_metric.avg:.4f} | "
                f"Eval Loss: {eval_loss:.4f} | LR: {current_lr:.2e}"
            )
            loss_metric.reset()
            model.train()

        # Save checkpoint
        if step % save_every == 0:
            ckpt_path = os.path.join(
                checkpoint_dir, f"{checkpoint_stem}_step_{step}.pt"
            )
            save_checkpoint(model, optimizer, scheduler, step, ckpt_path, model_config)
            pbar.write(f"Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(checkpoint_dir, checkpoint_name)
    save_checkpoint(model, optimizer, scheduler, step, final_path, model_config)
    pbar.write(f"Final checkpoint saved: {final_path}")
    pbar.close()

    print("SFT training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) for MiniLLM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_sft.yaml",
        help="Path to SFT config YAML",
    )
    args = parser.parse_args()
    train_sft(args.config)

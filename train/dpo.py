"""Direct Preference Optimization (DPO) training script for MiniLLM."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    checkpoint_model_state,
    make_dataloader_generator,
    require_file,
    set_seed,
    validate_checkpoint_model_config,
    validate_training_config,
    write_run_manifest,
    AvgMetric,
)


SFT_TEMPLATE = "{prompt} {response}"  # No template - raw text


def compute_logprobs(model, input_ids, targets):
    """Forward pass, compute average log-prob of response tokens (non -100).

    Uses F.log_softmax then gather to extract token log-probabilities.

    Args:
        model: MiniLLM model.
        input_ids: (B, T) token IDs.
        targets: (B, T) target token IDs, with -100 for masked positions.

    Returns:
        (B,) tensor of per-sample average log-probabilities for response tokens.
    """
    result = model(input_ids)
    logits = result["logits"]  # (B, T, V)

    # Create mask: 1 where target is not -100, 0 otherwise
    mask = (targets != -100).float()  # (B, T)

    # Replace -100 with 0 to avoid index out of bounds in gather
    safe_targets = targets.clamp(min=0)

    # Compute log probabilities via log_softmax then gather
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

    # Gather the log-prob of the target token at each position
    per_token_logps = log_probs.gather(
        dim=-1, index=safe_targets.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)

    # Compute average log-prob over valid (non-masked) tokens per sample
    token_counts = mask.sum(dim=-1).clamp(min=1.0)  # (B,)
    avg_logps = (per_token_logps * mask).sum(dim=-1) / token_counts  # (B,)

    return avg_logps


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1,
):
    """Compute DPO loss.

    DPO loss: -log sigmoid(beta * ((logp_policy_chosen - logp_policy_rejected)
                                   - (logp_ref_chosen - logp_ref_rejected)))

    Args:
        policy_chosen_logps: (B,) avg log-probs from policy model for chosen responses.
        policy_rejected_logps: (B,) avg log-probs from policy model for rejected responses.
        ref_chosen_logps: (B,) avg log-probs from reference model for chosen responses.
        ref_rejected_logps: (B,) avg log-probs from reference model for rejected responses.
        beta: DPO temperature parameter.

    Returns:
        Scalar loss (mean over batch).
    """
    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    ref_log_ratio = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_log_ratio - ref_log_ratio)
    loss = -F.logsigmoid(logits).mean()
    return loss


class DPODataset(Dataset):
    """DPO dataset.

    Reads JSONL with prompt/chosen/rejected fields.
    Tokenizes using the SFT template for both chosen and rejected responses.
    Labels use -100 for prompt portion.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: MiniLLMTokenizer,
        max_length: int = 256,
    ):
        require_file(data_path, "DPO dataset")
        if max_length <= 0:
            raise ValueError("DPO max_length must be positive")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                required = ("prompt", "chosen", "rejected")
                missing = {field for field in required if field not in item}
                if missing:
                    raise ValueError(
                        f"DPO sample is missing {', '.join(sorted(missing))}: {item}"
                    )
                if any(not isinstance(item[field], str) for field in required):
                    raise ValueError("DPO prompt, chosen, and rejected must be strings")
                if not item["chosen"].strip() or not item["rejected"].strip():
                    raise ValueError("DPO chosen and rejected responses must not be empty")
                if item["chosen"].strip() == item["rejected"].strip():
                    raise ValueError("DPO chosen and rejected responses must differ")
                self.samples.append(item)

        if not self.samples:
            raise ValueError(f"DPO dataset is empty: {Path(data_path).resolve()}")
        for index, item in enumerate(self.samples):
            for role in ("chosen", "rejected"):
                _, labels = self._tokenize_pair(item["prompt"], item[role])
                if not bool((labels != -100).any()):
                    raise ValueError(
                        f"DPO sample {index} {role} has no unmasked response target"
                    )

    def __len__(self):
        return len(self.samples)

    def _tokenize_pair(self, prompt, response):
        """Tokenize a prompt+response pair, returning input_ids and labels."""
        prompt_ids = self.tokenizer.encode(
            prompt.strip() + " ", add_special_tokens=False
        )
        response_ids = self.tokenizer.encode(
            response.strip(), add_special_tokens=False
        )
        prompt_ids = prompt_ids[: max(0, self.max_length - 1)]

        full_ids = (
            [self.tokenizer.bos_id()]
            + prompt_ids
            + response_ids
            + [self.tokenizer.eos_id()]
        )
        full_ids = full_ids[: self.max_length + 1]

        # Causal next-token alignment.  Response labels are retained while the
        # shifted targets that still belong to the prompt are ignored.
        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
        labels = full_ids[1:]
        labels[: len(prompt_ids)] = [-100] * len(prompt_ids)
        labels = torch.tensor(labels, dtype=torch.long)

        return input_ids, labels

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = item["prompt"]

        chosen_ids, chosen_labels = self._tokenize_pair(prompt, item["chosen"])
        rejected_ids, rejected_labels = self._tokenize_pair(prompt, item["rejected"])

        return chosen_ids, chosen_labels, rejected_ids, rejected_labels


def collate_dpo(batch):
    """Collate function that pads DPO batches to the same length."""
    chosen_ids, chosen_labels, rejected_ids, rejected_labels = zip(*batch)

    max_len = max(
        max(len(s) for s in chosen_ids),
        max(len(s) for s in rejected_ids),
    )

    def pad_sequence(sequences, pad_value=0):
        padded = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            if pad_value == -100:
                padded.append(torch.cat([seq, torch.full((pad_len,), -100, dtype=torch.long)]))
            else:
                padded.append(torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)]))
        return torch.stack(padded)

    return (
        pad_sequence(chosen_ids, 0),
        pad_sequence(chosen_labels, -100),
        pad_sequence(rejected_ids, 0),
        pad_sequence(rejected_labels, -100),
    )


@torch.no_grad()
def evaluate_loss(
    policy_model, ref_model, dataloader, device, beta, max_batches=None
):
    """Evaluate DPO loss on a dataloader. Returns average loss."""
    policy_model.eval()
    losses = []
    for i, (chosen_ids, chosen_labels, rejected_ids, rejected_labels) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        chosen_ids = chosen_ids.to(device)
        chosen_labels = chosen_labels.to(device)
        rejected_ids = rejected_ids.to(device)
        rejected_labels = rejected_labels.to(device)

        # Policy model log-probs
        policy_chosen_logps = compute_logprobs(policy_model, chosen_ids, chosen_labels)
        policy_rejected_logps = compute_logprobs(policy_model, rejected_ids, rejected_labels)

        # Reference model log-probs
        ref_chosen_logps = compute_logprobs(ref_model, chosen_ids, chosen_labels)
        ref_rejected_logps = compute_logprobs(ref_model, rejected_ids, rejected_labels)

        loss = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=beta,
        )
        losses.append(loss.item())
    policy_model.train()
    return sum(losses) / len(losses) if losses else float("inf")


def load_dpo_checkpoint(
    model: MiniLLM,
    path: str,
    device: str,
    model_config: MiniLLMConfig,
    role: str,
) -> dict:
    """Require and load a compatible DPO policy/reference initialization."""
    try:
        require_file(path, f"DPO {role} checkpoint")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"DPO {role} checkpoint is required and was not found: "
            f"{Path(path).expanduser().resolve()}. Train/download the intended "
            f"{role} checkpoint and update the configuration before starting DPO."
        ) from exc
    checkpoint = load_checkpoint(path, device)
    validate_checkpoint_model_config(checkpoint, model_config, f"DPO {role} checkpoint")
    model.load_state_dict(checkpoint_model_state(checkpoint))
    print(f"Loaded {role} model from {path}")
    return checkpoint


def train_dpo(config_path: str):
    """DPO training loop.

    1. Create policy model (from sft.pt, trainable)
    2. Create reference model (from sft.pt, frozen, no grad)
    3. For each batch: compute logprobs for chosen/rejected through both models
    4. Compute DPO loss, backprop through policy only
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
    require_file(cfg["data_path"], "DPO training dataset")
    require_file(cfg["eval_data_path"], "DPO validation dataset")

    # Tokenizer
    tokenizer = MiniLLMTokenizer(cfg["tokenizer_path"])

    # Create policy model (trainable)
    policy_model = MiniLLM(model_config).to(device)
    policy_init = cfg["policy_init"]
    load_dpo_checkpoint(policy_model, policy_init, device, model_config, "policy")

    # Create reference model (frozen)
    ref_model = MiniLLM(model_config).to(device)
    ref_init = cfg.get("ref_init", policy_init)
    load_dpo_checkpoint(ref_model, ref_init, device, model_config, "reference")

    # Freeze reference model
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print(f"Policy parameters (trainable): {policy_model.num_trainable_params():,}")
    print(f"Reference parameters (frozen): {ref_model.num_params():,}")

    # Datasets
    max_length = model_config.block_size
    beta = cfg.get("beta", 0.1)

    train_dataset = DPODataset(cfg["data_path"], tokenizer, max_length)
    eval_dataset = DPODataset(cfg["eval_data_path"], tokenizer, max_length)

    batch_size = cfg["batch_size"]
    grad_accum_steps = cfg["gradient_accumulation_steps"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_dpo,
        generator=make_dataloader_generator(seed),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_dpo,
    )

    # Optimizer and scheduler (only for policy model)
    max_steps = cfg["max_steps"]
    warmup_steps = cfg["warmup_steps"]
    lr = cfg["learning_rate"]
    min_lr = cfg["min_lr"]
    weight_decay = cfg["weight_decay"]

    optimizer = create_optimizer(policy_model, lr, weight_decay)
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
    save_every = cfg.get("save_every", 200)
    eval_every = cfg.get("eval_every", 100)
    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")
    checkpoint_name = cfg.get("checkpoint_name", "dpo.pt")

    os.makedirs(checkpoint_dir, exist_ok=True)
    write_run_manifest(
        checkpoint_dir,
        config_path=config_path,
        config=cfg,
        seed=seed,
        device=device,
        checkpoint_paths=[policy_init, ref_init],
    )

    step = 0
    loss_metric = AvgMetric()
    policy_model.train()

    pbar = tqdm(total=max_steps, desc="DPO Training")
    data_iter = iter(train_loader)

    while step < max_steps:
        try:
            chosen_ids, chosen_labels, rejected_ids, rejected_labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            chosen_ids, chosen_labels, rejected_ids, rejected_labels = next(data_iter)

        chosen_ids = chosen_ids.to(device)
        chosen_labels = chosen_labels.to(device)
        rejected_ids = rejected_ids.to(device)
        rejected_labels = rejected_labels.to(device)

        with autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            # Policy model log-probs (with gradients)
            policy_chosen_logps = compute_logprobs(
                policy_model, chosen_ids, chosen_labels
            )
            policy_rejected_logps = compute_logprobs(
                policy_model, rejected_ids, rejected_labels
            )

            # Reference model log-probs (no gradients)
            with torch.no_grad():
                ref_chosen_logps = compute_logprobs(
                    ref_model, chosen_ids, chosen_labels
                )
                ref_rejected_logps = compute_logprobs(
                    ref_model, rejected_ids, rejected_labels
                )

            loss = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=beta,
            )
            loss = loss / grad_accum_steps

        # Backward
        scaler.scale(loss).backward()

        loss_metric.update(loss.item() * grad_accum_steps)

        # Gradient accumulation
        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss_metric.avg:.4f}", "step": step})

        # Evaluation
        if step % eval_every == 0:
            eval_loss = evaluate_loss(
                policy_model, ref_model, eval_loader, device, beta, max_batches=50
            )
            current_lr = scheduler.get_last_lr()[0]
            pbar.write(
                f"Step {step} | Train Loss: {loss_metric.avg:.4f} | "
                f"Eval Loss: {eval_loss:.4f} | LR: {current_lr:.2e}"
            )
            loss_metric.reset()
            policy_model.train()

        # Save checkpoint
        if step % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"dpo_step_{step}.pt")
            save_checkpoint(
                policy_model, optimizer, scheduler, step, ckpt_path, model_config,
                scaler=scaler, training_config=cfg, checkpoint_type="dpo_training",
            )
            pbar.write(f"Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(checkpoint_dir, checkpoint_name)
    save_checkpoint(
        policy_model, optimizer, scheduler, step, final_path, model_config,
        scaler=scaler, training_config=cfg, checkpoint_type="dpo_training",
    )
    pbar.write(f"Final checkpoint saved: {final_path}")
    pbar.close()

    print("DPO training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training for MiniLLM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_instruction_dpo_v2.yaml",
        help="Path to DPO config YAML",
    )
    args = parser.parse_args()
    train_dpo(args.config)

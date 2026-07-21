"""Loss evaluation for MiniLLM."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tqdm import tqdm

from eval.metrics import compute_perplexity


@torch.no_grad()
def evaluate_loss(model, dataloader, device="cuda", max_batches=None):
    """Evaluate model loss on a dataset.

    Args:
        model: MiniLLM model instance.
        dataloader: DataLoader yielding batches with input_ids and targets.
        device: Device string (e.g. 'cuda' or 'cpu').
        max_batches: If set, only evaluate on this many batches.

    Returns:
        Dictionary with:
            - 'loss': average cross-entropy loss
            - 'perplexity': exp(loss)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        output = model(input_ids, targets=targets)
        loss = output["loss"].item()

        total_loss += loss
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    ppl = compute_perplexity(avg_loss)

    return {"loss": avg_loss, "perplexity": ppl}

"""Module for engine."""

from typing import Dict, List, Optional
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def _collect_logits_targets(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    progress_prefix: Optional[str] = None,
    progress_interval: int = 10,
    grad_accum_steps: int = 1,
) -> Dict:
    """
    Collects logits and targets from the model on the given data loader, optionally performing training if optimizer is provided.

    Args:
        model: The model to evaluate/train.
        loader: DataLoader for the data to evaluate/train on.
        device: Device to run the computations on.
        criterion: Loss function to compute the loss.
        optimizer: Optional optimizer for training. If None, the model is only evaluated.
        progress_prefix: Optional prefix for progress printouts during training.
        progress_interval: Interval (in batches) for printing progress during training.
        grad_accum_steps: Number of batches to accumulate gradients before stepping the optimizer during training.

    Returns:
        A dictionary containing metrics such as loss, accuracy, etc., as well as collected targets.
    """

    # Setup
    is_train = optimizer is not None
    progress_interval = max(1, int(progress_interval))
    grad_accum_steps = max(1, int(grad_accum_steps))
    model.train(is_train)

    # Initialization
    losses: List[float] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []
    all_preds: List[int] = []

    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        total_batches = len(loader)
        if is_train:
            optimizer.zero_grad()

        # Loop over batches
        for batch_idx, batch in enumerate(loader, start=1):

            # Move data to device
            move_start = time.perf_counter()
            patch_batches = [item["embeddings"].to(device) for item in batch]
            coord_batches = [item["coords"].to(device) for item in batch]
            targets = torch.tensor(
                [item["label"] for item in batch], device=device, dtype=torch.long
            )
            move_elapsed = time.perf_counter() - move_start

            # Forward pass
            forward_start = time.perf_counter()
            model_output = model(patch_batches, coord_batches)
            logits = (
                model_output[0] if isinstance(model_output, tuple) else model_output
            )
            cls_loss = criterion(logits, targets)
            loss = cls_loss
            forward_elapsed = time.perf_counter() - forward_start

            backward_elapsed = 0.0
            # Backward pass and optimization step (if training)
            if is_train:
                backward_start = time.perf_counter()
                # Scale loss by grad_accum_steps for gradient accumulation
                (loss / grad_accum_steps).backward()
                # Step optimizer every grad_accum_steps batches or on the last batch
                if batch_idx % grad_accum_steps == 0 or batch_idx == total_batches:
                    optimizer.step()
                    optimizer.zero_grad()
                backward_elapsed = time.perf_counter() - backward_start

            post_start = time.perf_counter()
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)

            losses.append(float(loss.item()))
            all_targets.extend(targets.detach().cpu().tolist())
            all_probs.extend(list(probs))
            all_preds.extend(preds.tolist())
            post_elapsed = time.perf_counter() - post_start

            if (
                is_train
                and progress_prefix
                and (
                    batch_idx == 1
                    or batch_idx % progress_interval == 0
                    or batch_idx == total_batches
                )
            ):
                running_loss = float(np.mean(losses)) if losses else float("nan")
                print(
                    f"[{progress_prefix}] batch {batch_idx}/{total_batches} "
                    f"running_loss={running_loss:.4f} "
                    f"move_time={move_elapsed:.2f}s "
                    f"forward_time={forward_elapsed:.2f}s "
                    f"backward_time={backward_elapsed:.2f}s "
                    f"post_time={post_elapsed:.2f}s"
                )

    # Compute metrics
    all_targets_np = np.array(all_targets)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.vstack(all_probs)

    acc = float(accuracy_score(all_targets_np, all_preds_np))
    balanced_acc = float(balanced_accuracy_score(all_targets_np, all_preds_np))
    macro_f1 = float(
        f1_score(all_targets_np, all_preds_np, average="macro", zero_division=0)
    )
    f1 = float(
        f1_score(all_targets_np, all_preds_np, average="weighted", zero_division=0)
    )

    num_classes = all_probs_np.shape[1]
    per_class_f1 = f1_score(
        all_targets_np,
        all_preds_np,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    conf_mat = confusion_matrix(
        all_targets_np,
        all_preds_np,
        labels=list(range(num_classes)),
    )

    try:
        if num_classes == 2:
            auc = float(roc_auc_score(all_targets_np, all_probs_np[:, 1]))
        else:
            auc = float(
                roc_auc_score(
                    all_targets_np,
                    all_probs_np,
                    multi_class="ovr",
                    average="weighted",
                )
            )
    except Exception:
        auc = float("nan")

    result = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "f1": f1,
        "auc": auc,
        "targets": all_targets_np,
        "preds": all_preds_np,
        "probs": all_probs_np,
        "per_class_f1": per_class_f1,
        "confusion_matrix": conf_mat,
    }
    return result


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    progress_prefix: Optional[str] = None,
    progress_interval: int = 10,
    grad_accum_steps: int = 1,
) -> Dict:
    """
    Trains the model for one epoch and returns metrics.

    Args:
        model: The model to train.
        loader: DataLoader for the training data.
        optimizer: Optimizer for updating model parameters.
        criterion: Loss function.
        device: Device to run the training on.
        progress_prefix: Optional prefix for progress printouts.
        progress_interval: Interval (in batches) for printing progress.
        grad_accum_steps: Number of batches to accumulate gradients before stepping the optimizer.
    
    Returns:
        A dictionary containing training metrics such as loss, accuracy, etc.
    """
    return _collect_logits_targets(
        model,
        loader,
        device,
        criterion,
        optimizer=optimizer,
        progress_prefix=progress_prefix,
        progress_interval=progress_interval,
        grad_accum_steps=grad_accum_steps,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    """
    Evaluates the model on the given data loader and returns metrics.

    Args:
        model: The model to evaluate.
        loader: DataLoader for the evaluation data.
        criterion: Loss function.
        device: Device to run the evaluation on.
    
    Returns:
        A dictionary containing evaluation metrics such as loss, accuracy, etc.
    """
    return _collect_logits_targets(model, loader, device, criterion, optimizer=None)

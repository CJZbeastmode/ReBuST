"""Training/evaluation engine for Mamba MIL."""

from typing import Dict, List, Optional
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


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
    is_train = optimizer is not None
    progress_interval = max(1, int(progress_interval))
    grad_accum_steps = max(1, int(grad_accum_steps))
    model.train(is_train)

    losses: List[float] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []
    all_preds: List[int] = []

    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        total_batches = len(loader)
        if is_train and progress_prefix:
            print(f"[{progress_prefix}] total_batches={total_batches}", flush=True)
        if is_train:
            optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader, start=1):
            move_start = time.perf_counter()
            patch_batches = [item["embeddings"].to(device) for item in batch]
            coord_batches = [item["coords"].to(device) for item in batch]
            targets = torch.tensor(
                [item["label"] for item in batch],
                dtype=torch.long,
                device=device,
            )
            move_elapsed = time.perf_counter() - move_start

            forward_start = time.perf_counter()
            model_output = model(patch_batches, coord_batches)
            logits = model_output[0] if isinstance(model_output, tuple) else model_output
            loss = criterion(logits, targets)
            forward_elapsed = time.perf_counter() - forward_start

            backward_elapsed = 0.0
            if is_train:
                backward_start = time.perf_counter()
                (loss / grad_accum_steps).backward()
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

    all_targets_np = np.array(all_targets)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.vstack(all_probs)

    acc = float(accuracy_score(all_targets_np, all_preds_np))
    balanced_acc = float(balanced_accuracy_score(all_targets_np, all_preds_np))
    macro_f1 = float(f1_score(all_targets_np, all_preds_np, average="macro", zero_division=0))
    f1 = float(f1_score(all_targets_np, all_preds_np, average="weighted", zero_division=0))

    num_classes = int(all_probs_np.shape[1])
    per_class_f1 = f1_score(
        all_targets_np,
        all_preds_np,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    conf_mat = confusion_matrix(all_targets_np, all_preds_np, labels=list(range(num_classes)))

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

    return {
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
    return _collect_logits_targets(model, loader, device, criterion, optimizer=None)

"""SASHA training pipeline.

Two stages:
1) Train HAFED-like MIL classifier on per-WSI embeddings.
2) Freeze classifier and train SASHA-style RL patch selector.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, WeightedRandomSampler

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.ablation_patch_selector.SASHA.data import (
    PTEmbeddingDirDataset,
    PTEmbeddingSample,
    build_items_and_label_map,
    build_items_with_label_map,
    collate_samples,
)
from src.ablation_patch_selector.SASHA.models import HAFEDClassifier, SashaPolicyValue


@dataclass
class EpisodeRollout:
    log_probs: List[torch.Tensor]
    values: List[torch.Tensor]
    rewards: List[float]
    entropies: List[torch.Tensor]
    selected_indices: List[int]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_candidate_pool(
    embeddings: torch.Tensor,
    hafed: HAFEDClassifier,
    max_candidates: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return candidate embeddings/coords indices chosen by HAFED attention."""
    n = int(embeddings.shape[0])
    if n <= 0:
        return embeddings, torch.zeros(0, dtype=torch.long), torch.zeros(0)

    with torch.no_grad():
        toks = embeddings.unsqueeze(0).to(device)
        mask = torch.ones(1, n, dtype=torch.bool, device=device)
        _, attn, _ = hafed(toks, mask)
        attn = attn.squeeze(0).detach().cpu()

    k = min(int(max_candidates), n)
    if k <= 0:
        k = n

    topv, topi = torch.topk(attn, k=k, largest=True)
    topi, order = torch.sort(topi)
    topv = topv[order]
    return embeddings[topi], topi, topv


def classify_subset_probability(
    hafed: HAFEDClassifier,
    subset_embeddings: torch.Tensor,
    true_label_idx: int,
    device: torch.device,
) -> float:
    if subset_embeddings.numel() == 0:
        return 0.0
    with torch.no_grad():
        toks = subset_embeddings.unsqueeze(0).to(device)
        mask = torch.ones(1, toks.shape[1], dtype=torch.bool, device=device)
        logits, _, _ = hafed(toks, mask)
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, int(true_label_idx)].item())


def run_selector_episode(
    sample: PTEmbeddingSample,
    candidate_embeddings: torch.Tensor,
    policy: SashaPolicyValue,
    hafed_frozen: HAFEDClassifier,
    device: torch.device,
    max_steps: int,
    gamma: float,
    deterministic: bool,
) -> Tuple[EpisodeRollout, float, int]:
    selected_mask = torch.zeros(candidate_embeddings.shape[0], dtype=torch.bool, device=device)
    chosen: List[int] = []

    log_probs: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    rewards: List[float] = []
    entropies: List[torch.Tensor] = []

    prev_prob = 0.0
    global_context = candidate_embeddings.mean(dim=0, keepdim=True).to(device)

    for step in range(max_steps):
        step_frac = torch.tensor([[float(step) / max(1, max_steps)]], dtype=torch.float32, device=device)

        logits, value = policy(
            candidate_embeddings.unsqueeze(0).to(device),
            selected_mask.unsqueeze(0),
            global_context,
            step_frac,
        )
        logits = logits.squeeze(0)

        # mask already selected token actions
        logits = logits.clone()
        logits[:-1] = logits[:-1].masked_fill(selected_mask, torch.finfo(logits.dtype).min)

        dist = Categorical(logits=logits)
        if deterministic:
            action = int(torch.argmax(logits).item())
            log_prob = torch.log_softmax(logits, dim=0)[action]
            entropy = -(torch.softmax(logits, dim=0) * torch.log_softmax(logits, dim=0)).sum()
        else:
            sampled = dist.sample()
            action = int(sampled.item())
            log_prob = dist.log_prob(sampled)
            entropy = dist.entropy()

        stop_action = candidate_embeddings.shape[0]
        reward = 0.0

        if action == stop_action:
            reward = prev_prob
            log_probs.append(log_prob)
            values.append(value.squeeze(0))
            rewards.append(float(reward))
            entropies.append(entropy)
            break

        if bool(selected_mask[action].item()):
            reward = -0.05
        else:
            new_mask = selected_mask.clone()
            new_mask[action] = True
            selected_mask = new_mask
            chosen.append(action)

            subset = candidate_embeddings[selected_mask.detach().cpu()].detach()
            curr_prob = classify_subset_probability(
                hafed=hafed_frozen,
                subset_embeddings=subset,
                true_label_idx=sample.label_idx,
                device=device,
            )
            reward = curr_prob - prev_prob
            prev_prob = curr_prob

        log_probs.append(log_prob)
        values.append(value.squeeze(0))
        rewards.append(float(reward))
        entropies.append(entropy)

    # discounted return from final rewards
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + float(gamma) * running
        returns.append(running)
    returns.reverse()

    # overwrite rewards with returns for optimization convenience
    rollout = EpisodeRollout(
        log_probs=log_probs,
        values=values,
        rewards=returns,
        entropies=entropies,
        selected_indices=chosen,
    )
    final_prob = prev_prob
    return rollout, final_prob, len(chosen)


def evaluate_hafed(
    model: HAFEDClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            for sample in batch:
                emb = sample.embeddings.to(device)
                mask = torch.ones(1, emb.shape[0], dtype=torch.bool, device=device)
                logits, _, _ = model(emb.unsqueeze(0), mask)
                target = torch.tensor([sample.label_idx], dtype=torch.long, device=device)
                loss = F.cross_entropy(logits, target)

                total += 1
                total_loss += float(loss.item())
                pred = int(torch.argmax(logits, dim=-1).item())
                correct += int(pred == sample.label_idx)

    if total == 0:
        return {"loss": float("inf"), "acc": 0.0}

    return {"loss": total_loss / total, "acc": correct / total}


def evaluate_selector(
    hafed: HAFEDClassifier,
    policy: SashaPolicyValue,
    loader: DataLoader,
    device: torch.device,
    max_candidates: int,
    max_steps: int,
) -> Dict[str, float]:
    policy.eval()
    hafed.eval()

    total = 0
    correct = 0
    mean_prob = 0.0
    mean_kept_ratio = 0.0

    for batch in loader:
        for sample in batch:
            emb = sample.embeddings.detach().cpu()
            candidates, _, _ = pick_candidate_pool(
                embeddings=emb,
                hafed=hafed,
                max_candidates=max_candidates,
                device=device,
            )
            if candidates.shape[0] == 0:
                continue

            rollout, final_prob, kept = run_selector_episode(
                sample=sample,
                candidate_embeddings=candidates,
                policy=policy,
                hafed_frozen=hafed,
                device=device,
                max_steps=max_steps,
                gamma=1.0,
                deterministic=True,
            )

            if kept <= 0:
                kept = 1
                top_idx = int(torch.argmax(candidates.norm(dim=-1)).item())
                rollout.selected_indices = [top_idx]

            subset = candidates[torch.tensor(rollout.selected_indices, dtype=torch.long)]
            with torch.no_grad():
                mask = torch.ones(1, subset.shape[0], dtype=torch.bool, device=device)
                logits, _, _ = hafed(subset.unsqueeze(0).to(device), mask)
                pred = int(torch.argmax(logits, dim=-1).item())

            total += 1
            correct += int(pred == sample.label_idx)
            mean_prob += float(final_prob)
            mean_kept_ratio += float(kept / max(1, emb.shape[0]))

    if total == 0:
        return {"acc": 0.0, "mean_prob": 0.0, "mean_kept_ratio": 0.0}

    return {
        "acc": correct / total,
        "mean_prob": mean_prob / total,
        "mean_kept_ratio": mean_kept_ratio / total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SASHA selector pipeline")

    parser.add_argument("--train-embeddings-dir", default="/Volumes/Xbox_HD/Data/med_img/train")
    parser.add_argument("--val-embeddings-dir", default="/Volumes/Xbox_HD/Data/med_img/val")
    parser.add_argument("--out-dir", default="data/models/ablation_patch_selector/sasha")
    parser.add_argument(
        "--input-format",
        type=str,
        default="auto",
        choices=["pt", "svs", "auto"],
        help="Input file type expected in train/val directories.",
    )
    parser.add_argument(
        "--svs-level-mode",
        type=str,
        default="root_only",
        choices=["root_only", "finest_only"],
        help="WSI level used when input-format includes .svs.",
    )
    parser.add_argument(
        "--svs-embed-backend",
        type=str,
        default="plip",
        choices=["plip", "conch"],
        help="Embedding backend for direct .svs loading.",
    )

    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)

    parser.add_argument("--max-patches-per-wsi", type=int, default=0)
    parser.add_argument("--patch-sample-mode", type=str, default="uniform", choices=["uniform", "head", "random"])

    parser.add_argument("--hafed-epochs", type=int, default=10)
    parser.add_argument("--hafed-lr", type=float, default=1e-4)

    parser.add_argument("--selector-epochs", type=int, default=8)
    parser.add_argument("--selector-lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=24)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--logit-adjust-tau", type=float, default=1.0,
                        help="Logit adjustment tau (0 = disabled)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_sasha(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_items, label_map = build_items_and_label_map(
        args.train_embeddings_dir,
        input_format=args.input_format,
    )
    val_items = build_items_with_label_map(
        args.val_embeddings_dir,
        label_map=label_map,
        strict=False,
        input_format=args.input_format,
    )

    train_ds = PTEmbeddingDirDataset(
        items=train_items,
        max_patches_per_wsi=args.max_patches_per_wsi,
        sample_mode=args.patch_sample_mode,
        sample_seed=args.seed,
        svs_level_mode=args.svs_level_mode,
        svs_embed_backend=args.svs_embed_backend,
    )
    val_ds = PTEmbeddingDirDataset(
        items=val_items,
        max_patches_per_wsi=args.max_patches_per_wsi,
        sample_mode="head",
        sample_seed=args.seed,
        svs_level_mode=args.svs_level_mode,
        svs_embed_backend=args.svs_embed_backend,
    )

    train_labels = [int(item[2]) for item in train_items]
    class_counts = np.bincount(train_labels, minlength=len(label_map)).astype(np.float32)
    class_counts[class_counts == 0.0] = 1.0
    class_weights_arr = (len(train_labels) / (len(label_map) * class_counts)).astype(np.float32)
    sample_weights = torch.tensor([class_weights_arr[int(item[2])] for item in train_items], dtype=torch.float32)
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, collate_fn=collate_samples)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_samples)

    print("[SASHA] ================================================")
    print(f"[SASHA] device={device} seed={args.seed} out_dir={args.out_dir}")
    print(
        f"[SASHA] input_format={args.input_format} "
        f"svs_level_mode={args.svs_level_mode} svs_embed_backend={args.svs_embed_backend}"
    )
    print(
        f"[SASHA] train_items={len(train_ds)} val_items={len(val_ds)} "
        f"num_classes={len(label_map)}"
    )
    print(f"[SASHA] label_map={label_map}")
    print("[SASHA] ================================================")

    # ---------------- Stage 1: HAFED ----------------
    print("[SASHA][HAFED] Stage-1 training start")
    hafed = HAFEDClassifier(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(label_map),
        num_heads=args.num_heads,
    ).to(device)
    hafed.set_logit_adjustment(torch.tensor(class_counts, device=device), tau=args.logit_adjust_tau)
    print(f"[SASHA] logit_adjust_tau={args.logit_adjust_tau} class_counts={class_counts.tolist()}")

    hafed_opt = torch.optim.AdamW(hafed.parameters(), lr=args.hafed_lr)

    best_val_acc = -1.0
    best_hafed_path = os.path.join(args.out_dir, "best_hafed.pt")

    for epoch in range(1, args.hafed_epochs + 1):
        epoch_start = time.perf_counter()
        hafed.train()
        running_loss = 0.0
        seen = 0
        total_samples = max(1, len(train_ds))
        log_interval = max(1, total_samples // 5)

        print(
            f"[SASHA][HAFED][E{epoch:03d}] ENTER "
            f"samples={total_samples} lr={args.hafed_lr:.2e}"
        )

        for batch in train_loader:
            for sample in batch:
                emb = sample.embeddings.to(device)
                mask = torch.ones(1, emb.shape[0], dtype=torch.bool, device=device)
                target = torch.tensor([sample.label_idx], dtype=torch.long, device=device)

                logits, _, _ = hafed(emb.unsqueeze(0), mask)
                loss = F.cross_entropy(logits, target)

                hafed_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(hafed.parameters(), 1.0)
                hafed_opt.step()

                running_loss += float(loss.item())
                seen += 1

                if seen == 1 or seen % log_interval == 0 or seen == total_samples:
                    avg_loss = running_loss / max(1, seen)
                    print(
                        f"[SASHA][HAFED][E{epoch:03d}] progress "
                        f"{seen}/{total_samples} avg_loss={avg_loss:.4f}"
                    )

        val_metrics = evaluate_hafed(hafed, val_loader, device=device)
        train_loss = running_loss / max(1, seen)
        improved = val_metrics["acc"] > best_val_acc
        epoch_secs = time.perf_counter() - epoch_start
        print(
            f"[HAFED] epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        )
        print(
            f"[SASHA][HAFED][E{epoch:03d}] EXIT "
            f"improved={improved} best_val_acc={max(best_val_acc, val_metrics['acc']):.4f} "
            f"time={epoch_secs:.2f}s"
        )

        if improved:
            best_val_acc = val_metrics["acc"]
            torch.save(
                {
                    "model_state_dict": hafed.state_dict(),
                    "label_map": label_map,
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_heads": args.num_heads,
                    "num_classes": len(label_map),
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                best_hafed_path,
            )
            print(f"[HAFED] saved best checkpoint: {best_hafed_path}")

    ckpt = torch.load(best_hafed_path, map_location=device)
    hafed.load_state_dict(ckpt["model_state_dict"])
    hafed.eval()
    for p in hafed.parameters():
        p.requires_grad = False
    print(f"[SASHA][HAFED] Stage-1 training end best_val_acc={best_val_acc:.4f}")

    # ---------------- Stage 2: RL selector ----------------
    print("[SASHA][SELECTOR] Stage-2 training start")
    policy = SashaPolicyValue(embed_dim=args.embed_dim, hidden_dim=args.hidden_dim).to(device)
    pol_opt = torch.optim.AdamW(policy.parameters(), lr=args.selector_lr)

    best_selector_score = -math.inf
    best_selector_path = os.path.join(args.out_dir, "best_selector.pt")

    for epoch in range(1, args.selector_epochs + 1):
        epoch_start = time.perf_counter()
        policy.train()
        total_actor = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_prob = 0.0
        total_kept = 0.0
        episodes = 0
        samples_seen = 0
        total_samples = max(1, len(train_ds))
        log_interval = max(1, total_samples // 5)

        print(
            f"[SASHA][SELECTOR][E{epoch:03d}] ENTER "
            f"samples={total_samples} lr={args.selector_lr:.2e}"
        )

        for batch in train_loader:
            for sample in batch:
                samples_seen += 1
                candidates, _, _ = pick_candidate_pool(
                    embeddings=sample.embeddings.detach().cpu(),
                    hafed=hafed,
                    max_candidates=args.max_candidates,
                    device=device,
                )

                if candidates.shape[0] <= 0:
                    continue

                rollout, final_prob, kept_count = run_selector_episode(
                    sample=sample,
                    candidate_embeddings=candidates,
                    policy=policy,
                    hafed_frozen=hafed,
                    device=device,
                    max_steps=args.max_steps,
                    gamma=args.gamma,
                    deterministic=False,
                )

                if not rollout.log_probs:
                    continue

                returns = torch.tensor(rollout.rewards, dtype=torch.float32, device=device)
                values = torch.stack(rollout.values)
                log_probs = torch.stack(rollout.log_probs)
                entropies = torch.stack(rollout.entropies)

                advantages = returns - values.detach()
                actor_loss = -(log_probs * advantages).mean()
                value_loss = F.mse_loss(values, returns)
                entropy_term = entropies.mean()

                loss = actor_loss + args.value_coef * value_loss - args.entropy_coef * entropy_term

                pol_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                pol_opt.step()

                episodes += 1
                total_actor += float(actor_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(entropy_term.item())
                total_prob += float(final_prob)
                total_kept += float(kept_count / max(1, candidates.shape[0]))

                if (
                    samples_seen == 1
                    or samples_seen % log_interval == 0
                    or samples_seen == total_samples
                ):
                    mean_actor = total_actor / max(1, episodes)
                    mean_value = total_value / max(1, episodes)
                    mean_entropy = total_entropy / max(1, episodes)
                    print(
                        f"[SASHA][SELECTOR][E{epoch:03d}] progress "
                        f"{samples_seen}/{total_samples} episodes={episodes} "
                        f"actor={mean_actor:.4f} value={mean_value:.4f} entropy={mean_entropy:.4f}"
                    )

        train_actor = total_actor / max(1, episodes)
        train_value = total_value / max(1, episodes)
        train_entropy = total_entropy / max(1, episodes)
        train_prob = total_prob / max(1, episodes)
        train_kept = total_kept / max(1, episodes)

        val_sel = evaluate_selector(
            hafed=hafed,
            policy=policy,
            loader=val_loader,
            device=device,
            max_candidates=args.max_candidates,
            max_steps=args.max_steps,
        )
        score = val_sel["acc"] - 0.1 * val_sel["mean_kept_ratio"]
        improved = score > best_selector_score
        epoch_secs = time.perf_counter() - epoch_start

        print(
            f"[SELECTOR] epoch={epoch} actor={train_actor:.4f} value={train_value:.4f} "
            f"entropy={train_entropy:.4f} train_prob={train_prob:.4f} train_kept={train_kept:.4f} "
            f"val_acc={val_sel['acc']:.4f} val_prob={val_sel['mean_prob']:.4f} "
            f"val_kept={val_sel['mean_kept_ratio']:.4f} score={score:.4f}"
        )
        print(
            f"[SASHA][SELECTOR][E{epoch:03d}] EXIT "
            f"improved={improved} best_score={max(best_selector_score, score):.4f} "
            f"time={epoch_secs:.2f}s"
        )

        if improved:
            best_selector_score = score
            torch.save(
                {
                    "model_state_dict": policy.state_dict(),
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "max_candidates": args.max_candidates,
                    "max_steps": args.max_steps,
                    "best_score": best_selector_score,
                    "epoch": epoch,
                },
                best_selector_path,
            )
            print(f"[SELECTOR] saved best checkpoint: {best_selector_path}")

    print(f"[SASHA][SELECTOR] Stage-2 training end best_score={best_selector_score:.4f}")

    config_path = os.path.join(args.out_dir, "sasha_train_config.json")
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "train_embeddings_dir": args.train_embeddings_dir,
                "val_embeddings_dir": args.val_embeddings_dir,
                "input_format": args.input_format,
                "svs_level_mode": args.svs_level_mode,
                "svs_embed_backend": args.svs_embed_backend,
                "label_map": label_map,
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "max_candidates": args.max_candidates,
                "max_steps": args.max_steps,
                "hafed_epochs": args.hafed_epochs,
                "selector_epochs": args.selector_epochs,
                "seed": args.seed,
            },
            fh,
            indent=2,
        )

    print(f"[DONE] best_hafed={best_hafed_path}")
    print(f"[DONE] best_selector={best_selector_path}")
    print(f"[DONE] config={config_path}")

    return {
        "best_hafed_path": best_hafed_path,
        "best_selector_path": best_selector_path,
        "config_path": config_path,
        "label_map": label_map,
        "best_hafed_val_acc": best_val_acc,
        "best_selector_score": best_selector_score,
    }


def main() -> None:
    args = parse_args()
    train_sasha(args)


if __name__ == "__main__":
    main()

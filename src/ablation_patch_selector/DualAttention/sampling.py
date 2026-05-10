"""Sampling utilities for Raza dual attention."""

from __future__ import annotations

from typing import Tuple

import torch


def _has_valid_coords(coords: torch.Tensor) -> bool:
    if coords is None or coords.numel() == 0:
        return False
    if coords.dim() != 2 or coords.shape[1] < 2:
        return False
    return bool(torch.any(coords.abs() > 0))


def _normalize_coords(coords: torch.Tensor) -> torch.Tensor:
    if coords.numel() == 0:
        return coords
    mean = coords.mean(dim=0, keepdim=True)
    std = coords.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (coords - mean) / std


def sample_attention_candidates(
    embeddings: torch.Tensor,
    coords: torch.Tensor,
    attn: torch.Tensor,
    num_tiles: int,
    pool_multiplier: int = 2,
    noise_low: float = 0.0,
    noise_high: float = 0.1,
    min_dist: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: selected_embeddings, selected_coords, selected_indices
    """
    n = int(embeddings.shape[0])
    if n <= 0:
        return embeddings, coords, torch.zeros(0, dtype=torch.long)

    num_tiles = max(1, int(num_tiles))
    pool_k = min(n, max(num_tiles, int(pool_multiplier) * num_tiles))

    attn = attn.detach().float()
    attn_min = float(attn.min().item())
    attn_max = float(attn.max().item())
    denom = max(1e-6, attn_max - attn_min)
    attn_norm = (attn - attn_min) / denom

    noise = torch.empty_like(attn_norm).uniform_(float(noise_low), float(noise_high))
    scores = attn_norm + noise

    top_scores, top_idx = torch.topk(scores, k=pool_k, largest=True)
    order = torch.argsort(top_scores, descending=True)
    top_idx = top_idx[order]

    if min_dist <= 0 or not _has_valid_coords(coords):
        top_idx = top_idx[:num_tiles]
        return embeddings[top_idx], coords[top_idx], top_idx

    coords_xy = coords[:, -2:].float()
    coords_xy = _normalize_coords(coords_xy)

    picked = []
    for idx in top_idx.tolist():
        if len(picked) >= num_tiles:
            break
        if not picked:
            picked.append(idx)
            continue
        candidate = coords_xy[idx]
        keep = True
        for chosen in picked:
            dist = torch.dist(candidate, coords_xy[chosen])
            if dist < float(min_dist):
                keep = False
                break
        if keep:
            picked.append(idx)

    if len(picked) < num_tiles:
        remaining = [i for i in top_idx.tolist() if i not in picked]
        picked.extend(remaining[: max(0, num_tiles - len(picked))])

    picked_idx = torch.tensor(picked[:num_tiles], dtype=torch.long)
    return embeddings[picked_idx], coords[picked_idx], picked_idx


def pairwise_distance_penalty(coords: torch.Tensor, min_dist: float) -> torch.Tensor:
    if coords.numel() == 0 or coords.shape[0] < 2 or min_dist <= 0:
        return torch.tensor(0.0, device=coords.device)

    coords_xy = coords[:, -2:].float()
    coords_xy = _normalize_coords(coords_xy)
    total = 0.0
    count = 0
    for i in range(coords_xy.shape[0]):
        for j in range(i + 1, coords_xy.shape[0]):
            dist = torch.dist(coords_xy[i], coords_xy[j])
            if dist < float(min_dist):
                total += (float(min_dist) - dist) / float(min_dist)
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=coords.device)
    return torch.tensor(total / count, device=coords.device)

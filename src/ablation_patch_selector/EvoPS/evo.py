"""Evolutionary patch selection utilities."""

from __future__ import annotations

import random
from typing import Iterable, List, Optional, Tuple

import torch

from src.ablation_patch_selector.SASHA.models import HAFEDClassifier


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _unique_sorted(indices: Iterable[int]) -> List[int]:
    return sorted(set(int(x) for x in indices))


def _clamp_budget(budget: int, n: int) -> int:
    if n <= 0:
        return 0
    budget = int(budget)
    if budget <= 0:
        return min(1, n)
    return min(budget, n)


def _fitness(
    hafed: HAFEDClassifier,
    candidate_embeddings: torch.Tensor,
    indices: List[int],
    target_idx: int,
    device: torch.device,
) -> float:
    if not indices:
        return 0.0
    subset = candidate_embeddings[torch.tensor(indices, dtype=torch.long)]
    with torch.no_grad():
        mask = torch.ones(1, subset.shape[0], dtype=torch.bool, device=device)
        logits, _, _ = hafed(subset.unsqueeze(0).to(device), mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        return float(probs[int(target_idx)].item())


def _predict_label(
    hafed: HAFEDClassifier,
    embeddings: torch.Tensor,
    device: torch.device,
) -> Tuple[int, torch.Tensor]:
    with torch.no_grad():
        mask = torch.ones(1, embeddings.shape[0], dtype=torch.bool, device=device)
        logits, _, _ = hafed(embeddings.unsqueeze(0).to(device), mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu()
    return int(torch.argmax(probs).item()), probs


def _seed_population(
    n: int,
    budget: int,
    population_size: int,
    scores: Optional[torch.Tensor],
    seed: int,
) -> List[List[int]]:
    _set_seed(seed)
    population: List[List[int]] = []

    if n <= 0:
        return population

    if scores is not None and scores.numel() == n:
        topk = torch.topk(scores, k=budget, largest=True).indices.tolist()
        population.append(_unique_sorted(topk))

    while len(population) < population_size:
        idx = torch.randperm(n)[:budget].tolist()
        population.append(_unique_sorted(idx))

    return population


def _tournament(population: List[List[int]], scores: List[float]) -> List[int]:
    if not population:
        return []
    k = min(3, len(population))
    candidates = random.sample(range(len(population)), k=k)
    best = max(candidates, key=lambda i: scores[i])
    return list(population[best])


def _crossover(parent_a: List[int], parent_b: List[int], budget: int) -> List[int]:
    pool = _unique_sorted(list(parent_a) + list(parent_b))
    if len(pool) <= budget:
        return pool
    picked = random.sample(pool, k=budget)
    return _unique_sorted(picked)


def _mutate(child: List[int], n: int, mutation_rate: float, budget: int) -> List[int]:
    if not child or n <= 0:
        return child
    if random.random() > mutation_rate:
        return child

    child_set = set(child)
    if len(child_set) >= n:
        return child

    replace_idx = random.randrange(len(child))
    available = [i for i in range(n) if i not in child_set]
    if not available:
        return child

    child[replace_idx] = random.choice(available)
    return _unique_sorted(child)[:budget]


def evo_select_subset(
    candidate_embeddings: torch.Tensor,
    hafed: HAFEDClassifier,
    device: torch.device,
    budget: int,
    population_size: int,
    generations: int,
    elite_fraction: float,
    mutation_rate: float,
    crossover_rate: float,
    seed: int,
    scores: Optional[torch.Tensor] = None,
    target_idx: Optional[int] = None,
) -> Tuple[List[int], int, float]:
    """Run EvoPS selection.

    Returns (selected_indices, predicted_label_idx, best_score).
    """
    if candidate_embeddings.dim() == 1:
        candidate_embeddings = candidate_embeddings.unsqueeze(0)
    n = int(candidate_embeddings.shape[0])
    budget = _clamp_budget(budget, n)

    if n <= 0:
        return [], 0, 0.0

    if budget >= n:
        pred_idx, _ = _predict_label(hafed, candidate_embeddings, device=device)
        best_score = _fitness(hafed, candidate_embeddings, list(range(n)), pred_idx, device)
        return list(range(n)), pred_idx, best_score

    if scores is None:
        scores = candidate_embeddings.norm(dim=-1).detach().cpu()

    if target_idx is None:
        pred_idx, _ = _predict_label(hafed, candidate_embeddings, device=device)
        target_idx = pred_idx
    else:
        pred_idx = int(target_idx)

    population = _seed_population(n, budget, population_size, scores, seed=seed)
    if not population:
        return [], pred_idx, 0.0

    elite_count = max(1, int(population_size * elite_fraction))

    best_indices: List[int] = []
    best_score = -1.0

    for _ in range(max(1, int(generations))):
        fitness_scores = [
            _fitness(hafed, candidate_embeddings, member, target_idx, device)
            for member in population
        ]
        ranked = sorted(range(len(population)), key=lambda i: fitness_scores[i], reverse=True)
        elites = [population[i] for i in ranked[:elite_count]]

        if fitness_scores[ranked[0]] > best_score:
            best_score = float(fitness_scores[ranked[0]])
            best_indices = list(population[ranked[0]])

        next_population = elites.copy()
        while len(next_population) < population_size:
            parent_a = _tournament(population, fitness_scores)
            parent_b = _tournament(population, fitness_scores)

            if random.random() < crossover_rate:
                child = _crossover(parent_a, parent_b, budget)
            else:
                child = list(parent_a)

            child = _mutate(child, n=n, mutation_rate=mutation_rate, budget=budget)
            if len(child) < budget:
                missing = [i for i in range(n) if i not in set(child)]
                if missing:
                    child.extend(random.sample(missing, k=min(len(missing), budget - len(child))))
                    child = _unique_sorted(child)
            next_population.append(child)

        population = next_population

    if not best_indices:
        best_indices = population[0]
        best_score = _fitness(hafed, candidate_embeddings, best_indices, target_idx, device)

    return _unique_sorted(best_indices)[:budget], pred_idx, float(best_score)

"""Data utilities for aggregation-transformer training/inference."""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WSIItem:
    case_id: str
    label: int


class WSIEmbeddingDataset(Dataset):
    def __init__(
        self,
        items: Sequence[WSIItem],
        embeddings_dir: str,
        max_patches_per_wsi: int = 0,
        patch_sample_mode: str = "uniform",
        sample_seed: int = 42,
    ):
        self.items = list(items)
        self.embeddings_dir = embeddings_dir
        self.max_patches_per_wsi = int(max_patches_per_wsi)
        self.patch_sample_mode = str(patch_sample_mode)
        self.sample_seed = int(sample_seed)

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _to_tensor(value: object, dim: int) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=torch.float32)
        else:
            value = value.detach().cpu().float()
        if value.dim() == dim - 1:
            value = value.unsqueeze(0)
        if value.dim() != dim:
            return None
        if value.numel() == 0:
            return None
        return value

    def _sample_indices(self, n: int, case_id: str) -> np.ndarray:
        if self.max_patches_per_wsi <= 0 or n <= self.max_patches_per_wsi:
            return np.arange(n, dtype=np.int64)

        k = self.max_patches_per_wsi
        if self.patch_sample_mode == "head":
            return np.arange(k, dtype=np.int64)

        if self.patch_sample_mode == "random":
            seed = (hash(case_id) + self.sample_seed) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(n, size=k, replace=False))
            return idx.astype(np.int64)

        idx = np.linspace(0, n - 1, num=k)
        return np.unique(idx.round().astype(np.int64))

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        pt_path = os.path.join(self.embeddings_dir, f"{item.case_id}.pt")
        loaded = torch.load(pt_path, map_location="cpu")

        if isinstance(loaded, dict):
            embeddings = self._to_tensor(loaded.get("embeddings"), dim=2)
            coords = self._to_tensor(loaded.get("coords"), dim=2)
        elif torch.is_tensor(loaded):
            embeddings = self._to_tensor(loaded, dim=2)
            coords = None
        else:
            embeddings = None
            coords = None

        if embeddings is None:
            embeddings = torch.zeros(1, 512, dtype=torch.float32)
        if coords is None:
            coords = torch.zeros(embeddings.shape[0], 3, dtype=torch.float32)

        n = int(embeddings.shape[0])
        if coords.shape[0] != n:
            coords = torch.zeros(n, 3, dtype=torch.float32)

        chosen_idx = self._sample_indices(n, item.case_id)
        embeddings = embeddings[chosen_idx]
        coords = coords[chosen_idx]

        return {
            "case_id": item.case_id,
            "label": int(item.label),
            "embeddings": embeddings,
            "coords": coords,
            "patch_count": int(embeddings.shape[0]),
        }


def collate_wsi_batch(batch: List[Dict]) -> List[Dict]:
    return batch


def build_items_from_pt_labels(embeddings_dir: str) -> Tuple[List[WSIItem], Dict[str, int]]:
    files = [
        f
        for f in sorted(os.listdir(embeddings_dir))
        if f.endswith(".pt") and not f.startswith(".") and not f.startswith("._")
    ]
    if not files:
        raise FileNotFoundError(f"No .pt files found in {embeddings_dir}")

    labels_by_case: Dict[str, str] = {}
    for fname in files:
        case_id = os.path.splitext(fname)[0]
        pt_path = os.path.join(embeddings_dir, fname)
        try:
            loaded = torch.load(pt_path, map_location="cpu")
            if isinstance(loaded, dict):
                label = loaded.get("label")
                if label is not None:
                    labels_by_case[case_id] = str(label)
        except Exception:
            continue

    if not labels_by_case:
        raise ValueError(f"No usable 'label' field found in PT files under {embeddings_dir}.")

    labels = sorted(set(labels_by_case.values()))
    label_map = {name: i for i, name in enumerate(labels)}
    items = [
        WSIItem(case_id=case_id, label=label_map[label])
        for case_id, label in sorted(labels_by_case.items())
    ]
    return items, label_map


def build_items_from_pt_labels_with_map(
    embeddings_dir: str,
    label_map: Dict[str, int],
    strict: bool = True,
) -> List[WSIItem]:
    files = [
        f
        for f in sorted(os.listdir(embeddings_dir))
        if f.endswith(".pt") and not f.startswith(".") and not f.startswith("._")
    ]
    if not files:
        raise FileNotFoundError(f"No .pt files found in {embeddings_dir}")

    items: List[WSIItem] = []
    unknown = set()
    for fname in files:
        case_id = os.path.splitext(fname)[0]
        pt_path = os.path.join(embeddings_dir, fname)
        try:
            loaded = torch.load(pt_path, map_location="cpu")
            if not isinstance(loaded, dict):
                continue
            label = loaded.get("label")
            if label is None:
                continue
            label = str(label)
            if label not in label_map:
                unknown.add(label)
                continue
            items.append(WSIItem(case_id=case_id, label=label_map[label]))
        except Exception:
            continue

    if strict and unknown:
        unknown_labels = ", ".join(sorted(unknown))
        raise ValueError(
            f"Found labels in {embeddings_dir} missing from train label_map: {unknown_labels}"
        )

    if not items:
        raise ValueError(f"No usable samples found in {embeddings_dir} with provided label_map.")

    return items

"""Module for data."""

import json
import os
from pathlib import Path
import random
from dataclasses import dataclass
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.wsi import WSI
from src.utils.embedder import Embedder

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None


@dataclass
class WSIItem:
    case_id: str
    label: int


class WSIEmbeddingDataset(Dataset):
    def __init__(
        self,
        items: Sequence[WSIItem],
        embeddings_dir: str,
        images_dir: Optional[str] = None,
    ):
        self.items = list(items)
        self.embeddings_dir = embeddings_dir
        self.images_dir = images_dir
        self._embedder: Optional[Embedder] = None

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _normalize_precomputed_tensors(
        embeddings: object,
        coords: object,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if embeddings is None or coords is None:
            return None

        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        else:
            embeddings = embeddings.detach().cpu().float()

        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, dtype=torch.float32)
        else:
            coords = coords.detach().cpu().float()

        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        if coords.dim() == 1:
            coords = coords.unsqueeze(0)

        if embeddings.numel() == 0 or coords.numel() == 0:
            return None
        if embeddings.shape[0] != coords.shape[0]:
            return None

        return embeddings, coords

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]

        pt_path = os.path.join(self.embeddings_dir, f"{item.case_id}.pt")

        loaded = torch.load(pt_path, map_location="cpu")

        if not isinstance(loaded, dict):
            raise ValueError(
                f"Unsupported PT schema for {pt_path}. Expected dict payload."
            )

        precomputed = self._normalize_precomputed_tensors(
            embeddings=loaded.get("embeddings"),
            coords=loaded.get("coords"),
        )
        if precomputed is not None:
            embeddings, coords = precomputed
        else:
            embeddings, coords = self._extract_from_active_patches(item.case_id, loaded)

        patch_count = int(embeddings.shape[0])

        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        return {
            "case_id": item.case_id,
            "label": item.label,
            "embeddings": embeddings,
            "coords": coords,
            "patch_count": patch_count,
        }

    def _extract_from_active_patches(
        self, case_id: str, loaded: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._embedder is None:
            self._embedder = Embedder(img_backend="plip")

        img_path = loaded.get("img_path")
        if (not img_path or not os.path.exists(img_path)) and self.images_dir:
            img_path = os.path.join(self.images_dir, f"{case_id}.svs")

        if not img_path or not os.path.exists(img_path):
            return torch.zeros(1, 512), torch.zeros(1, 3, dtype=torch.float32)

        wsi = WSI(
            img_path,
            multistage=bool(loaded.get("multistage", False)),
            embedder=self._embedder,
        )
        wsi.active_patches = loaded.get("active_patches", {})
        wsi.zoomed_patches = loaded.get("zoomed_patches", {})

        keys = self._extract_active_keys(loaded.get("active_patches", {}))
        embs: List[torch.Tensor] = []
        coords: List[torch.Tensor] = []

        for lvl, x, y in keys:
            try:
                patch = wsi.get_patch(lvl, x, y)
                emb = wsi.get_emb(patch)
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().float().view(-1)
                else:
                    emb = torch.tensor(emb, dtype=torch.float32).view(-1)
                if emb.numel() != 512:
                    continue
                embs.append(emb)
                coords.append(
                    torch.tensor([float(lvl), float(x), float(y)], dtype=torch.float32)
                )
            except Exception:
                continue

        if not embs:
            return torch.zeros(1, 512), torch.zeros(1, 3, dtype=torch.float32)

        return torch.stack(embs), torch.stack(coords)

    @staticmethod
    def _extract_active_keys(active_patches: object) -> List[Tuple[int, int, int]]:
        keys: List[Tuple[int, int, int]] = []

        if isinstance(active_patches, dict):
            iterable = active_patches.keys()
        elif isinstance(active_patches, list):
            iterable = active_patches
        else:
            iterable = []

        for key in iterable:
            if isinstance(key, tuple) and len(key) == 3:
                lvl, x, y = key
            elif isinstance(key, list) and len(key) == 3:
                lvl, x, y = key
            else:
                continue
            try:
                keys.append((int(lvl), int(x), int(y)))
            except Exception:
                continue

        keys.sort(key=lambda k: (k[0], k[2], k[1]))
        return keys


def collate_wsi_batch(batch: List[Dict]) -> List[Dict]:
    return batch


def build_items_from_labels_json(
    labels_json: str,
    embeddings_dir: str,
) -> Tuple[List[WSIItem], Dict[str, int]]:
    with open(labels_json, "r") as fh:
        labels_map = json.load(fh)

    available = [
        case_id
        for case_id, label in labels_map.items()
        if label is not None
        and os.path.exists(os.path.join(embeddings_dir, f"{case_id}.pt"))
        and not case_id.startswith("._")
    ]

    if not available:
        raise FileNotFoundError(
            f"No matching embedding files found in {embeddings_dir} for labels in {labels_json}."
        )

    projects = sorted({labels_map[c] for c in available})
    proj_to_int = {project: idx for idx, project in enumerate(projects)}

    items = [
        WSIItem(case_id=case_id, label=proj_to_int[labels_map[case_id]])
        for case_id in sorted(available)
    ]
    return items, proj_to_int


def build_items_from_pt_labels(
    embeddings_dir: str,
) -> Tuple[List[WSIItem], Dict[str, int]]:
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
        raise ValueError(
            f"No usable 'label' field found in PT files under {embeddings_dir}."
        )

    projects = sorted(set(labels_by_case.values()))
    proj_to_int = {project: idx for idx, project in enumerate(projects)}
    items = [
        WSIItem(case_id=case_id, label=proj_to_int[label])
        for case_id, label in sorted(labels_by_case.items())
    ]
    return items, proj_to_int


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
    unknown_labels: Dict[str, int] = {}

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
            label_str = str(label)
            if label_str not in label_map:
                unknown_labels[label_str] = unknown_labels.get(label_str, 0) + 1
                continue
            items.append(WSIItem(case_id=case_id, label=label_map[label_str]))
        except Exception:
            continue

    if strict and unknown_labels:
        unknown = ", ".join(sorted(unknown_labels.keys()))
        raise ValueError(
            f"Found labels in {embeddings_dir} that are missing from train label_map: {unknown}"
        )

    if not items:
        raise ValueError(
            f"No usable samples found in {embeddings_dir} with the provided label_map."
        )

    return items


def split_items(
    items: Sequence[WSIItem],
    train_count: int = 800,
    val_count: int = 100,
    test_count: int = 100,
    seed: int = 42,
    stratified: bool = True,
) -> Tuple[List[WSIItem], List[WSIItem], List[WSIItem]]:
    items = list(items)
    n_items = len(items)
    requested = train_count + val_count + test_count

    if requested != n_items:
        if requested > n_items:
            raise ValueError(
                f"Requested split counts ({requested}) exceed available items ({n_items})."
            )
        train_count = int(round(n_items * (train_count / requested)))
        val_count = int(round(n_items * (val_count / requested)))
        test_count = n_items - train_count - val_count

    rng = random.Random(seed)
    labels = [it.label for it in items]

    if stratified and train_test_split is not None:
        idx = list(range(n_items))
        train_idx, temp_idx = train_test_split(
            idx,
            train_size=train_count,
            stratify=labels,
            random_state=seed,
        )

        temp_labels = [labels[i] for i in temp_idx]
        val_fraction = val_count / max(val_count + test_count, 1)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_fraction,
            stratify=temp_labels,
            random_state=seed,
        )

        train_items = [items[i] for i in train_idx]
        val_items = [items[i] for i in val_idx]
        test_items = [items[i] for i in test_idx]
        return train_items, val_items, test_items

    shuffled = items[:]
    rng.shuffle(shuffled)
    train_items = shuffled[:train_count]
    val_items = shuffled[train_count : train_count + val_count]
    test_items = shuffled[
        train_count + val_count : train_count + val_count + test_count
    ]
    return train_items, val_items, test_items

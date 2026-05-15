"""Data utilities for SASHA training/inference over PT or raw SVS inputs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class PTEmbeddingSample:
    case_id: str
    label_str: str
    label_idx: int
    embeddings: torch.Tensor
    coords: torch.Tensor
    img_path: Optional[str]
    multistage: bool
    active_patches: object
    zoomed_patches: object
    source_pt_path: str


class PTEmbeddingDirDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Tuple[str, str, int]],
        max_patches_per_wsi: int = 0,
        sample_mode: str = "uniform",
        sample_seed: int = 42,
        svs_level_mode: str = "root_only",
        svs_embed_backend: str = "plip",
    ):
        self.items = list(items)
        self.max_patches_per_wsi = int(max_patches_per_wsi)
        self.sample_mode = str(sample_mode)
        self.sample_seed = int(sample_seed)
        self.svs_level_mode = str(svs_level_mode)
        self.svs_embed_backend = str(svs_embed_backend)
        self._embedder = None

    def __len__(self) -> int:
        return len(self.items)

    # Coerce value to float32 tensor of the requested rank
    @staticmethod
    def _to_tensor(value: object, dim: int) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if torch.is_tensor(value):
            tensor = value.detach().cpu().float()
        else:
            tensor = torch.tensor(value, dtype=torch.float32)
        if tensor.dim() == dim - 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != dim or tensor.numel() == 0:
            return None
        return tensor

    # Subsample patch indices according to max_patches_per_wsi and sample_mode
    def _sample_indices(self, n: int, case_id: str) -> torch.Tensor:
        if self.max_patches_per_wsi <= 0 or n <= self.max_patches_per_wsi:
            return torch.arange(n, dtype=torch.long)

        k = self.max_patches_per_wsi
        if self.sample_mode == "head":
            return torch.arange(k, dtype=torch.long)

        if self.sample_mode == "random":
            seed = (hash(case_id) + self.sample_seed) & 0xFFFFFFFF
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            idx = torch.randperm(n, generator=generator)[:k]
            idx, _ = torch.sort(idx)
            return idx

        # uniform
        idx_float = torch.linspace(0, n - 1, steps=k)
        idx = torch.unique(idx_float.round().long())
        if idx.numel() < k:
            needed = k - idx.numel()
            tail = torch.arange(n - needed, n, dtype=torch.long)
            idx = torch.unique(torch.cat([idx, tail], dim=0))
            idx, _ = torch.sort(idx)
            idx = idx[:k]
        return idx

    # Lazy-initialize embedding backend
    def _get_embedder(self):
        if self._embedder is None:
            from src.utils.embedder import Embedder

            self._embedder = Embedder(img_backend=self.svs_embed_backend)
        return self._embedder

    # Derive label string from SVS filename stem (last dash-separated token)
    @staticmethod
    def _svs_label_from_stem(stem: str) -> str:
        tokens = [tok for tok in stem.split("-") if tok]
        if tokens:
            return tokens[-1].upper()
        return stem.upper()

    # Load and embed patches directly from an SVS file
    def _load_from_svs(
        self, svs_path: str, label_str: str, label_idx: int
    ) -> PTEmbeddingSample:
        from src.utils.wsi import WSI

        embedder = self._get_embedder()
        wsi = WSI(svs_path, embedder=embedder)

        if self.svs_level_mode == "root_only":
            target_level = int(wsi.max_level)
        elif self.svs_level_mode == "finest_only":
            target_level = int(wsi.min_level)
        else:
            raise ValueError(f"Unknown svs_level_mode: {self.svs_level_mode}")

        keys = [
            (target_level, int(x), int(y)) for x, y in wsi.iterate_patches(target_level)
        ]
        if not keys:
            embeddings = torch.zeros(1, 512, dtype=torch.float32)
            coords = torch.zeros(1, 3, dtype=torch.float32)
            active = {(target_level, 0, 0): {}}
            return PTEmbeddingSample(
                case_id=str(Path(svs_path).stem),
                label_str=str(label_str),
                label_idx=int(label_idx),
                embeddings=embeddings,
                coords=coords,
                img_path=str(svs_path),
                multistage=False,
                active_patches=active,
                zoomed_patches={},
                source_pt_path=str(svs_path),
            )

        chosen = self._sample_indices(len(keys), Path(svs_path).stem)
        selected_keys = [keys[int(i)] for i in chosen.tolist()]

        embs: List[torch.Tensor] = []
        coord_rows: List[torch.Tensor] = []
        active = {}

        for lvl, x, y in selected_keys:
            try:
                patch = wsi.get_patch(lvl, x, y)
                emb = wsi.get_emb(patch)
                if torch.is_tensor(emb):
                    emb = emb.detach().cpu().float().view(-1)
                else:
                    emb = torch.tensor(emb, dtype=torch.float32).view(-1)
                if emb.numel() == 0:
                    continue
                embs.append(emb)
                coord_rows.append(
                    torch.tensor([float(lvl), float(x), float(y)], dtype=torch.float32)
                )
                active[(int(lvl), int(x), int(y))] = {}
            except Exception:
                continue

        if not embs:
            embeddings = torch.zeros(1, 512, dtype=torch.float32)
            coords = torch.zeros(1, 3, dtype=torch.float32)
            active = {(target_level, 0, 0): {}}
        else:
            embeddings = torch.stack(embs)
            coords = torch.stack(coord_rows)

        return PTEmbeddingSample(
            case_id=str(Path(svs_path).stem),
            label_str=str(label_str),
            label_idx=int(label_idx),
            embeddings=embeddings,
            coords=coords,
            img_path=str(svs_path),
            multistage=False,
            active_patches=active,
            zoomed_patches={},
            source_pt_path=str(svs_path),
        )

    def __getitem__(self, index: int) -> PTEmbeddingSample:
        sample_path, label_str, label_idx = self.items[index]
        ext = Path(sample_path).suffix.lower()

        if ext == ".svs":
            return self._load_from_svs(
                sample_path, label_str=label_str, label_idx=label_idx
            )

        # Load PT file
        pt_path = sample_path
        loaded = torch.load(pt_path, map_location="cpu")

        if not isinstance(loaded, dict):
            raise ValueError(f"Expected dict PT payload in {pt_path}")

        embeddings = self._to_tensor(loaded.get("embeddings"), dim=2)
        if embeddings is None:
            embeddings = torch.zeros(1, 512, dtype=torch.float32)

        coords = self._to_tensor(loaded.get("coords"), dim=2)
        if coords is None or coords.shape[0] != embeddings.shape[0]:
            coords = torch.zeros(embeddings.shape[0], 3, dtype=torch.float32)

        # Sample patch subset
        chosen = self._sample_indices(int(embeddings.shape[0]), Path(pt_path).stem)
        embeddings = embeddings[chosen]
        coords = coords[chosen]

        return PTEmbeddingSample(
            case_id=str(loaded.get("case_id") or Path(pt_path).stem),
            label_str=str(label_str),
            label_idx=int(label_idx),
            embeddings=embeddings,
            coords=coords,
            img_path=loaded.get("img_path"),
            multistage=bool(loaded.get("multistage", False)),
            active_patches=loaded.get("active_patches", {}),
            zoomed_patches=loaded.get("zoomed_patches", {}),
            source_pt_path=str(pt_path),
        )


# List .pt or .svs files in a directory
def list_input_files(input_dir: str, input_format: str = "pt") -> List[str]:
    input_format = str(input_format).lower()
    if input_format not in {"pt", "svs", "auto"}:
        raise ValueError("input_format must be one of {'pt', 'svs', 'auto'}")

    base = Path(input_dir)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    if input_format == "pt":
        exts = {".pt"}
    elif input_format == "svs":
        exts = {".svs"}
    else:
        exts = {".pt", ".svs"}

    files = [
        str(p)
        for p in sorted(base.iterdir())
        if p.is_file()
        and p.suffix.lower() in exts
        and not p.name.startswith(".")
        and not p.name.startswith("._")
    ]
    if not files:
        raise FileNotFoundError(
            f"No matching files ({sorted(exts)}) found in {input_dir}"
        )
    return files


# Read label string from a .pt or .svs file path
def read_label_from_path(sample_path: str) -> str:
    ext = Path(sample_path).suffix.lower()
    if ext == ".pt":
        loaded = torch.load(sample_path, map_location="cpu")
        if isinstance(loaded, dict) and loaded.get("label") is not None:
            return str(loaded.get("label"))
        raise ValueError(f"Missing 'label' in {sample_path}")

    if ext == ".svs":
        return PTEmbeddingDirDataset._svs_label_from_stem(Path(sample_path).stem)

    raise ValueError(f"Unsupported file extension: {sample_path}")


# Build item list and label map from a training directory
def build_items_and_label_map(
    train_dir: str,
    input_format: str = "pt",
) -> Tuple[List[Tuple[str, str, int]], Dict[str, int]]:
    train_files = list_input_files(train_dir, input_format=input_format)

    labels: Dict[str, str] = {}
    for sample_path in train_files:
        try:
            labels[sample_path] = read_label_from_path(sample_path)
        except Exception:
            continue

    if not labels:
        raise ValueError(f"No usable labels found in {train_dir}")

    uniq = sorted(set(labels.values()))
    label_map = {name: idx for idx, name in enumerate(uniq)}
    items = [
        (path, labels[path], label_map[labels[path]]) for path in sorted(labels.keys())
    ]
    return items, label_map


# Build item list from a directory using a pre-existing label map
def build_items_with_label_map(
    embeddings_dir: str,
    label_map: Dict[str, int],
    strict: bool = True,
    input_format: str = "pt",
) -> List[Tuple[str, str, int]]:
    files = list_input_files(embeddings_dir, input_format=input_format)

    items: List[Tuple[str, str, int]] = []
    unknown_labels = set()
    for path in files:
        try:
            label_str = read_label_from_path(path)
        except Exception:
            continue

        if label_str not in label_map:
            unknown_labels.add(label_str)
            continue

        items.append((path, label_str, int(label_map[label_str])))

    if strict and unknown_labels:
        raise ValueError(
            f"Found labels in {embeddings_dir} not in train label map: {sorted(unknown_labels)}"
        )

    if not items:
        raise ValueError(f"No usable PT samples in {embeddings_dir}")

    return items


# Pass-through collate — each WSI has a different patch count so batching is deferred
def collate_samples(batch: List[PTEmbeddingSample]) -> List[PTEmbeddingSample]:
    return batch

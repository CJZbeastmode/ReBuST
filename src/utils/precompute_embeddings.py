"""
Precompute embeddings — patch-level feature extraction
======================================================

Precompute patch embeddings for a list of images and store them on disk.

Design goals:
-------------
- Robust to individual patch failures
- Safe to run unattended on large WSI collections
- Embeddings stored on CPU as float32 for portability
"""

# ==========================================================================
# Imports
# ==========================================================================

import os
import glob
from typing import Callable, List, Tuple, Optional

import numpy as np
import torch
from src.utils.wsi import WSI


# ============================================================================
# precompute_embeddings
# ============================================================================
def precompute_embeddings(
    image_paths: List[str],
    embedder,
    extractor_fn: Callable,
    out_dir: str,
    patches_per_image: int = 200,
    label_fn: Optional[Callable] = None,
):
    """
    Core pipeline:
    - iterate over images
    - extract patches (user-defined strategy)
    - compute embeddings
    - save everything per-image

    Args:
        image_paths (List[str]): Paths to input images (WSI files).
        embedder (object): Embedder with an `img_emb(patch)` method.
        extractor_fn (Callable): Function yielding `(patch, coord)` tuples.
        out_dir (str): Output directory for per-image .npz files.
        patches_per_image (int): Number of patches to extract per image.
        label_fn (Optional[Callable]): Optional label callback.

    Returns:
        None: Writes .npz files to disk.
    """

    # Ensure output directory exists (best-effort)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        return

    for p in image_paths:
        embeds = []
        coords = []
        labels = []

        # Patch extraction loop
        try:
            patch_iter = extractor_fn(p, patches_per_image)
        except Exception:
            continue

        for i, item in enumerate(patch_iter):
            # Defensive unpacking
            try:
                patch, coord = item
            except Exception:
                continue

            # --------------------------------------------------
            # Embedding computation (fully guarded)
            # --------------------------------------------------
            try:
                with torch.inference_mode():
                    emb = embedder.img_emb(patch)
            except Exception:
                continue

            # Normalize embedding to numpy float32
            try:
                if isinstance(emb, torch.Tensor):
                    emb_arr = emb.detach().cpu().numpy().astype(np.float32)
                else:
                    emb_arr = np.asarray(emb, dtype=np.float32)
            except Exception:
                continue

            embeds.append(emb_arr)

            # Coordinate handling
            try:
                coords.append(np.asarray(coord, dtype=np.float32))
            except Exception:
                coords.append(np.zeros_like(emb_arr[:3], dtype=np.float32))

            # Optional label computation
            if label_fn is not None:
                try:
                    lab = float(label_fn(p, i, patch, coord))
                except Exception:
                    lab = 0.0
                labels.append(lab)

        # Skip images that produced no valid embeddings
        if len(embeds) == 0:
            continue

        # Stack arrays safely
        try:
            embeds = np.stack(embeds, axis=0)
            coords = np.stack(coords, axis=0)
        except Exception:
            continue

        # Write output file
        out_path = os.path.join(out_dir, os.path.basename(p) + ".npz")
        try:
            if labels:
                labels = np.array(labels, dtype=np.float32)
                np.savez_compressed(
                    out_path, embeds=embeds, coords=coords, labels=labels
                )
            else:
                np.savez_compressed(out_path, embeds=embeds, coords=coords)
        except Exception:
            continue


# ============================================================================
# precompute_from_wsi_folder
# ============================================================================
def precompute_from_wsi_folder(
    images_dir: str,
    embedder,
    out_dir: str,
    patches_per_image: Optional[int] = 200,
    levels: Optional[List[int]] = None,
    sample_all: bool = False,
    random_seed: Optional[int] = None,
    label_fn: Optional[Callable] = None,
):
    """
    Convenience wrapper for bulk embedding precomputation from a WSI directory.

    This function:
    -- Discovers WSI files in `images_dir`
    -- Builds a WSI-based patch extractor
    -- Delegates to `precompute_embeddings`

    The design intentionally separates extraction logic from embedding logic,
    enabling reuse with alternative sampling strategies.

    Args:
        images_dir (str): Directory containing WSI files.
        embedder (object): Embedder with an `img_emb(patch)` method.
        out_dir (str): Output directory for per-image .npz files.
        patches_per_image (Optional[int]): Patches per image (ignored if sample_all).
        levels (Optional[List[int]]): Pyramid levels to sample.
        sample_all (bool): If True, sample all patches at chosen levels.
        random_seed (Optional[int]): RNG seed for sampling.
        label_fn (Optional[Callable]): Optional label callback.

    Returns:
        None: Writes .npz files to disk.
    """

    # --------------------------------------------------
    # Discover WSI files
    # --------------------------------------------------
    exts = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".vsi")
    try:
        image_paths = [
            os.path.join(images_dir, f)
            for f in sorted(os.listdir(images_dir))
            if f.lower().endswith(exts)
        ]
    except Exception:
        image_paths = []

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def extractor_fn_for_wsi(path, n):
        """
        Patch extractor for a single WSI.
        """
        try:
            wsi = WSI(path)
        except Exception:
            return

        # Determine which pyramid levels to sample
        try:
            chosen_levels = levels if levels is not None else [wsi.max_level]
        except Exception:
            return

        # Collect candidate coordinates
        candidates = []
        for lvl in chosen_levels:
            try:
                for x, y in wsi.iterate_patches(lvl):
                    candidates.append((lvl, x, y))
            except Exception:
                continue

        # Randomized selection
        rng = np.random.default_rng(random_seed)
        if sample_all:
            selected = candidates
        else:
            total = len(candidates)
            if total == 0:
                selected = []
            else:
                try:
                    if n >= total:
                        idxs = rng.permutation(total)[:n]
                    else:
                        idxs = rng.choice(total, size=n, replace=False)
                    selected = [candidates[i] for i in idxs]
                except Exception:
                    selected = []

        # Patch extraction
        for lvl, x, y in selected:
            try:
                patch = wsi.get_patch(lvl, x, y)
                coord = np.array([lvl, x, y], dtype=np.float32)
                yield patch, coord
            except Exception:
                continue

    precompute_embeddings(
        image_paths,
        embedder,
        extractor_fn_for_wsi,
        out_dir,
        patches_per_image if not sample_all else -1,
        label_fn=label_fn,
    )


# ==================================================
# Dataset for precomputed embeddings
# ==================================================
class EmbeddingDataset(torch.utils.data.Dataset):
    """
    Torch Dataset for precomputed patch embeddings.

    Each `.npz` file contributes multiple samples. Each sample consists of:
    - state: concatenation of (coords, embedding) or embedding alone
    - label: optional supervision signal (defaults to 0.0)

    This dataset is designed for:
    - supervised baselines
    - RL
    """

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, npz_dir: str, include_coords: bool = True):
        """
        Initialize the dataset by scanning the directory for `.npz` files and building an index.

        Input:
        -- npz_dir: directory containing .npz files with precomputed embeddings
        -- include_coords: whether to include coordinates in the state representation
        """

        # Discover .npz files
        try:
            files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        except Exception:
            files = []

        self.files = files
        self.include_coords = include_coords

        # --------------------------------------------------
        # Build global index
        # --------------------------------------------------
        self.index = []  # list of (file_idx, item_idx)
        self._meta = []

        # Iterate over files to build index and gather metadata
        for i, fpath in enumerate(self.files):
            try:
                with np.load(fpath, mmap_mode="r") as data:
                    n = int(data["embeds"].shape[0])
                    self.index.extend([(i, j) for j in range(n)])
                    self._meta.append(
                        {
                            "n": n,
                            "has_labels": "labels" in data or "rewards" in data,
                            "embed_dim": int(data["embeds"].shape[1]),
                            "coord_dim": int(data["coords"].shape[1]),
                        }
                    )
            except Exception:
                continue

        # --------------------------------------------------
        # Derive state dimensionality
        # --------------------------------------------------
        if len(self._meta) > 0:
            meta0 = self._meta[0]
            self.state_dim = meta0["embed_dim"] + (
                meta0["coord_dim"] if self.include_coords else 0
            )
        else:
            self.state_dim = 0

    # ---------------------------------------------------------------------------
    # __len__
    # ---------------------------------------------------------------------------
    def __len__(self):
        """
        Total number of samples across all files.
        """
        return len(self.index)

    # ---------------------------------------------------------------------------
    # __getitem__
    # ---------------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Load a single sample from disk.

        Returns:
            state: torch.FloatTensor
            label: torch.FloatTensor (scalar)
        """
        file_idx, item_idx = self.index[idx]
        fpath = self.files[file_idx]

        try:
            data = np.load(fpath, mmap_mode="r")
            emb = data["embeds"][item_idx].astype(np.float32)
            coord = data["coords"][item_idx].astype(np.float32)

            if "labels" in data:
                label = float(data["labels"][item_idx])
            elif "rewards" in data:
                label = float(data["rewards"][item_idx])
            else:
                label = 0.0
            data.close()
        except Exception:
            # Hard fallback: zero sample
            emb = np.zeros(self.state_dim, dtype=np.float32)
            coord = np.zeros(0, dtype=np.float32)
            label = 0.0

        if self.include_coords:
            try:
                state = np.concatenate([coord, emb]).astype(np.float32)
            except Exception:
                state = emb.astype(np.float32)
        else:
            state = emb.astype(np.float32)

        return torch.tensor(state), torch.tensor(label)

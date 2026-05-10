"""Module for collect dataset."""

import sys
from pathlib import Path
import time
import argparse

from openslide import OpenSlideUnsupportedFormatError

# ---------------------------------------------------------------------
# Ensure the repository root is on sys.path so `src` is importable
# This allows running the script directly via:
#   python src/data_collection/collect_dataset.py
# ---------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import numpy as np
import torch
from src.utils.dynamic_patch_env import DynamicPatchEnv
from src.utils.wsi import WSI


CLASS_AWARE_SCORES = {"cancer_centroid_score", "contrastive_text_score"}


def resolve_image_class_from_path(image_path: str):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    normalized = stem.replace("_", "-")
    tokens = [tok.upper() for tok in normalized.split("-") if tok]
    for token in reversed(tokens):
        if token.isalpha() and 3 <= len(token) <= 6 and token != "TCGA":
            return token
    return None


# =====================================================================
# OPTIONAL BACKUP HELPER
# =====================================================================
def maybe_backup(states, scores, zoom_decisions, out_npz=None, image_classes=None):
    """
    Save a lightweight NPZ snapshot of the environment state.

    This is a best-effort diagnostic utility:
    - It must never crash the main pipeline
    - Failures are silently ignored
    - Intended for debugging long-running data collection

    The snapshot contains:
        - current level / coordinates
        - step counters
        - optional STOP / ZOOM scores
        - optional image embedding
    """
    try:
        states_np = np.stack(states, axis=0)
        scores_np = np.stack(scores, axis=0)  # shape (N, 2)
        zoom_decisions_np = np.asarray(zoom_decisions, dtype=np.int64)

        print("Saved dataset...")

        if out_npz is not None:
            os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
            payload = {
                "states": states_np,
                "scores": scores_np,
                "zoom_decision": zoom_decisions_np,
            }
            if image_classes is not None:
                payload["image_class"] = np.asarray(image_classes)
            np.savez_compressed(out_npz, **payload)

        print("Backup dataset saving complete.")
    except Exception as e:
        print(f"Warning: could not save dataset backup: {e}")


# =====================================================================
# DATASET COLLECTION
# =====================================================================
def collect_dataset(
    images_dir: str = "data/images",
    out_npz: str = "data/supervised_dataset/score_regressor_text_alignment_score.npz",
    max_samples: int = None,
    random_seed: int = None,
    score_module="text_align_score",
    score_module_aggregation=None,
    img_embedding_backend="plip",
):
    """
    Collect supervised training data for a Score Regressor.

    For each WSI:
        - iterate pyramid levels from max_level down to (but excluding) min_level
        - extract patch-level states
        - compute STOP / ZOOM scores via DynamicPatchEnv
        - store (state, score_pair, zoom_decision)

    IMPORTANT SEMANTICS:
        - min_level is NEVER used as a parent
        - only levels where ZOOM is valid are included
        - min_level patches appear only as children in inference, not here
    """
    print("Starting dataset collection for Score Regressor ...")

    # -----------------------------------------------------------------
    # Resolve image paths
    # -----------------------------------------------------------------
    if isinstance(images_dir, str) and os.path.isdir(images_dir):
        exts = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".vsi")
        image_paths = [
            os.path.join(images_dir, f)
            for f in sorted(os.listdir(images_dir))
            if f.lower().endswith(exts)
        ]
    else:
        # Allow passing an explicit list of files
        image_paths = list(images_dir)

    if len(image_paths) == 0:
        raise ValueError(f"No WSI files found in {images_dir}")

    # Storage buffers
    states = []
    scores = []
    zoom_decisions = []
    collect_image_class = score_module in CLASS_AWARE_SCORES
    image_classes = [] if collect_image_class else None
    collected = 0

    # -----------------------------------------------------------------
    # Main loop over WSIs
    # -----------------------------------------------------------------
    t0 = time.time()
    for image_path in image_paths:
        print(
            f"Processing image {image_path} ({image_paths.index(image_path)+1}/{len(image_paths)})"
        )

        try:
            wsi = WSI(image_path, img_embedding_backend=img_embedding_backend)
        except OpenSlideUnsupportedFormatError as e:
            print(f"[SKIP] OpenSlide cannot read: {image_path}")
            continue
        except FileNotFoundError:
            print(f"[SKIP] Missing file: {image_path}")
            continue
        except Exception as e:
            print(f"[SKIP] Unknown error opening {image_path}: {e}")
            continue

        env = DynamicPatchEnv(
            wsi,
            patch_score=score_module,
            patch_score_aggregation=score_module_aggregation,
            image_class=resolve_image_class_from_path(image_path),
        )

        max_level = wsi.max_level
        min_level = wsi.min_level

        # -------------------------------------------------------------
        # Iterate pyramid levels TOP-DOWN, excluding min_level
        # This ensures no terminal (non-zoomable) states are included
        # -------------------------------------------------------------
        for lvl in range(max_level, min_level, -1):
            print(f"  Inspecting Level {lvl}/{max_level}")

            # Extra safety: skip level 0 explicitly if present
            if lvl == 0:
                print("    (skipping level 0)")
                continue

            # ---------------------------------------------------------
            # Iterate all patches at this level
            # ---------------------------------------------------------
            for x, y in wsi.iterate_patches(lvl):

                # Compute STOP / ZOOM scores
                try:
                    valid, score_pair = env.calculate_score(lvl, x, y)
                except Exception as e:
                    print(f"Error calculating score at level {lvl}, x {x}, y {y}: {e}")
                    continue

                if not valid:
                    continue

                # Encode state (coords + embedding)
                state = env.encode_state(
                    wsi.get_patch(lvl, x, y),
                    lvl=lvl,
                    x=x,
                    y=y,
                )

                r_stop, r_zoom = score_pair
                zoom_decision = env.infer_zoom_decision(r_stop, r_zoom)

                print(
                    "Collected sample:",
                    lvl,
                    x,
                    y,
                    "Scores:",
                    score_pair,
                    "Zoom decision:",
                    zoom_decision,
                )

                # Append to dataset
                states.append(state.astype(np.float32))
                scores.append(np.asarray(score_pair, dtype=np.float32))
                zoom_decisions.append(zoom_decision)
                if collect_image_class:
                    image_classes.append(env.image_class or "UNKNOWN")
                collected += 1

                # Optional hard cap on total samples
                if max_samples is not None and collected >= max_samples:
                    break

        if max_samples is not None and collected >= max_samples:
            break

        t1 = time.time()
        print(
            f"Processed image ({image_paths.index(image_path)+1}/{len(image_paths)}) - Elapsed time: {t1 - t0:.2f} seconds"
        )

        # Optional per-image backup snapshot
        try:
            maybe_backup(
                states,
                scores,
                zoom_decisions,
                out_npz=out_npz,
                image_classes=image_classes,
            )
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Finalization
    # -----------------------------------------------------------------
    if len(states) == 0:
        raise ValueError("No valid samples collected from provided images")

    states_np = np.stack(states, axis=0)
    scores_np = np.stack(scores, axis=0)  # shape (N, 2)
    zoom_decisions_np = np.asarray(zoom_decisions, dtype=np.int64)

    print("Saved dataset...")

    if out_npz is not None:
        os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
        payload = {
            "states": states_np,
            "scores": scores_np,
            "zoom_decision": zoom_decisions_np,
        }
        if image_classes is not None:
            payload["image_class"] = np.asarray(image_classes)
        np.savez_compressed(out_npz, **payload)

    print("Dataset collection complete.")

    return (
        torch.tensor(states_np, dtype=torch.float32),
        torch.tensor(scores_np, dtype=torch.float32),
        torch.tensor(zoom_decisions_np, dtype=torch.int64),
    )


# =====================================================================
# CLI ENTRY POINT
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect supervised score-regressor dataset from WSIs"
    )

    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/images",
        help="Directory containing WSI files (.svs, .tif, .ndpi, ...)",
    )

    parser.add_argument(
        "--out-npz",
        type=str,
        required=True,
        help="Output .npz file for collected dataset",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional hard cap on total number of samples",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for shuffling image order",
    )

    parser.add_argument(
        "--score-module",
        type=str,
        required=True,
        help="Score module used by DynamicPatchEnv",
    )

    parser.add_argument(
        "--score-module-aggregation",
        type=str,
        default=None,
        help="Score module used by DynamicPatchEnv",
    )

    parser.add_argument(
        "--img-embedding-backend",
        type=str,
        default="plip",
        help="Score module used by DynamicPatchEnv",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    collect_dataset(
        out_npz=args.out_npz,
        score_module=args.score_module,
        score_module_aggregation=args.score_module_aggregation,
        img_embedding_backend=args.img_embedding_backend,
    )

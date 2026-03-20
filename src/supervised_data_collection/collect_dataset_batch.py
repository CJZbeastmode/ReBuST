import sys
from pathlib import Path
import time
import argparse
import os
import json

import numpy as np
import torch
from openslide import OpenSlideUnsupportedFormatError

# ---------------------------------------------------------------------
# Ensure repo root on path
# ---------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.dynamic_patch_env import DynamicPatchEnv
from src.utils.wsi import WSI


# =====================================================================
# BACKUP HELPER (NON-QUADRATIC)
# =====================================================================
def maybe_backup(states, scores, zoom_decisions, out_npz):
    try:
        np.savez_compressed(
            out_npz,
            states=np.stack(states),
            scores=np.stack(scores),
            zoom_decision=np.asarray(zoom_decisions),
        )
        print("[Backup] Snapshot saved.")
    except Exception as e:
        print(f"[Backup] Failed: {e}")


# =====================================================================
# DATASET COLLECTION (BATCHED, NO RANDOMNESS)
# =====================================================================
def collect_dataset(
    images_dir: str,
    out_npz: str,
    score_module: str,
    score_module_aggregation=None,
    img_embedding_backend="plip",
    max_samples: int = None,
    batch_size: int = 64,
    backup_every_n_images: int = 5,
):
    print("Starting dataset collection (batched, faithful)")

    # -----------------------------------------------------------------
    # Resolve image paths (NO SHUFFLING)
    # -----------------------------------------------------------------
    exts = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".vsi")
    image_paths = [
        os.path.join(images_dir, f)
        for f in sorted(os.listdir(images_dir))
        if f.lower().endswith(exts)
    ]

    if not image_paths:
        raise ValueError(f"No WSI files found in {images_dir}")

    # -----------------------------------------------------------------
    # Storage buffers
    # -----------------------------------------------------------------
    states = []
    scores = []
    zoom_decisions = []
    used_images = []

    t0 = time.time()

    # -----------------------------------------------------------------
    # Main loop over WSIs
    # -----------------------------------------------------------------
    for img_idx, image_path in enumerate(image_paths, start=1):
        print(f"\nProcessing image {image_path} ({img_idx}/{len(image_paths)})")

        try:
            wsi = WSI(image_path, img_embedding_backend=img_embedding_backend)
        except OpenSlideUnsupportedFormatError:
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
        )

        used_images.append(image_path)

        patch_buffer = []
        meta_buffer = []

        max_level = wsi.max_level
        min_level = wsi.min_level

        # -------------------------------------------------------------
        # Iterate pyramid TOP-DOWN (faithful to your logic)
        # -------------------------------------------------------------
        for lvl in range(max_level, min_level, -1):
            print(f"  Inspecting Level {lvl}/{max_level}")

            if lvl == 0:
                print("    (skipping level 0)")
                continue

            for x, y in wsi.iterate_patches(lvl):

                try:
                    valid, score_pair = env.calculate_score(lvl, x, y)
                except Exception as e:
                    print(f"Error calculating score at lvl={lvl}, x={x}, y={y}: {e}")
                    continue

                if not valid:
                    continue

                patch = wsi.get_patch(lvl, x, y)

                patch_buffer.append(patch)
                meta_buffer.append((lvl, x, y, score_pair))

                # -----------------------------------------------------
                # Flush batch
                # -----------------------------------------------------
                if len(patch_buffer) >= batch_size:
                    _flush_batch(
                        env,
                        patch_buffer,
                        meta_buffer,
                        states,
                        scores,
                        zoom_decisions,
                    )
                    patch_buffer.clear()
                    meta_buffer.clear()

                    if max_samples and len(states) >= max_samples:
                        break

            if max_samples and len(states) >= max_samples:
                break

        # Flush remaining patches
        if patch_buffer:
            _flush_batch(
                env,
                patch_buffer,
                meta_buffer,
                states,
                scores,
                zoom_decisions,
            )

        # Backup occasionally (not every image)
        if img_idx % backup_every_n_images == 0:
            maybe_backup(states, scores, zoom_decisions, out_npz)

        if max_samples and len(states) >= max_samples:
            break

        print(
            f"Processed image ({img_idx}/{len(image_paths)}) "
            f"- elapsed {time.time() - t0:.2f}s"
        )

    # -----------------------------------------------------------------
    # Final save
    # -----------------------------------------------------------------
    if not states:
        raise ValueError("No valid samples collected")

    print("\nFinalizing dataset...")
    np.savez_compressed(
        out_npz,
        states=np.stack(states),
        scores=np.stack(scores),
        zoom_decision=np.asarray(zoom_decisions),
    )

    with open(out_npz.replace(".npz", "_meta.json"), "w") as f:
        json.dump(
            {
                "used_images": used_images,
                "batch_size": batch_size,
            },
            f,
            indent=2,
        )

    print(
        f"Dataset complete: {len(states)} samples "
        f"in {time.time() - t0:.1f}s"
    )

    return (
        torch.tensor(np.stack(states), dtype=torch.float32),
        torch.tensor(np.stack(scores), dtype=torch.float32),
        torch.tensor(np.asarray(zoom_decisions), dtype=torch.int64),
    )


# =====================================================================
# BATCH FLUSH (CORE SPEEDUP)
# =====================================================================
def _flush_batch(env, patches, metas, states, scores, zoom_decisions):
    """
    Faithful batching:
    - scores already computed
    - only embeddings are batched
    """
    embeddings = env.embedder.embed_batch(patches)

    for emb, (lvl, x, y, score_pair) in zip(embeddings, metas):
        state = env.encode_state_from_embedding(
            emb, lvl=lvl, x=x, y=y
        )

        r_stop, r_zoom = score_pair
        zoom_decision = env.infer_zoom_decision(r_stop, r_zoom)

        states.append(state.astype(np.float32))
        scores.append(np.asarray(score_pair, dtype=np.float32))
        zoom_decisions.append(zoom_decision)


# =====================================================================
# CLI
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

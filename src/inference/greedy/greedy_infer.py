"""
Greedy inference utility

Usage:
  python src/greedy_infer.py --image data/images/TCGA-05-4382.svs --zoom-thresh 0.0 --max-depth 6

This performs deterministic recursive zooming using the score module:
- For each patch, compute `score = score_module.compute(action=1, parent_patch=patch, child_patches=[])`
- If score > zoom_thresh: zoom (recurse into children)
- Else: keep the patch

Outputs kept/discarded patches and a visualization (uses `visualize_patches.generate_visualization`).
"""

# ADDED COMMENT:
# This script implements a *deterministic* (non-RL) baseline for hierarchical
# patch selection. It is intended for:
#   - qualitative inspection of score behavior
#   - sanity-checking patch_score_module logic
#   - providing a non-learning baseline for comparisons

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src` package imports work when running
# this script directly (python src/inference/greedy_infer.py)
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ADDED COMMENT:
# Explicit path injection avoids brittle PYTHONPATH assumptions and allows
# this file to be executed both as a module and as a standalone script.

import argparse
import os
from src.utils.wsi import WSI
from src.utils.dynamic_patch_env import DynamicPatchEnv
from src.utils.embedder import Embedder
from src.utils.patch_scores import *

# ADDED COMMENT:
# Imports intentionally mirror the RL pipeline to ensure that greedy inference
# operates on exactly the same abstractions (WSI, Env, ScoreModules).


def greedy_infer_zoom(env, level, x, y, max_depth=10):
    # print(f"At level {level}, x {x}, y {y}")
    kept = []
    discarded = []

    patch = env.wsi.get_patch(level, x, y)

    score_exists, score = env.calculate_score(level, x, y)

    if not score_exists:
        print("Case 1: Invalid patch, keeping by default")
        kept.append((patch, {"level": level, "x": x, "y": y, "score": 0.0}))
        return kept, discarded

    s_stop, s_zoom = score[0], score[1]
    zoom_decision = env.infer_zoom_decision(s_stop, s_zoom)

    # --------------------
    # STOP
    # --------------------
    if zoom_decision == 0:
        kept.append((patch, {"level": level, "x": x, "y": y, "score": s_stop}))
        print("Score for STOP:", s_stop)
        return kept, discarded

    # --------------------
    # ZOOM
    # --------------------
    child_level = level - 1

    # ==================================================
    # TERMINAL SPLIT: reached min_level
    # ==================================================
    if child_level <= env.min_level or max_depth <= 0:
        print(f"Reached terminal level {child_level}, keeping all children")

        child_grids = env.wsi.get_child_grid(level, x, y)
        for grid in child_grids:
            for nx, ny in grid:
                try:
                    child_patch = env.wsi.get_patch(child_level, nx, ny)
                    kept.append(
                        (
                            child_patch,
                            {"level": child_level, "x": nx, "y": ny, "score": 0.0},
                        )
                    )
                except Exception:
                    continue

        # IMPORTANT: parent is NOT discarded here
        return kept, discarded

    # ==================================================
    # NORMAL (non-terminal) ZOOM
    # ==================================================
    discarded.append((patch, {"level": level, "x": x, "y": y, "score": s_zoom}))
    print("Score for ZOOM:", s_zoom)

    child_grids = env.wsi.get_child_grid(level, x, y)

    for grid in child_grids:
        for nx, ny in grid:
            k, d = greedy_infer_zoom(
                env,
                child_level,
                nx,
                ny,
                max_depth=max_depth - 1,
            )
            kept.extend(k)
            discarded.extend(d)

    return kept, discarded


def greedy_infer_wsi(
    image_path,
    max_depth=6,
    output_dir=None,
    score_module="text_align_score",
    viz_title=None,
):
    # ADDED COMMENT:
    # Entry point for whole-slide greedy inference. Iterates over all patches
    # at the coarsest level and applies greedy descent independently.
    print(f"Loading WSI: {image_path}")
    wsi = WSI(image_path)

    # ADDED COMMENT:
    # Greedy inference intentionally uses the same DynamicPatchEnv abstraction
    # as RL to ensure identical patch sampling, scaling, and scoring behavior.
    env = DynamicPatchEnv(wsi, patch_score=score_module)

    lvl = wsi.max_level
    width, height = wsi.levels_info[lvl]["size"]

    kept_all = []
    disc_all = []

    # ADDED COMMENT:
    # Independent greedy inference is run for each top-level patch.
    # There is no global budget or coordination between branches.
    for y in range(0, height, wsi.patch_size):
        for x in range(0, width, wsi.patch_size):
            k, d = greedy_infer_zoom(env, lvl, x, y, max_depth=max_depth)
            kept_all.extend(k)
            disc_all.extend(d)

    min_level = wsi.min_level

    min_w, min_h = wsi.levels_info[min_level]["size"]

    all_patches_count = (min_w // wsi.patch_size) * (min_h // wsi.patch_size)

    print(f"All patches:  {all_patches_count}")
    print(f"Kept patches: {len(kept_all)}")
    print(f"Discarded:    {all_patches_count - len(kept_all)}")

    # Visualization
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_html = os.path.join(output_dir, "visualization.html")
    else:
        out_html = f"data/visualizations/score_comparison/{score_module.replace('_', '-')}_viz.html"
        os.makedirs(os.path.dirname(out_html), exist_ok=True)

    wsi.active_patches.clear()
    wsi.zoomed_patches.clear()

    for _patch, meta in kept_all:
        key = (meta["level"], meta["x"], meta["y"])
        wsi.active_patches[key] = meta

    for _patch, meta in disc_all:
        key = (meta["level"], meta["x"], meta["y"])
        wsi.zoomed_patches[key] = meta

    wsi.visualize(
        output_html=out_html,
        image_label=viz_title,
        metadata={
            "Method": "Greedy",
            "Total patches": all_patches_count,
            "Kept patches": len(kept_all),
            "Discarded patches": len(disc_all),
        },
    )
    return kept_all, disc_all


if __name__ == "__main__":
    # ADDED COMMENT:
    # CLI wrapper for reproducible experimentation and debugging.
    parser = argparse.ArgumentParser(
        description="Greedy zoom inference based on score module"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="./data/to_test_image/test_img_1.svs",
        help="Path to .svs image",
    )
    parser.add_argument(
        "--max-depth", type=int, default=6, help="Maximum recursion depth"
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--score-module", type=str, default="text_align_score")
    parser.add_argument("--viz-title", type=str, default=None)
    args = parser.parse_args()

    greedy_infer_wsi(
        args.image,
        max_depth=args.max_depth,
        output_dir=args.output_dir,
        score_module=args.score_module,
        viz_title=args.viz_title,
    )

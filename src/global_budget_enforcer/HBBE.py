"""
HBBE — Hierarchical Balanced Budget Enforcer (multistage edition)

This module implements a balanced patch-selection algorithm for WSIs.

Key Features:
- Applies a depth-discount to refinement gain, encouraging balanced zoom depth.
- Uses the same WSI integration pattern as HUMBE (in-place mutation of active/zoomed patches).
- Stores STOP scores and metadata back on patches for downstream methods.
- Multistage: can be used as a first-stage selector before policy refinement.

Typical Usage:
    from src.utils.wsi import WSI
    from src.utils.patch_scores import PATCH_SCORE_MODULES
    from src.global_budget_enforcer.HBBE import hbbe

    wsi = WSI("slide.svs", multistage=True)
    score_module = PATCH_SCORE_MODULES["text_align_score"]()

    hbbe(
            wsi,
            score_module=score_module,
            budget_ratio=0.25,
            output_html="data/visualizations/hbbe.html",
            viz_metadata={"Image": "slide.svs"},
    )
"""

import sys
import math
import time
from pathlib import Path
import argparse

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.patch_scores import PATCH_SCORE_MODULES
from src.utils.wsi import WSI


# =============================================================================
# Core HBBE algorithm
# =============================================================================
def hbbe(
    wsi: WSI,
    score_module=None,
    budget_ratio: float = 0.3,
    batch_size: int = 8,
    log_every: int = 1,
    output_html: str | None = None,
    viz_metadata: dict | None = None,
) -> WSI:
    """
    Run HBBE on a ``WSI`` object (multistage=True), updating it in-place.

    Parameters:
        wsi (WSI): A freshly constructed (or reset) WSI instance .
            ``wsi.active_patches`` is expected to contain the flat root grid
        score_module (PatchScoreModule, optional): Scoring module with ``compute_stop`` and ``compute_zoom`` methods.
            Defaults to ``text_align_score``.
        budget_ratio (float): Fraction of total pyramid patches to retain (e.g. 0.25 = 25 %).
        batch_size (int): Max number of refinements applied per iteration.
        log_every (int): Print a progress line every N iterations.
        output_html (str, optional): If provided, path to save an HTML visualization of the final WSI state.
        viz_metadata (dict, optional): Extra key-value pairs forwarded to the visualizer header.

    Returns:
        WSI: The same WSI object, with ``active_patches`` and ``zoomed_patches`` updated to reflect the final selection.
    """

    if score_module is None:
        score_module = PATCH_SCORE_MODULES["text_align_score"]()

    t_start = time.time()

    # -------------------------
    # Track patch depths
    # -------------------------
    depth = {}

    # -------------------------
    # 1. Score root-level patches
    # -------------------------
    root_keys = list(wsi.active_patches.keys())

    # Iterate over root patches, compute and store their STOP scores in metadata.
    for lvl, x, y in root_keys:
        try:
            img = wsi.get_patch(lvl, x, y)
            s_stop = float(score_module.compute_stop(parent_patch=img))
        except Exception:
            s_stop = 0.0

        depth[(lvl, x, y)] = 0

        wsi.set_patch_metadata(lvl, x, y, {"score": s_stop, "zoomable": False})

    print(f"[HBBE] Initialized with {wsi.active_patch_count()} root patches")

    # -------------------------
    # 2. Compute budget
    # -------------------------
    total_patches = sum(
        sum(1 for _ in wsi.iterate_patches(lvl))
        for lvl, info in wsi.levels_info.items()
        if not info.get("frozen", False)
    )

    budget = math.floor(budget_ratio * total_patches)

    print(f"[HBBE] Total patches in pyramid : {total_patches}")
    print(f"[HBBE] Budget                   : {budget}")

    # Batched refinement loop
    #    Each iteration:
    #      a) Scan active patches for valid refinements.
    #      b) Rank candidates by depth-discounted efficiency.
    #      c) Apply top-K non-conflicting refinements via wsi.zoom_patch().
    iteration = 0

    # Iterate until we hit the budget or run out of candidates
    while wsi.active_patch_count() < budget:

        iteration += 1
        candidates = []

        # Snapshot current active keys so we can safely iterate while the
        # dict may be mutated later
        active_snapshot = list(wsi.active_patches.keys())

        # Iterate over active patches and score zoom candidates
        for lvl, x, y in active_snapshot:

            # Quit on root-level patches — no further zoom possible
            if lvl <= wsi.min_level:
                continue

            parent = (lvl, x, y)

            child_grids = wsi.get_child_grid(lvl, x, y)
            # Quit on patches with no children (e.g. leaves)
            if not child_grids:
                continue

            child_imgs = []
            child_coords = []

            for grid in child_grids:
                for cx, cy in grid:
                    try:
                        img_c = wsi.get_patch(lvl - 1, cx, cy)
                        child_imgs.append(img_c)
                        child_coords.append((lvl - 1, cx, cy))
                    except Exception:
                        continue

            # Quit on error — this parent is not a valid candidate
            if not child_imgs:
                continue

            try:
                parent_img = wsi.get_patch(lvl, x, y)
                s_stop = wsi.get_patch_metadata(lvl, x, y).get("score", 0.0)

                s_zoom = float(
                    score_module.compute_zoom(
                        parent_patch=parent_img,
                        child_patches=child_imgs,
                    )
                )
            except Exception:
                continue

            raw_gain = s_zoom - s_stop
            cost = len(child_imgs) - 1

            # Quit on non-positive gain/cost or budget overflow
            if raw_gain <= 0 or cost <= 0:
                continue
            if wsi.active_patch_count() + cost > budget:
                continue

            d = depth.get(parent, 0)

            # Depth-discounted gain
            discounted_gain = raw_gain / (1 + d)

            efficiency = discounted_gain / cost

            # Add candidate to list
            candidates.append(
                {
                    "parent": parent,
                    "children": child_coords,
                    "child_imgs": child_imgs,
                    "gain": discounted_gain,
                    "cost": cost,
                    "eff": efficiency,
                    "depth": d,
                }
            )

        # Quit if no candidates found
        if not candidates:
            print(f"[HBBE] No candidates at iter {iteration}, stopping.")
            break

        candidates.sort(key=lambda c: c["eff"], reverse=True)

        # Iterate over candidates in order of efficiency
        # Applying non-conflicting refinements until batch size is reached
        applied = 0
        for cand in candidates:

            if applied >= batch_size:
                break

            lvl, x, y = cand["parent"]
            cost = cand["cost"]

            # Quit on inactive parent
            if not wsi.is_active(lvl, x, y):
                continue

            # Quit on budget overflow
            if wsi.active_patch_count() + cost > budget:
                continue

            parent_depth = cand["depth"]

            # Apply zoom in-place on wsi
            wsi.zoom_patch(lvl, x, y)

            wsi.set_patch_metadata(lvl, x, y, {"zoomable": True})

            # Score and attach metadata to each new child
            for (c_lvl, cx, cy), img in zip(cand["children"], cand["child_imgs"]):

                if not wsi.is_active(c_lvl, cx, cy):
                    continue

                try:
                    sc = float(score_module.compute_stop(parent_patch=img))
                except Exception:
                    sc = 0.0

                depth[(c_lvl, cx, cy)] = parent_depth + 1

                wsi.set_patch_metadata(c_lvl, cx, cy, {"score": sc, "zoomable": False})

            applied += 1

        if iteration % log_every == 0:
            best_eff = candidates[0]["eff"]
            print(
                f"[HBBE][iter {iteration:03d}] "
                f"applied={applied} | "
                f"active={wsi.active_patch_count()}/{budget} | "
                f"zoomed={len(wsi.zoomed_patches)} | "
                f"best_eff={best_eff:.4f}"
            )

        if applied == 0:
            print(f"[HBBE] No refinements could be applied at iter {iteration}.")
            break

    elapsed = time.time() - t_start

    print(
        f"[HBBE] Done — {iteration} iterations | "
        f"active={wsi.active_patch_count()} | "
        f"zoomed={len(wsi.zoomed_patches)} | "
        f"elapsed={elapsed:.1f}s"
    )

    # -------------------------
    # Visualization
    # -------------------------
    if output_html is not None:

        score_name = getattr(score_module, "__class__", type(score_module)).__name__

        auto_meta = {
            "Method": "HBBE",
            "Score module": score_name,
            "Budget ratio": f"{budget_ratio:.0%}",
            "Active patches": str(wsi.active_patch_count()),
            "Zoomed patches": str(len(wsi.zoomed_patches)),
            "Elapsed": f"{elapsed:.1f}s",
        }

        if viz_metadata:
            auto_meta.update(viz_metadata)

        wsi.visualize(output_html=output_html, metadata=auto_meta)

    return wsi


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run HBBE on a WSI file.")
    parser.add_argument(
        "--image", default="data/to_test_image/test_img_1.svs", help="Path to .svs file"
    )
    parser.add_argument(
        "--budget", type=float, default=0.2, help="Budget ratio (default: 0.25)"
    )
    parser.add_argument(
        "--score",
        default="text_align_score",
        help="Score module key (default: text_align_score)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output",
        default="data/visualizations/hbbe.html",
        help="Output HTML visualization path",
    )
    args = parser.parse_args()

    # Run HBBE
    wsi = WSI(args.image, multistage=True)
    score_module = PATCH_SCORE_MODULES[args.score]()

    hbbe(
        wsi,
        score_module=score_module,
        budget_ratio=args.budget,
        batch_size=args.batch_size,
        output_html=args.output,
        viz_metadata={"Image": args.image},
    )

    print(f"\n[HBBE] Final active : {wsi.active_patch_count()} patches")
    print(f"[HBBE] Final zoomed : {len(wsi.zoomed_patches)} patches")

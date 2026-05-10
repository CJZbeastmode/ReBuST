"""
PQK — Priority Queue Knapsack (multistage edition)

This module implements a heap-based patch-selection algorithm for WSIs.

Key Features:
- Maintains a persistent max-heap of refinement candidates.
- Uses efficiency (gain / cost) as the queue priority.
- Applies zoom decisions directly via wsi.zoom_patch().
- Multistage: can be used as a first-stage selector before policy refinement.

Typical Usage:
    from src.utils.wsi import WSI
    from src.utils.patch_scores import PATCH_SCORE_MODULES
    from src.global_budget_enforcer.PQK import pqk

    wsi = WSI("slide.svs", multistage=True)
    score_module = PATCH_SCORE_MODULES["text_align_score"]()

    pqk(
        wsi,
        score_module=score_module,
        budget_ratio=0.25,
        output_html="data/visualizations/pqk.html",
        viz_metadata={"Image": "slide.svs"},
    )
"""

import sys
import math
import time
import heapq
from pathlib import Path
import argparse

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.patch_scores import PATCH_SCORE_MODULES
from src.utils.wsi import WSI


# =============================================================================
# Core PQK algorithm
# =============================================================================
def pqk(
    wsi: WSI,
    score_module=None,
    budget_ratio: float = 0.3,
    log_every: int = 1,
    output_html: str | None = None,
    viz_metadata: dict | None = None,
) -> WSI:
    """
    Run PQK on a WSI object, updating it in-place.

    Parameters:
        wsi (WSI): A freshly constructed (or reset) WSI instance .
            ``wsi.active_patches`` is expected to contain the flat root grid
        score_module (PatchScoreModule, optional): Scoring module with ``compute_stop`` and ``compute_zoom`` methods.
            Defaults to ``text_align_score``.
        budget_ratio (float): Fraction of total pyramid patches to retain (e.g. 0.25 = 25 %).
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
    # 1. Score root patches
    # -------------------------
    root_keys = list(wsi.active_patches.keys())

    # Iterate over root patches, compute and store their STOP scores in metadata.
    for lvl, x, y in root_keys:
        try:
            img = wsi.get_patch(lvl, x, y)
            s_stop = float(score_module.compute_stop(parent_patch=img))
        except Exception:
            s_stop = 0.0

        wsi.set_patch_metadata(lvl, x, y, {"score": s_stop, "zoomable": False})

    print(f"[PQK] Initialized with {wsi.active_patch_count()} root patches")

    # -------------------------
    # 2. Compute budget
    # -------------------------
    total_patches = sum(
        sum(1 for _ in wsi.iterate_patches(lvl))
        for lvl, info in wsi.levels_info.items()
        if not info.get("frozen", False)
    )

    budget = math.floor(budget_ratio * total_patches)

    print(f"[PQK] Total patches in pyramid : {total_patches}")
    print(f"[PQK] Budget                   : {budget}")

    # -------------------------
    # 3. Initialize priority queue
    # -------------------------
    pq = []
    counter = 0

    def compute_candidate(lvl, x, y):
        """Build one refinement candidate with gain/cost efficiency."""

        # Quit on root-level patches — no further zoom possible
        if lvl <= wsi.min_level:
            return None

        child_grids = wsi.get_child_grid(lvl, x, y)
        # Quit on patches with no children (e.g. leaves)
        if not child_grids:
            return None

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
            return None

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
            return None

        gain = s_zoom - s_stop
        cost = len(child_imgs) - 1

        # Quit on non-positive gain or cost
        if gain <= 0 or cost <= 0:
            return None

        eff = gain / cost

        return {
            "parent": (lvl, x, y),
            "children": child_coords,
            "child_imgs": child_imgs,
            "gain": gain,
            "cost": cost,
            "eff": eff,
        }

    # Push all root candidates
    for lvl, x, y in root_keys:
        cand = compute_candidate(lvl, x, y)
        if cand:
            heapq.heappush(pq, (-cand["eff"], counter, cand))
            counter += 1

    # Priority-queue refinement loop
    #    Each iteration:
    #      a) Pop the best available candidate by efficiency.
    #      b) Validate active/budget constraints.
    #      c) Apply zoom and push newly formed child candidates.
    # 4. Refinement loop
    iteration = 0

    # Iterate until queue is empty or budget is reached.
    while pq and wsi.active_patch_count() < budget:

        iteration += 1

        neg_eff, _, cand = heapq.heappop(pq)

        lvl, x, y = cand["parent"]
        cost = cand["cost"]

        # Skip stale candidates where parent is no longer active
        if not wsi.is_active(lvl, x, y):
            continue

        # Skip candidates that exceed remaining budget
        if wsi.active_patch_count() + cost > budget:
            continue

        # Apply zoom in-place on wsi
        wsi.zoom_patch(lvl, x, y)

        wsi.set_patch_metadata(lvl, x, y, {"zoomable": True})

        # Score and attach metadata to each new child
        for (c_lvl, cx, cy), img in zip(cand["children"], cand["child_imgs"]):

            # Quit on inactive child
            if not wsi.is_active(c_lvl, cx, cy):
                continue

            try:
                sc = float(score_module.compute_stop(parent_patch=img))
            except Exception:
                sc = 0.0

            wsi.set_patch_metadata(c_lvl, cx, cy, {"score": sc, "zoomable": False})

            # Push child candidate into the priority queue
            new_cand = compute_candidate(c_lvl, cx, cy)
            if new_cand:
                heapq.heappush(pq, (-new_cand["eff"], counter, new_cand))
                counter += 1

        if iteration % log_every == 0:
            best_eff = -neg_eff
            print(
                f"[PQK][iter {iteration:03d}] "
                f"active={wsi.active_patch_count()}/{budget} | "
                f"zoomed={len(wsi.zoomed_patches)} | "
                f"best_eff={best_eff:.4f}"
            )

    elapsed = time.time() - t_start

    print(
        f"[PQK] Done — {iteration} iterations | "
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
            "Method": "PQK",
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
    parser = argparse.ArgumentParser(description="Run PQK on a WSI file.")
    parser.add_argument(
        "--image", default="data/to_test_image/test_img_1.svs", help="Path to .svs file"
    )
    parser.add_argument(
        "--budget", type=float, default=0.05, help="Budget ratio (default: 0.25)"
    )
    parser.add_argument(
        "--score",
        default="text_align_score",
        help="Score module key (default: text_align_score)",
    )
    args = parser.parse_args()

    # Run PQK
    wsi = WSI(args.image, multistage=True)
    score_module = PATCH_SCORE_MODULES[args.score]()
    output = f"data/visualizations/pqk_{str(args.budget).replace('.', '_')}.html"

    pqk(
        wsi,
        score_module=score_module,
        budget_ratio=args.budget,
        output_html=output,
        viz_metadata={"Image": args.image},
    )

    print(f"\n[PQK] Final active : {wsi.active_patch_count()} patches")
    print(f"[PQK] Final zoomed : {len(wsi.zoomed_patches)} patches")

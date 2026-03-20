"""
PQK — Priority Queue Knapsack (multistage edition)

Approximate global budget enforcer using a priority queue over
refinement efficiency.

Differences from HUMBE
----------------------
* Uses a persistent max-heap of candidate refinements instead of
  scanning all active patches every iteration.
* Each candidate represents a possible zoom action.
* Efficiency = gain / cost (same objective as HUMBE).

Typical usage
-------------
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

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.patch_scores import PATCH_SCORE_MODULES
from src.utils.wsi import WSI


def pqk(
    wsi: WSI,
    score_module=None,
    budget_ratio: float = 0.3,
    log_every: int = 1,
    output_html: str | None = None,
    viz_metadata: dict | None = None,
) -> WSI:

    if score_module is None:
        score_module = PATCH_SCORE_MODULES["text_align_score"]()

    t_start = time.time()

    # --------------------------------------------------
    # 1. Score root patches
    # --------------------------------------------------
    root_keys = list(wsi.active_patches.keys())

    for lvl, x, y in root_keys:
        try:
            img = wsi.get_patch(lvl, x, y)
            s_stop = float(score_module.compute_stop(parent_patch=img))
        except Exception:
            s_stop = 0.0

        wsi.set_patch_metadata(lvl, x, y, {
            "score": s_stop,
            "zoomable": False
        })

    print(f"[PQK] Initialized with {wsi.active_patch_count()} root patches")

    # --------------------------------------------------
    # 2. Compute budget
    # --------------------------------------------------
    total_patches = sum(
        sum(1 for _ in wsi.iterate_patches(lvl))
        for lvl, info in wsi.levels_info.items()
        if not info.get("frozen", False)
    )

    budget = math.floor(budget_ratio * total_patches)

    print(f"[PQK] Total patches in pyramid : {total_patches}")
    print(f"[PQK] Budget                   : {budget}")

    # --------------------------------------------------
    # 3. Initialize priority queue
    # --------------------------------------------------
    pq = []
    counter = 0

    def compute_candidate(lvl, x, y):

        if lvl <= wsi.min_level:
            return None

        child_grids = wsi.get_child_grid(lvl, x, y)
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

        if not child_imgs:
            return None

        try:
            parent_img = wsi.get_patch(lvl, x, y)
            s_stop = wsi.get_patch_metadata(lvl, x, y).get("score", 0.0)

            s_zoom = float(score_module.compute_zoom(
                parent_patch=parent_img,
                child_patches=child_imgs,
            ))
        except Exception:
            return None

        gain = s_zoom - s_stop
        cost = len(child_imgs) - 1

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

    # push all root candidates
    for lvl, x, y in root_keys:
        cand = compute_candidate(lvl, x, y)
        if cand:
            heapq.heappush(pq, (-cand["eff"], counter, cand))
            counter += 1

    # --------------------------------------------------
    # 4. Refinement loop
    # --------------------------------------------------
    iteration = 0

    while pq and wsi.active_patch_count() < budget:

        iteration += 1

        neg_eff, _, cand = heapq.heappop(pq)

        lvl, x, y = cand["parent"]
        cost = cand["cost"]

        if not wsi.is_active(lvl, x, y):
            continue

        if wsi.active_patch_count() + cost > budget:
            continue

        # apply zoom
        wsi.zoom_patch(lvl, x, y)

        wsi.set_patch_metadata(lvl, x, y, {"zoomable": True})

        # score children
        for (c_lvl, cx, cy), img in zip(cand["children"], cand["child_imgs"]):

            if not wsi.is_active(c_lvl, cx, cy):
                continue

            try:
                sc = float(score_module.compute_stop(parent_patch=img))
            except Exception:
                sc = 0.0

            wsi.set_patch_metadata(c_lvl, cx, cy, {
                "score": sc,
                "zoomable": False
            })

            # push child candidate
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

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    if output_html is not None:

        score_name = getattr(score_module, '__class__', type(score_module)).__name__

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
# Standalone entry-point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PQK on a WSI file.")
    parser.add_argument("--image", default="data/to_test_image/test_img_1.svs", help="Path to .svs file")
    parser.add_argument("--budget", type=float, default=0.2,
                        help="Budget ratio (default: 0.25)")
    parser.add_argument("--score", default="text_align_score",
                        help="Score module key (default: text_align_score)")
    parser.add_argument("--output", default="data/visualizations/pqk.html",
                        help="Output HTML visualization path")
    args = parser.parse_args()

    wsi = WSI(args.image, multistage=True)
    score_module = PATCH_SCORE_MODULES[args.score]()

    pqk(
        wsi,
        score_module=score_module,
        budget_ratio=args.budget,
        output_html=args.output,
        viz_metadata={"Image": args.image},
    )

    print(f"\n[PQK] Final active : {wsi.active_patch_count()} patches")
    print(f"[PQK] Final zoomed : {len(wsi.zoomed_patches)} patches")

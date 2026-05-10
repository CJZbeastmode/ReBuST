"""
HUMBE — Batched Hierarchical Utility-Maximizing Budget Enforcer (multistage edition)

This module implements a patch-selection algorithm for WSIs.

Key Features:
- Zoom decisions applied via wsi.zoom_patch(), keeping active and zoomed patches authoritative.
- Stores STOP scores and metadata back on patches.
- Returns the mutated WSI object itself, ready for visualization.
- Multistage: can be run with multiple stages, e.g., HUMBE for initial selection followed by A2C for refinement.

Typical Usage:
    from src.utils.wsi import WSI
    from src.utils.patch_scores import PATCH_SCORE_MODULES
    from src.global_budget_enforcer.HUMBE import humbe

    wsi = WSI("slide.svs", multistage=True)
    score_module = PATCH_SCORE_MODULES["text_align_score"]()
    humbe(
        wsi,
        score_module=score_module,
        budget_ratio=0.25,
        output_html="data/visualizations/humbe.html",
        viz_metadata={"Image": "slide.svs"},
    )
    # or visualize manually afterwards:
    # wsi.visualize("out.html", metadata={"Method": "HUMBE"})
"""

# -------------------------
# 1, Imports and Path Setup
# -------------------------
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


# -------------------------
# 2, Core HUMBE Algorithm
# -------------------------
def humbe(
    wsi: WSI,
    score_module=None,
    budget_ratio: float = 0.3,
    batch_size: int = 64,
    log_every: int = 1,
    output_html: str | None = None,
    viz_metadata: dict | None = None,
    verbose: bool = True,
    viz_title: str | None = None,
) -> WSI:
    """
    Run HUMBE on a WSI object, updating it in-place.

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

    _vprint = print if verbose else lambda *a, **kw: None

    patch_img_cache: dict[tuple[int, int, int], object | None] = {}
    stop_score_cache: dict[tuple[int, int, int], float] = {}
    zoom_score_cache: dict[tuple, float] = {}

    max_patch_cache = 50000
    max_zoom_cache = 100000

    # -------------------------
    # 2.1, Helpers
    # -------------------------
    def _get_patch_cached(lvl: int, x: int, y: int):
        """Get patch image with caching, to speed up repeated access."""
        key = (lvl, x, y)
        if key in patch_img_cache:
            return patch_img_cache[key]
        try:
            img = wsi.get_patch(lvl, x, y)
        except Exception:
            img = None
        if len(patch_img_cache) >= max_patch_cache:
            patch_img_cache.clear()
        patch_img_cache[key] = img
        return img

    def _get_stop_score_cached(lvl: int, x: int, y: int, img=None) -> float:
        """Get STOP score with caching."""
        key = (lvl, x, y)
        if key in stop_score_cache:
            return stop_score_cache[key]
        if img is None:
            img = _get_patch_cached(lvl, x, y)
        if img is None:
            score = 0.0
        else:
            try:
                score = float(score_module.compute_stop(parent_patch=img))
            except Exception:
                score = 0.0
        stop_score_cache[key] = score
        return score

    # -------------------------
    # 3, Initialization & Budget Calculation
    # -------------------------
    t_start = time.time()

    root_keys = list(wsi.active_patches.keys())

    # Iterate over root patches, compute and store their STOP scores in metadata.
    for lvl, x, y in root_keys:
        s_stop = _get_stop_score_cached(lvl, x, y)
        # zoomable=False: this root patch was not zoomed by HUMBE.
        # A2C will not zoom it further — HUMBE already decided it is not
        # interesting enough to explore.  Only children that HUMBE actually
        # creates (in step 3) receive zoomable=True.
        wsi.set_patch_metadata(lvl, x, y, {"score": s_stop, "zoomable": False})

    _vprint(f"[HUMBE] Initialized with {wsi.active_patch_count()} root patches")

    # Compute budget (non-frozen patches × budget_ratio)
    total_patches = sum(
        sum(1 for _ in wsi.iterate_patches(lvl))
        for lvl, info in wsi.levels_info.items()
        if not info.get("frozen", False)
    )
    budget = math.floor(budget_ratio * total_patches)

    _vprint(f"[HUMBE] Total patches in pyramid : {total_patches}")
    _vprint(f"[HUMBE] Budget                   : {budget}")

    # -------------------------
    # 4, Batched Refinement Loop
    # -------------------------
    #    Each iteration:
    #      a) Scan wsi.active_patches for patches worth zooming.
    #      b) Rank candidates by gain / cost efficiency.
    #      c) Apply top-K non-conflicting refinements — each via
    #         wsi.zoom_patch(), which removes the parent and inserts children.
    iteration = 0

    # Iterate until we hit the budget or run out of candidates
    while wsi.active_patch_count() < budget:
        iteration += 1
        candidates = []

        # ---- collect candidates ----
        # Snapshot current active keys so we can safely iterate while the
        # dict may be mutated later
        active_snapshot = list(wsi.active_patches.keys())

        # Iterate over active patches and score zoom candidates
        for lvl, x, y in active_snapshot:
            # Quit on root-level patches — they have no parents and cannot be zoomed
            if lvl <= wsi.min_level:
                continue

            child_grids = wsi.get_child_grid(lvl, x, y)
            # Quit on patches with no children (e.g. leaves)
            if not child_grids:
                continue

            # Collect valid children
            child_imgs = []
            child_coords = []
            # Iterate over child grids
            for grid in child_grids:
                # Iterate over patches in this child grid
                for cx, cy in grid:
                    img_c = _get_patch_cached(lvl - 1, cx, cy)
                    # Quit on error — this child is not a valid candidate
                    if img_c is None:
                        continue
                    child_imgs.append(img_c)
                    child_coords.append((lvl - 1, cx, cy))

            # Quit on error — this parent is not a valid candidate
            if not child_imgs:
                continue

            # Compute efficiency
            parent_img = _get_patch_cached(lvl, x, y)
            # Quit on error — this parent is not a valid candidate
            if parent_img is None:
                continue

            # Get STOP and ZOOM scores with/without caching
            s_stop = _get_stop_score_cached(lvl, x, y, img=parent_img)
            zoom_key = ((lvl, x, y), tuple(child_coords))
            if zoom_key in zoom_score_cache:
                s_zoom = zoom_score_cache[zoom_key]
            else:
                try:
                    s_zoom = float(
                        score_module.compute_zoom(
                            parent_patch=parent_img,
                            child_patches=child_imgs,
                        )
                    )
                except Exception:
                    continue
                if len(zoom_score_cache) >= max_zoom_cache:
                    zoom_score_cache.clear()
                zoom_score_cache[zoom_key] = s_zoom

            # Compute gain and cost
            gain = s_zoom - s_stop
            cost = len(child_imgs) - 1

            # Quit on non-positive gain or cost
            if gain <= 0 or cost <= 0:
                continue
            # Quit on budget overflow
            if wsi.active_patch_count() + cost > budget:
                continue

            # Add candidate to list
            candidates.append(
                {
                    "parent": (lvl, x, y),
                    "children": child_coords,
                    "child_imgs": child_imgs,
                    "gain": gain,
                    "cost": cost,
                    "eff": gain / cost,
                }
            )

        # Quit if no candidates found
        if not candidates:
            _vprint(f"[HUMBE] No candidates at iter {iteration}, stopping.")
            break

        # Rank by efficiency (gain / cost)
        candidates.sort(key=lambda c: c["eff"], reverse=True)

        # Iterate over candidates in efficiency order
        applied = 0
        for cand in candidates:
            if applied >= batch_size:
                break

            lvl, x, y = cand["parent"]
            cost = cand["cost"]

            # Skip if this parent was already consumed by an earlier refinement in this batch
            if not wsi.is_active(lvl, x, y):
                continue
            # Skip if applying this refinement would exceed the budget
            if wsi.active_patch_count() + cost > budget:
                continue

            # Apply zoom in-place on wsi
            wsi.zoom_patch(lvl, x, y)

            # Mark parent as zoomed
            wsi.set_patch_metadata(lvl, x, y, {"zoomable": True})

            # Score and attach metadata to each new child
            for (c_lvl, cx, cy), img in zip(cand["children"], cand["child_imgs"]):
                if not wsi.is_active(c_lvl, cx, cy):
                    continue  # child fell outside padded bounds — already dropped
                sc = _get_stop_score_cached(c_lvl, cx, cy, img=img)
                # Children are HUMBE leaves: A2C should run its policy on each one
                wsi.set_patch_metadata(c_lvl, cx, cy, {"score": sc, "zoomable": False})

            applied += 1

        if iteration % log_every == 0:
            best_eff = candidates[0]["eff"]
            _vprint(
                f"[HUMBE][iter {iteration:03d}] "
                f"applied={applied} | "
                f"active={wsi.active_patch_count()}/{budget} | "
                f"zoomed={len(wsi.zoomed_patches)} | "
                f"best_eff={best_eff:.4f}"
            )

        if applied == 0:
            _vprint(f"[HUMBE] No refinements could be applied at iter {iteration}.")
            break

    # -------------------------
    # 5, Visualization & Output
    # -------------------------
    t_end = time.time()

    elapsed = t_end - t_start

    _vprint(
        f"[HUMBE] Done — {iteration} iterations | "
        f"active={wsi.active_patch_count()} | "
        f"zoomed={len(wsi.zoomed_patches)} | "
        f"elapsed={elapsed:.1f}s"
    )

    # Store HUMBE stats and visualize
    if output_html is not None:
        score_name = getattr(score_module, "__class__", type(score_module)).__name__
        auto_meta = {
            "Method": "HUMBE",
            "Score module": score_name,
            "Budget ratio": f"{budget_ratio:.0%}",
            "Active patches": str(wsi.active_patch_count()),
            "Zoomed patches": str(len(wsi.zoomed_patches)),
            "Elapsed": f"{elapsed:.1f}s",
        }
        if viz_metadata:
            auto_meta.update(viz_metadata)
        wsi.visualize(
            output_html=output_html, metadata=auto_meta, image_label=viz_title
        )

    return wsi


# -------------------------
# 6, Main Execution Block
# -------------------------

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run HUMBE on a WSI file.")
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--viz-title",
        type=str,
        default=None,
        help="Title of the output HTML visualization path",
    )

    args = parser.parse_args()

    # Run HUMBE
    wsi = WSI(args.image, multistage=True)
    score_module = PATCH_SCORE_MODULES[args.score]()
    output = f"data/visualizations/humbe/humbe_{str(args.budget).replace('.', '_')}.html"

    humbe(
        wsi,
        score_module=score_module,
        budget_ratio=args.budget,
        batch_size=args.batch_size,
        output_html=output,
        viz_metadata={"Image": args.image},
        viz_title=args.viz_title,
    )

    print(f"\n[HUMBE] Final active : {wsi.active_patch_count()} patches")
    print(f"[HUMBE] Final zoomed : {len(wsi.zoomed_patches)} patches")

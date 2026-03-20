"""
HUMBE — Batched Hierarchical Utility-Maximizing Budget Enforcer (multistage edition)

Differences from the original HUMBE
-------------------------------------
* Accepts a ``WSI`` object (with ``multistage=True``) and drives it in-place
  instead of maintaining a separate internal dict.
* Every zoom decision is applied via ``wsi.zoom_patch()``, which keeps
  ``wsi.active_patches`` and ``wsi.zoomed_patches`` authoritative
  throughout the algorithm — no separate bookkeeping structures are needed.
* Scores are stored back on each patch via ``wsi.set_patch_metadata()``.
* The return value is the **mutated ``wsi`` object itself**.  Call
  ``wsi.visualize()`` on it directly, or pass ``output_html`` to
  ``humbe`` to have visualization triggered automatically.

Typical usage
-------------
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
    # — or visualize manually afterwards:
    # wsi.visualize("out.html", metadata={"Method": "HUMBE"})
"""

import sys
import math
import time
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.patch_scores import PATCH_SCORE_MODULES
from src.utils.wsi import WSI


def humbe(
    wsi: WSI,
    score_module=None,
    budget_ratio: float = 0.3,
    batch_size: int = 8,
    log_every: int = 1,
    output_html: str | None = None,
    viz_metadata: dict | None = None,
    verbose: bool = True,
) -> WSI:
    """
    Run HUMBE on a ``WSI`` object (multistage=True), updating it in-place.

    Parameters
    ----------
    wsi : WSI
        A freshly constructed (or reset) WSI instance (multistage=True).
        ``wsi.active_patches`` is expected to contain the flat root grid
        (the default after ``WSI.__init__``).
    score_module : PatchScoreModule, optional
        Scoring module with ``compute_stop`` and ``compute_zoom`` methods.
        Defaults to ``text_align_score``.
    budget_ratio : float
        Fraction of total pyramid patches to retain (e.g. 0.25 = 25 %).
    batch_size : int
        Max number of refinements applied per iteration.
    log_every : int
        Print a progress line every N iterations.
    output_html : str, optional
        If provided, ``wsi.visualize()`` is called automatically after the
        algorithm finishes and the HTML is written to this path.
    viz_metadata : dict, optional
        Extra key-value pairs forwarded to the visualizer header.
        HUMBE-specific stats (budget, score module, elapsed time) are merged
        in automatically; this dict is for additional caller-supplied info.

    Returns
    -------
    WSI
        The same ``wsi`` object, with ``active_patches`` and
        ``zoomed_patches`` updated to reflect the final selection.
        Each entry in ``active_patches`` has its metadata set to
        ``{"score": float}`` (the STOP score of that patch).
    """

    if score_module is None:
        score_module = PATCH_SCORE_MODULES["text_align_score"]()

    _vprint = print if verbose else lambda *a, **kw: None

    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. Score all root-level patches already in wsi.active_patches
    #    and store each score in the patch metadata.
    # ------------------------------------------------------------------
    root_keys = list(wsi.active_patches.keys())   # snapshot — avoid iter+mutate
    for lvl, x, y in root_keys:
        try:
            img = wsi.get_patch(lvl, x, y)
            s_stop = float(score_module.compute_stop(parent_patch=img))
        except Exception:
            s_stop = 0.0
        # zoomable=False: this root patch was not zoomed by HUMBE.
        # A2C will not zoom it further — HUMBE already decided it is not
        # interesting enough to explore.  Only children that HUMBE actually
        # creates (in step 3) receive zoomable=True.
        wsi.set_patch_metadata(lvl, x, y, {"score": s_stop, "zoomable": False})

    _vprint(f"[HUMBE] Initialized with {wsi.active_patch_count()} root patches")

    # ------------------------------------------------------------------
    # 2. Compute budget  (total non-frozen patches × budget_ratio)
    # ------------------------------------------------------------------
    total_patches = sum(
        sum(1 for _ in wsi.iterate_patches(lvl))
        for lvl, info in wsi.levels_info.items()
        if not info.get("frozen", False)
    )
    budget = math.floor(budget_ratio * total_patches)

    _vprint(f"[HUMBE] Total patches in pyramid : {total_patches}")
    _vprint(f"[HUMBE] Budget                   : {budget}")

    # ------------------------------------------------------------------
    # 3. Batched refinement loop
    #
    #    Each iteration:
    #      a) Scan wsi.active_patches for patches worth zooming.
    #      b) Rank candidates by gain / cost efficiency.
    #      c) Apply top-K non-conflicting refinements — each via
    #         wsi.zoom_patch(), which atomically removes the parent
    #         and inserts children.
    # ------------------------------------------------------------------
    iteration = 0

    while wsi.active_patch_count() < budget:
        iteration += 1
        candidates = []

        # ---- collect candidates ----
        # Snapshot current active keys so we can safely iterate while the
        # dict may be mutated later (it is NOT mutated inside this loop).
        active_snapshot = list(wsi.active_patches.keys())

        for lvl, x, y in active_snapshot:
            if lvl <= wsi.min_level:
                continue

            child_grids = wsi.get_child_grid(lvl, x, y)
            if not child_grids:
                continue

            # Collect valid children
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
                continue

            # Compute efficiency
            try:
                parent_img = wsi.get_patch(lvl, x, y)
                s_stop = wsi.get_patch_metadata(lvl, x, y).get("score", 0.0)
                s_zoom = float(score_module.compute_zoom(
                    parent_patch=parent_img,
                    child_patches=child_imgs,
                ))
            except Exception:
                continue

            gain = s_zoom - s_stop
            # Cost = net extra patches: replace 1 parent with N children → N-1 extra.
            cost = len(child_imgs) - 1

            if gain <= 0 or cost <= 0:
                continue
            if wsi.active_patch_count() + cost > budget:
                continue

            candidates.append({
                "parent": (lvl, x, y),
                "children": child_coords,
                "child_imgs": child_imgs,
                "gain": gain,
                "cost": cost,
                "eff": gain / cost,
            })

        if not candidates:
            _vprint(f"[HUMBE] No candidates at iter {iteration}, stopping.")
            break

        # ---- rank by efficiency ----
        candidates.sort(key=lambda c: c["eff"], reverse=True)

        # ---- apply top-K non-conflicting refinements ----
        applied = 0

        for cand in candidates:
            if applied >= batch_size:
                break

            lvl, x, y = cand["parent"]
            cost = cand["cost"]

            # Skip if this parent was already consumed by an earlier refinement
            # in the same iteration batch.
            if not wsi.is_active(lvl, x, y):
                continue
            if wsi.active_patch_count() + cost > budget:
                continue

            # ---- apply zoom in-place on wsi ----
            wsi.zoom_patch(lvl, x, y)

            # Parent is now in zoomed_patches — mark it as zoomed.
            # (score was already written in step 1; just flip the flag)
            wsi.set_patch_metadata(lvl, x, y, {"zoomable": True})

            # ---- score and attach metadata to each new child ----
            for (c_lvl, cx, cy), img in zip(cand["children"], cand["child_imgs"]):
                if not wsi.is_active(c_lvl, cx, cy):
                    continue   # child fell outside padded bounds — already dropped
                try:
                    sc = float(score_module.compute_stop(parent_patch=img))
                except Exception:
                    sc = 0.0
                # Children are HUMBE leaves: A2C should run its policy on each one
                # (zoom further or stop). Setting zoomable=True allows Guard 3 to
                # pass and the model to run.
                wsi.set_patch_metadata(c_lvl, cx, cy, {"score": sc, "zoomable": False})

            applied += 1

        # ---- progress logging ----
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

    t_end = time.time()

    elapsed = t_end - t_start

    _vprint(
        f"[HUMBE] Done — {iteration} iterations | "
        f"active={wsi.active_patch_count()} | "
        f"zoomed={len(wsi.zoomed_patches)} | "
        f"elapsed={elapsed:.1f}s"
    )

    if output_html is not None:
        score_name = getattr(score_module, '__class__', type(score_module)).__name__
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
        wsi.visualize(output_html=output_html, metadata=auto_meta)

    return wsi


# =============================================================================
# Standalone entry-point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run HUMBE on a WSI file.")
    parser.add_argument("--image", default="data/to_test_image/test_img_1.svs", help="Path to .svs file")
    parser.add_argument("--budget", type=float, default=0.2,
                        help="Budget ratio (default: 0.25)")
    parser.add_argument("--score", default="text_align_score",
                        help="Score module key (default: text_align_score)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="data/visualizations/humbe.html",
                        help="Output HTML visualization path")
    args = parser.parse_args()

    wsi = WSI(args.image, multistage=True)
    score_module = PATCH_SCORE_MODULES[args.score]()

    humbe(
        wsi,
        score_module=score_module,
        budget_ratio=args.budget,
        batch_size=args.batch_size,
        output_html=args.output,
        viz_metadata={"Image": args.image},
    )

    print(f"\n[HUMBE] Final active : {wsi.active_patch_count()} patches")
    print(f"[HUMBE] Final zoomed : {len(wsi.zoomed_patches)} patches")

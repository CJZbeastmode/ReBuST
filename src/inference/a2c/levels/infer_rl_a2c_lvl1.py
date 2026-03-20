"""
infer_rl_a2c.py — A2C inference on a WSI object (multistage mode)

Differences from infer_rl_a2c.py
----------------------------------
* Accepts a ``WSI`` object (with ``multistage=True``, output of HUMBE)
  instead of a raw image path.  The HUMBE active-patch set is the starting
  point for the RL agent — each entry in ``wsi.active_patches`` is treated
  as one episode root.
* Zoom decisions are applied **in-place** via ``wsi.zoom_patch()``, which
  atomically updates ``wsi.active_patches`` and ``wsi.zoomed_patches``.
  No separate kept / zoomed bookkeeping is required.
* Per-patch zoom depth is bounded by ``wsi.min_level``.  Patches already at
  ``min_level`` are always STOPped immediately (no zoom action is possible).
* ``if __name__ == "__main__"`` is intentionally left as ``pass`` and will be
  implemented once the rest of the pipeline is in place.
"""

import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[4])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch
from torch.distributions import Categorical

from src.utils.wsi import WSI

# Shared model components — imported from infer_rl_a2c_lvl1.
# The checkpoint at data/models/rl/a2c_lvl1/a2c_lvl1_final.pt was trained
# with the lvl1 ActorCritic: 518-D input (515 base + 3 history features),
# 256-D hidden, no parent_proj.
from backup.infer_a2c.infer_rl_a2c_lvl1 import (  # noqa: E402
    HistoryTracker,
    encode_state,
    ActorCritic,
    load_a2c_model,
    _run_policy,
)


# ============================================================
# Per-patch recursive inference (operates in-place on wsi)
# ============================================================

@torch.no_grad()
def infer_patch_a2c(
    wsi: WSI,
    model: ActorCritic,
    history: HistoryTracker,
    level: int,
    x: int,
    y: int,
    device: torch.device,
    deterministic: bool = True,
    _depth_budget: int | None = None,
) -> None:
    """
    Recursively apply the A2C policy to a HUMBE-selected patch.

    On the first call ``_depth_budget`` is ``None`` and is read from the
    patch's HUMBE metadata (``"max_depth"``).  Each ZOOM step decrements the
    budget by 1 and recurses into every child.  When the budget reaches 0 the
    agent is forced to STOP.

    ``max_depth`` is set by HUMBE to ``leaf_level - min_level``: the
    **remaining** zoom capacity below each HUMBE leaf.  A patch that HUMBE
    left near ``min_level`` has little budget; a root patch that HUMBE never
    touched has the full pyramid depth available to A2C.

    Example (max_depth = remaining levels to min_level)::

        After HUMBE:  patch 1: max_depth 4  patch 2: max_depth 1
                      patch 3: max_depth 2  patch 4: max_depth 4

        Possible A2C: patch 1: stays at 4   patch 2: zooms to depth 3
                      patch 3: zooms to depth 3  patch 4: stays at 4
                      (patch 2: 2 extra zooms ≤ budget 1? only if budget ≥ 2)

        Impossible:   any patch ending at a SHALLOWER depth than HUMBE left
                      it — the architecture makes un-zooming impossible.

    Parameters
    ----------
    wsi : WSI
        In-place state object.
    model : ActorCritic
        Loaded policy network.
    history : HistoryTracker
        Shared episode history (tracks parent embeddings, depth, visited).
    level, x, y : int
        Coordinates of the patch to process.
    device : torch.device
    deterministic : bool
        If True use argmax, otherwise sample from the policy distribution.
    _depth_budget : int | None
        Remaining zoom budget.  Pass ``None`` on the first (external) call;
        it will be initialised from the patch's ``"max_depth"`` metadata field
        written by HUMBE.  Subsequent recursive calls pass the decremented
        value directly.
    """
    # ------------------------------------------------------------------
    # Guard 1 — patch might have been displaced by a sibling's zoom
    # ------------------------------------------------------------------
    if not wsi.is_active(level, x, y):
        return

    # ------------------------------------------------------------------
    # Guard 2 — already at the finest level; STOP unconditionally
    # ------------------------------------------------------------------
    if level <= wsi.min_level:
        print(f"[MIN-LEVEL-STOP] lvl={level} x={x} y={y}")
        return

    # ------------------------------------------------------------------
    # Guard 3 — initialise or check the per-patch HUMBE depth budget.
    #
    # On the first call _depth_budget is None; read it from the metadata
    # that HUMBE wrote (``"max_depth"`` = max_level - leaf_level).
    # Subsequent recursive calls pass the already-decremented value.
    # When the budget hits 0 the agent must STOP — the A2C is not allowed
    # to zoom farther than HUMBE originally decided for this patch.
    # ------------------------------------------------------------------
    if _depth_budget is None:
        meta = wsi.get_patch_metadata(level, x, y)
        # Fallback: if HUMBE hasn't set max_depth, use the full remaining
        # pyramid depth for this patch (level - min_level).
        _depth_budget = int(meta.get("max_depth", level - wsi.min_level))

    if _depth_budget <= 0:
        print(f"[BUDGET-STOP] lvl={level} x={x} y={y} (budget exhausted)")
        wsi.set_patch_metadata(level, x, y, {"score": 0.0})
        return

    # ------------------------------------------------------------------
    # Run policy  (state encoding + model forward handled by _run_policy)
    # ------------------------------------------------------------------
    action, p_stop, p_zoom, v = _run_policy(
        wsi, model, history, level, x, y, device, deterministic
    )

    # ------------------------------------------------------------------
    # STOP
    # ------------------------------------------------------------------
    if action == 0:
        print(
            f"[STOP] lvl={level} x={x} y={y} "
            f"p_stop={p_stop:.4f} p_zoom={p_zoom:.4f} v={v:.4f}"
        )
        wsi.set_patch_metadata(level, x, y, {"score": float(p_stop)})
        return

    # ------------------------------------------------------------------
    # ZOOM
    # ------------------------------------------------------------------
    child_level = level - 1
    print(
        f"[ZOOM] lvl={level} \u2192 {child_level} "
        f"x={x} y={y} p_stop={p_stop:.4f} p_zoom={p_zoom:.4f} v={v:.4f} "
        f"depth_budget={_depth_budget}\u2192{_depth_budget - 1}"
    )

    try:
        children = wsi.zoom_patch(level, x, y)   # mutates wsi in-place
    except (KeyError, ValueError) as e:
        print(f"[ZOOM-FAIL] lvl={level} x={x} y={y} :: {e}")
        return

    # Recurse into each child with the decremented depth budget
    child_depth_budget = _depth_budget - 1
    for (c_lvl, cx, cy) in children:
        infer_patch_a2c(
            wsi, model, history, c_lvl, cx, cy, device, deterministic,
            _depth_budget=child_depth_budget,
        )


# ============================================================
# Whole-WSI inference
# ============================================================

def infer_wsi_a2c(
    wsi: WSI,
    model_path: str,
    output_html: str | None = None,
    deterministic: bool = True,
    viz_metadata: dict | None = None,
) -> WSI:
    """
    Run A2C inference over all active patches in ``wsi``.

    HUMBE is assumed to have already populated ``wsi.active_patches``
    before this function is called.  Each active patch is treated as one
    episode root; the policy is applied recursively, zooming in-place.

    Parameters
    ----------
    wsi : WSI
        WSI object (multistage=True) with HUMBE output loaded into active_patches.
    model_path : str
        Path to the saved A2C checkpoint.
    output_html : str, optional
        If provided, ``wsi.visualize()`` is called after inference.
    deterministic : bool
        Argmax policy if True; sampled otherwise.
    viz_metadata : dict, optional
        Extra key-value pairs forwarded to the visualization header.

    Returns
    -------
    WSI
        The same ``wsi`` object, updated in-place.
    """
    device = torch.device(
        "cuda"  if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )

    model   = load_a2c_model(model_path, device)
    history = HistoryTracker()  # shared across all patches in this WSI

    # The HUMBE seed count is informational only; A2C may grow it via zooming.
    seed_count = wsi.active_patch_count()

    print(f"[INFER] Starting — {seed_count} seed patches")
    print(f"[INFER] Levels: max={wsi.max_level}  min={wsi.min_level}")

    # Snapshot the seed set; zoom_patch mutates active_patches during iteration
    seed_patches = list(wsi.active_patches.keys())

    for idx, (lvl, x, y) in enumerate(seed_patches):
        if idx % 50 == 0:
            print(f"[INFER] {idx}/{len(seed_patches)} seeds processed "
                  f"(active={wsi.active_patch_count()})")

        infer_patch_a2c(
            wsi, model, history, lvl, x, y, device, deterministic,
        )

    print(
        f"[INFER] Done — active={wsi.active_patch_count()} "
        f"zoomed={len(wsi.zoomed_patches)}"
    )

    if output_html is not None:
        meta = {
            "Method":        "HUMBE + A2C",
            "Model":         str(Path(model_path).name),
            "Policy":        "deterministic" if deterministic else "stochastic",
            "Active patches": str(wsi.active_patch_count()),
            "Zoomed patches": str(len(wsi.zoomed_patches)),
        }
        if viz_metadata:
            meta.update(viz_metadata)
        wsi.visualize(output_html, metadata=meta)

    return wsi


# ============================================================
# CLI entry-point  (to be implemented)
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="A2C Level-1 inference — works for both single-stage and multistage (HUMBE) modes"
    )
    parser.add_argument("--image",     type=str, default="data/to_test_image/test_img_1.svs")
    parser.add_argument("--model",     type=str, default="data/models/rl/a2c_lvl1/a2c_lvl1_final.pt")
    parser.add_argument("--output-viz-path", type=str,
                        default="data/visualizations/rl/a2c/viz_a2c_lvl1.html")
    parser.add_argument("--multistage", action="store_true",
                        help="Use multistage mode (requires HUMBE pre-processing). "
                             "Default: single-stage (forward recursive zoom from max_level).")
    parser.add_argument("--stochastic", action="store_true")

    args = parser.parse_args()

    wsi = WSI(args.image, multistage=args.multistage)

    infer_wsi_a2c(
        wsi,
        model_path=args.model,
        output_html=args.output_viz_path,
        deterministic=not args.stochastic,
    )
"""
sandbox.py — End-to-end pipeline: HUMBE → A2C refinement → Visualization

Pipeline
--------
1. Load WSI from a test slide.
2. Run HUMBE (global budget enforcer) — coarse patch selection, in-place.
3. Run A2C inference (infer_rl_a2c) — local refinement, in-place.
4. Visualize the final patch state as an HTML overlay.

Run from the repo root:
    python src/sandbox.py
    python src/sandbox.py --image data/to_test_image/test_img_1.svs \
                          --budget 0.15 \
                          --model  data/models/rl/a2c_baseline/a2c_baseline_final.pt
"""

import sys
import argparse
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.wsi import WSI
from src.inference.a2c.infer_rl_a2c import infer_wsi_a2c


# ============================================================
# Defaults — change here or override via CLI flags
# ============================================================

DEFAULT_IMAGE = "data/to_test_image/test_img_1.svs"
DEFAULT_MODEL = "data/models/rl/a2c/a2c.pt"
DEFAULT_BUDGET = 0.5  # fraction of total pyramid patches to keep after HUMBE_B
DEFAULT_SCORE = "text_align_score"
DEFAULT_MIN_LEVEL = 0  # finest level A2C/HUMBE may zoom into (0 = native full res)
DEFAULT_OUTFILE = "data/visualizations/pipelines/viz_a2c.html"


# ============================================================
# Pipeline
# ============================================================


def run_pipeline(
    image_path: str,
    model_path: str,
    budget_ratio: float = DEFAULT_BUDGET,
    score_key: str = DEFAULT_SCORE,
    min_level: int = DEFAULT_MIN_LEVEL,
    out: str = DEFAULT_OUTFILE,
    deterministic: bool = True,
) -> WSI:
    """
    Full pipeline: HUMBE coarse selection → A2C local refinement → visualization.

    Parameters
    ----------
    image_path    : str   Path to the .svs slide file.
    model_path    : str   Path to the A2C checkpoint (.pt).
    budget_ratio  : float HUMBE patch budget (fraction of full pyramid).
    score_key     : str   Key into PATCH_SCORE_MODULES for HUMBE_B scoring.
    min_level     : int   Finest pyramid level A2C/HUMBE may zoom into
                          (default 2 — avoids native full-resolution level 0).
    out_humbe     : str   HTML visualization path after HUMBE_B.
    out_final     : str   HTML visualization path after A2C refinement.
    deterministic : bool  Argmax policy if True; sampled otherwise.

    Returns
    -------
    WSI  The fully updated WSI object.
    """

    # ------------------------------------------------------------------
    # Step 1 — Load slide
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"[SANDBOX] Image : {image_path}")
    print(f"[SANDBOX] Model : {model_path}")
    print(f"[SANDBOX] Budget: {budget_ratio:.0%}")
    print("=" * 60)

    wsi = WSI(image_path)

    print(f"[SANDBOX] Levels  : max={wsi.max_level}  min={wsi.min_level}")
    print(f"[SANDBOX] Root patches at max_level: {wsi.active_patch_count()}")

    # ------------------------------------------------------------------
    # Step 3 — A2C: local per-patch refinement
    # ------------------------------------------------------------------
    print("\n[SANDBOX] ── Step 3: A2C refinement ───────────────────────")

    wsi = infer_wsi_a2c(
        wsi,
        model_path=model_path,
        output_html=out,
        deterministic=deterministic,
        viz_metadata={
            "Image": Path(image_path).name,
            "Budget": f"{budget_ratio:.0%}",
            "Stage": "after A2C Level-4",
        },
    )

    print(
        f"[SANDBOX] After A2C      — active={wsi.active_patch_count()}  "
        f"zoomed={len(wsi.zoomed_patches)}"
    )
    print(f"[SANDBOX] Final viz    → {out}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n[SANDBOX] Pipeline complete.")
    return wsi


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A2C refinement → visualization sandbox"
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Path to .svs slide (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Path to A2C checkpoint (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_BUDGET,
        help=f"HUMBE_B budget ratio (default: {DEFAULT_BUDGET})",
    )
    parser.add_argument(
        "--score",
        default=DEFAULT_SCORE,
        help=f"Patch score module key (default: {DEFAULT_SCORE})",
    )
    parser.add_argument("--out", default=DEFAULT_OUTFILE, help="HTML output path")
    parser.add_argument(
        "--min-level",
        type=int,
        default=DEFAULT_MIN_LEVEL,
        help=f"Finest zoom level allowed (default: {DEFAULT_MIN_LEVEL}, 0 = native full res)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling (default: deterministic argmax)",
    )
    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        model_path=args.model,
        budget_ratio=args.budget,
        score_key=args.score,
        min_level=args.min_level,
        out=args.out,
        deterministic=not args.stochastic,
    )

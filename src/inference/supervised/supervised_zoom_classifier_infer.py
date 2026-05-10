"""
Greedy inference using a trained Zoom Classifier
"""

import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import os
import torch
import webbrowser

from src.utils.wsi import WSI
from src.utils.dynamic_patch_env import DynamicPatchEnv
from src.training.supervised.zoom_classifier import ZoomClassifier


# -------------------------
# Model loading
# -------------------------


def load_regressor(model_path, device, state_dim=None):
    if state_dim is None:
        state_dim = 515
    model = ZoomClassifier(state_dim=state_dim, hidden=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(
        f"Loaded regressor from '{model_path}' "
        f"on device={device} (state_dim={state_dim})"
    )
    return model


# -------------------------
# Single-patch greedy zoom
# -------------------------


@torch.no_grad()
def greedy_infer_zoom_regressor(
    env,
    model,
    level,
    x,
    y,
    max_depth=6,
    device="cpu",
):
    kept, discarded = [], []

    patch = env.wsi.get_patch(level, x, y)
    print(f"Eval patch: level={level} x={x} y={y} max_depth={max_depth}")

    try:
        state = env.encode_state(patch, lvl=level, x=x, y=y)
    except Exception:
        kept.append((patch, {"level": level, "x": x, "y": y, "score": 0.0}))
        return kept, discarded

    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    s_stop, s_zoom = model(s).squeeze(0).tolist()
    zoom_decision = env.infer_zoom_decision(s_stop, s_zoom)

    # -------------------------
    # STOP
    # -------------------------
    if zoom_decision == 0:
        print(
            f"  Decision: STOP at level={level} x={x} y={y} " f"(s_stop={s_stop:.4f})"
        )
        kept.append((patch, {"level": level, "x": x, "y": y, "score": s_stop}))
        return kept, discarded

    # -------------------------
    # ZOOM
    # -------------------------
    print(f"  Decision: ZOOM at level={level} x={x} y={y} " f"(s_zoom={s_zoom:.4f})")

    child_level = level - 1

    # --------------------------------------------------
    # TERMINAL SPLIT: zoom reaches min_level
    # --------------------------------------------------
    if child_level == env.min_level:
        print(f"  TERMINAL: reached min_level={env.min_level} " f"from level={level}")

        child_grids = env.wsi.get_child_grid(level, x, y)

        for grid in child_grids:
            for nx, ny in grid:
                try:
                    child_patch = env.wsi.get_patch(child_level, nx, ny)
                    print(f"    KEEP MIN: lvl={child_level} x={nx} y={ny}")
                    kept.append(
                        (
                            child_patch,
                            {
                                "level": child_level,
                                "x": nx,
                                "y": ny,
                                "score": 0.0,
                            },
                        )
                    )
                except Exception:
                    print(f"    FAILED MIN PATCH x={nx} y={ny}")
                    continue

        # Parent is NOT discarded here
        return kept, discarded

    # --------------------------------------------------
    # NORMAL (non-terminal) ZOOM
    # --------------------------------------------------
    discarded.append((patch, {"level": level, "x": x, "y": y, "score": s_zoom}))

    child_grids = env.wsi.get_child_grid(level, x, y)

    for grid in child_grids:
        for nx, ny in grid:
            k, d = greedy_infer_zoom_regressor(
                env,
                model,
                child_level,
                nx,
                ny,
                max_depth=max_depth - 1,
                device=device,
            )
            kept.extend(k)
            discarded.extend(d)

    return kept, discarded


# -------------------------
# Whole-slide inference
# -------------------------


def greedy_infer_wsi_regressor(
    image_path,
    model_path,
    output_viz_path,
    max_depth=6,
    output_dir=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading WSI: {image_path}")
    wsi = WSI(image_path)
    env = DynamicPatchEnv(wsi)

    lvl = wsi.max_level
    width, height = wsi.levels_info[lvl]["size"]

    try:
        sample_patch = wsi.get_patch(lvl, 0, 0)
        state_dim = len(env.encode_state(sample_patch, lvl=lvl, x=0, y=0))
    except Exception:
        state_dim = 515

    model = load_regressor(model_path, device, state_dim=state_dim)

    kept_all, disc_all = [], []

    for y in range(0, height, wsi.patch_size):
        print(f"Scanning row y={y}/{height}")
        for x in range(0, width, wsi.patch_size):
            k, d = greedy_infer_zoom_regressor(
                env,
                model,
                lvl,
                x,
                y,
                max_depth=max_depth,
                device=device,
            )
            kept_all.extend(k)
            disc_all.extend(d)

    # --------------------------------------------------
    # REQUIRED METRIC (exactly as requested)
    # --------------------------------------------------
    min_lvl = env.min_level
    min_w, min_h = wsi.levels_info[min_lvl]["size"]

    total_min_patches = (min_w // wsi.patch_size) * (min_h // wsi.patch_size)

    kept_count = len(kept_all)
    kept_ratio = kept_count / max(1, total_min_patches)

    print("======================================")
    print(f"Min-level patches (ALL): {total_min_patches}")
    print(f"Kept patches (ALL):      {kept_count}")
    print(f"Kept / Min-level ratio:  {kept_ratio:.6f}")
    print("======================================")

    # Rebuild active_patches / zoomed_patches from inference results
    wsi.active_patches.clear()
    wsi.zoomed_patches.clear()
    for _patch, meta in kept_all:
        key = (meta["level"], meta["x"], meta["y"])
        wsi.active_patches[key] = {"score": meta.get("score", 0.0)}
    for _patch, meta in disc_all:
        key = (meta["level"], meta["x"], meta["y"])
        wsi.zoomed_patches[key] = {"score": meta.get("score", 0.0)}

    os.makedirs(os.path.dirname(os.path.abspath(output_viz_path)), exist_ok=True)
    out_html = wsi.visualize(
        output_html=output_viz_path,
        metadata={
            "Method": "Supervised Zoom Classifier",
            "Model": str(Path(model_path).name),
            "Kept patches": str(kept_count),
            "Kept/min ratio": f"{kept_ratio:.4f}",
        },
    )

    webbrowser.open(f"file://{os.path.abspath(out_html)}")
    return kept_all, disc_all


# -------------------------
# CLI (UNCHANGED)
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Greedy zoom inference using Score Regressor"
    )
    parser.add_argument(
        "--image", type=str, default="data/to_test_image/test_img_1.svs"
    )
    parser.add_argument(
        "--model", type=str, default="data/models/supervised/zoom_classifier.pth"
    )
    parser.add_argument(
        "--output-viz-path",
        type=str,
        default="data/visualizations/supervised/zoom_classifier/visualization.html",
    )
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    greedy_infer_wsi_regressor(
        args.image,
        args.model,
        args.output_viz_path,
        max_depth=args.max_depth,
        output_dir=args.output_dir,
    )

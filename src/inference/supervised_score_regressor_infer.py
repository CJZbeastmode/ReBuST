"""
Greedy inference using a trained Score Regressor
"""

import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import os
import torch
import webbrowser

from src.utils.wsi import WSI
from src.utils.dynamic_patch_env import DynamicPatchEnv
from src.training.supervised.score_regressor import ScoreRegressor


# ============================================================
# Model loading
# ============================================================


def load_regressor(model_path, device, state_dim=515):
    model = ScoreRegressor(state_dim=state_dim, hidden=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[MODEL] Loaded {model_path} on {device}")
    return model


# ============================================================
# Recursive greedy inference (FULL LOGGING)
# ============================================================


@torch.no_grad()
def greedy_infer_zoom_regressor(
    env,
    model,
    level,
    x,
    y,
    max_depth,
    device,
):
    kept, discarded = [], []

    print(f"[VISIT] lvl={level} x={x} y={y} depth={max_depth}")

    # --------------------------------------------------------
    # Hard termination guards
    # --------------------------------------------------------
    if max_depth <= 0:
        print(f"[DEPTH-STOP] lvl={level} x={x} y={y}")
        try:
            patch = env.wsi.get_patch(level, x, y)
            kept.append((patch, {"level": level, "x": x, "y": y, "score": 0.0}))
        except Exception:
            pass
        return kept, discarded

    if level < env.min_level or level > env.max_level:
        print(f"[OUT-OF-RANGE] lvl={level}")
        return kept, discarded

    # --------------------------------------------------------
    # Read patch
    # --------------------------------------------------------
    try:
        patch = env.wsi.get_patch(level, x, y)
    except Exception as e:
        print(f"[READ-FAIL] lvl={level} x={x} y={y} :: {e}")
        return kept, discarded

    # --------------------------------------------------------
    # Encode + forward
    # --------------------------------------------------------
    try:
        state = env.encode_state(patch, lvl=level, x=x, y=y)
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        s_stop, s_zoom = model(s).squeeze(0).tolist()
    except Exception as e:
        print(f"[MODEL-FAIL] lvl={level} x={x} y={y} :: {e}")
        kept.append((patch, {"level": level, "x": x, "y": y, "score": 0.0}))
        return kept, discarded

    zoom_decision = env.infer_zoom_decision(s_stop, s_zoom)

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------
    if zoom_decision == 0:
        print(
            f"[STOP] lvl={level} x={x} y={y} "
            f"s_stop={s_stop:.4f} s_zoom={s_zoom:.4f}"
        )
        kept.append(
            (
                patch,
                {
                    "level": level,
                    "x": x,
                    "y": y,
                    "score": float(s_stop),
                },
            )
        )
        return kept, discarded

    # --------------------------------------------------------
    # ZOOM
    # --------------------------------------------------------
    child_level = level - 1
    print(
        f"[ZOOM] lvl={level} -> {child_level} "
        f"x={x} y={y} s_stop={s_stop:.4f} s_zoom={s_zoom:.4f}"
    )

    # --------------------------------------------------------
    # TERMINAL: reaching min_level
    # --------------------------------------------------------
    if child_level == env.min_level:
        print(
            f"[TERMINAL] reached min_level={env.min_level} "
            f"from lvl={level} at x={x} y={y}"
        )

        child_grids = env.wsi.get_child_grid(level, x, y)
        for grid in child_grids:
            for nx, ny in grid:
                try:
                    cp = env.wsi.get_patch(child_level, nx, ny)
                    print(f"  [KEEP-MIN] lvl={child_level} x={nx} y={ny}")
                    kept.append(
                        (
                            cp,
                            {
                                "level": child_level,
                                "x": nx,
                                "y": ny,
                                "score": 0.0,
                            },
                        )
                    )
                except Exception:
                    print(f"  [MIN-READ-FAIL] x={nx} y={ny}")
                    continue

        # Parent is NOT discarded at terminal split
        return kept, discarded

    # --------------------------------------------------------
    # Normal recursive zoom
    # --------------------------------------------------------
    discarded.append(
        (
            patch,
            {
                "level": level,
                "x": x,
                "y": y,
                "score": float(s_zoom),
            },
        )
    )

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


# ============================================================
# Whole-slide inference + accounting
# ============================================================


def greedy_infer_wsi_regressor(
    image_path,
    model_path,
    output_viz_path,
    max_depth,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[WSI] Loading {image_path}")
    wsi = WSI(image_path)
    env = DynamicPatchEnv(wsi)

    print(f"[LEVELS] max_level={env.max_level} min_level={env.min_level}")

    lvl = env.max_level
    width, height = wsi.levels_info[lvl]["size"]

    model = load_regressor(model_path, device)

    kept_all, disc_all = [], []

    for y in range(0, height, wsi.patch_size):
        print(f"[ROW] y={y}/{height}")
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

    # --------------------------------------------------------
    # CORRECT METRIC (your definition)
    # --------------------------------------------------------
    min_lvl = env.min_level
    min_w, min_h = wsi.levels_info[min_lvl]["size"]

    total_min_level_patches = (min_w // wsi.patch_size) * (min_h // wsi.patch_size)

    kept_count = len(kept_all)
    kept_ratio = kept_count / max(1, total_min_level_patches)

    print("==========================================")
    print(f"[METRIC] Min-level patches (ALL): {total_min_level_patches}")
    print(f"[METRIC] Kept patches (ALL):      {kept_count}")
    print(f"[METRIC] Kept / Min-level ratio:  {kept_ratio:.6f}")
    print("==========================================")

    # --------------------------------------------------------
    # Visualization via wsi.visualize()
    # --------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_viz_path)), exist_ok=True)

    # Rebuild active_patches / zoomed_patches from inference results
    wsi.active_patches.clear()
    wsi.zoomed_patches.clear()
    for _patch, meta in kept_all:
        key = (meta["level"], meta["x"], meta["y"])
        wsi.active_patches[key] = {"score": meta.get("score", 0.0)}
    for _patch, meta in disc_all:
        key = (meta["level"], meta["x"], meta["y"])
        wsi.zoomed_patches[key] = {"score": meta.get("score", 0.0)}

    out_html = wsi.visualize(
        output_html=output_viz_path,
        metadata={
            "Method":           "Supervised Score Regressor",
            "Model":            str(Path(model_path).name),
            "Kept patches":     str(kept_count),
            "Kept/min ratio":   f"{kept_ratio:.4f}",
        },
    )

    webbrowser.open(f"file://{os.path.abspath(out_html)}")

    return kept_all, disc_all, kept_ratio


# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Greedy zoom inference using Score Regressor"
    )
    parser.add_argument(
        "--image", type=str, default="data/to_test_image/test_img_1.svs"
    )
    parser.add_argument("--model", type=str, default="data/models/supervised/score_regressor.pth")
    parser.add_argument(
        "--output-viz-path",
        type=str,
        default="data/visualizations/supervised/score_regressor/visualization.html",
    )
    parser.add_argument("--max-depth", type=int, default=6)

    args = parser.parse_args()

    greedy_infer_wsi_regressor(
        args.image,
        args.model,
        args.output_viz_path,
        max_depth=args.max_depth,
    )

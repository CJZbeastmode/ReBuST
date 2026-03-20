"""
Greedy inference using trained DQN (Q-learning) model
"""

import sys
from pathlib import Path


repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import os
import torch
import torch.nn as nn

from src.utils.wsi import WSI
from src.utils.dynamic_patch_env import DynamicPatchEnv
import webbrowser


# ============================================================
# Q Network (same as training)
# ============================================================
class QNet(nn.Module):
    """
    Q(s) -> [Q(stop), Q(zoom)]
    """

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        # FIXED: Add LayerNorm like A2C for more stable training
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        # Orthogonal init
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class LegacyQNet(nn.Module):
    """Older Q-network variant without `LayerNorm`.

    Some saved checkpoints in this repo were trained before the normalization
    layer was added, so inference must support both layouts.
    """

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Model loading
# ============================================================


def load_qnet_model(model_path, device, state_dim=515):
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Newer checkpoints use:
    #   net.0 = Linear, net.1 = LayerNorm, net.3 = Linear
    # Older checkpoints use:
    #   net.0 = Linear, net.1 = ReLU,      net.2 = Linear
    uses_layernorm = "net.1.weight" in state_dict and "net.3.weight" in state_dict

    if uses_layernorm:
        model = QNet(state_dim=state_dim)
        variant = "layernorm"
    else:
        model = LegacyQNet(state_dim=state_dim)
        variant = "legacy"

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[MODEL] Loaded Q-Network ({variant}) from {model_path} on {device}")
    return model


# ============================================================
# Recursive greedy inference
# ============================================================


@torch.no_grad()
def greedy_infer_q_learning(
    env,
    model,
    level,
    x,
    y,
    max_depth,
    device,
):
    """
    Recursively apply Q-learning policy to decide STOP or ZOOM.
    
    Returns:
        kept: list of (patch, metadata) tuples for kept patches
        discarded: list of (patch, metadata) tuples for discarded patches
    """
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
    # Encode + forward through Q-Network
    # --------------------------------------------------------
    try:
        state = env.encode_state(patch, lvl=level, x=x, y=y)
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model(s)
        
        # Get action (greedy: argmax)
        action = q_values.argmax(dim=1).item()
        
        # Get Q-values for logging
        q_stop, q_zoom = q_values.squeeze().tolist()
        
    except Exception as e:
        print(f"[MODEL-FAIL] lvl={level} x={x} y={y} :: {e}")
        kept.append((patch, {"level": level, "x": x, "y": y, "score": 0.0}))
        return kept, discarded

    # --------------------------------------------------------
    # STOP (action=0)
    # --------------------------------------------------------
    if action == 0:
        print(
            f"[STOP] lvl={level} x={x} y={y} "
            f"Q(stop)={q_stop:.4f} Q(zoom)={q_zoom:.4f}"
        )
        kept.append(
            (
                patch,
                {
                    "level": level,
                    "x": x,
                    "y": y,
                    "score": float(q_stop),
                },
            )
        )
        return kept, discarded

    # --------------------------------------------------------
    # ZOOM (action=1)
    # --------------------------------------------------------
    child_level = level - 1
    print(
        f"[ZOOM] lvl={level} -> {child_level} "
        f"x={x} y={y} Q(stop)={q_stop:.4f} Q(zoom)={q_zoom:.4f}"
    )

    # --------------------------------------------------------
    # Check if child_level is below min_level
    # --------------------------------------------------------
    if child_level < env.min_level:
        print(f"[BELOW-MIN] child_level={child_level} < min_level={env.min_level}, keeping current patch")
        kept.append(
            (
                patch,
                {
                    "level": level,
                    "x": x,
                    "y": y,
                    "score": float(q_stop),
                },
            )
        )
        return kept, discarded

    # --------------------------------------------------------
    # TERMINAL: reaching min_level
    # --------------------------------------------------------
    if child_level == env.min_level:
        print(
            f"[TERMINAL] reached min_level={env.min_level} "
            f"from lvl={level} at x={x} y={y}"
        )

        # Discard the parent patch since we're zooming
        discarded.append(
            (
                patch,
                {
                    "level": level,
                    "x": x,
                    "y": y,
                    "score": float(q_zoom),
                },
            )
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
                "score": float(q_zoom),
            },
        )
    )

    child_grids = env.wsi.get_child_grid(level, x, y)
    for grid in child_grids:
        for nx, ny in grid:
            k, d = greedy_infer_q_learning(
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


def greedy_infer_wsi_q_learning(
    image_path,
    model_path,
    output_viz_path,
    max_depth,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"[WSI] Loading {image_path}")
    wsi = WSI(image_path)
    env = DynamicPatchEnv(wsi, patch_score="text_align_score")

    print(f"[LEVELS] max_level={env.max_level} min_level={env.min_level}")

    lvl = env.max_level
    width, height = wsi.levels_info[lvl]["size"]

    model = load_qnet_model(model_path, device)

    kept_all, disc_all = [], []

    for y in range(0, height, wsi.patch_size):
        print(f"[ROW] y={y}/{height}")
        for x in range(0, width, wsi.patch_size):
            k, d = greedy_infer_q_learning(
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
            "Method":         "Q-learning (DQN)",
            "Model":          str(Path(model_path).name),
            "Kept patches":   str(kept_count),
            "Kept/min ratio": f"{kept_ratio:.4f}",
        },
    )

    webbrowser.open(f"file://{os.path.abspath(out_html)}")

    return kept_all, disc_all, kept_ratio


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Greedy zoom inference using trained DQN (Q-learning) model"
    )
    parser.add_argument(
        "--image", type=str, default="data/to_test_image/test_img_1.svs"
    )
    parser.add_argument("--model", type=str, default="data/models/dqn.pt")
    parser.add_argument(
        "--output-viz-path",
        type=str,
        default="data/visualizations/rl/q_learning/viz_orig.html",
    )
    parser.add_argument("--max-depth", type=int, default=6)

    args = parser.parse_args()

    greedy_infer_wsi_q_learning(
        args.image,
        args.model,
        args.output_viz_path,
        max_depth=args.max_depth,
    )

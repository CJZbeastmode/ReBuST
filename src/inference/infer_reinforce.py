"""
Greedy inference using trained REINFORCE model
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
from torch.distributions import Categorical
import webbrowser

from src.utils.wsi import WSI
from src.utils.dynamic_patch_env import DynamicPatchEnv
from src.utils.patched_image import PatchedImage


# ============================================================
# REINFORCE Policy Network (same as training)
# ============================================================
class PolicyWithBaseline(nn.Module):
    """
    REINFORCE policy with learned baseline.
    """

    def __init__(self, state_dim=515, hidden_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_dim, 2)     # STOP / ZOOM
        self.baseline = nn.Linear(hidden_dim, 1)  # b(s)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        nn.init.orthogonal_(self.baseline.weight, gain=1.0)
        nn.init.zeros_(self.baseline.bias)

    def forward(self, x):
        h = self.encoder(x)
        logits = self.actor(h)
        baseline = self.baseline(h).squeeze(-1)
        return logits, baseline


# ============================================================
# Model loading
# ============================================================


def load_reinforce_model(model_path, device, state_dim=515):
    model = PolicyWithBaseline(state_dim=state_dim, hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[MODEL] Loaded REINFORCE from {model_path} on {device}")
    return model


# ============================================================
# Recursive greedy inference
# ============================================================


@torch.no_grad()
def greedy_infer_reinforce(
    env,
    model,
    level,
    x,
    y,
    max_depth,
    device,
    deterministic=True,
):
    """
    Recursively apply REINFORCE policy to decide STOP or ZOOM.
    
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
    # Encode + forward through REINFORCE policy
    # --------------------------------------------------------
    try:
        state = env.encode_state(patch, lvl=level, x=x, y=y)
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, baseline = model(s)
        
        # Get action
        if deterministic:
            action = logits.argmax(dim=1).item()
        else:
            dist = Categorical(logits=logits)
            action = dist.sample().item()
        
        # Get probabilities for logging
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        p_stop, p_zoom = probs[0], probs[1]
        b = baseline.item()
        
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
            f"p_stop={p_stop:.4f} p_zoom={p_zoom:.4f} baseline={b:.4f}"
        )
        kept.append(
            (
                patch,
                {
                    "level": level,
                    "x": x,
                    "y": y,
                    "score": float(p_stop),
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
        f"x={x} y={y} p_stop={p_stop:.4f} p_zoom={p_zoom:.4f} baseline={b:.4f}"
    )

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
                    "score": float(p_zoom),
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
                "score": float(p_zoom),
            },
        )
    )

    child_grids = env.wsi.get_child_grid(level, x, y)
    for grid in child_grids:
        for nx, ny in grid:
            k, d = greedy_infer_reinforce(
                env,
                model,
                child_level,
                nx,
                ny,
                max_depth=max_depth - 1,
                device=device,
                deterministic=deterministic,
            )
            kept.extend(k)
            discarded.extend(d)

    return kept, discarded


# ============================================================
# Whole-slide inference + accounting
# ============================================================


def greedy_infer_wsi_reinforce(
    image_path,
    model_path,
    output_viz_path,
    max_depth,
    deterministic=True,
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

    model = load_reinforce_model(model_path, device)

    kept_all, disc_all = [], []

    for y in range(0, height, wsi.patch_size):
        print(f"[ROW] y={y}/{height}")
        for x in range(0, width, wsi.patch_size):
            k, d = greedy_infer_reinforce(
                env,
                model,
                lvl,
                x,
                y,
                max_depth=max_depth,
                device=device,
                deterministic=deterministic,
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
    # Create PatchedImage and visualize
    # --------------------------------------------------------
    # Extract coordinates from kept and discarded patches
    kept_coords = [(meta["level"], meta["x"], meta["y"]) for _, meta in kept_all]
    zoomed_coords = [(meta["level"], meta["x"], meta["y"]) for _, meta in disc_all]
    
    # Create PatchedImage object
    patched_image = PatchedImage(wsi, kept_coords, zoomed_coords)
    
    # Generate visualization
    os.makedirs(os.path.dirname(output_viz_path) or ".", exist_ok=True)
    out_html = patched_image.generate_visualization(output_html=output_viz_path)
    
    # Open in browser
    webbrowser.open(f"file://{os.path.abspath(out_html)}")

    return kept_all, disc_all, kept_ratio


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Greedy zoom inference using trained REINFORCE model"
    )
    parser.add_argument(
        "--image", type=str, default="data/to_test_image/test_img_1.svs"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="data/models/rl/reinforce/reinforce_epoch_6.pt"
    )
    parser.add_argument(
        "--output-viz-path",
        type=str,
        default="data/visualizations/rl/reinforce/viz_reinforce.html",
    )
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument(
        "--stochastic", 
        action="store_true", 
        help="Use stochastic policy (sample actions)"
    )

    args = parser.parse_args()

    greedy_infer_wsi_reinforce(
        args.image,
        args.model,
        args.output_viz_path,
        max_depth=args.max_depth,
        deterministic=not args.stochastic,
    )

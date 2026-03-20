"""
Greedy inference using trained A2C Level 2 model (Redundancy Avoidance)
"""

import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[4])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import webbrowser

from src.utils.wsi import WSI
from src.utils.dynamic_patch_env import DynamicPatchEnv
from src.utils.patched_image import PatchedImage

OVERLAP_THRESHOLD = 0.3


# ============================================================
# History Tracker (Level 2 - with redundancy detection)
# ============================================================
class HistoryTracker:
    """Track visited patches and detect redundancy during inference."""
    
    def __init__(self, grid_size=32):
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        """Reset history at episode start."""
        self.visited = {}
        self.last_action = 0
        self.depth = 0
        self.visited_locations = []
    
    def _hash_location(self, level, x, y):
        """Hash continuous coordinates to discrete grid."""
        x_bin = int(x * self.grid_size)
        y_bin = int(y * self.grid_size)
        return (level, x_bin, y_bin)
    
    def visit(self, level, x, y):
        """Record a visit to a location."""
        key = self._hash_location(level, x, y)
        self.visited[key] = self.visited.get(key, 0) + 1
        self.visited_locations.append((level, x, y))
        return self.visited[key]
    
    def get_visit_count(self, level, x, y):
        """Get visit count for a location."""
        key = self._hash_location(level, x, y)
        return self.visited.get(key, 0)
    
    def update_action(self, action):
        """Update the last action taken."""
        self.last_action = action
        if action == 1:  # ZOOM
            self.depth += 1
    
    def compute_redundancy_score(self, level, x, y, threshold=OVERLAP_THRESHOLD):
        """Compute redundancy score based on spatial overlap."""
        if len(self.visited_locations) == 0:
            return 0.0
        
        same_level_visits = [
            (l, vx, vy) for (l, vx, vy) in self.visited_locations 
            if abs(l - level) < 0.1
        ]
        
        if len(same_level_visits) == 0:
            return 0.0
        
        distances = []
        for (_, vx, vy) in same_level_visits:
            dist = np.sqrt((x - vx)**2 + (y - vy)**2)
            distances.append(dist)
        
        overlaps = sum(1 for d in distances if d < threshold)
        redundancy = min(overlaps / max(1, len(same_level_visits)), 1.0)
        
        return redundancy
    
    def compute_overlap_penalty(self, level, x, y):
        """Compute explicit penalty for being too close to visited regions."""
        if len(self.visited_locations) < 2:
            return 0.0
        
        recent_visits = self.visited_locations[-5:]
        
        for (l, vx, vy) in recent_visits:
            if abs(l - level) < 0.1:
                dist = np.sqrt((x - vx)**2 + (y - vy)**2)
                if dist < OVERLAP_THRESHOLD * 0.5:
                    return 1.0
                elif dist < OVERLAP_THRESHOLD:
                    return 0.5
        
        return 0.0
    
    def get_history_features(self, level, x, y):
        """Extract history features including redundancy."""
        visit_count = self.get_visit_count(level, x, y)
        visit_count_norm = min(visit_count, 5) / 5.0
        
        redundancy_score = self.compute_redundancy_score(level, x, y)
        overlap_penalty = self.compute_overlap_penalty(level, x, y)
        
        return np.array([
            visit_count_norm,
            float(self.last_action),
            self.depth / 10.0,
            redundancy_score,
            overlap_penalty,
        ], dtype=np.float32)


def get_history_aware_state(env_state, history, level_norm, x_norm, y_norm):
    """Augment environment state with history and redundancy features."""
    history_features = history.get_history_features(level_norm, x_norm, y_norm)
    return np.concatenate([env_state, history_features])


# ============================================================
# Actor-Critic Network (Level 2 - state_dim=520)
# ============================================================
class ActorCritic(nn.Module):
    """Level 2 Actor-Critic with state_dim=520."""

    def __init__(self, state_dim=520, hidden_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_dim, 2)
        self.critic = nn.Linear(hidden_dim, 1)

        nn.init.orthogonal_(self.actor.weight, gain=0.1)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.actor.bias)
        self.actor.bias.data[1] = 0.1
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        h = self.encoder(x)
        return self.actor(h), self.critic(h)


# ============================================================
# Model loading
# ============================================================
def load_a2c_model(model_path, device, state_dim=520):
    model = ActorCritic(state_dim=state_dim, hidden_dim=256)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"[MODEL] Loaded A2C Level 2 from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"[MODEL] Loaded from {model_path} on {device}")
    print(f"[MODEL] State dimension: {state_dim}")
    return model


# ============================================================
# Recursive greedy inference with redundancy awareness
# ============================================================
@torch.no_grad()
def greedy_infer_a2c(
    env,
    model,
    history,
    level,
    x,
    y,
    max_depth,
    device,
    deterministic=True,
):
    """Recursively apply A2C Level 2 policy with redundancy avoidance."""
    kept, discarded = [], []

    print(f"[VISIT] lvl={level} x={x} y={y} depth={max_depth}")

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

    try:
        patch = env.wsi.get_patch(level, x, y)
    except Exception as e:
        print(f"[READ-FAIL] lvl={level} x={x} y={y} :: {e}")
        return kept, discarded

    history.visit(level, x, y)

    try:
        env_state = env.encode_state(patch, lvl=level, x=x, y=y)
        
        level_norm = env_state[0]
        x_norm = env_state[1]
        y_norm = env_state[2]
        state = get_history_aware_state(env_state, history, level_norm, x_norm, y_norm)
        
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = model(s)
        
        if deterministic:
            action = logits.argmax(dim=1).item()
        else:
            dist = Categorical(logits=logits)
            action = dist.sample().item()
        
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        p_stop, p_zoom = probs[0], probs[1]
        v = value.item()
        
        # Get redundancy info for logging
        redundancy = history.compute_redundancy_score(level_norm, x_norm, y_norm)
        
    except Exception as e:
        print(f"[MODEL-FAIL] lvl={level} x={x} y={y} :: {e}")
        kept.append((patch, {"level": level, "x": x, "y": y, "score": 0.0}))
        return kept, discarded

    history.update_action(action)

    if action == 0:  # STOP
        print(
            f"[STOP] lvl={level} x={x} y={y} "
            f"p_stop={p_stop:.4f} p_zoom={p_zoom:.4f} v={v:.4f} redund={redundancy:.3f}"
        )
        kept.append((patch, {"level": level, "x": x, "y": y, "score": float(p_stop)}))
        return kept, discarded

    child_level = level - 1
    print(
        f"[ZOOM] lvl={level} -> {child_level} "
        f"x={x} y={y} p_stop={p_stop:.4f} p_zoom={p_zoom:.4f} v={v:.4f} redund={redundancy:.3f}"
    )

    if child_level == env.min_level:
        print(f"[TERMINAL] reached min_level={env.min_level}")
        discarded.append((patch, {"level": level, "x": x, "y": y, "score": float(p_zoom)}))

        child_grids = env.wsi.get_child_grid(level, x, y)
        for grid in child_grids:
            for nx, ny in grid:
                try:
                    cp = env.wsi.get_patch(child_level, nx, ny)
                    print(f"  [KEEP-MIN] lvl={child_level} x={nx} y={ny}")
                    kept.append((cp, {"level": child_level, "x": nx, "y": ny, "score": 0.0}))
                except Exception:
                    continue

        return kept, discarded

    discarded.append((patch, {"level": level, "x": x, "y": y, "score": float(p_zoom)}))

    child_grids = env.wsi.get_child_grid(level, x, y)
    for grid in child_grids:
        for nx, ny in grid:
            k, d = greedy_infer_a2c(
                env, model, history, child_level, nx, ny,
                max_depth=max_depth - 1, device=device, deterministic=deterministic
            )
            kept.extend(k)
            discarded.extend(d)

    return kept, discarded


# ============================================================
# Whole-slide inference
# ============================================================
def greedy_infer_wsi_a2c(image_path, model_path, output_viz_path, max_depth, deterministic=True):
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

    model = load_a2c_model(model_path, device, state_dim=520)

    history = HistoryTracker(grid_size=32)

    kept_all, disc_all = [], []

    for y in range(0, height, wsi.patch_size):
        print(f"[ROW] y={y}/{height}")
        for x in range(0, width, wsi.patch_size):
            k, d = greedy_infer_a2c(
                env, model, history, lvl, x, y,
                max_depth=max_depth, device=device, deterministic=deterministic
            )
            kept_all.extend(k)
            disc_all.extend(d)

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

    kept_coords = [(meta["level"], meta["x"], meta["y"]) for _, meta in kept_all]
    zoomed_coords = [(meta["level"], meta["x"], meta["y"]) for _, meta in disc_all]
    
    patched_image = PatchedImage(wsi, kept_coords, zoomed_coords)
    
    os.makedirs(os.path.dirname(output_viz_path) or ".", exist_ok=True)
    out_html = patched_image.generate_visualization(output_html=output_viz_path)
    
    webbrowser.open(f"file://{os.path.abspath(out_html)}")

    return kept_all, disc_all, kept_ratio


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with A2C Level 2 (Redundancy Avoidance)")
    parser.add_argument("--image", type=str, default="data/to_test_image/test_img_1.svs")
    parser.add_argument("--model", type=str, default="data/models/rl/a2c_lvl2/a2c_lvl2_final.pt")
    parser.add_argument("--output-viz-path", type=str, default="data/visualizations/rl/a2c/viz_a2c_lvl2.html")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--stochastic", action="store_true")

    args = parser.parse_args()

    greedy_infer_wsi_a2c(
        args.image, args.model, args.output_viz_path,
        max_depth=args.max_depth, deterministic=not args.stochastic
    )

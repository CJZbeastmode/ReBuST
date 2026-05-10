"""
infer_rl_a2c.py — A2C Level-4 inference on a WSI object (multistage mode)

Mirrors infer_rl_a2c.py with two differences:
  1. Operates on WSI with multistage=True (HUMBE pre-processed) instead of WSI + DynamicPatchEnv.
  2. Zoom permission is controlled per-patch by the ``zoomable`` flag in
     each patch's metadata (set by HUMBE), only when ``multistage=True``:
       zoomable=True  → A2C may zoom this patch.
       zoomable=False → A2C must stop here (patch is kept as-is).
     When ``multistage=False`` (single-stage mode), the ``zoomable`` flag is
     never stored or consulted — every patch is unconditionally eligible;
     use ``wsi.is_zoomable(lvl, x, y)`` to query zoom eligibility.

Additional constraint: A2C never zooms past min_level.  If child_level
would fall below min_level the patch is kept.  When child_level exactly
equals min_level (TERMINAL), all children are kept without running the
policy on them.

Everything else — HistoryTracker, ActorCritic, model loading,
state encoding, get_hierarchical_aware_state — is identical to the
original file.
"""

import sys
from pathlib import Path
import argparse

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.utils.wsi import WSI

OVERLAP_THRESHOLD = 0.3


# ============================================================
# History Tracker  (identical to infer_rl_a2c.py)
# ============================================================


class HistoryTracker:
    """Track visited patches, redundancy, and hierarchical context during inference."""

    def __init__(self, grid_size=32):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        """Reset history at episode start."""
        self.visited = {}
        self.last_action = 0
        self.depth = 0
        self.visited_locations = []
        self.parent_embedding = None
        self.current_embedding = None
        self.has_parent = False

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

    def update_action(self, action, current_embedding=None):
        """Update the last action taken and hierarchical context."""
        self.last_action = action

        if action == 1:  # ZOOM
            self.depth += 1
            if current_embedding is not None:
                self.parent_embedding = current_embedding.copy()
                self.has_parent = True

        if current_embedding is not None:
            self.current_embedding = current_embedding.copy()

    def compute_redundancy_score(self, level, x, y, threshold=OVERLAP_THRESHOLD):
        """Compute redundancy score based on spatial overlap."""
        if len(self.visited_locations) == 0:
            return 0.0

        same_level_visits = [
            (l, vx, vy)
            for (l, vx, vy) in self.visited_locations
            if abs(l - level) < 0.1
        ]

        if len(same_level_visits) == 0:
            return 0.0

        distances = []
        for _, vx, vy in same_level_visits:
            dist = np.sqrt((x - vx) ** 2 + (y - vy) ** 2)
            distances.append(dist)

        overlaps = sum(1 for d in distances if d < threshold)
        redundancy = min(overlaps / max(1, len(same_level_visits)), 1.0)

        return redundancy

    def compute_overlap_penalty(self, level, x, y):
        """Compute explicit penalty for being too close to visited regions."""
        if len(self.visited_locations) < 2:
            return 0.0

        recent_visits = self.visited_locations[-5:]

        for l, vx, vy in recent_visits:
            if abs(l - level) < 0.1:
                dist = np.sqrt((x - vx) ** 2 + (y - vy) ** 2)
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

        return np.array(
            [
                visit_count_norm,
                float(self.last_action),
                self.depth / 10.0,
                redundancy_score,
                overlap_penalty,
            ],
            dtype=np.float32,
        )

    def get_hierarchical_features(self):
        """Extract hierarchical context features."""
        if self.has_parent and self.parent_embedding is not None:
            return self.parent_embedding.copy(), 1.0
        else:
            return np.zeros(512, dtype=np.float32), 0.0


def get_hierarchical_aware_state(env_state, history, level_norm, x_norm, y_norm):
    """Augment environment state with history and hierarchical context features.
    Identical to infer_rl_a2c.py."""
    history_features = history.get_history_features(level_norm, x_norm, y_norm)
    parent_emb, has_parent = history.get_hierarchical_features()

    return np.concatenate([env_state, history_features, parent_emb, [has_parent]])


# ============================================================
# State encoding  (replaces DynamicPatchEnv.encode_state)
# ============================================================


def encode_state(wsi: WSI, embedder, patch, level: int, x: int, y: int) -> np.ndarray:
    """
    Replicate DynamicPatchEnv.encode_state for WSI (multistage mode).
    Returns a 515-D vector: [level_norm, x_norm, y_norm, emb(512)].
    """
    W, H = wsi.levels_info[level]["size"]
    level_norm = level / max(wsi.max_level, 1)
    x_norm = x / max(W, 1)
    y_norm = y / max(H, 1)

    try:
        emb = embedder.img_emb(patch)
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy().flatten()
        else:
            emb = np.asarray(emb, dtype=np.float32).flatten()
        if len(emb) != 512 or np.any(~np.isfinite(emb)):
            raise ValueError("bad embedding")
    except Exception:
        emb = np.zeros(512, dtype=np.float32)

    return np.concatenate([[level_norm, x_norm, y_norm], emb]).astype(np.float32)


# ============================================================
# Actor-Critic Network  (identical to infer_rl_a2c.py)
# ============================================================


class ActorCritic(nn.Module):
    """Level 4 Actor-Critic with parent embedding projection (same as Level 3).

    Input state: 1033-D [base(520) + parent_emb(512) + has_parent(1)]
    After projection: 585-D [base(520) + parent_proj(64) + has_parent(1)]
    """

    def __init__(
        self, base_state_dim=520, parent_emb_dim=512, parent_proj_dim=64, hidden_dim=512
    ):
        super().__init__()

        # Project parent embedding 512-D → 64-D
        self.parent_proj = nn.Sequential(
            nn.Linear(parent_emb_dim, parent_proj_dim),
            nn.ReLU(),
        )

        # Final state dimension after projection
        final_state_dim = base_state_dim + parent_proj_dim + 1  # 520 + 64 + 1 = 585

        self.encoder = nn.Sequential(
            nn.Linear(final_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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
        # Split state: [base(520), parent_emb(512), has_parent(1)]
        base_state = x[:, :520]
        parent_emb = x[:, 520:1032]
        has_parent = x[:, 1032:1033]

        # Project parent embedding 512-D → 64-D
        parent_proj = self.parent_proj(parent_emb)

        # Concatenate: [520, 64, 1] = 585-D
        x_proj = torch.cat([base_state, parent_proj, has_parent], dim=1)

        h = self.encoder(x_proj)
        return self.actor(h), self.critic(h)


# ============================================================
# Model loading  (identical to infer_rl_a2c.py)
# ============================================================


def load_a2c_model(model_path, device, verbose=True):
    _vprint = print if verbose else lambda *a, **kw: None
    model = ActorCritic(
        base_state_dim=520, parent_emb_dim=512, parent_proj_dim=64, hidden_dim=512
    )
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "unknown")
        gae_lambda = checkpoint.get("gae_lambda", "unknown")
        _vprint(f"[MODEL] Loaded A2C Level 4 from epoch {epoch}")
        _vprint(f"[MODEL] Training used GAE lambda: {gae_lambda}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    _vprint(f"[MODEL] Loaded from {model_path} on {device}")
    _vprint(f"[MODEL] Architecture: 1033-D input → 585-D (projected) → 512-D hidden")
    return model


# ============================================================
# Per-patch policy  (single step, no recursion)
# ============================================================


@torch.no_grad()
def _run_policy(wsi, model, history, level, x, y, device, deterministic, verbose=True):
    """
    Run the A2C policy on one patch and return (action, p_stop, p_zoom, v).
    Returns (0, 1.0, 0.0, 0.0) on any failure (safe STOP default).
    """
    _vprint = print if verbose else lambda *a, **kw: None
    try:
        patch = wsi.get_patch(level, x, y)
    except Exception as e:
        _vprint(f"[READ-FAIL] lvl={level} x={x} y={y} :: {e}")
        return 0, 1.0, 0.0, 0.0

    history.visit(level, x, y)

    try:
        env_state = encode_state(wsi, wsi.embedder, patch, level, x, y)
        level_norm = env_state[0]
        x_norm = env_state[1]
        y_norm = env_state[2]
        current_embedding = env_state[3:]

        state = get_hierarchical_aware_state(
            env_state, history, level_norm, x_norm, y_norm
        )
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = model(s)

        if deterministic:
            action = logits.argmax(dim=1).item()
        else:
            action = Categorical(logits=logits).sample().item()

        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        p_stop, p_zoom = probs[0], probs[1]
        v = value.item()

    except Exception as e:
        _vprint(f"[MODEL-FAIL] lvl={level} x={x} y={y} :: {e}")
        import traceback

        traceback.print_exc()
        return 0, 1.0, 0.0, 0.0

    history.update_action(action, current_embedding)
    return action, p_stop, p_zoom, v


# ============================================================
# Single-stage recursive zoom  (multistage=False)
# ============================================================


def _infer_recursive(
    wsi, model, history, lvl, x, y, device, deterministic, verbose=True
):
    """
    Recursively apply the A2C policy to (lvl, x, y), zooming forward.

    Used when wsi.multistage=False — no HUMBE pre-processing.  The patch
    must already be in wsi.active_patches when called.

    On STOP  : patch stays in active_patches (kept as-is).
    On ZOOM  : wsi.zoom_patch() moves parent → zoomed_patches, inserts
               children → active_patches, then recurse on each child.
    TERMINAL : when child_level == min_level all children are kept and
               recursion stops (no policy call on children).
    """
    _vprint = print if verbose else lambda *a, **kw: None
    if (lvl, x, y) not in wsi.active_patches:
        return  # already consumed by a prior zoom

    # Forced STOP: at the finest allowed level
    if lvl <= wsi.min_level:
        _vprint(f"[MIN-LEVEL] lvl={lvl} x={x} y={y} — kept")
        return

    child_level = lvl - 1

    action, p_stop, p_zoom, v = _run_policy(
        wsi, model, history, lvl, x, y, device, deterministic
    )

    if action == 0:
        print(
            f"[STOP] lvl={lvl} x={x} y={y} "
            f"p_stop={p_stop:.4f} p_zoom={p_zoom:.4f} v={v:.4f}"
        )
        return

    # ZOOM
    print(
        f"[ZOOM] lvl={lvl} \u2192 {child_level} x={x} y={y} "
        f"p_stop={p_stop:.4f} p_zoom={p_zoom:.4f} v={v:.4f}"
    )
    try:
        children = wsi.zoom_patch(lvl, x, y)
    except (KeyError, ValueError) as e:
        print(f"[ZOOM-FAIL] {e}")
        return

    # TERMINAL: children are at min_level — keep all, stop recursion
    if child_level == wsi.min_level:
        print(f"[TERMINAL] {len(children)} children at min_level={wsi.min_level} kept")
        return

    for c_lvl, cx, cy in children:
        _infer_recursive(wsi, model, history, c_lvl, cx, cy, device, deterministic)


# ============================================================
# Unzoom helpers  (multistage=True)
# ============================================================


def _remove_descendants(wsi, level, x, y):
    """
    Recursively remove all descendants of (level, x, y) from both
    active_patches and zoomed_patches.  Called after a patch is unzoomed
    so that none of its children linger in either dict.
    """
    child_level = level - 1
    if child_level < 0:
        return
    for grid in wsi.get_child_grid(level, x, y):
        for nx, ny in grid:
            key = (child_level, nx, ny)
            if key in wsi.active_patches:
                del wsi.active_patches[key]
            elif key in wsi.zoomed_patches:
                del wsi.zoomed_patches[key]
                _remove_descendants(wsi, child_level, nx, ny)


def _do_unzoom(wsi, level, x, y):
    """
    Move (level, x, y) from zoomed_patches back to active_patches and
    cascade-remove all of its descendants.
    """
    meta = wsi.zoomed_patches.pop((level, x, y), {})
    wsi.active_patches[(level, x, y)] = meta
    _remove_descendants(wsi, level, x, y)


# ============================================================
# Whole-slide inference  (dispatches on wsi.multistage)
# ============================================================


def infer_wsi_a2c(
    wsi: WSI,
    model_path: str,
    output_html: str | None = None,
    deterministic: bool = True,
    viz_metadata: dict | None = None,
    verbose: bool = True,
) -> WSI:
    """
    A2C inference that works for both pipeline modes:

    multistage=False  (single-stage / no HUMBE)
        Start from all patches at max_level (wsi.active_patches).  For each
        root patch, run the policy recursively, zooming forward into finer
        levels.  wsi.zoom_patch() tracks the tree: parents move to
        zoomed_patches, children are added to active_patches.

    multistage=True  (HUMBE pre-processed)
        Iterate wsi.zoomed_patches shallowest-first.  For each zoomed patch
        the policy decides whether to KEEP the zoom (action=1 — leave as-is)
        or UNZOOM (action=0 — move parent back to active and cascade-remove
        all descendants).
    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    _vprint = print if verbose else lambda *a, **kw: None
    _vprint(
        f"[INFER] Levels: max={wsi.max_level}  min={wsi.min_level}  "
        f"mode={'multistage' if wsi.multistage else 'single-stage'}"
    )

    model = load_a2c_model(model_path, device, verbose=verbose)
    history = HistoryTracker(grid_size=32)

    # ------------------------------------------------------------------
    # Single-stage path: recursive forward zoom from max_level
    # ------------------------------------------------------------------
    if not wsi.multistage:
        root_patches = sorted(wsi.active_patches.keys())  # snapshot
        total = len(root_patches)
        _vprint(f"[INFER] single-stage — {total} root patches at max_level")

        for idx, (lvl, x, y) in enumerate(root_patches):
            if idx % 50 == 0:
                _vprint(
                    f"[INFER] {idx}/{total} "
                    f"(active={wsi.active_patch_count()} "
                    f"zoomed={len(wsi.zoomed_patches)})"
                )
            # Skip if a prior zoom already consumed this patch
            if (lvl, x, y) not in wsi.active_patches:
                continue
            _infer_recursive(
                wsi, model, history, lvl, x, y, device, deterministic, verbose=verbose
            )

    # ------------------------------------------------------------------
    # Multistage path: unzoom pass over HUMBE's zoomed patches
    # ------------------------------------------------------------------
    else:
        # Snapshot and sort shallowest first (highest level = least deep).
        # Ensures cascaded unzooms remove deeper entries before we reach them.
        zoomed_seeds = sorted(wsi.zoomed_patches.keys(), key=lambda k: -k[0])
        total = len(zoomed_seeds)
        _vprint(f"[INFER] multistage — {total} zoomed patches")

        for idx, (lvl, x, y) in enumerate(zoomed_seeds):
            if idx % 50 == 0:
                _vprint(
                    f"[INFER] {idx}/{total} processed "
                    f"(active={wsi.active_patch_count()} "
                    f"zoomed={len(wsi.zoomed_patches)})"
                )

            # May have been cascade-removed by a shallower unzoom earlier
            if (lvl, x, y) not in wsi.zoomed_patches:
                _vprint(f"[SKIP] lvl={lvl} x={x} y={y} — already removed by cascade")
                continue

            # Edge case: patch somehow ended up zoomed at min_level
            if lvl <= wsi.min_level:
                _vprint(f"[UNZOOM-MINLVL] lvl={lvl} x={x} y={y}")
                _do_unzoom(wsi, lvl, x, y)
                continue

            action, p_stop, p_zoom, v = _run_policy(
                wsi, model, history, lvl, x, y, device, deterministic, verbose=verbose
            )

            if action == 0:
                _vprint(
                    f"[UNZOOM] lvl={lvl} x={x} y={y} "
                    f"p_stop={p_stop:.4f} p_zoom={p_zoom:.4f} v={v:.4f}"
                )
                _do_unzoom(wsi, lvl, x, y)
            else:
                _vprint(
                    f"[KEEP-ZOOMED] lvl={lvl} x={x} y={y} "
                    f"p_stop={p_stop:.4f} p_zoom={p_zoom:.4f} v={v:.4f}"
                )

    _vprint(
        f"[INFER] Done — active={wsi.active_patch_count()}  "
        f"zoomed={len(wsi.zoomed_patches)}"
    )

    if output_html is not None:
        mode_tag = (
            "A2C Level-4 (single-stage)"
            if not wsi.multistage
            else "HUMBE + A2C Level-4"
        )
        meta_viz = {
            "Method": mode_tag,
            "Model": str(Path(model_path).name),
            "Policy": "deterministic" if deterministic else "stochastic",
            "Active patches": str(wsi.active_patch_count()),
            "Zoomed patches": str(len(wsi.zoomed_patches)),
        }
        if viz_metadata:
            meta_viz.update(viz_metadata)
        wsi.visualize(output_html, metadata=meta_viz)

    return wsi


# ============================================================
# CLI entry-point  (mirrors infer_rl_a2c.py __main__)
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A2C Level-4 inference — works for both single-stage and multistage (HUMBE) modes"
    )
    parser.add_argument(
        "--image", type=str, default="data/to_test_image/test_img_1.svs"
    )
    parser.add_argument("--model", type=str, default="data/models/rl/a2c/a2c_final.pt")
    parser.add_argument(
        "--output-viz-path", type=str, default="data/visualizations/rl/a2c/viz_a2c.html"
    )
    parser.add_argument(
        "--multistage",
        action="store_true",
        help="Use multistage mode (requires HUMBE pre-processing). "
        "Default: single-stage (forward recursive zoom from max_level).",
    )
    parser.add_argument("--stochastic", action="store_true")

    args = parser.parse_args()

    wsi = WSI(args.image, multistage=args.multistage)

    infer_wsi_a2c(
        wsi,
        model_path=args.model,
        output_html=args.output_viz_path,
        deterministic=not args.stochastic,
    )

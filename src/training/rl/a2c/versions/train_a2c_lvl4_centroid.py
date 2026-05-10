"""
Actor-Critic (A2C) Level 4 — CancerTypeCentroidScore reward
=============================================================

Identical architecture to a2c.py (GAE, HistoryTracker, parent projection),
but the patch reward is driven by CancerTypeCentroidScore instead of a
generic text-alignment scorer.

Cancer type is automatically extracted from each WSI filename:
    TCGA-18-4083-LUSC.svs  →  LUSC

A shared CancerTypeCentroidScore is built per-image and injected into
DynamicPatchEnv as env.patch_score_module.

Usage
-----
    python src/training/rl/a2c/fail/a2c_lvl4_centroid.py
    python src/training/rl/a2c/fail/a2c_lvl4_centroid.py \\
        --images-dir data/images --epochs 10 --episodes-per-image 5
"""

import os
import sys
import argparse
from pathlib import Path
import time
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import optim

# ============================================================================
# Repo setup
# ============================================================================
repo_root = str(Path(__file__).resolve().parents[4])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.wsi import WSI
from src.utils.dynamic_patch_env import DynamicPatchEnv
from src.utils.embedder import Embedder
from src.utils.patch_scores import CancerTypeCentroidScore

# ============================================================================
# Hyperparameters (same as a2c.py)
# ============================================================================
GAMMA = 0.99
LR = 3e-4
ENTROPY_BETA = 0.08
VALUE_COEF = 0.5
GRAD_CLIP = 1.0
EPS = 1e-8
REDUNDANCY_PENALTY = 0.2
OVERLAP_THRESHOLD = 0.3
GAE_LAMBDA = 0.95
PARENT_PROJ_DIM = 64


# ============================================================================
# Cancer-type extraction
# ============================================================================
def extract_cancer_type_from_path(image_path: str) -> str:
    """
    Extract cancer type from a TCGA case filename.
    E.g. 'TCGA-18-4083-LUSC.svs' → 'LUSC'
    Falls back to an empty string when the pattern is not recognised.
    """
    stem = Path(image_path).stem  # e.g. 'TCGA-18-4083-LUSC'
    parts = stem.split("-")
    if len(parts) >= 4 and parts[3].isalpha():
        return parts[3].upper()
    last = parts[-1]
    if last.isalpha() and 2 <= len(last) <= 6:
        return last.upper()
    return ""


# ============================================================================
# History Tracker (identical to a2c.py)
# ============================================================================
class HistoryTracker:
    """Track visited patches, exploration history, and hierarchical context."""

    def __init__(self, grid_size=32):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.visited = {}
        self.last_action = 0
        self.depth = 0
        self.visited_locations = []
        self.parent_embedding = None
        self.current_embedding = None
        self.has_parent = False

    def _hash_location(self, level, x, y):
        x_bin = int(x * self.grid_size)
        y_bin = int(y * self.grid_size)
        return (level, x_bin, y_bin)

    def visit(self, level, x, y):
        key = self._hash_location(level, x, y)
        self.visited[key] = self.visited.get(key, 0) + 1
        self.visited_locations.append((level, x, y))
        return self.visited[key]

    def get_visit_count(self, level, x, y):
        key = self._hash_location(level, x, y)
        return self.visited.get(key, 0)

    def update_action(self, action, current_embedding=None):
        self.last_action = action
        if action == 1:  # ZOOM
            self.depth += 1
            if current_embedding is not None:
                self.parent_embedding = current_embedding.copy()
                self.has_parent = True
        if current_embedding is not None:
            self.current_embedding = current_embedding.copy()

    def compute_redundancy_score(self, level, x, y, threshold=OVERLAP_THRESHOLD):
        if len(self.visited_locations) == 0:
            return 0.0
        same_level_visits = [
            (l, vx, vy)
            for (l, vx, vy) in self.visited_locations
            if abs(l - level) < 0.1
        ]
        if len(same_level_visits) == 0:
            return 0.0
        distances = [
            np.sqrt((x - vx) ** 2 + (y - vy) ** 2) for (_, vx, vy) in same_level_visits
        ]
        overlaps = sum(1 for d in distances if d < threshold)
        return min(overlaps / max(1, len(same_level_visits)), 1.0)

    def compute_overlap_penalty(self, level, x, y):
        if len(self.visited_locations) < 2:
            return 0.0
        for l, vx, vy in self.visited_locations[-5:]:
            if abs(l - level) < 0.1:
                dist = np.sqrt((x - vx) ** 2 + (y - vy) ** 2)
                if dist < OVERLAP_THRESHOLD * 0.5:
                    return 1.0
                elif dist < OVERLAP_THRESHOLD:
                    return 0.5
        return 0.0

    def get_history_features(self, level, x, y):
        visit_count = self.get_visit_count(level, x, y)
        return np.array(
            [
                min(visit_count, 5) / 5.0,
                float(self.last_action),
                self.depth / 10.0,
                self.compute_redundancy_score(level, x, y),
                self.compute_overlap_penalty(level, x, y),
            ],
            dtype=np.float32,
        )

    def get_hierarchical_features(self):
        if self.has_parent and self.parent_embedding is not None:
            return self.parent_embedding.copy(), 1.0
        return np.zeros(512, dtype=np.float32), 0.0


# ============================================================================
# Actor-Critic Network (identical to a2c.py)
# ============================================================================
class ActorCritic(nn.Module):
    def __init__(
        self, base_state_dim=520, parent_emb_dim=512, parent_proj_dim=64, hidden_dim=512
    ):
        super().__init__()
        self.parent_proj = nn.Sequential(
            nn.Linear(parent_emb_dim, parent_proj_dim), nn.ReLU()
        )
        final_state_dim = base_state_dim + parent_proj_dim + 1
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

        nn.init.orthogonal_(self.parent_proj[0].weight, gain=1.0)
        nn.init.zeros_(self.parent_proj[0].bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.1)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.actor.bias)
        self.actor.bias.data[1] = 0.1
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        base_state = x[:, :520]
        parent_emb = x[:, 520:1032]
        has_parent = x[:, 1032:1033]
        x_proj = torch.cat(
            [base_state, self.parent_proj(parent_emb), has_parent], dim=1
        )
        h = self.encoder(x_proj)
        return self.actor(h), self.critic(h)


# ============================================================================
# State construction (identical to a2c.py)
# ============================================================================
def get_hierarchical_aware_state(env_state, history, level_norm, x_norm, y_norm):
    history_features = history.get_history_features(level_norm, x_norm, y_norm)
    parent_emb, has_parent = history.get_hierarchical_features()
    return np.concatenate([env_state, history_features, parent_emb, [has_parent]])


# ============================================================================
# Episode Rollout (identical to a2c.py)
# ============================================================================
def rollout_episode(
    env,
    model,
    device,
    redundancy_penalty=REDUNDANCY_PENALTY,
    overlap_threshold=OVERLAP_THRESHOLD,
):
    history = HistoryTracker(grid_size=32)
    states, actions, log_probs, values, rewards, raw_rewards = [], [], [], [], [], []

    env_state = env.reset()
    done = False
    steps = 0

    while not done:
        level_norm = env_state[0]
        x_norm = env_state[1]
        y_norm = env_state[2]
        current_embedding = env_state[3:]

        state = get_hierarchical_aware_state(
            env_state, history, level_norm, x_norm, y_norm
        )
        history.visit(level_norm, x_norm, y_norm)

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        logits, value = model(state_tensor)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        states.append(state)
        actions.append(action.item())
        log_probs.append(log_prob.squeeze())
        values.append(value.squeeze())

        next_env_state, reward, done, info = env.step(action.item())

        if action.item() == 1 and "s_stop" in info and "s_zoom" in info:
            s_stop = info["s_stop"]
            s_zoom = info["s_zoom"]
            if s_zoom is not None and s_stop is not None:
                reward = s_zoom - s_stop

        raw_reward = reward
        redundancy_score = history.compute_redundancy_score(
            level_norm, x_norm, y_norm, threshold=overlap_threshold
        )
        overlap_penalty_val = history.compute_overlap_penalty(
            level_norm, x_norm, y_norm
        )
        adjusted_reward = reward - redundancy_penalty * (
            redundancy_score + overlap_penalty_val
        )

        raw_rewards.append(raw_reward)
        rewards.append(adjusted_reward)
        steps += 1
        history.update_action(action.item(), current_embedding)

        if not done:
            env_state = next_env_state

    return {
        "states": states,
        "actions": actions,
        "log_probs": log_probs,
        "values": values,
        "rewards": rewards,
        "raw_rewards": raw_rewards,
        "terminal_reward": rewards[-1] if rewards else 0.0,
        "steps": steps,
        "info": info,
        "history": history,
    }


# ============================================================================
# GAE (identical to a2c.py)
# ============================================================================
def compute_gae_advantages_and_returns(trajectory, gamma=GAMMA, gae_lambda=GAE_LAMBDA):
    rewards = trajectory["rewards"]
    values = trajectory["values"]
    T = len(rewards)
    values_tensor = torch.stack(values)
    rewards_tensor = torch.tensor(
        rewards, dtype=torch.float32, device=values_tensor.device
    )

    advantages = torch.zeros(T, device=values_tensor.device)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = 0.0 if t == T - 1 else values_tensor[t + 1]
        delta = rewards_tensor[t] + gamma * next_value - values_tensor[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    returns = advantages + values_tensor
    return returns, advantages


def a2c_update(
    model,
    optimizer,
    trajectory,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    entropy_beta=ENTROPY_BETA,
    value_coef=VALUE_COEF,
):
    returns, advantages = compute_gae_advantages_and_returns(
        trajectory, gamma, gae_lambda
    )

    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

    log_probs = torch.stack(trajectory["log_probs"])
    values = torch.stack(trajectory["values"])

    states_tensor = torch.stack(
        [
            torch.tensor(s, dtype=torch.float32, device=values.device)
            for s in trajectory["states"]
        ]
    )
    logits, _ = model(states_tensor)
    dist = Categorical(logits=logits)
    entropy = dist.entropy()

    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_loss = -entropy.mean()
    loss = policy_loss + value_coef * value_loss + entropy_beta * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        mean_stop_prob = probs[:, 0].mean().item()
        mean_zoom_prob = probs[:, 1].mean().item()

    avg_raw_reward = (
        float(np.mean(trajectory["raw_rewards"])) if trajectory["raw_rewards"] else 0.0
    )
    avg_adj_reward = float(np.mean(trajectory["rewards"]))

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.mean().item(),
        "terminal_reward": trajectory["terminal_reward"],
        "episode_return": returns[0].item(),
        "steps": trajectory["steps"],
        "mean_value": values.mean().item(),
        "mean_advantage": advantages.mean().item(),
        "mean_stop_prob": mean_stop_prob,
        "mean_zoom_prob": mean_zoom_prob,
        "avg_raw_reward": avg_raw_reward,
        "redundancy_impact": avg_raw_reward - avg_adj_reward,
    }


# ============================================================================
# Training Loop
# ============================================================================
def train_a2c(
    images_dir,
    num_epochs=10,
    episodes_per_image=30,
    output_dir="data/models/rl/a2c/versions/a2c_lvl4",
    device=None,
    redundancy_penalty=REDUNDANCY_PENALTY,
    overlap_threshold=OVERLAP_THRESHOLD,
    gae_lambda=GAE_LAMBDA,
):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"[A2C-Centroid] Training on device: {device}")
    print(f"GAE lambda: {gae_lambda} | Redundancy penalty: {redundancy_penalty}")
    os.makedirs(output_dir, exist_ok=True)

    embedder = Embedder()

    model = ActorCritic(
        base_state_dim=520,
        parent_emb_dim=512,
        parent_proj_dim=PARENT_PROJ_DIM,
        hidden_dim=512,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    image_paths = sorted(glob(f"{images_dir}/*.svs"))
    print(f"Found {len(image_paths)} WSI images")
    if not image_paths:
        raise ValueError(f"No .svs images found in {images_dir}")

    global_step = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_metrics = {
            k: []
            for k in (
                "loss",
                "terminal_reward",
                "episode_return",
                "steps",
                "entropy",
                "redundancy_impact",
            )
        }

        for img_path in image_paths:
            img_name = Path(img_path).stem

            # ------------------------------------------------------------------
            # Resolve cancer type from filename
            # ------------------------------------------------------------------
            cancer_type = extract_cancer_type_from_path(img_path)
            if not cancer_type:
                print(
                    f"[WARN] Could not extract cancer type from '{img_name}' — skipping."
                )
                continue

            try:
                wsi = WSI(img_path, embedder=embedder)

                # Build the centroid scorer for this cancer type
                scorer = CancerTypeCentroidScore(
                    cancer_type=cancer_type,
                    weight=1.0,
                    embedder=embedder,
                )
                _ = scorer._get_centroid()  # force build now
                print(f"[INFO] {img_name}  cancer_type={cancer_type}  centroid ready.")

                # Build env using a safe placeholder scorer, then inject the real one
                env = DynamicPatchEnv(
                    wsi=wsi,
                    patch_score="text_align_score",
                    patch_size=256,
                    max_steps=8,
                )
                env.patch_score_module = (
                    scorer  # override with per-image centroid scorer
                )

                for ep in range(episodes_per_image):
                    trajectory = rollout_episode(
                        env, model, device, redundancy_penalty, overlap_threshold
                    )
                    metrics = a2c_update(
                        model, optimizer, trajectory, gae_lambda=gae_lambda
                    )

                    for key in epoch_metrics:
                        if key in metrics:
                            epoch_metrics[key].append(metrics[key])

                    global_step += 1
                    if global_step % 10 == 0:
                        print(
                            f"[Epoch {epoch+1}/{num_epochs}] Step {global_step} | "
                            f"{img_name} ({cancer_type}) | "
                            f"Loss: {metrics['loss']:.4f} | "
                            f"Terminal R: {metrics['terminal_reward']:.4f} | "
                            f"Steps: {metrics['steps']} | "
                            f"Entropy: {metrics['entropy']:.4f} | "
                            f"P(stop): {metrics['mean_stop_prob']:.3f} | "
                            f"Redund: {metrics['redundancy_impact']:.4f}"
                        )

            except Exception as e:
                print(f"[ERROR] Failed to process {img_name}: {e}")
                continue

        epoch_time = time.time() - epoch_start
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"{'='*80}")
        for key, vals in epoch_metrics.items():
            if vals:
                print(f"{key:20s}: {np.mean(vals):8.4f} ± {np.std(vals):6.4f}")
        print(f"{'='*80}\n")

        ckpt = os.path.join(output_dir, f"a2c_lvl4_centroid_epoch_{epoch+1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "gae_lambda": gae_lambda,
            },
            ckpt,
        )
        print(f"Checkpoint saved: {ckpt}\n")

    print("Training complete!")
    final_path = os.path.join(output_dir, "a2c_lvl4_centroid_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="A2C Level 4 — CancerTypeCentroidScore reward (cancer type auto-detected from filename)"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/images",
        help="Directory containing WSI images",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--episodes-per-image", type=int, default=5)
    parser.add_argument(
        "--output-dir", type=str, default="data/models/rl/a2c_lvl4_centroid"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/mps/cpu)"
    )
    parser.add_argument("--redundancy-penalty", type=float, default=REDUNDANCY_PENALTY)
    parser.add_argument("--overlap-threshold", type=float, default=OVERLAP_THRESHOLD)
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=GAE_LAMBDA,
        help="GAE lambda (0=TD(0), 1=Monte Carlo)",
    )

    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    )

    train_a2c(
        images_dir=args.images_dir,
        num_epochs=args.epochs,
        episodes_per_image=args.episodes_per_image,
        output_dir=args.output_dir,
        device=device,
        redundancy_penalty=args.redundancy_penalty,
        overlap_threshold=args.overlap_threshold,
        gae_lambda=args.gae_lambda,
    )


if __name__ == "__main__":
    main()

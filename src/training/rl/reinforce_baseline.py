"""
REINFORCE + Learned Baseline for Hierarchical Patch Zooming
Stopping-Time Decision Problem (Terminal Reward Only)

Key properties:
---------------
1. Pure Monte-Carlo REINFORCE (no TD, no bootstrapping)
2. Learned state-dependent baseline to reduce variance
3. Terminal reward optimization
4. Small zoom penalty to enforce stopping trade-off
5. Designed explicitly for optimal stopping problems

Actor learns:     π(STOP | s), π(ZOOM | s)
Baseline learns:  b(s) ≈ E[terminal return | s]
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import optim

# ============================================================================
# Repo setup
# ============================================================================
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.wsi import WSI
from src.utils.dynamic_patch_env import DynamicPatchEnv
from src.utils.embedder import Embedder

# ============================================================================
# Hyperparameters
# ============================================================================
GAMMA = 0.99
LR = 5e-4
VALUE_COEF = 0.5
GRAD_CLIP = 1.0
ZOOM_PENALTY = 0.01

# ============================================================================
# Model
# ============================================================================
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


# ============================================================================
# Rollout
# ============================================================================
def rollout_episode(env, model, device):
    states = []
    log_probs = []
    rewards = []

    state = env.reset()
    done = False

    while not done:
        state_t = torch.tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(state_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        next_state, reward, done, info = env.step(action.item())

        # Small cost for zooming (information is not free)
        if action.item() == 1:  # ZOOM
            reward -= ZOOM_PENALTY

        states.append(state)
        log_probs.append(log_prob)
        rewards.append(reward)

        if not done:
            state = next_state

    # Monte-Carlo returns
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + GAMMA * R
        returns.insert(0, R)

    return {
        "states": states,
        "log_probs": torch.stack(log_probs),
        "returns": torch.tensor(returns, device=device),
        "steps": len(returns),
        "terminal_reward": rewards[-1],
    }


# ============================================================================
# REINFORCE Update
# ============================================================================
def reinforce_update(model, optimizer, trajectory):
    states = torch.stack([
        torch.tensor(s, dtype=torch.float32, device=trajectory["returns"].device)
        for s in trajectory["states"]
    ])

    logits, baseline = model(states)

    returns = trajectory["returns"]
    log_probs = trajectory["log_probs"]

    advantages = returns - baseline.detach()

    # Policy gradient loss
    policy_loss = -(log_probs * advantages).mean()

    # Baseline regression
    baseline_loss = F.mse_loss(baseline, returns)

    loss = policy_loss + VALUE_COEF * baseline_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "baseline_loss": baseline_loss.item(),
        "mean_return": returns.mean().item(),
        "mean_advantage": advantages.mean().item(),
        "steps": trajectory["steps"],
    }


# ============================================================================
# Training Loop
# ============================================================================
def train(
    images_dir,
    patch_score,
    num_epochs,
    episodes_per_image,
    output_dir,
    device,
):
    os.makedirs(output_dir, exist_ok=True)

    embedder = Embedder()
    model = PolicyWithBaseline().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    image_paths = sorted(Path(images_dir).glob("*.svs"))
    if not image_paths:
        raise RuntimeError(f"No .svs images found in {images_dir}")

    global_step = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_steps = []

        for img_path in image_paths:
            wsi = WSI(str(img_path), embedder=embedder)

            env = DynamicPatchEnv(
                wsi=wsi,
                patch_score=patch_score,
                patch_size=256,
                max_steps=8,
            )

            for _ in range(episodes_per_image):
                traj = rollout_episode(env, model, device)
                metrics = reinforce_update(model, optimizer, traj)

                epoch_steps.append(metrics["steps"])
                global_step += 1

                if global_step % 20 == 0:
                    print(
                        f"[Epoch {epoch+1}/{num_epochs}] "
                        f"Step {global_step} | "
                        f"Loss {metrics['loss']:.3f} | "
                        f"Return {metrics['mean_return']:.3f} | "
                        f"Steps {metrics['steps']}"
                    )

        print(
            f"\nEpoch {epoch+1} finished in {time.time() - start_time:.1f}s | "
            f"Mean steps: {np.mean(epoch_steps):.2f}\n"
        )

        ckpt_path = os.path.join(output_dir, f"reinforce_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}\n")

    final_path = os.path.join(output_dir, "reinforce_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="REINFORCE + baseline for stopping-time patch zooming"
    )
    parser.add_argument("--images-dir", type=str, default="data/images")
    parser.add_argument("--patch-score", type=str, default="text_align_score")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--episodes-per-image", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="data/models/rl/reinforce")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Training on device: {device}")

    train(
        images_dir=args.images_dir,
        patch_score=args.patch_score,
        num_epochs=args.epochs,
        episodes_per_image=args.episodes_per_image,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()

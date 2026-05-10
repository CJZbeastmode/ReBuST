"""Module for train dqn."""

# B_qlearning_full.py

import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import sys
from pathlib import Path
import time

# ========================================================
# Repo path setup
# ========================================================
repo_root = str(Path(__file__).resolve().parents[4])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.wsi import WSI
from src.utils.dynamic_patch_env import DynamicPatchEnv
from src.utils.patch_scores import *

LAMBDA_LR = 0.05  # FIXED: Match A2C learning rate for faster adaptation
ZOOM_BUDGET = 0.5
WARMUP_EPOCHS = 2  # Warmup epochs with fixed lambda for exploration

# =========================
# Q NETWORK
# =========================


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


# =========================
# Reply Buffer
# =========================


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(list, zip(*batch))

    def __len__(self):
        return len(self.buffer)


# =========================
# AGENT
# =========================


class DQNAgent:
    def __init__(
        self,
        state_dim,
        lr=1e-4,
        gamma=0.99,
        eps=0.2,
        eps_min=0.01,
        eps_decay=0.999,
        buffer_size=50_000,
        target_update=5_000,
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma

        self.q = QNet(state_dim).to(device)
        self.q_target = QNet(state_dim).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.step_count = 0
        self.train_frequency = 4  # Train every N steps
        self.update_count = 0  # Track number of network updates

    def act(self, state):
        if random.random() < self.eps:
            return random.randint(0, 1)

        with torch.no_grad():
            q_vals = self.q(state.unsqueeze(0))
            return q_vals.argmax(dim=1).item()

    def store(self, s, a, r, s_next, done):
        self.buffer.push(
            (
                s.detach().cpu(),
                int(a),
                float(r),
                None if s_next is None else s_next.detach().cpu(),
                float(done),
            )
        )

    def train_step(self, batch_size=64):
        # Warm-up: wait until buffer has enough diverse samples
        # Set to ~2 epochs worth of data (80 images × 20 ep × 5 avg_steps = 8000)
        # Using 2000 to start training after ~first epoch with diverse experiences
        if len(self.buffer) < max(batch_size, 2000):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.stack(states).to(self.device)  # (B, D)
        actions = torch.tensor(actions, device=self.device).long()  # (B,)
        rewards = torch.tensor(rewards, device=self.device).float()  # (B,)
        dones = torch.tensor(dones, device=self.device).float()  # (B,)

        # Q(s,a)
        q_vals = self.q(states)  # (B, 2)
        q_sa = q_vals.gather(1, actions.unsqueeze(1)).squeeze(
            1
        )  # (B,) - select Q-value for taken action

        # Build q_next with masking for terminal transitions
        with torch.no_grad():
            q_next = torch.zeros(
                batch_size, device=self.device
            )  # default 0 for terminal

            non_terminal_idx = [i for i, ns in enumerate(next_states) if ns is not None]
            if len(non_terminal_idx) > 0:
                ns_batch = torch.stack([next_states[i] for i in non_terminal_idx]).to(
                    self.device
                )  # (B_nt, D)

                # FIXED: Double DQN - use online network for action selection
                # This reduces overestimation bias
                online_q_next = self.q(ns_batch)  # (B_nt, 2)
                best_actions = online_q_next.argmax(dim=1)  # (B_nt,)

                # Use target network for Q-value evaluation
                target_q_next = self.q_target(ns_batch)  # (B_nt, 2)
                q_next_vals = target_q_next.gather(
                    1, best_actions.unsqueeze(1)
                ).squeeze(
                    1
                )  # (B_nt,)

                q_next[torch.tensor(non_terminal_idx, device=self.device)] = q_next_vals

            target = (
                rewards + self.gamma * q_next
            )  # (B,) - q_next already 0 for terminals

        loss = nn.functional.mse_loss(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=1.0)  # Gradient clipping
        self.opt.step()

        self.update_count += 1
        # FIXED: Update target network based on training updates, not steps
        # Use smaller frequency (1000) for more stable learning
        if self.update_count % 1000 == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return loss.item()

    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    # =========================
    # INFERENCE
    # =========================

    @torch.no_grad()
    def infer(agent, state):
        state = torch.tensor(state, dtype=torch.float32, device=agent.device)
        q_vals = agent.q(state)
        action = torch.argmax(q_vals).item()
        return action, q_vals.cpu().numpy()


# =========================
# TRAINING LOOP
# =========================


def run_episode_dqn(env, agent, lambda_zoom, device, max_steps, batch_size=64):
    state = torch.tensor(env.reset(), dtype=torch.float32, device=device)

    ep_reward_raw = 0.0  # Accumulate raw rewards (before lambda penalty)
    ep_reward_eff = 0.0  # Accumulate constrained rewards (for monitoring)
    zoom_count = 0
    steps = 0
    training_steps = 0  # Track training iterations for this episode

    for _ in range(max_steps):
        action = agent.act(state)

        next_state, reward, done, info = env.step(action)

        # ---- cost definition (same as A2C) ----
        cost = 1.0 if action == 1 else 0.0
        zoom_count += int(cost)

        # FIXED: Separate raw reward from constrained reward for clarity
        # Raw reward for episode statistics
        ep_reward_raw += reward
        # Effective reward for learning (what agent learns from)
        reward_eff = reward - lambda_zoom * cost
        ep_reward_eff += reward_eff

        if next_state is not None:
            next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device)
        else:
            next_state_t = None

        agent.buffer.push(state, action, reward_eff, next_state_t, done)

        steps += 1

        if done:
            break

        state = next_state_t

    # Train at end of episode with full trajectory
    # Decay epsilon per episode (not per step) for stable learning
    if len(agent.buffer) >= max(batch_size, 2000):
        # Train multiple times per episode if buffer is large
        for _ in range(max(1, steps // agent.train_frequency)):
            loss = agent.train_step(batch_size=batch_size)
            if loss is not None:
                training_steps += 1
        agent.decay_eps()  # Decay once per episode

    zoom_fraction = zoom_count / max(1, steps)

    return {
        "reward_raw": ep_reward_raw,
        "reward_eff": ep_reward_eff,
        "zoom_frac": zoom_fraction,
        "training_steps": training_steps,
    }


# =========================
# MAIN
# =========================


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, default="data/images")
    parser.add_argument("--score-module", type=str, default="text_align_score")
    parser.add_argument("--img-train-count", type=int, default=80)
    parser.add_argument("--episodes-per-wsi", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default="data/models")
    parser.add_argument("--test-run", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------------------
    # Collect WSIs
    # --------------------------------------------------
    all_images = sorted(f for f in os.listdir(args.images_dir) if f.endswith(".svs"))
    train_images = all_images[: args.img_train_count]

    if args.test_run:
        train_images = train_images[:1]
        args.episodes_per_wsi = 1
        args.epochs = 1

    # --------------------------------------------------
    # Initialize env once to get state_dim
    # --------------------------------------------------
    t0 = time.time()
    from src.utils.embedder import Embedder

    # Create shared embedder to avoid loading multiple models
    embedder = Embedder(img_backend="plip", device=device)

    wsi0 = WSI(os.path.join(args.images_dir, train_images[0]), embedder=embedder)
    env0 = DynamicPatchEnv(wsi0, patch_score=args.score_module)
    state_dim = len(env0.reset())

    print(f"State dim: {state_dim}")

    # --------------------------------------------------
    # Create agent
    # --------------------------------------------------
    # CRITICAL FIX: Much slower epsilon decay to maintain exploration
    # Problem: Fast decay causes Q-values to bias toward STOP before learning zoom benefits
    agent = DQNAgent(
        state_dim=state_dim,
        device=device,
        eps=1.0,  # Start with full exploration
        eps_min=0.2,  # Keep higher minimum - need more exploration than A2C
        eps_decay=0.9999,  # Much slower decay - was 0.9995, caused premature exploitation
    )
    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    lambda_zoom = 2.0  # FIXED: Start with 2.0 like A2C for stronger initial exploration

    for epoch in range(args.epochs):
        # Warmup phase: fixed lambda for exploration
        is_warmup = epoch < WARMUP_EPOCHS

        if is_warmup:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} (WARMUP) ===")
            print(f"Fixed lambda ({lambda_zoom:.3f}) for exploration")
        else:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        epoch_rewards = []

        for img in train_images:
            wsi = WSI(os.path.join(args.images_dir, img), embedder=embedder)
            env = DynamicPatchEnv(wsi, patch_score=args.score_module)

            for ep in range(args.episodes_per_wsi):
                stats = run_episode_dqn(
                    env,
                    agent,
                    lambda_zoom,
                    device,
                    args.max_steps,
                    batch_size=args.batch_size,
                )

                epoch_rewards.append(stats["reward_raw"])

                # Dual update (constraint enforcement) - same as A2C
                # Only update lambda after warmup epochs
                if not is_warmup:
                    # Allow lambda to go to 0 to match A2C behavior
                    lambda_zoom += LAMBDA_LR * (stats["zoom_frac"] - ZOOM_BUDGET)
                    lambda_zoom = max(0.0, min(5.0, lambda_zoom))  # Match A2C: min=0.0

                print(
                    f"{img} | ep {ep+1} | "
                    f"R_raw={stats['reward_raw']:.3f} | "
                    f"R_eff={stats['reward_eff']:.3f} | "
                    f"zoom={stats['zoom_frac']:.2f} | "
                    f"lambda={lambda_zoom:.3f} | "
                    f"eps={agent.eps:.3f}"
                )

        if args.output:
            os.makedirs(args.output, exist_ok=True)
            path = os.path.join(args.output, "dqn_backup.pt")
            torch.save(agent.q.state_dict(), path)
            print(f"Model saved to {path}")

        t1 = time.time()
        elapsed = t1 - t0
        print(
            f"Epoch {epoch+1} mean reward: {sum(epoch_rewards)/len(epoch_rewards):.3f}"
        )
        print(f"Time elapsed after epoch {epoch+1}, image {img}: {elapsed:.1f} sec")

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        path = os.path.join(args.output, "dqn.pt")
        torch.save(agent.q.state_dict(), path)
        print(f"Model saved to {path}")


if __name__ == "__main__":
    main()


# 21701 sec

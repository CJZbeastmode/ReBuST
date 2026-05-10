"""Module for rl q learning fail."""

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
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.wsi import WSI
from src.utils.dynamic_patch_env import DynamicPatchEnv
from src.utils.patch_scores import *

LAMBDA_LR = 0.01
ZOOM_BUDGET = 0.5

# =========================
# Q NETWORK
# =========================


class QNet(nn.Module):
    """
    Q(s) -> [Q(stop), Q(zoom)]
    """

    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
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
        self.action_count = 0  # Separate counter for training frequency

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
        # Warm-up: wait until buffer has enough samples
        if len(self.buffer) < max(batch_size, 1000):
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
                q_next_vals = self.q_target(ns_batch).max(1)[0]  # (B_nt,)
                q_next[torch.tensor(non_terminal_idx, device=self.device)] = q_next_vals

            target = (
                rewards + self.gamma * q_next
            )  # (B,) - q_next already 0 for terminals

        loss = nn.functional.mse_loss(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=1.0)  # Gradient clipping
        self.opt.step()

        self.step_count += 1
        if self.step_count % self.target_update == 0:
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


def run_episode_dqn(env, agent, lambda_zoom, device, max_steps):
    state = torch.tensor(env.reset(), dtype=torch.float32, device=device)

    ep_reward = 0.0
    zoom_count = 0
    steps = 0

    for _ in range(max_steps):
        action = agent.act(state)

        next_state, reward, done, info = env.step(action)

        # ALTERED: Use raw reward directly (env step() returns immediate reward/score difference)
        # No manual delta computation needed

        # ---- cost definition (same as A2C) ----
        cost = 1.0 if action == 1 else 0.0
        zoom_count += int(cost)

        # ALTERED: Effective reward with zoom penalty
        reward_norm = float(np.clip(reward, -1.0, 1.0))

        reward_eff = reward_norm - lambda_zoom * cost
        ep_reward += reward_eff

        if next_state is not None:
            next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device)
        else:
            next_state_t = None

        agent.buffer.push(state, action, reward_eff, next_state_t, done)

        agent.action_count += 1

        # Train every N steps instead of every step
        if agent.action_count % agent.train_frequency == 0:
            agent.train_step(batch_size=64)

        # Decay epsilon per step
        agent.decay_eps()

        steps += 1

        if done:
            break

        state = next_state_t

    # Fix #5: zoom_fraction should use total steps taken (including STOP action)
    zoom_fraction = zoom_count / max(1, steps)

    return ep_reward, zoom_fraction


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
    # ALTERED: Higher initial epsilon and faster decay for better exploration
    agent = DQNAgent(
        state_dim=state_dim,
        device=device,
        eps=1.0,  # ALTERED: Start with full exploration
        eps_min=0.05,  # ALTERED: Lower minimum for more exploitation
        eps_decay=0.9995,  # ALTERED: Faster decay to reach exploitation phase
    )

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    lambda_zoom = 1.0

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        epoch_rewards = []

        for img in train_images:
            wsi = WSI(os.path.join(args.images_dir, img), embedder=embedder)
            env = DynamicPatchEnv(wsi, patch_score=args.score_module)

            for ep in range(args.episodes_per_wsi):
                ep_reward, zoom_frac = run_episode_dqn(
                    env, agent, lambda_zoom, device, args.max_steps
                )

                epoch_rewards.append(ep_reward)

                # ALTERED: dual update with proper lambda bounds
                lambda_zoom += LAMBDA_LR * (zoom_frac - ZOOM_BUDGET)
                lambda_zoom = max(0.0, min(5.0, lambda_zoom))

                print(
                    f"{img} | ep {ep+1} | "
                    f"R={ep_reward:.3f} | zoom={zoom_frac:.2f} | "
                    f"lambda={lambda_zoom:.3f} | eps={agent.eps:.3f}"
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

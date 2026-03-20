import os
import sys
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import time

# ============================================================
# FAILURE DEMONSTRATION CONFIGURATION
# ============================================================
# This configuration is intentionally set to demonstrate why
# REINFORCE-based contextual bandits fail for hierarchical
# patch selection tasks. Key failure modes:
#
# 1. SLOW BASELINE (beta=0.01): Causes advantage bias, leading
#    to premature convergence to suboptimal STOP-only policy
#
# 2. NO LAYER NORMALIZATION: Training instability with high-dim
#    states (515-dim) causes gradient variance and oscillation
#
# 3. WEAK ENTROPY (0.005): Policy collapses quickly to
#    deterministic behavior without sufficient exploration
#
# 4. SINGLE-SAMPLE UPDATES: High-variance gradients from
#    one context-action-reward tuple per update
#
# Expected failure pattern:
# - Initial random exploration (entropy ~0.693)
# - Rapid collapse to STOP-only (zoom_frac → 0)
# - Entropy drops to near-zero (deterministic policy)
# - Lambda oscillates but can't correct the collapsed policy
# - Final performance: worse than random baseline
# ============================================================

# ============================================================
# Repo path setup
# ============================================================
repo_root = str(Path(__file__).resolve().parents[4])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.wsi import WSI
from src.utils.dynamic_patch_env import DynamicPatchEnv

# ============================================================
# Policy Network
# ============================================================

class Policy(nn.Module):
    """
    π(a | s) for contextual bandit.
    Outputs logits for STOP / ZOOM.
    """

    def __init__(self, state_dim):
        super().__init__()
        # FAILURE DEMO: No LayerNorm - causes training instability with high-dim states
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Constrained Bandit Agent
# ============================================================

class ConstrainedBanditAgent:
    """
    Contextual bandit with:
    - REINFORCE
    - EMA baseline
    - Lagrangian zoom constraint
    """

    def __init__(
        self,
        state_dim,
        lr=1e-4,
        beta=0.01,  # FAILURE DEMO: Slow baseline adaptation causes advantage bias
        lambda_lr=0.05,
        zoom_budget=0.5,
        entropy_coef=0.005,  # FAILURE DEMO: Weak entropy allows quick collapse to suboptimal policy
        device="cpu",
    ):
        self.device = device

        self.policy = Policy(state_dim).to(device)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)

        # Baseline (EMA of rewards)
        self.baseline = 0.0
        self.beta = beta

        # Constraint
        # *PATCH: Initialize to 2.0 instead of 1.0 to match A2C (stronger initial zoom penalty)
        self.lambda_zoom = 2.0
        self.lambda_lr = lambda_lr
        self.zoom_budget = zoom_budget
        
        # Entropy regularization
        self.entropy_coef = entropy_coef

    def reinforce_update(self, state, action, reward_eff):
        """
        One REINFORCE update with entropy regularization.
        """
        
        # Convert reward_eff to float if it's a tensor
        reward_val = reward_eff.item() if torch.is_tensor(reward_eff) else float(reward_eff)

        # Update baseline (EMA) - use scalar values only
        # *PATCH: Proper EMA update with faster tracking (beta=0.1)
        self.baseline = (1 - self.beta) * self.baseline + self.beta * reward_val
        
        # Advantage = reward - baseline
        # *PATCH: Don't normalize advantage by std (causes instability with single samples)
        advantage = reward_val - self.baseline

        logits = self.policy(state.unsqueeze(0))
        dist = Categorical(logits=logits)

        # Policy loss with entropy bonus
        # *PATCH: Advantage is a scalar, just multiply with log_prob
        policy_loss = -dist.log_prob(action) * advantage
        # *PATCH: Strong entropy regularization to prevent collapse
        entropy_loss = -dist.entropy()
        
        loss = policy_loss + self.entropy_coef * entropy_loss

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.opt.step()

        return loss.item(), dist.entropy().item()


# ============================================================
# Training Loop
# ============================================================

def train_constrained_bandit(
    agent,
    images,
    images_dir,
    steps_per_image,
    device,
    embedder,
    backup_output,
    patch_score="text_align_score",
):
    """
    Train contextual bandit across multiple WSIs.
    """

    agent.policy.train()

    t0 = time.time()
    for img_idx, img in enumerate(images):
        wsi = WSI(os.path.join(images_dir, img), embedder=embedder)
        env = DynamicPatchEnv(wsi, patch_score=patch_score)

        print(f"\nInitialized DynamicPatchEnv with patch score module: {patch_score}")
        print(f"[IMAGE {img_idx+1}/{len(images)}] {img}")
        
        # Reset counters for this image
        zoom_count = 0
        step_count = 0
        rewards = []
        losses = []
        entropies = []

        for step in range(steps_per_image):

            # ----------------------------------------------
            # 1. Sample independent context
            # ----------------------------------------------
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

            # ----------------------------------------------
            # 2. Sample action from policy
            # ----------------------------------------------
            logits = agent.policy(state.unsqueeze(0))
            dist = Categorical(logits=logits)
            action = dist.sample()

            # ----------------------------------------------
            # 3. Immediate reward
            # ----------------------------------------------
            _, reward, _, info = env.step(action.item())
            rewards.append(reward)

            # ----------------------------------------------
            # 4. Cost (zoom usage)
            # ----------------------------------------------
            cost = 1.0 if action.item() == 1 else 0.0
            zoom_count += int(cost)
            step_count += 1

            # ----------------------------------------------
            # 5. Lagrangian reward
            # ----------------------------------------------
            reward_eff = reward - agent.lambda_zoom * cost

            # ----------------------------------------------
            # 6. REINFORCE update
            # ----------------------------------------------
            loss, entropy = agent.reinforce_update(state, action, reward_eff)
            losses.append(loss)
            entropies.append(entropy)

            # ----------------------------------------------
            # 7. Dual update (constraint) - update every step like A2C
            # ----------------------------------------------
            # *PATCH: Update lambda every step, not every 100 steps
            # Running average constraint violation for more stable updates
            zoom_fraction = zoom_count / max(1, step_count)
            constraint_violation = zoom_fraction - agent.zoom_budget

            agent.lambda_zoom += agent.lambda_lr * constraint_violation
            agent.lambda_zoom = max(0.0, agent.lambda_zoom)

            # ----------------------------------------------
            # 8. Logging - match A2C style
            # ----------------------------------------------
            log_count = 20
            if (step + 1) % log_count == 0:
                avg_reward = sum(rewards[-log_count:]) / len(rewards[-log_count:]) if len(rewards) >= log_count else sum(rewards) / max(1, len(rewards))
                avg_loss = sum(losses[-log_count:]) / len(losses[-log_count:]) if len(losses) >= log_count else sum(losses) / max(1, len(losses))
                avg_entropy = sum(entropies[-log_count:]) / len(entropies[-log_count:]) if len(entropies) >= log_count else sum(entropies) / max(1, len(entropies))
                
                print(
                    f"{img} | step {step+1:05d}/{steps_per_image} | "
                    f"loss={avg_loss:.3f} | "
                    f"reward={avg_reward:.3f} | "
                    f"zoom_frac={zoom_fraction:.2f} | "
                    f"lambda={agent.lambda_zoom:.3f} | "
                    f"entropy={avg_entropy:.3f}"
                )
        
        # Final stats for this image
        final_zoom_frac = zoom_count / max(1, step_count)
        final_avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        final_avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        t1 = time.time()
        elapsed = t1 - t0
        
        print(
            f"{img} | COMPLETED | "
            f"avg_reward={final_avg_reward:.3f} | "
            f"zoom_frac={final_zoom_frac:.2f} | "
            f"lambda={agent.lambda_zoom:.3f}"
        )
        print(f"Time elapsed after image {img_idx+1}/{len(images)}: {elapsed:.1f} sec")
        
        # Backup model after each image
        os.makedirs(backup_output, exist_ok=True)
        path = os.path.join(backup_output, "bandit_constrained_backup.pt")
        torch.save(agent.policy.state_dict(), path)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, default="data/images")
    parser.add_argument("--img-train-count", type=int, default=80)
    # FAILURE DEMO: Use 5000 steps - enough to show collapse but faster for overnight run
    parser.add_argument("--steps-per-image", type=int, default=200)
    parser.add_argument("--zoom-budget", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
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
    # Load images
    # --------------------------------------------------
    all_images = sorted(f for f in os.listdir(args.images_dir) if f.endswith(".svs"))
    images = all_images[: args.img_train_count]

    if args.test_run:
        images = images[:1]
        # *PATCH: Use 1000 steps even for test run (bandit needs sufficient samples)
        args.steps_per_image = 1000

    # --------------------------------------------------
    # Init shared embedder and env for state_dim
    # --------------------------------------------------
    from src.utils.embedder import Embedder
    
    embedder = Embedder(img_backend="plip", device=device)
    
    wsi0 = WSI(os.path.join(args.images_dir, images[0]), embedder=embedder)
    env0 = DynamicPatchEnv(wsi0)
    state_dim = len(env0.reset())

    print(f"State dim: {state_dim}")

    # --------------------------------------------------
    # Agent
    # --------------------------------------------------
    agent = ConstrainedBanditAgent(
        state_dim=state_dim,
        lr=args.lr,
        zoom_budget=args.zoom_budget,
        device=device,
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    train_constrained_bandit(
        agent=agent,
        images=images,
        images_dir=args.images_dir,
        steps_per_image=args.steps_per_image,
        device=device,
        backup_output=args.output,
        embedder=embedder,
    )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, "bandit_constrained.pt")
    torch.save(agent.policy.state_dict(), path)
    print(f"\nModel saved to {path}")


# ============================================================
if __name__ == "__main__":
    main()

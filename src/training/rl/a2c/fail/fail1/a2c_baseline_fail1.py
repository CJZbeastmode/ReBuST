"""
Actor-Critic (A2C) Training for Hierarchical Patch Zooming
Stopping-Time Decision Problem (No Budget, No Constraints)

This implementation focuses on learning when to stop zooming based on
expected terminal utility improvement.

Key principles:
---------------
1. NO budget constraints, NO CMDP formulation, NO Lagrange multipliers
2. Pure Actor-Critic with standard advantages
3. All intermediate rewards are zero
4. Terminal reward represents utility improvement
5. Critic learns: "What utility gain will I get if I continue optimally?"
6. Actor learns: "Should I zoom or stop at this patch?"

Training objective:
-------------------
- Sparse patches → STOP early (no utility from zooming)
- Informative patches → ZOOM deep (utility increases with resolution)
- Policy discovers this through trial-and-error with delayed rewards

Network architecture:
---------------------
- Shared encoder: state → hidden representation
- Actor head: hidden → logits(STOP, ZOOM)
- Critic head: hidden → expected future terminal reward
"""

import os
import sys
import argparse
from pathlib import Path
import time

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

# ============================================================================
# Hyperparameters
# ============================================================================
GAMMA = 0.99                # Discount factor (should be high for delayed rewards)
LR = 3e-4                   # Learning rate
ENTROPY_BETA = 0.01         # Entropy regularization (encourage exploration)
VALUE_COEF = 0.5            # Critic loss weight
GRAD_CLIP = 1.0             # Gradient clipping
EPS = 1e-8                  # Numerical stability


# ============================================================================
# Actor-Critic Network
# ============================================================================
class ActorCritic(nn.Module):
    """
    Standard Actor-Critic architecture with shared encoder.
    
    Input: state (515-D) = [embedding(512), x_norm, y_norm, depth_norm]
    Outputs:
        - Actor: logits for [STOP, ZOOM]
        - Critic: expected terminal reward V(s)
    """

    def __init__(self, state_dim=515, hidden_dim=256):
        super().__init__()

        # Shared feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Actor head: policy logits
        self.actor = nn.Linear(hidden_dim, 2)

        # Critic head: value function
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialization for stable early training
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor, shape (B, 515)
            Batch of states
        
        Returns
        -------
        logits : torch.Tensor, shape (B, 2)
            Action logits [STOP, ZOOM]
        value : torch.Tensor, shape (B, 1)
            State value estimates
        """
        h = self.encoder(x)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value


# ============================================================================
# Episode Rollout
# ============================================================================
def rollout_episode(env, model, device):
    """
    Execute one complete episode with the current policy.
    
    This function handles the reward structure from DynamicPatchEnv.
    
    Parameters
    ----------
    env : DynamicPatchEnv
        Environment instance
    model : ActorCritic
        Policy network
    device : torch.device
        Computation device
    
    Returns
    -------
    trajectory : dict
        Contains:
        - states: list of states
        - actions: list of actions
        - log_probs: list of log probabilities
        - values: list of value estimates
        - rewards: list of rewards (0 except terminal)
        - terminal_reward: final reward value
        - info: episode metadata
    """
    states = []
    actions = []
    log_probs = []
    values = []
    rewards = []

    state = env.reset()
    done = False
    steps = 0

    with torch.no_grad():
        while not done:
            # Convert state to tensor
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)

            # Forward pass
            logits, value = model(state_tensor)

            # Sample action
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Store trajectory
            states.append(state)
            actions.append(action.item())
            log_probs.append(log_prob.squeeze())
            values.append(value.squeeze())

            # Environment step
            next_state, reward, done, info = env.step(action.item())

            rewards.append(reward)
            steps += 1

            if not done:
                state = next_state

    # Extract terminal reward (last reward in trajectory)
    terminal_reward = rewards[-1] if rewards else 0.0

    return {
        'states': states,
        'actions': actions,
        'log_probs': log_probs,
        'values': values,
        'rewards': rewards,
        'terminal_reward': terminal_reward,
        'steps': steps,
        'info': info,
    }


# ============================================================================
# A2C Update
# ============================================================================
def compute_advantages_and_returns(trajectory, gamma=GAMMA):
    """
    Compute returns and advantages using TD error.
    
    For stopping-time problems with terminal rewards:
    - Returns are computed by bootstrapping from the terminal state
    - Advantages = TD errors = R_t + γV(s_{t+1}) - V(s_t)
    
    Since intermediate rewards are zero, this simplifies to:
    - For all steps before terminal: advantage ≈ γV(s_{t+1}) - V(s_t)
    - For terminal step: advantage = terminal_reward - V(s_terminal)
    
    Parameters
    ----------
    trajectory : dict
        Episode trajectory from rollout_episode
    gamma : float
        Discount factor
    
    Returns
    -------
    returns : torch.Tensor, shape (T,)
        Discounted returns for each timestep
    advantages : torch.Tensor, shape (T,)
        TD advantages
    """
    rewards = trajectory['rewards']
    values = trajectory['values']
    
    T = len(rewards)
    
    # Convert to tensors
    values_tensor = torch.stack(values)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=values_tensor.device)
    
    # Compute returns (backward pass)
    returns = torch.zeros(T, device=values_tensor.device)
    R = 0.0  # Bootstrap value (zero at terminal state)
    
    for t in reversed(range(T)):
        R = rewards_tensor[t] + gamma * R
        returns[t] = R
    
    # Compute advantages (TD error)
    advantages = torch.zeros(T, device=values_tensor.device)
    
    for t in range(T):
        if t == T - 1:
            # Terminal state: no next value
            next_value = 0.0
        else:
            next_value = values_tensor[t + 1]
        
        # TD error: r + γV(s') - V(s)
        advantages[t] = rewards_tensor[t] + gamma * next_value - values_tensor[t]
    
    return returns, advantages


def a2c_update(model, optimizer, trajectory, gamma=GAMMA, entropy_beta=ENTROPY_BETA, value_coef=VALUE_COEF):
    """
    Perform A2C update on a single episode trajectory.
    
    Loss components:
    ----------------
    1. Policy loss: -log π(a|s) * A(s,a)
    2. Value loss: MSE(V(s), R_t)
    3. Entropy regularization: -H(π(·|s))
    
    Parameters
    ----------
    model : ActorCritic
    optimizer : torch.optim.Optimizer
    trajectory : dict
        Episode trajectory
    gamma : float
        Discount factor
    entropy_beta : float
        Entropy regularization coefficient
    value_coef : float
        Value loss weight
    
    Returns
    -------
    metrics : dict
        Training metrics for logging
    """
    # Compute returns and advantages
    returns, advantages = compute_advantages_and_returns(trajectory, gamma)
    
    # Normalize advantages (improves stability)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)
    
    # Stack trajectory tensors
    log_probs = torch.stack(trajectory['log_probs'])
    values = torch.stack(trajectory['values'])
    
    # Compute entropy from stored log probs
    # Reconstruct distribution for entropy calculation
    states = trajectory['states']
    states_tensor = torch.stack([
        torch.tensor(s, dtype=torch.float32, device=values.device) for s in states
    ])
    
    logits, _ = model(states_tensor)
    dist = Categorical(logits=logits)
    entropy = dist.entropy()
    
    # ========================================================================
    # Loss computation
    # ========================================================================
    
    # 1. Policy loss (actor)
    policy_loss = -(log_probs * advantages.detach()).mean()
    
    # 2. Value loss (critic)
    value_loss = F.mse_loss(values, returns)
    
    # 3. Entropy bonus (exploration)
    entropy_loss = -entropy.mean()
    
    # Total loss
    loss = policy_loss + value_coef * value_loss + entropy_beta * entropy_loss
    
    # ========================================================================
    # Optimization step
    # ========================================================================
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    
    # ========================================================================
    # Metrics
    # ========================================================================
    metrics = {
        'loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.mean().item(),
        'terminal_reward': trajectory['terminal_reward'],
        'episode_return': returns[0].item(),  # Discounted return from start
        'steps': trajectory['steps'],
        'mean_value': values.mean().item(),
        'mean_advantage': advantages.mean().item(),
    }
    
    return metrics


# ============================================================================
# Training Loop
# ============================================================================
def train_a2c(
    images_dir,
    patch_score="text_align_score",
    num_epochs=10,
    episodes_per_image=5,
    output_dir='data/models/rl/a2c_baseline',
    device=None,
):
    """
    Main A2C training loop.
    
    Parameters
    ----------
    images_dir : str
        Path to WSI images
    patch_score : str
        Patch scoring module to use (e.g., 'text_align_score')
    num_epochs : int
        Number of training epochs
    episodes_per_image : int
        Episodes per WSI per epoch
    output_dir : str
        Model checkpoint directory
    device : torch.device
        Computation device
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Training on device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize embedder
    embedder = Embedder()
    
    print(f"Using patch score: {patch_score}")
    
    # Initialize model and optimizer
    model = ActorCritic(state_dim=515, hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load WSI images
    from glob import glob
    image_paths = sorted(glob(f"{images_dir}/*.svs"))
    print(f"Found {len(image_paths)} WSI images")
    
    if len(image_paths) == 0:
        raise ValueError(f"No .svs images found in {images_dir}")
    
    # Training loop
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_metrics = {
            'loss': [],
            'terminal_reward': [],
            'episode_return': [],
            'steps': [],
            'entropy': [],
        }
        
        for img_path in image_paths:
            img_name = Path(img_path).stem
            
            try:
                # Load WSI
                wsi = WSI(img_path, embedder=embedder)
                
                # Create environment
                env = DynamicPatchEnv(
                    wsi=wsi,
                    patch_score=patch_score,
                    patch_size=256,
                    max_steps=8,
                )
                
                # Run multiple episodes per image
                for ep in range(episodes_per_image):
                    # Rollout episode
                    trajectory = rollout_episode(env, model, device)
                    
                    # A2C update
                    metrics = a2c_update(model, optimizer, trajectory)
                    
                    # Accumulate metrics
                    for key in epoch_metrics:
                        if key in metrics:
                            epoch_metrics[key].append(metrics[key])
                    
                    global_step += 1
                    
                    # Log every 10 episodes
                    if global_step % 10 == 0:
                        print(
                            f"[Epoch {epoch+1}/{num_epochs}] "
                            f"Step {global_step} | "
                            f"Image: {img_name} | "
                            f"Loss: {metrics['loss']:.4f} | "
                            f"Terminal R: {metrics['terminal_reward']:.4f} | "
                            f"Return: {metrics['episode_return']:.4f} | "
                            f"Steps: {metrics['steps']} | "
                            f"Entropy: {metrics['entropy']:.4f}"
                        )
                
            except Exception as e:
                print(f"[ERROR] Failed to process {img_name}: {e}")
                continue
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"{'='*80}")
        
        for key, values in epoch_metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{key:20s}: {mean_val:8.4f} ± {std_val:6.4f}")
        
        print(f"{'='*80}\n")
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"a2c_baseline_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}\n")
    
    print("Training complete!")
    
    # Save final model
    final_path = os.path.join(output_dir, "a2c_baseline_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="A2C baseline training for stopping-time patch zooming"
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        default='data/images',
        help='Directory containing WSI images'
    )
    parser.add_argument(
        '--patch-score',
        type=str,
        default='text_align_score',
        help='Patch scoring module (e.g., text_align_score, img_sim_score)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--episodes-per-image',
        type=int,
        default=5,
        help='Episodes per image per epoch'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/models/rl/a2c_baseline',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/mps/cpu)'
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Train
    train_a2c(
        images_dir=args.images_dir,
        patch_score=args.patch_score,
        num_epochs=args.epochs,
        episodes_per_image=args.episodes_per_image,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == '__main__':
    main()

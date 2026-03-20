"""
Actor-Critic (A2C) Level 1: History Awareness (Raza-core)
==========================================================

CHANGES FROM BASELINE (a2c_baseline.py):
-----------------------------------------
✅ Added history-aware state representation:
   - Visited patch tracking (spatial hash map)
   - Visit count per location
   - Last action taken (one-hot encoded)
   - Depth trace (cumulative zoom depth)

✅ State dimension expanded: 515 → 518
   - Original: [level_norm, x_norm, y_norm, embedding(512)]
   - New: [level_norm, x_norm, y_norm, embedding(512), visit_count, last_action, depth]

✅ Network updated to handle 518-D input

✅ Episode rollout tracks history across trajectory

KEY CLAIM UNLOCKED:
-------------------
"Decisions depend on what has already been seen"
- Agent can avoid revisiting patches
- Policy learns exploration vs exploitation
- Foundation for redundancy avoidance (Level 2)

This is the SINGLE MOST IMPORTANT Raza aspect.
Without history, redundancy avoidance and exploration claims are weak.
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
GAMMA = 0.99                # Discount factor
LR = 5e-4                   # Learning rate
ENTROPY_BETA = 0.03         # Entropy regularization
VALUE_COEF = 0.5            # Critic loss weight
GRAD_CLIP = 1.0             # Gradient clipping
EPS = 1e-8                  # Numerical stability


# ============================================================================
# ✨ NEW: History Tracker
# ============================================================================
class HistoryTracker:
    """
    Track visited patches and exploration history during an episode.
    
    This enables the agent to:
    - Remember which locations it has visited
    - Count how many times it's seen each region
    - Track the depth of exploration
    - Encode the last action taken
    """
    
    def __init__(self, grid_size=32):
        """
        Parameters
        ----------
        grid_size : int
            Spatial resolution for hashing patch locations
            (higher = finer tracking, more memory)
        """
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        """Reset history at episode start."""
        self.visited = {}  # {(level, x_bin, y_bin): visit_count}
        self.last_action = 0  # 0=STOP, 1=ZOOM (starts with STOP)
        self.depth = 0  # Cumulative zoom depth
    
    def _hash_location(self, level, x, y):
        """
        Hash a continuous (level, x, y) to a discrete grid cell.
        
        This allows tracking approximate spatial regions without
        requiring exact coordinate matching.
        """
        x_bin = int(x * self.grid_size)
        y_bin = int(y * self.grid_size)
        return (level, x_bin, y_bin)
    
    def visit(self, level, x, y):
        """
        Record a visit to a location.
        
        Returns
        -------
        visit_count : int
            Number of times this location has been visited (including this time)
        """
        key = self._hash_location(level, x, y)
        self.visited[key] = self.visited.get(key, 0) + 1
        return self.visited[key]
    
    def get_visit_count(self, level, x, y):
        """Get visit count for a location (0 if never visited)."""
        key = self._hash_location(level, x, y)
        return self.visited.get(key, 0)
    
    def update_action(self, action):
        """Update the last action taken."""
        self.last_action = action
        if action == 1:  # ZOOM
            self.depth += 1
    
    def get_history_features(self, level, x, y):
        """
        Extract history features for state augmentation.
        
        Returns
        -------
        features : np.ndarray, shape (3,)
            [visit_count, last_action, depth]
        """
        visit_count = self.get_visit_count(level, x, y)
        # Normalize visit count (clip at 5 for stability)
        visit_count_norm = min(visit_count, 5) / 5.0
        
        return np.array([
            visit_count_norm,
            float(self.last_action),
            self.depth / 10.0  # Normalize depth (assuming max ~10 zooms)
        ], dtype=np.float32)


# ============================================================================
# Actor-Critic Network (UPDATED for history-aware state)
# ============================================================================
class ActorCritic(nn.Module):
    """
    History-aware Actor-Critic architecture.
    
    Input: state (518-D) = [embedding(512), x_norm, y_norm, depth_norm, 
                             visit_count, last_action, zoom_depth]
    Outputs:
        - Actor: logits for [STOP, ZOOM]
        - Critic: expected cumulative reward V(s)
    
    ✨ CHANGE: state_dim increased from 515 → 518 to include history
    """

    def __init__(self, state_dim=518, hidden_dim=256):
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

        # Initialization
        nn.init.orthogonal_(self.actor.weight, gain=0.1)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.actor.bias)
        self.actor.bias.data[1] = 0.1  # Small bias toward zoom
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor, shape (B, 518)
            Batch of history-aware states
        
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
# ✨ NEW: History-aware state construction
# ============================================================================
def get_history_aware_state(env_state, history, level_norm, x_norm, y_norm):
    """
    Augment environment state with history features.
    
    Parameters
    ----------
    env_state : np.ndarray, shape (515,)
        Original environment state [level, x, y, embedding(512)]
    history : HistoryTracker
        Episode history tracker
    level_norm : float
        Normalized level (for visit counting)
    x_norm : float
        Normalized x coordinate
    y_norm : float
        Normalized y coordinate
    
    Returns
    -------
    augmented_state : np.ndarray, shape (518,)
        [original(515), visit_count, last_action, depth]
    """
    # Get history features
    history_features = history.get_history_features(level_norm, x_norm, y_norm)
    
    # Concatenate: [env_state(515), history(3)] → 518-D
    return np.concatenate([env_state, history_features])


# ============================================================================
# Episode Rollout (UPDATED with history tracking)
# ============================================================================
def rollout_episode(env, model, device):
    """
    Execute one complete episode with history-aware policy.
    
    ✨ CHANGES:
    - HistoryTracker initialized and maintained throughout episode
    - State augmented with history features before policy evaluation
    - History updated after each action
    
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
        Contains states, actions, log_probs, values, rewards, and history info
    """
    # ========================================================================
    # ✨ BEGIN LEVEL 1 ADDITIONS
    # ========================================================================
    history = HistoryTracker(grid_size=32)
    # ========================================================================
    # ✨ END LEVEL 1 ADDITIONS
    # ========================================================================
    
    states = []
    actions = []
    log_probs = []
    values = []
    rewards = []

    env_state = env.reset()
    done = False
    steps = 0

    while not done:
        # ====================================================================
        # ✨ BEGIN LEVEL 1 ADDITIONS: Augment state with history
        # ====================================================================
        # Extract normalized coordinates from env_state
        level_norm = env_state[0]
        x_norm = env_state[1]
        y_norm = env_state[2]
        
        # Get history-aware state
        state = get_history_aware_state(env_state, history, level_norm, x_norm, y_norm)
        
        # Record visit
        history.visit(level_norm, x_norm, y_norm)
        # ====================================================================
        # ✨ END LEVEL 1 ADDITIONS
        # ====================================================================

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
        next_env_state, reward, done, info = env.step(action.item())
        
        # Reward shaping (same as baseline)
        if action.item() == 1 and 's_stop' in info and 's_zoom' in info:
            s_stop = info['s_stop']
            s_zoom = info['s_zoom']
            if s_zoom is not None and s_stop is not None:
                reward = s_zoom - s_stop

        rewards.append(reward)
        steps += 1
        
        # ====================================================================
        # ✨ BEGIN LEVEL 1 ADDITIONS: Update history
        # ====================================================================
        history.update_action(action.item())
        # ====================================================================
        # ✨ END LEVEL 1 ADDITIONS
        # ====================================================================

        if not done:
            env_state = next_env_state

    # Extract terminal reward
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
        'history': history,  # ✨ NEW: Include history for analysis
    }


# ============================================================================
# A2C Update (unchanged from baseline)
# ============================================================================
def compute_advantages_and_returns(trajectory, gamma=GAMMA):
    """
    Compute returns and advantages using TD(0).
    
    (Same as baseline - no changes needed)
    """
    rewards = trajectory['rewards']
    values = trajectory['values']
    
    T = len(rewards)
    
    values_tensor = torch.stack(values)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=values_tensor.device)
    
    # Compute returns
    returns = torch.zeros(T, device=values_tensor.device)
    R = 0.0
    
    for t in reversed(range(T)):
        R = rewards_tensor[t] + gamma * R
        returns[t] = R
    
    # Compute advantages
    advantages = torch.zeros(T, device=values_tensor.device)
    
    for t in range(T):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values_tensor[t + 1]
        
        advantages[t] = rewards_tensor[t] + gamma * next_value - values_tensor[t]
    
    return returns, advantages


def a2c_update(model, optimizer, trajectory, gamma=GAMMA, entropy_beta=ENTROPY_BETA, value_coef=VALUE_COEF):
    """
    Perform A2C update.
    
    (Same as baseline - no changes needed)
    """
    returns, advantages = compute_advantages_and_returns(trajectory, gamma)
    
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)
    
    log_probs = torch.stack(trajectory['log_probs'])
    values = torch.stack(trajectory['values'])
    
    # Recompute for entropy
    states = trajectory['states']
    states_tensor = torch.stack([
        torch.tensor(s, dtype=torch.float32, device=values.device) for s in states
    ])
    
    logits, _ = model(states_tensor)
    dist = Categorical(logits=logits)
    entropy = dist.entropy()
    
    # Loss computation
    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_loss = -entropy.mean()
    
    loss = policy_loss + value_coef * value_loss + entropy_beta * entropy_loss
    
    # Optimization
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    
    # Metrics
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        mean_stop_prob = probs[:, 0].mean().item()
        mean_zoom_prob = probs[:, 1].mean().item()
    
    metrics = {
        'loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.mean().item(),
        'terminal_reward': trajectory['terminal_reward'],
        'episode_return': returns[0].item(),
        'steps': trajectory['steps'],
        'mean_value': values.mean().item(),
        'mean_advantage': advantages.mean().item(),
        'mean_stop_prob': mean_stop_prob,
        'mean_zoom_prob': mean_zoom_prob,
    }
    
    return metrics


# ============================================================================
# Training Loop
# ============================================================================
def train_a2c(
    images_dir,
    patch_score="text_align_score",
    num_epochs=10,
    episodes_per_image=30,
    output_dir='data/models/rl/a2c_lvl1',
    device=None,
):
    """
    Main A2C Level 1 training loop.
    
    ✨ CHANGE: output_dir default is 'a2c_lvl1' to distinguish from baseline
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Training A2C Level 1 (History Awareness) on device: {device}")
    print(f"State dimension: 518 (515 baseline + 3 history features)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    embedder = Embedder()
    print(f"Using patch score: {patch_score}")
    
    # ========================================================================
    # ✨ LEVEL 1 CHANGE: state_dim = 518 (was 515)
    # ========================================================================
    model = ActorCritic(state_dim=518, hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    from glob import glob
    image_paths = sorted(glob(f"{images_dir}/*.svs"))
    print(f"Found {len(image_paths)} WSI images")
    
    if len(image_paths) == 0:
        raise ValueError(f"No .svs images found in {images_dir}")
    
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
                wsi = WSI(img_path, embedder=embedder)
                env = DynamicPatchEnv(
                    wsi=wsi,
                    patch_score=patch_score,
                    patch_size=256,
                    max_steps=8,
                )
                
                for ep in range(episodes_per_image):
                    trajectory = rollout_episode(env, model, device)
                    metrics = a2c_update(model, optimizer, trajectory)
                    
                    for key in epoch_metrics:
                        if key in metrics:
                            epoch_metrics[key].append(metrics[key])
                    
                    global_step += 1
                    
                    if global_step % 10 == 0:
                        print(
                            f"[Epoch {epoch+1}/{num_epochs}] "
                            f"Step {global_step} | "
                            f"Image: {img_name} | "
                            f"Loss: {metrics['loss']:.4f} | "
                            f"Terminal R: {metrics['terminal_reward']:.4f} | "
                            f"Steps: {metrics['steps']} | "
                            f"Entropy: {metrics['entropy']:.4f} | "
                            f"P(stop): {metrics['mean_stop_prob']:.3f}"
                        )
                
            except Exception as e:
                print(f"[ERROR] Failed to process {img_name}: {e}")
                continue
        
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
        
        checkpoint_path = os.path.join(output_dir, f"a2c_lvl1_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}\n")
    
    print("Training complete!")
    
    final_path = os.path.join(output_dir, "a2c_lvl1_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="A2C Level 1: History-aware patch zooming"
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
        help='Patch scoring module'
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
        default='data/models/rl/a2c_lvl1',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/mps/cpu)'
    )
    
    args = parser.parse_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
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

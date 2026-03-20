"""
Actor-Critic (A2C) Level 2: Redundancy Avoidance (Raza-justification)
======================================================================

CHANGES FROM LEVEL 1 (a2c_lvl1.py):
------------------------------------
✅ All Level 1 features retained:
   - Visited patch tracking
   - Visit count per location
   - Last action encoding
   - Depth trace

✅ NEW Level 2 additions:
   - Explicit redundancy penalty in reward function
   - Spatial overlap detection for visited regions
   - Redundancy score based on proximity to visited patches
   - Enhanced state with redundancy features (518 → 520)

✅ State dimension expanded: 518 → 520
   - Level 1: [level, x, y, embedding(512), visit_count, last_action, depth]
   - Level 2: [... Level 1 ..., redundancy_score, overlap_penalty]

✅ Reward shaping includes redundancy penalty:
   - Original reward adjusted by overlap with visited regions
   - Encourages exploration of novel spatial areas

KEY CLAIM UNLOCKED:
-------------------
"Agent avoids re-focusing on previously explored regions"
- Strong, reviewer-proof motivation for RL over greedy
- Builds on history awareness (Level 1 required)
- Still local, still interpretable
- Demonstrates intelligent spatial exploration

This is your SECOND extension, dependent on Level 1.
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

# ========================================================================
# ✨ NEW LEVEL 2: Redundancy parameters
# ========================================================================
REDUNDANCY_PENALTY = 0.2    # Penalty weight for revisiting regions
OVERLAP_THRESHOLD = 0.3     # Spatial proximity threshold for overlap
# ========================================================================
# ✨ END LEVEL 2 ADDITIONS
# ========================================================================


# ============================================================================
# History Tracker (EXTENDED for redundancy detection)
# ============================================================================
class HistoryTracker:
    """
    Track visited patches and exploration history during an episode.
    
    ✨ LEVEL 2 EXTENSIONS:
    - Compute spatial redundancy score
    - Detect overlap with previously visited regions
    - Track visited region centroids for proximity calculation
    """
    
    def __init__(self, grid_size=32):
        """
        Parameters
        ----------
        grid_size : int
            Spatial resolution for hashing patch locations
        """
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        """Reset history at episode start."""
        self.visited = {}  # {(level, x_bin, y_bin): visit_count}
        self.last_action = 0
        self.depth = 0
        # ====================================================================
        # ✨ BEGIN LEVEL 2 ADDITIONS
        # ====================================================================
        self.visited_locations = []  # List of (level, x, y) for overlap detection
        # ====================================================================
        # ✨ END LEVEL 2 ADDITIONS
        # ====================================================================
    
    def _hash_location(self, level, x, y):
        """Hash continuous coordinates to discrete grid."""
        x_bin = int(x * self.grid_size)
        y_bin = int(y * self.grid_size)
        return (level, x_bin, y_bin)
    
    def visit(self, level, x, y):
        """
        Record a visit to a location.
        
        Returns
        -------
        visit_count : int
            Number of times this location has been visited
        """
        key = self._hash_location(level, x, y)
        self.visited[key] = self.visited.get(key, 0) + 1
        
        # ====================================================================
        # ✨ BEGIN LEVEL 2 ADDITIONS: Track continuous locations
        # ====================================================================
        self.visited_locations.append((level, x, y))
        # ====================================================================
        # ✨ END LEVEL 2 ADDITIONS
        # ====================================================================
        
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
    
    # ========================================================================
    # ✨ BEGIN LEVEL 2 ADDITIONS: Redundancy detection methods
    # ========================================================================
    def compute_redundancy_score(self, level, x, y, threshold=OVERLAP_THRESHOLD):
        """
        Compute redundancy score based on spatial overlap with visited patches.
        
        The score measures how close the current location is to previously
        visited regions. Higher score = more redundant (closer to visited areas).
        
        Parameters
        ----------
        level : float
            Normalized level
        x : float
            Normalized x coordinate
        y : float
            Normalized y coordinate
        threshold : float
            Distance threshold for considering regions as overlapping
        
        Returns
        -------
        redundancy_score : float
            Score in [0, 1], where 1 = highly redundant (overlaps with many visits)
        """
        if len(self.visited_locations) == 0:
            return 0.0  # First visit, no redundancy
        
        # Compute distances to all visited locations at the same level
        same_level_visits = [
            (l, vx, vy) for (l, vx, vy) in self.visited_locations 
            if abs(l - level) < 0.1  # Same pyramid level (with tolerance)
        ]
        
        if len(same_level_visits) == 0:
            print(f"[REDUNDANCY] Redundancy calculated: 0.0")
            return 0.0  # No visits at this level yet
        
        # Compute spatial distances
        distances = []
        for (_, vx, vy) in same_level_visits:
            dist = np.sqrt((x - vx)**2 + (y - vy)**2)
            distances.append(dist)
        
        # Count overlaps (visits within threshold distance)
        overlaps = sum(1 for d in distances if d < threshold)
        
        # Normalize by total visits at this level (clip at 5 for stability)
        redundancy = min(overlaps / max(1, len(same_level_visits)), 1.0)
        
        print(f"[REDUNDANCY] Redundancy calculated: {redundancy}")
        return redundancy
    
    def compute_overlap_penalty(self, level, x, y):
        """
        Compute explicit penalty for being too close to visited regions.
        
        This is a binary-like penalty that strongly discourages
        immediate revisits to the same spatial area.
        
        Returns
        -------
        penalty : float
            Penalty value in [0, 1], where 1 = strong overlap
        """
        if len(self.visited_locations) < 2:
            return 0.0  # Need at least 2 visits for meaningful overlap
        
        # Check if current location was recently visited
        recent_visits = self.visited_locations[-5:]  # Last 5 visits
        
        for (l, vx, vy) in recent_visits:
            if abs(l - level) < 0.1:  # Same level
                dist = np.sqrt((x - vx)**2 + (y - vy)**2)
                if dist < OVERLAP_THRESHOLD * 0.5:  # Very close
                    return 1.0  # Strong penalty
                elif dist < OVERLAP_THRESHOLD:  # Moderately close
                    return 0.5  # Moderate penalty
        
        return 0.0  # No significant overlap
    # ========================================================================
    # ✨ END LEVEL 2 ADDITIONS
    # ========================================================================
    
    def get_history_features(self, level, x, y):
        """
        Extract history features for state augmentation.
        
        ✨ LEVEL 2: Now includes redundancy features
        
        Returns
        -------
        features : np.ndarray, shape (5,)
            [visit_count, last_action, depth, redundancy_score, overlap_penalty]
        """
        visit_count = self.get_visit_count(level, x, y)
        visit_count_norm = min(visit_count, 5) / 5.0
        
        # ====================================================================
        # ✨ BEGIN LEVEL 2 ADDITIONS: Add redundancy features
        # ====================================================================
        redundancy_score = self.compute_redundancy_score(level, x, y)
        overlap_penalty = self.compute_overlap_penalty(level, x, y)
        # ====================================================================
        # ✨ END LEVEL 2 ADDITIONS
        # ====================================================================
        
        return np.array([
            visit_count_norm,
            float(self.last_action),
            self.depth / 10.0,
            redundancy_score,      # ✨ NEW
            overlap_penalty,       # ✨ NEW
        ], dtype=np.float32)


# ============================================================================
# Actor-Critic Network (UPDATED for redundancy-aware state)
# ============================================================================
class ActorCritic(nn.Module):
    """
    Redundancy-aware Actor-Critic architecture.
    
    Input: state (520-D) = [embedding(512), x_norm, y_norm, depth_norm, 
                             visit_count, last_action, zoom_depth,
                             redundancy_score, overlap_penalty]
    Outputs:
        - Actor: logits for [STOP, ZOOM]
        - Critic: expected cumulative reward V(s)
    
    ✨ LEVEL 2 CHANGE: state_dim increased from 518 → 520 (added 2 redundancy features)
    """

    def __init__(self, state_dim=520, hidden_dim=256):
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
        x : torch.Tensor, shape (B, 520)
            Batch of redundancy-aware states
        
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
# Redundancy-aware state construction
# ============================================================================
def get_history_aware_state(env_state, history, level_norm, x_norm, y_norm):
    """
    Augment environment state with history and redundancy features.
    
    ✨ LEVEL 2: Now returns 520-D state (was 518-D in Level 1)
    
    Parameters
    ----------
    env_state : np.ndarray, shape (515,)
        Original environment state
    history : HistoryTracker
        Episode history tracker
    level_norm : float
        Normalized level
    x_norm : float
        Normalized x coordinate
    y_norm : float
        Normalized y coordinate
    
    Returns
    -------
    augmented_state : np.ndarray, shape (520,)
        [original(515), visit_count, last_action, depth, redundancy, overlap]
    """
    # Get history features (now includes redundancy)
    history_features = history.get_history_features(level_norm, x_norm, y_norm)
    
    # Concatenate: [env_state(515), history(5)] → 520-D
    return np.concatenate([env_state, history_features])


# ============================================================================
# Episode Rollout (UPDATED with redundancy-aware rewards)
# ============================================================================
def rollout_episode(env, model, device, redundancy_penalty=REDUNDANCY_PENALTY, overlap_threshold=OVERLAP_THRESHOLD):
    """
    Execute one complete episode with redundancy-aware policy.
    
    ✨ LEVEL 2 CHANGES:
    - Rewards adjusted by redundancy penalty
    - Explicit penalty for revisiting recently explored regions
    - Encourages spatial diversity in exploration
    
    Parameters
    ----------
    env : DynamicPatchEnv
        Environment instance
    model : ActorCritic
        Policy network
    device : torch.device
        Computation device
    redundancy_penalty : float
        Weight for redundancy penalty
    overlap_threshold : float
        Threshold for overlap detection
    
    Returns
    -------
    trajectory : dict
        Contains states, actions, rewards (with redundancy penalties), history
    """
    history = HistoryTracker(grid_size=32)
    
    states = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    # ====================================================================
    # ✨ BEGIN LEVEL 2 ADDITIONS: Track raw and adjusted rewards
    # ====================================================================
    raw_rewards = []  # Original rewards before redundancy adjustment
    # ====================================================================
    # ✨ END LEVEL 2 ADDITIONS
    # ====================================================================

    env_state = env.reset()
    done = False
    steps = 0

    while not done:
        # Extract normalized coordinates
        level_norm = env_state[0]
        x_norm = env_state[1]
        y_norm = env_state[2]
        
        # Get redundancy-aware state
        state = get_history_aware_state(env_state, history, level_norm, x_norm, y_norm)
        
        # Record visit
        history.visit(level_norm, x_norm, y_norm)

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
        
        # Reward shaping (from baseline)
        if action.item() == 1 and 's_stop' in info and 's_zoom' in info:
            s_stop = info['s_stop']
            s_zoom = info['s_zoom']
            if s_zoom is not None and s_stop is not None:
                reward = s_zoom - s_stop
        
        # ====================================================================
        # ✨ BEGIN LEVEL 2 ADDITIONS: Apply redundancy penalty to reward
        # ====================================================================
        raw_reward = reward
        
        # Get current redundancy features (using passed threshold)
        redundancy_score = history.compute_redundancy_score(level_norm, x_norm, y_norm, threshold=overlap_threshold)
        overlap_penalty_val = history.compute_overlap_penalty(level_norm, x_norm, y_norm)
        
        # Apply penalty: reduce reward if revisiting known regions
        # This encourages the agent to explore novel areas
        redundancy_penalty_amount = redundancy_penalty * (redundancy_score + overlap_penalty_val)
        adjusted_reward = reward - redundancy_penalty_amount
        
        raw_rewards.append(raw_reward)
        rewards.append(adjusted_reward)
        # ====================================================================
        # ✨ END LEVEL 2 ADDITIONS
        # ====================================================================
        
        steps += 1
        history.update_action(action.item())

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
        'raw_rewards': raw_rewards,  # ✨ NEW: Track unadjusted rewards
        'terminal_reward': terminal_reward,
        'steps': steps,
        'info': info,
        'history': history,
    }


# ============================================================================
# A2C Update (unchanged from Level 1)
# ============================================================================
def compute_advantages_and_returns(trajectory, gamma=GAMMA):
    """
    Compute returns and advantages using TD(0).
    
    (Same as Level 1 - no changes needed)
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
    
    (Same as Level 1 - no changes needed)
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
    
    # ====================================================================
    # ✨ BEGIN LEVEL 2 ADDITIONS: Add redundancy metrics
    # ====================================================================
    avg_raw_reward = np.mean(trajectory['raw_rewards']) if 'raw_rewards' in trajectory else 0.0
    avg_adjusted_reward = np.mean(trajectory['rewards'])
    redundancy_impact = avg_raw_reward - avg_adjusted_reward
    # ====================================================================
    # ✨ END LEVEL 2 ADDITIONS
    # ====================================================================
    
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
        'avg_raw_reward': avg_raw_reward,         # ✨ NEW
        'redundancy_impact': redundancy_impact,   # ✨ NEW
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
    output_dir='data/models/rl/a2c_lvl2',
    device=None,
    redundancy_penalty=REDUNDANCY_PENALTY,
    overlap_threshold=OVERLAP_THRESHOLD,
):
    """
    Main A2C Level 2 training loop.
    
    ✨ LEVEL 2: Trains redundancy-aware agent with spatial exploration incentives
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Training A2C Level 2 (Redundancy Avoidance) on device: {device}")
    print(f"State dimension: 520 (515 baseline + 3 history + 2 redundancy)")
    print(f"Redundancy penalty: {redundancy_penalty}")
    print(f"Overlap threshold: {overlap_threshold}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    embedder = Embedder()
    print(f"Using patch score: {patch_score}")
    
    # ========================================================================
    # ✨ LEVEL 2 CHANGE: state_dim = 520 (was 518 in Level 1)
    # ========================================================================
    model = ActorCritic(state_dim=520, hidden_dim=256).to(device)
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
            'redundancy_impact': [],  # ✨ NEW
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
                    trajectory = rollout_episode(env, model, device, redundancy_penalty, overlap_threshold)
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
        
        for key, values in epoch_metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{key:20s}: {mean_val:8.4f} ± {std_val:6.4f}")
        
        print(f"{'='*80}\n")
        
        checkpoint_path = os.path.join(output_dir, f"a2c_lvl2_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}\n")
    
    print("Training complete!")
    
    final_path = os.path.join(output_dir, "a2c_lvl2_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="A2C Level 2: Redundancy-aware patch zooming"
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
        default='data/models/rl/a2c_lvl2',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/mps/cpu)'
    )
    parser.add_argument(
        '--redundancy-penalty',
        type=float,
        default=REDUNDANCY_PENALTY,
        help='Redundancy penalty weight'
    )
    parser.add_argument(
        '--overlap-threshold',
        type=float,
        default=OVERLAP_THRESHOLD,
        help='Spatial overlap threshold'
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
        redundancy_penalty=args.redundancy_penalty,
        overlap_threshold=args.overlap_threshold,
    )


if __name__ == '__main__':
    main()

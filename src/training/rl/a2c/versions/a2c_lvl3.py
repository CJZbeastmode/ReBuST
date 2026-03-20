"""
Actor-Critic (A2C) Level 3: Contextual Memory (PAMIL-bridge)
==============================================================

CHANGES FROM LEVEL 2 (a2c_lvl2.py):
------------------------------------
✅ All Level 2 features retained:
   - Visited patch tracking
   - Visit count per location
   - Last action encoding
   - Depth trace
   - Redundancy penalty in reward function
   - Spatial overlap detection

✅ NEW Level 3 additions:
   - Parent patch embedding tracking (hierarchical context)
   - Parent-child embedding delta (relational features)
   - Has-parent indicator (for root vs child patches)
   - Enhanced state with hierarchical context (520 → 1033)

✅ State dimension expanded: 520 → 1033
   - Level 2: [level, x, y, embedding(512), history(5)]
   - Level 3: [... Level 2 ..., parent_embedding(512), has_parent(1)]

✅ Hierarchical context awareness:
   - Agent tracks which patch it zoomed from
   - Decisions consider parent-child relationships
   - Enables understanding of zoom history beyond just spatial location

KEY CLAIM UNLOCKED:
-------------------
"Agent decisions consider hierarchical zoom context, not just single patches"
- Clean conceptual bridge to PAMIL's hierarchical awareness
- Still local decision-making (no global bag aggregation)
- Interpretable parent-child relationships
- Demonstrates structured exploration through zoom tree

This is your THIRD extension, building on redundancy avoidance.
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
# Hyperparameters (FIXED)
# ============================================================================
GAMMA = 0.99                # Discount factor
LR = 3e-4                   # Learning rate (reduced for stability)
ENTROPY_BETA = 0.08         # Entropy regularization (increased to prevent STOP collapse)
VALUE_COEF = 0.5            # Critic loss weight
GRAD_CLIP = 1.0             # Gradient clipping
EPS = 1e-8                  # Numerical stability
REDUNDANCY_PENALTY = 0.2    # Penalty weight for revisiting regions
OVERLAP_THRESHOLD = 0.3     # Spatial proximity threshold for overlap
PARENT_PROJ_DIM = 64        # Project parent embedding from 512-D to 64-D


# ============================================================================
# History Tracker (EXTENDED for parent tracking)
# ============================================================================
class HistoryTracker:
    """
    Track visited patches, exploration history, and hierarchical context.
    
    ✨ LEVEL 3 EXTENSIONS:
    - Track parent patch embedding when zooming
    - Maintain zoom tree structure
    - Provide parent-child relational features
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
        self.visited_locations = []  # List of (level, x, y) for overlap detection
        
        # ====================================================================
        # ✨ BEGIN LEVEL 3 ADDITIONS
        # ====================================================================
        self.parent_embedding = None  # Embedding of patch we zoomed from (512-D)
        self.current_embedding = None  # Current patch embedding
        self.has_parent = False  # Whether current patch has a parent (not root)
        # ====================================================================
        # ✨ END LEVEL 3 ADDITIONS
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
        self.visited_locations.append((level, x, y))
        return self.visited[key]
    
    def get_visit_count(self, level, x, y):
        """Get visit count for a location (0 if never visited)."""
        key = self._hash_location(level, x, y)
        return self.visited.get(key, 0)
    
    def update_action(self, action, current_embedding=None):
        """
        Update the last action taken and hierarchical context.
        
        ✨ LEVEL 3: Now tracks parent embedding on ZOOM actions
        
        Parameters
        ----------
        action : int
            Action taken (0=STOP, 1=ZOOM)
        current_embedding : np.ndarray, optional
            Current patch embedding (512-D)
        """
        self.last_action = action
        
        # ====================================================================
        # ✨ BEGIN LEVEL 3 ADDITIONS: Track hierarchical context
        # ====================================================================
        if action == 1:  # ZOOM
            self.depth += 1
            # When we zoom, current patch becomes parent of next patch
            if current_embedding is not None:
                self.parent_embedding = current_embedding.copy()
                self.has_parent = True
        
        # Update current embedding for next iteration
        if current_embedding is not None:
            self.current_embedding = current_embedding.copy()
        # ====================================================================
        # ✨ END LEVEL 3 ADDITIONS
        # ====================================================================
    
    def compute_redundancy_score(self, level, x, y, threshold=OVERLAP_THRESHOLD):
        """
        Compute redundancy score based on spatial overlap with visited patches.
        
        (Same as Level 2 - no changes)
        """
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
        """
        Compute explicit penalty for being too close to visited regions.
        
        (Same as Level 2 - no changes)
        """
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
        """
        Extract history features for state augmentation.
        
        (Same as Level 2 - no changes to this part)
        
        Returns
        -------
        features : np.ndarray, shape (5,)
            [visit_count, last_action, depth, redundancy_score, overlap_penalty]
        """
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
    
    # ========================================================================
    # ✨ BEGIN LEVEL 3 ADDITIONS: Hierarchical context features
    # ========================================================================
    def get_hierarchical_features(self):
        """
        Extract hierarchical context features.
        
        Returns
        -------
        parent_emb : np.ndarray, shape (512,)
            Parent patch embedding (zeros if no parent)
        has_parent : float
            Binary indicator (1.0 if has parent, 0.0 otherwise)
        """
        if self.has_parent and self.parent_embedding is not None:
            return self.parent_embedding.copy(), 1.0
        else:
            # No parent (root patch) - return zero embedding
            return np.zeros(512, dtype=np.float32), 0.0
    # ========================================================================
    # ✨ END LEVEL 3 ADDITIONS
    # ========================================================================


# ============================================================================
# Actor-Critic Network (FIXED with projection)
# ============================================================================
class ActorCritic(nn.Module):
    """
    FIXED: Projects parent_embedding from 512-D → 64-D before concatenation.
    State: 1033-D → 585-D (515 + 5 + 64 + 1), hidden_dim increased to 512.
    """

    def __init__(self, base_state_dim=520, parent_emb_dim=512, parent_proj_dim=64, hidden_dim=512):
        super().__init__()
        
        # Project parent embedding to lower dimension
        self.parent_proj = nn.Sequential(
            nn.Linear(parent_emb_dim, parent_proj_dim),
            nn.ReLU(),
        )
        
        final_state_dim = base_state_dim + parent_proj_dim + 1  # 585

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
        """Split state, project parent embedding, forward pass."""
        base_state = x[:, :520]
        parent_emb = x[:, 520:1032]
        has_parent = x[:, 1032:1033]
        
        parent_proj = self.parent_proj(parent_emb)  # 512 → 64
        x_proj = torch.cat([base_state, parent_proj, has_parent], dim=1)  # 585
        
        h = self.encoder(x_proj)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value


# ============================================================================
# Hierarchical-aware state construction
# ============================================================================
def get_hierarchical_aware_state(env_state, history, level_norm, x_norm, y_norm):
    """
    Augment environment state with history and hierarchical context features.
    
    ✨ LEVEL 3: Now returns 1033-D state (was 520-D in Level 2)
    
    Parameters
    ----------
    env_state : np.ndarray, shape (515,)
        Original environment state [level, x, y, embedding(512)]
    history : HistoryTracker
        Episode history tracker with hierarchical context
    level_norm : float
        Normalized level
    x_norm : float
        Normalized x coordinate
    y_norm : float
        Normalized y coordinate
    
    Returns
    -------
    augmented_state : np.ndarray, shape (1033,)
        [original(515), history(5), parent_embedding(512), has_parent(1)]
    """
    # Get history features (redundancy, etc.)
    history_features = history.get_history_features(level_norm, x_norm, y_norm)
    
    # ====================================================================
    # ✨ BEGIN LEVEL 3 ADDITIONS: Add hierarchical context
    # ====================================================================
    parent_emb, has_parent = history.get_hierarchical_features()
    
    # Concatenate all features:
    # [env_state(515), history(5), parent_emb(512), has_parent(1)] → 1033-D
    return np.concatenate([
        env_state,           # 515-D
        history_features,    # 5-D
        parent_emb,          # 512-D (✨ NEW)
        [has_parent]         # 1-D (✨ NEW)
    ])
    # ====================================================================
    # ✨ END LEVEL 3 ADDITIONS
    # ====================================================================


# ============================================================================
# Episode Rollout (UPDATED with hierarchical context tracking)
# ============================================================================
def rollout_episode(env, model, device, redundancy_penalty=REDUNDANCY_PENALTY, overlap_threshold=OVERLAP_THRESHOLD):
    """
    Execute one complete episode with hierarchical-context-aware policy.
    
    ✨ LEVEL 3 CHANGES:
    - Track parent embeddings during zoom actions
    - Pass current embedding to history tracker
    - State now includes hierarchical context (1033-D)
    
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
        Contains states, actions, rewards, hierarchical context
    """
    history = HistoryTracker(grid_size=32)
    
    states = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    raw_rewards = []

    env_state = env.reset()
    done = False
    steps = 0

    while not done:
        # Extract normalized coordinates and embedding
        level_norm = env_state[0]
        x_norm = env_state[1]
        y_norm = env_state[2]
        current_embedding = env_state[3:]  # 512-D embedding
        
        # ====================================================================
        # ✨ LEVEL 3: Get hierarchical-aware state
        # ====================================================================
        state = get_hierarchical_aware_state(env_state, history, level_norm, x_norm, y_norm)
        # ====================================================================
        
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
        
        # Apply redundancy penalty (from Level 2)
        raw_reward = reward
        redundancy_score = history.compute_redundancy_score(level_norm, x_norm, y_norm, threshold=overlap_threshold)
        overlap_penalty_val = history.compute_overlap_penalty(level_norm, x_norm, y_norm)
        redundancy_penalty_amount = redundancy_penalty * (redundancy_score + overlap_penalty_val)
        adjusted_reward = reward - redundancy_penalty_amount
        
        raw_rewards.append(raw_reward)
        rewards.append(adjusted_reward)
        
        steps += 1
        
        # ====================================================================
        # ✨ LEVEL 3: Update action with current embedding for parent tracking
        # ====================================================================
        history.update_action(action.item(), current_embedding)
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
        'raw_rewards': raw_rewards,
        'terminal_reward': terminal_reward,
        'steps': steps,
        'info': info,
        'history': history,
    }


# ============================================================================
# A2C Update (same as Level 2)
# ============================================================================
def compute_advantages_and_returns(trajectory, gamma=GAMMA):
    """
    Compute returns and advantages using TD(0).
    
    (Same as Level 2 - no changes needed)
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
    
    (Same as Level 2 - no changes needed)
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
    
    avg_raw_reward = np.mean(trajectory['raw_rewards']) if 'raw_rewards' in trajectory else 0.0
    avg_adjusted_reward = np.mean(trajectory['rewards'])
    redundancy_impact = avg_raw_reward - avg_adjusted_reward
    
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
        'avg_raw_reward': avg_raw_reward,
        'redundancy_impact': redundancy_impact,
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
    output_dir='data/models/rl/a2c_lvl3',
    device=None,
    redundancy_penalty=REDUNDANCY_PENALTY,
    overlap_threshold=OVERLAP_THRESHOLD,
):
    """
    Main A2C Level 3 training loop.
    
    ✨ LEVEL 3: Trains hierarchical-context-aware agent
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Training A2C Level 3 (Contextual Memory - FIXED) on device: {device}")
    print(f"Raw state: 1033-D → Projected: 585-D (520 base + 64 proj_parent + 1 has_parent)")
    print(f"Hidden dim: 512 (was 256) | Entropy beta: {ENTROPY_BETA} (was 0.03)")
    print(f"Redundancy penalty: {redundancy_penalty} | Overlap threshold: {overlap_threshold}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    embedder = Embedder()
    print(f"Using patch score: {patch_score}")
    
    # FIXED: Use projection architecture
    model = ActorCritic(
        base_state_dim=520,
        parent_emb_dim=512,
        parent_proj_dim=PARENT_PROJ_DIM,
        hidden_dim=512
    ).to(device)
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
            'redundancy_impact': [],
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
        
        checkpoint_path = os.path.join(output_dir, f"a2c_lvl3_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}\n")
    
    print("Training complete!")
    
    final_path = os.path.join(output_dir, "a2c_lvl3_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="A2C Level 3: Hierarchical-context-aware patch zooming"
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
        default='data/models/rl/a2c_lvl3',
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

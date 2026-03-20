# Progressive A2C Architecture: Level 1 → Level 4

**Comprehensive Documentation of Evolutionary Enhancements**

Author: Jay (Implementation)  
Date: January 31, 2026  
Repository: PLIP-dynamic-patcher-MA

---

## Table of Contents

1. [Overview](#overview)
2. [Level 1: History Awareness (Raza-core)](#level-1-history-awareness)
3. [Level 2: Redundancy Avoidance (Raza-justification)](#level-2-redundancy-avoidance)
4. [Level 3: Contextual Memory (PAMIL-bridge)](#level-3-contextual-memory)
5. [Level 4: Multi-Step Returns (RL-maturity)](#level-4-multi-step-returns)
6. [Comparative Summary](#comparative-summary)
7. [Training Instructions](#training-instructions)
8. [Evaluation & Analysis](#evaluation--analysis)

---

## Overview

This document provides a **structural roadmap** of the progressive A2C implementations, detailing the exact changes made at each level. The progression follows a principled approach:

1. **Level 1**: Add basic history tracking (Raza paper foundation)
2. **Level 2**: Add explicit redundancy penalties (spatial exploration)
3. **Level 3**: Add hierarchical context (PAMIL-inspired parent-child relationships)
4. **Level 4**: Add multi-step returns (GAE for better credit assignment)

Each level **builds incrementally** on the previous one, ensuring clean conceptual separation and interpretability.

---

## Level 1: History Awareness

**File**: `a2c_lvl1.py`  
**Inspiration**: Raza et al. (2024) - "Core history tracking without complex logic"

### Key Changes from Baseline

#### 1. State Dimension Expansion
```
Baseline: 515-D [level, x, y, embedding(512)]
Level 1:  518-D [level, x, y, embedding(512), visit_count, last_action, depth]
```

#### 2. New Component: `HistoryTracker` Class

**Purpose**: Track exploration history within an episode

**Attributes**:
- `visited`: Dict mapping (level, x_bin, y_bin) → visit count
- `last_action`: Last action taken (0=STOP, 1=ZOOM)
- `depth`: Current zoom depth (increments on ZOOM)

**Methods**:
- `visit(level, x, y)`: Record a visit to a location
- `get_visit_count(level, x, y)`: Query visit count
- `update_action(action)`: Update last action and depth
- `get_history_features(level, x, y)`: Return [visit_count_norm, last_action, depth/10]

**Spatial Hashing**:
```python
def _hash_location(self, level, x, y):
    x_bin = int(x * self.grid_size)  # grid_size=32
    y_bin = int(y * self.grid_size)
    return (level, x_bin, y_bin)
```

#### 3. State Construction Function

**New**: `get_history_aware_state(env_state, history, level_norm, x_norm, y_norm)`

Concatenates:
- Environment state (515-D)
- History features (3-D)
→ Returns 518-D augmented state

#### 4. Modified Rollout

**Before** (baseline):
```python
state = env_state  # 515-D
```

**After** (Level 1):
```python
state = get_history_aware_state(env_state, history, level_norm, x_norm, y_norm)  # 518-D
history.visit(level_norm, x_norm, y_norm)
history.update_action(action.item())
```

#### 5. Actor-Critic Network

**Change**: `state_dim=518` (was 515)

**Architecture** (unchanged):
```
Input (518-D) → Encoder (256-D) → Actor (2-D logits) + Critic (1-D value)
```

### Claim Unlocked

> "Agent maintains memory of visited locations and exploration depth"

- Enables non-myopic decisions
- Foundation for spatial exploration strategies
- Clean, interpretable features

---

## Level 2: Redundancy Avoidance

**File**: `a2c_lvl2.py`  
**Inspiration**: Raza et al. (2024) - "Justify RL over greedy by avoiding redundant exploration"

### Key Changes from Level 1

#### 1. State Dimension Expansion
```
Level 1: 518-D [baseline(515), history(3)]
Level 2: 520-D [baseline(515), history(5)]
```

**New history features**: `redundancy_score` and `overlap_penalty`

#### 2. Extended `HistoryTracker`

**New Attributes**:
- `visited_locations`: List of (level, x, y) tuples for continuous coordinate tracking

**New Methods**:

**`compute_redundancy_score(level, x, y, threshold=0.3)`**:
- Computes spatial distance to all previously visited patches at the same level
- Returns normalized overlap count (0-1)
- Formula: `overlaps / total_visits_at_level`

**`compute_overlap_penalty(level, x, y)`**:
- Binary-like penalty for very recent revisits
- Checks last 5 visited locations
- Returns: 1.0 (very close), 0.5 (moderately close), 0.0 (no overlap)

**Updated `get_history_features()`**:
```python
return np.array([
    visit_count_norm,
    float(self.last_action),
    self.depth / 10.0,
    redundancy_score,      # NEW
    overlap_penalty,       # NEW
], dtype=np.float32)
```

#### 3. Reward Shaping

**New**: Explicit redundancy penalty applied to rewards

```python
raw_reward = reward

redundancy_score = history.compute_redundancy_score(level_norm, x_norm, y_norm)
overlap_penalty_val = history.compute_overlap_penalty(level_norm, x_norm, y_norm)

redundancy_penalty_amount = REDUNDANCY_PENALTY * (redundancy_score + overlap_penalty_val)
adjusted_reward = raw_reward - redundancy_penalty_amount
```

**Hyperparameters**:
- `REDUNDANCY_PENALTY = 0.2` (weight for penalty)
- `OVERLAP_THRESHOLD = 0.3` (spatial distance threshold)

#### 4. Tracking Metrics

**New metrics logged**:
- `avg_raw_reward`: Reward before redundancy adjustment
- `redundancy_impact`: Difference between raw and adjusted reward

**Purpose**: Analyze how much redundancy penalty affects training

#### 5. Actor-Critic Network

**Change**: `state_dim=520` (was 518)

### Claim Unlocked

> "Agent avoids re-focusing on previously explored regions"

- Strong motivation for RL over greedy (greedy has no memory)
- Interpretable: penalty directly encoded in reward
- Demonstrates intelligent spatial exploration

---

## Level 3: Contextual Memory

**File**: `a2c_lvl3.py`  
**Inspiration**: PAMIL (Zheng et al., 2024) - "Hierarchical parent-child relationships"

### Key Changes from Level 2

#### 1. State Dimension Expansion
```
Level 2: 520-D  [baseline(515), history(5)]
Level 3: 1033-D [baseline(515), history(5), parent_emb(512), has_parent(1)]
```

**Major expansion**: Add full parent patch embedding (512-D)

#### 2. Extended `HistoryTracker`

**New Attributes**:
- `parent_embedding`: Embedding (512-D) of the patch we zoomed from
- `current_embedding`: Current patch embedding
- `has_parent`: Boolean indicator (True if not root patch)

**Updated `update_action(action, current_embedding=None)`**:
```python
if action == 1:  # ZOOM
    self.depth += 1
    if current_embedding is not None:
        self.parent_embedding = current_embedding.copy()  # Save parent
        self.has_parent = True

if current_embedding is not None:
    self.current_embedding = current_embedding.copy()
```

**New Method**:
```python
def get_hierarchical_features(self):
    """
    Returns:
        parent_emb: 512-D embedding (zeros if no parent)
        has_parent: 1.0 or 0.0
    """
    if self.has_parent and self.parent_embedding is not None:
        return self.parent_embedding.copy(), 1.0
    else:
        return np.zeros(512, dtype=np.float32), 0.0
```

#### 3. State Construction

**Updated**: `get_hierarchical_aware_state(env_state, history, level_norm, x_norm, y_norm)`

```python
history_features = history.get_history_features(level_norm, x_norm, y_norm)  # 5-D
parent_emb, has_parent = history.get_hierarchical_features()  # 512-D + 1-D

return np.concatenate([
    env_state,           # 515-D
    history_features,    # 5-D
    parent_emb,          # 512-D (NEW)
    [has_parent]         # 1-D (NEW)
])  # Total: 1033-D
```

#### 4. Modified Rollout

**Change**: Pass `current_embedding` to `update_action()`

```python
current_embedding = env_state[3:]  # Extract 512-D embedding

state = get_hierarchical_aware_state(env_state, history, level_norm, x_norm, y_norm)

# After action execution:
history.update_action(action.item(), current_embedding)  # Pass embedding
```

#### 5. Actor-Critic Network

**Change**: `state_dim=1033` (was 520)

**Implication**: Network now processes parent-child relational information

### Claim Unlocked

> "Agent decisions consider hierarchical zoom context, not just single patches"

- Clean bridge to PAMIL's hierarchical awareness
- Enables understanding zoom tree structure
- Still local decisions (no global bag aggregation)

---

## Level 4: Multi-Step Returns

**File**: `a2c_lvl4.py`  
**Inspiration**: RL maturity - Standard PPO/A3C technique

### Key Changes from Level 3

#### 1. State Dimension (Unchanged)
```
Level 4: 1033-D (same as Level 3)
```

**No architectural changes** - improvements are in training algorithm only

#### 2. New Hyperparameter

```python
GAE_LAMBDA = 0.95  # GAE lambda parameter
```

**Meaning**:
- λ = 0: TD(0) (low variance, high bias)
- λ = 1: Monte Carlo (high variance, low bias)
- λ = 0.95: GAE (balanced)

#### 3. Advantage Computation Replacement

**Before** (Levels 1-3): `compute_advantages_and_returns()` using **TD(0)**

```python
# TD(0) advantage
for t in range(T):
    if t == T - 1:
        next_value = 0.0
    else:
        next_value = values_tensor[t + 1]
    
    advantages[t] = rewards_tensor[t] + gamma * next_value - values_tensor[t]
```

**After** (Level 4): `compute_gae_advantages_and_returns()` using **GAE(λ)**

```python
# GAE advantage
advantages = torch.zeros(T, device=values_tensor.device)
gae = 0.0

for t in reversed(range(T)):  # Backward pass
    if t == T - 1:
        next_value = 0.0
    else:
        next_value = values_tensor[t + 1]
    
    # TD error
    delta = rewards_tensor[t] + gamma * next_value - values_tensor[t]
    
    # GAE recursion
    gae = delta + gamma * gae_lambda * gae
    advantages[t] = gae

# Returns from advantages
returns = advantages + values_tensor
```

#### 4. Mathematical Formulation

**GAE Formula**:
$$
A_t^{GAE(\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

Where:
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**Interpretation**:
- Each advantage is an **exponentially-weighted sum** of TD errors
- Provides **eligibility traces** for credit assignment
- Smooths value estimates across time steps

#### 5. Updated A2C Update Function

**Signature change**:
```python
def a2c_update(model, optimizer, trajectory, 
               gamma=GAMMA, 
               gae_lambda=GAE_LAMBDA,  # NEW parameter
               entropy_beta=ENTROPY_BETA, 
               value_coef=VALUE_COEF):
```

**Call change**:
```python
returns, advantages = compute_gae_advantages_and_returns(trajectory, gamma, gae_lambda)
```

### Claim Unlocked

> "Agent learns long-horizon value propagation"

- Better credit assignment for delayed rewards
- Reduced variance in advantage estimates
- Standard technique in modern RL (PPO uses GAE)

---

## Comparative Summary

### State Dimensions

| Level    | Dimension | Components                                                                 |
|----------|-----------|---------------------------------------------------------------------------|
| Baseline | 515       | `[level, x, y, embedding(512)]`                                           |
| Level 1  | 518       | `[baseline(515), visit_count, last_action, depth]`                        |
| Level 2  | 520       | `[baseline(515), history(5): visit, action, depth, redundancy, overlap]`  |
| Level 3  | 1033      | `[baseline(515), history(5), parent_emb(512), has_parent(1)]`             |
| Level 4  | 1033      | `[same as Level 3]` (no state changes)                                    |

### Feature Evolution

| Feature                      | Lvl 1 | Lvl 2 | Lvl 3 | Lvl 4 |
|------------------------------|-------|-------|-------|-------|
| Visit count tracking         | ✅     | ✅     | ✅     | ✅     |
| Last action encoding         | ✅     | ✅     | ✅     | ✅     |
| Depth tracking               | ✅     | ✅     | ✅     | ✅     |
| Redundancy score             | ❌     | ✅     | ✅     | ✅     |
| Overlap penalty              | ❌     | ✅     | ✅     | ✅     |
| Parent embedding             | ❌     | ❌     | ✅     | ✅     |
| Has-parent indicator         | ❌     | ❌     | ✅     | ✅     |
| GAE (multi-step returns)     | ❌     | ❌     | ❌     | ✅     |

### Algorithm Evolution

| Component                    | Baseline | Lvl 1  | Lvl 2  | Lvl 3  | Lvl 4  |
|------------------------------|----------|--------|--------|--------|--------|
| Advantage estimation         | TD(0)    | TD(0)  | TD(0)  | TD(0)  | GAE(λ) |
| Reward shaping               | Baseline | Same   | +Redun | Same   | Same   |
| History tracking             | None     | Basic  | +Spat  | +Hier  | Same   |
| State dimensionality         | 515      | 518    | 520    | 1033   | 1033   |

### Hyperparameters

| Parameter               | Default | Tunable | Description                          |
|-------------------------|---------|---------|--------------------------------------|
| `GAMMA`                 | 0.99    | Yes     | Discount factor                      |
| `LR`                    | 5e-4    | Yes     | Learning rate                        |
| `ENTROPY_BETA`          | 0.03    | Yes     | Entropy regularization weight        |
| `VALUE_COEF`            | 0.5     | Yes     | Critic loss weight                   |
| `GRAD_CLIP`             | 1.0     | Yes     | Gradient clipping threshold          |
| `REDUNDANCY_PENALTY`    | 0.2     | Yes     | Redundancy penalty weight (Lvl 2+)   |
| `OVERLAP_THRESHOLD`     | 0.3     | Yes     | Spatial overlap threshold (Lvl 2+)   |
| `GAE_LAMBDA`            | 0.95    | Yes     | GAE lambda parameter (Lvl 4)         |

---

## Training Instructions

### Environment Setup

```bash
# Ensure Python environment is configured
cd /Users/jay/Desktop/MA

# Verify dependencies
python -c "import torch; import numpy; print('Dependencies OK')"
```

### Training Commands

#### Level 1: History Awareness
```bash
python src/training/rl/a2c/a2c_lvl1.py \
    --images-dir data/images \
    --epochs 10 \
    --episodes-per-image 5 \
    --output-dir data/models/rl/a2c_lvl1
```

#### Level 2: Redundancy Avoidance
```bash
python src/training/rl/a2c/a2c_lvl2.py \
    --images-dir data/images \
    --epochs 10 \
    --episodes-per-image 5 \
    --redundancy-penalty 0.2 \
    --overlap-threshold 0.3 \
    --output-dir data/models/rl/a2c_lvl2
```

#### Level 3: Contextual Memory
```bash
python src/training/rl/a2c/a2c_lvl3.py \
    --images-dir data/images \
    --epochs 10 \
    --episodes-per-image 5 \
    --redundancy-penalty 0.2 \
    --overlap-threshold 0.3 \
    --output-dir data/models/rl/a2c_lvl3
```

#### Level 4: Multi-Step Returns
```bash
python src/training/rl/a2c/a2c_lvl4.py \
    --images-dir data/images \
    --epochs 10 \
    --episodes-per-image 5 \
    --redundancy-penalty 0.2 \
    --overlap-threshold 0.3 \
    --gae-lambda 0.95 \
    --output-dir data/models/rl/a2c_lvl4
```

### Syntax Validation

Before training, validate syntax:

```bash
python -m py_compile src/training/rl/a2c/a2c_lvl1.py
python -m py_compile src/training/rl/a2c/a2c_lvl2.py
python -m py_compile src/training/rl/a2c/a2c_lvl3.py
python -m py_compile src/training/rl/a2c/a2c_lvl4.py
```

---

## Evaluation & Analysis

### Metrics Logged

All levels track:
- `loss`: Total A2C loss
- `policy_loss`: Policy gradient loss
- `value_loss`: Critic MSE loss
- `entropy`: Policy entropy (exploration measure)
- `terminal_reward`: Final step reward
- `episode_return`: Cumulative discounted return
- `steps`: Episode length
- `mean_stop_prob`: P(STOP action)
- `mean_zoom_prob`: P(ZOOM action)

**Level 2+** additionally tracks:
- `avg_raw_reward`: Reward before redundancy penalty
- `redundancy_impact`: Penalty amount applied

### Comparative Analysis Questions

1. **Does history tracking improve over baseline?**
   - Compare Level 1 vs Baseline: episode returns, exploration diversity

2. **Does redundancy penalty encourage spatial diversity?**
   - Analyze `redundancy_impact` in Level 2
   - Compare visited patch heatmaps: Level 2 vs Level 1

3. **Does hierarchical context improve decision quality?**
   - Compare Level 3 vs Level 2: value function stability
   - Analyze zoom tree structures

4. **Does GAE improve learning stability?**
   - Compare Level 4 vs Level 3: advantage variance, convergence speed
   - Ablation study: λ ∈ {0.0, 0.5, 0.95, 1.0}

### Visualization Recommendations

1. **Training Curves**: Plot loss, returns, entropy over time
2. **Spatial Heatmaps**: Visualize visited patch locations per level
3. **Zoom Tree**: Show parent-child relationships (Level 3+)
4. **Advantage Distributions**: Compare TD(0) vs GAE variance
5. **Redundancy Impact**: Plot penalty over episodes (Level 2+)

---

## Conclusion

This progressive architecture demonstrates **principled incremental enhancement** of A2C for hierarchical patch exploration:

- **Level 1** establishes history awareness (foundation)
- **Level 2** adds spatial exploration intelligence (RL advantage)
- **Level 3** bridges to PAMIL-style hierarchical reasoning
- **Level 4** matures the training algorithm with GAE

Each level is **independently functional**, **interpretable**, and **builds cleanly** on previous work. This structure supports ablation studies and provides clear narrative for thesis contributions.

---

**End of Documentation**

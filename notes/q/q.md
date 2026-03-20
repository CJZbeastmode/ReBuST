Below is a **README-style explanation for Q-learning with a zoom-budget constraint**, written to **mirror your A2C README**, but **honest to Q-learning** (no fake advantages, no policy gradients).
It is designed so you can **reuse phrasing, structure, and symbol definitions** in a thesis or report.

I’ll be explicit about **symbols**, **their meaning**, and **where they appear in code**.

---

# Constrained Q-Learning for STOP / ZOOM Decisions

## Overview

This implementation uses **Q-learning** to learn STOP vs ZOOM decisions in a hierarchical Whole Slide Image (WSI) environment, subject to a **zoom-budget constraint**.

The goal is to learn a policy that:

* maximizes task-specific reward (patch quality),
* while limiting excessive zoom actions (resource constraint).

The constraint is enforced using a **Lagrangian relaxation**, making the method comparable to constrained A2C, but within an **off-policy value-based** framework.

---

## Environment Assumptions

* **Actions**

  * `a = 0` → STOP (terminate exploration)
  * `a = 1` → ZOOM (descend one pyramid level)

* **State**

  * ( s_t \in \mathbb{R}^D )
  * Concatenation of:

    * normalized spatial coordinates ((\text{level}, x, y))
    * patch embedding (e.g. PLIP, 512-dim)

* **Episode termination**

  * STOP action
  * maximum zoom depth reached
  * invalid zoom (no children)

---

## Core Symbols and Their Meaning

### Time index

* ( t \in {0, 1, \dots, T-1} )
* Index within one episode

---

### State

* ( s_t )
* Environment observation at time (t)

**In code**

```python
state = torch.tensor(env.reset(), ...)
```

---

### Action

* ( a_t \in {0, 1} )

**Meaning**

* 0 = STOP
* 1 = ZOOM

**In code**

```python
action = agent.act(state)
```

---

### Reward (task reward)

* ( r_t )

**Meaning**

* Absolute score returned by the environment for the chosen action
* Depends only on ((s_t, a_t))

**Important**

* No zoom penalty is included here

**In code**

```python
next_state, reward, done, info = env.step(action)
```

---

### Cost (constraint signal)

* ( c_t )

[
c_t =
\begin{cases}
1 & \text{if } a_t = \text{ZOOM} \
0 & \text{if } a_t = \text{STOP}
\end{cases}
]

**Meaning**

* Counts resource usage (zoom operations)

**In code**

```python
cost = 1.0 if action == 1 else 0.0
```

---

### Zoom budget

* ( \alpha \in (0,1] )

**Meaning**

* Maximum allowed fraction of ZOOM actions per episode

[
\frac{1}{T} \sum_{t=0}^{T-1} c_t \le \alpha
]

**Example**

* ( \alpha = 0.5 ) → at most 50% zoom actions

---

### Lagrange multiplier (zoom price)

* ( \lambda \ge 0 )

**Meaning**

* Adaptive penalty applied to ZOOM actions
* Controls how expensive zooming is

**In code**

```python
lambda_zoom = 1.0
```

---

## Effective Reward (Constraint Integration)

Q-learning does **not** handle constraints natively.
We therefore modify the reward using **Lagrangian relaxation**.

### Effective reward

[
\tilde r_t = r_t - \lambda \cdot c_t
]

**Meaning**

* ZOOM actions are penalized proportional to current constraint pressure
* STOP actions are unaffected

**In code**

```python
reward_eff = reward - lambda_zoom * cost
```

This is the reward actually used in the Bellman update.

---

## Q-Function

### Definition

[
Q(s, a) = \mathbb{E}\left[ \sum_{k=0}^{\infty} \gamma^k \tilde r_{t+k}
;\middle|; s_t = s, a_t = a \right]
]

**Meaning**

* Expected future **penalized** return when taking action (a) in state (s)

---

### Neural approximation

The Q-network outputs:

[
Q_\theta(s) =
\begin{bmatrix}
Q(s, \text{STOP}) \
Q(s, \text{ZOOM})
\end{bmatrix}
]

**In code**

```python
q_vals = q_net(state)  # shape [2]
```

---

## Bellman Update (Q-learning)

### Target

[
y_t = \tilde r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a')
]

where:

* ( \theta^- ) are target network parameters
* If ( s_{t+1} ) is terminal, the second term is zero

**In code**

```python
target = rewards + gamma * (1 - done) * q_next
```

---

### Loss

[
\mathcal{L}*{Q} = \left(Q*\theta(s_t, a_t) - y_t\right)^2
]

**In code**

```python
loss = F.mse_loss(q_sa, target)
```

---

## Policy (Action Selection)

Q-learning does not learn a policy explicitly.

### ε-greedy policy

[
\pi(s) =
\begin{cases}
\text{random action} & \text{with probability } \varepsilon \
\arg\max_a Q(s,a) & \text{otherwise}
\end{cases}
]

**In code**

```python
if random.random() < eps:
    action = random.randint(0, 1)
else:
    action = argmax Q(s)
```

---

## Constraint Update (Dual Variable)

After each episode, update the zoom price:

### Constraint violation

[
g = \frac{1}{T}\sum_{t=0}^{T-1} c_t - \alpha
]

---

### Lagrange update

[
\lambda \leftarrow \max\left(0,; \lambda + \eta_\lambda \cdot g\right)
]

where:

* ( \eta_\lambda ) is `LAMBDA_LR`

**In code**

```python
lambda_zoom += LAMBDA_LR * (zoom_fraction - ZOOM_BUDGET)
lambda_zoom = max(0.0, lambda_zoom)
```

---

## Full Algorithm (High-Level)

1. Initialize Q-network (Q_\theta) and target network (Q_{\theta^-})
2. Initialize replay buffer
3. Initialize zoom price (\lambda)
4. For each episode:

   1. Reset environment
   2. For each step:

      * select action via ε-greedy
      * observe (r_t, c_t, s_{t+1})
      * compute (\tilde r_t = r_t - \lambda c_t)
      * store transition
      * update Q-network via Bellman loss
   3. Compute zoom fraction
   4. Update (\lambda)
5. Periodically update target network

---

## Key Differences to A2C (Important)

| Aspect              | A2C       | Q-learning  |
| ------------------- | --------- | ----------- |
| Learning type       | On-policy | Off-policy  |
| Value learned       | (V(s))    | (Q(s,a))    |
| Policy              | Explicit  | Implicit    |
| Entropy             | Yes       | No          |
| Constraint handling | Clean     | Approximate |
| Stability           | Higher    | Lower       |

---

## Interpretation in This Project

* Q-learning serves as a **baseline**
* Constraint is enforced **in expectation**, not strictly
* High variance is expected due to:

  * off-policy updates
  * replay buffer
  * non-stationary (\lambda)

Despite this, the formulation is:

* mathematically consistent
* comparable to constrained A2C
* defensible in an academic context

---

## One-Sentence Intuition

> **Q-learning estimates how good STOP or ZOOM is after paying a dynamically adjusted zoom price.**

---

If you want next, I can:

* shorten this into a **thesis-ready Methods subsection**
* add **pseudo-code**
* give a **comparison paragraph (A2C vs Q-learning)**
* or help you justify why Q-learning underperforms (if it does)

Just tell me.

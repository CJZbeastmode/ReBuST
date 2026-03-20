Below is a **thorough documentation (“Doku”)** for your **constrained contextual bandit (REINFORCE)** baseline, written in the same style as your A2C docs: **step-by-step**, **symbol-by-symbol**, and with a **fully concrete numeric example**.

---

# Constrained Contextual Bandit (REINFORCE) for STOP / ZOOM

## 1) What this baseline is

This baseline treats each STOP/ZOOM decision as a **contextual bandit** problem:

* You observe a context (the current patch state) (s)
* You choose one action (a \in {0,1})
* You get an immediate reward (r)
* The interaction ends immediately (no long trajectory)

This is intentionally simpler than A2C/Q-learning:

* It **does not model long-term consequences** of zooming.
* It optimizes **immediate patch score**.

In your project this is still useful as:

* sanity check for the score signal
* cheap baseline
* “lower-capacity” reference in evaluation

---

## 2) Key objects and symbols

### State (context)

* (s \in \mathbb{R}^D)

In your env, (s) contains:

* normalized coordinates: ((\text{level}, x, y))
* embedding vector (e.g., PLIP 512-dim)

**In code**

```python
state = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device)
```

---

### Action

* (a \in {0,1})

Meaning:

* (a=0): STOP
* (a=1): ZOOM

**In code**

```python
action = dist.sample()   # a ~ πθ(.|s)
```

---

### Reward (task reward)

* (r = r(s,a))

Meaning:

* immediate score returned by the environment for the selected action
* should depend only on current state and chosen action

**In code**

```python
_, reward, _, info = env.step(action.item())
```

---

### Cost (zoom usage)

* (c = c(s,a))

We define:
[
c =
\begin{cases}
1 & \text{if } a=\text{ZOOM} \
0 & \text{if } a=\text{STOP}
\end{cases}
]

**In code**

```python
cost = 1.0 if action.item() == 1 else 0.0
```

---

### Zoom budget

* (\alpha \in (0,1])

Meaning: the maximum allowed fraction of zoom actions.

[
\mathbb{E}[c] \le \alpha
]

Example:

* (\alpha = 0.5) means “zoom at most 50% of the time (in expectation).”

---

### Lagrange multiplier (zoom price)

* (\lambda \ge 0)

Meaning:

* adaptive “price” for zoom actions
* increases if you violate the budget
* decreases (or stays low) if you respect it

**In code**

```python
lambda_zoom = 1.0
```

---

## 3) Policy model

### Policy distribution

We learn a stochastic policy:

[
\pi_\theta(a \mid s)
]

Implemented as a neural network outputting logits:

[
\text{logits} = f_\theta(s) \in \mathbb{R}^2
]
[
\pi_\theta(a \mid s) = \text{softmax}(\text{logits})_a
]

**In code**

```python
logits = policy(state.unsqueeze(0))
dist = Categorical(logits=logits)
```

---

## 4) REINFORCE objective (bandit version)

We want to maximize expected reward:

[
J(\theta) = \mathbb{E}*{a\sim \pi*\theta(\cdot|s)}[r]
]

The policy gradient for a bandit is:

[
\nabla_\theta J(\theta)
=======================

\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s); r\right]
]

This is why the loss is:

[
\mathcal{L}(\theta) = -\log \pi_\theta(a|s); r
]

**In code**

```python
loss = -dist.log_prob(action) * reward
```

Negative sign because we *minimize* loss but want to *maximize* reward.

---

## 5) Baseline for variance reduction

REINFORCE gradients are noisy. A standard trick is subtracting a baseline (b) that does not depend on the action:

[
\nabla_\theta J(\theta)
=======================

\mathbb{E}\left[\nabla_\theta \log \pi(a|s); (r - b)\right]
]

We define the **advantage**:

[
A = r - b
]

### Baseline update (EMA)

Your code uses an exponential moving average baseline:

[
b \leftarrow (1-\beta) b + \beta r
]

**In code**

```python
baseline = (1 - beta) * baseline + beta * reward_eff
advantage = reward_eff - baseline
```

Meaning:

* if reward is higher than baseline → action is reinforced
* if reward is lower than baseline → action probability is reduced

---

## 6) Constraint integration (Lagrangian)

Bandits do not enforce constraints directly, so we use Lagrangian relaxation:

### Effective reward

[
\tilde r = r - \lambda c
]

Meaning:

* zoom actions “pay” a price (\lambda)
* stop actions pay 0 cost

**In code**

```python
reward_eff = reward - lambda_zoom * cost
```

Then REINFORCE uses (\tilde r) (not (r)):

[
\mathcal{L}(\theta) = -\log \pi_\theta(a|s); (\tilde r - b)
]

---

## 7) Dual update for λ (enforcing budget)

We track average zoom usage:

[
\hat c = \frac{\text{zoom_count}}{\text{step_count}}
]

Constraint violation:

[
g = \hat c - \alpha
]

Update:

[
\lambda \leftarrow \max(0,\lambda + \eta_\lambda \cdot g)
]

* If zoom usage is too high → (g>0) → (\lambda) increases → zoom becomes more expensive
* If zoom usage is low → (g<0) → (\lambda) decreases (or stops rising)

**In code**

```python
zoom_fraction = zoom_count / step_count
lambda_zoom += lambda_lr * (zoom_fraction - zoom_budget)
lambda_zoom = max(0.0, lambda_zoom)
```

---

# 8) Step-by-step algorithm (what happens every training iteration)

Each iteration is one “bandit sample”:

### Step 1 — reset (sample context)

* sample a random patch location and get its embedding state

```python
state = env.reset()
```

### Step 2 — policy forward pass

* compute logits and distribution

```python
logits = policy(state)
dist = Categorical(logits=logits)
```

### Step 3 — sample action

* stochastic exploration is built-in

```python
action = dist.sample()
```

### Step 4 — compute immediate reward from env

* execute one step to get reward for the chosen action

```python
_, reward, _, _ = env.step(action.item())
```

### Step 5 — compute zoom cost

```python
cost = 1.0 if action == 1 else 0.0
```

### Step 6 — effective reward (constraint)

```python
reward_eff = reward - lambda_zoom * cost
```

### Step 7 — update baseline and compute advantage

```python
baseline = EMA(baseline, reward_eff)
advantage = reward_eff - baseline
```

### Step 8 — REINFORCE update

```python
loss = -log_prob(action) * advantage
```

### Step 9 — update λ (dual step)

```python
lambda_zoom += lambda_lr * (zoom_fraction - zoom_budget)
```

---

# 9) Concrete numeric example (single training step)

Assume:

* Budget: (\alpha=0.5)
* Current λ: (\lambda=0.8)
* Baseline: (b=0.2)
* Baseline rate: (\beta=0.01)

Policy outputs probabilities:

[
\pi(\text{STOP}|s)=0.4,\quad \pi(\text{ZOOM}|s)=0.6
]

Sampled action:
[
a=\text{ZOOM}
]

Environment returns reward:
[
r = 0.7
]

Cost:
[
c=1
]

Effective reward:
[
\tilde r = 0.7 - 0.8\cdot 1 = -0.1
]

Update baseline:
[
b \leftarrow 0.99\cdot 0.2 + 0.01\cdot (-0.1)=0.198 - 0.001 = 0.197
]

Advantage:
[
A = \tilde r - b = -0.1 - 0.197 = -0.297
]

Log prob for zoom:
[
\log \pi(\text{ZOOM}|s) = \log(0.6) \approx -0.511
]

Loss:
[
\mathcal{L} = -\log \pi(a|s)\cdot A
= -(-0.511)\cdot(-0.297)
= -0.152
]

Interpretation:

* Advantage is negative → we reduce probability of choosing ZOOM in similar states
* Because zoom was too expensive at current λ

---

# 10) What this baseline learns (and what it cannot)

### Learns

* mapping from patch embedding → probability of STOP vs ZOOM
* adapts zoom usage via λ to meet budget in expectation

### Cannot learn

* long-term benefit of zooming deeper
* sequence planning (zoom now to get better later)
* depth-dependent policies except indirectly via state coords

So expect:

* bandit performs worse than A2C on problems where “zoom now helps later”

That’s normal.

---

# 11) Practical tips (so it actually trains)

* Use **small λ learning rate** (`lambda_lr = 0.01`)
* Log:

  * `zoom_fraction`
  * `lambda_zoom`
  * average reward
* If `lambda` explodes:

  * reduce `lambda_lr`
  * reduce reward clipping
* If zoom_fraction stays too high:

  * increase `lambda_lr` slightly
  * increase initial `lambda_zoom`

---

If you want the next step after docs: I can write a **Methods subsection** (thesis-ready) for bandit + constraint in 1–2 pages, with equations but without the long explanation.

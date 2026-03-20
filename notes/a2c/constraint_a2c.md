Perfect — let’s do this **cleanly, slowly, and symbol-by-symbol again**, but now with a **zoom-budget constraint integrated into A2C**.
I’ll use **one single approach** so you don’t get confused:

> **A2C + Lagrangian constraint (CMDP)**
> This is the *theoretically correct* way to enforce “≤ 50% / 70% zoom”.

No shortcuts, no mixing methods.

---

# 0. What changes compared to plain A2C?

Plain A2C:

* maximizes **reward**

Constrained A2C:

* maximizes **reward**
* while keeping **zoom usage below a budget**

So we now optimize **two things**:

1. reward (what you want)
2. cost (what you must limit)

---

# 1. New object: cost signal

We introduce a **cost per step**.

### Symbol: ( c_t )

[
c_t =
\begin{cases}
1 & \text{if } a_t = \text{ZOOM} \
0 & \text{if } a_t = \text{STOP}
\end{cases}
]

### Meaning (plain English)

> “Did I consume zoom budget at this step?”

---

### Example episode

| t | action | reward (r_t) | cost (c_t) |
| - | ------ | ------------ | ---------- |
| 0 | ZOOM   | +0.4         | 1          |
| 1 | ZOOM   | +0.1         | 1          |
| 2 | STOP   | 0.0          | 0          |

---

# 2. Total cost of an episode

### Symbol: ( C(\tau) )

[
C(\tau) = \sum_{t=0}^{T-1} c_t
]

### Meaning

> “How many zooms did I use in this episode?”

---

### Zoom fraction

[
f_{\text{zoom}} = \frac{C(\tau)}{T}
]

You want:

[
f_{\text{zoom}} \le \alpha
]

Where:

* ( \alpha = 0.5 ) → 50% zoom budget
* ( \alpha = 0.7 ) → 70% zoom budget

---

# 3. The constrained optimization problem

Now the **real math**.

### Objective (unchanged)

[
\max_\pi ;; \mathbb{E}*\pi\left[\sum*{t=0}^{T-1} \gamma^t r_t\right]
]

### Constraint (new)

[
\mathbb{E}_\pi[C(\tau)] \le \alpha T
]

This is called a **Constrained Markov Decision Process (CMDP)**.

---

# 4. Lagrangian relaxation (key idea)

We convert the constrained problem into an unconstrained one using a **Lagrange multiplier**.

---

## 4.1 New symbol: ( \lambda )

[
\lambda \ge 0
]

### Meaning

> “How expensive is zoom right now?”

* small ( \lambda ) → zoom is cheap
* large ( \lambda ) → zoom is heavily punished

---

## 4.2 Lagrangian objective

[
\mathcal{L}(\pi, \lambda)
=========================

## \mathbb{E}_\pi[R(\tau)]

\lambda \left(
\mathbb{E}_\pi[C(\tau)] - \alpha T
\right)
]

---

### Expand it

# [

\mathbb{E}*\pi\left[
\sum*{t} r_t
------------

\lambda \sum_{t} c_t
\right]
+
\lambda \alpha T
]

The last term does **not** affect gradients → ignore it.

---

# 5. Effective reward (this is what the agent sees)

### Symbol: ( r_t^{\text{eff}} )

[
r_t^{\text{eff}} = r_t - \lambda c_t
]

### Meaning

> “Reward minus zoom cost”

---

### Example

If:

* ( r_t = +0.4 )
* ( \lambda = 0.6 )
* action = ZOOM → ( c_t = 1 )

Then:
[
r_t^{\text{eff}} = 0.4 - 0.6 = -0.2
]

Zoom is no longer worth it.

---

# 6. A2C equations with constraint (actor side)

Nothing changes structurally — **only the reward**.

---

## 6.1 Return

[
G_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k}^{\text{eff}} + \gamma^n V(s_{t+n})
]

---

## 6.2 Advantage

[
\hat A_t = G_t - V(s_t)
]

### Meaning

> “Was this action better or worse **after accounting for zoom cost**?”

---

## 6.3 Policy gradient

[
\nabla_\theta J
===============

\mathbb{E}\left[
\nabla_\theta \log \pi_\theta(a_t\mid s_t); \hat A_t
\right]
]

Same as before — the constraint enters **only via ( r_t^{\text{eff}} )**.

---

# 7. Critic equations (unchanged, but learning new values)

[
\mathcal{L}_{\text{value}} =
\left(V(s_t) - G_t\right)^2
]

### Meaning

> “Predict future **penalized** return”

The critic learns that:

* deep zoom paths are expensive
* shallow paths are safer

---

# 8. Entropy (unchanged)

[
\mathcal{L}_{\text{entropy}} = -\beta H(\pi(\cdot\mid s_t))
]

---

# 9. Full constrained A2C loss

[
\mathcal{L} =
-\log \pi(a_t\mid s_t)\hat A_t
+
c_v (G_t - V(s_t))^2
--------------------

\beta H(\pi)
]

**Exactly the same form** — only the reward changed.

---

# 10. Updating the Lagrange multiplier ( \lambda )

This is the **dual update**.

### Constraint violation signal

[
g = \frac{C(\tau)}{T} - \alpha
]

---

### Update rule

[
\lambda \leftarrow \max(0,; \lambda + \eta g)
]

### Meaning

* If zoom fraction > budget → increase penalty
* If zoom fraction < budget → relax penalty
* Clamp to ≥ 0 (important)

---

### Example

Budget ( \alpha = 0.5 )

Episode:

```
T = 10
zoom_count = 7
```

[
g = 0.7 - 0.5 = 0.2
]

If:

* ( \lambda = 0.4 )
* ( \eta = 0.1 )

[
\lambda \leftarrow 0.4 + 0.1·0.2 = 0.42
]

Zoom becomes more expensive next episode.

---

# 11. Full algorithm (mental picture)

1. Initialize policy ( \pi_\theta ), value ( V_\phi ), ( \lambda )
2. Run episode using current policy
3. Compute:

   * rewards ( r_t )
   * costs ( c_t )
   * effective rewards ( r_t^{\text{eff}} )
4. Compute A2C loss → update ( \theta, \phi )
5. Measure zoom fraction
6. Update ( \lambda )
7. Repeat

---

# 12. Why this is correct (intuition)

> **Actor:** “Choose actions that give good reward *after paying zoom tax*.”
> **Critic:** “Predict future value *including zoom cost*.”
> **λ:** “Raise taxes if zoom is abused, lower taxes if it’s underused.”

This is **not a heuristic**. This is textbook CMDP optimization.

---

# 13. Why this is better than your current penalty

| Your current approach    | Lagrangian A2C                 |
| ------------------------ | ------------------------------ |
| History-dependent reward | Markovian                      |
| Manual tuning            | Automatic                      |
| Critic instability       | Stable                         |
| No guarantees            | Budget enforced in expectation |

---

# 14. Final takeaway (remember this)

> **Constraints live outside the policy.
> The policy only sees prices.**

If you want, next I can:

* map this **exactly** to your Python variables
* give a drop-in patch for your script
* or show how to log constraint satisfaction properly for a thesis

Just say which.

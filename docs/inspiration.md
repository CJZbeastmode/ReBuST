Good, *this* is the right question.
Short answer first, then the mapping.

**Yes** — some of these papers are *structurally compatible* with a **DQN-like** formulation, others are **strictly A2C / policy-gradient territory**, and mixing that up is exactly why DQN keeps collapsing to STOP in your experiments.

Below is a **hard separation**, no hand-waving.

---

## 1️⃣ Papers that can inspire a **DQN-style** agent

These papers treat patch selection as **value / utility estimation** or **search**, not as a stochastic policy.

### ✅ **DC-WSI (Fan et al., 2024)**

**Why DQN-compatible**

* Implicitly estimates **patch value** (“representative” vs redundant)
* Selection is **deterministic** once scores are known
* No stochastic policy, no entropy terms

**How it maps to DQN**

* Your Q(s, zoom) ≈ *expected marginal utility of zooming here*
* Could inspire:

  * Patch-value heads
  * Q-value interpreted as *expected bag improvement*

**But**

* DC-WSI is *non-sequential*
* Replay buffer breaks the bag semantics

➡️ **Usable inspiration, but only conceptually**

---

### ✅ **SPLICE (Alsaafin et al., 2024)**

**Why DQN-compatible**

* Greedy, value-based selection
* Explicit redundancy penalty
* Sequential but deterministic

**How it maps**

* Q(s, zoom) ≈ novelty gain
* STOP when marginal gain ≤ 0

**This fits DQN better than PAMIL**

* No stochasticity
* No trajectory-level credit assignment

➡️ **Best heuristic inspiration for DQN**

---

### ⚠️ **EvoPS (Hashemian & Bidgoli, 2025)**

**Why partially compatible**

* Optimizes patch *sets*
* Evaluates subsets by final performance

**Why not clean DQN**

* Objective is **set-level**, not step-level
* Evolution ≠ temporal difference

**But**

* Supports the idea that *patch utility exists*
* Justifies learning a **value signal**

➡️ **Conceptual support, not algorithmic**

---

## 2️⃣ Papers that are **A2C / policy-gradient only**

These **cannot** be translated into DQN without breaking their logic.

### ❌ **PAMIL (Zheng et al., 2024)**

**Why DQN fails here**

* Reward depends on **trajectory history**
* Contrastive reward is non-Markovian
* Explicit stochastic policy required

**What it needs**

* Entropy regularization
* On-policy updates
* Memory of past actions

➡️ **Pure A2C / PPO territory**

---

### ❌ **Raza et al. (2024)** — Dual attention + RL

**Why DQN is wrong**

* Hard attention trained via REINFORCE
* Actions are **glimpses**, not choices among discrete Q-values
* Reward delayed to final classification

**Why A2C works**

* Handles sparse terminal reward
* Handles hierarchical decisions (low → high mag)

➡️ **Strong A2C inspiration**

---

### ❌ **AAMM (Dang et al., 2025)**

**Why neither DQN nor A2C (directly)**

* Anomaly detection, not policy learning
* No action space

**But**

* Could be used as a **reward shaping module**
* Or pre-filter before RL

➡️ **Auxiliary, not core RL inspiration**

---

## 3️⃣ Clean mapping (this is the key table)

| Paper       | DQN | A2C / PPO | Why                     |
| ----------- | --- | --------- | ----------------------- |
| DC-WSI      | ✅   | ⚠️        | Value-oriented, static  |
| SPLICE      | ✅   | ⚠️        | Greedy, marginal gain   |
| EvoPS       | ⚠️  | ⚠️        | Set-level optimization  |
| PAMIL       | ❌   | ✅         | Stochastic, non-Markov  |
| Raza et al. | ❌   | ✅         | Sparse reward, glimpses |
| AAMM        | ❌   | ❌         | No policy               |

---

## 4️⃣ What this means for *your* code

### Your **DQN**

Use it only if:

* You interpret Q as **patch utility**
* You remove:

  * replay buffer (or make it episode-local)
  * lambda from TD targets
* You stop pretending it’s MIL-compatible

**DQN = greedy search baseline**, nothing more.

---

### Your **A2C**

This is where:

* PAMIL ideas belong
* Contrastive rewards belong
* Zoom budgets make sense
* Your thesis novelty actually lives

---

## 5️⃣ Blunt conclusion

* **Yes**, DC-WSI + SPLICE can inspire a *value-based* DQN baseline.
* **No**, PAMIL and Raza *cannot* — forcing them into DQN is mathematically wrong.
* **All serious dynamic patch papers converge toward policy gradients**, not Q-learning.

If you want, next I can:

* Rewrite your DQN section as a **“greedy value baseline”** (clean, defensible)
* Show how to **PAMIL-ify your A2C** step by step
* Help you phrase this split convincingly for reviewers

Just tell me which one.

# Chapter 6: PPO — The Industry Standard

## Why PPO Exists

Every algorithm we've built has a specific weakness:

- **DQN**: stable but limited to discrete actions
- **REINFORCE**: handles any action space but wildly unstable
- **A2C**: combines both approaches but inherits the instability

The root cause of the instability: **unbounded policy updates**. One gradient step can change the policy dramatically. If episode 500 produces a bad gradient, the policy jumps to a completely different behavior — one the critic hasn't calibrated for, which produces more bad gradients, which causes more jumping.

PPO's solution is elegant: **limit how much the policy can change per update**. The new policy must stay "close" to the old policy. This single constraint transforms Actor-Critic from fragile to robust.

PPO (Proximal Policy Optimization) was published by OpenAI in 2017. It has since become the default algorithm for:
- ChatGPT's RLHF (Reinforcement Learning from Human Feedback)
- OpenAI Five (Dota 2 at superhuman level)
- Most production robotics RL
- Nearly all modern RL research as a baseline

Its dominance comes from a rare combination: simple to implement, stable to train, and competitive with much more complex algorithms.

---

## Part 1: The Core Idea — The Probability Ratio

### Old Policy vs New Policy

In A2C, we collect an episode with the current policy, compute one gradient, and update. The episode data is then discarded. This is wasteful — each episode of experience is used exactly once.

PPO changes this. It collects a batch of experience, then makes **multiple gradient updates** on the same batch. But this creates a problem: after the first update, the policy has changed. The data was collected with the old policy, but we're now updating a new (slightly different) policy. How do we account for this mismatch?

The answer: the **probability ratio**.

```
r(θ) = π_new(a|s) / π_old(a|s)
```

This ratio measures how much the policy has changed for a specific (state, action) pair:

```
r(θ) = 1.0  → new policy assigns same probability as old → no change
r(θ) = 1.5  → new policy is 50% more likely to take this action
r(θ) = 0.5  → new policy is 50% less likely to take this action
r(θ) = 3.0  → new policy is 3× more likely → big change! (dangerous)
```

### Computing the Ratio

In practice, we compute the ratio using log-probabilities (numerically more stable):

```python
ratio = torch.exp(new_log_probs - old_log_probs)
```

This works because `exp(log(a) - log(b)) = exp(log(a/b)) = a/b`.

We store `old_log_probs` when collecting the experience. After updating the network, we recompute `new_log_probs` for the same (state, action) pairs using the updated network. The ratio tells us how much the policy shifted.

---

## Part 2: The Clipped Surrogate Objective

### The Unclipped Objective

The simplest way to use the ratio is the **importance-sampled policy gradient**:

```
L_unclipped = r(θ) × A
```

Where A is the advantage (same as A2C). This is mathematically correct — it adjusts the gradient to account for the fact that the data came from the old policy, not the new one.

But there's no limit on r(θ). If the new policy is very different from the old one (r = 5.0 or r = 0.1), the gradient is scaled by a large factor, causing a huge update — exactly the instability we're trying to prevent.

### The Clipped Objective

PPO's key innovation: clip the ratio to a narrow range around 1.0.

```
L_clipped = clip(r(θ), 1-ε, 1+ε) × A
```

Where ε = 0.2 (the standard clip range). The clipped ratio stays between 0.8 and 1.2:

```
r(θ) = 0.5  → clipped to 0.8    (can't decrease probability too much)
r(θ) = 0.9  → stays 0.9         (within range, no clipping)
r(θ) = 1.0  → stays 1.0         (no change)
r(θ) = 1.1  → stays 1.1         (within range, no clipping)
r(θ) = 2.0  → clipped to 1.2    (can't increase probability too much)
```

### The Minimum: Conservative Updates

PPO takes the **minimum** of the unclipped and clipped objectives:

```
L_PPO = min(r(θ) × A, clip(r(θ), 1-ε, 1+ε) × A)
```

Why the minimum? It creates a **pessimistic bound**. Let's trace through all four cases:

**Case 1: Positive advantage, ratio increasing (action is good, policy is making it more likely)**
```
A = +2.0, r(θ) = 1.5

Unclipped: 1.5 × 2.0 = 3.0
Clipped:   1.2 × 2.0 = 2.4    (r clipped from 1.5 to 1.2)
min(3.0, 2.4) = 2.4            ← clipping limits the benefit
```
The policy wants to make this action much more likely (r = 1.5), but PPO says "you can only go up to r = 1.2." This prevents over-committing to one good experience.

**Case 2: Positive advantage, ratio decreasing (action is good, but policy is making it less likely)**
```
A = +2.0, r(θ) = 0.7

Unclipped: 0.7 × 2.0 = 1.4
Clipped:   0.8 × 2.0 = 1.6    (r clipped from 0.7 to 0.8)
min(1.4, 1.6) = 1.4            ← no clipping (unclipped is already smaller)
```
The policy is moving away from a good action — the full gradient is applied to correct this. No clipping needed because the policy is already moving in a conservative direction.

**Case 3: Negative advantage, ratio increasing (action is bad, policy is making it more likely)**
```
A = -2.0, r(θ) = 1.5

Unclipped: 1.5 × (-2.0) = -3.0
Clipped:   1.2 × (-2.0) = -2.4
min(-3.0, -2.4) = -3.0         ← full gradient applied (moving toward bad action)
```
The policy is making a bad action more likely — full correction force. No clipping because we want to fix this.

**Case 4: Negative advantage, ratio decreasing (action is bad, policy is making it less likely)**
```
A = -2.0, r(θ) = 0.5

Unclipped: 0.5 × (-2.0) = -1.0
Clipped:   0.8 × (-2.0) = -1.6
min(-1.6, -1.0) = -1.6         ← clipping limits how much we decrease
```
The policy is already moving away from the bad action. PPO limits how far it goes — don't overcorrect.

### The Pattern

```
Good action, moving toward it:     CLIPPED (don't over-commit)
Good action, moving away from it:  NOT CLIPPED (full correction)
Bad action, moving toward it:      NOT CLIPPED (full correction)
Bad action, moving away from it:   CLIPPED (don't overcorrect)
```

Clipping only activates when the policy change is "too beneficial" — when the gradient would push too hard in a direction that's already been rewarded. It never prevents corrections — if the policy is moving in a bad direction, the full gradient is applied.

### In Code

```python
# Probability ratio
ratio = torch.exp(new_log_probs - mb_old_log_probs)

# Unclipped objective
surr1 = ratio * mb_advantages

# Clipped objective
clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
surr2 = clipped_ratio * mb_advantages

# Take the minimum (pessimistic bound)
actor_loss = -torch.min(surr1, surr2).mean()
```

The negative sign is because PyTorch minimizes loss, but we want to maximize the objective.

---

## Part 3: Generalized Advantage Estimation (GAE)

### The Problem with Monte Carlo Advantages

Our A2C used Monte Carlo returns: `A = Gₜ - V(sₜ)`. Zero bias, but high variance from using the full episode return.

### The Problem with TD Advantages

The alternative is TD(0): `A = rₜ + γV(sₜ₊₁) - V(sₜ)`. Low variance (one step of randomness), but biased (relies on the critic's possibly inaccurate V(sₜ₊₁)).

### GAE: The Best of Both

GAE computes a **weighted average** of all n-step advantage estimates:

```
δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)                    ← TD error at step t

A_GAE(t) = δₜ + (γλ)δₜ₊₁ + (γλ)²δₜ₊₂ + ...    ← exponentially weighted sum of TD errors
```

The parameter λ (lambda, typically 0.95) controls the tradeoff:

```
λ = 0:    A = δₜ                                 ← pure TD(0), low variance, high bias
λ = 1:    A = Gₜ - V(sₜ)                         ← pure Monte Carlo, zero bias, high variance
λ = 0.95: A = smooth blend                        ← standard choice, good balance
```

### How GAE Works: A Traced Example

Consider a 4-step sequence with γ = 0.99, λ = 0.95:

```
Rewards:    [1, 1, 1, 1]
Values:     [4.0, 3.5, 2.8, 1.5]
Next value:  0.5  (V at the state after step 3)

TD errors (δ):
  δ₃ = 1 + 0.99 × 0.5 - 1.5  = 1 + 0.495 - 1.5 = -0.005
  δ₂ = 1 + 0.99 × 1.5 - 2.8  = 1 + 1.485 - 2.8 = -0.315
  δ₁ = 1 + 0.99 × 2.8 - 3.5  = 1 + 2.772 - 3.5 =  0.272
  δ₀ = 1 + 0.99 × 3.5 - 4.0  = 1 + 3.465 - 4.0 =  0.465

GAE advantages (computed backwards):
  A₃ = δ₃                                          = -0.005
  A₂ = δ₂ + 0.99 × 0.95 × A₃  = -0.315 + 0.9405 × (-0.005) = -0.320
  A₁ = δ₁ + 0.99 × 0.95 × A₂  = 0.272 + 0.9405 × (-0.320)  = -0.029
  A₀ = δ₀ + 0.99 × 0.95 × A₁  = 0.465 + 0.9405 × (-0.029)  =  0.438
```

Step 0 has a positive advantage (0.438) — the actual sequence was better than the critic expected. Step 2 has a negative advantage (-0.320) — the critic overestimated that state's value.

### The Code

```python
def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns
```

### Line-by-Line

```python
next_non_terminal = 1.0 - dones[t]
```
- If the episode ended at step t (`dones[t] = 1.0`), then `next_non_terminal = 0.0`. This zeros out any future value — there IS no future after a terminal state.

```python
delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
```
- The TD error: actual reward + discounted next value - current value estimate.
- If done: `delta = rewards[t] - values[t]` (no future term).

```python
gae = delta + gamma * lam * next_non_terminal * gae
```
- The recursive GAE formula. Each step's advantage is its own TD error plus the discounted, λ-decayed advantage from the next step.
- `gamma * lam = 0.99 × 0.95 = 0.9405` — the decay rate. After 10 steps, the weight is `0.9405^10 ≈ 0.54`. After 50 steps: `0.9405^50 ≈ 0.05`. Distant TD errors contribute very little.

```python
returns = advantages + values
```
- The return targets for the critic. `returns = advantages + values` because `advantages = returns - values`, so `returns = advantages + values`. This gives us both the advantages (for the actor) and the returns (for the critic) in one computation.

---

## Part 4: Multiple Epochs — PPO's Efficiency Advantage

### The Waste in A2C

A2C collects an episode, computes one gradient, updates, and throws the data away. Each transition is used for exactly one gradient step. This is incredibly wasteful.

### PPO Reuses Data

PPO collects a batch of experience (2048 transitions), then makes **multiple passes** (epochs) over the same data:

```
Collect 2048 transitions with current policy (π_old)

Epoch 1:  shuffle data, split into mini-batches of 64, update on each
Epoch 2:  shuffle again, update again on the same data
Epoch 3:  shuffle again, update again
Epoch 4:  shuffle again, update again

Discard data. Collect new batch with updated policy. Repeat.
```

Four epochs × ~32 mini-batches per epoch = ~128 gradient steps from a single batch of experience. Compare to A2C's 1 gradient step per episode.

### Why Clipping Makes This Safe

In A2C, reusing data would be dangerous — the second update uses data from the old policy, and without any safeguard, the policy could drift far from what generated the data.

PPO's clipping prevents this. As the policy changes during epochs 1-4, the ratio r(θ) moves away from 1.0. Once it hits the clip boundary (0.8 or 1.2), the gradient is zeroed out for that (state, action) pair. The policy can't keep changing in the same direction — the clip acts as a natural stopping point.

This is why we track **clip fraction** — the percentage of (state, action) pairs where clipping activates:

```
Clip fraction 0.05:  5% of updates clipped → policy is barely changing (safe, maybe too conservative)
Clip fraction 0.15:  15% clipped → healthy amount of policy change
Clip fraction 0.30:  30% clipped → significant change (getting aggressive)
Clip fraction 0.50:  50% clipped → too much change (probably need fewer epochs or smaller LR)
```

In our training, clip fraction typically ranged from 0.05 to 0.25 — healthy territory.

---

## Part 5: The Rollout Buffer

### Collecting Experience

Unlike DQN's replay buffer (stores everything, samples randomly from history), PPO uses a **rollout buffer** that stores only the current batch and is discarded after use.

```python
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states = []
        # ... clear all lists
```

We store `log_probs` and `values` at collection time — these are π_old's outputs, needed for computing the ratio during updates.

### On-Policy vs Off-Policy

PPO is **on-policy**: it can only learn from data collected by the current policy. Once the policy is updated and a new batch is collected, the old data is useless. This is fundamentally different from DQN's off-policy approach, where the replay buffer stores months of old experience.

```
DQN (off-policy):  Buffer holds 50,000 transitions from many past policies
                   Random sampling breaks correlation
                   Very sample-efficient (each transition used many times over its lifetime)

PPO (on-policy):   Buffer holds 2048 transitions from the current policy ONLY
                   Used for 4 epochs, then discarded
                   Less sample-efficient, but more stable and handles continuous actions
```

---

## Part 6: The Full PPO Update

```python
def update(self, next_value):
    states, actions, rewards, dones, old_log_probs, old_values = self.buffer.get()

    # Compute GAE advantages
    advantages, returns = compute_gae(
        rewards, old_values, dones, next_value, self.gamma, self.lam
    )

    # Normalize advantages across the full rollout
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Multiple epochs over the same data
    for epoch in range(self.n_epochs):
        indices = torch.randperm(len(states))

        for start in range(0, len(states), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]

            # Get mini-batch
            mb_states = states[batch_idx]
            mb_actions = actions[batch_idx]
            mb_old_log_probs = old_log_probs[batch_idx]
            mb_advantages = advantages[batch_idx]
            mb_returns = returns[batch_idx]

            # Evaluate under CURRENT policy
            new_log_probs, new_values, entropy = self.network.evaluate_actions(
                mb_states, mb_actions
            )

            # Clipped surrogate objective
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            surr2 = clipped_ratio * mb_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = F.mse_loss(new_values, mb_returns.detach())

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = actor_loss + self.value_coeff * critic_loss + self.entropy_coeff * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

    self.buffer.clear()
```

### Why Advantage Normalization Works Here (But Not in A2C)

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

In A2C, normalization killed the signal because all advantages were near zero (one episode, similar returns). In PPO, we normalize across **2048 transitions spanning many episodes**. This batch contains transitions from episodes that scored 50 and episodes that scored 300 — plenty of variance. Normalization ensures the gradient scale doesn't depend on the absolute reward magnitude.

### The Combined Loss

```python
loss = actor_loss + self.value_coeff * critic_loss + self.entropy_coeff * entropy_loss
```

Three terms combined into one loss with one optimizer:

- `actor_loss` — the clipped surrogate objective (policy improvement)
- `value_coeff × critic_loss` — critic training (value prediction)
- `entropy_coeff × entropy_loss` — exploration maintenance

**Why does a shared optimizer work here but not in A2C?** The clipping mechanism constrains how much the actor can change per update. This prevents the actor's gradients from overwhelming the critic's gradients. In A2C, without clipping, a single large actor gradient could completely destabilize the shared network. PPO's clipping provides the stability that allows shared training.

---

## Part 7: Timesteps vs Episodes

### PPO Counts Timesteps

Previous algorithms trained for a fixed number of episodes. PPO trains for a fixed number of **timesteps** — individual environment steps, regardless of episode boundaries.

```python
while timestep < total_timesteps:
    # Collect 2048 steps (might span many short episodes or part of one long one)
    for _ in range(rollout_steps):
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.buffer.store(state, action, reward, done, log_prob, value)
        timestep += 1

        if done:
            state, _ = env.reset()    # start new episode within the rollout
        else:
            state = next_state

    # Update on the collected batch
    agent.update(next_value)
```

A single rollout of 2048 steps might contain:
- 100 short episodes (20 steps each) early in training
- 10 medium episodes (200 steps each) mid-training
- 4 long episodes (500 steps each) when the agent is good

The rollout seamlessly spans episode boundaries — it just keeps collecting until 2048 steps are reached.

### Why This Matters

Counting timesteps instead of episodes gives a fairer comparison:
- A bad agent plays 100 short episodes in 2048 steps
- A good agent plays 4 long episodes in 2048 steps
- Both use the same amount of environment interaction (2048 steps)

With episode counting, the bad agent would get 100 updates while the good agent gets only 4 — unfairly penalizing good performance.

---

## Part 8: Our Results

### Training Progression (200K timesteps)

```
Timestep   2,048 | reward:   20.9 | clip: 0.069    ← random behavior
Timestep  16,384 | reward:  107.9 | clip: 0.293    ← learning fast
Timestep  65,536 | reward:  206.2 | clip: 0.081    ← solid performance
Timestep  75,776 | reward:  293.8 | clip: 0.078    ← improving steadily
Timestep 169,984 | reward:  371.2 | clip: 0.082    ← approaching mastery
Timestep 176,128 | reward:  347.1 | clip: 0.158    ← stable high performance

Evaluation: 224.3 ± 4.9
```

### The Key Observation: Stability

Compare the evaluation standard deviations:

```
REINFORCE: 500.0 ± 0.0     (solved, but lucky — training was chaotic)
DQN:       500.0 ± 0.0     (solved perfectly)
A2C:       241.9 ± 88.7    (huge variance — 139 to 392)
PPO:       224.3 ± 4.9     (incredibly tight — 217 to 237)
```

PPO's ± 4.9 is 18× less variance than A2C's ± 88.7. The clipping mechanism prevents large policy changes, which prevents the oscillations that plagued A2C.

PPO hasn't reached 500 yet (it was still climbing — would need more timesteps), but the consistency of its performance is remarkable. Every evaluation episode scores within a narrow 20-point band. This **predictability** is why PPO is preferred for production systems. A robot that sometimes works perfectly and sometimes fails catastrophically is worse than one that consistently works well.

---

## Part 9: PPO's Hyperparameters

| Parameter | Value | What It Controls |
|-----------|-------|-----------------|
| `clip_range` (ε) | 0.2 | How much policy can change per update. Lower = more conservative. |
| `n_epochs` | 4 | How many times to reuse each batch. More = more sample-efficient but risk overfitting. |
| `rollout_steps` | 2048 | How many transitions to collect before updating. Larger = more stable advantages. |
| `batch_size` | 64 | Mini-batch size within each epoch. |
| `gamma` | 0.99 | Discount factor (same as all previous algorithms). |
| `lambda` (λ) | 0.95 | GAE parameter. 0 = TD(0), 1 = Monte Carlo. |
| `lr` | 3e-4 | Learning rate. |
| `entropy_coeff` | 0.01 | Exploration bonus. |
| `value_coeff` | 0.5 | How much to weight critic loss vs actor loss. |

The standard values above work well for a wide range of problems without tuning. This is another reason PPO is popular — it's not as sensitive to hyperparameters as DQN or A2C.

---

## Summary

PPO's three innovations over A2C:

```
Problem: Unbounded policy updates      → Solution: Clipped surrogate objective
  The ratio r(θ) = π_new/π_old is clamped to [1-ε, 1+ε]
  Prevents catastrophic policy changes

Problem: High-variance advantages      → Solution: GAE (Generalized Advantage Estimation)
  Weighted average of n-step advantages
  λ parameter controls bias-variance tradeoff

Problem: Wasteful single-use data      → Solution: Multiple epochs per batch
  Reuse each batch for 4 gradient epochs
  Clipping makes this safe (limits how much the policy can drift)
```

The PPO training loop:

```
While timesteps < total:

    ┌─── Collect Phase ───────────────────────────────┐
    │  For 2048 steps:                                 │
    │    action, log_prob, value = network(state)      │
    │    next_state, reward, done = env.step(action)   │
    │    Store (state, action, reward, done,            │
    │           log_prob, value) in buffer              │
    └──────────────────────────────────────────────────┘

    ┌─── Update Phase ────────────────────────────────┐
    │  Compute GAE advantages from buffer              │
    │  Normalize advantages                            │
    │                                                  │
    │  For epoch in range(4):                          │
    │    Shuffle data, split into mini-batches          │
    │    For each mini-batch:                          │
    │      ratio = π_new(a|s) / π_old(a|s)            │
    │      clipped = clip(ratio, 0.8, 1.2)            │
    │      actor_loss = -min(ratio×A, clipped×A)      │
    │      critic_loss = MSE(V, returns)               │
    │      entropy_loss = -entropy                     │
    │      total = actor + 0.5×critic + 0.01×entropy  │
    │      gradient step                               │
    │                                                  │
    │  Clear buffer                                    │
    └──────────────────────────────────────────────────┘
```

## What's Next

In [Chapter 7](07_comparison.md), we put all four deep RL algorithms side by side on the same environment and analyze the results. We'll look at training curves, final performance, sample efficiency, stability, and discuss when to use which algorithm in practice.
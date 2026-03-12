# Chapter 5: Actor-Critic — Combining Value-Based and Policy-Based RL

## The Problem We're Solving

REINFORCE works, but its training curves swing wildly — 500 reward one episode, 250 the next. The culprit: **high variance from Monte Carlo returns**. Each episode's return is one noisy sample from a complex distribution. One bad episode creates a gradient that can partially undo many good episodes' progress.

The running-average baseline helped but wasn't enough. The baseline is a single number — the same for every state. But some states are inherently better than others. Being near the goal deserves a high baseline; being near a penalty deserves a low one. We need a **state-dependent** baseline.

That's exactly what the critic provides.

## Part 1: The Two-Network Architecture

### Actor and Critic

```
┌─────────────────────────┐     ┌─────────────────────────┐
│         ACTOR            │     │         CRITIC           │
│                          │     │                          │
│  Input: state            │     │  Input: state            │
│  Output: action probs    │     │  Output: value V(s)      │
│                          │     │                          │
│  "What should I do?"     │     │  "How good is this       │
│                          │     │   state?"                │
└─────────────────────────┘     └─────────────────────────┘
           │                                │
           ▼                                ▼
    Sample action aₜ              Compute advantage:
    from π(a|s)                   A = Gₜ - V(sₜ)
                                  "Was this better or worse
                                   than expected?"
```

The **actor** is the same policy network from REINFORCE — it outputs action probabilities and we sample from them.

The **critic** is new — a value network that estimates V(s), the expected total future reward from state s. This replaces REINFORCE's crude running-average baseline with a learned, state-dependent function.

### The Advantage Function

The key concept that ties actor and critic together is the **advantage**:

```
A(sₜ, aₜ) = Gₜ - V(sₜ)
```

In words: "How much BETTER was the actual return compared to what the critic expected from this state?"

```
If A > 0:  The return exceeded expectations → this was a GOOD action
           → increase its probability

If A < 0:  The return fell short of expectations → this was a BAD action
           → decrease its probability

If A ≈ 0:  The return matched expectations → average action
           → barely change anything
```

Compare to REINFORCE's approaches:

```
REINFORCE (no baseline):   weight = Gₜ          (always positive → always increase)
REINFORCE (avg baseline):  weight = Gₜ - avg    (same baseline for all states)
Actor-Critic:              weight = Gₜ - V(sₜ)  (different baseline per state)
```

The state-dependent baseline is much more informative. The critic knows that being near the goal (V ≈ 9.0) is very different from being at the start (V ≈ 5.0). A return of 7.0 is great from the start (advantage +2.0) but disappointing near the goal (advantage -2.0).

### A Concrete Example

The agent is in two different states across two episodes:

```
State A (near start):    V(A) = 3.0    "I expect about 3.0 total reward from here"
State B (near goal):     V(B) = 9.0    "I expect about 9.0 total reward from here"
```

Both episodes happen to get return 6.0:

```
REINFORCE (baseline = 4.5):
  State A: weight = 6.0 - 4.5 = +1.5  → increase action probability
  State B: weight = 6.0 - 4.5 = +1.5  → increase action probability (same!)

Actor-Critic:
  State A: advantage = 6.0 - 3.0 = +3.0  → strongly increase (did much better than expected)
  State B: advantage = 6.0 - 9.0 = -3.0  → strongly DECREASE (did much worse than expected)
```

REINFORCE treats both identically. Actor-Critic recognizes that 6.0 is great from state A but terrible from state B. The critic provides context that the fixed baseline can't.

---

## Part 2: Why Separate Networks?

### The Competing Gradients Problem

Our first A2C attempt used a **shared** network — one backbone with two output heads (actor and critic). This seems efficient: the shared layers learn to understand the state, and both heads benefit.

In practice, it caused a catastrophic failure — the agent got stuck at reward 10, worse than random. The problem: actor and critic gradients **compete**. The actor wants features that help choose good actions. The critic wants features that help predict values. These aren't the same thing, and when the gradients point in different directions, the shared layers get pulled apart and learn nothing useful.

### The Solution: Separate Everything

```python
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
```

Two completely independent networks. Two separate optimizers. No shared weights. Each network can learn freely without interfering with the other.

### Why Tanh Instead of ReLU?

Notice we use `nn.Tanh()` instead of `nn.ReLU()`. Tanh outputs values between -1 and 1, providing smoother gradients than ReLU (which has zero gradient for negative inputs). Policy gradient methods are sensitive to gradient quality — smoother gradients mean smoother learning. This is a common choice in policy gradient implementations.

### Separate Learning Rates

```python
self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-3)
self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
```

The actor learns faster (3e-3) than the critic (1e-3). This was a critical tuning choice we discovered through failure:

- **First attempt**: critic at 5e-3, actor at 1e-3. The critic converged too quickly to predicting the bad initial returns (~22), and the advantage signal went to zero. Actor never improved.
- **Fix**: actor at 3e-3, critic at 1e-3. The actor moves first, exploring and finding better behavior. The critic catches up, providing increasingly accurate baselines.

The intuition: the actor needs to be "ahead" of the critic. If the critic perfectly predicts the current (bad) returns, every advantage is zero and the actor can't learn. A slightly inaccurate critic produces non-zero advantages that drive actor improvement.

---

## Part 3: The Update — Step by Step

### Collecting an Episode

Like REINFORCE, we collect a full episode before updating:

```python
def select_action(self, state):
    state_t = torch.FloatTensor(state).unsqueeze(0)
    probs = self.actor(state_t)
    dist = Categorical(probs)
    action = dist.sample()

    self.log_probs.append(dist.log_prob(action))
    self.states.append(state)
    self.entropies.append(dist.entropy())

    return action.item()
```

We store three things per step:
- `log_probs` — needed for the actor gradient
- `states` — needed for the critic's value estimates
- `entropies` — needed for the entropy bonus (explained below)

### Computing Returns

Same as REINFORCE — Monte Carlo returns computed backwards:

```python
def compute_returns(self):
    returns = []
    G = 0
    for r in reversed(self.rewards):
        G = r + self.gamma * G
        returns.insert(0, G)
    return torch.FloatTensor(returns)
```

### The Update

```python
def update(self):
    returns = self.compute_returns()

    states_t = torch.FloatTensor(np.array(self.states))
    log_probs = torch.stack(self.log_probs)
    entropies = torch.stack(self.entropies)

    # Critic's value estimates
    values = self.critic(states_t)

    # Advantage = actual return - critic's prediction
    advantages = returns - values.detach()

    # ---- Update Actor ----
    actor_loss = -(log_probs * advantages).mean()
    entropy_bonus = entropies.mean()
    actor_total = actor_loss - self.entropy_coeff * entropy_bonus

    self.actor_optimizer.zero_grad()
    actor_total.backward()
    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
    self.actor_optimizer.step()

    # ---- Update Critic ----
    critic_loss = F.mse_loss(values, returns.detach())

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
    self.critic_optimizer.step()
```

### Line-by-Line: The Critical Details

```python
values = self.critic(states_t)
```
- Forward pass through the critic for every state in the episode.
- Shape: `(T,)` — one value per timestep.
- These are the critic's predictions of "how much total reward will follow from each state."

```python
advantages = returns - values.detach()
```
- **The advantage**: how much better (or worse) the actual return was compared to the critic's prediction.
- **`values.detach()`** is critical. Without `.detach()`, the actor's gradient would flow back through the critic (via the advantage computation), messing up the critic's weights. `.detach()` creates a copy of the values tensor that's disconnected from the computation graph — the actor's gradient stops here.

```python
actor_loss = -(log_probs * advantages).mean()
```
- Same policy gradient as REINFORCE, but weighted by advantage instead of raw returns.
- `advantages` can be negative (return was below the critic's prediction), which means "decrease this action's probability."
- `.mean()` averages over all timesteps in the episode.

```python
entropy_bonus = entropies.mean()
actor_total = actor_loss - self.entropy_coeff * entropy_bonus
```
- **Entropy bonus**: encourages the policy to maintain some randomness.
- Entropy measures how "spread out" the distribution is. A policy that always picks the same action has zero entropy. A 50/50 policy has maximum entropy.
- We subtract `entropy_coeff × entropy` from the loss. Since we minimize loss and entropy is positive, this INCREASES entropy — pushing the policy away from always picking the same action.
- Without this, the policy can **collapse**: it commits to one action early, stops exploring, and never discovers better strategies. This was the exact failure mode in our first A2C attempts.
- `entropy_coeff = 0.01` — a small encouragement. Enough to prevent collapse, not enough to override the policy gradient signal.

```python
critic_loss = F.mse_loss(values, returns.detach())
```
- Train the critic to predict actual returns.
- `returns.detach()` — the critic's gradient shouldn't affect how returns are computed (they're observed data, not predictions).
- This is supervised learning: the input is a state, the target is the actual return that followed. The critic learns to approximate the value function V(s) ≈ E[Gₜ | sₜ = s].

### Why Two Separate Backward Passes?

```python
# Actor update
self.actor_optimizer.zero_grad()
actor_total.backward()
self.actor_optimizer.step()

# Critic update
self.critic_optimizer.zero_grad()
critic_loss.backward()
self.critic_optimizer.step()
```

We update the actor and critic completely independently. The actor's gradient doesn't touch the critic's weights, and vice versa. This is the clean separation that makes training stable.

If we combined them into one loss (`total_loss = actor_loss + critic_loss`) and did one backward pass with one optimizer, the gradients would interact through shared optimizer state (Adam's momentum terms). Separate optimizers keep them fully independent.

---

## Part 4: The Advantage Normalization Trap

### What Went Wrong

Our first working A2C had zero actor loss — the actor wasn't updating at all. The culprit was advantage normalization:

```python
# THIS KILLED THE GRADIENT:
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

When all episodes score similarly (~22 reward) and the critic quickly learns to predict ~22, all advantages are near zero. Normalizing near-zero values produces even smaller values (dividing by a tiny standard deviation doesn't help when the mean is already subtracted). The actor loss rounds to 0.0000 and learning stops completely.

### The Fix

Remove advantage normalization entirely:

```python
# Raw advantages — let the critic's errors drive learning
advantages = returns - values.detach()
```

This was counterintuitive — normalization usually helps in deep learning. But here it was removing the only learning signal. The raw advantages (even if they're all similar) have enough variation step-to-step (early steps have higher returns than late steps) to provide meaningful gradients.

### The Lesson

Don't blindly apply techniques from supervised learning to RL. The dynamics are different:
- In supervised learning, you have diverse, well-distributed targets
- In RL, especially early in training, all episodes look similar and targets are narrow
- Normalization that's helpful for diverse data can destroy a narrow signal

---

## Part 5: The Bias-Variance Tradeoff

### Where Each Algorithm Sits

The advantage can be computed different ways, each trading off bias and variance:

**Monte Carlo (what we use — REINFORCE and our A2C):**
```
A = Gₜ - V(sₜ)

Gₜ is the TRUE discounted return (observed from the episode)
V(sₜ) is the critic's estimate

Variance: HIGH — Gₜ is one sample from a noisy distribution
Bias: LOW — Gₜ is the real return, not an estimate
```

**TD(0) — One-step temporal difference:**
```
A = rₜ + γ × V(sₜ₊₁) - V(sₜ)

Uses one real reward, then bootstraps from the critic for everything else

Variance: LOW — only one step of randomness
Bias: HIGH — V(sₜ₊₁) might be inaccurate
```

**n-step returns — A middle ground:**
```
A = rₜ + γrₜ₊₁ + ... + γⁿ⁻¹rₜ₊ₙ₋₁ + γⁿV(sₜ₊ₙ) - V(sₜ)

Uses n real rewards, then bootstraps

Variance: MEDIUM (n steps of randomness)
Bias: MEDIUM (bootstrap after n steps)
```

This spectrum from Monte Carlo to TD(0) is fundamental to RL. Every algorithm chooses a point on it. PPO (next chapter) uses **GAE** (Generalized Advantage Estimation), which smoothly blends across this entire spectrum.

### Visualizing the Spectrum

```
Full Monte Carlo                    ←── Our A2C is here
  ↕ Bias: none,  Variance: high
n-step (n=20)
  ↕
n-step (n=5)
  ↕
n-step (n=1) = TD(0)
  ↕ Bias: high,  Variance: low
```

Our A2C uses Monte Carlo advantages (full episode returns). This gives zero bias (we use real observed returns) but high variance (one episode's return is a noisy estimate). The next chapter's PPO will use GAE to find a better point on this spectrum.

---

## Part 6: Our Results and What They Show

### Training Progress

```
Episode   100 | avg reward:    39.2
Episode   300 | avg reward:    54.8
Episode   500 | avg reward:    74.2
Episode   700 | avg reward:   149.7
Episode   800 | avg reward:   363.3
Episode   900 | avg reward:   363.2
Episode  1000 | avg reward:   260.6

Evaluation: 241.9 ± 88.7
```

The good news: A2C clearly learned — from 39 to 363, with evaluation averaging 241. Compare to random (22) and our broken first attempts (10).

The bad news: **the instability problem.** The agent peaked at 363 around episode 800, then dropped to 260 by episode 1000. The evaluation shows ± 88.7 — some episodes score 392, others 139.

### What's Happening

The actor makes a large policy update that overshoots. The policy changes too much in one step — maybe it becomes too confident (always picking one action) or shifts to a bad region of the policy space. The critic, which was calibrated for the old policy, now provides inaccurate advantages for the new policy. This leads to more bad updates, and the cycle continues.

Sometimes the agent recovers (the critic recalibrates and the actor finds its way back). Sometimes it doesn't, and performance stays low for a while.

### The Missing Piece

What we need is a way to **limit how much the policy can change in a single update**. If the policy can only take small steps, it can't accidentally destroy itself. One bad episode can nudge the policy slightly in a bad direction, but it can't catastrophically shift it to a completely different behavior.

This is exactly what PPO provides.

---

## Part 7: Comparing Everything So Far

| Aspect | DQN | REINFORCE | A2C |
|--------|-----|-----------|-----|
| Approach | Value-based | Policy-based | Both combined |
| Networks | Q-network + target | Policy only | Actor + Critic |
| Updates per episode | Many (every step) | 1 (end of episode) | 1 (end of episode) |
| Variance | Low (replay buffer) | Very high | High (but less than REINFORCE) |
| Bias | Some (bootstrapping) | None | None (Monte Carlo) |
| Stability | High (once tuned) | Low (oscillates) | Medium (still oscillates) |
| Action space | Discrete only | Any | Any |
| Sample efficiency | High (replay) | Low (use once) | Low (use once) |

The pattern: DQN is stable but limited. REINFORCE is flexible but unstable. A2C is flexible and better than REINFORCE, but still unstable. We keep gaining capability while fighting instability.

PPO resolves this tension once and for all.

---

## Summary

The Actor-Critic architecture:

```
For each episode:
    While not done:
        probs = actor(state)                      ← policy distribution
        action = sample(probs)                     ← stochastic action
        log_prob = log(probs[action])              ← record for gradient
        entropy = -Σ probs × log(probs)            ← measure randomness
        next_state, reward = env.step(action)

    Compute returns: Gₜ = rₜ + γGₜ₊₁              ← backwards pass
    Compute values: V(sₜ) = critic(sₜ)            ← critic forward pass
    Compute advantages: Aₜ = Gₜ - V(sₜ)           ← how much better than expected?

    Actor loss = -mean(log_probs × advantages) - entropy_coeff × mean(entropy)
    Critic loss = MSE(values, returns)

    Update actor (separate optimizer)
    Update critic (separate optimizer)
```

Key ideas:

| Concept | What It Is | Why It Matters |
|---------|-----------|----------------|
| Advantage | Gₜ - V(sₜ) | State-dependent baseline → lower variance than REINFORCE |
| Separate networks | Independent actor and critic | Prevents competing gradients |
| Separate optimizers | Different learning rates | Actor can lead, critic follows |
| Entropy bonus | Reward for maintaining randomness | Prevents policy collapse |
| .detach() | Stops gradient flow between networks | Keeps actor and critic independent |

## What's Next

In [Chapter 6](06_ppo.md), we solve the instability problem with **PPO (Proximal Policy Optimization)**. The core idea: clip the policy update so the new policy can never be too different from the old one. This single constraint transforms Actor-Critic from a fragile research algorithm into the robust, reliable industry standard used in robotics, game AI, and ChatGPT's RLHF training.
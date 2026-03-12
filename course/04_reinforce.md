# Chapter 4: REINFORCE — Policy Gradients from Scratch

## A Fundamentally Different Philosophy

Everything we've built so far follows the same pattern:

```
Value-based approach:
  1. Estimate Q(s, a) — "how good is each action?"
  2. Act greedily — pick the action with the highest Q-value
```

REINFORCE throws this out entirely:

```
Policy-based approach:
  1. Have a neural network output action probabilities directly
  2. Sample actions from these probabilities
  3. If a sampled action led to high reward, make it more probable
  4. If it led to low reward, make it less probable
```

No Q-values. No value estimation at all. The network IS the policy — it directly maps states to action probabilities.

### Why Bother?

DQN solved CartPole perfectly. Why do we need a new approach?

**Problem 1: Discrete actions only.** DQN outputs a Q-value for each action: `[Q(left), Q(right)]`. For a robot arm with continuous joint angles (turn 15.7° vs 15.8° vs 15.9°...), you'd need infinite outputs. Policy networks can output continuous distributions naturally — for example, a Gaussian with mean and standard deviation.

**Problem 2: Deterministic policies.** DQN's policy is `argmax(Q)` — always the same action in the same state. Sometimes you want **stochastic** policies — a probability distribution over actions. This is useful for exploration (naturally trying different things) and for games against opponents (a predictable policy can be exploited).

**Problem 3: The policy is what we actually want.** Q-values are a means to an end — we estimate them only to derive a policy. Policy gradient methods skip the middleman and optimize what we actually care about.

---

## Part 1: The Policy Network

### From Q-Values to Probabilities

The DQN network outputs Q-values (arbitrary real numbers):
```
State → Network → [2.1, 3.8]     ← Q-values
                   Pick argmax → action 1 (right)
```

The policy network outputs **probabilities** (positive, sum to 1):
```
State → Network → [0.25, 0.75]   ← probabilities
                   Sample → action 1 (right, 75% of the time)
```

The difference: DQN always picks the same action (argmax). The policy network **samples** — it picks action 1 most of the time but occasionally picks action 0. This built-in randomness IS the exploration strategy. No need for epsilon-greedy.

### Softmax: From Raw Scores to Probabilities

The network's final layer outputs raw scores (logits). We convert these to probabilities using **softmax**:

```
logits = [1.0, 2.0]

softmax([1.0, 2.0]) = [e^1.0 / (e^1.0 + e^2.0),  e^2.0 / (e^1.0 + e^2.0)]
                     = [2.72 / (2.72 + 7.39),       7.39 / (2.72 + 7.39)]
                     = [0.27, 0.73]
```

Properties of softmax:
- All outputs are positive
- All outputs sum to 1 (a valid probability distribution)
- Larger logits get larger probabilities
- The ratio between probabilities is exponential: a logit difference of 1 means ~2.7× probability ratio

### The Code

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def get_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
```

### Line-by-Line: get_action

```python
probs = self.forward(state_t)
```
- Forward pass: state → logits → softmax → probabilities.
- For CartPole: output might be `[0.35, 0.65]` — 35% chance of left, 65% chance of right.

```python
dist = Categorical(probs)
```
- Create a **categorical distribution** from the probabilities. This is PyTorch's way of representing a discrete probability distribution that we can sample from and compute log-probabilities of.

```python
action = dist.sample()
```
- Sample one action from the distribution. With probs `[0.35, 0.65]`, this returns 0 about 35% of the time and 1 about 65% of the time. Unlike DQN's argmax (which always returns 1), sampling provides natural exploration.

```python
log_prob = dist.log_prob(action)
```
- Compute `log π(a|s)` — the log-probability of the sampled action under the current policy. This is the crucial quantity needed for the policy gradient update.
- If `probs = [0.35, 0.65]` and `action = 1`: `log_prob = log(0.65) ≈ -0.43`.
- **Why log-probability?** Two reasons: (1) logs turn products into sums, simplifying the math, and (2) the gradient of log-probability has a convenient form that enables the policy gradient theorem.

---

## Part 2: The Policy Gradient Theorem

### The Objective

We want to maximize the **expected total return** under our policy:

```
J(θ) = E_π [ G₀ ]  = "average total reward when following policy π"
```

Where θ are the network's parameters and G₀ is the return from the start.

To maximize J(θ), we need its gradient with respect to θ — the direction to push the parameters to increase expected reward.

### The Derivation (Intuitive Version)

The policy gradient theorem states:

```
∇J(θ) = E_π [ Σₜ  ∇log π(aₜ|sₜ; θ) × Gₜ ]
```

In words: the gradient of expected return equals the expected value of the sum over all timesteps of the gradient of the log-probability of the action taken, weighted by the return from that timestep onward.

Let's unpack each piece:

**`∇log π(aₜ|sₜ; θ)`** — "the direction to push θ to make action aₜ more likely in state sₜ."

- This is the gradient of the log-probability of the action we took.
- It points in the direction that would increase π(aₜ|sₜ).
- If we follow this gradient, action aₜ becomes more probable in state sₜ.

**`Gₜ`** — "how good was the outcome from this point on?"

- The discounted return from timestep t: Gₜ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ...
- This WEIGHTS the gradient. Large Gₜ → push hard (make this action much more likely). Small Gₜ → push weakly.

**The product `∇log π × Gₜ`** — "increase the probability of action aₜ proportionally to how good the outcome was."

- Action led to high return → strongly increase its probability
- Action led to low return → weakly increase (or don't change) its probability
- The best actions get reinforced the most — hence "REINFORCE"

### Why It Works: A Thought Experiment

Imagine two episodes in the same starting state:

```
Episode 1: action = LEFT,  return = 500    (great outcome)
Episode 2: action = RIGHT, return = 20     (bad outcome)
```

The policy gradient update:
- Episode 1: push parameters to make LEFT more likely (weighted by 500)
- Episode 2: push parameters to make RIGHT more likely (weighted by 20)

The LEFT push is 25× stronger than the RIGHT push. Over many episodes, the policy converges to preferring LEFT in this state — because LEFT consistently leads to higher returns.

### The Log-Probability Trick

Why `log π` instead of just `π`? There's a mathematical trick at work. We want to differentiate an expectation:

```
∇ E_π[G] = ∇ Σ_τ P(τ) × G(τ)
```

Where τ is a trajectory and P(τ) is its probability under the policy. The trick uses the identity:

```
∇ P(τ) = P(τ) × ∇ log P(τ)
```

This converts the gradient of a probability (hard to compute) into an expectation of a log-probability gradient (easy to compute with sampling). We can estimate the expectation by sampling trajectories — which is exactly what the agent does by playing episodes.

This is sometimes called the **likelihood ratio trick** or **score function estimator**.

---

## Part 3: Computing Returns

### Discounted Returns

After an episode, we compute the return at each timestep:

```
Rewards: [1, 1, 1, 1, 1]  (5-step episode in CartPole)
γ = 0.99

G₄ = 1                                         = 1.000
G₃ = 1 + 0.99 × 1                              = 1.990
G₂ = 1 + 0.99 × 1 + 0.99² × 1                 = 2.970
G₁ = 1 + 0.99 × 1 + 0.99² × 1 + 0.99³ × 1    = 3.940
G₀ = 1 + 0.99 × 1 + 0.99² × 1 + 0.99³ × 1 + 0.99⁴ × 1 = 4.900
```

We compute this backwards efficiently:

```python
def compute_returns(self):
    returns = []
    G = 0
    for r in reversed(self.rewards):
        G = r + self.gamma * G
        returns.insert(0, G)
    return returns
```

### Line-by-Line

```python
for r in reversed(self.rewards):
    G = r + self.gamma * G
    returns.insert(0, G)
```
- Start from the last reward and work backwards.
- `G = r + γ × G` — the return at timestep t is the reward at t plus discounted return from t+1.
- `returns.insert(0, G)` — prepend to the list (since we're going backwards).

Trace through the 5-step example:
```
Start: G = 0
Step 4 (r=1): G = 1 + 0.99 × 0     = 1.000   returns = [1.000]
Step 3 (r=1): G = 1 + 0.99 × 1.000  = 1.990   returns = [1.990, 1.000]
Step 2 (r=1): G = 1 + 0.99 × 1.990  = 2.970   returns = [2.970, 1.990, 1.000]
Step 1 (r=1): G = 1 + 0.99 × 2.970  = 3.940   returns = [3.940, 2.970, 1.990, 1.000]
Step 0 (r=1): G = 1 + 0.99 × 3.940  = 4.900   returns = [4.900, 3.940, 2.970, 1.990, 1.000]
```

---

## Part 4: The REINFORCE Update

### The Loss Function

```python
def update(self):
    returns = self.compute_returns()
    returns_t = torch.FloatTensor(returns)

    # Apply baseline
    if self.use_baseline:
        returns_t = returns_t - self.baseline

    # Normalize returns
    if len(returns_t) > 1 and returns_t.std() > 0:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

    # Policy gradient loss
    policy_loss = []
    for log_prob, G in zip(self.log_probs, returns_t):
        policy_loss.append(-log_prob * G)

    loss = torch.stack(policy_loss).sum()

    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
    self.optimizer.step()
```

### Line-by-Line

```python
policy_loss.append(-log_prob * G)
```
This single line is the entire REINFORCE algorithm. Let's understand the negative sign:

- `log_prob` is negative (log of a number between 0 and 1 is always negative)
- `G` is positive (total reward)
- `-log_prob * G` is positive (negative × positive, then negated)
- PyTorch **minimizes** loss, but we want to **maximize** expected return
- Minimizing `-log_prob × G` is equivalent to maximizing `log_prob × G`
- Maximizing `log_prob × G` means: increase log-probability of actions with high returns

If G is large: the loss is very negative → strong gradient → big update to make this action more likely.
If G is small: the loss is barely negative → weak gradient → small update.

```python
loss = torch.stack(policy_loss).sum()
```
- Sum the loss across all timesteps. The policy gradient is a sum over the trajectory.

### Why REINFORCE Is Monte Carlo

REINFORCE is a **Monte Carlo** method — it uses **complete episode returns** to compute the gradient. The agent must finish an entire episode before it can update.

Compare to DQN, which updates after every single step using bootstrapped estimates. REINFORCE has:
- **Zero bias**: the returns Gₜ are the actual observed returns, not estimates
- **High variance**: one episode might score 500 (everything went right) and the next 50 (one unlucky early action), even with the same policy

This high variance is REINFORCE's fundamental weakness.

---

## Part 5: The Baseline — Reducing Variance

### The Problem

Consider a CartPole agent that always scores between 100 and 200. Every action gets a positive return, so every action's probability gets increased. But some actions are better than others! We need to distinguish "above average" from "below average," not just "positive" from "negative."

### The Solution

Subtract a **baseline** b from the returns:

```
∇J(θ) = E[ Σₜ ∇log π(aₜ|sₜ) × (Gₜ - b) ]
```

This is mathematically valid — subtracting a constant baseline doesn't change the gradient direction (provably). But it dramatically reduces variance:

- `Gₜ - b > 0` → return was ABOVE average → increase action probability
- `Gₜ - b < 0` → return was BELOW average → decrease action probability
- `Gₜ - b ≈ 0` → return was average → barely change anything

### Choosing the Baseline

The simplest baseline: the running average of episode returns.

```python
self.baseline = 0.0
self.baseline_count = 0

# After each episode:
episode_return = returns[0]
self.baseline_count += 1
self.baseline += (episode_return - self.baseline) / self.baseline_count
```

This is the same incremental mean update from Chapter 1 (bandits). The baseline tracks "what's typical" so the agent can focus on "what's better or worse than typical."

### A Concrete Example

Without baseline (all returns are positive):
```
Episode 1: return = 150 → increase ALL action probabilities (weight: 150)
Episode 2: return = 80  → increase ALL action probabilities (weight: 80)
Episode 3: return = 200 → increase ALL action probabilities (weight: 200)
```

Every episode pushes in the same direction (increase everything). The signal is dominated by the magnitude of returns, not the quality of actions.

With baseline (baseline ≈ 143):
```
Episode 1: return - baseline = 150 - 143 = +7   → slightly increase probabilities
Episode 2: return - baseline = 80 - 143 = -63   → DECREASE probabilities (bad episode!)
Episode 3: return - baseline = 200 - 143 = +57  → increase probabilities
```

Now the signal distinguishes good episodes from bad episodes. Episode 2's actions get their probabilities *decreased* because the return was below average. Much more informative.

### Return Normalization

Beyond the baseline, we also normalize returns to have zero mean and unit standard deviation:

```python
if len(returns_t) > 1 and returns_t.std() > 0:
    returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
```

This ensures the gradient magnitude doesn't depend on the absolute scale of rewards. Whether returns are in the range [10, 20] or [1000, 2000], the normalized gradient has similar magnitude. This stabilizes training, especially when combined with a fixed learning rate.

---

## Part 6: Our Results

### REINFORCE Without Baseline

```
Episode  400 | avg reward: 458.7
Episode  500 | avg reward: 252.7    ← dropped 200 points!
Episode  700 | avg reward: 491.6
Episode  800 | avg reward: 439.7    ← dropped again

Evaluation: 500.0 ± 0.0
```

The agent learned (evaluation is perfect) but the training curve is wild — swinging between 250 and 500. One bad episode can partially undo hundreds of good ones.

### REINFORCE With Baseline

```
Episode  400 | avg reward: 443.6
Episode  700 | avg reward: 249.3    ← still unstable!
Episode  900 | avg reward: 185.4
Episode 1000 | avg reward: 479.4

Evaluation: 500.0 ± 0.0
```

The baseline helped (slightly faster convergence on average) but the instability remains. Both versions oscillate between good and bad performance throughout training.

### The Core Problem

The training curves swing wildly because **one episode's gradient can undo many episodes' worth of learning**. If episode 500 scores 50 while the previous 50 episodes averaged 480, the loss from episode 500 creates a massive gradient that shifts the policy away from the good behavior it had developed.

DQN didn't have this problem because:
1. The replay buffer smooths out individual episodes
2. The target network provides stable learning targets
3. Multiple training steps per episode (higher data efficiency)

REINFORCE has none of these stabilizers. Each episode gets one gradient update, and that update can be arbitrarily large.

This instability is exactly what the next chapter addresses.

---

## Part 7: REINFORCE vs DQN — A Side-by-Side Comparison

| Aspect | DQN | REINFORCE |
|--------|-----|-----------|
| Learns | Q-values (then derives policy) | Policy directly |
| Action selection | argmax(Q) + ε-greedy | Sample from π(a\|s) |
| Updates per step | 1 (on a batch of 64) | 1 (on the full episode) |
| When it updates | Every step | End of episode only |
| Data reuse | Yes (replay buffer) | No (each episode used once) |
| Exploration | ε-greedy (manual) | Built-in (stochastic policy) |
| Continuous actions | No | Yes (future chapters) |
| Variance | Low (replay + target net) | High (full episode returns) |
| Bias | Some (bootstrapping) | None (true returns) |

The fundamental tradeoff: DQN has low variance but some bias (from bootstrapping). REINFORCE has zero bias but high variance (from using full episode returns). The next two chapters combine both approaches to get low variance AND low bias.

---

## Summary

The REINFORCE algorithm:

```
For each episode:
    states, actions, log_probs, rewards = [], [], [], []

    While not done:
        probs = policy_network(state)           ← forward pass
        action = sample from probs              ← stochastic action selection
        log_prob = log(prob of action)           ← record for gradient
        next_state, reward = env.step(action)    ← take action
        Store: state, action, log_prob, reward

    Compute discounted returns: Gₜ = rₜ + γGₜ₊₁     ← backwards pass
    Subtract baseline: Gₜ ← Gₜ - b                   ← variance reduction
    Normalize: Gₜ ← (Gₜ - mean) / std                ← stability

    loss = -Σₜ log π(aₜ|sₜ) × Gₜ                     ← policy gradient
    loss.backward()                                     ← compute gradients
    optimizer.step()                                    ← update parameters
```

Key ideas:
1. **The policy gradient theorem** — the mathematical basis for directly optimizing policies
2. **The log-probability trick** — makes the gradient computable from sampled trajectories
3. **Monte Carlo returns** — unbiased but high variance
4. **Baselines** — reduce variance without introducing bias
5. **The variance problem** — REINFORCE's fundamental limitation, motivating Actor-Critic

## What's Next

In [Chapter 5](05_actor_critic.md), we add a **critic** — a value network that provides step-by-step feedback instead of waiting for the full episode return. The critic replaces the crude running-average baseline with a learned, state-dependent baseline. This is the **Actor-Critic** architecture, which combines value-based and policy-based methods into something more powerful than either alone.
# Chapter 1: Multi-Armed Bandits — Exploration vs Exploitation

## The Problem

You walk into a casino with 10 slot machines. Each machine has a different (unknown) probability of paying out. You have 2000 pulls total. How do you maximize your winnings?

This is the **multi-armed bandit** problem — the simplest possible RL scenario. There are no states, no transitions, no episodes. Just one repeated decision: which arm do I pull?

Despite its simplicity, this problem captures the fundamental tension in all of RL: **exploration vs exploitation**. Every algorithm we build in this course addresses this tension differently.

## Why Start Here?

Bandits strip away everything except the core dilemma:

- No states → no need to think about transitions or planning
- No sequences → no need for temporal reasoning
- One action per step → no compounding effects

This lets us focus purely on the question: "How do I learn which option is best when I don't know the payoffs?"

## The Environment

```python
class BanditEnvironment:
    def __init__(self, k: int = 10, seed: int = 42):
        self.k = k
        self.rng = np.random.RandomState(seed)
        self.probs = self.rng.uniform(0.1, 0.9, size=k)
        self.best_arm = int(np.argmax(self.probs))
        self.best_prob = self.probs[self.best_arm]

    def pull(self, arm: int) -> float:
        return 1.0 if self.rng.random() < self.probs[arm] else 0.0
```

### Line-by-Line

```python
self.probs = self.rng.uniform(0.1, 0.9, size=k)
```
- Each arm gets a random payout probability between 0.1 and 0.9.
- These probabilities are **hidden from the agent**. The agent only observes the binary outcome (1 or 0) of each pull.
- With our seed, the 10 arms get probabilities like: `[0.40, 0.86, 0.69, 0.58, 0.22, 0.22, 0.15, 0.79, 0.58, 0.67]`
- Arm 1 (probability 0.86) is the best, but the agent doesn't know this.

```python
def pull(self, arm: int) -> float:
    return 1.0 if self.rng.random() < self.probs[arm] else 0.0
```
- Pull arm → generate a random number between 0 and 1 → if it's less than the arm's probability, pay out 1.0, otherwise 0.0.
- This is a **Bernoulli reward**: you either win (1) or lose (0).
- Arm 1 pays out 86% of the time, arm 6 pays out only 15% of the time. But you'd need many pulls to figure this out.

### The Challenge

If you knew the probabilities, you'd always pull arm 1. But you don't. After 5 pulls of arm 3 with results [1, 0, 1, 1, 0], you estimate its probability as 3/5 = 0.60. But that estimate is noisy — the true probability might be 0.40 or 0.80. Do you stick with arm 3 or try arm 7 (which you've only pulled once)?

---

## Strategy 1: Random (Baseline)

```python
class RandomAgent:
    def __init__(self, k: int):
        self.k = k

    def select_arm(self) -> int:
        return self.rng.randint(0, self.k)

    def update(self, arm: int, reward: float):
        pass  # Random agent doesn't learn
```

Pick uniformly at random every time. Never learns, never exploits. This is our baseline — any smart strategy should beat this.

**Expected performance**: the average of all arm probabilities. With our arms: `(0.40 + 0.86 + 0.69 + ... + 0.67) / 10 ≈ 0.52`. Our results confirmed this: 0.532 average reward.

---

## Strategy 2: Epsilon-Greedy

### The Idea

Most of the time, pick the arm you think is best (**exploit**). But with a small probability ε, pick a random arm instead (**explore**).

```
With probability (1 - ε):  pick the arm with highest estimated value
With probability ε:        pick a random arm
```

This is the simplest possible balance of exploration and exploitation. ε controls the ratio: ε = 0 means pure exploitation (never explore), ε = 1 means pure exploration (never exploit).

### Estimating Arm Values

How do we know which arm is "best"? We track the **running average** of rewards for each arm:

```
After pulling arm 3 five times with rewards [1, 0, 1, 1, 0]:
  Q(arm 3) = (1 + 0 + 1 + 1 + 0) / 5 = 0.60
```

This is our **estimate** of arm 3's true probability. With more pulls, the estimate converges to the true value (by the law of large numbers).

### Incremental Mean Update

Computing the average from scratch every time is wasteful. Instead, we use the **incremental update** formula:

```
new_average = old_average + (1/n) × (reward - old_average)
```

Where n is the number of times this arm has been pulled. Let's trace through:

```
Pull 1: reward = 1    Q = 0 + (1/1)(1 - 0) = 1.0
Pull 2: reward = 0    Q = 1.0 + (1/2)(0 - 1.0) = 0.5
Pull 3: reward = 1    Q = 0.5 + (1/3)(1 - 0.5) = 0.667
Pull 4: reward = 1    Q = 0.667 + (1/4)(1 - 0.667) = 0.75
Pull 5: reward = 0    Q = 0.75 + (1/5)(0 - 0.75) = 0.60
```

Same result (0.60) as computing the full average, but we only need the previous estimate and the count.

### The Code

```python
class EpsilonGreedyAgent:
    def __init__(self, k: int, epsilon: float = 0.1, decay: float = 0.999):
        self.k = k
        self.epsilon = epsilon
        self.decay = decay
        self.q_values = np.zeros(k)
        self.arm_counts = np.zeros(k)

    def select_arm(self) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.k)
        else:
            max_q = np.max(self.q_values)
            best_arms = np.where(self.q_values == max_q)[0]
            return self.rng.choice(best_arms)

    def update(self, arm: int, reward: float):
        self.arm_counts[arm] += 1
        n = self.arm_counts[arm]
        self.q_values[arm] += (1.0 / n) * (reward - self.q_values[arm])
        self.epsilon *= self.decay
```

### Line-by-Line

```python
self.q_values = np.zeros(k)
self.arm_counts = np.zeros(k)
```
- `q_values[i]` = our current estimate of arm i's payout probability. Starts at 0 for all arms.
- `arm_counts[i]` = how many times we've pulled arm i. Needed for the incremental mean.

```python
if self.rng.random() < self.epsilon:
    return self.rng.randint(0, self.k)
```
- With probability ε, explore: pick a random arm.

```python
else:
    max_q = np.max(self.q_values)
    best_arms = np.where(self.q_values == max_q)[0]
    return self.rng.choice(best_arms)
```
- With probability (1 - ε), exploit: pick the arm with the highest estimated value.
- The `np.where` + `choice` handles ties: if multiple arms have the same estimated value (common early on when many arms are at 0), we pick randomly among them rather than always picking the lowest index.

```python
self.q_values[arm] += (1.0 / n) * (reward - self.q_values[arm])
```
- The incremental mean update. `(reward - self.q_values[arm])` is the **error** — how far the reward was from our estimate. We adjust by `1/n` of this error.

```python
self.epsilon *= self.decay
```
- **Decay epsilon over time.** Early on, we want lots of exploration (ε is large). Later, once we've found the best arm, we want mostly exploitation (ε is small).
- With decay = 0.998 and starting ε = 0.15: after 500 pulls, ε ≈ 0.055. After 1000 pulls, ε ≈ 0.020.

### The Fundamental Limitation

Epsilon-greedy explores **uniformly** — when it explores, every arm has equal chance. This wastes pulls on arms we already know are bad. Arm 6 (probability 0.15) gets just as many exploration pulls as arm 1 (probability 0.86). We need a smarter exploration strategy.

---

## Strategy 3: Upper Confidence Bound (UCB)

### The Idea

Instead of exploring randomly, explore **where we're most uncertain**. If we've pulled arm 3 a hundred times, we have a good estimate. If we've pulled arm 7 twice, we barely know anything. UCB says: give uncertain arms a **bonus** to encourage pulling them.

The formula:

```
UCB(arm) = Q(arm) + c × √(ln(t) / N(arm))
```

Where:
- `Q(arm)` = estimated value (same as epsilon-greedy)
- `c` = exploration constant (controls how much we value uncertainty)
- `t` = total number of pulls so far
- `N(arm)` = number of times this arm has been pulled
- `√(ln(t) / N(arm))` = the **uncertainty bonus**

Always pick the arm with the highest UCB score.

### How the Bonus Works

The bonus term `√(ln(t) / N(arm))` has two elegant properties:

**Property 1: The bonus shrinks as you pull an arm more.** N(arm) increases, so the fraction decreases, so the bonus decreases. The more you know about an arm, the less reason to explore it.

```
After 1 pull:    bonus ∝ √(ln(100) / 1)   = √4.6  ≈ 2.14
After 10 pulls:  bonus ∝ √(ln(100) / 10)  = √0.46 ≈ 0.68
After 50 pulls:  bonus ∝ √(ln(100) / 50)  = √0.09 ≈ 0.30
```

**Property 2: The bonus grows slowly over time.** t increases (in the numerator via ln), so arms that haven't been pulled in a while gradually get a higher bonus. No arm is permanently forgotten.

```
At t=100:   bonus for arm with N=5: √(ln(100)/5)  = √0.92 ≈ 0.96
At t=1000:  bonus for arm with N=5: √(ln(1000)/5) = √1.38 ≈ 1.18
At t=10000: bonus for arm with N=5: √(ln(10000)/5)= √1.84 ≈ 1.36
```

### A Worked Example

After 100 total pulls, suppose:

```
Arm 1: Q = 0.80, pulled 40 times → UCB = 0.80 + 2 × √(ln(100)/40) = 0.80 + 0.68 = 1.48
Arm 2: Q = 0.50, pulled 5 times  → UCB = 0.50 + 2 × √(ln(100)/5)  = 0.50 + 1.92 = 2.42  ← PICK THIS
Arm 3: Q = 0.70, pulled 30 times → UCB = 0.70 + 2 × √(ln(100)/30) = 0.70 + 0.78 = 1.48
```

Arm 2 has the lowest estimated value (0.50) but the highest UCB score (2.42) because it's been pulled very few times. UCB says: "We barely know anything about arm 2 — its true value might be much higher than 0.50. Let's find out." This is **optimism in the face of uncertainty**.

After pulling arm 2 a few more times, its bonus decreases and other arms might take the lead.

### The Code

```python
class UCBAgent:
    def __init__(self, k: int, c: float = 2.0):
        self.k = k
        self.c = c
        self.q_values = np.zeros(k)
        self.arm_counts = np.zeros(k)
        self.total_pulls = 0

    def select_arm(self) -> int:
        self.total_pulls += 1

        unpulled = np.where(self.arm_counts == 0)[0]
        if len(unpulled) > 0:
            return int(unpulled[0])

        bonus = self.c * np.sqrt(np.log(self.total_pulls) / self.arm_counts)
        ucb_values = self.q_values + bonus
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        self.arm_counts[arm] += 1
        n = self.arm_counts[arm]
        self.q_values[arm] += (1.0 / n) * (reward - self.q_values[arm])
```

### Line-by-Line

```python
unpulled = np.where(self.arm_counts == 0)[0]
if len(unpulled) > 0:
    return int(unpulled[0])
```
- If any arm has never been pulled, pull it first. This avoids division by zero in the bonus formula and ensures every arm gets at least one observation.
- With 10 arms, the first 10 pulls are guaranteed to try each arm once.

```python
bonus = self.c * np.sqrt(np.log(self.total_pulls) / self.arm_counts)
```
- Vectorized computation: computes the bonus for ALL arms simultaneously.
- `np.log` is natural log (ln).
- `self.c = 2.0` is the exploration constant. Higher c = more exploration.

```python
ucb_values = self.q_values + bonus
return int(np.argmax(ucb_values))
```
- Add estimated value and uncertainty bonus, then pick the highest.
- No randomness at all — UCB is fully deterministic given the history. All exploration comes from the bonus term.

### Why UCB Had Higher Regret Than Epsilon-Greedy in Our Results

Our results showed UCB with 284.1 regret vs epsilon-greedy's 59.1. This might seem backwards — isn't UCB the "smarter" algorithm? The answer is nuanced.

UCB explores **more systematically** — it keeps pulling undersampled arms to reduce uncertainty. This is optimal in the long run (over millions of pulls) but costly in the short run (2000 pulls). Epsilon-greedy with aggressive decay committed to the best arm faster, wasting fewer pulls on exploration.

With 100,000 pulls instead of 2,000, UCB would likely overtake epsilon-greedy because its systematic exploration finds the true optimum more reliably.

---

## Strategy 4: Thompson Sampling

### The Idea

Thompson Sampling takes a **Bayesian** approach. Instead of maintaining a single estimate for each arm, it maintains a **probability distribution** over possible values.

Each step:
1. For each arm, sample a random value from its distribution
2. Pick the arm whose sample is highest
3. After observing the reward, update the distribution

Arms with uncertain distributions (few pulls) sometimes produce very high samples → natural exploration. Arms with known-good distributions consistently produce high samples → natural exploitation. The exploration-exploitation balance emerges automatically from the uncertainty in the distributions.

### The Beta Distribution

For binary rewards (win/lose), the **Beta distribution** is the perfect tool. Beta(α, β) is a distribution over probabilities (values between 0 and 1), parameterized by:
- α = number of successes + 1
- β = number of failures + 1

```
Beta(1, 1):   uniform — know nothing (prior)
Beta(2, 1):   1 win, 0 losses → skewed right (probably a good arm)
Beta(1, 2):   0 wins, 1 loss → skewed left (probably a bad arm)
Beta(10, 3):  9 wins, 2 losses → peaked near 0.75 (fairly confident)
Beta(50, 10): 49 wins, 9 losses → tightly peaked near 0.83 (very confident)
```

As you collect more data, the distribution narrows — you become more confident about the arm's true probability.

### A Visual Intuition

Imagine each arm has a "belief cloud" representing your uncertainty:

```
After 0 pulls (all arms):
  Arm 0: [==========] could be anything from 0 to 1
  Arm 1: [==========]
  Arm 2: [==========]

After 20 pulls (arm 0: 12 wins, arm 1: 3 wins, arm 2: not pulled):
  Arm 0:        [====]         peaked around 0.6, fairly confident
  Arm 1:  [====]               peaked around 0.3, somewhat confident
  Arm 2: [==========]          still totally uncertain

Thompson Sampling would probably pick arm 0 (high mean),
but might pick arm 2 (uncertain — could be great!)
```

### The Code

```python
class ThompsonSamplingAgent:
    def __init__(self, k: int):
        self.k = k
        self.alpha = np.ones(k)   # successes + 1
        self.beta = np.ones(k)    # failures + 1

    def select_arm(self) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
```

### Line-by-Line

```python
self.alpha = np.ones(k)
self.beta = np.ones(k)
```
- Start with Beta(1, 1) for every arm = uniform distribution = "I know nothing."

```python
samples = self.rng.beta(self.alpha, self.beta)
return int(np.argmax(samples))
```
- Sample one random value from each arm's Beta distribution.
- Pick the arm whose sample is highest.
- This is the entire algorithm. No epsilon, no bonus terms, no manual exploration. The randomness in the sampling IS the exploration.

```python
if reward > 0:
    self.alpha[arm] += 1
else:
    self.beta[arm] += 1
```
- Win → increment α (more evidence of success).
- Loss → increment β (more evidence of failure).
- The Beta distribution automatically narrows as evidence accumulates.

### Why Thompson Sampling Is Elegant

Thompson Sampling has a beautiful theoretical property: it explores each arm in proportion to the probability that it IS the best arm. If arm 3 has a 20% chance of being optimal (given current evidence), Thompson Sampling pulls it roughly 20% of the time. This is provably near-optimal.

Compare to epsilon-greedy, which explores all arms equally (regardless of their potential), and UCB, which explores based on uncertainty alone (regardless of the arm's estimated value). Thompson Sampling balances both factors naturally.

---

## Measuring Performance: Regret

### What Is Regret?

**Regret** measures how much reward you lost compared to the optimal strategy (always pulling the best arm):

```
Regret at step t = (optimal reward per step) × t - (total reward so far)
                 = best_prob × t - Σ rewards
```

A perfect agent (that knows the best arm from the start) has zero regret. A random agent accumulates regret linearly. A good learning agent accumulates regret that grows sublinearly — it starts with regret (while exploring) but the rate of regret decreases as it finds the best arm.

### Computing Regret in Code

```python
def compute_regret(rewards, best_prob, n_steps):
    optimal_rewards = best_prob * np.arange(1, n_steps + 1)
    actual_cumulative = np.cumsum(rewards)
    return optimal_rewards - actual_cumulative
```

- `optimal_rewards`: what you'd earn if you always pulled the best arm
- `actual_cumulative`: what you actually earned
- The difference is regret — always non-negative for a suboptimal strategy

### What We Observed

```
Random:           656.1 regret — linear growth, never learns
Epsilon-Greedy:    59.1 regret — fast convergence, early commitment
UCB:              284.1 regret — systematic exploration, slower convergence
Thompson:          80.1 regret — elegant balance, near-optimal
```

The ideal regret curve is logarithmic — fast initial growth (while exploring) that flattens as the agent locks onto the best arm. Both epsilon-greedy (with decay) and Thompson Sampling achieved this.

---

## Summary

| Strategy | Explores How? | Exploits How? | Balance |
|----------|---------------|---------------|---------|
| Random | Always exploring | Never exploits | No balance — pure baseline |
| ε-Greedy | Random ε% of the time | Best known arm (1-ε)% | Manual balance via ε |
| UCB | Uncertain arms get bonus | Highest (value + bonus) | Automatic via uncertainty |
| Thompson | Samples from beliefs | Highest sample wins | Automatic via distributions |

Key takeaways:

1. **Exploration is necessary but costly.** You must explore to find the best option, but every exploration pull is a pull that could have been exploitation.

2. **Simple works surprisingly well.** Epsilon-greedy with decay had the lowest regret in our experiment. Don't overcomplicate when the problem is simple.

3. **Uncertainty drives smart exploration.** Both UCB and Thompson Sampling use uncertainty to guide exploration — they explore where it matters most.

4. **These ideas recur everywhere in RL.** Epsilon-greedy reappears in Q-Learning and DQN. Uncertainty-driven exploration reappears in Bayesian RL and curiosity-driven methods. The Thompson Sampling principle (match exploration to probability of being optimal) underlies posterior sampling RL.

## What's Next

In [Chapter 2](02_q_learning.md), we add states and sequential decisions. The bandit problem has no notion of "where you are" — just pick an arm. In GridWorld, the agent has a position, actions move it, and the goal is to navigate to a target. This requires the agent to plan ahead — to value not just immediate rewards but future states. This introduces the **Bellman equation**, the mathematical backbone of all value-based RL.
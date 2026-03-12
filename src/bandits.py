"""
Multi-Armed Bandits — The simplest RL problem.

The setup: K slot machines, each with an unknown payout probability.
You have N pulls. Maximize total reward.

The dilemma: do you EXPLOIT the best arm you've found so far,
or EXPLORE other arms that might be better?

We implement four strategies and compare them.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict


# -----------------------------------------------------------------------
# The Environment
# -----------------------------------------------------------------------

class BanditEnvironment:
    """
    K-armed bandit. Each arm has a fixed (hidden) probability of paying out.
    Pull an arm → get reward 1 with that probability, else 0.
    """

    def __init__(self, k: int = 10, seed: int = 42):
        self.k = k
        self.rng = np.random.RandomState(seed)
        # Each arm's true payout probability (unknown to the agent)
        self.probs = self.rng.uniform(0.1, 0.9, size=k)
        self.best_arm = int(np.argmax(self.probs))
        self.best_prob = self.probs[self.best_arm]

    def pull(self, arm: int) -> float:
        """Pull an arm. Returns 1.0 (win) or 0.0 (lose)."""
        return 1.0 if self.rng.random() < self.probs[arm] else 0.0

    def __repr__(self):
        probs_str = ", ".join(f"{p:.2f}" for p in self.probs)
        return (
            f"BanditEnvironment(k={self.k})\n"
            f"  Probabilities: [{probs_str}]\n"
            f"  Best arm: {self.best_arm} (p={self.best_prob:.2f})"
        )


# -----------------------------------------------------------------------
# Strategy 1: Random (Baseline)
# -----------------------------------------------------------------------

class RandomAgent:
    """Pick an arm uniformly at random every time. The baseline."""

    def __init__(self, k: int):
        self.k = k
        self.rng = np.random.RandomState(0)

    def select_arm(self) -> int:
        return self.rng.randint(0, self.k)

    def update(self, arm: int, reward: float):
        pass  # Random agent doesn't learn


# -----------------------------------------------------------------------
# Strategy 2: Epsilon-Greedy
# -----------------------------------------------------------------------

class EpsilonGreedyAgent:
    """
    With probability (1-ε): pick the arm with highest estimated value (exploit)
    With probability ε: pick a random arm (explore)

    As we gather more data, we decay ε so we explore less over time.
    """

    def __init__(self, k: int, epsilon: float = 0.1, decay: float = 0.999):
        self.k = k
        self.epsilon = epsilon
        self.decay = decay
        self.rng = np.random.RandomState(1)

        # Track estimated value of each arm
        self.q_values = np.zeros(k)     # running average reward per arm
        self.arm_counts = np.zeros(k)   # how many times each arm was pulled

    def select_arm(self) -> int:
        if self.rng.random() < self.epsilon:
            # Explore: random arm
            return self.rng.randint(0, self.k)
        else:
            # Exploit: best known arm (break ties randomly)
            max_q = np.max(self.q_values)
            best_arms = np.where(self.q_values == max_q)[0]
            return self.rng.choice(best_arms)

    def update(self, arm: int, reward: float):
        # Incremental mean update:
        # new_avg = old_avg + (1/n) * (reward - old_avg)
        self.arm_counts[arm] += 1
        n = self.arm_counts[arm]
        self.q_values[arm] += (1.0 / n) * (reward - self.q_values[arm])

        # Decay epsilon
        self.epsilon *= self.decay


# -----------------------------------------------------------------------
# Strategy 3: Upper Confidence Bound (UCB)
# -----------------------------------------------------------------------

class UCBAgent:
    """
    Pick the arm that maximizes: estimated_value + exploration_bonus

    The exploration bonus is high for arms we haven't tried much.
    As an arm gets pulled more, its bonus shrinks — we become more
    confident in our estimate and don't need to explore it.

    Formula: UCB(a) = Q(a) + c × √(ln(t) / N(a))

    where:
      Q(a) = estimated value of arm a
      t    = total pulls so far
      N(a) = times arm a was pulled
      c    = exploration constant (higher = more exploration)
    """

    def __init__(self, k: int, c: float = 2.0):
        self.k = k
        self.c = c
        self.q_values = np.zeros(k)
        self.arm_counts = np.zeros(k)
        self.total_pulls = 0

    def select_arm(self) -> int:
        self.total_pulls += 1

        # If any arm hasn't been pulled, pull it (infinite bonus)
        unpulled = np.where(self.arm_counts == 0)[0]
        if len(unpulled) > 0:
            return int(unpulled[0])

        # UCB formula
        bonus = self.c * np.sqrt(np.log(self.total_pulls) / self.arm_counts)
        ucb_values = self.q_values + bonus
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        self.arm_counts[arm] += 1
        n = self.arm_counts[arm]
        self.q_values[arm] += (1.0 / n) * (reward - self.q_values[arm])


# -----------------------------------------------------------------------
# Strategy 4: Thompson Sampling
# -----------------------------------------------------------------------

class ThompsonSamplingAgent:
    """
    Bayesian approach. For each arm, maintain a probability distribution
    (Beta distribution) over its true payout rate.

    Each step:
    1. Sample a random value from each arm's distribution
    2. Pick the arm whose sample is highest

    Arms with uncertain distributions (few pulls) produce high samples
    sometimes → natural exploration. Arms with known-good distributions
    produce consistently high samples → natural exploitation.

    The Beta distribution is perfect for binary rewards (0 or 1):
      Beta(α, β) where α = successes + 1, β = failures + 1
    """

    def __init__(self, k: int):
        self.k = k
        self.rng = np.random.RandomState(2)

        # Beta distribution parameters for each arm
        # Start with Beta(1,1) = uniform distribution (no prior knowledge)
        self.alpha = np.ones(k)   # successes + 1
        self.beta = np.ones(k)    # failures + 1

    def select_arm(self) -> int:
        # Sample from each arm's Beta distribution
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        # Update the Beta distribution
        if reward > 0:
            self.alpha[arm] += 1   # one more success
        else:
            self.beta[arm] += 1    # one more failure


# -----------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------

def run_experiment(env, agent, n_steps: int = 1000):
    """Run an agent on a bandit environment for n_steps."""
    rewards = np.zeros(n_steps)
    arms_chosen = np.zeros(n_steps, dtype=int)

    for t in range(n_steps):
        arm = agent.select_arm()
        reward = env.pull(arm)
        agent.update(arm, reward)

        rewards[t] = reward
        arms_chosen[t] = arm

    return rewards, arms_chosen


def compute_regret(rewards, best_prob, n_steps):
    """
    Regret = what you COULD have earned - what you DID earn.

    If you always pulled the best arm, expected reward = best_prob × n_steps.
    Regret measures how far from optimal you were.
    """
    optimal_rewards = best_prob * np.arange(1, n_steps + 1)
    actual_cumulative = np.cumsum(rewards)
    return optimal_rewards - actual_cumulative


# -----------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------

def plot_results(results, env, n_steps, save_path="notebooks/bandits_comparison.png"):
    """Plot comparison of all strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Multi-Armed Bandit Comparison (K={env.k}, N={n_steps})", fontsize=14)

    colors = {"Random": "#999999", "ε-Greedy": "#e74c3c",
              "UCB": "#3498db", "Thompson": "#2ecc71"}

    # Plot 1: Cumulative reward
    ax = axes[0, 0]
    for name, (rewards, _) in results.items():
        cumulative = np.cumsum(rewards)
        ax.plot(cumulative, label=name, color=colors[name], linewidth=2)
    ax.set_title("Cumulative Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Regret
    ax = axes[0, 1]
    for name, (rewards, _) in results.items():
        regret = compute_regret(rewards, env.best_prob, n_steps)
        ax.plot(regret, label=name, color=colors[name], linewidth=2)
    ax.set_title("Cumulative Regret (lower is better)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Regret")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Running average reward (smoothed)
    ax = axes[1, 0]
    window = 50
    for name, (rewards, _) in results.items():
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax.plot(smoothed, label=name, color=colors[name], linewidth=2)
    ax.axhline(y=env.best_prob, color="black", linestyle="--", alpha=0.5,
               label=f"Optimal ({env.best_prob:.2f})")
    ax.set_title(f"Average Reward (rolling {window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Avg Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Arm selection frequency
    ax = axes[1, 1]
    bar_width = 0.2
    x = np.arange(env.k)
    for i, (name, (_, arms)) in enumerate(results.items()):
        counts = np.bincount(arms, minlength=env.k) / n_steps
        ax.bar(x + i * bar_width, counts, bar_width, label=name,
               color=colors[name], alpha=0.8)
    # Mark the best arm
    ax.axvline(x=env.best_arm, color="black", linestyle="--", alpha=0.3)
    ax.set_title("Arm Selection Frequency")
    ax.set_xlabel("Arm")
    ax.set_ylabel("Fraction of Pulls")
    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels([f"{i}\n({env.probs[i]:.2f})" for i in range(env.k)],
                       fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.close()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    K = 10           # number of arms
    N = 2000         # number of pulls
    SEED = 42

    # Create environment
    env = BanditEnvironment(k=K, seed=SEED)
    print(env)
    print()

    # Create agents
    agents = {
        "Random": RandomAgent(K),
        "ε-Greedy": EpsilonGreedyAgent(K, epsilon=0.15, decay=0.998),
        "UCB": UCBAgent(K, c=2.0),
        "Thompson": ThompsonSamplingAgent(K),
    }

    # Run experiments
    results = {}
    for name, agent in agents.items():
        # Each agent gets its own fresh environment with the same seed
        test_env = BanditEnvironment(k=K, seed=SEED)
        rewards, arms = run_experiment(test_env, agent, n_steps=N)
        total = np.sum(rewards)
        final_regret = compute_regret(rewards, env.best_prob, N)[-1]
        print(f"{name:>12s}: total reward = {total:.0f}, "
              f"final regret = {final_regret:.1f}, "
              f"avg reward = {total/N:.3f}")
        results[name] = (rewards, arms)

    print(f"\n  Optimal avg reward: {env.best_prob:.3f}")

    # Plot
    plot_results(results, env, N)
    print("\nDone! Check notebooks/bandits_comparison.png")
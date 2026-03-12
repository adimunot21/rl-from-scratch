"""
REINFORCE — Policy Gradient from Scratch.

A fundamentally different approach from DQN:

DQN:       Learn Q-values → derive policy (pick action with max Q)
REINFORCE: Learn the policy DIRECTLY → no Q-values at all

The policy is a neural network that outputs a probability distribution
over actions. We sample from this distribution, observe the outcome,
and adjust the probabilities:
  - Actions that led to HIGH returns → increase their probability
  - Actions that led to LOW returns → decrease their probability

The math:  ∇J(θ) = E[ Σₜ ∇log π(aₜ|sₜ) · Gₜ ]

Where:
  π(aₜ|sₜ) = probability the policy assigns to action aₜ in state sₜ
  Gₜ        = total return from timestep t onward
  ∇log π    = direction to push the parameters to make aₜ more likely

Multiply by Gₜ:
  - If Gₜ is large (good outcome): push hard to make aₜ more likely
  - If Gₜ is small (bad outcome): push weakly (or not at all)

This is called the "log-probability trick" or "likelihood ratio trick."
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------
# Policy Network
# -----------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    Outputs a probability distribution over actions.

    Unlike DQN's Q-network (which outputs values), this outputs
    probabilities via softmax. The agent SAMPLES from these
    probabilities rather than taking argmax.

    Input:  state (4 values for CartPole)
    Output: action probabilities (2 values — prob of left, prob of right)
    """

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
        """
        Sample an action from the policy distribution.

        Returns the action AND its log-probability (needed for the update).
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state_t)

        # Categorical creates a distribution we can sample from
        dist = Categorical(probs)
        action = dist.sample()            # sample one action
        log_prob = dist.log_prob(action)   # log π(a|s)

        return action.item(), log_prob


# -----------------------------------------------------------------------
# REINFORCE Agent
# -----------------------------------------------------------------------

class REINFORCEAgent:
    """
    REINFORCE with optional baseline subtraction.

    The baseline reduces variance without introducing bias.
    We use the running average of returns as a simple baseline.

    Without baseline:
      ∇J = E[ Σₜ ∇log π(aₜ|sₜ) · Gₜ ]

    With baseline:
      ∇J = E[ Σₜ ∇log π(aₜ|sₜ) · (Gₜ - b) ]

    Where b is the baseline (average return). This means:
      - Returns above average → positive signal → increase probability
      - Returns below average → negative signal → decrease probability

    The gradient still points in the right direction, but with much
    less variance — learning is faster and more stable.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        use_baseline: bool = True,
    ):
        self.gamma = gamma
        self.use_baseline = use_baseline

        self.policy = PolicyNetwork(state_dim, n_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Episode storage (cleared after each update)
        self.log_probs = []
        self.rewards = []

        # Running baseline
        self.baseline = 0.0
        self.baseline_count = 0

    def select_action(self, state):
        """Sample action from current policy."""
        action, log_prob = self.policy.get_action(state)
        self.log_probs.append(log_prob)
        return action

    def store_reward(self, reward):
        """Store reward for current step."""
        self.rewards.append(reward)

    def compute_returns(self):
        """
        Compute discounted returns for each timestep.

        Gₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + ... + γᵀ⁻ᵗ·rₜ

        We compute this backwards for efficiency:
          G_T = r_T
          G_{T-1} = r_{T-1} + γ · G_T
          G_{T-2} = r_{T-2} + γ · G_{T-1}
          ...
        """
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def update(self):
        """
        Update the policy using the REINFORCE algorithm.

        This is called once per episode (after the episode is complete).
        REINFORCE is a MONTE CARLO method — it needs full episodes.
        """
        returns = self.compute_returns()
        returns_t = torch.FloatTensor(returns)

        # Update running baseline
        episode_return = returns[0]  # G₀ = total episode return
        self.baseline_count += 1
        self.baseline += (episode_return - self.baseline) / self.baseline_count

        # Apply baseline: center the returns
        if self.use_baseline:
            returns_t = returns_t - self.baseline

        # Normalize returns for training stability
        if len(returns_t) > 1 and returns_t.std() > 0:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Compute policy gradient loss
        # loss = -Σₜ log π(aₜ|sₜ) · Gₜ
        # The negative sign is because PyTorch minimizes loss,
        # but we want to MAXIMIZE expected return
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns_t):
            policy_loss.append(-log_prob * G)

        loss = torch.stack(policy_loss).sum()

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # Clear episode storage
        self.log_probs = []
        self.rewards = []

        return loss.item()


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_reinforce(env_name="CartPole-v1", n_episodes=1000, use_baseline=True):
    """Train REINFORCE on a Gymnasium environment."""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = REINFORCEAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        lr=1e-3,
        gamma=0.99,
        use_baseline=use_baseline,
    )

    label = "REINFORCE" + (" + baseline" if use_baseline else "")
    print(f"Training {label} on {env_name}")
    n_params = sum(p.numel() for p in agent.policy.parameters())
    print(f"Policy network parameters: {n_params:,}")
    print()

    episode_rewards = []
    losses = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.store_reward(reward)

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        # Update AFTER the full episode (Monte Carlo)
        loss = agent.update()
        episode_rewards.append(total_reward)
        losses.append(loss)

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(episode_rewards[-100:])
            print(f"Episode {ep+1:5d} | "
                  f"avg reward: {avg_r:7.1f} | "
                  f"loss: {np.mean(losses[-100:]):.4f}")

    env.close()
    return agent, episode_rewards, losses


def evaluate_reinforce(agent, env_name="CartPole-v1", n_episodes=20):
    """Evaluate trained policy (greedy — pick most probable action)."""
    env = gym.make(env_name)
    rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        total = 0
        while True:
            # Greedy: pick most probable action (no sampling)
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs = agent.policy(state_t)
            action = probs.argmax(dim=1).item()

            state, reward, term, trunc, _ = env.step(action)
            total += reward
            if term or trunc:
                break
        rewards.append(total)

    env.close()
    return rewards


def plot_comparison(results, env_name, save_path="notebooks/reinforce_comparison.png"):
    """Plot REINFORCE with and without baseline."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    window = 50
    colors = {"REINFORCE": "#e74c3c", "REINFORCE + baseline": "#3498db"}

    for name, (rewards, losses) in results.items():
        color = colors.get(name, "#333333")

        # Rewards
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax1.plot(rewards, alpha=0.15, color=color)
        ax1.plot(range(window-1, len(rewards)), smoothed,
                 linewidth=2, color=color, label=name)

    ax1.set_title(f"Episode Reward — {env_name}")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Raw rewards variance comparison (last 200 episodes)
    ax2.set_title("Reward Distribution (last 200 episodes)")
    data = []
    labels = []
    for name, (rewards, _) in results.items():
        data.append(rewards[-200:])
        labels.append(name)
    ax2.boxplot(data, labels=labels)
    ax2.set_ylabel("Reward")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("REINFORCE: Effect of Baseline", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.close()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    ENV = "CartPole-v1"
    N_EPISODES = 1000

    results = {}

    # Train WITHOUT baseline
    print("=" * 50)
    agent_no_bl, rewards_no_bl, losses_no_bl = train_reinforce(
        env_name=ENV, n_episodes=N_EPISODES, use_baseline=False
    )
    results["REINFORCE"] = (rewards_no_bl, losses_no_bl)

    print(f"\nEvaluation (no baseline):")
    eval_r = evaluate_reinforce(agent_no_bl, env_name=ENV)
    print(f"  Avg: {np.mean(eval_r):.1f} ± {np.std(eval_r):.1f}")

    # Train WITH baseline
    print("\n" + "=" * 50)
    agent_bl, rewards_bl, losses_bl = train_reinforce(
        env_name=ENV, n_episodes=N_EPISODES, use_baseline=True
    )
    results["REINFORCE + baseline"] = (rewards_bl, losses_bl)

    print(f"\nEvaluation (with baseline):")
    eval_r = evaluate_reinforce(agent_bl, env_name=ENV)
    print(f"  Avg: {np.mean(eval_r):.1f} ± {np.std(eval_r):.1f}")

    # Plot comparison
    plot_comparison(results, ENV)

    # Save
    torch.save(agent_bl.policy.state_dict(), "checkpoints/reinforce_cartpole.pt")
    print("Saved checkpoint to checkpoints/reinforce_cartpole.pt")
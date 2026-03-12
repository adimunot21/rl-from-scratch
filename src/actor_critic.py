"""
Advantage Actor-Critic (A2C) — Combining Value-Based and Policy-Based RL.

The problem with REINFORCE:
  - Uses full episode returns (high variance, slow learning)
  - A single bad episode can ruin the policy
  - The baseline is a crude running average, not state-dependent

The fix — two networks that help each other:

  ACTOR:  the policy network (same as REINFORCE)
          outputs action probabilities
          updated using advantage-weighted policy gradient

  CRITIC: a value network (estimates V(s) — how good is this state?)
          provides a LEARNED, STATE-DEPENDENT baseline for the actor
          updated by minimizing prediction error

The key concept: ADVANTAGE

  A(s, a) = Gₜ - V(s)
          = "how much BETTER was the actual return than what the critic expected?"

  If A > 0: the return was better than expected → increase action probability
  If A < 0: the return was worse than expected → decrease action probability

This is like REINFORCE, but the baseline V(s) is:
  1. Learned (improves over time, unlike the fixed running average)
  2. State-dependent (different baseline for different situations)

Both properties dramatically reduce variance compared to vanilla REINFORCE.
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
# Separate Actor and Critic Networks
# -----------------------------------------------------------------------

class Actor(nn.Module):
    """Policy network — outputs action probabilities."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
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
    """Value network — estimates V(s), how good a state is."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
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


# -----------------------------------------------------------------------
# A2C Agent
# -----------------------------------------------------------------------

class A2CAgent:
    """
    Advantage Actor-Critic.

    Collects full episodes (like REINFORCE), then uses the critic to
    compute advantages for the policy update.

    Two separate optimizers for actor and critic — this prevents
    competing gradients from destabilizing training.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        actor_lr: float = 1e-3,
        critic_lr: float = 5e-3,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
    ):
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff

        self.actor = Actor(state_dim, n_actions)
        self.critic = Critic(state_dim)

        # Separate optimizers — critic can learn faster than actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Episode storage
        self.log_probs = []
        self.rewards = []
        self.states = []
        self.entropies = []

    def select_action(self, state):
        """Sample action from the actor's distribution."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state_t)
        dist = Categorical(probs)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        self.states.append(state)
        self.entropies.append(dist.entropy())

        return action.item()

    def store_reward(self, reward):
        """Store reward for current step."""
        self.rewards.append(reward)

    def compute_returns(self):
        """Compute discounted returns for each timestep."""
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)

    def update(self):
        """
        Update actor and critic after a complete episode.

        1. Compute actual returns (Monte Carlo)
        2. Get critic's value estimates for each state
        3. Advantage = returns - values (how much better than expected?)
        4. Update actor to increase probability of above-average actions
        5. Update critic to predict returns more accurately
        """
        returns = self.compute_returns()

        states_t = torch.FloatTensor(np.array(self.states))
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        # Critic's value estimates
        values = self.critic(states_t)

        # Advantage = actual return - critic's prediction
        advantages = returns - values.detach()  # detach: don't backprop actor loss into critic

        # NOTE: We do NOT normalize advantages here.
        # When all episodes have similar returns, normalization crushes
        # the signal to zero. Raw advantages let the critic's errors
        # drive meaningful actor updates.

        # ---- Update Actor ----
        # Increase probability of actions with positive advantage
        actor_loss = -(log_probs * advantages).mean()
        entropy_bonus = entropies.mean()
        actor_total = actor_loss - self.entropy_coeff * entropy_bonus

        self.actor_optimizer.zero_grad()
        actor_total.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ---- Update Critic ----
        # Train to predict actual returns
        critic_loss = F.mse_loss(values, returns.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Clear episode storage
        self.log_probs = []
        self.rewards = []
        self.states = []
        self.entropies = []

        return actor_loss.item(), critic_loss.item()


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_a2c(env_name="CartPole-v1", n_episodes=1000):
    """Train A2C on a Gymnasium environment."""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = A2CAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        actor_lr=3e-3,
        critic_lr=1e-3,
        gamma=0.99,
        entropy_coeff=0.01,
    )

    actor_params = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())
    print(f"Training A2C on {env_name}")
    print(f"Actor parameters: {actor_params:,}")
    print(f"Critic parameters: {critic_params:,}")
    print()

    episode_rewards = []
    actor_losses = []
    critic_losses = []

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

        # Update after complete episode
        a_loss, c_loss = agent.update()
        episode_rewards.append(total_reward)
        actor_losses.append(a_loss)
        critic_losses.append(c_loss)

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(episode_rewards[-100:])
            avg_al = np.mean(actor_losses[-100:])
            avg_cl = np.mean(critic_losses[-100:])
            print(f"Episode {ep+1:5d} | "
                  f"avg reward: {avg_r:7.1f} | "
                  f"actor loss: {avg_al:8.4f} | "
                  f"critic loss: {avg_cl:8.4f}")

    env.close()
    return agent, episode_rewards, actor_losses, critic_losses


def evaluate_a2c(agent, env_name="CartPole-v1", n_episodes=20):
    """Evaluate trained agent (greedy — pick most probable action)."""
    env = gym.make(env_name)
    rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        total = 0
        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs = agent.actor(state_t)
            action = probs.argmax(dim=1).item()
            state, reward, term, trunc, _ = env.step(action)
            total += reward
            if term or trunc:
                break
        rewards.append(total)

    env.close()
    return rewards


def plot_a2c_results(rewards, actor_losses, critic_losses, env_name,
                     save_path="notebooks/a2c_training.png"):
    """Plot A2C training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    window = 50

    # Rewards
    ax = axes[0]
    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
    ax.plot(rewards, alpha=0.2, color="#3498db")
    ax.plot(range(window-1, len(rewards)), smoothed, linewidth=2, color="#3498db")
    ax.set_title(f"Episode Reward — {env_name}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)

    # Actor loss
    ax = axes[1]
    if actor_losses:
        sm = np.convolve(actor_losses, np.ones(window)/window, mode="valid")
        ax.plot(actor_losses, alpha=0.1, color="#e74c3c")
        ax.plot(range(window-1, len(actor_losses)), sm, linewidth=2, color="#e74c3c")
    ax.set_title("Actor Loss")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Critic loss
    ax = axes[2]
    if critic_losses:
        sm = np.convolve(critic_losses, np.ones(window)/window, mode="valid")
        ax.plot(critic_losses, alpha=0.1, color="#2ecc71")
        ax.plot(range(window-1, len(critic_losses)), sm, linewidth=2, color="#2ecc71")
    ax.set_title("Critic Loss")
    ax.set_xlabel("Episode")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Advantage Actor-Critic (A2C) Training", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.close()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    ENV = "CartPole-v1"

    # Train
    agent, rewards, a_losses, c_losses = train_a2c(
        env_name=ENV,
        n_episodes=1000,
    )

    # Evaluate
    print(f"\n{'='*50}")
    print("EVALUATION (20 episodes)")
    print(f"{'='*50}")
    eval_rewards = evaluate_a2c(agent, env_name=ENV)
    print(f"Avg reward: {np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}")
    print(f"Max: {max(eval_rewards):.0f}, Min: {min(eval_rewards):.0f}")

    # Plot
    plot_a2c_results(rewards, a_losses, c_losses, ENV)

    # Save
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
    }, "checkpoints/a2c_cartpole.pt")
    print("Saved checkpoint to checkpoints/a2c_cartpole.pt")
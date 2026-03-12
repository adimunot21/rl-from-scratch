"""
Proximal Policy Optimization (PPO) — The Industry Standard.

The problem with A2C:
  - One bad policy update can destroy a good policy
  - No mechanism to limit how much the policy changes per update
  - Wastes data: each batch of experience is used for exactly one update

PPO fixes all three with one elegant idea: the CLIPPED SURROGATE OBJECTIVE.

Instead of:  L = log π(a|s) · A          (vanilla policy gradient)

PPO uses:    L = min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)

Where:
  r(θ) = π_new(a|s) / π_old(a|s)    "probability ratio"
  ε = 0.2                            "clip range"

What this does:
  - If the new policy is too different from the old one (r far from 1),
    the gradient is clipped — the update is limited.
  - This prevents catastrophic policy updates.
  - Because updates are limited, we can safely reuse the SAME batch
    of data for MULTIPLE gradient steps (epochs) — much more data-efficient.

PPO also uses Generalized Advantage Estimation (GAE) for smoother advantages.

This is the algorithm behind:
  - OpenAI Five (Dota 2)
  - ChatGPT's RLHF training
  - Most modern robotics RL
  - Virtually all production RL systems
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
# Actor-Critic Network
# -----------------------------------------------------------------------

class PPONetwork(nn.Module):
    """
    Shared-backbone actor-critic for PPO.

    Using a shared backbone here (unlike our A2C where we separated them)
    because PPO's clipped objective prevents the destabilization that
    hurt our shared A2C. The clipping acts as a stabilizer.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value.squeeze(-1)

    def get_action(self, state):
        """Sample action, return action + log_prob + value."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        logits, value = self.forward(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate_actions(self, states, actions):
        """
        Re-evaluate actions under the CURRENT policy.

        This is the key to PPO: we collected data with the OLD policy,
        but now we evaluate those same actions with the NEW (updated) policy
        to compute the probability ratio r(θ).
        """
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


# -----------------------------------------------------------------------
# Rollout Buffer
# -----------------------------------------------------------------------

class RolloutBuffer:
    """
    Stores a batch of experience for PPO training.

    Unlike DQN's replay buffer (random sampling from history),
    PPO uses ALL collected data for multiple epochs, then discards it.
    This is an ON-POLICY method — data must come from the current policy.
    """

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
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def get(self):
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.dones),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.values),
        )

    def __len__(self):
        return len(self.states)


# -----------------------------------------------------------------------
# GAE: Generalized Advantage Estimation
# -----------------------------------------------------------------------

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE).

    GAE is a weighted average of n-step advantage estimates:

      A_GAE = Σₖ (γλ)ᵏ · δₜ₊ₖ

    Where δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)  (TD error at step t)

    The parameter λ (lambda) controls the bias-variance tradeoff:
      λ = 0: A = δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)   (1-step TD, low variance, high bias)
      λ = 1: A = Gₜ - V(sₜ)                       (Monte Carlo, zero bias, high variance)
      λ = 0.95: smooth blend (the standard choice)

    Returns:
      advantages: (T,) tensor
      returns: (T,) tensor (advantages + values = targets for critic)
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        # If episode ended at step t, there's no future value
        next_non_terminal = 1.0 - dones[t]

        # TD error: actual reward + discounted next value - current value
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]

        # GAE: exponentially weighted sum of TD errors
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae

    returns = advantages + torch.FloatTensor(values) if isinstance(values, list) else advantages + values
    return advantages, returns


# -----------------------------------------------------------------------
# PPO Agent
# -----------------------------------------------------------------------

class PPOAgent:
    """
    PPO with clipped surrogate objective and GAE.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_range: float = 0.2,
        n_epochs: int = 4,
        batch_size: int = 64,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        rollout_steps: int = 2048,
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.rollout_steps = rollout_steps

        self.network = PPONetwork(state_dim, n_actions)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def select_action(self, state):
        """Select action and store transition data."""
        action, log_prob, value = self.network.get_action(state)
        return action, log_prob, value

    def update(self, next_value):
        """
        PPO update — the heart of the algorithm.

        1. Compute GAE advantages from the collected rollout
        2. For multiple epochs:
           a. Split data into mini-batches
           b. Compute probability ratio r(θ) = π_new / π_old
           c. Compute clipped surrogate loss
           d. Update network
        3. Clear the buffer
        """
        states, actions, rewards, dones, old_log_probs, old_values = self.buffer.get()

        # Compute GAE advantages
        advantages, returns = compute_gae(
            rewards.tolist(), old_values.tolist(), dones.tolist(),
            next_value, self.gamma, self.lam
        )

        # Normalize advantages across the full rollout
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_samples = len(states)
        all_actor_losses = []
        all_critic_losses = []
        all_clip_fracs = []

        # Multiple epochs over the same data (PPO's sample efficiency advantage)
        for epoch in range(self.n_epochs):
            # Random mini-batch indices
            indices = torch.randperm(total_samples)

            for start in range(0, total_samples, self.batch_size):
                end = min(start + self.batch_size, total_samples)
                batch_idx = indices[start:end]

                # Get mini-batch
                mb_states = states[batch_idx]
                mb_actions = actions[batch_idx]
                mb_old_log_probs = old_log_probs[batch_idx]
                mb_advantages = advantages[batch_idx]
                mb_returns = returns[batch_idx]

                # Evaluate actions under CURRENT policy
                new_log_probs, new_values, entropy = self.network.evaluate_actions(
                    mb_states, mb_actions
                )

                # ---- The Clipped Surrogate Objective ----

                # Probability ratio: how has the policy changed?
                # r = π_new(a|s) / π_old(a|s)
                # = exp(log π_new - log π_old)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Two surrogate objectives:
                # 1. Unclipped: r(θ) · A
                surr1 = ratio * mb_advantages

                # 2. Clipped: clip(r(θ), 1-ε, 1+ε) · A
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range,
                                            1.0 + self.clip_range)
                surr2 = clipped_ratio * mb_advantages

                # Take the MINIMUM of the two
                # This is conservative: if the unclipped objective suggests
                # a big change but the clipped one disagrees, we take the
                # smaller update. Prevents catastrophic policy changes.
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss: predict returns accurately
                critic_loss = F.mse_loss(new_values, mb_returns.detach())

                # Entropy bonus: encourage exploration
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (actor_loss
                        + self.value_coeff * critic_loss
                        + self.entropy_coeff * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                # Track statistics
                all_actor_losses.append(actor_loss.item())
                all_critic_losses.append(critic_loss.item())
                clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                all_clip_fracs.append(clip_frac)

        self.buffer.clear()

        return {
            "actor_loss": np.mean(all_actor_losses),
            "critic_loss": np.mean(all_critic_losses),
            "clip_frac": np.mean(all_clip_fracs),
        }


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_ppo(env_name="CartPole-v1", total_timesteps=100000):
    """
    Train PPO. Unlike previous algorithms that count episodes,
    PPO counts TIMESTEPS — it collects a fixed number of steps,
    then updates, regardless of episode boundaries.
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = PPOAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_range=0.2,
        n_epochs=4,
        batch_size=64,
        entropy_coeff=0.01,
        value_coeff=0.5,
        rollout_steps=2048,
    )

    n_params = sum(p.numel() for p in agent.network.parameters())
    print(f"Training PPO on {env_name}")
    print(f"Network parameters: {n_params:,}")
    print(f"Rollout steps: {agent.rollout_steps}")
    print(f"Epochs per update: {agent.n_epochs}")
    print(f"Total timesteps: {total_timesteps:,}")
    print()

    episode_rewards = []
    update_stats = []
    timestep = 0
    episode_count = 0

    state, _ = env.reset()
    current_ep_reward = 0

    while timestep < total_timesteps:
        # ---- Collect rollout ----
        for _ in range(agent.rollout_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.store(state, action, reward, float(done), log_prob, value)
            current_ep_reward += reward
            timestep += 1
            state = next_state

            if done:
                episode_rewards.append(current_ep_reward)
                episode_count += 1
                current_ep_reward = 0
                state, _ = env.reset()

            if timestep >= total_timesteps:
                break

        # ---- Update ----
        # Get value estimate for the last state (bootstrap)
        with torch.no_grad():
            _, _, next_value = agent.network.get_action(state)

        stats = agent.update(next_value)
        update_stats.append(stats)

        # Log
        if episode_rewards:
            recent = episode_rewards[-20:] if len(episode_rewards) >= 20 else episode_rewards
            avg_r = np.mean(recent)
            print(f"Timestep {timestep:7d} | "
                  f"episodes: {episode_count:4d} | "
                  f"avg reward: {avg_r:7.1f} | "
                  f"actor loss: {stats['actor_loss']:8.4f} | "
                  f"critic loss: {stats['critic_loss']:8.2f} | "
                  f"clip frac: {stats['clip_frac']:.3f}")

    env.close()
    return agent, episode_rewards, update_stats


def evaluate_ppo(agent, env_name="CartPole-v1", n_episodes=20):
    """Evaluate trained PPO agent."""
    env = gym.make(env_name)
    rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        total = 0
        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logits, _ = agent.network(state_t)
            action = logits.argmax(dim=1).item()  # greedy
            state, reward, term, trunc, _ = env.step(action)
            total += reward
            if term or trunc:
                break
        rewards.append(total)

    env.close()
    return rewards


def plot_ppo_results(rewards, stats, env_name,
                     save_path="notebooks/ppo_training.png"):
    """Plot PPO training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    window = 20

    # Episode rewards
    ax = axes[0, 0]
    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
    ax.plot(rewards, alpha=0.2, color="#3498db")
    ax.plot(range(window-1, len(rewards)), smoothed, linewidth=2, color="#3498db")
    ax.set_title(f"Episode Reward — {env_name}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)

    # Actor loss per update
    ax = axes[0, 1]
    actor_losses = [s["actor_loss"] for s in stats]
    ax.plot(actor_losses, linewidth=2, color="#e74c3c")
    ax.set_title("Actor Loss per Update")
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Critic loss per update
    ax = axes[1, 0]
    critic_losses = [s["critic_loss"] for s in stats]
    ax.plot(critic_losses, linewidth=2, color="#2ecc71")
    ax.set_title("Critic Loss per Update")
    ax.set_xlabel("Update")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)

    # Clip fraction per update
    ax = axes[1, 1]
    clip_fracs = [s["clip_frac"] for s in stats]
    ax.plot(clip_fracs, linewidth=2, color="#9b59b6")
    ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="Typical max")
    ax.set_title("Clip Fraction (how often clipping activates)")
    ax.set_xlabel("Update")
    ax.set_ylabel("Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("PPO Training", fontsize=14)
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
    agent, rewards, stats = train_ppo(
        env_name=ENV,
        total_timesteps=200000,
    )

    # Evaluate
    print(f"\n{'='*50}")
    print("EVALUATION (20 episodes)")
    print(f"{'='*50}")
    eval_rewards = evaluate_ppo(agent, env_name=ENV)
    print(f"Avg reward: {np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}")
    print(f"Max: {max(eval_rewards):.0f}, Min: {min(eval_rewards):.0f}")

    # Plot
    plot_ppo_results(rewards, stats, ENV)

    # Save
    torch.save(agent.network.state_dict(), "checkpoints/ppo_cartpole.pt")
    print("Saved checkpoint to checkpoints/ppo_cartpole.pt")
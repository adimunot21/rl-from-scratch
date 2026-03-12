"""
Deep Q-Network (DQN) — Q-Learning with Neural Networks.

The problem: Q-tables can't handle continuous or large state spaces.
CartPole has 4 continuous values (position, velocity, angle, angular velocity).
There are infinite possible states — we can't store a value for each one.

The solution: replace the Q-table with a neural network that takes a state
as input and outputs Q-values for each action. The network GENERALIZES —
it can estimate Q-values for states it's never seen before.

Three key innovations from DeepMind's 2015 paper:

1. EXPERIENCE REPLAY: Store transitions in a buffer, sample random batches.
   Without this, the network trains on correlated consecutive transitions,
   which breaks the i.i.d. assumption of gradient descent.

2. TARGET NETWORK: A frozen copy of the Q-network used to compute targets.
   Without this, the target Q-values shift every update (moving target problem),
   making training unstable.

3. EPSILON-GREEDY with decay: Start with random exploration, gradually
   shift to exploitation as the network learns.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from collections import deque
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------
# Replay Buffer
# -----------------------------------------------------------------------

class ReplayBuffer:
    """
    Stores past experiences for random sampling.

    Why random sampling? During a CartPole episode, consecutive transitions
    are highly correlated (the state barely changes between steps). Training
    a neural network on correlated data causes it to "forget" older patterns
    as it overfits to recent ones.

    By sampling RANDOM batches from a large buffer of past experiences, we
    break this correlation. Each batch contains transitions from many different
    episodes and time steps — much more diverse training data.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# -----------------------------------------------------------------------
# Q-Network
# -----------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Neural network that approximates Q(s, a) for all actions simultaneously.

    Input:  state vector (4 values for CartPole)
    Output: Q-value for each action (2 values for CartPole — left, right)

    Architecture: simple MLP (multi-layer perceptron)
      state → Linear(128) → ReLU → Linear(128) → ReLU → Linear(n_actions)

    This is intentionally simple. DQN doesn't need a fancy architecture —
    the innovations are in the TRAINING PROCEDURE, not the network.
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
        return self.net(x)


# -----------------------------------------------------------------------
# DQN Agent
# -----------------------------------------------------------------------

class DQNAgent:
    """
    The complete DQN agent with all three innovations.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # The Q-network (the one we train)
        self.q_net = QNetwork(state_dim, n_actions).to(self.device)

        # The target network (frozen copy, updated periodically)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # never trains — only used for computing targets

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=buffer_size)

        self.train_steps = 0

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        # Use the Q-network to pick the best action
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        One training step: sample a batch, compute targets, update Q-network.

        This is where all three DQN innovations come together.
        """
        if len(self.buffer) < self.batch_size:
            return None  # not enough data yet

        # --- Innovation 1: Experience Replay ---
        # Sample a RANDOM batch from the buffer (breaks correlation)
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a) for the actions we actually took
        # q_net(states) gives Q for ALL actions → .gather picks the ones we took
        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # --- Innovation 2: Target Network ---
        # Compute targets using the FROZEN target network
        # target = r + γ · max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1).values
            # If done, there's no future reward
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        # Loss: Huber loss (smooth_l1) is more robust than MSE
        # MSE squares large errors, amplifying Q-value overestimates
        # Huber loss is linear for large errors, preventing the explosion
        loss = F.smooth_l1_loss(current_q, target)

        # Update the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1

        # --- Innovation 2 (continued): Periodically update target network ---
        # Copy the Q-network weights to the target network
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Reduce exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_dqn(env_name="CartPole-v1", n_episodes=500, device="cpu"):
    """Train DQN on a Gymnasium environment."""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        lr=5e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.990,
        buffer_size=50000,
        batch_size=64,
        target_update_freq=1000,
        device=device,
    )

    print(f"Training DQN on {env_name}")
    print(f"State dim: {state_dim}, Actions: {n_actions}")
    n_params = sum(p.numel() for p in agent.q_net.parameters())
    print(f"Q-Network parameters: {n_params:,}")
    print()

    episode_rewards = []
    episode_lengths = []
    losses = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        ep_losses = []

        while True:
            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store in replay buffer
            agent.store_transition(state, action, reward, next_state, done)

            # Train on a batch
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if ep_losses:
            losses.append(np.mean(ep_losses))

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(episode_rewards[-50:])
            avg_l = np.mean(episode_lengths[-50:])
            avg_loss = np.mean(losses[-50:]) if losses else 0
            print(f"Episode {ep+1:4d} | "
                  f"avg reward: {avg_r:7.1f} | "
                  f"avg length: {avg_l:5.1f} | "
                  f"epsilon: {agent.epsilon:.3f} | "
                  f"loss: {avg_loss:.4f}")

    env.close()
    return agent, episode_rewards, episode_lengths, losses


def evaluate_dqn(agent, env_name="CartPole-v1", n_episodes=20):
    """Evaluate trained agent with no exploration."""
    env = gym.make(env_name)
    old_eps = agent.epsilon
    agent.epsilon = 0.0

    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total = 0
        while True:
            action = agent.select_action(state)
            state, reward, term, trunc, _ = env.step(action)
            total += reward
            if term or trunc:
                break
        rewards.append(total)

    env.close()
    agent.epsilon = old_eps
    return rewards


def plot_dqn_results(rewards, losses, env_name,
                     save_path="notebooks/dqn_training.png"):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Rewards
    window = 30
    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
    ax1.plot(rewards, alpha=0.3, color="#3498db")
    ax1.plot(range(window-1, len(rewards)), smoothed, linewidth=2, color="#3498db")
    ax1.set_title(f"DQN on {env_name} — Episode Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)

    # Loss
    if losses:
        smoothed_loss = np.convolve(losses, np.ones(window)/window, mode="valid")
        ax2.plot(losses, alpha=0.3, color="#e74c3c")
        ax2.plot(range(window-1, len(losses)), smoothed_loss,
                 linewidth=2, color="#e74c3c")
    ax2.set_title("Training Loss")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("MSE Loss")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Deep Q-Network Training", fontsize=14)
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
    agent, rewards, lengths, losses = train_dqn(
        env_name=ENV,
        n_episodes=800,
        device="cpu",
    )

    # Evaluate
    print(f"\n{'='*50}")
    print("EVALUATION (20 episodes, no exploration)")
    print(f"{'='*50}")
    eval_rewards = evaluate_dqn(agent, env_name=ENV)
    print(f"Avg reward: {np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}")
    print(f"Max reward: {max(eval_rewards):.0f}")
    print(f"Min reward: {min(eval_rewards):.0f}")

    # Plot
    plot_dqn_results(rewards, losses, ENV)

    # Save checkpoint
    torch.save({
        "q_net_state": agent.q_net.state_dict(),
        "target_net_state": agent.target_net.state_dict(),
        "epsilon": agent.epsilon,
    }, "checkpoints/dqn_cartpole.pt")
    print("Saved checkpoint to checkpoints/dqn_cartpole.pt")
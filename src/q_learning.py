"""
Q-Learning — Tabular Reinforcement Learning.

The Q-table stores a value for every (state, action) pair:
  Q(s, a) = "expected total future reward if I take action a in state s
             and then act optimally afterward"

The Bellman update rule:
  Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') - Q(s, a) ]

Where:
  α = learning rate (how fast we update — 0.1 typical)
  γ = discount factor (how much we value future vs immediate reward — 0.99 typical)
  r = reward received after taking action a in state s
  s' = next state we ended up in
  max_a' Q(s', a') = value of the best action in the next state

The term in brackets is the "temporal difference error" (TD error):
  δ = r + γ · max_a' Q(s', a') - Q(s, a)

This is the difference between:
  - What we EXPECTED to get: Q(s, a)
  - What we ACTUALLY got: r + γ · max_a' Q(s', a')

If δ > 0: we got more than expected → increase Q(s, a)
If δ < 0: we got less than expected → decrease Q(s, a)
"""

import numpy as np
from collections import defaultdict
from src.gridworld import GridWorldEnv, visualize_q_values, visualize_training


class QLearningAgent:
    """
    Tabular Q-Learning with epsilon-greedy exploration.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr                  # α — learning rate
        self.gamma = gamma            # γ — discount factor
        self.epsilon = epsilon        # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: maps (state, action) → value
        # defaultdict(float) returns 0.0 for unseen (state, action) pairs
        self.q_table = defaultdict(float)

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)

        # Pick action with highest Q-value for this state
        q_values = [self.q_table[(state, a)] for a in range(self.n_actions)]
        max_q = max(q_values)
        # Break ties randomly
        best_actions = [a for a in range(self.n_actions) if q_values[a] == max_q]
        return np.random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """
        The Bellman update — the core of Q-learning.

        Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') - Q(s, a) ]
        """
        # Current estimate
        current_q = self.q_table[(state, action)]

        # Target: what we actually observed
        if done:
            # Terminal state — no future rewards
            target = reward
        else:
            # r + γ · max_a' Q(s', a')
            next_q_values = [self.q_table[(next_state, a)] for a in range(self.n_actions)]
            target = reward + self.gamma * max(next_q_values)

        # TD error
        td_error = target - current_q

        # Update
        self.q_table[(state, action)] = current_q + self.lr * td_error

        return td_error

    def decay_epsilon(self):
        """Reduce exploration rate over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(env, agent, n_episodes=2000, verbose=True):
    """Train the Q-learning agent."""
    episode_rewards = []
    episode_lengths = []
    td_errors = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        ep_td_errors = []

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            td_error = agent.update(state, action, reward, next_state, done)
            ep_td_errors.append(abs(td_error))

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        td_errors.append(np.mean(ep_td_errors))

        if verbose and (ep + 1) % 200 == 0:
            avg_r = np.mean(episode_rewards[-200:])
            avg_l = np.mean(episode_lengths[-200:])
            print(f"Episode {ep+1:5d} | "
                  f"avg reward: {avg_r:7.2f} | "
                  f"avg length: {avg_l:5.1f} | "
                  f"epsilon: {agent.epsilon:.3f} | "
                  f"avg |TD|: {np.mean(td_errors[-200:]):.4f}")

    return episode_rewards, episode_lengths


def evaluate(env, agent, n_episodes=100):
    """Evaluate the trained agent (no exploration)."""
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # pure exploitation

    rewards = []
    lengths = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state
            if terminated or truncated:
                break
        rewards.append(total_reward)
        lengths.append(steps)

    agent.epsilon = old_epsilon
    return rewards, lengths


def show_episode(env, agent):
    """Show one episode step by step."""
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    state, _ = env.reset()
    print("Start:")
    print(env.render_text())
    print()

    total_reward = 0
    step = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step += 1
        state = next_state

        if terminated:
            print(f"Step {step}: {env.action_names[action]} → GOAL! "
                  f"Reward: {reward:.1f}")
            print(env.render_text())
            break
        elif truncated:
            print(f"Step {step}: Truncated (max steps)")
            break

    print(f"\nTotal reward: {total_reward:.1f}, Steps: {step}")
    agent.epsilon = old_epsilon


if __name__ == "__main__":
    # Create environment and agent
    env = GridWorldEnv()
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        lr=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    )

    print(f"GridWorld: {env.rows}×{env.cols} = {env.observation_space.n} states")
    print(f"Actions: {env.action_space.n}")
    print(f"Training for 2000 episodes...\n")

    # Train
    rewards, lengths = train(env, agent, n_episodes=2000)

    # Evaluate
    print(f"\n{'='*50}")
    print("EVALUATION (100 episodes, no exploration)")
    print(f"{'='*50}")
    eval_rewards, eval_lengths = evaluate(env, agent, n_episodes=100)
    print(f"Avg reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Avg length: {np.mean(eval_lengths):.1f} ± {np.std(eval_lengths):.1f}")
    success_rate = sum(1 for r in eval_rewards if r > 5) / len(eval_rewards)
    print(f"Success rate: {success_rate:.0%}")

    # Show one episode
    print(f"\n{'='*50}")
    print("SAMPLE EPISODE")
    print(f"{'='*50}")
    show_episode(env, agent)

    # Visualize
    visualize_q_values(env, agent.q_table)
    visualize_training(rewards, lengths)

    # Print Q-table stats
    print(f"\nQ-table entries: {len(agent.q_table)}")
    print(f"Non-zero entries: {sum(1 for v in agent.q_table.values() if v != 0)}")
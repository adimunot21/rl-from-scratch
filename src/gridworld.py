"""
GridWorld Environment — Following the Gymnasium API.

An 8×8 grid where an agent navigates from start to goal,
avoiding walls and penalty tiles.

Grid legend:
  S = start
  G = goal (+10 reward)
  . = empty (-0.1 reward per step — incentivizes short paths)
  # = wall (can't enter)
  X = penalty (-5 reward, then episode continues)

Actions: 0=up, 1=right, 2=down, 3=left
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Default 8×8 grid layout
DEFAULT_GRID = [
    "S . . . . . . .",
    ". . # . . X . .",
    ". . # . . . . .",
    ". . . . # # . .",
    ". X . . # . . .",
    ". . . . . . X .",
    ". . # . . . . .",
    ". . # . . . . G",
]


class GridWorldEnv(gym.Env):
    """
    Custom GridWorld following the Gymnasium API.

    Implementing the Gymnasium API means our environment works with
    any RL algorithm that expects a Gymnasium environment — including
    the ones we'll build in later phases.
    """

    metadata = {"render_modes": ["text", "rgb_array"]}

    def __init__(self, grid=None, max_steps=200):
        super().__init__()

        self.grid = self._parse_grid(grid or DEFAULT_GRID)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.max_steps = max_steps

        # Find start and goal positions
        self.start_pos = None
        self.goal_pos = None
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == "S":
                    self.start_pos = (r, c)
                elif self.grid[r][c] == "G":
                    self.goal_pos = (r, c)

        assert self.start_pos is not None, "Grid must have a start (S)"
        assert self.goal_pos is not None, "Grid must have a goal (G)"

        # Gymnasium API: define action and observation spaces
        self.action_space = spaces.Discrete(4)  # up, right, down, left
        self.observation_space = spaces.Discrete(self.rows * self.cols)

        # Action deltas: (row_change, col_change)
        self.action_deltas = {
            0: (-1, 0),   # up
            1: (0, 1),    # right
            2: (1, 0),    # down
            3: (0, -1),   # left
        }
        self.action_names = {0: "↑", 1: "→", 2: "↓", 3: "←"}

        # State tracking
        self.agent_pos = None
        self.steps = 0

    def _parse_grid(self, grid_lines):
        """Parse grid from list of strings."""
        return [line.split() for line in grid_lines]

    def _pos_to_state(self, pos):
        """Convert (row, col) to a single integer state."""
        return pos[0] * self.cols + pos[1]

    def _state_to_pos(self, state):
        """Convert integer state back to (row, col)."""
        return (state // self.cols, state % self.cols)

    def reset(self, seed=None, options=None):
        """Reset the environment. Returns (observation, info)."""
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.steps = 0
        return self._pos_to_state(self.agent_pos), {}

    def step(self, action):
        """
        Take an action. Returns (observation, reward, terminated, truncated, info).

        terminated = reached goal or penalty-ended
        truncated = hit max steps
        """
        self.steps += 1
        dr, dc = self.action_deltas[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        # Check bounds and walls
        if (0 <= new_r < self.rows and
            0 <= new_c < self.cols and
            self.grid[new_r][new_c] != "#"):
            self.agent_pos = (new_r, new_c)

        # Determine reward and termination
        cell = self.grid[self.agent_pos[0]][self.agent_pos[1]]

        if cell == "G":
            return self._pos_to_state(self.agent_pos), 10.0, True, False, {}
        elif cell == "X":
            return self._pos_to_state(self.agent_pos), -5.0, False, False, {}
        else:
            # Small negative reward per step to incentivize short paths
            reward = -0.1
            truncated = self.steps >= self.max_steps
            return self._pos_to_state(self.agent_pos), reward, False, truncated, {}

    def render_text(self):
        """Render the grid as text."""
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) == self.agent_pos:
                    row.append("A")
                else:
                    row.append(self.grid[r][c])
            lines.append(" ".join(row))
        return "\n".join(lines)


def visualize_q_values(env, q_table, save_path="notebooks/q_values.png"):
    """
    Visualize the Q-values as a heatmap with policy arrows.

    For each cell:
    - Color = max Q-value (how good is this state?)
    - Arrow = best action (which direction to go?)
    """
    rows, cols = env.rows, env.cols
    max_q = np.full((rows, cols), np.nan)
    best_actions = np.full((rows, cols), -1, dtype=int)

    for r in range(rows):
        for c in range(cols):
            state = env._pos_to_state((r, c))
            cell = env.grid[r][c]
            if cell == "#":
                continue
            q_vals = [q_table.get((state, a), 0.0) for a in range(4)]
            max_q[r, c] = max(q_vals)
            best_actions[r, c] = int(np.argmax(q_vals))

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw heatmap
    masked = np.ma.masked_invalid(max_q)
    im = ax.imshow(masked, cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Max Q-value")

    # Draw grid details and arrows
    arrow_dx = {0: 0, 1: 0.3, 2: 0, 3: -0.3}
    arrow_dy = {0: -0.3, 1: 0, 2: 0.3, 3: 0}

    for r in range(rows):
        for c in range(cols):
            cell = env.grid[r][c]

            if cell == "#":
                ax.add_patch(patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, fill=True,
                    facecolor="black", edgecolor="black"
                ))
            elif cell == "G":
                ax.text(c, r, "G", ha="center", va="center",
                        fontsize=16, fontweight="bold", color="white")
            elif cell == "S":
                ax.text(c, r, "S", ha="center", va="center",
                        fontsize=16, fontweight="bold", color="blue")
            elif cell == "X":
                ax.text(c, r, "X", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="red")

            # Draw policy arrow for non-special cells
            if cell not in ("#", "G") and best_actions[r, c] >= 0:
                a = best_actions[r, c]
                ax.arrow(c, r, arrow_dx[a] * 0.6, arrow_dy[a] * 0.6,
                         head_width=0.15, head_length=0.08,
                         fc="black", ec="black", alpha=0.7)

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_title("Q-Values & Learned Policy", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved Q-value visualization to {save_path}")
    plt.close()


def visualize_training(episode_rewards, episode_lengths,
                       save_path="notebooks/q_learning_training.png"):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Smooth with rolling average
    window = 50
    def smooth(data):
        return np.convolve(data, np.ones(window)/window, mode="valid")

    ax1.plot(smooth(episode_rewards), linewidth=2, color="#3498db")
    ax1.set_title("Episode Reward (smoothed)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.grid(True, alpha=0.3)

    ax2.plot(smooth(episode_lengths), linewidth=2, color="#e74c3c")
    ax2.set_title("Episode Length (smoothed)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Q-Learning Training Progress", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved training curves to {save_path}")
    plt.close()


if __name__ == "__main__":
    # Quick test
    env = GridWorldEnv()
    obs, _ = env.reset()
    print(f"Grid size: {env.rows}×{env.cols}")
    print(f"Start: {env.start_pos} (state {obs})")
    print(f"Goal: {env.goal_pos}")
    print(f"States: {env.observation_space.n}")
    print(f"Actions: {env.action_space.n}")
    print(f"\nInitial grid:")
    print(env.render_text())

    # Take a few random steps
    print(f"\n--- Random walk ---")
    total_reward = 0
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        pos = env._state_to_pos(obs)
        print(f"  Action: {env.action_names[action]}, "
              f"New pos: {pos}, Reward: {reward:.1f}, Done: {term}")

    print(f"\nGrid after random walk:")
    print(env.render_text())
    print("\nEnvironment test passed!")
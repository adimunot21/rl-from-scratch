"""
Shared plotting utilities and algorithm comparison.

Run: python -m src.utils
Generates a full comparison dashboard of all RL algorithms.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(data, window=30):
    """Rolling average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode="valid")


def plot_algorithm_comparison(all_results, env_name,
                              save_path="notebooks/algorithm_comparison.png"):
    """
    Compare all RL algorithms on the same plot.

    all_results: dict of {name: episode_rewards_list}
    """
    colors = {
        "DQN": "#e74c3c",
        "REINFORCE": "#f39c12",
        "A2C": "#2ecc71",
        "PPO": "#3498db",
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ---- Plot 1: Training curves ----
    ax = axes[0]
    window = 30
    for name, rewards in all_results.items():
        color = colors.get(name, "#333333")
        ax.plot(rewards, alpha=0.1, color=color)
        sm = smooth(rewards, window)
        ax.plot(range(window-1, len(rewards)), sm,
                linewidth=2.5, color=color, label=name)
    ax.set_title(f"Training Curves — {env_name}", fontsize=13)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: Final performance box plots ----
    ax = axes[1]
    last_n = 100
    data = []
    labels = []
    box_colors = []
    for name, rewards in all_results.items():
        tail = rewards[-last_n:] if len(rewards) >= last_n else rewards
        data.append(tail)
        labels.append(name)
        box_colors.append(colors.get(name, "#333333"))

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(f"Final Performance (last {last_n} episodes)", fontsize=13)
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3, axis="y")

    # ---- Plot 3: Summary stats table ----
    ax = axes[2]
    ax.axis("off")

    headers = ["Algorithm", "Mean", "Std", "Max", "Min", "Episodes"]
    table_data = []
    for name, rewards in all_results.items():
        tail = rewards[-last_n:] if len(rewards) >= last_n else rewards
        table_data.append([
            name,
            f"{np.mean(tail):.1f}",
            f"{np.std(tail):.1f}",
            f"{np.max(tail):.0f}",
            f"{np.min(tail):.0f}",
            f"{len(rewards)}",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Color the algorithm name cells
    for i, name in enumerate(all_results.keys()):
        table[i+1, 0].set_facecolor(colors.get(name, "#333333"))
        table[i+1, 0].set_text_props(color="white", fontweight="bold")

    ax.set_title(f"Summary Statistics (last {last_n} episodes)", fontsize=13,
                 pad=20)

    plt.suptitle("RL Algorithm Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison to {save_path}")
    plt.close()


if __name__ == "__main__":
    import torch
    import gymnasium as gym

    ENV = "CartPole-v1"
    all_results = {}

    # ---- DQN ----
    print("=" * 60)
    print("Training DQN...")
    print("=" * 60)
    from src.dqn import train_dqn
    _, dqn_rewards, _, _ = train_dqn(env_name=ENV, n_episodes=800, device="cpu")
    all_results["DQN"] = dqn_rewards

    # ---- REINFORCE ----
    print("\n" + "=" * 60)
    print("Training REINFORCE...")
    print("=" * 60)
    from src.reinforce import train_reinforce
    _, reinforce_rewards, _ = train_reinforce(env_name=ENV, n_episodes=800, use_baseline=True)
    all_results["REINFORCE"] = reinforce_rewards

    # ---- A2C ----
    print("\n" + "=" * 60)
    print("Training A2C...")
    print("=" * 60)
    from src.actor_critic import train_a2c
    _, a2c_rewards, _, _ = train_a2c(env_name=ENV, n_episodes=800)
    all_results["A2C"] = a2c_rewards

    # ---- PPO ----
    print("\n" + "=" * 60)
    print("Training PPO...")
    print("=" * 60)
    from src.ppo import train_ppo
    _, ppo_rewards, _ = train_ppo(env_name=ENV, total_timesteps=100000)
    # PPO counts timesteps not episodes, so we just use what it produces
    all_results["PPO"] = ppo_rewards

    # ---- Plot ----
    print("\n" + "=" * 60)
    print("Generating comparison...")
    print("=" * 60)
    plot_algorithm_comparison(all_results, ENV)

    print("\nDone! Check notebooks/algorithm_comparison.png")
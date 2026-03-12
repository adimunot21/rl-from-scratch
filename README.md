# Reinforcement Learning From Scratch

A complete implementation of every major RL algorithm in PyTorch — from multi-armed bandits to PPO. No `stable-baselines3`, no `rllib`, no RL libraries. Every algorithm, every update rule, every training loop written by hand.

Progresses through the entire RL algorithm family tree, where each algorithm exists because the previous one had a specific limitation.

## The Algorithm Family Tree

```
Multi-Armed Bandit
  "How do I choose between options with unknown rewards?"
       │
       ▼
Q-Learning (Tabular)
  "What if my choices have long-term consequences across states?"
       │
       ▼
Deep Q-Network (DQN)
  "What if there are too many states to store in a table?"
       │
       ├──────────────────────────────┐
       ▼                              ▼
  REINFORCE                      (value-based path)
  "What if I directly optimize         │
   the policy instead of               │
   estimating values?"                  │
       │                                │
       ▼                                │
  Actor-Critic (A2C)  ◄────────────────┘
  "Can I combine both approaches
   to get lower variance?"
       │
       ▼
  PPO (Proximal Policy Optimization)
  "Can I make Actor-Critic stable
   and reliable enough for production?"
```

## What's Inside

### Phase 1: Multi-Armed Bandits
Four strategies compared on a 10-armed bandit: Random, Epsilon-Greedy, Upper Confidence Bound (UCB), and Thompson Sampling. Introduces the exploration vs exploitation tradeoff — the foundational dilemma of all RL.

### Phase 2: GridWorld + Q-Learning
Custom 8×8 GridWorld environment (following the Gymnasium API) with walls, penalties, and a goal. Tabular Q-Learning agent learns the optimal path using the Bellman equation. **100% success rate, optimal 14-step path.**

### Phase 3: Deep Q-Network (DQN)
Replaces the Q-table with a neural network for continuous state spaces. Implements all three innovations from the DeepMind 2015 paper: experience replay, target network, and epsilon-greedy decay. **Solves CartPole with perfect 500/500 score.**

### Phase 4: REINFORCE (Policy Gradient)
A fundamentally different approach — directly optimizing the policy instead of estimating values. Implements the policy gradient theorem with and without baseline subtraction. **Demonstrates the high-variance problem that motivates Actor-Critic.**

### Phase 5: Advantage Actor-Critic (A2C)
Combines value-based and policy-based methods. The actor chooses actions, the critic evaluates them. Separate networks with separate learning rates. **Demonstrates the instability problem that motivates PPO.**

### Phase 6: PPO (Proximal Policy Optimization)
The industry-standard algorithm. Clipped surrogate objective prevents catastrophic policy updates. Generalized Advantage Estimation (GAE) for smooth advantage computation. Multiple epochs per batch for sample efficiency. **The most stable training of all algorithms.**

### Phase 7: Algorithm Comparison
All four deep RL algorithms trained on CartPole and compared side-by-side: training curves, final performance distributions, and summary statistics.

## Results

### Multi-Armed Bandits (10 arms, 2000 pulls)
| Strategy | Avg Reward | Regret |
|----------|-----------|--------|
| Random | 0.532 | 656.1 |
| ε-Greedy | 0.831 | 59.1 |
| UCB | 0.719 | 284.1 |
| Thompson Sampling | 0.821 | 80.1 |
| *Optimal* | *0.861* | *0* |

### Q-Learning on GridWorld
| Metric | Value |
|--------|-------|
| Success rate | 100% |
| Optimal path length | 14 steps |
| Reward | 8.70 / 10.0 |
| Q-table entries | 224 (56 states × 4 actions) |

### Deep RL on CartPole-v1
| Algorithm | Mean Reward | Std | Max | Stability |
|-----------|-----------|-----|-----|-----------|
| DQN | 500.0 | ± 0.0 | 500 | Solved perfectly |
| REINFORCE | 500.0 | ± 0.0 | 500 | Wild training, stable eval |
| A2C | 241.9 | ± 88.7 | 392 | Oscillating |
| PPO | 224.3 | ± 4.9 | 237 | Most consistent (18× less variance than A2C) |

## Project Structure

```
rl-from-scratch/
├── src/
│   ├── bandits.py          ← Multi-armed bandits (4 strategies)
│   ├── gridworld.py        ← Custom GridWorld environment
│   ├── q_learning.py       ← Tabular Q-Learning agent
│   ├── dqn.py              ← Deep Q-Network (replay buffer, target net)
│   ├── reinforce.py        ← REINFORCE with baseline
│   ├── actor_critic.py     ← Advantage Actor-Critic (A2C)
│   ├── ppo.py              ← Proximal Policy Optimization
│   └── utils.py            ← Comparison dashboard
├── notebooks/
│   ├── bandits_comparison.png
│   ├── q_values.png
│   ├── q_learning_training.png
│   ├── dqn_training.png
│   ├── reinforce_comparison.png
│   ├── a2c_training.png
│   ├── ppo_training.png
│   └── algorithm_comparison.png
├── course/                 ← Detailed written course (8 chapters)
├── checkpoints/            ← Saved model weights (local only)
└── requirements.txt
```

## Setup

```bash
conda create -n rl python=3.11 -y
conda activate rl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium "numpy<2" matplotlib tqdm
```

## Usage

```bash
# Phase 1: Multi-armed bandits
python -m src.bandits

# Phase 2: GridWorld + Q-Learning
python -m src.gridworld
python -m src.q_learning

# Phase 3: Deep Q-Network
python -m src.dqn

# Phase 4: REINFORCE
python -m src.reinforce

# Phase 5: Actor-Critic
python -m src.actor_critic

# Phase 6: PPO
python -m src.ppo

# Phase 7: Full comparison
python -m src.utils
```

## Key Takeaways

1. **Exploration vs exploitation** is the fundamental RL dilemma. Every algorithm addresses it differently — epsilon-greedy, UCB, entropy bonuses, or Bayesian sampling.

2. **Value-based methods (DQN)** are powerful but limited to discrete actions. They estimate "how good" each action is and pick the best one.

3. **Policy-based methods (REINFORCE)** directly optimize the policy and handle continuous actions, but suffer from high variance.

4. **Actor-Critic (A2C)** combines both approaches but introduces instability — competing gradients between actor and critic can cause collapse.

5. **PPO** solves the stability problem with the clipped surrogate objective. It's the industry standard because it's stable, sample-efficient, and general-purpose.

6. **There is no single best algorithm.** Each has tradeoffs. Knowing when to use which one is the real engineering skill.

## Course: Learn RL From Scratch

This project includes a detailed written course explaining every concept, every line of code, and every design decision. Written for someone who knows Python but is new to RL.

| Chapter | Topic | What You'll Learn |
|---------|-------|-------------------|
| [0: Introduction](course/00_introduction.md) | What is RL? | Agent-environment loop, MDPs, the RL landscape |
| [1: Multi-Armed Bandits](course/01_bandits.md) | Exploration vs exploitation | Epsilon-greedy, UCB, Thompson Sampling, regret |
| [2: GridWorld + Q-Learning](course/02_q_learning.md) | Tabular RL | Bellman equation, temporal difference, Q-tables |
| [3: Deep Q-Network](course/03_dqn.md) | Function approximation | Replay buffers, target networks, the deadly triad |
| [4: REINFORCE](course/04_reinforce.md) | Policy gradients | Policy gradient theorem, log-prob trick, baselines |
| [5: Actor-Critic](course/05_actor_critic.md) | Combining approaches | Advantage function, variance reduction, two-network design |
| [6: PPO](course/06_ppo.md) | The industry standard | Clipped objective, GAE, trust regions, multiple epochs |
| [7: Comparison](course/07_comparison.md) | Putting it all together | When to use which algorithm, tradeoffs, next steps |

## Built With
- PyTorch (neural networks and autograd)
- Gymnasium (CartPole environment)
- Custom GridWorld (following Gymnasium API)
- No RL libraries — everything from scratch
# Chapter 0: Introduction — What Is Reinforcement Learning?

## Three Kinds of Machine Learning

Machine learning has three main paradigms. Understanding where RL fits helps you see why it exists.

**Supervised Learning**: You have labeled data. Input → correct output. "Here are 10,000 photos of cats and dogs with labels — learn to classify new photos." The model learns from examples where the right answer is known.

**Unsupervised Learning**: You have unlabeled data. "Here are 10,000 photos — find patterns, clusters, structure." No right answers, just data.

**Reinforcement Learning**: You have an agent in an environment. No labeled data. No dataset of correct actions. Instead, the agent takes actions, receives rewards, and must figure out what works through trial and error. "Here's a game — play it a million times and figure out how to win."

The key difference: in supervised learning, someone tells you the right answer. In RL, you discover it by trying things and observing the consequences.

## The Agent-Environment Loop

Every RL problem has the same structure:

```
┌─────────┐    action aₜ    ┌─────────────┐
│         │────────────────▶│             │
│  Agent  │                 │ Environment │
│         │◀────────────────│             │
└─────────┘  state sₜ₊₁    └─────────────┘
              reward rₜ
```

At each time step t:

1. The agent observes the current **state** sₜ (what the world looks like right now)
2. The agent chooses an **action** aₜ (what to do)
3. The environment transitions to a new **state** sₜ₊₁ and gives a **reward** rₜ
4. Repeat

The agent's goal: choose actions that maximize the **total cumulative reward** over time. Not just immediate reward — total reward. This is what makes RL hard: an action that gives high immediate reward might lead to terrible future states.

### A Concrete Example: CartPole

A pole is balanced on a cart. The cart can move left or right.

```
        ╱
       ╱    ← pole (keep it upright!)
      ╱
   ┌──┴──┐
   │ cart │
   └──┬──┘
───────────────── track ─────────────────
```

**State** (what the agent sees): 4 numbers
- Cart position on the track
- Cart velocity (how fast it's moving)
- Pole angle (how far from vertical)
- Pole angular velocity (how fast it's falling)

**Actions** (what the agent can do): 2 choices
- Push the cart left
- Push the cart right

**Reward**: +1 for every time step the pole stays upright

**Episode ends**: when the pole falls past 15° or the cart goes off the edge

A random agent (choosing left/right randomly) scores about 20 — the pole falls almost immediately. A trained agent scores 500 (the maximum). The gap between 20 and 500 is what the agent must learn to bridge through experience.

## Key Concepts

### Policy

A **policy** π is the agent's strategy — a mapping from states to actions. It answers: "Given that I'm in state s, what action should I take?"

A policy can be:
- **Deterministic**: π(s) = a — always the same action for the same state
- **Stochastic**: π(a|s) = probability of taking action a in state s — the agent might take different actions in the same state, with certain probabilities

Stochastic policies are important because they naturally handle exploration (sometimes trying unusual actions) and can represent uncertainty.

### Value

The **value** of a state V(s) is the expected total future reward starting from that state, assuming the agent follows its current policy:

```
V(s) = E[rₜ + rₜ₊₁ + rₜ₊₂ + ... | sₜ = s]
```

States that lead to high future reward have high value. States near a cliff edge have low value. The value function tells the agent "how good is it to be here?"

### Q-Value

The **Q-value** Q(s, a) adds the action dimension: "how good is it to take action a in state s, then follow the current policy?"

```
Q(s, a) = E[rₜ + rₜ₊₁ + rₜ₊₂ + ... | sₜ = s, aₜ = a]
```

If you know Q(s, a) for all actions, the optimal policy is simple: always pick the action with the highest Q-value.

### Discount Factor (γ)

A dollar today is worth more than a dollar next year. Similarly, immediate reward is often more valuable than distant future reward. The **discount factor** γ (gamma, typically 0.99) encodes this:

```
Total return = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + γ³·rₜ₊₃ + ...
```

With γ = 0.99:
- Reward right now: full value (×1.0)
- Reward in 10 steps: ×0.99¹⁰ = ×0.90 (90% of face value)
- Reward in 100 steps: ×0.99¹⁰⁰ = ×0.37 (37% of face value)
- Reward in 500 steps: ×0.99⁵⁰⁰ = ×0.007 (almost worthless)

Why discount at all? Three reasons:
1. **Mathematical**: without discounting, infinite-horizon returns can be infinite, which breaks the math
2. **Practical**: distant rewards are uncertain — the world might change
3. **Behavioral**: it incentivizes the agent to achieve goals sooner rather than later

### Episodes

An **episode** is one complete run of the agent in the environment — from start to termination. In CartPole, an episode starts with the pole upright and ends when it falls. In chess, an episode is one complete game.

Some environments have natural episodes (games, tasks with goals). Others are **continuing** (a thermostat controlling temperature forever). We focus on episodic environments in this course.

### Exploration vs. Exploitation

This is THE central dilemma of RL:

**Exploitation**: Use what you already know. Pick the action you believe gives the highest reward. This is good when your knowledge is accurate.

**Exploration**: Try something new. Pick an action you're uncertain about to learn more. This is good when your knowledge might be wrong or incomplete.

Too much exploitation → you never discover better strategies. You might eat at the same restaurant every night and never find the amazing place around the corner.

Too much exploration → you never use what you've learned. You eat at a random restaurant every night and waste most evenings on bad food.

Every RL algorithm balances this tradeoff differently. This is the first thing we implement (Chapter 1: Bandits).

## The RL Algorithm Landscape

There are two fundamentally different approaches to RL:

### Value-Based Methods

"Estimate how good each action is, then pick the best one."

1. Learn Q(s, a) — the value of each action in each state
2. Act greedily: pick a = argmax Q(s, a)

Examples: Q-Learning, DQN

Strengths: stable, well-understood, sample-efficient
Weaknesses: can only handle discrete actions (you need to enumerate all actions to find the max)

### Policy-Based Methods

"Directly learn a mapping from states to actions."

1. Parameterize a policy π(a|s) (often with a neural network)
2. Evaluate how well the policy performs
3. Adjust the parameters to improve performance

Examples: REINFORCE, Policy Gradient

Strengths: handles continuous actions, can learn stochastic policies
Weaknesses: high variance, sample-inefficient

### Actor-Critic Methods

"Combine both approaches."

1. **Actor**: a policy network that chooses actions (policy-based)
2. **Critic**: a value network that evaluates actions (value-based)
3. The critic helps the actor learn faster by providing step-by-step feedback

Examples: A2C, PPO, SAC

This is where most modern RL lives. PPO (which we'll build) is the industry standard.

## The Markov Decision Process (MDP)

The mathematical framework underlying all of RL is the **Markov Decision Process**. An MDP is defined by:

- **S**: set of states
- **A**: set of actions
- **P(s'|s, a)**: transition probabilities — given state s and action a, what's the probability of ending up in state s'?
- **R(s, a)**: reward function — what reward do we get for taking action a in state s?
- **γ**: discount factor

The **Markov property** is the key assumption: the future depends only on the current state, not on the history of how we got here. Given the current state of a chess board, it doesn't matter what moves led to it — the best next move is the same.

```
P(sₜ₊₁ | sₜ, aₜ, sₜ₋₁, aₜ₋₁, ...) = P(sₜ₊₁ | sₜ, aₜ)
```

This assumption makes RL tractable. Without it, the agent would need to remember its entire history.

## What We'll Build

Over the next 7 chapters:

| Phase | Algorithm | Key Concept |
|-------|-----------|-------------|
| 1 | Multi-Armed Bandits | Exploration vs exploitation |
| 2 | Q-Learning on GridWorld | Bellman equation, tabular RL |
| 3 | Deep Q-Network (DQN) | Function approximation, replay buffers |
| 4 | REINFORCE | Policy gradients, the log-prob trick |
| 5 | Actor-Critic (A2C) | Advantage function, variance reduction |
| 6 | PPO | Clipped objective, trust regions, GAE |
| 7 | Comparison | When to use which algorithm |

Each phase builds on the previous one. Each algorithm exists because the previous one had a specific limitation. By the end, you'll understand not just how each algorithm works, but why it was invented.

## Environment Setup

### Step 1: Create a Conda Environment

```bash
conda create -n rl python=3.11 -y
conda activate rl
```

### Step 2: Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium "numpy<2" matplotlib tqdm
```

- **PyTorch**: neural network framework (used from Phase 3 onward)
- **Gymnasium**: provides standard RL environments (CartPole, LunarLander)
- **NumPy**: numerical computing
- **Matplotlib**: plotting

### Step 3: Create the Project Structure

```bash
mkdir -p ~/projects/rl-from-scratch/{src,checkpoints,notebooks,course}
cd ~/projects/rl-from-scratch
touch src/__init__.py src/bandits.py src/gridworld.py src/q_learning.py
touch src/dqn.py src/reinforce.py src/actor_critic.py src/ppo.py src/utils.py
```

### Step 4: Verify

```bash
python -c "
import torch
import gymnasium as gym
import numpy as np
env = gym.make('CartPole-v1')
obs, _ = env.reset()
print(f'CartPole state: {obs}')
print(f'Action space: {env.action_space}')
env.close()
print('All good!')
"
```

## What's Next

In [Chapter 1](01_bandits.md), we start with the simplest possible RL problem: the multi-armed bandit. No states, no transitions, no episodes — just repeated choices with uncertain rewards. This is where we learn exploration vs exploitation, the concept that underlies everything in RL.
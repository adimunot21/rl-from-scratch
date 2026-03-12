# Chapter 7: Putting It All Together — When to Use What

## The Full Picture

Over the past six chapters, we built every major RL algorithm from scratch:

```
Chapter 1: Multi-Armed Bandits       → exploration vs exploitation
Chapter 2: Q-Learning (Tabular)      → states, Bellman equation, TD learning
Chapter 3: DQN                       → function approximation, replay, target networks
Chapter 4: REINFORCE                 → policy gradients, the log-prob trick
Chapter 5: A2C                       → actor-critic, advantage function
Chapter 6: PPO                       → clipped objective, GAE, multiple epochs
```

Each algorithm was the answer to a specific problem in the previous one. Now we step back and compare them all — not just on performance numbers, but on fundamental properties that determine which algorithm you should choose for a given problem.

## Part 1: Our Results on CartPole

### Training Curves

In our comparison run, all four deep RL algorithms trained on CartPole-v1:

```
DQN:        800 episodes  → peaked at ~274 avg, final ~182 avg
REINFORCE:  800 episodes  → peaked at ~490 avg, final ~490 avg
A2C:        800 episodes  → peaked at ~167 avg, final ~167 avg
PPO:        100K steps    → peaked at ~228 avg, final ~146 avg
```

On dedicated runs with more training:

```
DQN:        800 episodes  → 500.0 ± 0.0  (perfect)
REINFORCE:  1000 episodes → 500.0 ± 0.0  (perfect, but chaotic training)
A2C:        1000 episodes → 241.9 ± 88.7 (learning but unstable)
PPO:        200K steps    → 224.3 ± 4.9  (stable but still climbing)
```

### What the Numbers Reveal

CartPole is a **simple** problem — 4 continuous state dimensions, 2 discrete actions, short episodes. This simplicity actually reveals the tradeoffs more clearly than a hard problem would:

**DQN excels on simple discrete problems.** The replay buffer and target network give it rock-solid stability. With enough episodes, it reaches perfect performance. But it needed careful tuning — our first attempt (wrong loss function, wrong target update frequency) scored 19.7.

**REINFORCE reaches peak performance but can't maintain it.** The training curve swings from 500 to 200 and back. In evaluation mode (greedy action selection, no sampling randomness), it scores perfectly — the learned policy is good. But during training, the stochastic sampling and high-variance gradients keep destabilizing it.

**A2C improves over REINFORCE in theory but struggled in practice.** We spent three iterations debugging it — shared networks failed, advantage normalization killed gradients, wrong learning rate ratios prevented learning. When it finally worked, it showed clear improvement over REINFORCE's variance, but still oscillated.

**PPO trades peak performance for consistency.** It didn't reach 500, but its ± 4.9 standard deviation is extraordinary. In 20 evaluation episodes, every single one scored between 217 and 237. No algorithm was this predictable.

---

## Part 2: Properties That Matter

### Sample Efficiency

How many environment interactions does the algorithm need to learn?

```
Most efficient → Least efficient:

DQN            ~400K steps to solve (800 episodes × ~500 steps)
  └─ Replay buffer: each transition used in ~many batches over its lifetime
  
PPO            ~170K steps to reach 370 avg
  └─ Each batch used for 4 epochs (4× reuse)
  
REINFORCE      ~500K steps to reach 490 avg (1000 episodes × ~500 steps)
  └─ Each episode used once, then discarded
  
A2C            ~400K steps to reach 240 avg (1000 episodes × ~400 steps)
  └─ Each episode used once, then discarded
```

DQN is the most sample-efficient because its replay buffer lets it reuse each transition many times. PPO is next because of its multiple epochs. REINFORCE and A2C discard data after one use.

**When this matters:** Real-world environments where each interaction is expensive — physical robots, expensive simulations, or systems where data collection takes time. DQN's replay buffer wins here.

### Stability

How reliably does the algorithm learn without collapsing?

```
Most stable → Least stable:

PPO         Training curve is smooth, evaluation is tight (± 4.9)
DQN         Training curve is smooth (after tuning), evaluation is perfect
REINFORCE   Evaluation is perfect, but training oscillates wildly
A2C         Both training and evaluation are volatile (± 88.7)
```

**When this matters:** Production systems where you need predictable behavior. A self-driving car that works 95% of the time and crashes 5% of the time is worse than one that works 85% of the time consistently. PPO's predictability is why it dominates production RL.

### Sensitivity to Hyperparameters

How much tuning does the algorithm need?

```
Least sensitive → Most sensitive:

PPO         Standard hyperparameters work across many problems
REINFORCE   Simple algorithm, few hyperparameters, but LR matters
DQN         Target update freq, buffer size, LR, epsilon decay — all critical
A2C         Learning rate ratio, entropy coeff, normalization — very finicky
```

We experienced this firsthand:
- **DQN**: MSE → Huber loss, target_update 10 → 1000, buffer 10K → 50K. Three changes needed.
- **A2C**: Three full rewrites. Shared → separate networks, normalization on → off, LR ratios flipped.
- **PPO**: Standard hyperparameters from the paper worked on the first try.
- **REINFORCE**: Worked out of the box.

**When this matters:** When you don't have time for extensive tuning. Research labs might have weeks to tune DQN. A startup shipping a product needs something that works with default settings. PPO wins.

### Action Space

What types of actions can the algorithm handle?

```
Discrete actions only:      DQN (requires argmax over all actions)
Any action space:           REINFORCE, A2C, PPO (output distributions directly)
```

This is the most fundamental divide. If your problem has continuous actions (robot joint angles, steering angles, force magnitudes), DQN is simply not an option without modification. Policy gradient methods (REINFORCE, A2C, PPO) handle continuous actions naturally by outputting Gaussian distributions instead of categorical distributions.

**When this matters:** Robotics (continuous torques and angles), autonomous driving (continuous steering and throttle), game AI with analog controls.

---

## Part 3: The Algorithm Decision Tree

```
                        What's your problem?
                              │
                    ┌─────────┴─────────┐
                    │                   │
            Discrete actions?     Continuous actions?
                    │                   │
              ┌─────┴─────┐            │
              │           │            │
          Small state   Large/         │
          space?      continuous       │
              │        state?          │
              │           │            │
         Q-Learning      │            │
         (tabular)       │            │
                         │            │
                    ┌────┴────┐       │
                    │         │       │
               Need max   Need        │
               sample    stability?   │
               efficiency?  │         │
                    │       │         │
                   DQN     PPO ◄──────┘
                         (default choice)
```

**The practical answer for most problems: start with PPO.** It handles discrete and continuous actions, is stable with default hyperparameters, and performs competitively on most tasks. Only switch to DQN if you need maximum sample efficiency with discrete actions, or to specialized algorithms if PPO's performance ceiling isn't high enough.

---

## Part 4: What Each Algorithm Taught Us

### Bandits → Exploration Is Fundamental

Without exploration, you never discover the best option. Without exploitation, you never use what you've learned. Every RL algorithm addresses this tradeoff:

```
Epsilon-greedy:     random exploration with probability ε (DQN, Q-Learning)
Entropy bonus:      reward the policy for maintaining randomness (A2C, PPO)
Stochastic policy:  sampling from a distribution IS exploration (REINFORCE, A2C, PPO)
UCB / Thompson:     explore where uncertainty is highest (bandits, Bayesian RL)
```

### Q-Learning → The Bellman Equation Is the Foundation

Every value-based method rests on the recursive relationship:

```
Q(s, a) = R(s, a) + γ × max Q(s', a')
```

This single equation enables temporal credit assignment — propagating the value of future rewards back to present actions. Without it, the agent would only learn from immediate rewards.

### DQN → Stability Requires Engineering

Three innovations (replay buffer, target network, Huber loss) were needed to make neural networks work with Q-learning. The naive combination exploded catastrophically. The lesson: combining two powerful ideas (neural networks + RL) requires careful engineering at the interface.

### REINFORCE → You Can Optimize Policies Directly

The policy gradient theorem opened a completely new approach to RL. Instead of estimating values and deriving policies, optimize the policy directly. The math is elegant (the log-prob trick), the implementation is simple, but the variance is brutal.

### A2C → Combining Approaches Is Hard

Actor-Critic should be the best of both worlds. In theory, it is. In practice, the two networks can fight each other, gradients can vanish, and subtle hyperparameter choices (normalization, learning rate ratios) can make or break the algorithm. We learned more from A2C's failures than from any algorithm's successes.

### PPO → Constraints Enable Progress

By limiting how much the policy can change per update, PPO paradoxically enables faster, more reliable learning. The clipping constraint doesn't slow progress — it prevents the catastrophic regressions that wasted most of A2C's training time recovering from bad updates.

This principle applies beyond RL: in optimization, constraints often help rather than hinder.

---

## Part 5: Beyond CartPole — Where Each Algorithm Shines

CartPole is a toy problem. Here's where each algorithm excels in the real world:

### DQN and Its Variants

**Best for:** Discrete action spaces with complex observations.

- Atari games (screen pixels → discrete button presses)
- Board games (board state → discrete move selection)
- Recommendation systems (user features → select from discrete catalog)

Variants like Double DQN, Dueling DQN, and Rainbow extend the basic algorithm with better value estimation, prioritized replay, and distributional Q-values.

### REINFORCE and Vanilla Policy Gradients

**Best for:** Simple problems, prototyping, understanding.

Rarely used in production because of high variance. But important as the foundation for all policy gradient methods. Understanding REINFORCE deeply makes A2C and PPO intuitive.

### A2C (and A3C — the asynchronous parallel version)

**Best for:** Problems where you can run many environments in parallel.

A3C (Asynchronous Advantage Actor-Critic) runs many copies of the agent in parallel, each in its own environment. The parallel experiences provide diverse data that stabilizes training without a replay buffer. Popular for game AI and simulated robotics where parallel environments are cheap.

### PPO

**Best for:** Almost everything.

- Robotics (continuous control, needs stability)
- Game AI (Dota 2, hide-and-seek, Minecraft)
- RLHF for language models (ChatGPT, Claude)
- Autonomous driving (continuous actions, safety-critical)
- Any new problem where you don't know which algorithm to use

PPO's dominance comes from its combination of generality (any action space), stability (clipped updates), simplicity (easy to implement), and performance (competitive with specialized algorithms). It's the "safe default" that works well enough for most applications.

---

## Part 6: What We Didn't Cover

This course covered the core family tree of RL algorithms. Here's what lies beyond:

### Model-Based RL

Everything we built is **model-free** — the agent doesn't try to understand how the environment works. It just learns what to do through trial and error.

**Model-based RL** first learns a model of the environment (how states transition, what rewards occur), then plans using that model. This is much more sample-efficient (you can "imagine" trajectories without real interaction) but requires an accurate model, which is hard to learn for complex environments.

Examples: MuZero (AlphaGo's successor), Dreamer, World Models.

### Off-Policy Actor-Critic (SAC, TD3)

Our Actor-Critic methods (A2C, PPO) are on-policy — they discard data after use. **SAC** (Soft Actor-Critic) and **TD3** (Twin Delayed DDPG) combine actor-critic with a replay buffer, getting the best of both worlds: policy gradient flexibility with DQN's sample efficiency.

SAC is the main competitor to PPO for continuous control tasks, often achieving better final performance with fewer samples.

### Multi-Agent RL

All our agents operated alone. Multi-agent RL deals with multiple agents interacting — competing (adversarial games), cooperating (robot teams), or both. This introduces game theory and non-stationarity (the environment changes as other agents learn).

### Offline RL

All our algorithms require interacting with the environment during training. **Offline RL** learns entirely from a fixed dataset of past experience — no new interactions. This is critical for domains where experimentation is dangerous or expensive (healthcare, autonomous driving, financial trading).

### Hierarchical RL

Complex tasks can be decomposed into subtasks. Hierarchical RL learns at multiple levels: a high-level policy selects subgoals, and low-level policies achieve them. This enables transfer learning (the same low-level skills work for different high-level goals) and tackles problems too long-horizon for flat RL.

---

## Part 7: The Complete Algorithm Table

| | Bandits | Q-Learning | DQN | REINFORCE | A2C | PPO |
|--|---------|-----------|-----|-----------|-----|-----|
| **States** | None | Discrete | Continuous | Continuous | Continuous | Continuous |
| **Actions** | Discrete | Discrete | Discrete | Any | Any | Any |
| **Learns** | Arm values | Q-table | Q-network | Policy | Policy + Value | Policy + Value |
| **Updates** | Every pull | Every step | Every step | End of episode | End of episode | Every N steps |
| **Data reuse** | N/A | None | Replay buffer | None | None | Multiple epochs |
| **Exploration** | ε-greedy/UCB/Thompson | ε-greedy | ε-greedy | Stochastic sampling | Stochastic + entropy | Stochastic + entropy |
| **Stability** | High | High | Medium | Low | Low-Medium | High |
| **Key innovation** | Uncertainty handling | Bellman equation | Replay + target net | Policy gradient theorem | Advantage function | Clipped objective |
| **Weakness** | No states | Table size | Discrete only | High variance | Instability | Sample efficiency |

---

## Course Complete

You've built the entire RL algorithm family tree from scratch:

1. **Multi-armed bandits** — 4 strategies for the exploration-exploitation tradeoff
2. **Custom GridWorld environment** — following the Gymnasium API
3. **Q-Learning** — the Bellman equation, TD learning, Q-tables
4. **Deep Q-Network** — replay buffers, target networks, function approximation
5. **REINFORCE** — policy gradients, the log-prob trick, baselines
6. **Actor-Critic (A2C)** — advantage function, separate networks, entropy bonus
7. **PPO** — clipped surrogate objective, GAE, multiple epochs

Every algorithm was written from scratch using only PyTorch's basic building blocks. You understand not just how each algorithm works, but **why** it was invented — what specific problem in the previous algorithm motivated its creation.

This progression — from bandits to PPO — mirrors the actual historical development of the field. The same problems we encountered (DQN's exploding loss, REINFORCE's variance, A2C's instability) are the same problems that motivated researchers to develop each successive algorithm. By building them yourself and hitting these walls firsthand, you've internalized the intuitions that drive RL research and engineering.

### Where To Go Next

The natural extensions from here:

- **Continuous control**: Modify PPO to output Gaussian distributions for continuous action spaces. Apply to MuJoCo environments (robot locomotion, manipulation).
- **SAC (Soft Actor-Critic)**: Combine actor-critic with a replay buffer for maximum sample efficiency on continuous control.
- **Multi-environment training**: Run PPO across 8-16 parallel environments simultaneously for faster, more stable training.
- **Real robotics**: Apply PPO to a simulated robot (PyBullet, Isaac Gym) and explore sim-to-real transfer.
- **RLHF**: Use PPO to fine-tune a language model from human preferences — this is how ChatGPT is trained.

Each of these builds directly on the algorithms and intuitions you've developed in this course.
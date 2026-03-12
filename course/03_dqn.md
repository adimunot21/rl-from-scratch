# Chapter 3: Deep Q-Network — Neural Networks Meet RL

## The Leap from Tables to Networks

In Chapter 2, Q-Learning stored a value for every (state, action) pair in a table. This works for 64 states. It doesn't work for CartPole, where the state is 4 continuous numbers — the cart's position, velocity, pole angle, and angular velocity. There are infinite possible combinations of these 4 numbers.

The solution: replace the Q-table with a **neural network** that takes a state as input and outputs Q-values for all actions. The network **generalizes** — given a state it's never seen before, it estimates Q-values based on similar states it has seen.

```
Q-Table approach:
  (state 7, action 2) → look up row 7, column 2 → Q = 3.5

Neural network approach:
  [0.02, -0.15, 0.08, 0.31] → neural network → [Q_left, Q_right]
                                                  [2.1,    3.8]
```

This seems straightforward, but naively combining neural networks with Q-Learning is catastrophically unstable. The DeepMind team discovered this the hard way in 2013, and their 2015 paper introduced three innovations that made it work. We implement all three.

## Part 1: The Q-Network

### Architecture

The Q-Network is a simple multi-layer perceptron (MLP):

```
State (4 numbers)
    │
    ▼
Linear(4 → 128) + ReLU       ← hidden layer 1
    │
    ▼
Linear(128 → 128) + ReLU     ← hidden layer 2
    │
    ▼
Linear(128 → 2)              ← output: Q-value for each action
    │
    ▼
[Q(s, left), Q(s, right)]    ← two numbers
```

### The Code

```python
class QNetwork(nn.Module):
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
```

### Line-by-Line

```python
nn.Linear(state_dim, hidden_dim)
```
- A fully connected layer: multiplies the input by a weight matrix and adds a bias. Input dimension 4 (CartPole state), output dimension 128.
- Parameters: 4×128 weights + 128 biases = 640 parameters.

```python
nn.ReLU()
```
- Activation function: `ReLU(x) = max(0, x)`. Passes positive values through, zeros out negative values. This introduces non-linearity — without it, stacking linear layers would just produce another linear function.

```python
nn.Linear(hidden_dim, n_actions)
```
- Final layer: 128 → 2. Outputs one Q-value per action. No activation function after this layer — Q-values can be any real number (positive or negative).

### How It Replaces the Q-Table

```
Q-Table lookup:
  Q(state=7, action=2) → table[7][2] → single number

Q-Network forward pass:
  Q(state=[0.02, -0.15, 0.08, 0.31]) → network([0.02, -0.15, 0.08, 0.31]) → [2.1, 3.8]
  Q-value for left = 2.1
  Q-value for right = 3.8
  Best action = right (higher Q-value)
```

The network outputs Q-values for ALL actions simultaneously. To get Q(s, a), we index into the output: `q_values[action]`. To find the best action: `argmax(q_values)`.

### Parameter Count

```
Layer 1: 4 × 128 + 128   =    640
Layer 2: 128 × 128 + 128 = 16,512
Layer 3: 128 × 2 + 2     =    258
                            ──────
Total:                     17,410 parameters
```

Compare to the Q-table's 256 entries (64 states × 4 actions). The network has 68× more parameters, but can handle infinite states.

---

## Part 2: Why Naive Deep Q-Learning Fails

### The Problem: Training Instability

If you simply replace the Q-table with a network and apply the same update rule, training explodes. We saw this firsthand — our first DQN attempt had loss growing from 0.01 to 4.9 billion, and the agent scored 19.7 (worse than random).

Three problems conspire to create this instability:

### Problem 1: Correlated Data

In Q-Learning, the agent takes consecutive steps: state 5 → state 6 → state 7 → state 8. These states are nearly identical (the cart moved slightly, the pole tilted slightly). Training a neural network on these consecutive, correlated samples is like studying for an exam by reading the same paragraph over and over — you memorize that paragraph but forget everything else.

Technically: gradient descent assumes training samples are **independent and identically distributed (i.i.d.)**. Consecutive RL transitions violate this assumption massively.

### Problem 2: Moving Targets

In supervised learning, the targets (labels) are fixed. "This image is a cat" doesn't change during training.

In Q-Learning, the target depends on the Q-network itself:

```
target = reward + γ × max Q_network(next_state, a')
                        ↑
                  THIS changes every time we update the network
```

Every time we update the network's weights, ALL the targets shift. The network is chasing a moving target. It adjusts toward the current target, which causes the target to move, which requires another adjustment, which moves the target again... This can spiral out of control.

### Problem 3: Q-Value Overestimation

The `max` in the target is biased upward. If Q-values have random noise (which they always do during training), `max` preferentially selects the overestimated values:

```
True Q-values:     [5.0, 5.0, 5.0]
Estimated Q-values: [4.8, 5.3, 4.9]   (random noise around true values)
max of estimates:   5.3                 (overestimates the true max of 5.0)
```

This overestimation compounds: overestimated targets lead to overestimated Q-values, which lead to even more overestimated targets. This is the positive feedback loop that caused our loss to explode to billions.

### The Deadly Triad

These three problems together are known as the **deadly triad** — the combination of:
1. Function approximation (neural network instead of table)
2. Bootstrapping (using estimated values in the target)
3. Off-policy learning (learning about the optimal policy while following an exploratory policy)

Any two of three are fine. All three together are dangerous. DQN's innovations specifically address this triad.

---

## Part 3: Innovation 1 — Experience Replay

### The Solution to Correlated Data

Instead of training on transitions as they arrive, store them in a **replay buffer** and sample **random batches** for training:

```
Step 1: Agent takes action, observes (s, a, r, s', done)
Step 2: Store this transition in a buffer (max 50,000 transitions)
Step 3: Sample a random batch of 64 transitions from the buffer
Step 4: Train the network on this random batch
```

Random sampling breaks the correlation between consecutive transitions. A single batch might contain transitions from episode 1, episode 50, and episode 200 — completely different situations. This satisfies the i.i.d. assumption.

### The Replay Buffer Code

```python
class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )
```

### Line-by-Line

```python
self.buffer = deque(maxlen=capacity)
```
- `deque` with `maxlen` automatically drops the oldest transitions when full. This means the buffer always contains the most recent 50,000 transitions. Old, potentially outdated experiences are gradually replaced.

```python
def push(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))
```
- Store one complete transition: the state we were in, the action we took, the reward we got, the state we ended up in, and whether the episode ended. This is everything needed for a Q-learning update.

```python
batch = random.sample(self.buffer, batch_size)
states, actions, rewards, next_states, dones = zip(*batch)
```
- `random.sample` picks 64 random transitions from the buffer **without replacement**.
- `zip(*batch)` transposes from a list of tuples to a tuple of lists. Before: `[(s1,a1,r1,s1',d1), (s2,a2,...)]`. After: `([s1,s2,...], [a1,a2,...], ...)`.

### Why Capacity Matters

Too small (1000): only recent experience. The network forgets early lessons.
Too large (1,000,000): stale transitions from when the policy was much worse. These can pollute training.
50,000 is a good balance for CartPole — enough diversity without too much staleness.

---

## Part 4: Innovation 2 — Target Network

### The Solution to Moving Targets

Keep a **frozen copy** of the Q-network that's only updated periodically (every 1000 steps). Use this frozen copy to compute targets:

```
Q-network (updated every step):
  Used for: selecting actions, computing current Q-values

Target network (updated every 1000 steps):
  Used for: computing targets in the Bellman equation
```

The target becomes:

```
target = reward + γ × max Q_TARGET(next_state, a')
                         ↑
              FROZEN network — doesn't change for 1000 steps
```

For 1000 steps, the targets are stable — the network can make real progress toward them. Then the target network is updated (copied from the Q-network), and the process repeats with new, slightly better targets.

### In Code

```python
# Create both networks
self.q_net = QNetwork(state_dim, n_actions)
self.target_net = QNetwork(state_dim, n_actions)
self.target_net.load_state_dict(self.q_net.state_dict())
self.target_net.eval()
```

```python
# Compute targets using the FROZEN target network
with torch.no_grad():
    next_q = self.target_net(next_states_t).max(dim=1).values
    target = rewards_t + self.gamma * next_q * (1.0 - dones_t)
```

```python
# Periodically sync target network
if self.train_steps % self.target_update_freq == 0:
    self.target_net.load_state_dict(self.q_net.state_dict())
```

### Line-by-Line

```python
self.target_net.load_state_dict(self.q_net.state_dict())
```
- Copy all weights from the Q-network to the target network. They start identical.

```python
self.target_net.eval()
```
- Set to evaluation mode. This network never trains — it's only used for forward passes to compute targets. `.eval()` disables dropout and batch normalization effects (not relevant for our simple network, but good practice).

```python
with torch.no_grad():
    next_q = self.target_net(next_states_t).max(dim=1).values
```
- `torch.no_grad()` disables gradient tracking — we don't need gradients for the target computation, and skipping them saves memory.
- `self.target_net(next_states_t)` — compute Q-values for all next states using the FROZEN network.
- `.max(dim=1).values` — take the maximum Q-value across actions for each state.

```python
target = rewards_t + self.gamma * next_q * (1.0 - dones_t)
```
- The Bellman target.
- `(1.0 - dones_t)` — if the episode is done, there's no future value. `dones` is 1.0 when done, so `1.0 - 1.0 = 0.0` zeros out the future term.

### Why 1000 Steps?

Our first attempt used `target_update_freq=10` — the target network updated every 10 steps, which meant the targets were barely more stable than without a target network at all. The loss exploded to billions.

At 1000 steps, the target network stays frozen for thousands of training updates. The Q-network can meaningfully converge toward these stable targets before they shift. The loss stayed in the range 0.01–0.23.

---

## Part 5: The Training Step — Putting It Together

```python
def train_step(self):
    if len(self.buffer) < self.batch_size:
        return None  # not enough data yet

    # Sample random batch (Innovation 1)
    states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

    # Convert to tensors
    states_t = torch.FloatTensor(states)
    actions_t = torch.LongTensor(actions)
    rewards_t = torch.FloatTensor(rewards)
    next_states_t = torch.FloatTensor(next_states)
    dones_t = torch.FloatTensor(dones)

    # Current Q-values for the actions we took
    current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

    # Target Q-values using frozen network (Innovation 2)
    with torch.no_grad():
        next_q = self.target_net(next_states_t).max(dim=1).values
        target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

    # Huber loss (Innovation 3: robustness)
    loss = F.smooth_l1_loss(current_q, target)

    # Update
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
    self.optimizer.step()
```

### The gather Operation

This is the trickiest line:

```python
current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
```

Let's trace it step by step:

```python
self.q_net(states_t)
# Shape: (64, 2) — Q-values for both actions, for all 64 states in the batch
# Example row: [2.1, 3.8] — Q(left)=2.1, Q(right)=3.8
```

```python
actions_t.unsqueeze(1)
# Shape: (64,) → (64, 1) — the actions we actually took
# Example: [[1], [0], [1], [0], ...] — right, left, right, left, ...
```

```python
.gather(1, actions_t.unsqueeze(1))
# For each row, pick the Q-value at the action index
# Row 0: actions=[1] → pick Q[1] = 3.8
# Row 1: actions=[0] → pick Q[0] = some_value
# Shape: (64, 1)
```

```python
.squeeze(1)
# Remove the extra dimension: (64, 1) → (64,)
```

Result: a tensor of 64 Q-values, one per transition in the batch — the Q-value the network currently assigns to the action we actually took.

### Huber Loss vs MSE

We use `F.smooth_l1_loss` (Huber loss) instead of `F.mse_loss`:

```
MSE loss:    L = (prediction - target)²
Huber loss:  L = 0.5 × (prediction - target)²    if |error| < 1
             L = |prediction - target| - 0.5       if |error| ≥ 1
```

The key difference: for large errors, MSE squares them (error of 100 → loss of 10,000), while Huber is linear (error of 100 → loss of 99.5). This prevents a single bad target from creating an enormous gradient that destabilizes the network.

This was the critical fix for our training. With MSE, a single overestimated Q-value created a huge loss, a huge gradient, and a huge weight update — which created even more overestimation. Huber loss caps the gradient magnitude, breaking the positive feedback loop.

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
```

Additional safety net: if the total gradient magnitude exceeds 1.0, scale all gradients down proportionally. This preserves the gradient direction but caps the step size. Belt and suspenders with Huber loss — both prevent oversized updates.

---

## Part 6: The Training Loop

```python
for ep in range(n_episodes):
    state, _ = env.reset()

    while True:
        action = agent.select_action(state)           # ε-greedy
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, done)
        loss = agent.train_step()                      # sample batch, update network

        state = next_state
        if done:
            break

    agent.decay_epsilon()
```

Every single step:
1. Choose action (ε-greedy using the Q-network)
2. Take action, observe transition
3. Store transition in replay buffer
4. Sample a random batch from the buffer
5. Compute targets using the frozen target network
6. Compute loss and update the Q-network
7. Periodically copy Q-network weights to target network

This is much more data-efficient than tabular Q-Learning, which updates one (state, action) pair per step. Here, every step trains on 64 random transitions — 64× the learning per step.

---

## Part 7: Our Results

### The Unstable Run (MSE, target update every 10)

```
Episode   50  | reward:  29.2 | loss: 147
Episode  150  | reward: 109.6 | loss: 214,049
Episode  300  | reward:  10.9 | loss: 205,825,861
Episode  500  | reward:  20.1 | loss: 4,921,056,801

Evaluation: 19.7 ± 2.1  ← worse than random
```

The loss grew by a factor of ~30 million. The Q-values spiraled to infinity. The agent learned nothing.

### The Stable Run (Huber loss, target update every 1000)

```
Episode   50  | reward:  23.6 | loss: 0.013
Episode  250  | reward: 181.0 | loss: 0.034
Episode  500  | reward: 143.7 | loss: 0.095
Episode  750  | reward: 500.0 | loss: 0.144

Evaluation: 500.0 ± 0.0  ← perfect score, every episode
```

Loss stayed between 0.01 and 0.22. The agent learned steadily and solved CartPole completely.

### What Changed

| Parameter | Unstable | Stable | Why |
|-----------|----------|--------|-----|
| Loss function | MSE | Huber | Caps gradient for large errors |
| Target update | Every 10 steps | Every 1000 steps | Stable targets |
| Gradient clip | 10.0 | 1.0 | Tighter safety net |
| Learning rate | 1e-3 | 5e-4 | Smaller, safer updates |
| Buffer size | 10,000 | 50,000 | More diverse samples |

No single change would have fixed it alone. The instability was a feedback loop — overestimation → large loss → large update → more overestimation. Breaking the loop required addressing it at multiple points: Huber loss caps the gradient, slow target updates prevent runaway targets, and a large buffer provides diverse training data.

---

## Part 8: DQN's Limitations

### Discrete Actions Only

DQN outputs Q-values for each action: `[Q(left), Q(right)]`. The policy is `argmax` — pick the highest. This requires enumerating all possible actions.

For CartPole (2 actions) or Atari games (18 actions), this works. For a robot arm with continuous joint angles (infinite possible actions), you can't enumerate them. You'd need to search over a continuous action space, which is expensive.

This limitation motivates **policy gradient methods** (Chapter 4): instead of estimating Q-values and deriving a policy, directly output the action.

### Sample Efficiency vs Stability

DQN is relatively sample-efficient — the replay buffer lets it reuse each transition many times. But it requires careful tuning (target update frequency, learning rate, buffer size, epsilon decay) and can be fragile. Our experience showed this — three hyperparameter choices (MSE, target_update=10, clip=10) turned a perfect solver into a complete failure.

---

## Summary

The three innovations that make DQN work:

```
Problem: Correlated data         → Solution: Experience Replay
  Store transitions, sample random batches
  Breaks temporal correlation between consecutive updates

Problem: Moving targets           → Solution: Target Network
  Frozen copy of Q-network, updated every 1000 steps
  Provides stable targets for the Q-network to learn toward

Problem: Overestimation explosion → Solution: Huber Loss + Clipping
  Linear loss for large errors (not quadratic)
  Gradient clipping as additional safety net
```

The DQN training loop:

```
For each step:
    1. ε-greedy action selection using Q-network
    2. Store (s, a, r, s', done) in replay buffer
    3. Sample random batch of 64 transitions
    4. Compute targets: r + γ × max Q_target(s', a')
    5. Compute Huber loss between Q-network predictions and targets
    6. Gradient descent step on Q-network
    7. Every 1000 steps: copy Q-network → target network
    8. Decay epsilon
```

## What's Next

In [Chapter 4](04_reinforce.md), we take a completely different approach. Instead of estimating Q-values and deriving a policy, we **directly optimize the policy** using the policy gradient theorem. This handles continuous actions, naturally produces stochastic policies, and introduces a fundamentally different philosophy of RL — but comes with its own challenges.
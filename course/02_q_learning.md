# Chapter 2: GridWorld + Q-Learning — Sequential Decisions and the Bellman Equation

## From Bandits to Sequential Decisions

In the bandit problem, each action was independent — pulling arm 3 didn't change what happened when you pulled arm 7 next. Real RL problems aren't like this. A robot turning left puts it in a different position than turning right. A chess move changes the entire board. **Actions have consequences that persist into the future.**

This chapter introduces two fundamental new concepts:

1. **States** — the agent is "somewhere" and its situation changes based on actions
2. **Long-term value** — a good action isn't just one that gives immediate reward, but one that leads to good future states

We implement both through a custom GridWorld environment and the Q-Learning algorithm.

## Part 1: The GridWorld Environment

### The Grid

```
S . . . . . . .
. . # . . X . .
. . # . . . . .
. . . . # # . .
. X . . # . . .
. . . . . . X .
. . # . . . . .
. . # . . . . G
```

- `S` = start position (top-left corner)
- `G` = goal (+10 reward, episode ends)
- `.` = empty cell (-0.1 reward per step)
- `#` = wall (can't enter)
- `X` = penalty (-5 reward, episode continues)

The agent can move in 4 directions: up, right, down, left. If it tries to walk into a wall or off the edge, it stays in place (but still pays the -0.1 step cost).

### Why -0.1 Per Step?

Without a step penalty, the agent has no incentive to take the shortest path. It could wander randomly for 1000 steps and still get +10 when it eventually stumbles onto the goal. The -0.1 penalty means every step costs something: a 14-step path gives 10.0 - 13×0.1 = 8.7 reward, while a 50-step path gives 10.0 - 49×0.1 = 5.1 reward. Shorter is better.

### Following the Gymnasium API

We build our environment as a proper Gymnasium environment. This means it follows a standard interface that any RL algorithm can use:

```python
env = GridWorldEnv()
state, info = env.reset()         # start a new episode
state, reward, terminated, truncated, info = env.step(action)  # take one action
```

This is worth understanding because every RL environment you'll ever encounter uses this same interface — CartPole, LunarLander, Atari games, robotics simulators, all of them.

### The Code

```python
class GridWorldEnv(gym.Env):
    def __init__(self, grid=None, max_steps=200):
        super().__init__()
        self.grid = self._parse_grid(grid or DEFAULT_GRID)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.max_steps = max_steps

        # Gymnasium API: define action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.rows * self.cols)

        # Action deltas: (row_change, col_change)
        self.action_deltas = {
            0: (-1, 0),   # up
            1: (0, 1),    # right
            2: (1, 0),    # down
            3: (0, -1),   # left
        }
```

### Line-by-Line

```python
self.action_space = spaces.Discrete(4)
```
- Tells Gymnasium (and any algorithm using this environment) that there are exactly 4 possible actions. `env.action_space.sample()` returns a random action (0, 1, 2, or 3).

```python
self.observation_space = spaces.Discrete(self.rows * self.cols)
```
- The state is a single integer from 0 to 63 (for an 8×8 grid). State 0 is row 0, column 0 (top-left). State 63 is row 7, column 7 (bottom-right).
- We convert between (row, col) and integer state using: `state = row × cols + col` and `row = state // cols, col = state % cols`.

```python
self.action_deltas = {
    0: (-1, 0),   # up: row decreases by 1
    1: (0, 1),    # right: column increases by 1
    2: (1, 0),    # down: row increases by 1
    3: (0, -1),   # left: column decreases by 1
}
```
- Row 0 is the top of the grid, so "up" means row - 1. This is standard for grid representations where (0,0) is the top-left corner.

### The Step Function

```python
def step(self, action):
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
        reward = -0.1
        truncated = self.steps >= self.max_steps
        return self._pos_to_state(self.agent_pos), reward, False, truncated, {}
```

The return values follow the Gymnasium standard:
- `observation` — the new state (integer 0–63)
- `reward` — immediate reward for this step
- `terminated` — did the episode end naturally? (reached goal)
- `truncated` — did the episode end artificially? (hit max steps)
- `info` — extra information (empty dict for us)

The distinction between `terminated` and `truncated` matters for learning. If terminated (reached goal), there's no future value — the episode is truly over. If truncated (hit step limit), the agent might have received more reward if allowed to continue. The Q-learning update handles these differently.

---

## Part 2: The Bellman Equation

### The Key Insight

Consider the agent at cell (2, 3) in our grid. What's the value of being here? It depends on what happens next — which depends on what happens after that — which depends on what happens after *that*...

The **Bellman equation** breaks this recursive chain with a beautiful trick. The value of a state equals the immediate reward plus the discounted value of the next state:

```
V(s) = R(s, a) + γ × V(s')
```

Or for Q-values (state-action values):

```
Q(s, a) = R(s, a) + γ × max_a' Q(s', a')
```

In words: "The value of taking action a in state s equals:
- the immediate reward R(s, a), plus
- the discounted value of the best action in the next state"

### A Concrete Example

Consider three cells in a line, with the goal at the right:

```
Cell A → Cell B → Cell G (goal, +10)
  -0.1     -0.1      +10
```

With γ = 0.99:

```
Q(G, any) = 10.0                             (goal — terminal)
Q(B, right) = -0.1 + 0.99 × 10.0 = 9.79     (one step from goal)
Q(A, right) = -0.1 + 0.99 × 9.79 = 9.59     (two steps from goal)
```

The values decrease as you move further from the goal — each extra step costs -0.1 and adds a γ discount. The agent learns: "Cell B is worth 9.79, Cell A is worth 9.59, Cell G is worth 10.0" and always moves toward higher values.

### Why max_a'?

```
Q(s, a) = R(s, a) + γ × max_a' Q(s', a')
                          ↑
                 "assume we act optimally from here on"
```

The `max` assumes the agent will take the best action in the next state. This is **optimistic** — it says "the value of being in state s' is the value of the best thing I can do there." This is correct for the optimal Q-function: if you're acting optimally, you will pick the best action.

---

## Part 3: Q-Learning — The Algorithm

### The Q-Table

Q-Learning stores a value for every (state, action) pair. For our 8×8 grid with 4 actions:

```
         Up    Right   Down   Left
State 0: [0.00, 0.00,  0.00,  0.00]    ← top-left corner (start)
State 1: [0.00, 0.00,  0.00,  0.00]    ← one step right of start
  ...
State 63:[0.00, 0.00,  0.00,  0.00]    ← bottom-right (goal)
```

Initially all zeros. Through experience, these values get updated to reflect the true value of each (state, action) combination.

After training, the table might look like:

```
         Up     Right   Down    Left
State 0: [-1.2,  8.5,   7.3,   -1.2]   ← right and down are good (toward goal)
State 1: [5.1,   8.7,   6.2,    7.5]   ← right is best (moving toward goal)
  ...
```

The policy is implicit: at each state, pick the action with the highest Q-value.

### The Update Rule

After taking action a in state s, observing reward r, and ending up in state s':

```
Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') - Q(s, a)]
```

Let's break this apart:

```
target = r + γ × max_a' Q(s', a')     "what we actually experienced"
current = Q(s, a)                       "what we predicted"
td_error = target - current             "how wrong we were"
Q(s, a) ← Q(s, a) + α × td_error      "adjust our prediction"
```

- `α` (learning rate, typically 0.1) — how much we adjust per update. Too high: values oscillate. Too low: learning is slow.
- `γ` (discount, typically 0.99) — how much we value future rewards.
- `td_error` (temporal difference error) — the surprise. Positive: things went better than expected. Negative: things went worse.

### A Concrete Update

The agent is at state 9 (row 1, col 1), takes action "right" (1), gets reward -0.1, and ends up at state 10 (row 1, col 2 — but that's a wall, so it stays at 9). Wait, let's pick a valid example.

The agent is at state 1 (row 0, col 1), takes action "down" (2), gets reward -0.1, ends up at state 9 (row 1, col 1).

Current Q-values:
```
Q(state 1, down) = 0.0      (never been here before)
Q(state 9, up)   = 0.0
Q(state 9, right)= 0.3      (from a previous update)
Q(state 9, down) = 0.5
Q(state 9, left) = 0.1
```

The update:
```
target = -0.1 + 0.99 × max(0.0, 0.3, 0.5, 0.1)
       = -0.1 + 0.99 × 0.5
       = -0.1 + 0.495
       = 0.395

td_error = 0.395 - 0.0 = 0.395

Q(state 1, down) ← 0.0 + 0.1 × 0.395 = 0.0395
```

The Q-value went from 0.0 to 0.0395. Not a big change from one update, but after hundreds of episodes where the agent passes through this state, the value converges to the true value.

### The Code

```python
class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(float)

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        q_values = [self.q_table[(state, a)] for a in range(self.n_actions)]
        max_q = max(q_values)
        best_actions = [a for a in range(self.n_actions) if q_values[a] == max_q]
        return np.random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[(state, action)]
        if done:
            target = reward
        else:
            next_q_values = [self.q_table[(next_state, a)] for a in range(self.n_actions)]
            target = reward + self.gamma * max(next_q_values)
        td_error = target - current_q
        self.q_table[(state, action)] = current_q + self.lr * td_error
        return td_error
```

### Line-by-Line: The Update

```python
self.q_table = defaultdict(float)
```
- A dictionary that returns 0.0 for any key that hasn't been set yet. This means unseen (state, action) pairs automatically have Q-value 0 — optimistic enough to encourage exploration.

```python
if done:
    target = reward
```
- If the episode is over (reached goal or truncated), there are no future rewards. The target is just the immediate reward. For the goal: target = 10.0. This is the "anchor" that values propagate backward from.

```python
else:
    next_q_values = [self.q_table[(next_state, a)] for a in range(self.n_actions)]
    target = reward + self.gamma * max(next_q_values)
```
- If not done, the target is immediate reward plus discounted best future value. This is the Bellman equation in code.
- `max(next_q_values)` — we assume optimal behavior from the next state onward.

```python
td_error = target - current_q
self.q_table[(state, action)] = current_q + self.lr * td_error
```
- TD error: how surprised were we? Positive = better than expected, negative = worse.
- Update: nudge the Q-value toward the target by `lr × td_error`.

### Epsilon Decay

```python
def decay_epsilon(self):
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

Same concept as the bandit chapter, but now much more important. The agent starts with ε = 1.0 (100% random) and decays to ε = 0.01 (1% random). This is critical because:

1. **Early training** (ε ≈ 1.0): The Q-values are all zero — there's nothing to exploit. Random exploration discovers the grid layout.
2. **Mid training** (ε ≈ 0.3): Q-values have rough estimates. The agent mostly follows them but still explores alternative paths.
3. **Late training** (ε ≈ 0.01): Q-values are accurate. The agent follows the learned optimal path with rare exploration.

---

## Part 4: How Learning Progresses

### Training Timeline

**Episodes 1–100 (ε ≈ 1.0 → 0.37)**:
```
Average reward: -8.41, average length: 59 steps
```
The agent stumbles randomly. It sometimes finds the goal, sometimes hits penalties, often gets truncated at 200 steps. But every step updates a Q-value. The Q-table is slowly filling in.

**Episodes 100–400 (ε ≈ 0.37 → 0.13)**:
```
Average reward: 6.90, average length: 18 steps
```
The agent consistently finds the goal and paths are getting shorter. Q-values near the goal are accurate (propagated from the +10 terminal reward). Values further away are still rough.

**Episodes 400–2000 (ε ≈ 0.13 → 0.01)**:
```
Average reward: 8.66, average length: 14 steps
```
Converged. The optimal 14-step path is found. TD errors drop to near zero — the Q-values are stable and self-consistent.

### How Values Propagate

Values propagate **backward from the goal**, one step per episode:

```
Episode 1: Agent stumbles onto goal from cell B.
           Q(B, right) gets a big positive update (reward 10.0)

Episode 5: Agent passes through cell A, then B, then goal.
           Q(A, right) gets updated using Q(B, right) — which is now positive
           → Q(A, right) becomes positive too

Episode 20: Agent passes through cell Z → ... → A → B → goal.
            Q(Z, ...) gets updated using values that have already propagated
            → the "wave" of positive values spreads backward from the goal
```

After enough episodes, every cell has accurate Q-values pointing toward the goal, like a flow field. The agent can start anywhere and follow the arrows to the goal.

---

## Part 5: Visualizing What Was Learned

### The Q-Value Heatmap

After training, we plot the maximum Q-value at each cell as a color, with arrows showing the best action:

```
  High value (green) → close to goal, on the optimal path
  Low value (red)    → far from goal, near penalties, or in dead ends
  Arrows             → the direction the agent would move (its policy)
```

The heatmap shows a smooth gradient from low (far from goal) to high (near goal), with dips near penalty cells and walls. The arrows trace the optimal path from start to goal.

### Reading the Policy

The arrows ARE the policy. At each cell, the arrow points in the direction of the action with the highest Q-value. Following these arrows from S to G traces the optimal path — the same 14-step path the agent learned.

---

## Part 6: Why Q-Learning Works

### Temporal Difference Learning

Q-Learning is a form of **temporal difference (TD) learning** — it learns from the *difference* between consecutive predictions, not from complete episodes.

Contrast with **Monte Carlo** methods (like REINFORCE in Chapter 4):
- Monte Carlo: play a full episode, compute the total return, then update
- TD: take one step, update immediately using the estimated value of the next state

TD advantages:
- **Learns every step** (don't need to wait for episode end)
- **Works for continuing tasks** (no natural episodes)
- **Lower variance** (one step of randomness vs an entire episode)

TD disadvantages:
- **Biased** (relies on estimated values that might be wrong)
- The estimated next-state value Q(s', a') is itself an estimate — we're updating estimates with estimates. This is called **bootstrapping**.

### Off-Policy Learning

Q-Learning is **off-policy** — it learns the optimal Q-values regardless of what the agent actually does. The update uses `max_a' Q(s', a')` (the best action), not the action the agent actually took in the next state.

This means the agent can explore wildly (high ε) while learning the optimal policy. The exploration doesn't corrupt the learned values because the update always assumes optimal future behavior.

Compare to **on-policy** methods (like SARSA, which we don't implement but is worth knowing about). SARSA's update uses the action the agent *actually took* next:

```
Q-Learning (off-policy): Q(s,a) ← Q(s,a) + α[r + γ × max_a' Q(s',a') - Q(s,a)]
SARSA (on-policy):       Q(s,a) ← Q(s,a) + α[r + γ × Q(s',a') - Q(s,a)]
                                                        ↑ actual next action, not max
```

### Convergence Guarantee

With a Q-table (finite states, finite actions), Q-Learning is **guaranteed to converge** to the optimal Q-values under mild conditions:
1. Every (state, action) pair is visited infinitely often (exploration)
2. The learning rate decreases appropriately over time

Our epsilon-greedy exploration with ε_min = 0.01 ensures condition 1 — there's always a small chance of exploring any action. The fixed learning rate α = 0.1 doesn't satisfy condition 2 perfectly, but works well in practice for this problem size.

---

## Part 7: The Limitations — Why We Need Deep RL

### The Table Size Problem

Our GridWorld has 64 states × 4 actions = 256 entries. Easy.

What about bigger problems?

```
10×10 grid:        100 states × 4 actions = 400 entries          ✓ fine
100×100 grid:      10,000 states × 4 actions = 40,000 entries    ✓ manageable
Atari game:        ~10^60 possible screen states                  ✗ impossible
CartPole:          infinite states (continuous values)            ✗ impossible
Chess:             ~10^47 possible board positions                ✗ impossible
```

For continuous state spaces like CartPole (where the state is 4 real numbers: position, velocity, angle, angular velocity), there are infinite possible states. You can't have a table with infinite rows.

### The Generalization Problem

Even if you could store a huge table, it would be wasteful. In CartPole, the states (0.01, 0.02, -0.03, 0.04) and (0.011, 0.021, -0.031, 0.041) are nearly identical — they should have nearly identical Q-values. But a table treats them as completely unrelated entries. No information transfers from one state to nearby states.

What we need is a **function** that takes a state as input and outputs Q-values — a function that can **generalize** from states it has seen to states it hasn't. That function is a neural network.

This is exactly what the next chapter covers: replacing the Q-table with a neural network, creating the Deep Q-Network (DQN).

---

## Summary

| Concept | What It Is | Why It Matters |
|---------|-----------|----------------|
| State | Where the agent is | Actions have different values in different situations |
| Q-value | Expected return for (state, action) | Tells the agent what to do |
| Bellman equation | Q(s,a) = R + γ·max Q(s',a') | Recursive relationship that makes learning possible |
| TD error | target - prediction | The learning signal — how surprised the agent was |
| Q-table | Dictionary of (state, action) → value | Stores everything the agent has learned |
| ε-greedy | Random ε%, greedy (1-ε)% | Balances exploration and exploitation |
| Discount (γ) | Future reward multiplier | Makes agents prefer sooner rewards |
| Off-policy | Learn optimal values regardless of exploration | Can explore aggressively while learning correctly |

The training loop:

```
For each episode:
    state = env.reset()
    While not done:
        action = ε-greedy(state)           ← explore or exploit
        next_state, reward, done = env.step(action)
        
        target = reward + γ × max Q(next_state, a')    ← Bellman
        td_error = target - Q(state, action)
        Q(state, action) += α × td_error               ← update
        
        state = next_state
    Decay ε                                 ← less exploration over time
```

## What's Next

In [Chapter 3](03_dqn.md), we replace the Q-table with a neural network to handle continuous state spaces. This is the Deep Q-Network (DQN) — the algorithm that made headlines when DeepMind used it to play Atari games at superhuman level. We'll implement its three key innovations: experience replay, target networks, and the transition from tabular to deep RL.
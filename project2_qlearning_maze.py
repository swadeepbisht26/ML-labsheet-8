"""
Project 2: Q-Learning based Maze Solver
Lab Sheet-08 | MCA III Semester

The maze is implemented as a Gymnasium environment, and a tabular Q-Learning
agent learns the shortest path from Start (S) to Goal (G).

Run:  python project2_qlearning_maze.py
Outputs: q_table.npy, maze_learning_curves.png, maze_policy.png,
         maze_hyperparameter_comparison.png, maze_hyperparameter_results.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

SEED = 42

# 0 = free, 1 = wall, S = start, G = goal
MAZE = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 0, 0],
])
START = (0, 0)
GOAL = (7, 7)
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}   # up, down, left, right
ARROWS = {0: "^", 1: "v", 2: "<", 3: ">"}


# ---------------------------------------------------------------- Environment
class MazeEnv(gym.Env):
    """Grid maze. Reward design:
       +100 reaching goal, -10 hitting wall/boundary, -1 per step (encourages short paths)."""

    def __init__(self, maze=MAZE, start=START, goal=GOAL, max_steps=200):
        super().__init__()
        self.maze, self.start, self.goal = maze, start, goal
        self.rows, self.cols = maze.shape
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.rows * self.cols)

    def _state(self, pos):
        return pos[0] * self.cols + pos[1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos, self.steps = self.start, 0
        return self._state(self.pos), {}

    def step(self, action):
        self.steps += 1
        dr, dc = ACTIONS[int(action)]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        if not (0 <= r < self.rows and 0 <= c < self.cols) or self.maze[r, c] == 1:
            reward = -10.0                      # wall / boundary: stay in place
        else:
            self.pos = (r, c)
            reward = 100.0 if self.pos == self.goal else -1.0
        terminated = self.pos == self.goal
        truncated = self.steps >= self.max_steps
        return self._state(self.pos), reward, terminated, truncated, {}


# ---------------------------------------------------------------- Q-Learning
def train(alpha=0.1, gamma=0.95, eps_start=1.0, eps_min=0.05, eps_decay=0.995,
          episodes=2000, seed=SEED):
    env = MazeEnv()
    rng = np.random.default_rng(seed)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    eps = eps_start
    rewards, steps_hist, success = [], [], []

    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            a = rng.integers(4) if rng.random() < eps else int(np.argmax(q[s]))
            s2, r, term, trunc, _ = env.step(a)
            # Bellman update: Q(s,a) += alpha * [r + gamma * max Q(s',.) - Q(s,a)]
            target = r + (0.0 if term else gamma * np.max(q[s2]))
            q[s, a] += alpha * (target - q[s, a])
            s, total, done = s2, total + r, term or trunc
        eps = max(eps_min, eps * eps_decay)
        rewards.append(total)
        steps_hist.append(env.steps)
        success.append(1 if term else 0)
    return q, np.array(rewards), np.array(steps_hist), np.array(success)


def greedy_path(q, max_len=100):
    env = MazeEnv()
    s, _ = env.reset()
    path, pos = [env.pos], env.pos
    for _ in range(max_len):
        s, _, term, trunc, _ = env.step(int(np.argmax(q[s])))
        path.append(env.pos)
        if term:
            return path, True
        if trunc:
            break
    return path, False


def shortest_path_bfs():
    """Ground-truth optimum to verify the agent's learned path."""
    from collections import deque
    dq, seen = deque([(START, 0)]), {START}
    while dq:
        (r, c), d = dq.popleft()
        if (r, c) == GOAL:
            return d
        for dr, dc in ACTIONS.values():
            n = (r + dr, c + dc)
            if (0 <= n[0] < MAZE.shape[0] and 0 <= n[1] < MAZE.shape[1]
                    and MAZE[n] == 0 and n not in seen):
                seen.add(n)
                dq.append((n, d + 1))
    return None


def moving_avg(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode="valid")


# ---------------------------------------------------------------- Main
if __name__ == "__main__":
    # 1. Train final agent
    q, rewards, steps, success = train()
    np.save("q_table.npy", q)
    print("Saved Q-table -> q_table.npy")

    # 2. Learning curves
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(rewards, alpha=0.3); ax[0].plot(range(49, len(rewards)), moving_avg(rewards), "r")
    ax[0].set(title="Reward per Episode", xlabel="Episode", ylabel="Total reward")
    ax[1].plot(steps, alpha=0.3); ax[1].plot(range(49, len(steps)), moving_avg(steps), "r")
    ax[1].set(title="Steps per Episode", xlabel="Episode", ylabel="Steps")
    ax[2].plot(range(49, len(success)), moving_avg(success) * 100, "g")
    ax[2].set(title="Success Rate (50-ep avg)", xlabel="Episode", ylabel="% reaching goal")
    plt.tight_layout(); plt.savefig("maze_learning_curves.png", dpi=150); plt.close()

    # 3. Learned path vs optimal
    path, reached = greedy_path(q)
    optimal = shortest_path_bfs()
    print(f"Learned path length: {len(path) - 1} steps (reached goal: {reached})")
    print(f"BFS optimal length : {optimal} steps")

    # 4. Visualise maze, policy arrows and path
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(MAZE, cmap="Greys")
    px, py = [p[1] for p in path], [p[0] for p in path]
    ax.plot(px, py, "r-o", markersize=4, linewidth=2, label="Learned path")
    for r in range(MAZE.shape[0]):
        for c in range(MAZE.shape[1]):
            if MAZE[r, c] == 0 and (r, c) != GOAL:
                ax.text(c, r + 0.3, ARROWS[int(np.argmax(q[r * MAZE.shape[1] + c]))],
                        ha="center", color="blue", fontsize=9)
    ax.text(START[1], START[0] - 0.2, "S", ha="center", color="green", fontsize=14, weight="bold")
    ax.text(GOAL[1], GOAL[0], "G", ha="center", va="center", color="orange", fontsize=14, weight="bold")
    ax.set_title(f"Learned Policy (path = {len(path) - 1} steps, optimal = {optimal})")
    ax.legend(loc="upper right"); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig("maze_policy.png", dpi=150); plt.close()

    # 5. Hyperparameter tuning / comparison (3 seeds each for reliability)
    settings = {
        "alpha=0.1, gamma=0.95 (baseline)": dict(alpha=0.1, gamma=0.95),
        "alpha=0.5, gamma=0.95": dict(alpha=0.5, gamma=0.95),
        "alpha=0.01, gamma=0.95": dict(alpha=0.01, gamma=0.95),
        "alpha=0.1, gamma=0.5": dict(alpha=0.1, gamma=0.5),
        "alpha=0.1, eps_decay=0.9": dict(alpha=0.1, gamma=0.95, eps_decay=0.9),
    }
    rows, curves = [], {}
    for name, hp in settings.items():
        all_rewards, finals, lens = [], [], []
        for sd in (1, 2, 3):
            qh, rw, _, _ = train(seed=sd, **hp)
            all_rewards.append(rw)
            finals.append(rw[-100:].mean())
            p, ok = greedy_path(qh)
            lens.append(len(p) - 1 if ok else np.nan)
        curves[name] = moving_avg(np.mean(all_rewards, axis=0))
        rows.append({"setting": name,
                     "avg_final_reward(last100)": round(float(np.mean(finals)), 2),
                     "avg_learned_path_len": round(float(np.nanmean(lens)), 1)
                     if not np.all(np.isnan(lens)) else "fail",
                     "optimal_len": optimal})
    df = pd.DataFrame(rows)
    df.to_csv("maze_hyperparameter_results.csv", index=False)
    print("\n=== Hyperparameter comparison (mean of 3 seeds) ===")
    print(df.to_string(index=False))

    plt.figure(figsize=(9, 5))
    for name, cv in curves.items():
        plt.plot(range(49, 49 + len(cv)), cv, label=name)
    plt.title("Effect of Hyperparameters on Learning"); plt.xlabel("Episode")
    plt.ylabel("Reward (50-ep moving avg)"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("maze_hyperparameter_comparison.png", dpi=150); plt.close()
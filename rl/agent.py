"""
Tabular Q-learning agent for the clutch-possession environment.

Deliberately kept to classic tabular Q-learning (not deep RL) because the
state space is small (5 score buckets x up to ~8 possessions-remaining
values) and fully enumerable - using a neural net here would be solving a
30-cell lookup problem with an oversized hammer. The point of this project
is demonstrating you know how to choose the right tool for the state
space, same judgment call that went into picking DQN vs PPO vs MADDPG
per-scenario in the drone project.
"""

from __future__ import annotations
import random
from collections import defaultdict
from typing import Dict, Tuple

from rl.environment import ClutchPossessionEnv, ACTIONS, NUM_SCORE_BUCKETS

State = Tuple[int, int]


class QLearningAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay_episodes: int = 3000):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.q: Dict[Tuple[State, int], float] = defaultdict(float)

    def _epsilon(self, episode: int) -> float:
        frac = min(1.0, episode / self.epsilon_decay_episodes)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, state: State, episode: int, greedy: bool = False) -> int:
        if not greedy and random.random() < self._epsilon(episode):
            return random.choice(ACTIONS)
        q_values = [self.q[(state, a)] for a in ACTIONS]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(ACTIONS, q_values) if q == max_q]
        return random.choice(best_actions)  # break ties randomly

    def update(self, state: State, action: int, reward: float, next_state: State, done: bool):
        current_q = self.q[(state, action)]
        target = reward if done else reward + self.gamma * max(
            self.q[(next_state, a)] for a in ACTIONS
        )
        self.q[(state, action)] = current_q + self.alpha * (target - current_q)


def train(env: ClutchPossessionEnv, agent: QLearningAgent, num_episodes: int,
          eval_every: int = 200, eval_episodes: int = 200):
    """Trains the agent and periodically evaluates greedy win-rate, so we
    get a learning curve (win-rate vs training episodes) for the eval report."""
    learning_curve = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, episode)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        if episode % eval_every == 0:
            win_rate = evaluate_policy(env, agent, eval_episodes, greedy=True)
            learning_curve.append({"episode": episode, "win_rate": win_rate})

    return learning_curve


def evaluate_policy(env: ClutchPossessionEnv, agent: QLearningAgent,
                     num_episodes: int, greedy: bool = True) -> float:
    wins = 0
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, episode=10**9, greedy=greedy)  # force greedy/epsilon_end
            state, reward, done = env.step(action)
        if reward > 0:
            wins += 1
    return round(wins / num_episodes, 3)


def evaluate_random_baseline(env: ClutchPossessionEnv, num_episodes: int) -> float:
    wins = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = random.choice(ACTIONS)
            state, reward, done = env.step(action)
        if reward > 0:
            wins += 1
    return round(wins / num_episodes, 3)


def evaluate_fixed_baseline(env: ClutchPossessionEnv, num_episodes: int, fixed_action: int) -> float:
    """A naive fixed-strategy baseline, e.g. 'always shoot two-pointers'."""
    wins = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            state, reward, done = env.step(fixed_action)
        if reward > 0:
            wins += 1
    return round(wins / num_episodes, 3)

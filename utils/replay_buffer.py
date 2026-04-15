"""
utils/replay_buffer.py

Replay buffer implementations for the traffic RL project.

Two buffers are provided:

1. RolloutBuffer  — stores a single on-policy rollout (used by PPO / GRPO).
                    Cleared after every policy update.
                    Supports advantage computation via GAE.

2. ReplayBuffer   — fixed-size circular buffer for off-policy algorithms
                    (provided for completeness; not used by PPO/GRPO but
                    useful if you add DQN or SAC baselines later).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────
# On-policy rollout buffer (PPO / GRPO)
# ──────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores transitions collected during one on-policy rollout.
    Supports GAE (Generalised Advantage Estimation) for PPO.

    Usage:
        buf = RolloutBuffer(obs_dim=20, capacity=2048)
        buf.add(obs, action, log_prob, reward, value, done)
        ...
        advantages, returns = buf.compute_gae(last_value)
        # iterate:
        for batch in buf.get_batches(batch_size=64):
            ...
        buf.clear()
    """

    def __init__(self, obs_dim: int, capacity: int = 2048):
        self.obs_dim   = obs_dim
        self.capacity  = capacity
        self.clear()

    def clear(self):
        self.observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions      = np.zeros(self.capacity, dtype=np.int64)
        self.log_probs    = np.zeros(self.capacity, dtype=np.float32)
        self.rewards      = np.zeros(self.capacity, dtype=np.float32)
        self.values       = np.zeros(self.capacity, dtype=np.float32)
        self.dones        = np.zeros(self.capacity, dtype=np.float32)
        self.advantages   = np.zeros(self.capacity, dtype=np.float32)
        self.returns      = np.zeros(self.capacity, dtype=np.float32)
        self._ptr         = 0
        self._full        = False

    @property
    def size(self) -> int:
        return self.capacity if self._full else self._ptr

    def add(
        self,
        obs:      np.ndarray,
        action:   int,
        log_prob: float,
        reward:   float,
        value:    float,
        done:     bool,
    ):
        """Add one transition. Raises if buffer is full."""
        if self._ptr >= self.capacity:
            raise BufferError("RolloutBuffer is full. Call clear() before adding more data.")

        self.observations[self._ptr] = obs
        self.actions[self._ptr]      = action
        self.log_probs[self._ptr]    = log_prob
        self.rewards[self._ptr]      = reward
        self.values[self._ptr]       = value
        self.dones[self._ptr]        = float(done)
        self._ptr += 1

        if self._ptr == self.capacity:
            self._full = True

    def compute_gae(
        self,
        last_value: float = 0.0,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalised Advantage Estimates (GAE) and discounted returns.

        Args:
            last_value: Bootstrap value V(s_{T+1}). Use 0 if episode ended,
                        or V(s_{T+1}) from the critic if truncated.
            gamma:      Discount factor.
            gae_lambda: GAE lambda (trade-off between bias and variance).

        Returns:
            (advantages, returns) — both shape (size,)
        """
        n = self.size
        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value        = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value        = self.values[t + 1]

            delta        = (self.rewards[t]
                            + gamma * next_value * next_non_terminal
                            - self.values[t])
            last_gae     = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values[:n]
        self.advantages[:n] = advantages
        self.returns[:n]    = returns
        return advantages, returns

    def get_batches(self, batch_size: int = 64):
        """
        Yield randomised mini-batches of transitions.

        Each batch is a dict with keys:
          observations, actions, log_probs, advantages, returns
        """
        n       = self.size
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            idx = indices[start: start + batch_size]
            yield {
                "observations": self.observations[idx],
                "actions":      self.actions[idx],
                "log_probs":    self.log_probs[idx],
                "advantages":   self.advantages[idx],
                "returns":      self.returns[idx],
                "values":       self.values[idx],
            }

    def is_full(self) -> bool:
        return self._full or self._ptr >= self.capacity


# ──────────────────────────────────────────────────────────────
# Off-policy circular replay buffer (DQN / SAC baselines)
# ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Fixed-size circular buffer for off-policy algorithms.

    Stores (s, a, r, s', done) transitions. When full, oldest
    transitions are overwritten.

    Usage:
        buf = ReplayBuffer(obs_dim=20, capacity=50_000)
        buf.add(obs, action, reward, next_obs, done)
        if len(buf) >= batch_size:
            batch = buf.sample(batch_size)
    """

    def __init__(self, obs_dim: int, capacity: int = 50_000):
        self.obs_dim   = obs_dim
        self.capacity  = capacity
        self._obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions  = np.zeros(capacity, dtype=np.int64)
        self._rewards  = np.zeros(capacity, dtype=np.float32)
        self._dones    = np.zeros(capacity, dtype=np.float32)
        self._ptr      = 0
        self._size     = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ):
        idx = self._ptr % self.capacity
        self._obs[idx]      = obs
        self._next_obs[idx] = next_obs
        self._actions[idx]  = action
        self._rewards[idx]  = reward
        self._dones[idx]    = float(done)
        self._ptr           = (self._ptr + 1) % self.capacity
        self._size          = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        """
        Sample a random mini-batch.

        Returns:
            dict with keys: observations, actions, rewards, next_observations, dones
        """
        if self._size < batch_size:
            raise ValueError(
                f"Buffer has {self._size} transitions but batch_size={batch_size}."
            )
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "observations":      self._obs[idx],
            "actions":           self._actions[idx],
            "rewards":           self._rewards[idx],
            "next_observations": self._next_obs[idx],
            "dones":             self._dones[idx],
        }

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= batch_size

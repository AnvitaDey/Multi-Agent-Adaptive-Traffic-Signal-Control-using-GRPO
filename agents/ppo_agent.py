"""
PPO agent for single-intersection traffic signal control.
Uses stable-baselines3 for a reliable, tested PPO implementation.
Custom network architecture suited to traffic observations.
"""

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import numpy as np


class TrafficFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for traffic observations.
    
    Applies layer normalisation and a small MLP before the PPO actor/critic heads.
    This is important because traffic features have different scales and
    temporal structure.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        obs_dim = int(np.prod(observation_space.shape))

        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class MetricsCallback(BaseCallback):
    """Logs per-episode traffic metrics to a CSV file."""

    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self._episode_rewards: list[float] = []
        self._current_reward  = 0.0

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]

        # Episode ended
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]
            with open(self.log_path, "a") as f:
                f.write(
                    f"{self.num_timesteps},"
                    f"{self._current_reward:.4f},"
                    f"{info.get('queue', 0)},"
                    f"{info.get('wait', 0):.2f},"
                    f"{info.get('arrived', 0)}\n"
                )
            self._current_reward = 0.0

        return True


def build_ppo_agent(env: gym.Env, log_dir: str = "results/ppo") -> PPO:
    """
    Build a PPO agent with traffic-tuned hyperparameters.
    
    Key hyperparameter choices:
      - n_steps=2048: Collect 2048 transitions before each update.
                      Longer rollouts improve advantage estimates in traffic.
      - n_epochs=10:  Multiple SGD passes over the same rollout data.
      - ent_coef=0.01: Encourages exploration early in training.
      - clip_range=0.2: Standard PPO clipping.
      - gae_lambda=0.95: High lambda for lower-bias advantage estimates.
    """
    policy_kwargs = {
        "features_extractor_class": TrafficFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
    }

    agent = PPO(
        policy          = "MlpPolicy",
        env             = Monitor(env),
        learning_rate   = 3e-4,
        n_steps         = 2048,
        batch_size      = 256,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.02,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        policy_kwargs   = policy_kwargs,
        verbose         = 1,
        tensorboard_log = log_dir,
    )
    return agent

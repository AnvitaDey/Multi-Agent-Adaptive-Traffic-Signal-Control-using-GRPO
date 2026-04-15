"""
agents/grpo_agent.py

Group Relative Policy Optimization (GRPO) for multi-agent traffic control.

Key fixes:
  - TrafficPolicyNet: deeper network (256->256->128)
  - GRPOUpdater: n_epochs support, per-step advantage weighting
  - AgentRollout: per-step discounted returns
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field


class TrafficPolicyNet(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, obs):
        return torch.distributions.Categorical(logits=self.net(obs))

    def get_log_prob(self, obs, actions):
        return self.forward(obs).log_prob(actions)

    def get_action(self, obs):
        dist   = self.forward(obs)
        action = dist.sample()
        return action, dist.log_prob(action)


@dataclass
class AgentRollout:
    observations: list = field(default_factory=list)
    actions:      list = field(default_factory=list)
    log_probs:    list = field(default_factory=list)
    rewards:      list = field(default_factory=list)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()

    def compute_return(self, gamma: float = 0.99) -> float:
        G = 0.0
        for r in reversed(self.rewards):
            G = r + gamma * G
        return G

    def compute_discounted_returns(self, gamma: float = 0.99) -> list:
        returns, G = [], 0.0
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return returns


class GRPOUpdater:
    """
    GRPO with per-step group-relative advantages and multi-epoch updates.
    
    For each timestep t, normalises returns across all agents:
        A_i_t = (G_i_t - mean_t) / (std_t + eps)
    This gives dense gradient signal vs scalar episode-level advantage.
    """

    def __init__(
        self,
        agents:       list,
        optimizers:   list,
        clip_eps:     float = 0.2,
        gamma:        float = 0.99,
        entropy_coef: float = 0.01,
        n_epochs:     int   = 4,
    ):
        self.agents       = agents
        self.optimizers   = optimizers
        self.clip_eps     = clip_eps
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.n_epochs     = n_epochs

    def update(self, rollouts: list) -> dict:
        n_agents = len(rollouts)
        assert n_agents == len(self.agents)

        # Episode-level returns for logging
        ep_returns = np.array([r.compute_return(self.gamma) for r in rollouts])

        # Per-step discounted returns — shape (n_agents, T)
        all_step_returns = [r.compute_discounted_returns(self.gamma) for r in rollouts]
        min_len = min(len(r) for r in all_step_returns)

        step_returns = np.array([r[:min_len] for r in all_step_returns])

        # Group-relative advantage per timestep
        mu_t    = step_returns.mean(axis=0, keepdims=True)
        sigma_t = step_returns.std(axis=0,  keepdims=True) + 1e-8
        adv_matrix = (step_returns - mu_t) / sigma_t  # (n_agents, T)

        losses = []

        for i, (agent, rollout) in enumerate(zip(self.agents, rollouts)):
            if len(rollout.observations) == 0:
                continue

            T      = min_len
            obs_t  = torch.tensor(np.array(rollout.observations[:T]), dtype=torch.float32)
            act_t  = torch.tensor(np.array(rollout.actions[:T]),      dtype=torch.long)
            old_lp = torch.tensor(np.array(rollout.log_probs[:T]),    dtype=torch.float32)
            adv_t  = torch.tensor(adv_matrix[i],                      dtype=torch.float32)

            agent_losses = []

            for _ in range(self.n_epochs):
                new_lp = agent.get_log_prob(obs_t, act_t)
                ratio  = torch.exp(new_lp - old_lp.detach())

                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean()

                entropy = agent.forward(obs_t).entropy().mean()
                loss    = policy_loss - self.entropy_coef * entropy

                self.optimizers[i].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                self.optimizers[i].step()
                agent_losses.append(loss.item())

            losses.append(np.mean(agent_losses))

        return {
            "group_returns":     ep_returns.tolist(),
            "group_mean_return": float(ep_returns.mean()),
            "advantages":        adv_matrix.mean(axis=1).tolist(),
            "mean_loss":         float(np.mean(losses)) if losses else 0.0,
        }

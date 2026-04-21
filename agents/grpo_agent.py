"""
agents/grpo_agent.py

Group Relative Policy Optimization (GRPO) for multi-agent traffic control.

Improvements over base version:
  - ValueNet baseline → reduces advantage variance, especially on heavy traffic
  - LR scheduler support (cosine decay)
  - adv_matrix clamp kept as numpy, converted to tensor once (no accidental double-convert)
  - Entropy annealing hook exposed via set_entropy_coef()
  - get_action() returns detached tensors — safe for numpy conversion
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────
#  Policy network
# ─────────────────────────────────────────────────────────────

class TrafficPolicyNet(nn.Module):
    """
    Actor network.  256 → 256 → hidden → num_actions
    LayerNorm on input handles the mixed scales of queue/wait/phase features.
    """

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
        # Small init on last layer → near-uniform policy at start
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.net(obs))

    def get_log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(obs).log_prob(actions)

    def get_action(self, obs: torch.Tensor):
        """Returns (action_tensor, log_prob_tensor) — both detached from graph."""
        with torch.no_grad():
            dist   = self.forward(obs)
            action = dist.sample()
            lp     = dist.log_prob(action)
        return action, lp


# ─────────────────────────────────────────────────────────────
#  Value (baseline) network  — reduces GRPO advantage variance
# ─────────────────────────────────────────────────────────────

class TrafficValueNet(nn.Module):
    """
    Critic network — same architecture as actor but outputs scalar V(s).
    Used as a baseline to reduce advantage variance on heavy traffic.
    """

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)   # (B,)


# ─────────────────────────────────────────────────────────────
#  Rollout buffer
# ─────────────────────────────────────────────────────────────

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
        """Scalar episode return (for logging)."""
        G = 0.0
        for r in reversed(self.rewards):
            G = r + gamma * G
        return G

    def compute_discounted_returns(self, gamma: float = 0.99) -> list:
        """Per-step discounted returns G_t (for advantage computation)."""
        returns, G = [], 0.0
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return returns


# ─────────────────────────────────────────────────────────────
#  GRPO Updater
# ─────────────────────────────────────────────────────────────

class GRPOUpdater:
    """
    GRPO with:
      - Per-step group-relative advantages (dense signal)
      - Optional value-baseline subtraction (reduces variance on heavy traffic)
      - Multi-epoch PPO-style clipped updates
      - Advantage clamping to [-5, 5] (training stability)

    Advantage formula (per timestep t, across N agents):
        raw_G_i_t  = G_i_t  [- V(s_i_t)  if use_baseline]
        A_i_t      = (raw_G_i_t - mean_t(raw_G)) / (std_t(raw_G) + eps)
    """

    def __init__(
        self,
        agents:        list,               # list of TrafficPolicyNet
        optimizers:    list,               # list of Adam (policy)
        value_nets:    list  = None,       # list of TrafficValueNet  (optional)
        value_optims:  list  = None,       # list of Adam (critic)
        clip_eps:      float = 0.2,
        gamma:         float = 0.99,
        entropy_coef:  float = 0.01,
        n_epochs:      int   = 6,
        vf_coef:       float = 0.5,        # weight on critic loss
        adv_clip:      float = 5.0,
    ):
        self.agents       = agents
        self.optimizers   = optimizers
        self.value_nets   = value_nets   or []
        self.value_optims = value_optims or []
        self.clip_eps     = clip_eps
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.n_epochs     = n_epochs
        self.vf_coef      = vf_coef
        self.adv_clip     = adv_clip
        self.use_baseline = len(self.value_nets) == len(self.agents)

    def set_entropy_coef(self, coef: float):
        """Allow caller to anneal entropy coefficient during training."""
        self.entropy_coef = coef

    def update(self, rollouts: list) -> dict:
        n_agents = len(rollouts)
        assert n_agents == len(self.agents)

        # ── Episode-level returns (logging only) ──────────────────
        ep_returns = np.array([r.compute_return(self.gamma) for r in rollouts])

        # ── Per-step discounted returns  shape: (n_agents, T) ─────
        all_step_returns = [r.compute_discounted_returns(self.gamma) for r in rollouts]
        min_len          = min(len(r) for r in all_step_returns)
        step_returns     = np.array([r[:min_len] for r in all_step_returns])  # (N, T)
        
        ret_mean = step_returns.mean()
        ret_std  = step_returns.std() + 1e-8
        step_returns = (step_returns - ret_mean) / ret_std

        # ── Optional value baseline subtraction ───────────────────
        if self.use_baseline:
            baselines = []
            for i, vnet in enumerate(self.value_nets):
                obs_t = torch.tensor(
                    np.array(rollouts[i].observations[:min_len]), dtype=torch.float32
                )
                with torch.no_grad():
                    v = vnet(obs_t).cpu().numpy()
                baselines.append(v)
            baseline_arr = np.array(baselines)            # (N, T)
            raw_adv      = step_returns - baseline_arr
        else:
            raw_adv = step_returns                        # (N, T)

        # ── Group-relative normalisation ──────────────────────────
        mu_t       = raw_adv.mean(axis=0, keepdims=True)
        sigma_t    = raw_adv.std(axis=0,  keepdims=True) + 1e-8
        adv_matrix = np.clip((raw_adv - mu_t) / sigma_t,
                             -self.adv_clip, self.adv_clip)   # (N, T)  numpy

        losses, vf_losses = [], []

        for i, (agent, rollout) in enumerate(zip(self.agents, rollouts)):
            if len(rollout.observations) == 0:
                continue

            T      = min_len
            obs_t  = torch.tensor(np.array(rollout.observations[:T]), dtype=torch.float32)
            act_t  = torch.tensor(np.array(rollout.actions[:T]),      dtype=torch.long)
            old_lp = torch.tensor(np.array(rollout.log_probs[:T]),    dtype=torch.float32)
            adv_t  = torch.tensor(adv_matrix[i],                      dtype=torch.float32)
            ret_t  = torch.tensor(step_returns[i],                    dtype=torch.float32)

            agent_losses = []

            for _ in range(self.n_epochs):
                # ── Policy loss (clipped PPO surrogate) ───────────
                new_lp = agent.get_log_prob(obs_t, act_t)
                ratio  = torch.exp(new_lp - old_lp.detach())

                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Entropy bonus ─────────────────────────────────
                dist    = agent.forward(obs_t)
                entropy = dist.entropy().mean()
                loss    = policy_loss - self.entropy_coef * entropy

                self.optimizers[i].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                self.optimizers[i].step()
                agent_losses.append(loss.item())

            losses.append(float(np.mean(agent_losses)))

            # ── Critic update (independent of policy epochs) ──────
            if self.use_baseline:
                vnet   = self.value_nets[i]
                voptim = self.value_optims[i]
                for _ in range(self.n_epochs):
                    v_pred = vnet(obs_t)
                    vf_loss = nn.functional.mse_loss(v_pred, ret_t)
                    voptim.zero_grad()
                    (self.vf_coef * vf_loss).backward()
                    nn.utils.clip_grad_norm_(vnet.parameters(), 0.5)
                    voptim.step()
                    vf_losses.append(vf_loss.item())

        return {
            "group_returns":     ep_returns.tolist(),
            "group_mean_return": float(ep_returns.mean()),
            "advantages":        adv_matrix.mean(axis=1).tolist(),  # numpy, safe
            "mean_loss":         float(np.mean(losses))     if losses    else 0.0,
            "mean_vf_loss":      float(np.mean(vf_losses))  if vf_losses else 0.0,
        }
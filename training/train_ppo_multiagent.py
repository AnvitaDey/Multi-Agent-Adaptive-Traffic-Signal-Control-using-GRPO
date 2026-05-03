"""
training/train_ppo_multiagent.py  (v4 — fixed hyperparams, LR decay, clean logging)
"""

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import csv, yaml
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback
)
from envs.multi_traffic_env import MultiTrafficEnv
from utils.sumo_utils import patch_sumocfg_routes


# ─────────────────────────────────────────────
# Linear LR schedule: starts at lr, ends at 0
# ─────────────────────────────────────────────
def linear_schedule(initial_lr: float):
    def schedule(progress: float) -> float:
        # progress goes 1.0 → 0.0 as training proceeds
        return initial_lr * progress
    return schedule


# ─────────────────────────────────────────────
# CSV logger callback — writes per-rollout stats
# ─────────────────────────────────────────────
class CSVLoggerCallback(BaseCallback):
    def __init__(self, csv_path: str, tls_id: str, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.tls_id   = tls_id
        self._write_header = not os.path.exists(csv_path)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Called after each rollout — log key metrics."""
        stats = self.logger.name_to_value
        ep_rew  = stats.get("rollout/ep_rew_mean", float("nan"))
        ep_len  = stats.get("rollout/ep_len_mean", float("nan"))
        val_los = stats.get("train/value_loss",    float("nan"))
        pol_los = stats.get("train/policy_gradient_loss", float("nan"))
        entropy = stats.get("train/entropy_loss",  float("nan"))

        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if self._write_header:
                w.writerow([
                    "tls_id", "timesteps",
                    "ep_rew_mean", "ep_len_mean",
                    "value_loss", "policy_gradient_loss", "entropy_loss"
                ])
                self._write_header = False
            w.writerow([
                self.tls_id,
                self.num_timesteps,
                round(ep_rew,  4) if ep_rew  == ep_rew  else "",
                round(ep_len,  1) if ep_len  == ep_len  else "",
                round(val_los, 6) if val_los == val_los else "",
                round(pol_los, 6) if pol_los == pol_los else "",
                round(entropy, 6) if entropy == entropy else "",
            ])


def load_config(path="config/experiment_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def get_mixed_cfgs(cfg):
    os.makedirs("results/tmp", exist_ok=True)
    cfgs = []
    for density in ["light", "moderate", "heavy"]:
        out = f"results/tmp/simulation_{density}.sumocfg"
        patch_sumocfg_routes(
            cfg["sumo"]["base_cfg"],
            cfg["sumo"]["routes"][density],
            out
        )
        cfgs.append(out)
    return cfgs


class SingleAgentWrapper(gym.Env):
    """Wraps MultiTrafficEnv for a single agent."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, mixed_cfgs, tls_id, all_tls_ids, max_steps=2000):
        super().__init__()
        self.tls_id      = tls_id
        self.all_tls_ids = all_tls_ids

        self._env = MultiTrafficEnv(
            sumo_cfg=mixed_cfgs,
            tls_ids=all_tls_ids,
            max_steps=max_steps
        )

        obs_dim = self._env.obs_dim(tls_id)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        obs_dict = self._env.reset()
        self._episode_reward = 0.0
        self._episode_length = 0
        return obs_dict[self.tls_id], {}

    def step(self, action):
        action_dict = {tls: 0 for tls in self.all_tls_ids}
        action_dict[self.tls_id] = int(action)

        obs_dict, rew_dict, done_dict, info_dict = self._env.step(action_dict)

        rew  = rew_dict[self.tls_id]
        done = done_dict["__all__"]
        info = info_dict[self.tls_id]

        self._episode_reward += rew
        self._episode_length += 1

        if done:
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._episode_length,
            }

        return obs_dict[self.tls_id], rew, done, False, info

    def close(self):
        self._env.close()


def parse_steps_from_ckpt(ckpt_path: str) -> int:
    fname = os.path.basename(ckpt_path).replace(".zip", "")
    parts = fname.split("_")
    try:
        return int(parts[-2])
    except (IndexError, ValueError):
        return 0


def main():
    cfg        = load_config()
    tls_ids    = cfg["sumo"]["tls_ids"]
    max_steps  = cfg["episode"]["max_decision_steps"]

    log_dir   = "results/ppo_multiagent"
    final_dir = os.path.join(log_dir, "final")
    tb_dir    = os.path.join(log_dir, "tb_logs")

    # ── Key training settings ──────────────────────────────
    # n_steps=4096 → 2 full episodes per rollout → stable advantages
    # lr=3e-4 with linear decay → starts aggressive, ends conservative
    # ent_coef=0.05 → strong early exploration, decays naturally via schedule
    # batch_size=256 → larger mini-batches → smoother gradient updates
    total_ts   = 500_000
    ckpt_every = 100_000   # checkpoint every 100k steps

    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(tb_dir,    exist_ok=True)

    mixed_cfgs = get_mixed_cfgs(cfg)

    print("=" * 60)
    print("  PPO Multi-Agent — mixed density, v4")
    print(f"  Timesteps/agent : {total_ts:,}")
    print(f"  Agents          : {tls_ids}")
    print("=" * 60)

    done_agents = [
        t for t in tls_ids
        if os.path.exists(os.path.join(final_dir, f"{t}.zip"))
    ]
    todo_agents = [t for t in tls_ids if t not in done_agents]

    if done_agents:
        print(f"\n  Already complete: {done_agents}")
    print(f"  To train: {todo_agents}\n")

    if not todo_agents:
        print("  All agents already trained.")
        return

    log_file = os.path.join(log_dir, "training_log.csv")

    for tls in todo_agents:
        print(f"\n{'─'*60}")
        print(f"  Agent: {tls}")
        print(f"{'─'*60}")

        ckpt_dir = os.path.join(log_dir, f"checkpoints_{tls}")
        os.makedirs(ckpt_dir, exist_ok=True)

        train_env = Monitor(
            SingleAgentWrapper(mixed_cfgs, tls, tls_ids, max_steps),
            filename=os.path.join(log_dir, f"monitor_{tls}.csv"),
        )

        # ── Resume or fresh start ──────────────────────────
        ckpts = sorted([
            f for f in os.listdir(ckpt_dir) if f.endswith(".zip")
        ])

        if ckpts:
            latest_ckpt = os.path.join(ckpt_dir, ckpts[-1])
            steps_done  = parse_steps_from_ckpt(latest_ckpt)
            remaining   = max(0, total_ts - steps_done)

            print(f"  Resuming: {ckpts[-1]}  ({steps_done:,} done, {remaining:,} left)")
            agent = PPO.load(latest_ckpt, env=train_env)

        else:
            print("  Starting fresh")
            remaining = total_ts

            agent = PPO(
                policy     = "MlpPolicy",
                env        = train_env,

                # ── LR schedule: 3e-4 → 0 over training ──
                learning_rate = linear_schedule(3e-4),

                # ── Rollout: 2 full episodes before each update ──
                # max_steps=2000 so 4096 ≈ 2 episodes → stable advantages
                n_steps    = 4096,
                batch_size = 256,
                n_epochs   = 10,

                # ── Discount ──
                gamma      = 0.99,
                gae_lambda = 0.95,

                # ── PPO clip ──
                clip_range = 0.2,

                # ── Entropy: high to force exploration early ──
                # 0.05 >> 0.005 from before — this was killing learning
                ent_coef   = 0.05,

                vf_coef       = 0.5,
                max_grad_norm = 0.5,

                # ── Network: single hidden layer, not too deep ──
                # LayerNorm input → helps with varied reward scales
                policy_kwargs = dict(
                    net_arch     = [256, 256],
                    activation_fn= __import__("torch").nn.ReLU,
                ),

                verbose         = 1,
                tensorboard_log = tb_dir,
            )

        if remaining > 0:
            callbacks = [
                CheckpointCallback(
                    save_freq   = ckpt_every,
                    save_path   = ckpt_dir,
                    name_prefix = f"ppo_{tls}",
                    verbose     = 0,
                ),
                CSVLoggerCallback(
                    csv_path = log_file,
                    tls_id   = tls,
                ),
            ]

            agent.learn(
                total_timesteps    = remaining,
                callback           = callbacks,
                progress_bar       = True,
                reset_num_timesteps= False,
                tb_log_name        = f"PPO_{tls}",
            )
        else:
            print("  Already at target — skipping learn()")

        agent.save(os.path.join(final_dir, tls))
        print(f"  Saved: {final_dir}/{tls}.zip")
        train_env.close()

    print(f"\n{'='*60}")
    print(f"  Done. Models → {final_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
"""
training/train_ppo_multiagent.py  (v3 — mixed-density, resumable)

Each agent trains on randomly sampled light/moderate/heavy traffic each episode.
Skips agents that already have a final model.
500k timesteps per agent.
Saves to results/ppo_multiagent/
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
import csv, yaml, random
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.multi_traffic_env import MultiTrafficEnv
from utils.sumo_utils import patch_sumocfg_routes

def load_config(path="config/experiment_config.yaml"):
    with open(path) as f: return yaml.safe_load(f)

def get_mixed_cfgs(cfg):
    os.makedirs("results/tmp", exist_ok=True)
    cfgs = []
    for density in ["light", "moderate", "heavy"]:
        out = f"results/tmp/simulation_{density}.sumocfg"
        patch_sumocfg_routes(cfg["sumo"]["base_cfg"],
                             cfg["sumo"]["routes"][density], out)
        cfgs.append(out)
    return cfgs

class SingleAgentWrapper(gym.Env):
    """Single-agent wrapper — randomly picks light/moderate/heavy each episode."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, mixed_cfgs, tls_id, all_tls_ids, max_steps=2000):
        super().__init__()
        self.tls_id      = tls_id
        self.all_tls_ids = all_tls_ids
        # Pass list of configs — MultiTrafficEnv picks randomly each reset()
        self._env = MultiTrafficEnv(sumo_cfg=mixed_cfgs, tls_ids=all_tls_ids,
                                    max_steps=max_steps)
        obs_dim = self._env.obs_dim(tls_id)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        obs_dict = self._env.reset()
        return obs_dict[self.tls_id], {}

    def step(self, action):
        action_dict = {tls: 0 for tls in self.all_tls_ids}
        action_dict[self.tls_id] = int(action)
        obs_dict, rew_dict, done_dict, info_dict = self._env.step(action_dict)
        return (obs_dict[self.tls_id], rew_dict[self.tls_id],
                done_dict["__all__"], False, info_dict[self.tls_id])

    def close(self):
        self._env.close()

def main():
    cfg        = load_config()
    tls_ids    = cfg["sumo"]["tls_ids"]
    max_steps  = cfg["episode"]["max_decision_steps"]
    hidden_dim = cfg["ppo"]["hidden_dim"]
    log_dir    = "results/ppo_multiagent"
    final_dir  = os.path.join(log_dir, "final")
    total_ts   = 500_000
    ckpt_every = 100_000

    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    mixed_cfgs = get_mixed_cfgs(cfg)

    print("=" * 60)
    print("  STAGE 2b — Independent Multi-Agent PPO (mixed-density, resumable)")
    print(f"  Training on: light + moderate + heavy (random each episode)")
    print(f"  Timesteps/agent: {total_ts:,}")
    print("=" * 60)

    done_agents = [t for t in tls_ids
                   if os.path.exists(os.path.join(final_dir, f"{t}.zip"))]
    todo_agents = [t for t in tls_ids if t not in done_agents]

    if done_agents:
        print(f"\n  Already complete (skipping): {done_agents}")
    print(f"  To train: {todo_agents}\n")

    if not todo_agents:
        print("  All agents already trained.")
        return

    log_file     = os.path.join(log_dir, "training_log.csv")
    write_header = not os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        if write_header:
            csv.writer(f).writerow(["tls_id","timesteps","ep_rew_mean",
                                     "ep_len_mean","value_loss"])

    for tls in todo_agents:
        print(f"\n{'─'*60}")
        print(f"  Training agent: {tls}  (mixed light/moderate/heavy)")
        print(f"{'─'*60}")

        ckpt_dir = os.path.join(log_dir, f"checkpoints_{tls}")
        os.makedirs(ckpt_dir, exist_ok=True)

        train_env = Monitor(
            SingleAgentWrapper(mixed_cfgs, tls, tls_ids, max_steps),
            filename=os.path.join(log_dir, f"monitor_{tls}.csv"),
        )

        agent = PPO(
            policy="MlpPolicy", env=train_env,
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, hidden_dim]),
            verbose=1,
        )

        ckpt_cb = CheckpointCallback(
            save_freq=ckpt_every, save_path=ckpt_dir,
            name_prefix=f"ppo_{tls}", verbose=0,
        )

        agent.learn(total_timesteps=total_ts, callback=[ckpt_cb], progress_bar=True)

        agent.save(os.path.join(final_dir, tls))
        print(f"  --> Saved: {final_dir}/{tls}.zip")
        train_env.close()

        try:
            stats = agent.logger.name_to_value
            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    tls, total_ts,
                    round(stats.get("rollout/ep_rew_mean", 0), 4),
                    round(stats.get("rollout/ep_len_mean", 0), 1),
                    round(stats.get("train/value_loss", 0), 6),
                ])
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"  All agents trained on mixed traffic.")
    print(f"  Final models → {final_dir}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

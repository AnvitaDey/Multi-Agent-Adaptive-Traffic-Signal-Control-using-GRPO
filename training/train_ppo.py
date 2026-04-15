"""
training/train_ppo.py

Stage 2 — PPO single-agent training.
Controls one intersection (B1, centre) with RL.
All other intersections run fixed-time control.
"""

import os
import sys

# Make project root importable regardless of where script is run from
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.traffic_env import TrafficEnv
from agents.ppo_agent  import build_ppo_agent, MetricsCallback


def load_config(path: str = "config/experiment_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    sumo_cfg       = cfg["sumo"]["base_cfg"]
    tls_ids        = cfg["sumo"]["tls_ids"]
    controlled_tls = cfg["sumo"]["controlled_tls"]
    max_steps      = cfg["episode"]["max_decision_steps"]
    log_dir        = cfg["ppo"]["log_dir"]
    total_timesteps= cfg["ppo"]["total_timesteps"]

    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("  STAGE 2 — PPO Single-Agent Training")
    print(f"  Controlled intersection : {controlled_tls}")
    print(f"  Fixed-time intersections: {[t for t in tls_ids if t != controlled_tls]}")
    print(f"  Total timesteps         : {total_timesteps}")
    print(f"  Log directory           : {log_dir}")
    print("=" * 60)

    # Training environment
    train_env = Monitor(
        TrafficEnv(
            sumo_cfg       = sumo_cfg,
            controlled_tls = controlled_tls,
            all_tls_ids    = tls_ids,
            max_steps      = max_steps,
            use_gui        = False,
        ),
        filename=os.path.join(log_dir, "monitor.csv"),
    )

    # Evaluation environment (separate instance)
    eval_env = Monitor(
        TrafficEnv(
            sumo_cfg       = sumo_cfg,
            controlled_tls = controlled_tls,
            all_tls_ids    = tls_ids,
            max_steps      = max_steps,
            use_gui        = False,
        )
    )

    # Build agent
    agent = build_ppo_agent(train_env, log_dir=log_dir)

    # Callbacks
    metrics_cb = MetricsCallback(
        log_path=os.path.join(log_dir, "episode_metrics.csv")
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=cfg["ppo"]["checkpoint_every"],
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="ppo_traffic",
        verbose=1,
    )

    # Train
    print("\nStarting training...\n")
    agent.learn(
        total_timesteps=total_timesteps,
        callback=[metrics_cb, eval_cb, checkpoint_cb],
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(log_dir, "final_model")
    agent.save(final_path)
    print(f"\nTraining complete. Final model saved → {final_path}")
    print(f"Best model saved  → {os.path.join(log_dir, 'best_model')}")
    print(f"Episode metrics   → {os.path.join(log_dir, 'episode_metrics.csv')}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
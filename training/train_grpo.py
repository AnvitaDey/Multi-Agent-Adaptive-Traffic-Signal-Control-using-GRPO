"""
training/train_grpo.py  (v3 — mixed-density training)

Each episode randomly selects light/moderate/heavy traffic.
The agent learns to adapt to all conditions in one model.
Saves to results/grpo_mixed/
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
import csv, yaml
import numpy as np
import torch
from tqdm import tqdm
from envs.multi_traffic_env import MultiTrafficEnv
from agents.grpo_agent import TrafficPolicyNet, AgentRollout, GRPOUpdater
from utils.sumo_utils import patch_sumocfg_routes

def load_config(path="config/experiment_config.yaml"):
    with open(path) as f: return yaml.safe_load(f)

def get_mixed_cfgs(cfg):
    """Return patched sumocfg paths for all 3 densities."""
    os.makedirs("results/tmp", exist_ok=True)
    cfgs = []
    for density in ["light", "moderate", "heavy"]:
        out = f"results/tmp/simulation_{density}.sumocfg"
        patch_sumocfg_routes(cfg["sumo"]["base_cfg"],
                             cfg["sumo"]["routes"][density], out)
        cfgs.append(out)
    return cfgs

def main():
    cfg          = load_config()
    tls_ids      = cfg["sumo"]["tls_ids"]
    hidden_dim   = cfg["grpo"]["hidden_dim"]
    max_steps    = cfg["episode"]["max_decision_steps"]
    num_episodes = 4000
    lr           = 1e-4
    gamma        = 0.99
    clip_eps     = 0.2
    entropy_coef = 0.01
    n_epochs     = 4
    ckpt_every   = 100
    log_dir      = "results/grpo_mixed"
    os.makedirs(log_dir, exist_ok=True)

    # All 3 density configs — randomly sampled each episode
    mixed_cfgs = get_mixed_cfgs(cfg)

    print("=" * 60)
    print("  STAGE 3 — GRPO Mixed-Density Training")
    print(f"  Episodes : {num_episodes}  |  LR: {lr}  |  Epochs/ep: {n_epochs}")
    print(f"  Training on: light + moderate + heavy (random each episode)")
    print("=" * 60)

    env      = MultiTrafficEnv(sumo_cfg=mixed_cfgs, tls_ids=tls_ids, max_steps=max_steps)
    obs_dict = env.reset()
    tls_list = list(tls_ids)
    agents, optimizers = {}, {}

    for tls in tls_list:
        obs_dim         = len(obs_dict[tls])
        agents[tls]     = TrafficPolicyNet(obs_dim=obs_dim, num_actions=2, hidden=hidden_dim)
        optimizers[tls] = torch.optim.Adam(agents[tls].parameters(), lr=lr)
        print(f"  Agent {tls}: obs_dim={obs_dim}")
    
    # 🔁 RESUME LOGIC
    start_ep = 0

    ckpts = sorted([
        d for d in os.listdir(log_dir)
        if d.startswith("checkpoint_ep")
    ])

    if ckpts:
        latest = ckpts[-1]
        ckpt_path = os.path.join(log_dir, latest, "checkpoint.pt")

        print(f"🔁 Resuming from {latest}")

        checkpoint = torch.load(ckpt_path)

        start_ep = checkpoint["episode"] + 1

        for tls in tls_list:
            agents[tls].load_state_dict(checkpoint["agents"][tls])
            optimizers[tls].load_state_dict(checkpoint["optimizers"][tls])

    updater = GRPOUpdater(
        agents=[agents[t] for t in tls_list],
        optimizers=[optimizers[t] for t in tls_list],
        clip_eps=clip_eps, gamma=gamma,
        entropy_coef=entropy_coef, n_epochs=n_epochs,
    )

    log_file = os.path.join(log_dir, "training_log.csv")
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["episode","density","mean_return","group_std",
                                 "mean_queue","mean_wait","throughput","mean_loss"])

    print("\nStarting mixed-density GRPO training...\n")

    for ep in tqdm(range(start_ep, um_episodes), desc="GRPO", unit="ep", colour="green"):
        obs_dict  = env.reset()
        # Track which density was used this episode
        current_density = ["light","moderate","heavy"][
            mixed_cfgs.index(env._current_cfg)
        ] if env._current_cfg in mixed_cfgs else "unknown"

        rollouts  = {tls: AgentRollout() for tls in tls_list}
        done      = False
        ep_queues, ep_waits, ep_throughput = [], [], 0

        while not done:
            action_dict = {}
            for tls in tls_list:
                obs_t = torch.tensor(obs_dict[tls], dtype=torch.float32).unsqueeze(0)
                action, lp = agents[tls].get_action(obs_t)
                action_dict[tls] = action.item()
                rollouts[tls].observations.append(obs_dict[tls].copy())
                rollouts[tls].actions.append(action.item())
                rollouts[tls].log_probs.append(lp.item())

            next_obs, reward_dict, done_dict, info_dict = env.step(action_dict)
            done = done_dict["__all__"]
            for tls in tls_list:
                rollouts[tls].rewards.append(reward_dict[tls])
                ep_queues.append(info_dict[tls]["queue"])
                ep_waits.append(info_dict[tls]["wait"])
            ep_throughput += info_dict[tls_list[0]]["arrived"]
            obs_dict = next_obs

        stats       = updater.update([rollouts[t] for t in tls_list])
        mean_return = float(np.mean(stats["group_returns"]))
        group_std   = float(np.std(stats["group_returns"]))
        mean_queue  = float(np.mean(ep_queues)) if ep_queues else 0.0
        mean_wait   = float(np.mean(ep_waits))  if ep_waits  else 0.0

        print(f"Ep {ep:4d}/{num_episodes} [{current_density:8s}] | "
              f"Return: {mean_return:7.2f} | Std: {group_std:5.2f} | "
              f"Queue: {mean_queue:5.2f} | Wait: {mean_wait:6.2f}s | "
              f"TP: {ep_throughput:5d} | Loss: {stats['mean_loss']:.4f}")

        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([ep, current_density,
                                     f"{mean_return:.4f}", f"{group_std:.4f}",
                                     f"{mean_queue:.4f}", f"{mean_wait:.4f}",
                                     ep_throughput, f"{stats['mean_loss']:.6f}"])

        if ep>0 and ep % ckpt_every == 0:
            ckpt_dir = os.path.join(log_dir, f"checkpoint_ep{ep}")
            os.makedirs(ckpt_dir, exist_ok=True)
            
            checkpoint = {
                "episode": ep,
                "agents": {tls: agents[tls].state_dict() for tls in tls_list},
                "optimizers": {tls: optimizers[tls].state_dict() for tls in tls_list},
            }
            torch.save(checkpoint, os.path.join(ckpt_dir, "checkpoint.pt"))
            print(f"  --> Checkpoint: {ckpt_dir}")

    final_dir = os.path.join(log_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    for tls in tls_list:
        torch.save(agents[tls].state_dict(), os.path.join(final_dir, f"{tls}.pt"))

    print(f"\nDone. Final → {final_dir}/  |  Log → {log_file}")
    env.close()

if __name__ == "__main__":
    main()

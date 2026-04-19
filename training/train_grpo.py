"""
training/train_grpo.py  (v4 — mixed-density, TRUE resume support)

Each episode randomly selects light/moderate/heavy traffic.
The agent learns to adapt to all conditions in one model.

Key improvements over v3:
  - TRUE resume: single checkpoint.pt with everything inside
  - CSV log preserved on resume (append mode, no wipe)
  - ValueNet baseline per agent (reduces variance, better heavy-traffic perf)
  - Cosine LR annealing (lr 3e-4 → 1e-5 over training)
  - Entropy annealing (0.05 → 0.005 over training)
  - 6000 episodes, lr=3e-4, n_epochs=6
  - Checkpoint every 100 episodes
  - tqdm.write() used inside loop so progress bar stays visible
Saves to results/grpo_mixed/
"""

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import csv, yaml
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_sched
from tqdm import tqdm
from envs.multi_traffic_env import MultiTrafficEnv
from agents.grpo_agent import TrafficPolicyNet, TrafficValueNet, AgentRollout, GRPOUpdater
from utils.sumo_utils import patch_sumocfg_routes


def load_config(path="config/experiment_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def get_mixed_cfgs(cfg):
    os.makedirs("results/tmp", exist_ok=True)
    cfgs = []
    for density in ["light", "moderate", "heavy"]:
        out = f"results/tmp/simulation_{density}.sumocfg"
        patch_sumocfg_routes(cfg["sumo"]["base_cfg"],
                             cfg["sumo"]["routes"][density], out)
        cfgs.append(out)
    return cfgs


# ── Checkpoint helpers ─────────────────────────────────────────────────────

def save_checkpoint(ckpt_path, ep, tls_list,
                    agents, optimizers,
                    value_nets, value_optims,
                    schedulers, v_schedulers):
    """Single file — all states in one dict. Either fully saves or doesn't."""
    checkpoint = {
        "episode":      ep,
        "agents":       {tls: agents[tls].state_dict()       for tls in tls_list},
        "optimizers":   {tls: optimizers[tls].state_dict()   for tls in tls_list},
        "value_nets":   {tls: value_nets[tls].state_dict()   for tls in tls_list},
        "value_optims": {tls: value_optims[tls].state_dict() for tls in tls_list},
        "schedulers":   {tls: schedulers[tls].state_dict()   for tls in tls_list},
        "v_schedulers": {tls: v_schedulers[tls].state_dict() for tls in tls_list},
    }
    torch.save(checkpoint, ckpt_path)


def load_checkpoint(ckpt_path, tls_list,
                    agents, optimizers,
                    value_nets, value_optims,
                    schedulers, v_schedulers):
    """Load everything from single checkpoint file. Returns saved episode."""
    checkpoint = torch.load(ckpt_path, weights_only=False)

    for tls in tls_list:
        agents[tls].load_state_dict(checkpoint["agents"][tls])
        optimizers[tls].load_state_dict(checkpoint["optimizers"][tls])
        value_nets[tls].load_state_dict(checkpoint["value_nets"][tls])
        value_optims[tls].load_state_dict(checkpoint["value_optims"][tls])
        schedulers[tls].load_state_dict(checkpoint["schedulers"][tls])
        v_schedulers[tls].load_state_dict(checkpoint["v_schedulers"][tls])

    return checkpoint["episode"]


def find_latest_checkpoint(log_dir):
    """
    Return path to the latest checkpoint.pt, or None.
    Looks for checkpoint_ep{N:05d}/checkpoint.pt — zero-padded so sort is correct.
    """
    entries = sorted([
        d for d in os.listdir(log_dir)
        if d.startswith("checkpoint_ep") and
           os.path.isfile(os.path.join(log_dir, d, "checkpoint.pt"))
    ])
    if not entries:
        return None
    return os.path.join(log_dir, entries[-1], "checkpoint.pt")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    cfg        = load_config()
    tls_ids    = cfg["sumo"]["tls_ids"]
    hidden_dim = cfg["grpo"]["hidden_dim"]
    max_steps  = cfg["episode"]["max_decision_steps"]

    # ── Hyperparameters ────────────────────────────────────────────
    num_episodes  = 6000
    lr            = 3e-4
    lr_min        = 1e-5
    gamma         = 0.99
    clip_eps      = 0.2
    entropy_start = 0.05
    entropy_end   = 0.005
    n_epochs      = 6
    ckpt_every    = 100
    log_dir       = "results/grpo_mixed"

    os.makedirs(log_dir, exist_ok=True)

    mixed_cfgs = get_mixed_cfgs(cfg)

    print("=" * 60)
    print("  STAGE 3 — GRPO Mixed-Density Training (resumable)")
    print(f"  Episodes : {num_episodes}  |  LR: {lr}→{lr_min}  |  Epochs/ep: {n_epochs}")
    print(f"  Training on: light + moderate + heavy (random each episode)")
    print("=" * 60)

    # ── Build env ──────────────────────────────────────────────────
    env      = MultiTrafficEnv(sumo_cfg=mixed_cfgs, tls_ids=tls_ids, max_steps=max_steps)
    obs_dict = env.reset()
    tls_list = list(tls_ids)

    # ── Build networks & optimizers ────────────────────────────────
    agents, optimizers       = {}, {}
    value_nets, value_optims = {}, {}

    for tls in tls_list:
        obs_dim = len(obs_dict[tls])
        agents[tls]       = TrafficPolicyNet(obs_dim=obs_dim, num_actions=2, hidden=hidden_dim)
        value_nets[tls]   = TrafficValueNet(obs_dim=obs_dim,  hidden=hidden_dim)
        optimizers[tls]   = torch.optim.Adam(agents[tls].parameters(),     lr=lr)
        value_optims[tls] = torch.optim.Adam(value_nets[tls].parameters(), lr=lr)
        print(f"  Agent {tls}: obs_dim={obs_dim}")

    # Cosine LR schedulers
    schedulers = {
        tls: lr_sched.CosineAnnealingLR(optimizers[tls],   T_max=num_episodes, eta_min=lr_min)
        for tls in tls_list
    }
    v_schedulers = {
        tls: lr_sched.CosineAnnealingLR(value_optims[tls], T_max=num_episodes, eta_min=lr_min)
        for tls in tls_list
    }

    # ── RESUME LOGIC ───────────────────────────────────────────────
    start_ep    = 0
    latest_ckpt = find_latest_checkpoint(log_dir)

    if latest_ckpt:
        print(f"\n  🔁 Resuming from: {latest_ckpt}")
        start_ep = load_checkpoint(
            latest_ckpt, tls_list,
            agents, optimizers,
            value_nets, value_optims,
            schedulers, v_schedulers,
        ) + 1
        print(f"     Continuing from episode {start_ep} / {num_episodes}")
    else:
        print("\n  🚀 No checkpoint found — starting fresh")

    # ── Build updater AFTER loading weights ───────────────────────
    updater = GRPOUpdater(
        agents       = [agents[t]       for t in tls_list],
        optimizers   = [optimizers[t]   for t in tls_list],
        value_nets   = [value_nets[t]   for t in tls_list],
        value_optims = [value_optims[t] for t in tls_list],
        clip_eps     = clip_eps,
        gamma        = gamma,
        entropy_coef = entropy_start,
        n_epochs     = n_epochs,
    )

    # ── CSV log — append on resume, write header only once ────────
    log_file     = os.path.join(log_dir, "training_log.csv")
    write_header = not os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        if write_header:
            csv.writer(f).writerow([
                "episode", "density", "mean_return", "group_std",
                "mean_queue", "mean_wait", "throughput",
                "mean_loss", "mean_vf_loss", "entropy_coef", "lr"
            ])

    print(f"\n  Starting from episode {start_ep}...\n")

    # ── Training loop ──────────────────────────────────────────────
    # tqdm bar stays at the bottom; all in-loop output uses tqdm.write()
    # so the bar never gets broken or pushed around.
    pbar = tqdm(range(start_ep, num_episodes), desc="GRPO",
                unit="ep", colour="green", dynamic_ncols=True)

    for ep in pbar:

        # ── Anneal entropy and read current LR ────────────────────
        progress     = ep / max(num_episodes - 1, 1)
        entropy_coef = entropy_start + (entropy_end - entropy_start) * progress
        updater.set_entropy_coef(entropy_coef)
        current_lr   = optimizers[tls_list[0]].param_groups[0]["lr"]

        # ── Episode rollout ────────────────────────────────────────
        obs_dict = env.reset()
        current_density = (
            ["light", "moderate", "heavy"][mixed_cfgs.index(env._current_cfg)]
            if env._current_cfg in mixed_cfgs else "unknown"
        )

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

        # ── GRPO update ────────────────────────────────────────────
        stats       = updater.update([rollouts[t] for t in tls_list])
        mean_return = float(np.mean(stats["group_returns"]))
        group_std   = float(np.std(stats["group_returns"]))
        mean_queue  = float(np.mean(ep_queues)) if ep_queues else 0.0
        mean_wait   = float(np.mean(ep_waits))  if ep_waits  else 0.0

        # ── Step LR schedulers ─────────────────────────────────────
        for tls in tls_list:
            schedulers[tls].step()
            v_schedulers[tls].step()

        # ── Update progress bar suffix ─────────────────────────────
        pbar.set_postfix({
            "density": current_density,
            "ret":     f"{mean_return:.1f}",
            "queue":   f"{mean_queue:.1f}",
            "loss":    f"{stats['mean_loss']:.4f}",
            "lr":      f"{current_lr:.1e}",
        })

        # ── Per-episode line (won't break the bar) ─────────────────
        tqdm.write(
            f"Ep {ep:4d}/{num_episodes} [{current_density:8s}] | "
            f"Return: {mean_return:7.2f} | Std: {group_std:5.2f} | "
            f"Queue: {mean_queue:5.2f} | Wait: {mean_wait:6.2f}s | "
            f"TP: {ep_throughput:5d} | Loss: {stats['mean_loss']:.4f} | "
            f"VF: {stats['mean_vf_loss']:.4f} | "
            f"Ent: {entropy_coef:.4f} | LR: {current_lr:.2e}"
        )

        # ── CSV logging ────────────────────────────────────────────
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([
                ep, current_density,
                f"{mean_return:.4f}",  f"{group_std:.4f}",
                f"{mean_queue:.4f}",   f"{mean_wait:.4f}",
                ep_throughput,
                f"{stats['mean_loss']:.6f}",
                f"{stats['mean_vf_loss']:.6f}",
                f"{entropy_coef:.6f}", f"{current_lr:.2e}",
            ])

        # ── Checkpoint (skip ep 0 — no training done yet) ─────────
        if ep > 0 and ep % ckpt_every == 0:
            ckpt_dir  = os.path.join(log_dir, f"checkpoint_ep{ep:05d}")
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_checkpoint(
                ckpt_path, ep, tls_list,
                agents, optimizers,
                value_nets, value_optims,
                schedulers, v_schedulers,
            )
            tqdm.write(f"  --> Checkpoint saved: {ckpt_dir}/checkpoint.pt")

    pbar.close()

    # ── Final save ─────────────────────────────────────────────────
    final_dir = os.path.join(log_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    for tls in tls_list:
        torch.save(agents[tls].state_dict(),
                   os.path.join(final_dir, f"{tls}.pt"))
        torch.save(value_nets[tls].state_dict(),
                   os.path.join(final_dir, f"{tls}_value.pt"))

    print(f"\nDone. Final → {final_dir}/  |  Log → {log_file}")
    env.close()


if __name__ == "__main__":
    main()
"""
experiments/run_experiments.py

Master experiment runner — PPO single-agent removed, PPO multiagent added.

Stages:
  fixed_time    — rule-based, all 9 intersections
  actuated      — rule-based, all 9 intersections
  ppo_multiagent — independent PPO, all 9 intersections (fair GRPO baseline)
  grpo          — group-relative PPO, all 9 intersections
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from controllers.fixed_time import run_fixed_time_baseline
from controllers.actuated   import run_actuated_baseline
from evaluation.metrics     import (
    MultiRunMetrics,
    build_comparison_table, save_comparison_csv,
)
from utils.sumo_utils import patch_sumocfg_routes


def load_config(path: str = "config/experiment_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_sumo_cfg_for_density(cfg: dict, density: str) -> str:
    base_cfg   = cfg["sumo"]["base_cfg"]
    route_file = cfg["sumo"]["routes"].get(density)
    if not route_file:
        raise ValueError(f"No route file for density '{density}'")
    tmp_dir    = "results/tmp"
    output_cfg = os.path.join(tmp_dir, f"simulation_{density}.sumocfg")
    os.makedirs(tmp_dir, exist_ok=True)
    patch_sumocfg_routes(base_cfg, route_file, output_cfg)
    return output_cfg


# ──────────────────────────────────────────────────────────────
# Stage 1: Fixed-Time Baseline
# ──────────────────────────────────────────────────────────────

def run_stage_fixed_time(cfg: dict, densities: list, n_runs: int) -> list:
    print("\n" + "=" * 60)
    print("  STAGE 1 — Fixed-Time Baseline")
    print("=" * 60)

    tls_ids  = cfg["sumo"]["tls_ids"]
    green_t  = cfg["fixed_time"]["green_time"]
    yellow_t = cfg["fixed_time"]["yellow_time"]
    results  = []

    for density in densities:
        tracker  = MultiRunMetrics()
        sumo_cfg = get_sumo_cfg_for_density(cfg, density)
        for run in range(n_runs):
            print(f"  [fixed_time | {density} | run {run+1}/{n_runs}]", end=" ", flush=True)
            m = run_fixed_time_baseline(
                sumo_cfg=sumo_cfg, tls_ids=tls_ids,
                green_time=green_t, yellow_time=yellow_t,
            )
            tracker.add(m)
            print(f"queue={m['avg_queue_length']:.1f}  wait={m['avg_waiting_time']:.1f}s  tp={m['throughput']}")
        summary = tracker.summarise()
        results.append({"controller": "fixed_time", "density": density, **summary})
    return results


# ──────────────────────────────────────────────────────────────
# Stage 1b: Actuated Baseline
# ──────────────────────────────────────────────────────────────

def run_stage_actuated(cfg: dict, densities: list, n_runs: int) -> list:
    print("\n" + "=" * 60)
    print("  STAGE 1b — Actuated Baseline")
    print("=" * 60)

    tls_ids = cfg["sumo"]["tls_ids"]
    results = []

    for density in densities:
        tracker  = MultiRunMetrics()
        sumo_cfg = get_sumo_cfg_for_density(cfg, density)
        for run in range(n_runs):
            print(f"  [actuated | {density} | run {run+1}/{n_runs}]", end=" ", flush=True)
            m = run_actuated_baseline(sumo_cfg=sumo_cfg, tls_ids=tls_ids)
            tracker.add(m)
            print(f"queue={m['avg_queue_length']:.1f}  wait={m['avg_waiting_time']:.1f}s  tp={m['throughput']}")
        summary = tracker.summarise()
        results.append({"controller": "actuated", "density": density, **summary})
    return results


# ──────────────────────────────────────────────────────────────
# Stage 2: PPO Multi-Agent (independent, all 9 intersections)
# ──────────────────────────────────────────────────────────────

def run_stage_ppo_multiagent(cfg: dict, densities: list, n_runs: int) -> list:
    print("\n" + "=" * 60)
    print("  STAGE 2 — PPO Multi-Agent Evaluation (independent)")
    print("=" * 60)

    try:
        from stable_baselines3 import PPO as SB3_PPO
        from envs.multi_traffic_env import MultiTrafficEnv
    except ImportError as e:
        print(f"  [SKIP] Dependencies not available: {e}")
        return []

    tls_ids   = cfg["sumo"]["tls_ids"]
    final_dir = "results/ppo_multiagent/final"

    if not os.path.exists(final_dir):
        print(f"  [SKIP] No PPO multiagent models found at {final_dir}")
        print("  Run training/train_ppo_multiagent.py first.")
        return []

    # Check all 9 agents exist
    missing = [tls for tls in tls_ids
               if not os.path.exists(os.path.join(final_dir, f"{tls}.zip"))]
    if missing:
        print(f"  [SKIP] Missing models for: {missing}")
        print("  Wait for train_ppo_multiagent.py to finish.")
        return []

    print(f"  Loading models from: {final_dir}")
    agents = {tls: SB3_PPO.load(os.path.join(final_dir, tls)) for tls in tls_ids}
    results = []

    for density in densities:
        tracker  = MultiRunMetrics()
        sumo_cfg = get_sumo_cfg_for_density(cfg, density)

        for run in range(n_runs):
            print(f"  [ppo_multiagent | {density} | run {run+1}/{n_runs}]", end=" ", flush=True)

            env      = MultiTrafficEnv(sumo_cfg=sumo_cfg, tls_ids=tls_ids,
                                       max_steps=cfg["episode"]["max_decision_steps"])
            obs_dict = env.reset()
            done            = False
            all_queues      = []
            all_waits       = []
            total_arrived   = 0
            total_teleports = 0
            peak_queue      = 0

            while not done:
                action_dict = {}
                for tls in tls_ids:
                    action, _ = agents[tls].predict(obs_dict[tls], deterministic=True)
                    action_dict[tls] = int(action)

                obs_dict, _, done_dict, info_dict = env.step(action_dict)
                done = done_dict["__all__"]

                step_q = [info_dict[tls]["queue"] for tls in tls_ids]
                step_w = [info_dict[tls]["wait"]  for tls in tls_ids]
                all_queues.append(float(np.mean(step_q)))
                all_waits.append(float(np.mean(step_w)))
                total_arrived   += info_dict[tls_ids[0]]["arrived"]
                total_teleports += info_dict[tls_ids[0]]["teleports"]
                peak_queue       = max(peak_queue, max(step_q))

            env.close()

            m = {
                "avg_queue_length": float(np.mean(all_queues)),
                "avg_waiting_time": float(np.mean(all_waits)),
                "throughput":       total_arrived,
                "total_teleports":  total_teleports,
                "peak_queue":       peak_queue,
            }
            tracker.add(m)
            print(f"queue={m['avg_queue_length']:.1f}  "
                  f"wait={m['avg_waiting_time']:.1f}s  "
                  f"tp={m['throughput']}")

        summary = tracker.summarise()
        results.append({"controller": "ppo_multiagent", "density": density, **summary})
    return results


# ──────────────────────────────────────────────────────────────
# Stage 3: GRPO Multi-Agent
# ──────────────────────────────────────────────────────────────

def run_stage_grpo(cfg: dict, densities: list, n_runs: int) -> list:
    print("\n" + "=" * 60)
    print("  STAGE 3 — GRPO Multi-Agent Evaluation")
    print("=" * 60)

    try:
        from envs.multi_traffic_env import MultiTrafficEnv
        from agents.grpo_agent      import TrafficPolicyNet
    except ImportError as e:
        print(f"  [SKIP] Dependencies not available: {e}")
        return []

    tls_ids = cfg["sumo"]["tls_ids"]
    hidden  = cfg["grpo"]["hidden_dim"]

    # Prefer grpo_mixed/final > grpo_v2/final > grpo/final > latest checkpoint
    model_dir = None
    for candidate in ["results/grpo_mixed/final", "results/grpo_v2/final", "results/grpo/final"]:
        if os.path.exists(candidate) and any(
            f.endswith(".pt") for f in os.listdir(candidate)
        ):
            model_dir = candidate
            break

    if model_dir is None:
        log_dir   = cfg["grpo"]["log_dir"]
        ckpt_dirs = [d for d in os.listdir(log_dir)
                     if d.startswith("checkpoint_ep")
                     and os.path.isdir(os.path.join(log_dir, d))
                     ] if os.path.exists(log_dir) else []
        if ckpt_dirs:
            latest    = sorted(ckpt_dirs, key=lambda d: int(d.replace("checkpoint_ep", "")))[-1]
            model_dir = os.path.join(log_dir, latest)

    if model_dir is None:
        print("  [SKIP] No GRPO models found. Run training/train_grpo.py first.")
        return []

    print(f"  Loading models from: {model_dir}")
    results = []

    for density in densities:
        tracker  = MultiRunMetrics()
        sumo_cfg = get_sumo_cfg_for_density(cfg, density)

        for run in range(n_runs):
            print(f"  [grpo | {density} | run {run+1}/{n_runs}]", end=" ", flush=True)

            env      = MultiTrafficEnv(sumo_cfg=sumo_cfg, tls_ids=tls_ids,
                                       max_steps=cfg["episode"]["max_decision_steps"])
            obs_dict = env.reset()

            agents = {}
            for tls in tls_ids:
                net = TrafficPolicyNet(obs_dim=len(obs_dict[tls]), hidden=hidden)
                f   = os.path.join(model_dir, f"{tls}.pt")
                if os.path.exists(f):
                    net.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
                net.eval()
                agents[tls] = net

            done            = False
            all_queues      = []
            all_waits       = []
            total_arrived   = 0
            total_teleports = 0
            peak_queue      = 0

            while not done:
                action_dict = {}
                with torch.no_grad():
                    for tls in tls_ids:
                        obs_t  = torch.tensor(obs_dict[tls], dtype=torch.float32).unsqueeze(0)
                        action, _ = agents[tls].get_action(obs_t)
                        action_dict[tls] = action.item()

                obs_dict, _, done_dict, info_dict = env.step(action_dict)
                done = done_dict["__all__"]

                step_q = [info_dict[tls]["queue"] for tls in tls_ids]
                step_w = [info_dict[tls]["wait"]  for tls in tls_ids]
                all_queues.append(float(np.mean(step_q)))
                all_waits.append(float(np.mean(step_w)))
                total_arrived   += info_dict[tls_ids[0]]["arrived"]
                total_teleports += info_dict[tls_ids[0]]["teleports"]
                peak_queue       = max(peak_queue, max(step_q))

            env.close()

            m = {
                "avg_queue_length": float(np.mean(all_queues)),
                "avg_waiting_time": float(np.mean(all_waits)),
                "throughput":       total_arrived,
                "total_teleports":  total_teleports,
                "peak_queue":       peak_queue,
            }
            tracker.add(m)
            print(f"queue={m['avg_queue_length']:.1f}  "
                  f"wait={m['avg_waiting_time']:.1f}s  "
                  f"tp={m['throughput']}")

        summary = tracker.summarise()
        results.append({"controller": "grpo", "density": density, **summary})
    return results


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiment_config.yaml")
    parser.add_argument("--stages", nargs="+",
                        choices=["fixed_time", "actuated", "ppo_multiagent", "grpo"],
                        default=["fixed_time", "actuated", "ppo_multiagent", "grpo"])
    parser.add_argument("--densities", nargs="+",
                        choices=["light", "moderate", "heavy"],
                        default=["light", "moderate", "heavy"])
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.smoke_test:
        n_runs = 1
        cfg["episode"]["max_decision_steps"] = 100
        print("[smoke-test] 1 run, 100 decision steps.")
    else:
        n_runs = args.n_runs or cfg["evaluation"]["episodes_per_density"]

    os.makedirs("results", exist_ok=True)
    all_results = []

    if "fixed_time"      in args.stages: all_results += run_stage_fixed_time(cfg, args.densities, n_runs)
    if "actuated"        in args.stages: all_results += run_stage_actuated(cfg, args.densities, n_runs)
    if "ppo_multiagent"  in args.stages: all_results += run_stage_ppo_multiagent(cfg, args.densities, n_runs)
    if "grpo"            in args.stages: all_results += run_stage_grpo(cfg, args.densities, n_runs)

    if all_results:
        print("\n\n" + "=" * 60)
        print("  RESULTS SUMMARY")
        print("=" * 60)
        print(build_comparison_table(all_results))
        out_path = cfg["evaluation"]["comparison_csv"]
        save_comparison_csv(all_results, out_path)
    else:
        print("\nNo results collected.")


if __name__ == "__main__":
    main()

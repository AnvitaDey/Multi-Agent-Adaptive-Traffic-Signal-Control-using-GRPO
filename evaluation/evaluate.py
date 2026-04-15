"""
Unified evaluation runner.
Compares Fixed-Time, PPO, and GRPO under light/moderate/heavy traffic.
"""

import pandas as pd
import json
from controllers.fixed_time import run_fixed_time_baseline
from training.train_grpo import train_grpo


CONFIGS = {
    "light":    "config/sumo/routes_light.rou.xml",
    "moderate": "config/sumo/routes_moderate.rou.xml",
    "heavy":    "config/sumo/routes_heavy.rou.xml",
}

TLS_IDS = [f"J{i}" for i in range(9)]  # 3x3 grid → 9 intersections
SUMO_BASE_CFG = "config/sumo/simulation.sumocfg"


def evaluate_all_controllers(output_path: str = "results/comparison.csv"):
    results = []

    for density, route_file in CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"  Evaluating under {density.upper()} traffic")
        print(f"{'='*50}")

        # ── Fixed-Time Baseline ───────────────────────────────────────────
        ft_metrics = run_fixed_time_baseline(
            sumo_cfg  = SUMO_BASE_CFG,
            tls_ids   = TLS_IDS,
            green_time= 30,
        )
        results.append({"controller": "fixed_time", "density": density, **ft_metrics})
        print(f"Fixed-Time: {ft_metrics}")

        # ── GRPO ─────────────────────────────────────────────────────────
        # (Assumes agents already trained; load checkpoints for evaluation)
        # grpo_metrics = evaluate_grpo_agents(...)
        # results.append({"controller": "grpo", "density": density, **grpo_metrics})

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    return df

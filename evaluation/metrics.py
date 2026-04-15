"""
evaluation/metrics.py

Metrics collection, aggregation, and comparison utilities.
Used by all controllers (fixed-time, PPO, GRPO) for fair evaluation.

Collected metrics:
  - avg_waiting_time   : mean per-vehicle waiting time across all lanes (seconds)
  - avg_queue_length   : mean number of halted vehicles per lane
  - throughput         : total vehicles that completed their journey
  - avg_travel_time    : mean travel time for completed trips (seconds)
  - total_teleports    : vehicles forcibly relocated by SUMO (congestion proxy)
  - total_reward       : cumulative RL reward (0.0 for fixed-time baseline)
"""

import csv
import os
import numpy as np
import traci
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Per-step snapshot
# ──────────────────────────────────────────────────────────────

@dataclass
class StepMetrics:
    """Raw metrics captured at a single simulation step."""
    step:            int
    total_queue:     float    # sum of halted vehicles across all monitored lanes
    total_wait:      float    # sum of waiting times across all monitored lanes
    arrived:         int      # vehicles that arrived at destination this step
    teleports:       int      # vehicles that teleported this step
    reward:          float = 0.0


# ──────────────────────────────────────────────────────────────
# Episode-level collector
# ──────────────────────────────────────────────────────────────

class EpisodeMetrics:
    """
    Collects per-step metrics during a simulation episode
    and aggregates them into episode-level statistics at the end.

    Usage:
        collector = EpisodeMetrics(tls_ids)
        # inside simulation loop:
        collector.record(step, reward=r)
        # after episode:
        summary = collector.aggregate()
    """

    def __init__(self, tls_ids: list[str]):
        self.tls_ids  = tls_ids
        self._records: list[StepMetrics] = []

    def reset(self):
        self._records.clear()

    def record(self, step: int, reward: float = 0.0):
        """
        Capture current SUMO state.
        Must be called while TraCI is active.
        """
        total_queue = 0.0
        total_wait  = 0.0

        for tls in self.tls_ids:
            lanes = set(traci.trafficlight.getControlledLanes(tls))
            for lane in lanes:
                total_queue += traci.lane.getLastStepHaltingNumber(lane)
                total_wait  += traci.lane.getWaitingTime(lane)

        self._records.append(StepMetrics(
            step        = step,
            total_queue = total_queue,
            total_wait  = total_wait,
            arrived     = traci.simulation.getArrivedNumber(),
            teleports   = traci.simulation.getStartingTeleportNumber(),
            reward      = reward,
        ))

    def aggregate(self) -> dict:
        """
        Compute episode-level statistics from collected step records.

        Returns:
            dict with keys:
              avg_queue_length, avg_waiting_time, throughput,
              total_teleports, total_reward, peak_queue,
              num_steps
        """
        if not self._records:
            return _empty_metrics()

        queues     = [r.total_queue for r in self._records]
        waits      = [r.total_wait  for r in self._records]
        rewards    = [r.reward      for r in self._records]
        throughput = sum(r.arrived    for r in self._records)
        teleports  = sum(r.teleports  for r in self._records)

        return {
            "avg_queue_length": float(np.mean(queues)),
            "avg_waiting_time": float(np.mean(waits)),
            "peak_queue":       float(np.max(queues)),
            "throughput":       int(throughput),
            "total_teleports":  int(teleports),
            "total_reward":     float(np.sum(rewards)),
            "mean_reward":      float(np.mean(rewards)),
            "num_steps":        len(self._records),
        }


# ──────────────────────────────────────────────────────────────
# Multi-run aggregator
# ──────────────────────────────────────────────────────────────

class MultiRunMetrics:
    """
    Aggregates metrics across multiple evaluation runs of the same controller
    under the same traffic density. Reports mean ± std for each metric.

    Usage:
        tracker = MultiRunMetrics()
        for run in range(N):
            summary = run_episode(...)
            tracker.add(summary)
        print(tracker.summarise())
    """

    def __init__(self):
        self._runs: list[dict] = []

    def add(self, episode_summary: dict):
        self._runs.append(episode_summary)

    def summarise(self) -> dict:
        if not self._runs:
            return {}

        keys = [k for k in self._runs[0] if isinstance(self._runs[0][k], (int, float))]
        result = {}
        for k in keys:
            vals = [r[k] for r in self._runs if k in r]
            result[f"{k}_mean"] = float(np.mean(vals))
            result[f"{k}_std"]  = float(np.std(vals))
        result["num_runs"] = len(self._runs)
        return result


# ──────────────────────────────────────────────────────────────
# CSV Logger
# ──────────────────────────────────────────────────────────────

class MetricsCSVLogger:
    """
    Writes episode metrics to a CSV file row by row.
    Creates the file with a header on first write.

    Usage:
        logger = MetricsCSVLogger("results/ppo/training_log.csv")
        logger.write(episode=1, controller="ppo", density="moderate",
                     **episode_summary)
    """

    COLUMNS = [
        "episode", "controller", "density",
        "avg_queue_length", "avg_waiting_time", "peak_queue",
        "throughput", "total_teleports",
        "total_reward", "mean_reward", "num_steps",
    ]

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS,
                                        extrasaction="ignore")
                writer.writeheader()

    def write(self, **kwargs):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS,
                                    extrasaction="ignore")
            writer.writerow(kwargs)


# ──────────────────────────────────────────────────────────────
# Comparison table
# ──────────────────────────────────────────────────────────────

def build_comparison_table(results: list[dict]) -> str:
    """
    Format a list of result dicts (each with 'controller' and 'density' keys)
    into a readable comparison table for console output.

    Args:
        results: list of dicts, each with at minimum:
                 controller, density, avg_queue_length_mean,
                 avg_waiting_time_mean, throughput_mean

    Returns:
        Formatted multi-line string.
    """
    header = (
        f"{'Controller':<15} {'Density':<10} "
        f"{'AvgQueue':>10} {'AvgWait(s)':>12} "
        f"{'Throughput':>12} {'Teleports':>10}"
    )
    sep    = "-" * len(header)
    rows   = [header, sep]

    for r in results:
        rows.append(
            f"{r.get('controller','?'):<15} "
            f"{r.get('density','?'):<10} "
            f"{r.get('avg_queue_length_mean', r.get('avg_queue_length', 0)):>10.2f} "
            f"{r.get('avg_waiting_time_mean', r.get('avg_waiting_time', 0)):>12.2f} "
            f"{r.get('throughput_mean',       r.get('throughput', 0)):>12.0f} "
            f"{r.get('total_teleports_mean',  r.get('total_teleports', 0)):>10.0f}"
        )

    return "\n".join(rows)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _empty_metrics() -> dict:
    return {
        "avg_queue_length": 0.0,
        "avg_waiting_time": 0.0,
        "peak_queue":       0.0,
        "throughput":       0,
        "total_teleports":  0,
        "total_reward":     0.0,
        "mean_reward":      0.0,
        "num_steps":        0,
    }


def save_comparison_csv(results: list[dict], path: str):
    """Save a flat list of result dicts to CSV."""
    if not results:
        return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    keys = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Comparison results saved → {path}")

"""Aggregates per-step SUMO metrics into episode-level statistics."""

import traci
import numpy as np


class MetricsLogger:
    def __init__(self, tls_ids: list[str]):
        self.tls_ids = tls_ids
        self._records: list[dict] = []

    def record(self, step: int):
        total_queue = 0
        total_wait  = 0
        for tls in self.tls_ids:
            lanes = traci.trafficlight.getControlledLanes(tls)
            for lane in set(lanes):
                total_queue += traci.lane.getLastStepHaltingNumber(lane)
                total_wait  += traci.lane.getWaitingTime(lane)

        self._records.append({
            "step":      step,
            "queue":     total_queue,
            "wait":      total_wait,
            "arrived":   traci.simulation.getArrivedNumber(),
            "teleports": traci.simulation.getStartingTeleportNumber(),
        })

    def aggregate(self) -> dict:
        if not self._records:
            return {}
        queues     = [r["queue"]   for r in self._records]
        waits      = [r["wait"]    for r in self._records]
        throughput = sum(r["arrived"] for r in self._records)
        teleports  = sum(r["teleports"] for r in self._records)
        return {
            "avg_queue_length": float(np.mean(queues)),
            "avg_waiting_time": float(np.mean(waits)),
            "throughput":       throughput,
            "total_teleports":  teleports,
            "peak_queue":       float(np.max(queues)),
        }

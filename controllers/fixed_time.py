"""
Fixed-time traffic signal controller.
Cycles through phases at fixed intervals. No learning, no state observation.
This is the canonical baseline for all RL comparisons.
"""

import traci
import numpy as np
from utils.logger import MetricsLogger


class FixedTimeController:
    """
    Cycles each intersection through its phases at a fixed green duration.
    
    Args:
        tls_ids:       List of traffic light IDs in the network.
        green_time:    Duration (in SUMO seconds) for each green phase.
        yellow_time:   Duration (in SUMO seconds) for yellow transitions.
    """

    def __init__(self, tls_ids: list[str], green_time: int = 30, yellow_time: int = 3):
        self.tls_ids = tls_ids
        self.green_time = green_time
        self.yellow_time = yellow_time

        # Per-intersection phase tracking
        self._phase_index: dict[str, int] = {tls: 0 for tls in tls_ids}
        self._phase_timer: dict[str, int] = {tls: 0 for tls in tls_ids}
        self._in_yellow: dict[str, bool] = {tls: False for tls in tls_ids}
        self._num_phases: dict[str, int] = {}

    def reset(self):
        """Call once after traci.start() to read phase counts from SUMO."""
        for tls in self.tls_ids:
            logic = traci.trafficlight.getAllProgramLogics(tls)[0]
            # Count only green phases (skip yellows already defined in SUMO logic)
            # We manage our own phase cycling — force a simple 2-phase structure
            self._num_phases[tls] = len([
                p for p in logic.phases
                if 'G' in p.state or 'g' in p.state
            ])
            self._phase_index[tls] = 0
            self._phase_timer[tls] = 0
            self._in_yellow[tls] = False

    def step(self, sim_step: int):
        """
        Called every SUMO simulation step.
        Advances phase timers and switches signals when time expires.
        """
        for tls in self.tls_ids:
            self._phase_timer[tls] += 1

            if self._in_yellow[tls]:
                if self._phase_timer[tls] >= self.yellow_time:
                    # Transition to next green phase
                    self._phase_index[tls] = (
                        self._phase_index[tls] + 1
                    ) % self._num_phases[tls]
                    self._set_green(tls)
                    self._in_yellow[tls] = False
                    self._phase_timer[tls] = 0
            else:
                if self._phase_timer[tls] >= self.green_time:
                    # Start yellow transition
                    self._set_yellow(tls)
                    self._in_yellow[tls] = True
                    self._phase_timer[tls] = 0

    def _set_green(self, tls: str):
        """Set the current green phase for a traffic light."""
        phase_idx = self._phase_index[tls]
        traci.trafficlight.setPhase(tls, phase_idx * 2)  # green phases at even indices

    def _set_yellow(self, tls: str):
        """Set yellow transition phase."""
        phase_idx = self._phase_index[tls]
        traci.trafficlight.setPhase(tls, phase_idx * 2 + 1)  # yellow at odd indices


def run_fixed_time_baseline(
    sumo_cfg: str,
    tls_ids: list[str],
    green_time: int = 30,
    yellow_time: int = 3,
    max_steps: int = 20000,
    use_gui: bool = False,
) -> dict:
    """
    Run a complete fixed-time simulation episode and return aggregated metrics.
    
    Returns:
        dict with keys: avg_waiting_time, avg_queue_length, throughput,
                        avg_travel_time, total_teleports
    """
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    traci.start([sumo_binary, "-c", sumo_cfg, "--no-warnings", "true"])

    controller = FixedTimeController(tls_ids, green_time, yellow_time)
    controller.reset()
    logger = MetricsLogger(tls_ids)

    for step in range(max_steps):
        traci.simulationStep()
        controller.step(step)
        logger.record(step)

    metrics = logger.aggregate()
    traci.close()
    return metrics

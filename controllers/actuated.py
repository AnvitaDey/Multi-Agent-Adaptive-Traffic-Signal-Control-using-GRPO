"""
controllers/actuated.py

SUMO-actuated traffic signal controller — a stronger baseline than fixed-time.

Actuated control extends fixed-time by extending the current green phase
if vehicles are still arriving on the active approach, up to a maximum green cap.
This is a rule-based heuristic widely used in real-world traffic engineering.

Using actuated control as a second baseline helps answer the question:
  "Does RL learn something beyond simple demand-responsive heuristics?"

Two implementations are provided:

1. SumoActuatedController  — delegates to SUMO's built-in actuated logic
                              by switching TLS programs to "actuated" type.
                              Simplest option; requires the net.xml to define
                              an actuated program or we inject one via TraCI.

2. PythonActuatedController — pure-Python actuated logic implemented via TraCI.
                              More flexible; does not require SUMO program support.
                              Used when the network only defines a static program.
"""

import traci
from utils.sumo_utils import get_unique_controlled_lanes, get_num_green_phases
from utils.logger import MetricsLogger


# ──────────────────────────────────────────────────────────────
# Pure-Python actuated controller
# ──────────────────────────────────────────────────────────────

class PythonActuatedController:
    """
    Demand-responsive traffic signal controller.

    Logic (per intersection, per step):
      - If current green phase has been active for at least min_green steps:
        - Check vehicle arrivals on active lanes.
        - If arrivals > threshold AND phase_timer < max_green: extend green.
        - Otherwise: begin yellow transition → switch to next phase.
      - Yellow is a fixed-duration intergreen.

    Args:
        tls_ids:    List of traffic light IDs to control.
        min_green:  Minimum SUMO steps a phase must remain green.
        max_green:  Maximum SUMO steps before forced phase switch.
        yellow:     SUMO steps for yellow intergreen.
        threshold:  Vehicle arrival count above which green is extended.
    """

    def __init__(
        self,
        tls_ids:   list[str],
        min_green: int = 15,
        max_green: int = 60,
        yellow:    int = 3,
        threshold: int = 2,
    ):
        self.tls_ids   = tls_ids
        self.min_green = min_green
        self.max_green = max_green
        self.yellow    = yellow
        self.threshold = threshold

        self._phase_index:  dict[str, int]  = {}
        self._phase_timer:  dict[str, int]  = {}
        self._in_yellow:    dict[str, bool] = {}
        self._yellow_timer: dict[str, int]  = {}
        self._num_phases:   dict[str, int]  = {}
        self._lanes:        dict[str, list] = {}

    def reset(self):
        """Initialise state after traci.start()."""
        for tls in self.tls_ids:
            self._lanes[tls]        = get_unique_controlled_lanes(tls)
            self._num_phases[tls]   = get_num_green_phases(tls)
            self._phase_index[tls]  = 0
            self._phase_timer[tls]  = 0
            self._in_yellow[tls]    = False
            self._yellow_timer[tls] = 0
            traci.trafficlight.setPhase(tls, 0)

    def step(self, sim_step: int):
        """Call once per SUMO simulation step."""
        for tls in self.tls_ids:
            if self._in_yellow[tls]:
                self._yellow_timer[tls] += 1
                if self._yellow_timer[tls] >= self.yellow:
                    self._advance_phase(tls)
            else:
                self._phase_timer[tls] += 1
                timer = self._phase_timer[tls]

                if timer >= self.max_green:
                    # Forced switch: max green exceeded
                    self._start_yellow(tls)

                elif timer >= self.min_green:
                    # Check demand on active lanes
                    demand = self._active_demand(tls)
                    if demand <= self.threshold:
                        # Low demand → switch early
                        self._start_yellow(tls)
                    # else: high demand → extend green (do nothing)

    # ── Private helpers ──────────────────────────────────────

    def _active_demand(self, tls: str) -> int:
        """Count vehicles on lanes active during current phase."""
        lanes = self._lanes[tls]
        phase_state = traci.trafficlight.getRedYellowGreenState(tls)

        active_count = 0
        for i, lane in enumerate(lanes):
            if i < len(phase_state) and phase_state[i] in ("G", "g"):
                active_count += traci.lane.getLastStepVehicleNumber(lane)
        return active_count

    def _start_yellow(self, tls: str):
        yellow_phase = self._phase_index[tls] * 2 + 1
        try:
            traci.trafficlight.setPhase(tls, yellow_phase)
        except traci.exceptions.TraCIException:
            # If yellow phase doesn't exist, go straight to next green
            self._advance_phase(tls)
            return
        self._in_yellow[tls]    = True
        self._yellow_timer[tls] = 0

    def _advance_phase(self, tls: str):
        self._phase_index[tls] = (
            self._phase_index[tls] + 1
        ) % self._num_phases[tls]
        green_phase = self._phase_index[tls] * 2
        traci.trafficlight.setPhase(tls, green_phase)
        self._in_yellow[tls]  = False
        self._phase_timer[tls] = 0


# ──────────────────────────────────────────────────────────────
# Evaluation runner
# ──────────────────────────────────────────────────────────────

def run_actuated_baseline(
    sumo_cfg:    str,
    tls_ids:     list[str],
    min_green:   int = 15,
    max_green:   int = 60,
    yellow:      int = 3,
    threshold:   int = 2,
    max_steps:   int = 20000,
    use_gui:     bool = False,
) -> dict:
    """
    Run a complete actuated-control simulation episode.

    Args:
        sumo_cfg:  Path to .sumocfg file.
        tls_ids:   List of traffic light IDs.
        min_green: Minimum SUMO steps per green phase.
        max_green: Maximum SUMO steps per green phase.
        yellow:    SUMO steps per yellow phase.
        threshold: Vehicle count below which phase switch is triggered early.
        max_steps: Maximum simulation steps.
        use_gui:   Launch SUMO-GUI.

    Returns:
        dict with aggregated episode metrics.
    """
    binary = "sumo-gui" if use_gui else "sumo"
    traci.start([binary, "-c", sumo_cfg,
                 "--no-warnings", "true",
                 "--time-to-teleport", "300"])

    controller = PythonActuatedController(
        tls_ids   = tls_ids,
        min_green = min_green,
        max_green = max_green,
        yellow    = yellow,
        threshold = threshold,
    )
    controller.reset()
    logger = MetricsLogger(tls_ids)

    for step in range(max_steps):
        traci.simulationStep()
        controller.step(step)
        logger.record(step)

        if traci.simulation.getMinExpectedNumber() == 0:
            break

    metrics = logger.aggregate()
    traci.close()
    return metrics

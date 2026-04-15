"""
envs/traffic_env.py
"""

import gymnasium as gym
import numpy as np
import traci
import uuid
from controllers.fixed_time import FixedTimeController

MIN_GREEN_STEPS    = 10
YELLOW_STEPS       = 2
STEPS_PER_DECISION = 10


class TrafficEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, sumo_cfg, controlled_tls, all_tls_ids, max_steps=2000, use_gui=False):
        super().__init__()
        self.sumo_cfg       = sumo_cfg
        self.controlled_tls = controlled_tls
        self.all_tls_ids    = all_tls_ids
        self.other_tls      = [t for t in all_tls_ids if t != controlled_tls]
        self.max_steps      = max_steps
        self.use_gui        = use_gui
        self._label         = f"sumo_{uuid.uuid4().hex[:8]}"

        self._lanes         = []
        self._num_phases    = 2
        self._sumo_running  = False

        obs_dim = self._probe_obs_dim()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = gym.spaces.Discrete(2)

        self._step_count   = 0
        self._phase_index  = 0
        self._phase_timer  = 0
        self._in_yellow    = False
        self._yellow_timer = 0
        self._fixed_ctrl   = None


    def _probe_obs_dim(self):
        probe_label = f"probe_{uuid.uuid4().hex[:8]}"
        binary = "sumo-gui" if self.use_gui else "sumo"

        traci.start([binary, "-c", self.sumo_cfg,
                     "--no-warnings", "true",
                     "--time-to-teleport", "300"], label=probe_label)

        conn = traci.getConnection(probe_label)

        lanes = self._deduplicate(conn.trafficlight.getControlledLanes(self.controlled_tls))
        logic = conn.trafficlight.getAllProgramLogics(self.controlled_tls)[0]

        num_phases = max(2, len([p for p in logic.phases if "G" in p.state or "g" in p.state]))

        traci.switch(probe_label)
        traci.close()

        self._lanes_probe      = lanes
        self._num_phases_probe = num_phases

        return len(lanes) * 2 + num_phases


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self._sumo_running:
            try:
                traci.switch(self._label)
                traci.close()
            except:
                pass

        binary = "sumo-gui" if self.use_gui else "sumo"

        traci.start([binary, "-c", self.sumo_cfg,
                     "--no-warnings", "true",
                     "--time-to-teleport", "300"], label=self._label)

        traci.switch(self._label)

        self._sumo_running = True
        self._lanes        = self._lanes_probe
        self._num_phases   = self._num_phases_probe

        self._fixed_ctrl   = FixedTimeController(self.other_tls)
        self._fixed_ctrl.reset()

        self._step_count   = 0
        self._phase_index  = 0
        self._phase_timer  = 0
        self._in_yellow    = False
        self._yellow_timer = 0

        traci.trafficlight.setPhase(self.controlled_tls, 0)

        return self._get_obs(), {}


    def step(self, action):
        traci.switch(self._label)

        if action == 1 and self._phase_timer >= MIN_GREEN_STEPS and not self._in_yellow:
            self._start_yellow_transition()

        metrics_before = self._snapshot_metrics()

        for _ in range(STEPS_PER_DECISION):
            traci.simulationStep()
            self._fixed_ctrl.step(int(traci.simulation.getTime()))
            self._advance_yellow_if_needed()

        metrics_after = self._snapshot_metrics()

        reward = self._compute_reward(metrics_before, metrics_after, action)

        self._step_count  += 1
        self._phase_timer += 1

        terminated = traci.simulation.getMinExpectedNumber() == 0
        truncated  = self._step_count >= self.max_steps

        if terminated or truncated:
            traci.close()
            self._sumo_running = False

        return self._get_obs(), reward, terminated, truncated, {}


    def _compute_reward(self, before, after, action):
        delta_queue = after["queue"] - before["queue"]
        delta_wait  = (after["wait"] - before["wait"]) / 200.0

        switch_penalty = 0.2 if action == 1 else 0.0

        return float(
            -0.6 * delta_queue
            -0.4 * delta_wait
            -switch_penalty
        )


    def _snapshot_metrics(self):
        return {
            "queue": sum(traci.lane.getLastStepHaltingNumber(l) for l in self._lanes),
            "wait":  sum(traci.lane.getWaitingTime(l) for l in self._lanes),
        }


    def _get_obs(self):
        if not self._sumo_running:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        q = [min(traci.lane.getLastStepHaltingNumber(l)/20.0, 1.0) for l in self._lanes]
        w = [min(traci.lane.getWaitingTime(l)/200.0, 1.0) for l in self._lanes]

        ph = np.zeros(self._num_phases)
        ph[self._phase_index % self._num_phases] = 1

        return np.concatenate([q, w, ph]).astype(np.float32)


    def _start_yellow_transition(self):
        traci.trafficlight.setPhase(self.controlled_tls, self._phase_index * 2 + 1)
        self._in_yellow    = True
        self._yellow_timer = 0


    def _advance_yellow_if_needed(self):
        if not self._in_yellow:
            return

        self._yellow_timer += 1

        if self._yellow_timer >= YELLOW_STEPS:
            self._phase_index = (self._phase_index + 1) % self._num_phases
            traci.trafficlight.setPhase(self.controlled_tls, self._phase_index * 2)

            self._in_yellow   = False
            self._phase_timer = 0


    @staticmethod
    def _deduplicate(lanes):
        seen, unique = set(), []
        for l in lanes:
            if l not in seen:
                seen.add(l)
                unique.append(l)
        return unique
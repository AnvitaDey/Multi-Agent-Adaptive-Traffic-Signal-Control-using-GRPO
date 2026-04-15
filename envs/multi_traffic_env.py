"""
envs/multi_traffic_env.py

Multi-agent SUMO environment — supports mixed-density training.

Key addition: if a list of sumo_cfg paths is passed, a random one is
selected at the start of each episode. This enables mixed-traffic training
where the agent sees light/moderate/heavy conditions in random order.
"""

import uuid
import random
import traci
import numpy as np


MIN_GREEN_STEPS    = 15
YELLOW_STEPS       =  2
STEPS_PER_DECISION = 10


class MultiTrafficEnv:

    def __init__(self, sumo_cfg, tls_ids, max_steps=2000, use_gui=False):
        """
        Args:
            sumo_cfg: str (single config) OR list of str (mixed training).
                      If a list, a random config is chosen each episode.
        """
        self.tls_ids   = tls_ids
        self.max_steps = max_steps
        self.use_gui   = use_gui
        self._label    = f"multi_{uuid.uuid4().hex[:8]}"

        # Support both single config and list of configs
        if isinstance(sumo_cfg, list):
            self._cfg_list = sumo_cfg
        else:
            self._cfg_list = [sumo_cfg]

        self._current_cfg  = self._cfg_list[0]

        self._lanes        = {}
        self._num_phases   = {}
        self._phase_index  = {}
        self._phase_timer  = {}
        self._in_yellow    = {}
        self._yellow_timer = {}
        self._step_count   = 0
        self._sumo_running = False

        # Probe using first config to get obs dims
        self._obs_dims = self._probe_obs_dims(self._cfg_list[0])

    def _probe_obs_dims(self, sumo_cfg):
        probe_label = f"probe_{uuid.uuid4().hex[:8]}"
        binary = "sumo-gui" if self.use_gui else "sumo"
        traci.start([binary, "-c", sumo_cfg, "--no-warnings", "true",
                     "--time-to-teleport", "300"], label=probe_label)
        conn = traci.getConnection(probe_label)
        dims        = {}
        lanes_cache = {}
        phase_cache = {}
        for tls in self.tls_ids:
            lanes = self._dedup(conn.trafficlight.getControlledLanes(tls))
            logic = conn.trafficlight.getAllProgramLogics(tls)[0]
            n_ph  = max(2, len([p for p in logic.phases if "G" in p.state or "g" in p.state]))
            lanes_cache[tls] = lanes
            phase_cache[tls] = n_ph
            dims[tls]        = len(lanes) * 2 + n_ph
        traci.switch(probe_label)
        traci.close()
        self._lanes_probe  = lanes_cache
        self._phases_probe = phase_cache
        print(f"[MultiTrafficEnv:{self._label}] obs dims: "
              + ", ".join(f"{t}={d}" for t, d in dims.items()))
        return dims

    def reset(self):
        if self._sumo_running:
            try:
                traci.switch(self._label)
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

        # Pick a random config each episode — this is mixed-traffic training
        self._current_cfg = random.choice(self._cfg_list)

        binary = "sumo-gui" if self.use_gui else "sumo"
        traci.start([binary, "-c", self._current_cfg, "--no-warnings", "true",
                     "--time-to-teleport", "300"], label=self._label)
        traci.switch(self._label)
        self._sumo_running = True
        self._step_count   = 0

        for tls in self.tls_ids:
            self._lanes[tls]        = self._lanes_probe[tls]
            self._num_phases[tls]   = self._phases_probe[tls]
            self._phase_index[tls]  = 0
            self._phase_timer[tls]  = 0
            self._in_yellow[tls]    = False
            self._yellow_timer[tls] = 0
            traci.trafficlight.setPhase(tls, 0)

        return {tls: self._get_obs(tls) for tls in self.tls_ids}

    def step(self, action_dict):
        traci.switch(self._label)
        for tls, action in action_dict.items():
            if (action == 1 and not self._in_yellow[tls]
                    and self._phase_timer[tls] >= MIN_GREEN_STEPS):
                self._start_yellow(tls)

        metrics_before = {tls: self._snap_local(tls) for tls in self.tls_ids}
        for _ in range(STEPS_PER_DECISION):
            traci.simulationStep()
            for tls in self.tls_ids:
                self._tick_yellow(tls)
        metrics_after = {tls: self._snap_local(tls) for tls in self.tls_ids}

        global_arrived   = traci.simulation.getArrivedNumber()
        global_teleports = traci.simulation.getStartingTeleportNumber()

        self._step_count += 1
        for tls in self.tls_ids:
            if not self._in_yellow[tls]:
                self._phase_timer[tls] += 1

        rewards = {
            tls: self._local_reward(metrics_before[tls], metrics_after[tls])
            for tls in self.tls_ids
        }

        sim_done     = traci.simulation.getMinExpectedNumber() == 0
        episode_done = sim_done or (self._step_count >= self.max_steps)

        if episode_done:
            traci.close()
            self._sumo_running = False

        dones            = {tls: episode_done for tls in self.tls_ids}
        dones["__all__"] = episode_done

        obs   = {tls: self._get_obs(tls) for tls in self.tls_ids}
        infos = {tls: {**metrics_after[tls], "arrived": global_arrived,
                       "teleports": global_teleports} for tls in self.tls_ids}

        return obs, rewards, dones, infos

    def close(self):
        if self._sumo_running:
            try:
                traci.switch(self._label)
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

    def obs_dim(self, tls):
        return self._obs_dims.get(tls, 10)

    def _get_obs(self, tls):
        if not self._sumo_running:
            return np.zeros(self._obs_dims[tls], dtype=np.float32)
        lanes = self._lanes[tls]
        q  = [min(traci.lane.getLastStepHaltingNumber(l) / 20.0, 1.0) for l in lanes]
        w  = [min(traci.lane.getWaitingTime(l) / 200.0, 1.0)          for l in lanes]
        ph = np.zeros(self._num_phases[tls], dtype=np.float32)
        ph[self._phase_index[tls] % self._num_phases[tls]] = 1.0
        return np.concatenate([np.array(q, dtype=np.float32),
                                np.array(w, dtype=np.float32), ph])

    def _snap_local(self, tls):
        lanes = self._lanes[tls]
        return {
            "queue": sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes),
            "wait":  sum(traci.lane.getWaitingTime(l)           for l in lanes),
        }

    def _local_reward(self, before, after):
        delta_queue = after["queue"] - before["queue"]
        delta_wait  = (after["wait"] - before["wait"]) / 200.0
        return float(- 0.5 * delta_queue - 0.5 * delta_wait)

    def _start_yellow(self, tls):
        try:
            traci.trafficlight.setPhase(tls, self._phase_index[tls] * 2 + 1)
            self._in_yellow[tls]    = True
            self._yellow_timer[tls] = 0
        except Exception:
            pass

    def _tick_yellow(self, tls):
        if not self._in_yellow[tls]:
            return
        self._yellow_timer[tls] += 1
        if self._yellow_timer[tls] >= YELLOW_STEPS:
            self._phase_index[tls] = (self._phase_index[tls] + 1) % self._num_phases[tls]
            try:
                traci.trafficlight.setPhase(tls, self._phase_index[tls] * 2)
            except Exception:
                pass
            self._in_yellow[tls]   = False
            self._phase_timer[tls] = 0

    @staticmethod
    def _dedup(lanes):
        seen, unique = set(), []
        for l in lanes:
            if l not in seen:
                seen.add(l)
                unique.append(l)
        return unique

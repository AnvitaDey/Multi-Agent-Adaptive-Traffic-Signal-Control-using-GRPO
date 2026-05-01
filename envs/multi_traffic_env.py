"""
envs/multi_traffic_env.py  (v4 — fixed reward, fixed throughput)
"""

import uuid
import random
import traci
import numpy as np


MIN_GREEN_STEPS    = 10   # was 15 — allow more frequent switching
YELLOW_STEPS       = 2
STEPS_PER_DECISION = 10


class MultiTrafficEnv:

    def __init__(self, sumo_cfg, tls_ids, max_steps=2000, use_gui=False):
        self.tls_ids   = tls_ids
        self.max_steps = max_steps
        self.use_gui   = use_gui
        self._label    = f"multi_{uuid.uuid4().hex[:8]}"

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

        # Track per-episode arrived count per decision step
        self._prev_arrived = 0

        self._obs_dims = self._probe_obs_dims(self._cfg_list[0])

    # ──────────── PROBE ──────────── #

    def _probe_obs_dims(self, sumo_cfg):
        probe_label = f"probe_{uuid.uuid4().hex[:8]}"
        binary = "sumo-gui" if self.use_gui else "sumo"

        traci.start([binary, "-c", sumo_cfg,
                     "--no-warnings", "true",
                     "--time-to-teleport", "300"], label=probe_label)

        conn = traci.getConnection(probe_label)

        dims        = {}
        lanes_cache = {}
        phase_cache = {}

        for tls in self.tls_ids:
            lanes = self._dedup(conn.trafficlight.getControlledLanes(tls))
            logic = conn.trafficlight.getAllProgramLogics(tls)[0]
            n_ph  = max(2, len([
                p for p in logic.phases
                if "G" in p.state or "g" in p.state
            ]))

            lanes_cache[tls] = lanes
            phase_cache[tls] = n_ph
            # obs = queue per lane + wait per lane + phase one-hot
            dims[tls]        = len(lanes) * 2 + n_ph

        traci.switch(probe_label)
        traci.close()

        self._lanes_probe  = lanes_cache
        self._phases_probe = phase_cache

        print(f"[MultiTrafficEnv] obs dims: "
              + ", ".join(f"{t}={d}" for t, d in dims.items()))

        return dims

    # ──────────── RESET ──────────── #

    def reset(self):
        if self._sumo_running:
            try:
                traci.switch(self._label)
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

        self._current_cfg = random.choice(self._cfg_list)

        binary = "sumo-gui" if self.use_gui else "sumo"
        traci.start([binary, "-c", self._current_cfg,
                     "--no-warnings", "true",
                     "--time-to-teleport", "300"], label=self._label)

        traci.switch(self._label)
        self._sumo_running = True
        self._step_count   = 0
        self._prev_arrived = 0   # reset throughput counter

        for tls in self.tls_ids:
            self._lanes[tls]        = self._lanes_probe[tls]
            self._num_phases[tls]   = self._phases_probe[tls]
            self._phase_index[tls]  = 0
            self._phase_timer[tls]  = 0
            self._in_yellow[tls]    = False
            self._yellow_timer[tls] = 0
            traci.trafficlight.setPhase(tls, 0)

        return {tls: self._get_obs(tls) for tls in self.tls_ids}

    # ──────────── STEP ──────────── #

    def step(self, action_dict):
        traci.switch(self._label)

        # Apply phase switch actions
        for tls, action in action_dict.items():
            if (action == 1
                    and not self._in_yellow[tls]
                    and self._phase_timer[tls] >= MIN_GREEN_STEPS):
                self._start_yellow(tls)

        # Snapshot BEFORE sim steps
        metrics_before = {
            tls: self._snap_local(tls) for tls in self.tls_ids
        }

        # Run simulation
        for _ in range(STEPS_PER_DECISION):
            traci.simulationStep()
            for tls in self.tls_ids:
                self._tick_yellow(tls)

        # Snapshot AFTER sim steps
        metrics_after = {
            tls: self._snap_local(tls) for tls in self.tls_ids
        }

        # Throughput: vehicles that completed trips THIS decision step
        total_arrived    = traci.simulation.getArrivedNumber()
        step_arrived     = max(0, total_arrived - self._prev_arrived)
        self._prev_arrived = total_arrived

        global_teleports = traci.simulation.getStartingTeleportNumber()

        # Update timers
        self._step_count += 1
        for tls in self.tls_ids:
            if not self._in_yellow[tls]:
                self._phase_timer[tls] += 1

        # Build rewards — pure local, no catastrophic global term
        rewards = {}
        for tls in self.tls_ids:
            rewards[tls] = self._local_reward(
                metrics_before[tls],
                metrics_after[tls],
                action_dict[tls],
                step_arrived          # ← now correctly passed in
            )

        # Done
        sim_done     = traci.simulation.getMinExpectedNumber() == 0
        episode_done = sim_done or (self._step_count >= self.max_steps)

        if episode_done:
            traci.close()
            self._sumo_running = False

        dones            = {tls: episode_done for tls in self.tls_ids}
        dones["__all__"] = episode_done

        obs = {tls: self._get_obs(tls) for tls in self.tls_ids}

        infos = {
            tls: {
                **metrics_after[tls],
                "arrived":   step_arrived,
                "teleports": global_teleports,
            }
            for tls in self.tls_ids
        }

        return obs, rewards, dones, infos

    # ──────────── REWARD ──────────── #

    def _local_reward(self, before, after, action, arrived=0):
        """
        Reward components — all bounded to reasonable scale:

        1. delta_queue  : change in halting vehicles  → range roughly [-5, +5]
           weight -0.5  → contribution [-2.5, +2.5]

        2. delta_wait   : change in cumulative wait, normalised by 200s
           weight -0.3  → contribution roughly [-1.5, +1.5]

        3. throughput   : vehicles arrived this step, normalised by 10
           weight +0.3  → contribution [0, ~1.5] (rarely >5 per step)

        4. teleport_pen : stuck vehicles being teleported — bad
           weight -1.0  → rare but significant penalty

        5. switch_pen   : discourage pointless switching
           fixed  -0.1  → small constant

        Total range: roughly [-6, +4] per step
        Episode total (2000 steps): roughly [-12000, +8000]
        """
        delta_queue = after["queue"] - before["queue"]
        delta_wait  = (after["wait"]  - before["wait"]) / 200.0

        # Throughput: normalise by ~10 vehicles/step max expected
        throughput_bonus = arrived / 10.0

        switch_penalty = 0.1 if action == 1 else 0.0

        return float(
            -0.5 * delta_queue
            -0.3 * delta_wait
            +0.3 * throughput_bonus
            -switch_penalty
        )

    # ──────────── OBS ──────────── #

    def _get_obs(self, tls):
        if not self._sumo_running:
            return np.zeros(self._obs_dims[tls], dtype=np.float32)

        lanes = self._lanes[tls]

        q  = [min(traci.lane.getLastStepHaltingNumber(l) / 20.0, 1.0)
              for l in lanes]
        w  = [min(traci.lane.getWaitingTime(l) / 200.0,          1.0)
              for l in lanes]

        ph = np.zeros(self._num_phases[tls], dtype=np.float32)
        ph[self._phase_index[tls] % self._num_phases[tls]] = 1.0

        return np.concatenate([np.array(q), np.array(w), ph])

    # ──────────── HELPERS ──────────── #

    def _snap_local(self, tls):
        lanes = self._lanes[tls]
        return {
            "queue": sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes),
            "wait":  sum(traci.lane.getWaitingTime(l)           for l in lanes),
        }

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
            self._phase_index[tls] = (
                (self._phase_index[tls] + 1) % self._num_phases[tls]
            )
            try:
                traci.trafficlight.setPhase(tls, self._phase_index[tls] * 2)
            except Exception:
                pass
            self._in_yellow[tls]   = False
            self._phase_timer[tls] = 0

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

    @staticmethod
    def _dedup(lanes):
        seen, unique = set(), []
        for l in lanes:
            if l not in seen:
                seen.add(l)
                unique.append(l)
        return unique
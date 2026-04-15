"""
utils/sumo_utils.py

SUMO and TraCI utility functions used across the project.

Covers:
  - Starting / stopping SUMO safely
  - Querying lane and intersection state
  - Generating route files for different traffic densities
  - Reading traffic light phase structure
  - Patching a .sumocfg to point at a different route file at runtime
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from typing import Optional
import traci


# ──────────────────────────────────────────────────────────────
# SUMO process management
# ──────────────────────────────────────────────────────────────

def start_sumo(
    sumo_cfg:         str,
    use_gui:          bool = False,
    time_to_teleport: int  = 300,
    extra_args:       Optional[list[str]] = None,
) -> None:
    """
    Start a SUMO simulation via TraCI.

    Args:
        sumo_cfg:         Path to .sumocfg file.
        use_gui:          If True, launches sumo-gui (for debugging).
        time_to_teleport: Seconds before SUMO teleports a stuck vehicle.
                          Set to -1 to disable teleports entirely.
        extra_args:       Additional command-line flags passed to SUMO.
    """
    binary = "sumo-gui" if use_gui else "sumo"
    cmd = [
        binary,
        "-c", sumo_cfg,
        "--no-warnings", "true",
        "--time-to-teleport", str(time_to_teleport),
    ]
    if extra_args:
        cmd.extend(extra_args)
    traci.start(cmd)


def close_sumo() -> None:
    """Close the TraCI connection if it is currently open."""
    try:
        traci.close()
    except Exception:
        pass


def is_sumo_running() -> bool:
    """Return True if a TraCI connection is currently active."""
    try:
        traci.simulation.getTime()
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────
# Traffic light / phase utilities
# ──────────────────────────────────────────────────────────────

def get_tls_ids_from_network(net_xml: str) -> list:
    """
    Parse a SUMO network file and return all traffic light IDs.

    Args:
        net_xml: Path to network.net.xml

    Returns:
        List of traffic light node IDs.
    """
    tree = ET.parse(net_xml)
    root = tree.getroot()
    tls_ids = []
    for junction in root.findall("junction"):
        if junction.attrib.get("type") == "traffic_light":
            tls_ids.append(junction.attrib["id"])
    return sorted(tls_ids)


def get_green_phase_indices(tls_id: str) -> list:
    """
    Return the indices of green phases for a traffic light.
    Green phases are those whose state string contains 'G' or 'g'.

    Args:
        tls_id: Traffic light ID (TraCI must be active).

    Returns:
        List of phase indices that are green.
    """
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    return [
        i for i, phase in enumerate(logic.phases)
        if "G" in phase.state or "g" in phase.state
    ]


def get_num_green_phases(tls_id: str) -> int:
    """Return the number of distinct green phases for a traffic light."""
    return max(2, len(get_green_phase_indices(tls_id)))


def get_unique_controlled_lanes(tls_id: str) -> list:
    """
    Return the unique lanes controlled by a traffic light,
    preserving original order.

    SUMO's getControlledLanes() often returns duplicates because the same
    lane can appear in multiple signal links. This deduplicates them.
    """
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    seen, unique = set(), []
    for lane in lanes:
        if lane not in seen:
            seen.add(lane)
            unique.append(lane)
    return unique


# ──────────────────────────────────────────────────────────────
# Lane state queries
# ──────────────────────────────────────────────────────────────

def get_lane_queue(lane_id: str) -> int:
    """Number of halted vehicles on a lane (speed < 0.1 m/s)."""
    return traci.lane.getLastStepHaltingNumber(lane_id)


def get_lane_wait(lane_id: str) -> float:
    """Accumulated waiting time of all vehicles on a lane (seconds)."""
    return traci.lane.getWaitingTime(lane_id)


def get_lane_density(lane_id: str) -> float:
    """Vehicle density on a lane: vehicles / metre."""
    length = traci.lane.getLength(lane_id)
    count  = traci.lane.getLastStepVehicleNumber(lane_id)
    if length <= 0:
        return 0.0
    return count / length


def get_intersection_metrics(tls_id: str) -> dict:
    """
    Aggregate queue length and waiting time across all unique lanes
    of an intersection.

    Returns:
        dict with keys: queue, wait, density, num_lanes
    """
    lanes   = get_unique_controlled_lanes(tls_id)
    queue   = sum(get_lane_queue(l)   for l in lanes)
    wait    = sum(get_lane_wait(l)    for l in lanes)
    density = sum(get_lane_density(l) for l in lanes)
    return {
        "queue":     queue,
        "wait":      wait,
        "density":   density,
        "num_lanes": len(lanes),
    }


def get_network_metrics(tls_ids: list) -> dict:
    """
    Aggregate metrics across the entire network of intersections.

    Returns:
        dict with keys: total_queue, total_wait, arrived, teleports
    """
    total_queue = 0
    total_wait  = 0.0
    for tls in tls_ids:
        m = get_intersection_metrics(tls)
        total_queue += m["queue"]
        total_wait  += m["wait"]
    return {
        "total_queue": total_queue,
        "total_wait":  total_wait,
        "arrived":     traci.simulation.getArrivedNumber(),
        "teleports":   traci.simulation.getStartingTeleportNumber(),
    }


# ──────────────────────────────────────────────────────────────
# Route file generation
# ──────────────────────────────────────────────────────────────

DENSITY_VEHICLE_COUNTS = {
    "light":    500,
    "moderate": 1334,
    "heavy":    2500,
}

DENSITY_PERIOD = {
    "light":    4.0,
    "moderate": 1.5,
    "heavy":    0.8,
}


def generate_routes(
    net_xml:    str,
    output_dir: str,
    density:    str = "moderate",
    end_time:   int = 3600,
    seed:       int = 42,
) -> str:
    """
    Generate a SUMO route file using randomTrips.py.

    Args:
        net_xml:    Path to network .net.xml file.
        output_dir: Directory to write the output .rou.xml file.
        density:    One of 'light', 'moderate', 'heavy'.
        end_time:   Simulation end time in seconds.
        seed:       Random seed for reproducibility.

    Returns:
        Path to the generated routes file.

    Requires:
        SUMO_HOME environment variable to be set.
    """
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError(
            "SUMO_HOME environment variable is not set. "
            "Set it to your SUMO installation directory."
        )

    random_trips = os.path.join(sumo_home, "tools", "randomTrips.py")
    if not os.path.exists(random_trips):
        raise FileNotFoundError(f"randomTrips.py not found at {random_trips}")

    os.makedirs(output_dir, exist_ok=True)

    period     = DENSITY_PERIOD.get(density, 1.5)
    route_file = os.path.join(output_dir, f"routes_{density}.rou.xml")
    trips_file = os.path.join(output_dir, f"trips_{density}.trips.xml")

    subprocess.run([
        "python3", random_trips,
        "-n", net_xml,
        "-o", trips_file,
        "-e", str(end_time),
        "-p", str(period),
        "--seed", str(seed),
        "--validate",
    ], check=True)

    subprocess.run([
        "duarouter",
        "-n", net_xml,
        "-t", trips_file,
        "-o", route_file,
        "--ignore-errors",
        "--no-warnings",
    ], check=True)

    if os.path.exists(trips_file):
        os.remove(trips_file)

    print(f"Routes generated → {route_file}")
    return route_file


# ──────────────────────────────────────────────────────────────
# Config file utilities
# ──────────────────────────────────────────────────────────────

def patch_sumocfg_routes(
    template_cfg: str,
    route_file:   str,
    output_cfg:   str,
) -> str:
    """
    Create a copy of a .sumocfg file pointing to a different route file.
    Uses absolute paths so SUMO can find all files regardless of where
    the temp config is written.

    Args:
        template_cfg: Path to the base .sumocfg file.
        route_file:   Path to the new .rou.xml file.
        output_cfg:   Path to write the patched .sumocfg.

    Returns:
        Path to the patched .sumocfg.
    """
    tree = ET.parse(template_cfg)
    root = tree.getroot()

    input_elem = root.find("input")
    if input_elem is None:
        raise ValueError(f"No <input> element found in {template_cfg}")

    # Resolve all file references to absolute paths so SUMO can find them
    # from wherever the temp config is saved
    base_dir = os.path.abspath(os.path.dirname(template_cfg))

    # Fix net-file path
    net_elem = input_elem.find("net-file")
    if net_elem is not None:
        net_val = net_elem.attrib.get("value", "")
        if not os.path.isabs(net_val):
            net_elem.set("value", os.path.join(base_dir, net_val))

    # Fix route-files path
    route_elem = input_elem.find("route-files")
    if route_elem is not None:
        route_elem.set("value", os.path.abspath(route_file))
    else:
        new_elem = ET.SubElement(input_elem, "route-files")
        new_elem.set("value", os.path.abspath(route_file))

    os.makedirs(
        os.path.dirname(output_cfg) if os.path.dirname(output_cfg) else ".",
        exist_ok=True
    )
    tree.write(output_cfg)
    return output_cfg


def validate_sumo_config(sumo_cfg: str) -> bool:
    """
    Quick validation: check that the .sumocfg file exists and
    its referenced network and route files can be found.

    Returns True if all referenced files exist, False otherwise.
    """
    if not os.path.exists(sumo_cfg):
        print(f"[validate] Config not found: {sumo_cfg}")
        return False

    base_dir = os.path.dirname(os.path.abspath(sumo_cfg))
    tree     = ET.parse(sumo_cfg)
    root     = tree.getroot()

    ok = True
    input_elem = root.find("input")
    if input_elem is None:
        print("[validate] No <input> section in config.")
        return False

    for tag in ("net-file", "route-files"):
        elem = input_elem.find(tag)
        if elem is None:
            print(f"[validate] Missing <{tag}> in config.")
            ok = False
            continue
        val = elem.attrib.get("value", "")
        path = val if os.path.isabs(val) else os.path.join(base_dir, val)
        if not os.path.exists(path):
            print(f"[validate] Referenced file not found: {path}")
            ok = False

    if ok:
        print(f"[validate] Config OK: {sumo_cfg}")
    return ok
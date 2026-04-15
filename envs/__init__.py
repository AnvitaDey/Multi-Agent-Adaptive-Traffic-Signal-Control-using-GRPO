"""
envs/__init__.py

Exports both environment classes so they can be imported cleanly:

    from envs import TrafficEnv, MultiTrafficEnv
"""

from envs.traffic_env       import TrafficEnv
from envs.multi_traffic_env import MultiTrafficEnv

__all__ = ["TrafficEnv", "MultiTrafficEnv"]

# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Gymnasium environment registration for the Go2 obstacle course tasks."""

import gymnasium as gym

from sim.agents.go2_ppo_cfg import Go2FlatPPORunnerCfg, Go2PPORunnerCfg

# --- Rough / Obstacle terrain ---
gym.register(
    id="Go2-Obstacle-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "sim.envs.go2_obstacle_env_cfg:Go2ObstacleEnvCfg",
        "rsl_rl_cfg_entry_point": "sim.agents.go2_ppo_cfg:Go2PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Obstacle-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "sim.envs.go2_obstacle_env_cfg:Go2ObstacleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "sim.agents.go2_ppo_cfg:Go2PPORunnerCfg",
    },
)

# --- Flat terrain ---
gym.register(
    id="Go2-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "sim.envs.go2_obstacle_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "sim.agents.go2_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "sim.envs.go2_obstacle_env_cfg:Go2FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "sim.agents.go2_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)

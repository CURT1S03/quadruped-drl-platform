# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Run a trained Go2 policy in evaluation / play mode.

Usage:
    isaaclab -p sim/scripts/play.py --task Go2-Obstacle-Play-v0 --checkpoint path/to/model.pt
    isaaclab -p sim/scripts/play.py --task Go2-Flat-Play-v0 --checkpoint path/to/model.pt --num_envs 4
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure project root is on sys.path so 'sim' package is importable
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# --------------------------------------------------------------------------- #
# 1. Parse CLI & launch simulator                                             #
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Evaluate a trained Go2 policy.")
parser.add_argument("--task", type=str, default="Go2-Obstacle-Play-v0", help="Play-variant task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
parser.add_argument("--num_envs", type=int, default=None, help="Override environment count.")
parser.add_argument("--num_steps", type=int, default=5000, help="Number of simulation steps to run.")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
# 2. Imports after simulator launch                                           #
# --------------------------------------------------------------------------- #
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import sim.envs.go2_obstacle_env  # noqa: F401


def main():
    task_entry = gym.spec(args_cli.task)
    env_cfg_cls = task_entry.kwargs["env_cfg_entry_point"]
    agent_cfg_cls = task_entry.kwargs["rsl_rl_cfg_entry_point"]

    import importlib

    if isinstance(env_cfg_cls, str):
        mod, cls = env_cfg_cls.rsplit(":", 1)
        env_cfg: ManagerBasedRLEnvCfg = getattr(importlib.import_module(mod), cls)()
    else:
        env_cfg = env_cfg_cls()

    if isinstance(agent_cfg_cls, str):
        mod, cls = agent_cfg_cls.rsplit(":", 1)
        agent_cfg: RslRlOnPolicyRunnerCfg = getattr(importlib.import_module(mod), cls)()
    else:
        agent_cfg = agent_cfg_cls()

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load checkpoint
    log_dir = os.path.dirname(args_cli.checkpoint)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)

    print(f"[INFO] Running inference with checkpoint: {args_cli.checkpoint}")
    print(f"[INFO] Environments: {env_cfg.scene.num_envs}, Steps: {args_cli.num_steps}")

    # Get the policy
    policy = runner.get_inference_policy(device=agent_cfg.device)

    obs, _ = env.get_observations()
    for step in range(args_cli.num_steps):
        actions = policy(obs)
        obs, _, _, _, _ = env.step(actions)

    print("[INFO] Evaluation complete.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Train a Go2 quadruped with RSL-RL PPO.

Usage:
    # From project root, via Isaac Lab's Python:
    isaaclab -p sim/scripts/train.py --task Go2-Obstacle-v0 --num_envs 4096 --headless
    isaaclab -p sim/scripts/train.py --task Go2-Flat-v0 --num_envs 2048 --max_iterations 300
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

# Ensure project root is on sys.path so 'sim' package is importable
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# --------------------------------------------------------------------------- #
# 1. Parse CLI & launch the simulator BEFORE any Isaac Lab imports            #
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Train Go2 quadruped with RSL-RL PPO.")
parser.add_argument("--task", type=str, default="Go2-Obstacle-v0", help="Registered Gym task id.")
parser.add_argument("--num_envs", type=int, default=None, help="Override number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override max training iterations.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--log_dir", type=str, default=None, help="Custom log directory.")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
parser.add_argument("--video", action="store_true", help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
# 2. Now safe to import Isaac Lab and RL modules                              #
# --------------------------------------------------------------------------- #
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

# Register our custom environments
import sim.envs.go2_obstacle_env  # noqa: F401

logger = logging.getLogger(__name__)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # ---------------------------------------------------------------------- #
    # Resolve task configs via gymnasium registry                             #
    # ---------------------------------------------------------------------- #
    task_entry = gym.spec(args_cli.task)
    env_cfg_cls = task_entry.kwargs["env_cfg_entry_point"]
    agent_cfg_cls = task_entry.kwargs["rsl_rl_cfg_entry_point"]

    # Import the config classes
    if isinstance(env_cfg_cls, str):
        module_path, cls_name = env_cfg_cls.rsplit(":", 1)
        import importlib
        env_cfg: ManagerBasedRLEnvCfg = getattr(importlib.import_module(module_path), cls_name)()
    else:
        env_cfg = env_cfg_cls()

    if isinstance(agent_cfg_cls, str):
        module_path, cls_name = agent_cfg_cls.rsplit(":", 1)
        import importlib
        agent_cfg: RslRlOnPolicyRunnerCfg = getattr(importlib.import_module(module_path), cls_name)()
    else:
        agent_cfg = agent_cfg_cls()

    # CLI overrides
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    env_cfg.seed = args_cli.seed

    # ---------------------------------------------------------------------- #
    # Logging directory                                                       #
    # ---------------------------------------------------------------------- #
    if args_cli.log_dir:
        log_root = args_cli.log_dir
    else:
        log_root = os.path.join("logs", "runs", agent_cfg.experiment_name)
    log_root = os.path.abspath(log_root)

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # ---------------------------------------------------------------------- #
    # Create environment                                                      #
    # ---------------------------------------------------------------------- #
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ---------------------------------------------------------------------- #
    # Create runner                                                           #
    # ---------------------------------------------------------------------- #
    import importlib.metadata
    rsl_rl_version = importlib.metadata.version("rsl-rl-lib")
    handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if args_cli.resume:
        print(f"[INFO] Resuming from: {args_cli.resume}")
        runner.load(args_cli.resume)

    # Save configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # ---------------------------------------------------------------------- #
    # Train (with JSON telemetry output for backend parsing)                  #
    # ---------------------------------------------------------------------- #
    start_time = time.time()
    print(f"[INFO] Starting training: {agent_cfg.max_iterations} iterations, {env_cfg.scene.num_envs} envs")

    # Patch the runner's log method to also emit JSON to stdout
    original_log = runner.log if hasattr(runner, "log") else None

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    elapsed = time.time() - start_time
    print(f"[INFO] Training complete in {elapsed:.1f}s")

    # Emit final summary as JSON for backend consumption
    summary = {
        "event": "training_complete",
        "log_dir": log_dir,
        "elapsed_seconds": round(elapsed, 2),
        "max_iterations": agent_cfg.max_iterations,
        "num_envs": env_cfg.scene.num_envs,
    }
    print(f"TELEMETRY:{json.dumps(summary)}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Export a trained RSL-RL policy to TorchScript for deployment in Isaac Sim.

The exported .pt file is compatible with Isaac Sim's PolicyController
(exts/isaacsim.robot.policy.examples/controllers/policy_controller.py).

Produces a zip bundle containing:
  - policy.pt          — TorchScript actor network
  - metadata.json      — obs/action dims, robot info, training config

Usage:
    python sim/scripts/export_policy.py --checkpoint logs/runs/go2_obstacle_course/2024-.../model_1500.pt
    python sim/scripts/export_policy.py --checkpoint logs/runs/custom_mybot/.../model_500.pt --auto
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from datetime import datetime

# Ensure project root is on sys.path so 'sim' package is importable
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch


def main():
    parser = argparse.ArgumentParser(description="Export RSL-RL policy to TorchScript .pt")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to RSL-RL model checkpoint.")
    parser.add_argument("--output", type=str, default=None, help="Output path. Defaults to <checkpoint_dir>/exported_policy.zip")
    parser.add_argument("--obs_dim", type=int, default=None, help="Observation dimension. Auto-detected from robot_info.json if available.")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    checkpoint_dir = os.path.dirname(args.checkpoint)

    # ── Auto-detect obs_dim from robot_info.json if available ────────── #
    robot_info = None
    obs_dim = args.obs_dim

    # Look for robot_info.json in the params/ subdirectory of the run
    run_dir = checkpoint_dir
    robot_info_path = os.path.join(run_dir, "params", "robot_info.json")
    if not os.path.exists(robot_info_path):
        # Checkpoint might be nested one level deeper
        parent = os.path.dirname(run_dir)
        robot_info_path = os.path.join(parent, "params", "robot_info.json")

    if os.path.isfile(robot_info_path):
        with open(robot_info_path, "r") as f:
            robot_info = json.load(f)
        if obs_dim is None:
            obs_dim = robot_info.get("obs_dim", 235)
        print(f"[INFO] Auto-detected from robot_info.json: obs_dim={obs_dim}, robot={robot_info.get('robot_name', 'unknown')}")
    else:
        if obs_dim is None:
            obs_dim = 235  # Go2 default
        print(f"[INFO] No robot_info.json found, using obs_dim={obs_dim}")

    # Load the RSL-RL checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # RSL-RL saves the actor-critic model state dict
    # We need to extract the actor (policy) network
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Reconstruct the actor network architecture
    # RSL-RL ActorCritic uses separate actor and critic MLPs
    # Actor layers are prefixed with "actor."
    actor_state = {}
    for key, value in state_dict.items():
        if key.startswith("actor."):
            actor_state[key.replace("actor.", "")] = value

    if not actor_state:
        print("[ERROR] Could not extract actor weights from checkpoint.")
        print(f"  Available keys: {list(state_dict.keys())[:20]}")
        sys.exit(1)

    # Infer network architecture from weight shapes
    layers = []
    layer_idx = 0
    while f"{layer_idx}.weight" in actor_state:
        weight = actor_state[f"{layer_idx}.weight"]
        layers.append(weight.shape)
        layer_idx += 1
        # Skip activation layers (they don't have weights)
        if f"{layer_idx}.weight" not in actor_state:
            layer_idx += 1
            if f"{layer_idx}.weight" not in actor_state:
                break

    print(f"[INFO] Detected actor layers: {layers}")

    # Build a scriptable actor module
    class ExportedPolicy(torch.nn.Module):
        def __init__(self, state_dict: dict, obs_dim: int):
            super().__init__()
            # Reconstruct the sequential network
            modules = []
            idx = 0
            while f"{idx}.weight" in state_dict:
                in_features = state_dict[f"{idx}.weight"].shape[1]
                out_features = state_dict[f"{idx}.weight"].shape[0]
                modules.append(torch.nn.Linear(in_features, out_features))
                idx += 1
                # Check if next is an activation (no weight key at idx)
                if f"{idx}.weight" not in state_dict:
                    # Check if this is the last layer (no more weights after)
                    if f"{idx + 1}.weight" in state_dict:
                        modules.append(torch.nn.ELU())
                        idx += 1
            self.network = torch.nn.Sequential(*modules)
            # Load weights
            self.network.load_state_dict(state_dict)

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.network(obs)

    policy = ExportedPolicy(actor_state, obs_dim)
    policy.eval()

    # Trace the model for TorchScript export
    dummy_input = torch.zeros(1, obs_dim)
    scripted = torch.jit.trace(policy, dummy_input)

    # ── Build metadata ───────────────────────────────────────────────── #
    action_dim = layers[-1][0] if layers else 0
    metadata = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "actor_layers": [list(s) for s in layers],
        "checkpoint_path": os.path.basename(args.checkpoint),
        "exported_at": datetime.now().isoformat(),
    }
    if robot_info:
        metadata["robot_name"] = robot_info.get("robot_name", "unknown")
        metadata["num_dof"] = robot_info.get("num_dof", action_dim)
        metadata["standing_height"] = robot_info.get("standing_height")
        metadata["foot_body_names"] = robot_info.get("foot_body_names")
        metadata["terrain_name"] = robot_info.get("terrain_name")
        metadata["use_height_scan"] = robot_info.get("use_height_scan")
    else:
        metadata["robot_name"] = "go2"
        metadata["num_dof"] = 12
        metadata["standing_height"] = 0.34

    # ── Determine output path ────────────────────────────────────────── #
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(checkpoint_dir, "exported_policy.zip")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save TorchScript to a temp file, then bundle into zip
    pt_path = output_path.replace(".zip", ".pt") if output_path.endswith(".zip") else output_path + ".pt"
    scripted.save(pt_path)

    if output_path.endswith(".zip"):
        metadata_json = json.dumps(metadata, indent=2)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(pt_path, "policy.pt")
            zf.writestr("metadata.json", metadata_json)
        # Clean up standalone .pt
        os.remove(pt_path)
        print(f"[INFO] Exported policy bundle to: {output_path}")
    else:
        # Also save metadata alongside the .pt
        meta_path = pt_path.replace(".pt", "_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[INFO] Exported TorchScript policy to: {pt_path}")
        print(f"[INFO] Metadata saved to: {meta_path}")

    print(f"[INFO] Input dim: {obs_dim}, Output dim: {action_dim}")


if __name__ == "__main__":
    main()

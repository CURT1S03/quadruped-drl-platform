# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Export a trained RSL-RL policy to TorchScript for deployment in Isaac Sim.

The exported .pt file is compatible with Isaac Sim's PolicyController
(exts/isaacsim.robot.policy.examples/controllers/policy_controller.py).

Usage:
    python sim/scripts/export_policy.py --checkpoint logs/runs/go2_obstacle_course/2024-.../model_1500.pt
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure project root is on sys.path so 'sim' package is importable
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch


def main():
    parser = argparse.ArgumentParser(description="Export RSL-RL policy to TorchScript .pt")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to RSL-RL model checkpoint.")
    parser.add_argument("--output", type=str, default=None, help="Output .pt path. Defaults to <checkpoint_dir>/exported_policy.pt")
    parser.add_argument("--obs_dim", type=int, default=235, help="Observation dimension (48 base + 187 height scan for rough).")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

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

    policy = ExportedPolicy(actor_state, args.obs_dim)
    policy.eval()

    # Trace the model for TorchScript export
    dummy_input = torch.zeros(1, args.obs_dim)
    scripted = torch.jit.trace(policy, dummy_input)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(os.path.dirname(args.checkpoint), "exported_policy.pt")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scripted.save(output_path)
    print(f"[INFO] Exported TorchScript policy to: {output_path}")
    print(f"[INFO] Input dim: {args.obs_dim}, Output dim: {layers[-1][0] if layers else '?'}")


if __name__ == "__main__":
    main()

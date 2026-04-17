# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Custom reward functions for Go2 quadruped locomotion.

These augment the base rewards from isaaclab.envs.mdp with
obstacle-course-specific terms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Reward long steps (feet air time above threshold) when the robot is commanded to move.

    This encourages the robot to lift its feet and take proper strides rather
    than shuffling. Only active when a non-zero base velocity command is given.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # True when foot just made contact this step
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # Reward air time above the threshold
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # Only reward when the robot is commanded to move
    command = env.command_manager.get_command(command_name)
    is_moving = torch.norm(command[:, :2], dim=1) > 0.1
    return reward * is_moving


def feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Penalize feet that experience large lateral forces on contact.

    Lateral force spikes indicate the foot is catching on terrain edges,
    which destabilizes the gait.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    # Latest timestep forces
    forces = net_forces[:, 0, :, :]
    # Check if foot is in contact (vertical force > threshold)
    in_contact = forces[:, :, 2].abs() > threshold
    # Lateral force magnitude
    lateral_force = torch.norm(forces[:, :, :2], dim=-1)
    # Only penalize stumbles during ground contact
    stumble = (lateral_force > 4.0 * forces[:, :, 2].abs()) & in_contact
    return torch.sum(stumble.float(), dim=1)


def base_height_target(
    env: ManagerBasedRLEnv,
    target_height: float = 0.34,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward maintaining the base at a target height above ground.

    Uses L2 penalty on deviation from the target height.
    The target of 0.34m is the nominal standing height for Go2.
    """
    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    return torch.square(base_height - target_height)


def joint_velocity_limits(
    env: ManagerBasedRLEnv,
    soft_ratio: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joints approaching their velocity limits.

    Uses a soft threshold at `soft_ratio` of the actual limit to encourage
    the policy to stay well within operational bounds.
    """
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel
    vel_limits = asset.data.soft_joint_vel_limits
    # Compute ratio of current velocity to limit
    vel_ratio = torch.abs(joint_vel) / vel_limits
    # Only penalize when exceeding soft threshold
    exceed = torch.clamp(vel_ratio - soft_ratio, min=0.0)
    return torch.sum(torch.square(exceed), dim=1)

# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Parameterized environment configuration factory for custom robots.

Generates ManagerBasedRLEnvCfg instances dynamically based on:
  - A robot URDF + metadata (via robot_loader.py)
  - A terrain config (preset name or YAML path)

This allows training arbitrary quadrupeds without writing new Python env configs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.envs import mdp

from sim.envs.robot_loader import RobotMetadata
from sim.rewards import locomotion_rewards
from sim.terrains.terrain_presets import get_terrain_cfg, FLAT_TERRAIN_CFG


def build_foot_regex(foot_names: list[str]) -> str:
    """Build a regex pattern that matches any of the given foot body names."""
    if len(foot_names) == 1:
        return foot_names[0]
    escaped = [name.replace(".", r"\.") for name in foot_names]
    return "(" + "|".join(escaped) + ")"


def build_custom_env_cfg(
    robot_meta: RobotMetadata,
    terrain_name: str = "obstacle",
    terrain_yaml: str | None = None,
    num_envs: int = 4096,
    use_height_scan: bool = True,
) -> ManagerBasedRLEnvCfg:
    """Build a complete ManagerBasedRLEnvCfg for a custom robot.

    Args:
        robot_meta: Parsed robot metadata from robot_loader.
        terrain_name: Preset terrain name ("flat", "obstacle", "easy", "hard", "stairs", "slopes").
        terrain_yaml: Path to custom terrain YAML (overrides terrain_name).
        num_envs: Number of parallel environments.
        use_height_scan: Whether to include height scanner observations.

    Returns:
        A fully configured ManagerBasedRLEnvCfg instance.
    """
    foot_regex = build_foot_regex(robot_meta.foot_body_names)
    is_flat = terrain_name == "flat" and terrain_yaml is None

    # --- Terrain ---
    if terrain_yaml:
        terrain_gen_cfg = get_terrain_cfg(yaml_path=terrain_yaml)
    else:
        terrain_gen_cfg = get_terrain_cfg(preset=terrain_name)

    # --- Robot ArticulationCfg from URDF ---
    robot_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=robot_meta.urdf_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
            ),
            activate_contact_sensors=True,
            fix_base=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, robot_meta.standing_height + 0.05),
        ),
    )

    # --- Scene ---
    @configclass
    class CustomSceneCfg(InteractiveSceneCfg):
        if is_flat:
            terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                terrain_generator=None,
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
                debug_vis=False,
            )
        else:
            terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=terrain_gen_cfg,
                max_init_terrain_level=5,
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                    project_uvw=True,
                    texture_scale=(0.25, 0.25),
                ),
                debug_vis=False,
            )

        robot: ArticulationCfg = robot_cfg

        contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=True,
        )

        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )

    # Add height scanner only if requested and not flat terrain
    if use_height_scan and not is_flat:
        CustomSceneCfg.height_scanner = RayCasterCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{robot_meta.base_body_name}",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
    else:
        CustomSceneCfg.height_scanner = None

    # --- Commands ---
    @configclass
    class CommandsCfg:
        base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
        )

    # --- Actions ---
    @configclass
    class ActionsCfg:
        joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.25,
            use_default_offset=True,
        )

    # --- Observations ---
    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
            projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
            velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

    # Add height scan observation if height scanner is present
    if use_height_scan and not is_flat:
        ObservationsCfg.PolicyCfg.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

    # --- Events ---
    @configclass
    class EventCfg:
        physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.8, 0.8),
                "dynamic_friction_range": (0.6, 0.6),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )

        add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=robot_meta.base_body_name),
                "mass_distribution_params": (-1.0, 3.0),
                "operation": "add",
            },
        )

        base_external_force_torque = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=robot_meta.base_body_name),
                "force_range": (0.0, 0.0),
                "torque_range": (-0.0, 0.0),
            },
        )

        reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                },
            },
        )

        reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (1.0, 1.0),
                "velocity_range": (0.0, 0.0),
            },
        )

        push_robot = None

    # --- Rewards ---
    @configclass
    class RewardsCfg:
        track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_exp,
            weight=1.5,
            params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        )
        track_ang_vel_z_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=0.75,
            params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        )
        lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
        dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
        dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
        action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
        dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.5)

        feet_air_time = RewTerm(
            func=locomotion_rewards.feet_air_time,
            weight=0.01,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_regex),
                "command_name": "base_velocity",
                "threshold": 0.5,
            },
        )

        base_height = RewTerm(
            func=locomotion_rewards.base_height_target,
            weight=-0.5,
            params={"target_height": robot_meta.standing_height},
        )

        feet_stumble = RewTerm(
            func=locomotion_rewards.feet_stumble,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_regex),
            },
        )

    # --- Terminations ---
    @configclass
    class TerminationsCfg:
        time_out = DoneTerm(func=mdp.time_out, time_out=True)
        base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=robot_meta.base_body_name),
                "threshold": 1.0,
            },
        )

    # --- Curriculum ---
    @configclass
    class CurriculumCfg:
        pass

    # --- Assemble top-level config ---
    @configclass
    class CustomEnvCfg(ManagerBasedRLEnvCfg):
        scene: CustomSceneCfg = CustomSceneCfg(num_envs=num_envs, env_spacing=2.5)
        observations: ObservationsCfg = ObservationsCfg()
        actions: ActionsCfg = ActionsCfg()
        commands: CommandsCfg = CommandsCfg()
        rewards: RewardsCfg = RewardsCfg()
        terminations: TerminationsCfg = TerminationsCfg()
        events: EventCfg = EventCfg()
        curriculum: CurriculumCfg = CurriculumCfg()

        def __post_init__(self):
            self.decimation = 4
            self.episode_length_s = 20.0
            self.sim.dt = 0.005
            self.sim.render_interval = self.decimation
            self.sim.physics_material = self.scene.terrain.physics_material
            self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
            if self.scene.height_scanner is not None:
                self.scene.height_scanner.update_period = self.decimation * self.sim.dt
            if self.scene.contact_forces is not None:
                self.scene.contact_forces.update_period = self.sim.dt
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    return CustomEnvCfg()


def compute_obs_dim(robot_meta: RobotMetadata, use_height_scan: bool = True) -> int:
    """Compute the expected observation dimension for a robot configuration.

    Base observations:
        base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
        + joint_pos(num_dof) + joint_vel(num_dof) + actions(num_dof)
    Height scan: 187 (from 1.6m x 1.0m grid at 0.1m resolution)
    """
    base_obs = 3 + 3 + 3 + 3  # lin_vel, ang_vel, gravity, commands
    joint_obs = robot_meta.num_dof * 3  # pos, vel, actions
    height_scan = 187 if use_height_scan else 0
    return base_obs + joint_obs + height_scan

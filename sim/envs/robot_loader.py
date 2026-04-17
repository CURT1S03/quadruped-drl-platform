# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Utility to load custom robot configurations from URDF + metadata JSON.

Expected directory layout for a custom robot:
    sim/assets/robots/<robot_name>/
        robot.urdf          — the URDF file
        metadata.json       — sidecar with foot names, standing height, etc.

metadata.json schema:
{
    "name": "my_robot",
    "foot_body_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    "base_body_name": "base",
    "standing_height": 0.34,
    "num_legs": 4
}
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RobotMetadata:
    """Parsed robot metadata used when building custom env configs."""

    name: str
    urdf_path: str
    foot_body_names: list[str]
    base_body_name: str = "base"
    standing_height: float = 0.34
    num_legs: int = 4
    num_dof: int = 0  # auto-detected from URDF if 0


def _count_revolute_joints(urdf_path: str | Path) -> int:
    """Count the number of revolute/continuous joints in a URDF file."""
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    count = 0
    for joint in root.findall("joint"):
        jtype = joint.get("type", "fixed")
        if jtype in ("revolute", "continuous", "prismatic"):
            count += 1
    return count


def _validate_urdf(urdf_path: str | Path) -> list[str]:
    """Run basic validation on a URDF file. Returns list of error messages."""
    errors = []
    path = Path(urdf_path)
    if not path.exists():
        errors.append(f"URDF file not found: {path}")
        return errors
    if path.suffix.lower() not in (".urdf", ".xacro"):
        errors.append(f"Unexpected file extension: {path.suffix}")

    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
    except ET.ParseError as e:
        errors.append(f"XML parse error: {e}")
        return errors

    if root.tag != "robot":
        errors.append(f"Root element is '{root.tag}', expected 'robot'")

    links = {link.get("name") for link in root.findall("link")}
    if not links:
        errors.append("No <link> elements found in URDF")

    joints = root.findall("joint")
    movable = [j for j in joints if j.get("type", "fixed") != "fixed"]
    if not movable:
        errors.append("No movable joints found in URDF (all are fixed)")

    # Check parent/child references
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and parent.get("link") not in links:
            errors.append(f"Joint '{joint.get('name')}' references unknown parent link '{parent.get('link')}'")
        if child is not None and child.get("link") not in links:
            errors.append(f"Joint '{joint.get('name')}' references unknown child link '{child.get('link')}'")

    return errors


def load_robot_metadata(robot_dir: str | Path) -> RobotMetadata:
    """Load robot metadata from a robot asset directory.

    Args:
        robot_dir: Path to directory containing robot.urdf and metadata.json.

    Returns:
        Populated RobotMetadata with auto-detected DOF count.

    Raises:
        FileNotFoundError: If required files are missing.
        ValueError: If URDF validation fails.
    """
    robot_dir = Path(robot_dir)

    # Find URDF file
    urdf_path = robot_dir / "robot.urdf"
    if not urdf_path.exists():
        # Try to find any .urdf file
        urdf_files = list(robot_dir.glob("*.urdf"))
        if not urdf_files:
            raise FileNotFoundError(f"No URDF file found in {robot_dir}")
        urdf_path = urdf_files[0]

    # Validate URDF
    errors = _validate_urdf(urdf_path)
    if errors:
        raise ValueError(f"URDF validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    # Load metadata
    meta_path = robot_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {robot_dir}. "
            "Create one with: name, foot_body_names, base_body_name, standing_height, num_legs"
        )

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Auto-detect DOF from URDF
    num_dof = _count_revolute_joints(urdf_path)

    return RobotMetadata(
        name=meta.get("name", robot_dir.name),
        urdf_path=str(urdf_path.resolve()),
        foot_body_names=meta["foot_body_names"],
        base_body_name=meta.get("base_body_name", "base"),
        standing_height=meta.get("standing_height", 0.34),
        num_legs=meta.get("num_legs", 4),
        num_dof=num_dof,
    )


def list_available_robots(assets_dir: str | Path) -> list[dict]:
    """List all robot directories that have a valid metadata.json.

    Returns list of dicts with name, path, num_dof, standing_height.
    """
    assets_dir = Path(assets_dir)
    robots = []

    if not assets_dir.exists():
        return robots

    for sub in sorted(assets_dir.iterdir()):
        if sub.is_dir() and (sub / "metadata.json").exists():
            try:
                meta = load_robot_metadata(sub)
                robots.append({
                    "name": meta.name,
                    "path": str(sub),
                    "num_dof": meta.num_dof,
                    "standing_height": meta.standing_height,
                    "num_legs": meta.num_legs,
                    "foot_body_names": meta.foot_body_names,
                })
            except (ValueError, FileNotFoundError):
                continue  # Skip invalid robot dirs

    return robots

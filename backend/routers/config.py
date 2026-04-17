# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Config router — expose default parameters, available tasks, robots, and terrains."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from backend.config import settings
from backend.schemas import DefaultConfig, RobotInfo, TerrainInfo

router = APIRouter(prefix="/api/config", tags=["config"])

ROBOTS_DIR = settings.project_root / "sim" / "assets" / "robots"
TERRAINS_DIR = settings.project_root / "sim" / "assets" / "terrains"


@router.get("/defaults", response_model=DefaultConfig)
async def get_defaults():
    return DefaultConfig(
        default_num_envs=settings.default_num_envs,
        default_max_iterations=settings.default_max_iterations,
    )


@router.get("/robots", response_model=list[RobotInfo])
async def list_robots():
    """List all available custom robot configurations."""
    robots = []
    if not ROBOTS_DIR.exists():
        return robots

    for sub in sorted(ROBOTS_DIR.iterdir()):
        meta_path = sub / "metadata.json"
        if sub.is_dir() and meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                # Count DOF from URDF if present
                num_dof = meta.get("num_dof", 0)
                if num_dof == 0:
                    urdf_path = sub / "robot.urdf"
                    if not urdf_path.exists():
                        urdfs = list(sub.glob("*.urdf"))
                        urdf_path = urdfs[0] if urdfs else None
                    if urdf_path and urdf_path.exists():
                        import xml.etree.ElementTree as ET
                        tree = ET.parse(str(urdf_path))
                        root = tree.getroot()
                        num_dof = sum(
                            1 for j in root.findall("joint")
                            if j.get("type", "fixed") in ("revolute", "continuous", "prismatic")
                        )

                robots.append(RobotInfo(
                    name=meta.get("name", sub.name),
                    path=str(sub),
                    num_dof=num_dof,
                    standing_height=meta.get("standing_height", 0.34),
                    num_legs=meta.get("num_legs", 4),
                    foot_body_names=meta.get("foot_body_names", []),
                ))
            except (json.JSONDecodeError, KeyError):
                continue

    return robots


@router.get("/terrains", response_model=list[TerrainInfo])
async def list_terrains():
    """List all available terrain presets and custom terrain configs."""
    terrains = []

    # Built-in presets
    for name in ["flat", "easy", "obstacle", "hard", "stairs", "slopes"]:
        terrains.append(TerrainInfo(name=name, type="preset"))

    # Custom YAML files
    if TERRAINS_DIR.exists():
        for f in sorted(TERRAINS_DIR.iterdir()):
            if f.suffix in (".yaml", ".yml") and f.is_file():
                terrains.append(TerrainInfo(name=f.stem, type="custom", path=str(f)))

    return terrains


@router.post("/upload-robot", response_model=RobotInfo)
async def upload_robot(
    name: str = Form(...),
    urdf: UploadFile = File(...),
    metadata: UploadFile = File(...),
):
    """Upload a custom robot URDF and metadata.json."""
    # Sanitize name
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_").strip()
    if not safe_name:
        raise HTTPException(400, "Invalid robot name")

    robot_dir = ROBOTS_DIR / safe_name
    if robot_dir.exists():
        raise HTTPException(409, f"Robot '{safe_name}' already exists")

    robot_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Validate and save metadata
        meta_content = await metadata.read()
        meta_dict = json.loads(meta_content)
        if "foot_body_names" not in meta_dict:
            raise HTTPException(400, "metadata.json must contain 'foot_body_names'")

        meta_path = robot_dir / "metadata.json"
        with open(meta_path, "wb") as f:
            f.write(meta_content)

        # Save URDF
        urdf_content = await urdf.read()
        urdf_filename = urdf.filename or "robot.urdf"
        if not urdf_filename.endswith((".urdf", ".xacro")):
            raise HTTPException(400, "URDF file must have .urdf or .xacro extension")
        urdf_path = robot_dir / "robot.urdf"
        with open(urdf_path, "wb") as f:
            f.write(urdf_content)

        # Validate URDF XML
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(str(urdf_path))
            root = tree.getroot()
            if root.tag != "robot":
                raise HTTPException(400, "URDF root element must be 'robot'")
        except ET.ParseError as e:
            raise HTTPException(400, f"Invalid URDF XML: {e}")

        # Count DOF
        num_dof = sum(
            1 for j in root.findall("joint")
            if j.get("type", "fixed") in ("revolute", "continuous", "prismatic")
        )
        if num_dof == 0:
            raise HTTPException(400, "URDF has no movable joints")

        return RobotInfo(
            name=meta_dict.get("name", safe_name),
            path=str(robot_dir),
            num_dof=num_dof,
            standing_height=meta_dict.get("standing_height", 0.34),
            num_legs=meta_dict.get("num_legs", 4),
            foot_body_names=meta_dict["foot_body_names"],
        )

    except HTTPException:
        # Clean up on validation error
        shutil.rmtree(robot_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(robot_dir, ignore_errors=True)
        raise HTTPException(500, f"Failed to process robot upload: {e}")


@router.post("/upload-terrain", response_model=TerrainInfo)
async def upload_terrain(
    name: str = Form(...),
    terrain_yaml: UploadFile = File(...),
):
    """Upload a custom terrain YAML configuration."""
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_").strip()
    if not safe_name:
        raise HTTPException(400, "Invalid terrain name")

    TERRAINS_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = TERRAINS_DIR / f"{safe_name}.yaml"

    if yaml_path.exists():
        raise HTTPException(409, f"Terrain '{safe_name}' already exists")

    try:
        content = await terrain_yaml.read()

        # Validate YAML structure
        import yaml
        data = yaml.safe_load(content)
        if not isinstance(data, dict) or "sub_terrains" not in data:
            raise HTTPException(400, "Terrain YAML must contain 'sub_terrains' key")

        for sub_name, sub_cfg in data["sub_terrains"].items():
            if "type" not in sub_cfg:
                raise HTTPException(400, f"Sub-terrain '{sub_name}' missing 'type' field")
            if "proportion" not in sub_cfg:
                raise HTTPException(400, f"Sub-terrain '{sub_name}' missing 'proportion' field")

        with open(yaml_path, "wb") as f:
            f.write(content)

        return TerrainInfo(name=safe_name, type="custom", path=str(yaml_path))

    except HTTPException:
        yaml_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        yaml_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Failed to process terrain upload: {e}")

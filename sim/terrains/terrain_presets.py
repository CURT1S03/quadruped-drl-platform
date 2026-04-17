# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Terrain presets and YAML-based terrain configuration loading.

Provides named presets (flat, easy, obstacle, hard, stairs, slopes) and
a function to load custom terrain configs from YAML files.

YAML terrain config schema:
    size: [8.0, 8.0]
    border_width: 20.0
    num_rows: 10
    num_cols: 20
    sub_terrains:
      pyramid_stairs:
        type: MeshPyramidStairsTerrainCfg
        proportion: 0.3
        step_height_range: [0.05, 0.18]
        step_width: 0.3
      random_rough:
        type: HfRandomUniformTerrainCfg
        proportion: 0.4
        noise_range: [0.01, 0.06]
      ...
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from sim.terrains.obstacle_terrain_cfg import OBSTACLE_TERRAINS_CFG

# ─── Map from YAML type names to Isaac Lab terrain cfg classes ──────────── #
TERRAIN_TYPE_MAP = {
    "MeshPyramidStairsTerrainCfg": terrain_gen.MeshPyramidStairsTerrainCfg,
    "MeshInvertedPyramidStairsTerrainCfg": terrain_gen.MeshInvertedPyramidStairsTerrainCfg,
    "MeshRandomGridTerrainCfg": terrain_gen.MeshRandomGridTerrainCfg,
    "HfRandomUniformTerrainCfg": terrain_gen.HfRandomUniformTerrainCfg,
    "HfPyramidSlopedTerrainCfg": terrain_gen.HfPyramidSlopedTerrainCfg,
    "HfInvertedPyramidSlopedTerrainCfg": terrain_gen.HfInvertedPyramidSlopedTerrainCfg,
}


# ─── Named presets ─────────────────────────────────────────────────────────── #

FLAT_TERRAIN_CFG = None  # Signals to use terrain_type="plane"

EASY_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5,
            noise_range=(0.005, 0.02),
            noise_step=0.005,
            border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.3,
            slope_range=(0.0, 0.15),
            platform_width=2.0,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.45,
            grid_height_range=(0.01, 0.04),
            platform_width=2.0,
        ),
    },
)
"""Easy terrain — gentle rough ground, mild slopes, and low boxes."""

HARD_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.1, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.1, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.45,
            grid_height_range=(0.05, 0.18),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.04, 0.1),
            noise_step=0.01,
            border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.2, 0.6),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)
"""Hard terrain — tall stairs, high boxes, steep slopes, aggressive rough ground."""

STAIRS_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)
"""Stairs-only terrain — ascending and descending stairs."""

SLOPES_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.0, 0.5),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.0, 0.5),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)
"""Slopes-only terrain — ascending and descending slopes."""

# ─── Preset registry ──────────────────────────────────────────────────────── #
TERRAIN_PRESETS: dict[str, TerrainGeneratorCfg | None] = {
    "flat": FLAT_TERRAIN_CFG,
    "easy": EASY_TERRAIN_CFG,
    "obstacle": OBSTACLE_TERRAINS_CFG,
    "hard": HARD_TERRAIN_CFG,
    "stairs": STAIRS_TERRAIN_CFG,
    "slopes": SLOPES_TERRAIN_CFG,
}


def list_terrain_presets() -> list[str]:
    """Return the names of all available terrain presets."""
    return list(TERRAIN_PRESETS.keys())


def _load_terrain_from_yaml(yaml_path: str | Path) -> TerrainGeneratorCfg:
    """Load a terrain configuration from a YAML file."""
    import yaml

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Terrain YAML not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    sub_terrains = {}
    for name, cfg in data.get("sub_terrains", {}).items():
        terrain_type = cfg.pop("type")
        if terrain_type not in TERRAIN_TYPE_MAP:
            raise ValueError(
                f"Unknown terrain type '{terrain_type}'. "
                f"Available: {list(TERRAIN_TYPE_MAP.keys())}"
            )
        cls = TERRAIN_TYPE_MAP[terrain_type]
        # Convert list values to tuples where needed
        for key, val in cfg.items():
            if isinstance(val, list):
                cfg[key] = tuple(val)
        sub_terrains[name] = cls(**cfg)

    return TerrainGeneratorCfg(
        size=tuple(data.get("size", [8.0, 8.0])),
        border_width=data.get("border_width", 20.0),
        num_rows=data.get("num_rows", 10),
        num_cols=data.get("num_cols", 20),
        horizontal_scale=data.get("horizontal_scale", 0.1),
        vertical_scale=data.get("vertical_scale", 0.005),
        slope_threshold=data.get("slope_threshold", 0.75),
        use_cache=data.get("use_cache", False),
        sub_terrains=sub_terrains,
    )


def get_terrain_cfg(
    preset: str | None = None,
    yaml_path: str | Path | None = None,
) -> TerrainGeneratorCfg | None:
    """Get a terrain configuration by preset name or YAML path.

    Args:
        preset: Name of a terrain preset.
        yaml_path: Path to a custom terrain YAML file (overrides preset).

    Returns:
        TerrainGeneratorCfg or None (for flat terrain).
    """
    if yaml_path:
        return _load_terrain_from_yaml(yaml_path)

    if preset:
        if preset not in TERRAIN_PRESETS:
            raise ValueError(f"Unknown terrain preset '{preset}'. Available: {list(TERRAIN_PRESETS.keys())}")
        return TERRAIN_PRESETS[preset]

    return OBSTACLE_TERRAINS_CFG  # Default


def list_available_terrains(assets_dir: str | Path) -> list[dict]:
    """List all terrain presets AND custom YAML terrain configs from assets dir.

    Returns list of dicts with name, type ("preset" or "custom"), and path (for custom).
    """
    terrains = []

    # Built-in presets
    for name in TERRAIN_PRESETS:
        terrains.append({"name": name, "type": "preset", "path": None})

    # Custom YAML files from assets directory
    assets_dir = Path(assets_dir)
    if assets_dir.exists():
        for yaml_file in sorted(assets_dir.glob("*.yaml")):
            terrains.append({
                "name": yaml_file.stem,
                "type": "custom",
                "path": str(yaml_file),
            })
        for yaml_file in sorted(assets_dir.glob("*.yml")):
            terrains.append({
                "name": yaml_file.stem,
                "type": "custom",
                "path": str(yaml_file),
            })

    return terrains

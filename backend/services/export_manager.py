# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Export manager — handles policy export to TorchScript + metadata bundles."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from backend.config import settings

logger = logging.getLogger(__name__)


def export_checkpoint(
    checkpoint_path: str,
    output_path: str | None = None,
    obs_dim: int | None = None,
) -> str:
    """Run the export_policy.py script to produce a .zip bundle.

    Args:
        checkpoint_path: Path to the RSL-RL model checkpoint (.pt file).
        output_path: Where to save the export. Defaults to alongside the checkpoint.
        obs_dim: Override observation dimension. Auto-detected if None.

    Returns:
        Path to the exported .zip bundle.

    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
        RuntimeError: If export subprocess fails.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(checkpoint_path), "exported_policy.zip"
        )

    export_script = str(settings.project_root / "sim" / "scripts" / "export_policy.py")

    cmd = [
        sys.executable,
        export_script,
        "--checkpoint", checkpoint_path,
        "--output", output_path,
    ]
    if obs_dim is not None:
        cmd.extend(["--obs_dim", str(obs_dim)])

    logger.info(f"Exporting policy: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(settings.project_root),
    )

    if result.returncode != 0:
        logger.error(f"Export failed: {result.stderr}")
        raise RuntimeError(f"Export failed: {result.stderr or result.stdout}")

    logger.info(f"Export output: {result.stdout.strip()}")

    if not os.path.isfile(output_path):
        raise RuntimeError(f"Export completed but output file not found: {output_path}")

    return output_path


def get_export_metadata(export_path: str) -> dict | None:
    """Read metadata from an exported policy zip bundle.

    Returns the parsed metadata dict, or None if not a valid export.
    """
    import zipfile

    if not os.path.isfile(export_path):
        return None

    if export_path.endswith(".zip"):
        try:
            with zipfile.ZipFile(export_path, "r") as zf:
                if "metadata.json" in zf.namelist():
                    return json.loads(zf.read("metadata.json"))
        except (zipfile.BadZipFile, json.JSONDecodeError):
            return None

    # Try companion metadata file
    meta_path = export_path.replace(".pt", "_metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)

    return None

# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Checkpoint manager — scans disk for saved policy checkpoints."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from backend.config import settings

# RSL-RL saves checkpoints as model_{iteration}.pt
CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt$")


@dataclass
class CheckpointInfo:
    iteration: int
    file_path: str
    file_size_bytes: int
    created_at: datetime


def scan_checkpoints(log_dir: str | Path) -> list[CheckpointInfo]:
    """Scan a log directory for RSL-RL checkpoint files."""
    log_dir = Path(log_dir)
    checkpoints = []

    if not log_dir.exists():
        return checkpoints

    for pt_file in log_dir.rglob("model_*.pt"):
        match = CHECKPOINT_PATTERN.search(pt_file.name)
        if match:
            iteration = int(match.group(1))
            stat = pt_file.stat()
            checkpoints.append(
                CheckpointInfo(
                    iteration=iteration,
                    file_path=str(pt_file),
                    file_size_bytes=stat.st_size,
                    created_at=datetime.fromtimestamp(stat.st_mtime),
                )
            )

    checkpoints.sort(key=lambda c: c.iteration)
    return checkpoints


def find_best_checkpoint(log_dir: str | Path) -> CheckpointInfo | None:
    """Find the latest checkpoint in a log directory."""
    checkpoints = scan_checkpoints(log_dir)
    return checkpoints[-1] if checkpoints else None

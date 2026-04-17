# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Config router — expose default parameters and available tasks."""

from __future__ import annotations

from fastapi import APIRouter

from backend.config import settings
from backend.schemas import DefaultConfig

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("/defaults", response_model=DefaultConfig)
async def get_defaults():
    return DefaultConfig(
        default_num_envs=settings.default_num_envs,
        default_max_iterations=settings.default_max_iterations,
    )

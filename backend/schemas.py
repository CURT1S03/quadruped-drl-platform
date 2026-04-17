# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ─── Training ──────────────────────────────────────────────────────────────── #

class TrainingStartRequest(BaseModel):
    name: str = Field(default="", description="Human-friendly run name.")
    task: str = Field(default="Go2-Obstacle-v0", description="Gym task id.")
    num_envs: int = Field(default=4096, ge=1, le=65536)
    max_iterations: int = Field(default=1500, ge=1, le=100000)
    learning_rate: float = Field(default=1e-3, gt=0, le=1.0)
    headless: bool = Field(default=True)


class TrainingStatusResponse(BaseModel):
    state: str
    run_id: int | None = None
    iteration: int = 0
    max_iterations: int = 0
    log_dir: str | None = None


class RunSummary(BaseModel):
    id: int
    name: str
    status: str
    task: str
    num_envs: int
    best_reward: float | None
    total_iterations: int
    created_at: datetime
    finished_at: datetime | None

    class Config:
        from_attributes = True


class RunDetail(RunSummary):
    config_json: str
    log_dir: str | None
    checkpoints: list[CheckpointResponse] = []

    class Config:
        from_attributes = True


# ─── Checkpoints ───────────────────────────────────────────────────────────── #

class CheckpointResponse(BaseModel):
    id: int
    run_id: int
    iteration: int
    file_path: str
    mean_reward: float | None
    created_at: datetime

    class Config:
        from_attributes = True


class EvaluateRequest(BaseModel):
    task: str = Field(default="Go2-Obstacle-Play-v0")
    num_envs: int = Field(default=4, ge=1, le=256)
    num_steps: int = Field(default=2000, ge=100, le=50000)


# ─── Metrics ───────────────────────────────────────────────────────────────── #

class MetricEntry(BaseModel):
    iteration: int
    metric_name: str
    metric_value: float
    timestamp: datetime

    class Config:
        from_attributes = True


# ─── Config ────────────────────────────────────────────────────────────────── #

class DefaultConfig(BaseModel):
    tasks: list[str] = [
        "Go2-Obstacle-v0",
        "Go2-Flat-v0",
        "Go2-Obstacle-Play-v0",
        "Go2-Flat-Play-v0",
    ]
    default_num_envs: int
    default_max_iterations: int
    default_learning_rate: float = 1e-3
    terrain_types: list[str] = ["obstacle", "flat"]


# Forward reference resolution
RunDetail.model_rebuild()

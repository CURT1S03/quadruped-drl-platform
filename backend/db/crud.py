# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""CRUD operations for training runs, checkpoints, and metrics."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.db.models import Checkpoint, Metric, TrainingRun


# ─── Training Runs ─────────────────────────────────────────────────────────── #

async def mark_stale_runs_failed(db: AsyncSession) -> int:
    """Mark any 'running' or 'queued' runs as 'failed' (cleanup after crash/restart)."""
    stmt = (
        update(TrainingRun)
        .where(TrainingRun.status.in_(["running", "queued"]))
        .values(status="failed", finished_at=datetime.utcnow())
    )
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount

async def create_run(
    db: AsyncSession,
    name: str,
    task: str,
    config_json: str,
    num_envs: int,
    log_dir: str | None = None,
) -> TrainingRun:
    run = TrainingRun(
        name=name,
        task=task,
        config_json=config_json,
        num_envs=num_envs,
        log_dir=log_dir,
        status="queued",
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return run


async def get_run(db: AsyncSession, run_id: int) -> TrainingRun | None:
    stmt = (
        select(TrainingRun)
        .where(TrainingRun.id == run_id)
        .options(selectinload(TrainingRun.checkpoints), selectinload(TrainingRun.metrics))
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def list_runs(db: AsyncSession, limit: int = 50, offset: int = 0) -> Sequence[TrainingRun]:
    stmt = (
        select(TrainingRun)
        .order_by(TrainingRun.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def update_run_status(
    db: AsyncSession,
    run_id: int,
    status: str,
    best_reward: float | None = None,
    total_iterations: int | None = None,
    finished_at: datetime | None = None,
) -> TrainingRun | None:
    run = await db.get(TrainingRun, run_id)
    if run is None:
        return None
    run.status = status
    if best_reward is not None:
        run.best_reward = best_reward
    if total_iterations is not None:
        run.total_iterations = total_iterations
    if finished_at is not None:
        run.finished_at = finished_at
    await db.commit()
    await db.refresh(run)
    return run


# ─── Checkpoints ───────────────────────────────────────────────────────────── #

async def create_checkpoint(
    db: AsyncSession,
    run_id: int,
    iteration: int,
    file_path: str,
    mean_reward: float | None = None,
) -> Checkpoint:
    ckpt = Checkpoint(
        run_id=run_id,
        iteration=iteration,
        file_path=file_path,
        mean_reward=mean_reward,
    )
    db.add(ckpt)
    await db.commit()
    await db.refresh(ckpt)
    return ckpt


async def list_checkpoints(
    db: AsyncSession,
    run_id: int | None = None,
) -> Sequence[Checkpoint]:
    stmt = select(Checkpoint).order_by(Checkpoint.created_at.desc())
    if run_id is not None:
        stmt = stmt.where(Checkpoint.run_id == run_id)
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_checkpoint(db: AsyncSession, checkpoint_id: int) -> Checkpoint | None:
    return await db.get(Checkpoint, checkpoint_id)


# ─── Metrics ───────────────────────────────────────────────────────────────── #

async def create_metrics_batch(
    db: AsyncSession,
    run_id: int,
    iteration: int,
    metrics: dict[str, float],
) -> list[Metric]:
    records = []
    for name, value in metrics.items():
        m = Metric(run_id=run_id, iteration=iteration, metric_name=name, metric_value=value)
        db.add(m)
        records.append(m)
    await db.commit()
    return records


async def get_metrics_for_run(
    db: AsyncSession,
    run_id: int,
    metric_name: str | None = None,
) -> Sequence[Metric]:
    stmt = select(Metric).where(Metric.run_id == run_id).order_by(Metric.iteration)
    if metric_name:
        stmt = stmt.where(Metric.metric_name == metric_name)
    result = await db.execute(stmt)
    return result.scalars().all()

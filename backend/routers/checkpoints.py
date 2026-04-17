# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Checkpoint router — list, inspect, and evaluate saved policy checkpoints."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import crud
from backend.db.database import get_db
from backend.schemas import CheckpointResponse, EvaluateRequest
from backend.services.checkpoint_manager import scan_checkpoints
from backend.services.sim_manager import SimManager, SimState

router = APIRouter(prefix="/api/checkpoints", tags=["checkpoints"])

_sim_manager: SimManager | None = None


def set_services(sim_manager: SimManager):
    global _sim_manager
    _sim_manager = sim_manager


@router.get("/", response_model=list[CheckpointResponse])
async def list_checkpoints(
    run_id: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    checkpoints = await crud.list_checkpoints(db, run_id=run_id)
    return checkpoints


@router.get("/{checkpoint_id}", response_model=CheckpointResponse)
async def get_checkpoint(checkpoint_id: int, db: AsyncSession = Depends(get_db)):
    ckpt = await crud.get_checkpoint(db, checkpoint_id)
    if not ckpt:
        raise HTTPException(404, "Checkpoint not found")
    return ckpt


@router.post("/{checkpoint_id}/evaluate")
async def evaluate_checkpoint(
    checkpoint_id: int,
    req: EvaluateRequest = EvaluateRequest(),
    db: AsyncSession = Depends(get_db),
):
    if _sim_manager is None:
        raise HTTPException(503, "Services not initialized")
    if _sim_manager.state != SimState.IDLE:
        raise HTTPException(409, f"Cannot evaluate while in state: {_sim_manager.state}")

    ckpt = await crud.get_checkpoint(db, checkpoint_id)
    if not ckpt:
        raise HTTPException(404, "Checkpoint not found")
    if not os.path.isfile(ckpt.file_path):
        raise HTTPException(404, f"Checkpoint file not found on disk: {ckpt.file_path}")

    try:
        _sim_manager.start_evaluation(
            checkpoint_path=ckpt.file_path,
            task=req.task,
            num_envs=req.num_envs,
            num_steps=req.num_steps,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e))

    return {"status": "evaluating", "checkpoint_id": checkpoint_id, "file_path": ckpt.file_path}


@router.post("/scan/{run_id}")
async def scan_and_register_checkpoints(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Scan the run's log directory for new checkpoint files and register them in the DB."""
    run = await crud.get_run(db, run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    if not run.log_dir:
        raise HTTPException(400, "Run has no log directory")

    disk_checkpoints = scan_checkpoints(run.log_dir)
    existing = await crud.list_checkpoints(db, run_id=run_id)
    existing_iters = {c.iteration for c in existing}

    created = []
    for ckpt_info in disk_checkpoints:
        if ckpt_info.iteration not in existing_iters:
            ckpt = await crud.create_checkpoint(
                db,
                run_id=run_id,
                iteration=ckpt_info.iteration,
                file_path=ckpt_info.file_path,
            )
            created.append(ckpt)

    return {"scanned": len(disk_checkpoints), "new": len(created)}

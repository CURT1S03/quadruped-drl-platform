# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Checkpoint router — list, inspect, evaluate, export, and download saved policy checkpoints."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import crud
from backend.db.database import get_db
from backend.schemas import CheckpointResponse, EvaluateRequest, ExportResponse
from backend.services.checkpoint_manager import scan_checkpoints
from backend.services.export_manager import export_checkpoint, get_export_metadata
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


@router.post("/{checkpoint_id}/export", response_model=ExportResponse)
async def export_policy(
    checkpoint_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Export a checkpoint to a TorchScript policy bundle (.zip with policy.pt + metadata.json)."""
    ckpt = await crud.get_checkpoint(db, checkpoint_id)
    if not ckpt:
        raise HTTPException(404, "Checkpoint not found")
    if not os.path.isfile(ckpt.file_path):
        raise HTTPException(404, f"Checkpoint file not found on disk: {ckpt.file_path}")

    try:
        export_path = export_checkpoint(ckpt.file_path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    metadata = get_export_metadata(export_path) or {}

    return ExportResponse(
        run_id=ckpt.run_id,
        checkpoint_id=checkpoint_id,
        export_path=export_path,
        obs_dim=metadata.get("obs_dim", 235),
        action_dim=metadata.get("action_dim", 12),
        robot_name=metadata.get("robot_name", "go2"),
    )


@router.get("/{checkpoint_id}/download")
async def download_export(
    checkpoint_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Download the exported policy bundle for a checkpoint."""
    ckpt = await crud.get_checkpoint(db, checkpoint_id)
    if not ckpt:
        raise HTTPException(404, "Checkpoint not found")

    # Look for existing export alongside the checkpoint
    ckpt_dir = os.path.dirname(ckpt.file_path)
    export_path = os.path.join(ckpt_dir, "exported_policy.zip")

    if not os.path.isfile(export_path):
        # Try to export on-demand
        try:
            export_path = export_checkpoint(ckpt.file_path)
        except Exception as e:
            raise HTTPException(500, f"Export failed: {e}")

    if not os.path.isfile(export_path):
        raise HTTPException(404, "Export file not found")

    filename = f"policy_iter{ckpt.iteration}.zip"
    return FileResponse(
        path=export_path,
        media_type="application/zip",
        filename=filename,
    )

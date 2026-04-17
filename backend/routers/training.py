# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Training router — start, stop, and query training status."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import crud
from backend.db.database import get_db
from backend.schemas import RunDetail, RunSummary, TrainingStartRequest, TrainingStatusResponse
from backend.services.sim_manager import SimManager, SimState
from backend.services.telemetry_collector import TelemetryCollector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])

# These are injected from main.py via the app state
_sim_manager: SimManager | None = None
_telemetry: TelemetryCollector | None = None


def set_services(sim_manager: SimManager, telemetry: TelemetryCollector):
    global _sim_manager, _telemetry
    _sim_manager = sim_manager
    _telemetry = telemetry


@router.post("/start", response_model=TrainingStatusResponse)
async def start_training(
    req: TrainingStartRequest,
    db: AsyncSession = Depends(get_db),
):
    if _sim_manager is None or _telemetry is None:
        raise HTTPException(503, "Services not initialized")
    if _sim_manager.state != SimState.IDLE:
        raise HTTPException(409, f"Already in state: {_sim_manager.state}")

    # Create DB record
    run_name = req.name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = req.model_dump(mode="json")
    run = await crud.create_run(
        db,
        name=run_name,
        task=req.task,
        config_json=json.dumps(config),
        num_envs=req.num_envs,
    )

    # Reset telemetry for new run
    _telemetry.clear()

    # Define output callback that parses metrics and persists them
    async def _persist_metrics(metrics: dict):
        if "iteration" in metrics and "mean_reward" in metrics:
            async with get_db().__anext__() as session:  # type: ignore
                await crud.create_metrics_batch(session, run.id, metrics["iteration"], metrics)

    def on_output(line: str):
        parsed = _telemetry.process_line(line)
        if parsed:
            logger.info(f"[TELEMETRY] {parsed}")
        if "Learning iteration" in line or "TELEMETRY:" in line or "ERROR" in line.upper():
            logger.info(f"[TRAIN] {line}")

    # Start subprocess
    try:
        # Resolve robot config path if custom robot specified
        robot_config_path = None
        if req.robot_name:
            from backend.config import settings as _settings
            robot_dir = _settings.project_root / "sim" / "assets" / "robots" / req.robot_name
            if not robot_dir.exists():
                raise HTTPException(404, f"Robot '{req.robot_name}' not found")
            robot_config_path = str(robot_dir)

        # Resolve terrain config
        terrain_config = None
        if req.terrain_preset:
            # Check if it's a custom YAML file
            terrain_yaml = _settings.project_root / "sim" / "assets" / "terrains" / f"{req.terrain_preset}.yaml"
            if terrain_yaml.exists():
                terrain_config = str(terrain_yaml)
            else:
                terrain_config = req.terrain_preset  # Preset name

        log_dir = _sim_manager.start_training(
            run_id=run.id,
            task=req.task,
            num_envs=req.num_envs,
            max_iterations=req.max_iterations,
            headless=req.headless,
            on_output=on_output,
            robot_config=robot_config_path,
            terrain_config=terrain_config,
            learning_rate=req.learning_rate,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        logger.error(f"Failed to start training subprocess: {e}")
        await crud.update_run_status(db, run.id, "failed", finished_at=datetime.utcnow())
        raise HTTPException(500, f"Failed to start training: {e}")

    # Update run record
    await crud.update_run_status(db, run.id, "running")
    run.log_dir = log_dir
    await db.commit()

    return TrainingStatusResponse(
        state=_sim_manager.state.value,
        run_id=run.id,
        iteration=0,
        max_iterations=req.max_iterations,
        log_dir=log_dir,
    )


@router.post("/stop")
async def stop_training(db: AsyncSession = Depends(get_db)):
    if _sim_manager is None:
        raise HTTPException(503, "Services not initialized")
    if _sim_manager.state == SimState.IDLE:
        raise HTTPException(400, "No training in progress")

    run_id = _sim_manager.current_run_id
    await _sim_manager.stop()

    if run_id:
        await crud.update_run_status(
            db, run_id, "stopped", finished_at=datetime.utcnow()
        )

    return {"status": "stopped", "run_id": run_id}


@router.get("/status", response_model=TrainingStatusResponse)
async def get_status():
    if _sim_manager is None or _telemetry is None:
        raise HTTPException(503, "Services not initialized")

    # Check if subprocess died
    _sim_manager.poll()

    return TrainingStatusResponse(
        state=_sim_manager.state.value,
        run_id=_sim_manager.current_run_id,
        iteration=_telemetry.current_iteration,
        max_iterations=_telemetry.max_iterations,
        log_dir=_sim_manager.log_dir,
    )


@router.get("/runs", response_model=list[RunSummary])
async def list_runs(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    runs = await crud.list_runs(db, limit=limit, offset=offset)
    return runs


@router.get("/runs/{run_id}", response_model=RunDetail)
async def get_run(run_id: int, db: AsyncSession = Depends(get_db)):
    run = await crud.get_run(db, run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run

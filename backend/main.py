# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""FastAPI application entry point.

Run with:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.db.database import async_session, init_db
from backend.db import crud
from backend.routers import checkpoints, config, telemetry, training
from backend.services.sim_manager import SimManager
from backend.services.telemetry_collector import TelemetryCollector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Shared service instances ──────────────────────────────────────────────── #
sim_manager = SimManager()
telemetry_collector = TelemetryCollector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database ready.")

    # Clean up any runs left as "running" from a previous crash
    async with async_session() as db:
        stale = await crud.mark_stale_runs_failed(db)
        if stale:
            logger.info(f"Marked {stale} stale run(s) as failed.")

    # Wire up services into routers
    training.set_services(sim_manager, telemetry_collector)
    checkpoints.set_services(sim_manager)
    telemetry.set_services(telemetry_collector)

    yield

    # Cleanup: stop any running subprocess
    logger.info("Shutting down...")
    await sim_manager.stop(timeout=10.0)


# ─── App ───────────────────────────────────────────────────────────────────── #
app = FastAPI(
    title="Quadruped DRL Training Platform",
    description="Orchestrate Go2 quadruped reinforcement learning training in Isaac Lab",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(training.router)
app.include_router(checkpoints.router)
app.include_router(telemetry.router)
app.include_router(config.router)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "sim_state": sim_manager.state.value,
        "current_run_id": sim_manager.current_run_id,
    }

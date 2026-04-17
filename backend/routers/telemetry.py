# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""WebSocket telemetry endpoint — streams live training metrics to clients."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.telemetry_collector import TelemetryCollector

logger = logging.getLogger(__name__)

router = APIRouter(tags=["telemetry"])

_telemetry: TelemetryCollector | None = None


def set_services(telemetry: TelemetryCollector):
    global _telemetry
    _telemetry = telemetry


@router.websocket("/ws/telemetry")
async def telemetry_ws(ws: WebSocket):
    if _telemetry is None:
        await ws.close(code=1011, reason="Services not initialized")
        return

    await ws.accept()
    queue = _telemetry.subscribe()
    logger.info("WebSocket client connected for telemetry")

    try:
        while True:
            # Wait for next metrics entry
            try:
                data = await asyncio.wait_for(queue.get(), timeout=30.0)
                await ws.send_text(json.dumps(data))
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                await ws.send_text(json.dumps({"event": "heartbeat"}))
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        _telemetry.unsubscribe(queue)

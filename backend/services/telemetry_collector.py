# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Telemetry collector — parses training output and broadcasts to WebSocket clients."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import deque
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Regex patterns for parsing RSL-RL training output
# RSL-RL 5.x logs lines like:  "  Learning iteration 100/1500  ..."
ITERATION_PATTERN = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
# RSL-RL 5.x: "Mean value loss: 0.1083" (older: "value loss: ...")
REWARD_PATTERN = re.compile(r"(?:mean reward|Mean reward):\s*([-\d.]+)")
EP_LEN_PATTERN = re.compile(r"(?:mean episode length|Mean episode length):\s*([-\d.]+)")
LOSS_PATTERN = re.compile(r"(?:Mean value loss|value loss):\s*([-\d.]+)")
POLICY_LOSS_PATTERN = re.compile(r"(?:Mean surrogate loss|surrogate loss):\s*([-\d.]+)")
LR_PATTERN = re.compile(r"(?:learning rate|Learning rate):\s*([-\d.eE+]+)")
STEPS_PATTERN = re.compile(r"Total steps:\s*(\d+)")
FPS_PATTERN = re.compile(r"Steps per second:\s*(\d+)")
EPISODE_REWARD_PATTERN = re.compile(r"Episode_Reward/(\S+):\s*([-\d.]+)")
# Our custom JSON telemetry prefix
TELEMETRY_PREFIX = "TELEMETRY:"


class TelemetryCollector:
    """Collects metrics from training output and broadcasts to subscribers."""

    def __init__(self, buffer_size: int = 500):
        self._subscribers: set[asyncio.Queue] = set()
        self._buffer: deque[dict[str, Any]] = deque(maxlen=buffer_size)
        self._current_iteration: int = 0
        self._max_iterations: int = 0
        self._lock = asyncio.Lock()
        # Accumulator: collects all metrics for the current iteration block
        self._pending: dict[str, Any] = {}

    def subscribe(self) -> asyncio.Queue:
        """Create a new subscription queue. Returns recent buffer + live updates."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        # Send buffered history
        for entry in self._buffer:
            queue.put_nowait(entry)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    @property
    def latest_metrics(self) -> dict[str, Any] | None:
        return self._buffer[-1] if self._buffer else None

    @property
    def current_iteration(self) -> int:
        return self._current_iteration

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    def process_line(self, line: str) -> dict[str, Any] | None:
        """Parse a single line of training output. Returns metrics dict if found.

        Accumulates metrics across lines for a single iteration block, then
        flushes the combined snapshot when the next iteration header appears
        or a separator line (``----``) is encountered.
        """
        # Check for our custom JSON telemetry (bypass accumulation)
        if line.startswith(TELEMETRY_PREFIX):
            try:
                data = json.loads(line[len(TELEMETRY_PREFIX):])
                self._broadcast(data)
                return data
            except json.JSONDecodeError:
                pass

        # Detect separator lines that mark end of an iteration block
        if line.startswith("------"):
            return self._flush_pending()

        # Parse RSL-RL output patterns
        metrics: dict[str, Any] = {}

        iter_match = ITERATION_PATTERN.search(line)
        if iter_match:
            # New iteration header — flush any previously accumulated data
            self._flush_pending()
            self._current_iteration = int(iter_match.group(1))
            self._max_iterations = int(iter_match.group(2))
            metrics["iteration"] = self._current_iteration
            metrics["max_iterations"] = self._max_iterations

        reward_match = REWARD_PATTERN.search(line)
        if reward_match:
            metrics["mean_reward"] = float(reward_match.group(1))

        ep_len_match = EP_LEN_PATTERN.search(line)
        if ep_len_match:
            metrics["mean_episode_length"] = float(ep_len_match.group(1))

        loss_match = LOSS_PATTERN.search(line)
        if loss_match:
            metrics["value_loss"] = float(loss_match.group(1))

        policy_loss_match = POLICY_LOSS_PATTERN.search(line)
        if policy_loss_match:
            metrics["policy_loss"] = float(policy_loss_match.group(1))

        lr_match = LR_PATTERN.search(line)
        if lr_match:
            metrics["learning_rate"] = float(lr_match.group(1))

        steps_match = STEPS_PATTERN.search(line)
        if steps_match:
            metrics["total_steps"] = int(steps_match.group(1))

        fps_match = FPS_PATTERN.search(line)
        if fps_match:
            metrics["steps_per_second"] = int(fps_match.group(1))

        ep_reward_match = EPISODE_REWARD_PATTERN.search(line)
        if ep_reward_match:
            metrics[f"reward/{ep_reward_match.group(1)}"] = float(ep_reward_match.group(2))

        if metrics:
            # Merge into the accumulator instead of broadcasting immediately
            self._pending.update(metrics)

        return None  # actual broadcast happens on flush

    def _flush_pending(self) -> dict[str, Any] | None:
        """Flush the accumulated pending metrics as a single combined message."""
        if not self._pending:
            return None
        snapshot = self._pending.copy()
        snapshot["timestamp"] = datetime.utcnow().isoformat()
        self._pending.clear()
        self._buffer.append(snapshot)
        self._broadcast(snapshot)
        return snapshot

    def _broadcast(self, data: dict[str, Any]) -> None:
        """Send data to all subscribers. Drop if queue is full."""
        dead_queues = set()
        for queue in self._subscribers:
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                dead_queues.add(queue)
        # Remove dead subscribers
        self._subscribers -= dead_queues

    def clear(self) -> None:
        """Reset state for a new training run."""
        self._flush_pending()
        self._buffer.clear()
        self._pending.clear()
        self._current_iteration = 0
        self._max_iterations = 0

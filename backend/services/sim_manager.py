# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Simulation manager — wraps Isaac Lab training/inference as subprocesses."""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from backend.config import settings

logger = logging.getLogger(__name__)


class SimState(str, enum.Enum):
    IDLE = "idle"
    TRAINING = "training"
    EVALUATING = "evaluating"
    STOPPING = "stopping"


class SimManager:
    """Manages Isaac Lab training/inference subprocess lifecycle."""

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._state: SimState = SimState.IDLE
        self._current_run_id: int | None = None
        self._log_dir: str | None = None
        self._on_output: Callable[[str], None] | None = None
        self._reader_task: asyncio.Task | None = None
        self._headless: bool = True

    @property
    def state(self) -> SimState:
        return self._state

    @property
    def current_run_id(self) -> int | None:
        return self._current_run_id

    @property
    def log_dir(self) -> str | None:
        return self._log_dir

    def start_training(
        self,
        run_id: int,
        task: str = "Go2-Obstacle-v0",
        num_envs: int | None = None,
        max_iterations: int | None = None,
        log_dir: str | None = None,
        headless: bool = True,
        extra_args: dict | None = None,
        on_output: Callable[[str], None] | None = None,
        robot_config: str | None = None,
        terrain_config: str | None = None,
        learning_rate: float | None = None,
    ) -> str:
        """Launch the training subprocess.

        Returns the resolved log directory path.
        """
        if self._state != SimState.IDLE:
            raise RuntimeError(f"Cannot start training while in state: {self._state}")

        # Build the command
        isaaclab_cmd = self._resolve_python()
        train_script = str(settings.project_root / "sim" / "scripts" / "train.py")

        cmd = [*isaaclab_cmd, train_script, "--task", task]
        if headless:
            cmd.append("--headless")
        if num_envs is not None:
            cmd.extend(["--num_envs", str(num_envs)])
        if max_iterations is not None:
            cmd.extend(["--max_iterations", str(max_iterations)])
        if log_dir:
            cmd.extend(["--log_dir", log_dir])
        if robot_config:
            cmd.extend(["--robot_config", robot_config])
        if terrain_config:
            cmd.extend(["--terrain_config", terrain_config])
        if learning_rate is not None:
            cmd.extend(["--learning_rate", str(learning_rate)])

        resolved_log_dir = log_dir or str(
            settings.log_dir / f"run_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        if not log_dir:
            cmd.extend(["--log_dir", resolved_log_dir])

        logger.info(f"Starting training: {' '.join(cmd)}")

        # Build env with unbuffered Python output
        proc_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        # On Windows, non-headless (GUI) mode needs CREATE_NEW_CONSOLE so
        # Isaac Sim's Vulkan/GPU window has proper desktop session access.
        # Headless mode uses CREATE_NEW_PROCESS_GROUP for clean CTRL_BREAK
        # termination without needing a window.
        if sys.platform == "win32":
            if headless:
                cflags = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                cflags = subprocess.CREATE_NEW_CONSOLE
        else:
            cflags = 0

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(settings.project_root),
            env=proc_env,
            creationflags=cflags,
        )
        logger.info(f"Subprocess PID: {self._process.pid}")

        self._state = SimState.TRAINING
        self._current_run_id = run_id
        self._log_dir = resolved_log_dir
        self._on_output = on_output
        self._headless = headless

        # Start async reader for stdout
        if on_output:
            loop = asyncio.get_event_loop()
            self._reader_task = loop.create_task(self._read_output())

        return resolved_log_dir

    def start_evaluation(
        self,
        checkpoint_path: str,
        task: str = "Go2-Obstacle-Play-v0",
        num_envs: int = 4,
        num_steps: int = 2000,
    ) -> None:
        """Launch an inference subprocess."""
        if self._state != SimState.IDLE:
            raise RuntimeError(f"Cannot start evaluation while in state: {self._state}")

        isaaclab_cmd = self._resolve_python()
        play_script = str(settings.project_root / "sim" / "scripts" / "play.py")

        cmd = [
            *isaaclab_cmd, play_script,
            "--task", task,
            "--checkpoint", checkpoint_path,
            "--num_envs", str(num_envs),
            "--num_steps", str(num_steps),
        ]

        logger.info(f"Starting evaluation: {' '.join(cmd)}")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(settings.project_root),
        )
        self._state = SimState.EVALUATING

    async def stop(self, timeout: float = 30.0) -> None:
        """Gracefully stop the running subprocess."""
        if self._process is None or self._state == SimState.IDLE:
            return

        self._state = SimState.STOPPING
        logger.info("Sending termination signal to subprocess...")

        # Try graceful termination first
        if sys.platform == "win32":
            if self._headless:
                # Headless uses CREATE_NEW_PROCESS_GROUP — send CTRL_BREAK
                self._process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # Non-headless uses CREATE_NEW_CONSOLE — terminate directly
                self._process.terminate()
        else:
            self._process.send_signal(signal.SIGTERM)

        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self._process.wait),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Subprocess did not exit gracefully, killing...")
            self._process.kill()
            self._process.wait()

        self._cleanup()

    def poll(self) -> int | None:
        """Check if subprocess has exited. Returns exit code or None if still running."""
        if self._process is None:
            return None
        rc = self._process.poll()
        if rc is not None and self._state not in (SimState.IDLE, SimState.STOPPING):
            self._cleanup()
        return rc

    def _cleanup(self):
        """Reset state after subprocess exits."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
        self._process = None
        self._state = SimState.IDLE
        self._current_run_id = None
        self._on_output = None
        self._reader_task = None
        self._headless = True

    async def _read_output(self):
        """Continuously read stdout from the subprocess and forward to callback."""
        if self._process is None or self._process.stdout is None:
            logger.warning("_read_output: no process or stdout to read")
            return
        logger.info("_read_output: starting output reader task")
        line_count = 0
        try:
            loop = asyncio.get_event_loop()
            while True:
                line = await loop.run_in_executor(None, self._process.stdout.readline)
                if not line:
                    # EOF — process likely exited
                    logger.info(f"_read_output: EOF after {line_count} lines")
                    break
                line_count += 1
                line = line.strip()
                if line and self._on_output:
                    self._on_output(line)
        except asyncio.CancelledError:
            logger.info(f"_read_output: cancelled after {line_count} lines")
        except Exception as e:
            logger.error(f"Error reading subprocess output after {line_count} lines: {e}")

    def _resolve_python(self) -> list[str]:
        """Resolve the Isaac Lab Python executable as a command list."""
        # Isaac Lab provides isaaclab.bat / isaaclab.sh wrapper
        # On Windows, .bat files must be invoked via cmd /c for subprocess.Popen
        if sys.platform == "win32":
            isaaclab_bat = Path(settings.isaaclab_path) / "isaaclab.bat"
            if isaaclab_bat.exists():
                return ["cmd", "/c", str(isaaclab_bat), "-p"]
            # Fallback: use Isaac Sim's python.bat
            return ["cmd", "/c", str(Path(settings.isaacsim_path) / "python.bat")]
        else:
            isaaclab_sh = Path(settings.isaaclab_path) / "isaaclab.sh"
            if isaaclab_sh.exists():
                return [str(isaaclab_sh), "-p"]
            return [str(Path(settings.isaacsim_path) / "python.sh")]

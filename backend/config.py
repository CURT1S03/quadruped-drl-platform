# Copyright (c) 2024, Quadruped DRL Training Platform
# SPDX-License-Identifier: MIT

"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    isaacsim_path: str = r"a:\IsaacSim\isaac-sim-standalone-5.1.0-windows-x86_64"
    isaaclab_path: str = r"A:\IsaacLab"
    project_root: Path = Path(__file__).resolve().parent.parent

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    database_url: str = "sqlite+aiosqlite:///./logs/platform.db"

    default_num_envs: int = 4096
    default_max_iterations: int = 2000
    default_checkpoint_interval: int = 500
    log_dir: Path = Path("logs/runs")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Quadruped DRL Training Platform

A full-stack platform for training a **Unitree Go2** quadruped robot via Deep Reinforcement Learning (PPO) in NVIDIA Isaac Lab, with a FastAPI backend for orchestration and a React dashboard for real-time monitoring.

## Architecture

```
┌─────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Isaac Lab + Isaac   │────▶│   FastAPI Backend │────▶│  React Dashboard │
│  Sim (GPU Training)  │     │   (Orchestration) │     │  (Visualization) │
│                      │     │                   │     │                  │
│  • Go2 Environment   │     │  • REST API       │     │  • Live Charts   │
│  • PPO (RSL-RL)      │     │  • WebSocket      │     │  • Controls      │
│  • Terrain Curriculum│     │  • SQLite DB      │     │  • Run History   │
└─────────────────────┘     └──────────────────┘     └──────────────────┘
```

## Prerequisites

- **NVIDIA Isaac Sim 5.1.0** installed at `a:\IsaacSim\isaac-sim-standalone-5.1.0-windows-x86_64`
- **NVIDIA Isaac Lab** cloned at `A:\IsaacLab` (see Setup below)
- **Python 3.11** (via conda) with PyTorch + CUDA
- **Node.js 18+** for the frontend

## Setup

### 1. Isaac Lab Installation

```bash
# Clone Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git A:\IsaacLab
cd A:\IsaacLab

# Set Isaac Sim path
set ISAACSIM_PATH=a:\IsaacSim\isaac-sim-standalone-5.1.0-windows-x86_64

# Install Isaac Lab (use the installer script)
isaaclab.bat --install

# Install RSL-RL
isaaclab.bat -p -m pip install rsl-rl-lib
```

### 2. Backend

```bash
cd backend
pip install -r requirements.txt

# Start the API server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Usage

### Train from CLI

```bash
# Obstacle terrain (full training)
isaaclab -p sim/scripts/train.py --task Go2-Obstacle-v0 --num_envs 4096 --headless

# Flat terrain (quick test)
isaaclab -p sim/scripts/train.py --task Go2-Flat-v0 --num_envs 2048 --max_iterations 300

# Evaluate a checkpoint
isaaclab -p sim/scripts/play.py --task Go2-Obstacle-Play-v0 --checkpoint logs/runs/.../model_1500.pt

# Export policy for Isaac Sim deployment
python sim/scripts/export_policy.py --checkpoint logs/runs/.../model_1500.pt
```

### Train from Dashboard

1. Start the backend: `uvicorn backend.main:app`
2. Start the frontend: `cd frontend && npm run dev`
3. Open http://localhost:5173
4. Configure parameters and click **Start Training**
5. Watch live reward curves and metrics

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `sim/envs/` | Isaac Lab environment configs (Go2 obstacle/flat) |
| `sim/rewards/` | Custom reward functions |
| `sim/terrains/` | Terrain generation configs |
| `sim/agents/` | RSL-RL PPO hyperparameter configs |
| `sim/scripts/` | Training, evaluation, and export entry points |
| `backend/` | FastAPI server (REST + WebSocket + SQLite) |
| `frontend/` | React + TypeScript dashboard |
| `logs/runs/` | Training outputs, checkpoints, TensorBoard logs |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/training/start` | Start a training run |
| POST | `/api/training/stop` | Stop current training |
| GET | `/api/training/status` | Current simulation state |
| GET | `/api/training/runs` | List all runs |
| GET | `/api/checkpoints/` | List checkpoints |
| POST | `/api/checkpoints/{id}/evaluate` | Run inference with checkpoint |
| WS | `/ws/telemetry` | Live training metrics stream |
| GET | `/api/config/defaults` | Default parameters |

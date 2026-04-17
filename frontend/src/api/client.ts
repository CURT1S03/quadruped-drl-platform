/** Typed REST API client for the FastAPI backend. */

import type {
  CheckpointResponse,
  DefaultConfig,
  EvaluateRequest,
  ExportResponse,
  RobotInfo,
  RunDetail,
  RunSummary,
  TerrainInfo,
  TrainingStartRequest,
  TrainingStatusResponse,
} from "../types";

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

// ─── Training ──────────────────────────────────────────────────────────────── //

export function startTraining(req: TrainingStartRequest) {
  return request<TrainingStatusResponse>("/training/start", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function stopTraining() {
  return request<{ status: string; run_id: number | null }>("/training/stop", {
    method: "POST",
  });
}

export function getTrainingStatus() {
  return request<TrainingStatusResponse>("/training/status");
}

// ─── Runs ──────────────────────────────────────────────────────────────────── //

export function listRuns(limit = 50, offset = 0) {
  return request<RunSummary[]>(`/training/runs?limit=${limit}&offset=${offset}`);
}

export function getRunDetail(runId: number) {
  return request<RunDetail>(`/training/runs/${runId}`);
}

// ─── Checkpoints ───────────────────────────────────────────────────────────── //

export function listCheckpoints(runId?: number) {
  const qs = runId != null ? `?run_id=${runId}` : "";
  return request<CheckpointResponse[]>(`/checkpoints/${qs}`);
}

export function evaluateCheckpoint(checkpointId: number, req?: EvaluateRequest) {
  return request<{ status: string; checkpoint_id: number }>(
    `/checkpoints/${checkpointId}/evaluate`,
    { method: "POST", body: JSON.stringify(req ?? {}) }
  );
}

export function scanCheckpoints(runId: number) {
  return request<{ scanned: number; new: number }>(`/checkpoints/scan/${runId}`, {
    method: "POST",
  });
}

// ─── Config ────────────────────────────────────────────────────────────────── //

export function getDefaults() {
  return request<DefaultConfig>("/config/defaults");
}

// ─── Robots & Terrains ─────────────────────────────────────────────────────── //

export function listRobots() {
  return request<RobotInfo[]>("/config/robots");
}

export function listTerrains() {
  return request<TerrainInfo[]>("/config/terrains");
}

export async function uploadRobot(name: string, urdf: File, metadata: File): Promise<RobotInfo> {
  const form = new FormData();
  form.append("name", name);
  form.append("urdf", urdf);
  form.append("metadata", metadata);
  const res = await fetch(`${BASE}/config/upload-robot`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function uploadTerrain(name: string, terrainYaml: File): Promise<TerrainInfo> {
  const form = new FormData();
  form.append("name", name);
  form.append("terrain_yaml", terrainYaml);
  const res = await fetch(`${BASE}/config/upload-terrain`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return res.json();
}

// ─── Export ────────────────────────────────────────────────────────────────── //

export function exportCheckpoint(checkpointId: number) {
  return request<ExportResponse>(`/checkpoints/${checkpointId}/export`, {
    method: "POST",
  });
}

export function getExportDownloadUrl(checkpointId: number) {
  return `${BASE}/checkpoints/${checkpointId}/download`;
}

// ─── Health ────────────────────────────────────────────────────────────────── //

export function getHealth() {
  return request<{ status: string; sim_state: string; current_run_id: number | null }>(
    "/health"
  );
}

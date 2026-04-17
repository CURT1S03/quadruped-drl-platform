/** Shared TypeScript interfaces mirroring backend Pydantic schemas. */

export type SimState = "idle" | "training" | "evaluating" | "stopping";
export type RunStatus = "queued" | "running" | "completed" | "failed" | "stopped";

export interface TrainingStartRequest {
  name?: string;
  task?: string;
  num_envs?: number;
  max_iterations?: number;
  learning_rate?: number;
  headless?: boolean;
}

export interface TrainingStatusResponse {
  state: SimState;
  run_id: number | null;
  iteration: number;
  max_iterations: number;
  log_dir: string | null;
}

export interface RunSummary {
  id: number;
  name: string;
  status: RunStatus;
  task: string;
  num_envs: number;
  best_reward: number | null;
  total_iterations: number;
  created_at: string;
  finished_at: string | null;
}

export interface RunDetail extends RunSummary {
  config_json: string;
  log_dir: string | null;
  checkpoints: CheckpointResponse[];
}

export interface CheckpointResponse {
  id: number;
  run_id: number;
  iteration: number;
  file_path: string;
  mean_reward: number | null;
  created_at: string;
}

export interface EvaluateRequest {
  task?: string;
  num_envs?: number;
  num_steps?: number;
}

export interface DefaultConfig {
  tasks: string[];
  default_num_envs: number;
  default_max_iterations: number;
  default_learning_rate: number;
  terrain_types: string[];
}

export interface TelemetryData {
  event?: string;
  iteration?: number;
  max_iterations?: number;
  mean_reward?: number;
  mean_episode_length?: number;
  value_loss?: number;
  policy_loss?: number;
  learning_rate?: number;
  timestamp?: string;
}

export interface MetricEntry {
  iteration: number;
  metric_name: string;
  metric_value: number;
  timestamp: string;
}

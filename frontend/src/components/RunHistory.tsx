import type { RunSummary } from "../types";
import { RunStatusBadge } from "./StatusBadge";

interface Props {
  runs: RunSummary[];
  selectedRunId: number | null;
  onSelect: (runId: number) => void;
}

export function RunHistory({ runs, selectedRunId, onSelect }: Props) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-gray-400">
        Run History
      </h2>

      {runs.length === 0 ? (
        <p className="text-sm text-gray-600">No training runs yet.</p>
      ) : (
        <div className="space-y-2 overflow-y-auto" style={{ maxHeight: "300px" }}>
          {runs.map((run) => (
            <button
              key={run.id}
              onClick={() => onSelect(run.id)}
              className={`w-full rounded border p-3 text-left transition ${
                selectedRunId === run.id
                  ? "border-nvidia/50 bg-nvidia/10"
                  : "border-gray-800 bg-gray-800/50 hover:border-gray-700"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{run.name}</span>
                <RunStatusBadge status={run.status} />
              </div>
              <div className="mt-1 flex gap-4 text-xs text-gray-500">
                <span>{run.task}</span>
                <span>{run.num_envs} envs</span>
                <span>{run.total_iterations} iters</span>
                {run.best_reward != null && (
                  <span className="text-nvidia">Best: {run.best_reward.toFixed(2)}</span>
                )}
              </div>
              <div className="mt-1 text-xs text-gray-600">
                {new Date(run.created_at).toLocaleString()}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

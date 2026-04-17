import type { SimState } from "../types";
import { Activity, Circle, Loader2, Pause, Square } from "lucide-react";

const config: Record<SimState, { color: string; label: string; icon: typeof Circle }> = {
  idle: { color: "text-gray-400", label: "Idle", icon: Circle },
  training: { color: "text-nvidia", label: "Training", icon: Activity },
  evaluating: { color: "text-blue-400", label: "Evaluating", icon: Loader2 },
  stopping: { color: "text-yellow-400", label: "Stopping", icon: Pause },
};

export function StatusBadge({ state }: { state: SimState }) {
  const { color, label, icon: Icon } = config[state] ?? config.idle;

  return (
    <span className={`inline-flex items-center gap-1.5 text-sm font-medium ${color}`}>
      <Icon
        size={14}
        className={state === "training" || state === "evaluating" ? "animate-pulse" : ""}
      />
      {label}
    </span>
  );
}

export function RunStatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    queued: "bg-gray-600",
    running: "bg-nvidia/20 text-nvidia",
    completed: "bg-green-900/30 text-green-400",
    failed: "bg-red-900/30 text-red-400",
    stopped: "bg-yellow-900/30 text-yellow-400",
  };

  return (
    <span
      className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${colors[status] ?? "bg-gray-700"}`}
    >
      {status}
    </span>
  );
}

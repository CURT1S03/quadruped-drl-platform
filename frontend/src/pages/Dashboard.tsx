import { useState } from "react";
import { Cpu, Wifi, WifiOff } from "lucide-react";
import { StatusBadge } from "../components/StatusBadge";
import { TrainingControls } from "../components/TrainingControls";
import { MetricChart } from "../components/MetricChart";
import { CheckpointTable } from "../components/CheckpointTable";
import { RunHistory } from "../components/RunHistory";
import { useTelemetry } from "../hooks/useTelemetry";
import { useTrainingStatus, useRuns, useCheckpoints } from "../hooks/useTrainingApi";
import type { SimState } from "../types";

export function Dashboard() {
  const { data: status } = useTrainingStatus();
  const { data: runs } = useRuns();
  const { connected, latest, history } = useTelemetry();
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);

  const activeRunId = status?.run_id ?? selectedRunId;
  const { data: checkpoints } = useCheckpoints(activeRunId ?? undefined);

  const simState: SimState = (status?.state as SimState) ?? "idle";
  const isRunning = simState === "training" || simState === "evaluating";

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/50 px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Cpu size={20} className="text-nvidia" />
            <h1 className="text-lg font-bold">
              Quadruped DRL <span className="text-nvidia">Training Platform</span>
            </h1>
          </div>
          <div className="flex items-center gap-4">
            <StatusBadge state={simState} />
            <span className="flex items-center gap-1 text-xs text-gray-500">
              {connected ? (
                <Wifi size={12} className="text-green-500" />
              ) : (
                <WifiOff size={12} className="text-red-500" />
              )}
              WS
            </span>
            {status?.iteration != null && status.max_iterations > 0 && (
              <span className="font-mono text-xs text-gray-400">
                {status.iteration.toLocaleString()} / {status.max_iterations.toLocaleString()}
              </span>
            )}
          </div>
        </div>

        {/* Progress bar */}
        {isRunning && status && status.max_iterations > 0 && (
          <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-gray-800">
            <div
              className="h-full bg-nvidia transition-all duration-500"
              style={{
                width: `${Math.min(100, (status.iteration / status.max_iterations) * 100)}%`,
              }}
            />
          </div>
        )}
      </header>

      {/* Main grid */}
      <main className="mx-auto max-w-screen-2xl gap-4 p-4 lg:grid lg:grid-cols-12">
        {/* Left column — controls + run history */}
        <div className="space-y-4 lg:col-span-3">
          <TrainingControls simState={simState} />
          <RunHistory
            runs={runs ?? []}
            selectedRunId={activeRunId}
            onSelect={setSelectedRunId}
          />
        </div>

        {/* Center column — live charts */}
        <div className="mt-4 space-y-4 lg:col-span-6 lg:mt-0">
          {/* Summary cards */}
          <div className="grid grid-cols-3 gap-3">
            <SummaryCard
              label="Mean Reward"
              value={latest?.mean_reward?.toFixed(2) ?? "—"}
              color="text-nvidia"
            />
            <SummaryCard
              label="Episode Length"
              value={latest?.mean_episode_length?.toFixed(1) ?? "—"}
              color="text-blue-400"
            />
            <SummaryCard
              label="Learning Rate"
              value={latest?.learning_rate?.toExponential(2) ?? "—"}
              color="text-purple-400"
            />
          </div>

          <MetricChart data={history} dataKey="mean_reward" title="Mean Reward" color="#76b900" />
          <MetricChart
            data={history}
            dataKey="mean_episode_length"
            title="Episode Length"
            color="#60a5fa"
          />
          <div className="grid grid-cols-2 gap-4">
            <MetricChart data={history} dataKey="value_loss" title="Value Loss" color="#f87171" />
            <MetricChart
              data={history}
              dataKey="policy_loss"
              title="Policy Loss"
              color="#fb923c"
            />
          </div>
        </div>

        {/* Right column — checkpoints */}
        <div className="mt-4 lg:col-span-3 lg:mt-0">
          <CheckpointTable checkpoints={checkpoints ?? []} disabled={isRunning} />
        </div>
      </main>
    </div>
  );
}

function SummaryCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-3">
      <div className="text-xs uppercase text-gray-500">{label}</div>
      <div className={`mt-1 text-xl font-bold ${color}`}>{value}</div>
    </div>
  );
}

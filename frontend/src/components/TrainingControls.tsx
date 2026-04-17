import { useState } from "react";
import { Play, Square } from "lucide-react";
import { useStartTraining, useStopTraining, useDefaults } from "../hooks/useTrainingApi";
import type { SimState, TrainingStartRequest } from "../types";

interface Props {
  simState: SimState;
}

export function TrainingControls({ simState }: Props) {
  const { data: defaults } = useDefaults();
  const startMutation = useStartTraining();
  const stopMutation = useStopTraining();

  const [form, setForm] = useState<TrainingStartRequest>({
    name: "",
    task: "Go2-Obstacle-v0",
    num_envs: 4096,
    max_iterations: 1500,
    learning_rate: 0.001,
    headless: true,
  });

  const isRunning = simState === "training" || simState === "evaluating";

  const handleStart = () => {
    startMutation.mutate({ ...form, name: form.name || undefined });
  };

  const handleStop = () => {
    stopMutation.mutate();
  };

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-gray-400">
        Training Controls
      </h2>

      <div className="grid grid-cols-2 gap-3">
        {/* Task */}
        <div>
          <label className="mb-1 block text-xs text-gray-500">Task</label>
          <select
            className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm"
            value={form.task}
            onChange={(e) => setForm({ ...form, task: e.target.value })}
            disabled={isRunning}
          >
            {(defaults?.tasks ?? ["Go2-Obstacle-v0", "Go2-Flat-v0"]).map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>

        {/* Run name */}
        <div>
          <label className="mb-1 block text-xs text-gray-500">Run Name</label>
          <input
            type="text"
            className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm"
            placeholder="(auto)"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            disabled={isRunning}
          />
        </div>

        {/* Num envs */}
        <div>
          <label className="mb-1 block text-xs text-gray-500">Environments</label>
          <input
            type="number"
            className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm"
            value={form.num_envs}
            onChange={(e) => setForm({ ...form, num_envs: Number(e.target.value) })}
            min={1}
            max={65536}
            disabled={isRunning}
          />
        </div>

        {/* Max iterations */}
        <div>
          <label className="mb-1 block text-xs text-gray-500">Max Iterations</label>
          <input
            type="number"
            className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm"
            value={form.max_iterations}
            onChange={(e) => setForm({ ...form, max_iterations: Number(e.target.value) })}
            min={1}
            disabled={isRunning}
          />
        </div>

        {/* Learning rate */}
        <div>
          <label className="mb-1 block text-xs text-gray-500">Learning Rate</label>
          <input
            type="number"
            className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm"
            value={form.learning_rate}
            onChange={(e) => setForm({ ...form, learning_rate: Number(e.target.value) })}
            step={0.0001}
            min={0.00001}
            max={1}
            disabled={isRunning}
          />
        </div>

        {/* Headless */}
        <div className="flex items-end">
          <label className="inline-flex cursor-pointer items-center gap-2 text-sm">
            <input
              type="checkbox"
              className="rounded border-gray-600"
              checked={form.headless}
              onChange={(e) => setForm({ ...form, headless: e.target.checked })}
              disabled={isRunning}
            />
            Headless
          </label>
        </div>
      </div>

      {/* Action buttons */}
      <div className="mt-4 flex gap-2">
        <button
          onClick={handleStart}
          disabled={isRunning || startMutation.isPending}
          className="flex items-center gap-1.5 rounded bg-nvidia px-4 py-2 text-sm font-medium text-black transition hover:bg-nvidia/80 disabled:opacity-40"
        >
          <Play size={14} />
          Start Training
        </button>
        <button
          onClick={handleStop}
          disabled={!isRunning || stopMutation.isPending}
          className="flex items-center gap-1.5 rounded bg-red-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-red-700 disabled:opacity-40"
        >
          <Square size={14} />
          Stop
        </button>
      </div>

      {startMutation.isError && (
        <p className="mt-2 text-xs text-red-400">{String(startMutation.error)}</p>
      )}
    </div>
  );
}

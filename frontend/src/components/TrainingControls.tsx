import { useRef, useState } from "react";
import { Play, Square, Upload } from "lucide-react";
import {
  useStartTraining,
  useStopTraining,
  useDefaults,
  useRobots,
  useTerrains,
  useUploadRobot,
  useUploadTerrain,
} from "../hooks/useTrainingApi";
import type { SimState, TrainingStartRequest } from "../types";

interface Props {
  simState: SimState;
}

export function TrainingControls({ simState }: Props) {
  const { data: defaults } = useDefaults();
  const { data: robots } = useRobots();
  const { data: terrains } = useTerrains();
  const startMutation = useStartTraining();
  const stopMutation = useStopTraining();
  const uploadRobotMutation = useUploadRobot();
  const uploadTerrainMutation = useUploadTerrain();

  const urdfInputRef = useRef<HTMLInputElement>(null);
  const metadataInputRef = useRef<HTMLInputElement>(null);
  const terrainInputRef = useRef<HTMLInputElement>(null);

  const [form, setForm] = useState<TrainingStartRequest>({
    name: "",
    task: "Go2-Obstacle-v0",
    num_envs: 4096,
    max_iterations: 1500,
    learning_rate: 0.001,
    headless: true,
    robot_name: null,
    terrain_preset: null,
  });

  const [uploadRobotName, setUploadRobotName] = useState("");
  const [uploadTerrainName, setUploadTerrainName] = useState("");

  const isRunning = simState === "training" || simState === "evaluating";
  const useCustomRobot = form.robot_name !== null && form.robot_name !== "";

  const handleStart = () => {
    startMutation.mutate({
      ...form,
      name: form.name || undefined,
      robot_name: useCustomRobot ? form.robot_name : undefined,
      terrain_preset: form.terrain_preset || undefined,
    });
  };

  const handleStop = () => {
    stopMutation.mutate();
  };

  const handleUploadRobot = () => {
    const urdfFile = urdfInputRef.current?.files?.[0];
    const metaFile = metadataInputRef.current?.files?.[0];
    if (!urdfFile || !metaFile || !uploadRobotName) return;
    uploadRobotMutation.mutate(
      { name: uploadRobotName, urdf: urdfFile, metadata: metaFile },
      {
        onSuccess: (robot) => {
          setForm({ ...form, robot_name: robot.name });
          setUploadRobotName("");
          if (urdfInputRef.current) urdfInputRef.current.value = "";
          if (metadataInputRef.current) metadataInputRef.current.value = "";
        },
      }
    );
  };

  const handleUploadTerrain = () => {
    const file = terrainInputRef.current?.files?.[0];
    if (!file || !uploadTerrainName) return;
    uploadTerrainMutation.mutate(
      { name: uploadTerrainName, terrainYaml: file },
      {
        onSuccess: (terrain) => {
          setForm({ ...form, terrain_preset: terrain.name });
          setUploadTerrainName("");
          if (terrainInputRef.current) terrainInputRef.current.value = "";
        },
      }
    );
  };

  const selectedRobot = robots?.find((r) => r.name === form.robot_name);

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

        {/* Robot selector */}
        <div>
          <label className="mb-1 block text-xs text-gray-500">Robot</label>
          <select
            className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm"
            value={form.robot_name ?? ""}
            onChange={(e) =>
              setForm({ ...form, robot_name: e.target.value || null })
            }
            disabled={isRunning}
          >
            <option value="">Default (Go2)</option>
            {robots?.map((r) => (
              <option key={r.name} value={r.name}>
                {r.name} ({r.num_dof} DOF)
              </option>
            ))}
          </select>
        </div>

        {/* Terrain selector */}
        <div>
          <label className="mb-1 block text-xs text-gray-500">Terrain</label>
          <select
            className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm"
            value={form.terrain_preset ?? ""}
            onChange={(e) =>
              setForm({ ...form, terrain_preset: e.target.value || null })
            }
            disabled={isRunning}
          >
            <option value="">Default (obstacle)</option>
            {terrains?.map((t) => (
              <option key={t.name} value={t.name}>
                {t.name}{t.is_custom ? " (custom)" : ""}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Upload sections */}
      <details className="mt-3">
        <summary className="cursor-pointer text-xs text-gray-500 hover:text-gray-300">
          Upload Custom Robot / Terrain
        </summary>
        <div className="mt-2 space-y-3 rounded border border-gray-800 bg-gray-800/50 p-3">
          {/* Upload robot */}
          <div>
            <p className="mb-1 text-xs font-medium text-gray-400">Upload Robot (URDF + metadata.json)</p>
            <div className="flex gap-2">
              <input
                type="text"
                className="w-28 rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs"
                placeholder="Robot name"
                value={uploadRobotName}
                onChange={(e) => setUploadRobotName(e.target.value)}
              />
              <input ref={urdfInputRef} type="file" accept=".urdf,.xml" className="hidden" />
              <button
                type="button"
                onClick={() => urdfInputRef.current?.click()}
                className="rounded border border-gray-700 px-2 py-1 text-xs hover:bg-gray-700"
              >
                URDF
              </button>
              <input ref={metadataInputRef} type="file" accept=".json" className="hidden" />
              <button
                type="button"
                onClick={() => metadataInputRef.current?.click()}
                className="rounded border border-gray-700 px-2 py-1 text-xs hover:bg-gray-700"
              >
                metadata.json
              </button>
              <button
                type="button"
                onClick={handleUploadRobot}
                disabled={uploadRobotMutation.isPending}
                className="flex items-center gap-1 rounded bg-nvidia/80 px-2 py-1 text-xs font-medium text-black hover:bg-nvidia disabled:opacity-40"
              >
                <Upload size={12} /> Upload
              </button>
            </div>
            {uploadRobotMutation.isError && (
              <p className="mt-1 text-xs text-red-400">{String(uploadRobotMutation.error)}</p>
            )}
          </div>

          {/* Upload terrain */}
          <div>
            <p className="mb-1 text-xs font-medium text-gray-400">Upload Terrain (YAML)</p>
            <div className="flex gap-2">
              <input
                type="text"
                className="w-28 rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs"
                placeholder="Terrain name"
                value={uploadTerrainName}
                onChange={(e) => setUploadTerrainName(e.target.value)}
              />
              <input ref={terrainInputRef} type="file" accept=".yaml,.yml" className="hidden" />
              <button
                type="button"
                onClick={() => terrainInputRef.current?.click()}
                className="rounded border border-gray-700 px-2 py-1 text-xs hover:bg-gray-700"
              >
                YAML file
              </button>
              <button
                type="button"
                onClick={handleUploadTerrain}
                disabled={uploadTerrainMutation.isPending}
                className="flex items-center gap-1 rounded bg-nvidia/80 px-2 py-1 text-xs font-medium text-black hover:bg-nvidia disabled:opacity-40"
              >
                <Upload size={12} /> Upload
              </button>
            </div>
            {uploadTerrainMutation.isError && (
              <p className="mt-1 text-xs text-red-400">{String(uploadTerrainMutation.error)}</p>
            )}
          </div>
        </div>
      </details>

      {selectedRobot && (
        <p className="mt-2 text-xs text-gray-500">
          Robot: {selectedRobot.name} · {selectedRobot.num_dof} DOF · {selectedRobot.num_legs} legs
        </p>
      )}

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

import { Download, Package, Play } from "lucide-react";
import type { CheckpointResponse } from "../types";
import { useEvaluateCheckpoint, useExportCheckpoint } from "../hooks/useTrainingApi";
import { getExportDownloadUrl } from "../api/client";

interface Props {
  checkpoints: CheckpointResponse[];
  disabled?: boolean;
}

export function CheckpointTable({ checkpoints, disabled }: Props) {
  const evaluateMutation = useEvaluateCheckpoint();
  const exportMutation = useExportCheckpoint();

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-gray-400">
        Checkpoints
      </h2>

      {checkpoints.length === 0 ? (
        <p className="text-sm text-gray-600">No checkpoints yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-xs uppercase text-gray-500">
                <th className="pb-2 pr-4">Iteration</th>
                <th className="pb-2 pr-4">Mean Reward</th>
                <th className="pb-2 pr-4">Created</th>
                <th className="pb-2">Actions</th>
              </tr>
            </thead>
            <tbody>
              {checkpoints.map((ckpt) => (
                <tr key={ckpt.id} className="border-b border-gray-800/50">
                  <td className="py-2 pr-4 font-mono text-gray-300">
                    {ckpt.iteration.toLocaleString()}
                  </td>
                  <td className="py-2 pr-4">
                    {ckpt.mean_reward != null ? ckpt.mean_reward.toFixed(2) : "—"}
                  </td>
                  <td className="py-2 pr-4 text-gray-500">
                    {new Date(ckpt.created_at).toLocaleString()}
                  </td>
                  <td className="py-2 flex gap-1">
                    <button
                      onClick={() => evaluateMutation.mutate({ id: ckpt.id })}
                      disabled={disabled || evaluateMutation.isPending}
                      className="inline-flex items-center gap-1 rounded bg-blue-600/20 px-2 py-1 text-xs text-blue-400 transition hover:bg-blue-600/30 disabled:opacity-40"
                    >
                      <Play size={12} />
                      Evaluate
                    </button>
                    <button
                      onClick={() => exportMutation.mutate({ id: ckpt.id })}
                      disabled={disabled || exportMutation.isPending}
                      className="inline-flex items-center gap-1 rounded bg-green-600/20 px-2 py-1 text-xs text-green-400 transition hover:bg-green-600/30 disabled:opacity-40"
                    >
                      <Package size={12} />
                      Export
                    </button>
                    <a
                      href={getExportDownloadUrl(ckpt.id)}
                      className="inline-flex items-center gap-1 rounded bg-gray-600/20 px-2 py-1 text-xs text-gray-400 transition hover:bg-gray-600/30"
                    >
                      <Download size={12} />
                      Download
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

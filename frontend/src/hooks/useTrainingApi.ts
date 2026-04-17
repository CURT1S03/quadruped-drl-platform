/** React Query hooks for training API operations. */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as api from "../api/client";
import type { TrainingStartRequest } from "../types";

export function useTrainingStatus() {
  return useQuery({
    queryKey: ["trainingStatus"],
    queryFn: api.getTrainingStatus,
    refetchInterval: 3000,
  });
}

export function useStartTraining() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: TrainingStartRequest) => api.startTraining(req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["trainingStatus"] });
      qc.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}

export function useStopTraining() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.stopTraining,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["trainingStatus"] });
      qc.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}

export function useRuns(limit = 50) {
  return useQuery({
    queryKey: ["runs", limit],
    queryFn: () => api.listRuns(limit),
    refetchInterval: 10000,
  });
}

export function useRunDetail(runId: number | null) {
  return useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.getRunDetail(runId!),
    enabled: runId != null,
  });
}

export function useCheckpoints(runId?: number) {
  return useQuery({
    queryKey: ["checkpoints", runId],
    queryFn: () => api.listCheckpoints(runId),
    refetchInterval: 15000,
  });
}

export function useEvaluateCheckpoint() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, req }: { id: number; req?: Parameters<typeof api.evaluateCheckpoint>[1] }) =>
      api.evaluateCheckpoint(id, req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["trainingStatus"] });
    },
  });
}

export function useDefaults() {
  return useQuery({
    queryKey: ["defaults"],
    queryFn: api.getDefaults,
    staleTime: Infinity,
  });
}

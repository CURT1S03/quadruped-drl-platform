/** React hook for consuming live telemetry data via WebSocket. */

import { useCallback, useEffect, useRef, useState } from "react";
import type { TelemetryData } from "../types";
import { telemetrySocket } from "../api/websocket";

const MAX_HISTORY = 2000;

export function useTelemetry() {
  const [connected, setConnected] = useState(false);
  const [latest, setLatest] = useState<TelemetryData | null>(null);
  const [history, setHistory] = useState<TelemetryData[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    telemetrySocket.connect();

    const unsub = telemetrySocket.subscribe((data) => {
      const td = data as TelemetryData;
      if (td.event === "heartbeat") return;

      setLatest(td);
      if (td.iteration != null) {
        setHistory((prev) => {
          const next = [...prev, td];
          return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
        });
      }
    });

    // Poll connection status
    intervalRef.current = setInterval(() => {
      setConnected(telemetrySocket.connected);
    }, 1000);

    return () => {
      unsub();
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
    setLatest(null);
  }, []);

  return { connected, latest, history, clearHistory };
}

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { TelemetryData } from "../types";

interface Props {
  data: TelemetryData[];
  dataKey: keyof TelemetryData;
  title: string;
  color?: string;
  yDomain?: [number | "auto", number | "auto"];
}

export function MetricChart({ data, dataKey, title, color = "#76b900", yDomain }: Props) {
  const chartData = data
    .filter((d) => d.iteration != null && d[dataKey] != null)
    .map((d) => ({
      iteration: d.iteration,
      value: d[dataKey] as number,
    }));

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-gray-400">
        {title}
      </h3>
      {chartData.length === 0 ? (
        <div className="flex h-48 items-center justify-center text-sm text-gray-600">
          Waiting for data...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="iteration"
              stroke="#6b7280"
              tick={{ fontSize: 11 }}
              label={{ value: "Iteration", position: "insideBottom", offset: -5, fill: "#6b7280" }}
            />
            <YAxis
              stroke="#6b7280"
              tick={{ fontSize: 11 }}
              domain={yDomain ?? ["auto", "auto"]}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1f2937",
                border: "1px solid #374151",
                borderRadius: "0.375rem",
                fontSize: "12px",
              }}
              labelStyle={{ color: "#9ca3af" }}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke={color}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

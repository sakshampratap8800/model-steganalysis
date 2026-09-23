import { RadialBarChart, RadialBar, PolarAngleAxis } from "recharts";

interface RiskScoreGaugeProps {
  score: number;
  size?: number;
  showLabel?: boolean;
}

function getRiskColor(score: number): string {
  if (score <= 30) return "#10b981";   // safe-500 (BENIGN)
  if (score <= 65) return "#f59e0b";   // warning-500 (SUSPICIOUS)
  return "#ef4444";                     // danger-500 (HIGH_RISK)
}

function getRiskLabel(score: number): string {
  if (score <= 30) return "BENIGN";
  if (score <= 65) return "SUSPICIOUS";
  return "HIGH RISK";
}

export default function RiskScoreGauge({ score, size = 180, showLabel = true }: RiskScoreGaugeProps) {
  const clampedScore = Math.max(0, Math.min(100, score));
  const color = getRiskColor(clampedScore);
  const label = getRiskLabel(clampedScore);

  const data = [{ value: clampedScore, fill: color }];

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <RadialBarChart
          width={size}
          height={size}
          cx={size / 2}
          cy={size / 2}
          innerRadius={size * 0.35}
          outerRadius={size * 0.48}
          barSize={size * 0.08}
          data={data}
          startAngle={220}
          endAngle={-40}
        >
          <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
          <RadialBar
            background={{ fill: "#1e293b" }}
            dataKey="value"
            angleAxisId={0}
            cornerRadius={6}
          />
        </RadialBarChart>
        {/* Center overlay */}
        <div
          className="absolute inset-0 flex flex-col items-center justify-center"
          style={{ pointerEvents: "none" }}
        >
          <span
            className="font-mono font-bold tabular-nums leading-none"
            style={{ fontSize: size * 0.18, color }}
          >
            {clampedScore.toFixed(0)}
          </span>
          <span className="text-xs text-slate-500 font-mono mt-0.5">/ 100</span>
        </div>
      </div>
      {showLabel && (
        <div
          className="text-xs font-bold tracking-widest font-mono"
          style={{ color }}
        >
          {label}
        </div>
      )}
    </div>
  );
}

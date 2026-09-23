import clsx from "clsx";
import type { EvidenceFamilyScore } from "../types";

interface EvidenceBarProps {
  evidence: EvidenceFamilyScore;
  showDescription?: boolean;
}

const FAMILY_LABELS: Record<string, string> = {
  bit_representation: "Bit Representation",
  trojan_signature: "Trojan Signature",
  structural: "Structural Analysis",
  reference_deviation: "Reference Deviation",
  behavioral: "Behavioral Analysis",
  model_xray: "Model X-Ray",
};

const LEVEL_CONFIG = {
  Low: { bar: "bg-safe-500", text: "text-safe-400", badge: "bg-safe-500/10 text-safe-400 border-safe-500/30" },
  Medium: { bar: "bg-warning-500", text: "text-warning-400", badge: "bg-warning-500/10 text-warning-400 border-warning-500/30" },
  High: { bar: "bg-danger-500", text: "text-danger-400", badge: "bg-danger-500/10 text-danger-400 border-danger-500/30" },
  Unavailable: { bar: "bg-slate-700", text: "text-slate-500", badge: "bg-slate-700/50 text-slate-500 border-slate-600/30" },
};

export default function EvidenceBar({ evidence, showDescription = true }: EvidenceBarProps) {
  const lvl = LEVEL_CONFIG[evidence.level];
  const label = FAMILY_LABELS[evidence.family] ?? evidence.family;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-sm font-medium text-slate-200 truncate">{label}</span>
          {!evidence.available && (
            <span className="text-xs text-slate-600 font-mono">(unavailable)</span>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className={clsx("text-sm font-mono font-semibold", lvl.text)}>
            {evidence.available ? `${evidence.score.toFixed(1)}` : "—"}
          </span>
          <span
            className={clsx(
              "px-2 py-0.5 text-xs font-semibold rounded-full border",
              lvl.badge
            )}
          >
            {evidence.level}
          </span>
        </div>
      </div>
      {/* Progress bar */}
      <div className="h-1.5 w-full bg-navy-950 rounded-full overflow-hidden">
        <div
          className={clsx("h-full rounded-full transition-all duration-700", lvl.bar)}
          style={{ width: evidence.available ? `${evidence.score}%` : "0%" }}
        />
      </div>
      {showDescription && evidence.description && (
        <p className="text-xs text-slate-500 leading-relaxed">{evidence.description}</p>
      )}
    </div>
  );
}

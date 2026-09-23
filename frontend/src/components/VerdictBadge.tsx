import clsx from "clsx";
import type { VerdictLevel } from "../types";

interface VerdictBadgeProps {
  verdict: VerdictLevel;
  size?: "sm" | "md" | "lg";
  showIcon?: boolean;
}

const CONFIG: Record<
  VerdictLevel,
  { label: string; classes: string; iconPath: string }
> = {
  BENIGN: {
    label: "Benign",
    classes: "bg-safe-500/10 text-safe-400 border-safe-500/30",
    iconPath:
      "M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z",
  },
  SUSPICIOUS: {
    label: "Suspicious",
    classes: "bg-warning-500/10 text-warning-400 border-warning-500/30",
    iconPath:
      "M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z",
  },
  HIGH_RISK: {
    label: "High Risk",
    classes: "bg-danger-600/10 text-danger-400 border-danger-600/30",
    iconPath:
      "M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z",
  },
  UNKNOWN: {
    label: "Unknown",
    classes: "bg-slate-700/50 text-slate-400 border-slate-600/30",
    iconPath:
      "M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z",
  },
};

export default function VerdictBadge({
  verdict,
  size = "md",
  showIcon = true,
}: VerdictBadgeProps) {
  const cfg = CONFIG[verdict];
  const sizeClasses = {
    sm: "px-2 py-0.5 text-xs gap-1",
    md: "px-3 py-1 text-sm gap-1.5",
    lg: "px-4 py-1.5 text-base gap-2",
  }[size];
  const iconSize = { sm: "w-3 h-3", md: "w-4 h-4", lg: "w-5 h-5" }[size];

  return (
    <span
      className={clsx(
        "inline-flex items-center font-semibold rounded-full border",
        cfg.classes,
        sizeClasses
      )}
    >
      {showIcon && (
        <svg
          className={iconSize}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d={cfg.iconPath} />
        </svg>
      )}
      {cfg.label}
    </span>
  );
}

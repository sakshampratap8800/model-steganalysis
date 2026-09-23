import clsx from "clsx";
import type { ScanStatus } from "../types";

interface ScanStatusBadgeProps {
  status: ScanStatus;
  size?: "sm" | "md";
}

const CONFIG: Record<ScanStatus, { label: string; classes: string; showSpinner?: boolean }> = {
  queued: {
    label: "Queued",
    classes: "bg-slate-700/50 text-slate-300 border-slate-600/30",
  },
  running: {
    label: "Running",
    classes: "bg-cyber-600/10 text-cyber-400 border-cyber-600/30",
    showSpinner: true,
  },
  complete: {
    label: "Complete",
    classes: "bg-safe-500/10 text-safe-400 border-safe-500/30",
  },
  failed: {
    label: "Failed",
    classes: "bg-danger-600/10 text-danger-400 border-danger-600/30",
  },
};

export default function ScanStatusBadge({ status, size = "md" }: ScanStatusBadgeProps) {
  const cfg = CONFIG[status];
  const sizeClasses = {
    sm: "px-2 py-0.5 text-xs gap-1",
    md: "px-3 py-1 text-sm gap-1.5",
  }[size];

  return (
    <span
      className={clsx(
        "inline-flex items-center font-medium rounded-full border",
        cfg.classes,
        sizeClasses
      )}
    >
      {cfg.showSpinner ? (
        <span className="w-2.5 h-2.5 rounded-full border border-current border-t-transparent animate-spin shrink-0" />
      ) : (
        <span
          className={clsx("w-2 h-2 rounded-full shrink-0", {
            "bg-slate-500": status === "queued",
            "bg-safe-500": status === "complete",
            "bg-danger-500": status === "failed",
          })}
        />
      )}
      {cfg.label}
    </span>
  );
}

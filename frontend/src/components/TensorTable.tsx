import { useState } from "react";
import clsx from "clsx";
import type { TensorProfile } from "../types";

interface TensorTableProps {
  tensors: TensorProfile[];
  maxRows?: number;
}

type SortKey = keyof Pick<TensorProfile, "name" | "parameter_count" | "mean" | "std_dev" | "sparsity">;
type SortDir = "asc" | "desc";

function fmt(n: number, digits = 4): string {
  if (Math.abs(n) >= 1e6) return n.toExponential(2);
  return n.toFixed(digits);
}

function fmtCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export default function TensorTable({ tensors, maxRows }: TensorTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("parameter_count");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [page, setPage] = useState(0);
  const pageSize = maxRows ?? 20;

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
    setPage(0);
  };

  const sorted = [...tensors].sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    const cmp = typeof av === "string"
      ? av.localeCompare(bv as string)
      : (av as number) - (bv as number);
    return sortDir === "asc" ? cmp : -cmp;
  });

  const totalPages = Math.ceil(sorted.length / pageSize);
  const visible = sorted.slice(page * pageSize, (page + 1) * pageSize);

  const SortIcon = ({ k }: { k: SortKey }) =>
    sortKey === k ? (
      <span className="ml-1 text-cyber-400">{sortDir === "asc" ? "▲" : "▼"}</span>
    ) : (
      <span className="ml-1 text-slate-700">▼</span>
    );

  const th = "px-3 py-2.5 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider whitespace-nowrap select-none cursor-pointer hover:text-cyber-400 transition-colors";

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto rounded-lg border border-slate-700/50">
        <table className="w-full text-sm">
          <thead className="bg-navy-950">
            <tr>
              <th className={th} onClick={() => handleSort("name")}>
                Name <SortIcon k="name" />
              </th>
              <th className={clsx(th, "text-right")}>Shape</th>
              <th className={clsx(th, "text-right")} onClick={() => handleSort("parameter_count")}>
                Params <SortIcon k="parameter_count" />
              </th>
              <th className={clsx(th, "text-right")} onClick={() => handleSort("mean")}>
                Mean <SortIcon k="mean" />
              </th>
              <th className={clsx(th, "text-right")} onClick={() => handleSort("std_dev")}>
                Std Dev <SortIcon k="std_dev" />
              </th>
              <th className={clsx(th, "text-right")} onClick={() => handleSort("sparsity")}>
                Sparsity <SortIcon k="sparsity" />
              </th>
              <th className={clsx(th, "text-right")}>dtype</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {visible.map((t, i) => (
              <tr key={i} className="hover:bg-navy-800/50 transition-colors">
                <td className="px-3 py-2 font-mono text-xs text-slate-300 max-w-[300px] truncate" title={t.name}>
                  {t.name}
                </td>
                <td className="px-3 py-2 font-mono text-xs text-slate-400 text-right whitespace-nowrap">
                  [{t.shape.join(", ")}]
                </td>
                <td className="px-3 py-2 font-mono text-xs text-slate-300 text-right">
                  {fmtCount(t.parameter_count)}
                </td>
                <td className="px-3 py-2 font-mono text-xs text-slate-400 text-right">
                  {fmt(t.mean)}
                </td>
                <td className="px-3 py-2 font-mono text-xs text-slate-400 text-right">
                  {fmt(t.std_dev)}
                </td>
                <td className="px-3 py-2 text-right">
                  <SparsityBar value={t.sparsity} />
                </td>
                <td className="px-3 py-2 font-mono text-xs text-slate-500 text-right">
                  {t.dtype}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {totalPages > 1 && (
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>
            Showing {page * pageSize + 1}–{Math.min((page + 1) * pageSize, sorted.length)} of {sorted.length}
          </span>
          <div className="flex gap-1">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-2 py-1 rounded border border-slate-700 hover:border-cyber-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              ‹ Prev
            </button>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="px-2 py-1 rounded border border-slate-700 hover:border-cyber-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Next ›
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function SparsityBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const color = pct > 80 ? "bg-warning-500" : pct > 50 ? "bg-cyber-500" : "bg-slate-600";
  return (
    <div className="flex items-center gap-2 justify-end">
      <div className="w-16 h-1.5 bg-navy-950 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-xs text-slate-400 w-10 text-right">
        {(pct).toFixed(1)}%
      </span>
    </div>
  );
}

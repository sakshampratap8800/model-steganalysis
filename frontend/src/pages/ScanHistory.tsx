import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import ScanStatusBadge from '../components/ScanStatusBadge';
import VerdictBadge from '../components/VerdictBadge';
import EmptyState from '../components/EmptyState';
import ErrorState from '../components/ErrorState';
import LoadingState from '../components/LoadingState';
import { scansApi } from '../services/api';
import type { ScanRecord } from '../types';

export default function ScanHistory() {
  const [scans, setScans] = useState<ScanRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  // Pagination
  const [page, setPage] = useState(0);
  const [total, setTotal] = useState(0);
  const limit = 20;

  const fetchHistory = () => {
    setLoading(true);
    scansApi.list(page * limit, limit)
      .then(data => {
        setScans(data.items);
        setTotal(data.total);
        setError(null);
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchHistory();
  }, [page]);

  const totalPages = Math.ceil(total / limit);

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <h1 className="text-3xl font-bold text-slate-100">Scan History</h1>
      
      <div className="bg-navy-800 rounded-xl border border-slate-700 overflow-hidden">
        {loading && scans.length === 0 ? (
          <div className="p-8"><LoadingState message="Loading scan history..." /></div>
        ) : error ? (
          <div className="p-8"><ErrorState message={error} onRetry={fetchHistory} /></div>
        ) : scans.length === 0 ? (
          <div className="p-8">
            <EmptyState 
              title="No Scans Yet" 
              description="Upload a model to begin scanning."
              action={{ label: "Upload Model", onClick: () => { window.location.href = '/upload'; } }}
            />
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-slate-300">
                <thead className="bg-navy-900 text-slate-400 whitespace-nowrap">
                  <tr>
                    <th className="px-6 py-3 font-medium">Filename</th>
                    <th className="px-6 py-3 font-medium">Scan ID</th>
                    <th className="px-6 py-3 font-medium">Status</th>
                    <th className="px-6 py-3 font-medium">Verdict</th>
                    <th className="px-6 py-3 font-medium text-right">Risk Score</th>
                    <th className="px-6 py-3 font-medium text-right">Size</th>
                    <th className="px-6 py-3 font-medium">Date</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700/50">
                  {scans.map(scan => (
                    <tr key={scan.id} className="hover:bg-slate-800/50 transition-colors">
                      <td className="px-6 py-4 max-w-[200px] truncate">
                        <Link to={`/scan/${scan.id}`} className="text-cyber-400 hover:underline font-medium">
                          {scan.filename}
                        </Link>
                      </td>
                      <td className="px-6 py-4 text-slate-500 font-mono text-xs">{scan.id}</td>
                      <td className="px-6 py-4"><ScanStatusBadge status={scan.status} /></td>
                      <td className="px-6 py-4">
                        {scan.verdict ? <VerdictBadge verdict={scan.verdict} size="sm" /> : <span className="text-slate-600">—</span>}
                      </td>
                      <td className="px-6 py-4 text-right font-mono">
                        {scan.overall_score !== undefined && scan.overall_score !== null ? scan.overall_score.toFixed(1) : <span className="text-slate-600">—</span>}
                      </td>
                      <td className="px-6 py-4 text-right text-slate-400 text-xs whitespace-nowrap">
                        {(scan.file_size_bytes / (1024 * 1024)).toFixed(1)} MB
                      </td>
                      <td className="px-6 py-4 text-slate-500 whitespace-nowrap text-xs">
                        {new Date(scan.created_at).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {totalPages > 1 && (
              <div className="px-6 py-4 bg-navy-900/50 border-t border-slate-700/50 flex items-center justify-between text-xs text-slate-400">
                <span>
                  Showing {page * limit + 1} to {Math.min((page + 1) * limit, total)} of {total} results
                </span>
                <div className="flex gap-2">
                  <button 
                    onClick={() => setPage(p => Math.max(0, p - 1))}
                    disabled={page === 0}
                    className="px-3 py-1.5 rounded border border-slate-700 hover:border-cyber-500 hover:text-cyber-400 disabled:opacity-30 disabled:hover:border-slate-700 disabled:hover:text-slate-400 transition-colors"
                  >
                    Previous
                  </button>
                  <button 
                    onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                    disabled={page >= totalPages - 1}
                    className="px-3 py-1.5 rounded border border-slate-700 hover:border-cyber-500 hover:text-cyber-400 disabled:opacity-30 disabled:hover:border-slate-700 disabled:hover:text-slate-400 transition-colors"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

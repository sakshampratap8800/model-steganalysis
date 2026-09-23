import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import ScanStatusBadge from '../components/ScanStatusBadge';
import VerdictBadge from '../components/VerdictBadge';
import EmptyState from '../components/EmptyState';
import ErrorState from '../components/ErrorState';
import LoadingState from '../components/LoadingState';
import { scansApi } from '../services/api';
import type { ScanRecord } from '../types';

export default function Dashboard() {
  const [scans, setScans] = useState<ScanRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    scansApi.list(0, 5)
      .then(data => {
        setScans(data.items);
        setError(null);
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <h1 className="text-3xl font-bold text-slate-100">Dashboard</h1>
      
      <div className="bg-navy-800 p-6 rounded-xl border border-slate-700">
        <h2 className="text-xl font-semibold mb-4 text-white">Recent Scans</h2>
        
        {loading ? (
          <LoadingState message="Loading recent scans..." />
        ) : error ? (
          <ErrorState message={error} onRetry={() => window.location.reload()} />
        ) : scans.length === 0 ? (
          <EmptyState 
            title="No Scans Yet" 
            description="Upload a model to begin scanning."
            action={{ label: "Upload Model", onClick: () => { window.location.href = '/upload'; } }}
          />
        ) : (
          <div className="divide-y divide-slate-700/50">
            {scans.map(scan => (
              <div key={scan.id} className="py-4 flex items-center justify-between">
                <div>
                  <Link to={`/scan/${scan.id}`} className="text-cyber-400 font-medium hover:underline">
                    {scan.filename}
                  </Link>
                  <div className="text-xs text-slate-500 mt-1 font-mono">{scan.id}</div>
                </div>
                <div className="flex items-center gap-4">
                  {scan.verdict && <VerdictBadge verdict={scan.verdict} size="sm" />}
                  <ScanStatusBadge status={scan.status} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

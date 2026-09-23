import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import RiskScoreGauge from '../components/RiskScoreGauge';
import VerdictBadge from '../components/VerdictBadge';
import ScanStatusBadge from '../components/ScanStatusBadge';
import EvidenceBar from '../components/EvidenceBar';
import TensorTable from '../components/TensorTable';
import { scansApi } from '../services/api';
import { usePolling } from '../hooks/usePolling';
import type { ScanReport, ScanStatusResponse } from '../types';
import ErrorState from '../components/ErrorState';
import LoadingState from '../components/LoadingState';

export default function ScanDetails() {
  const { scanId } = useParams();
  const [report, setReport] = useState<ScanReport | null>(null);
  
  const { data: status, error: statusError, loading: statusLoading } = usePolling<ScanStatusResponse>(
    () => scansApi.getStatus(scanId!),
    2000,
    {
      enabled: !!scanId,
      stopCondition: (data) => data.status === 'complete' || data.status === 'failed',
    }
  );

  useEffect(() => {
    if (status?.status === 'complete' && scanId) {
      scansApi.getReport(scanId).then(setReport).catch(console.error);
    }
  }, [status?.status, scanId]);

  if (statusError) return <ErrorState message={statusError.message} />;
  if (status?.status === 'failed') return <ErrorState message={status?.message || status?.progress_message || "Scan failed."} />;
  if (!status || (statusLoading && !status)) return <LoadingState message="Loading scan details..." />;

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="flex justify-between items-center bg-navy-800 p-6 rounded-xl border border-slate-700">
        <div>
          <h1 className="text-2xl font-bold text-white mb-2">Scan Details: {scanId}</h1>
          <ScanStatusBadge status={status.status} />
        </div>
        
        {report && (
          <div className="flex items-center gap-6">
            <VerdictBadge verdict={report.verdict} />
            <div className="w-32 h-32">
              <RiskScoreGauge score={report.risk_score} />
            </div>
          </div>
        )}
      </div>

      {status.status === 'running' && (
        <div className="bg-navy-800 p-6 rounded-xl border border-slate-700 animate-pulse text-center text-slate-300">
          Analyzing model... {status.message || status.progress_message || 'Please wait.'}
        </div>
      )}

      {report && (
        <>
          <div className="bg-navy-800 p-6 rounded-xl border border-slate-700">
            <h2 className="text-xl font-semibold mb-4 text-white">Global Evidence</h2>
            <div className="space-y-4">
              {report.evidence_families && report.evidence_families.map((ev, idx) => (
                <div key={idx}>
                  <div className="flex justify-between text-sm mb-1 text-slate-300">
                    <span>{ev.family}</span>
                  </div>
                  <EvidenceBar evidence={ev} showDescription={false} />
                  <p className="text-xs text-slate-500 mt-1">{ev.description}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-navy-800 p-6 rounded-xl border border-slate-700">
            <h2 className="text-xl font-semibold mb-4 text-white">Tensor Analysis</h2>
            <TensorTable tensors={report.model_profile?.tensors || []} />
          </div>
        </>
      )}
    </div>
  );
}

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
}

export default function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-16 text-center">
      <div className="w-14 h-14 rounded-full bg-danger-600/10 border border-danger-600/30 flex items-center justify-center">
        <svg className="w-7 h-7 text-danger-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round"
            d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
        </svg>
      </div>
      <div>
        <h3 className="text-base font-semibold text-danger-400">Error</h3>
        <p className="text-sm text-slate-400 mt-1 max-w-md font-mono">{message}</p>
      </div>
      {onRetry && (
        <button onClick={onRetry} className="cyber-button">
          Retry
        </button>
      )}
    </div>
  );
}

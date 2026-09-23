interface EmptyStateProps {
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export default function EmptyState({ title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-20 text-center">
      <div className="w-16 h-16 rounded-full bg-navy-800 border border-slate-700 flex items-center justify-center">
        <svg className="w-8 h-8 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round"
            d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 15.803 7.5 7.5 0 0015.803 15.803z" />
        </svg>
      </div>
      <div>
        <h3 className="text-base font-semibold text-slate-300">{title}</h3>
        {description && <p className="text-sm text-slate-500 mt-1 max-w-md">{description}</p>}
      </div>
      {action && (
        <button onClick={action.onClick} className="cyber-button mt-2">
          {action.label}
        </button>
      )}
    </div>
  );
}

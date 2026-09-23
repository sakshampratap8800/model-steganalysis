interface LoadingStateProps {
  message?: string;
  size?: "sm" | "md" | "lg";
}

export default function LoadingState({ message = "Loading...", size = "md" }: LoadingStateProps) {
  const spinnerSize = { sm: "w-6 h-6", md: "w-10 h-10", lg: "w-16 h-16" }[size];
  const textSize = { sm: "text-xs", md: "text-sm", lg: "text-base" }[size];

  return (
    <div className="flex flex-col items-center justify-center gap-4 py-16 text-slate-400">
      <div className="relative">
        <div
          className={`${spinnerSize} rounded-full border-2 border-navy-700 border-t-cyber-500 animate-spin`}
        />
        <div
          className={`absolute inset-0 rounded-full border-2 border-transparent border-r-cyber-700 animate-spin`}
          style={{ animationDirection: "reverse", animationDuration: "1.5s" }}
        />
      </div>
      <span className={`${textSize} font-mono text-slate-400`}>{message}</span>
    </div>
  );
}

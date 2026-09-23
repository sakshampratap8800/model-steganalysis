import { useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import clsx from "clsx";

// ─── Icons ────────────────────────────────────────────────────────────────────
function ShieldIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
    </svg>
  );
}

function GridIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25a2.25 2.25 0 01-2.25-2.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25A2.25 2.25 0 0113.5 18v-2.25z" />
    </svg>
  );
}

function PlusCircleIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M12 9v6m3-3H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );
}

function ClockIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );
}

function MenuIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
    </svg>
  );
}

// ─── Nav Items ────────────────────────────────────────────────────────────────
const NAV_ITEMS = [
  { to: "/", label: "Dashboard", Icon: GridIcon, exact: true },
  { to: "/scan/new", label: "New Scan", Icon: PlusCircleIcon, exact: false },
  { to: "/history", label: "History", Icon: ClockIcon, exact: false },
] as const;

// ─── Layout ───────────────────────────────────────────────────────────────────
interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const location = useLocation();

  return (
    <div className="flex h-screen overflow-hidden bg-navy-950 font-sans">
      {/* ── Sidebar ─────────────────────────────────────────────────────── */}
      <aside
        className={clsx(
          "flex flex-col bg-navy-900 border-r border-slate-700/50 transition-all duration-300 shrink-0",
          sidebarOpen ? "w-60" : "w-16"
        )}
      >
        {/* Logo */}
        <div className="flex items-center gap-3 px-4 py-5 border-b border-slate-700/50">
          <ShieldIcon className="w-8 h-8 text-cyber-500 shrink-0" />
          {sidebarOpen && (
            <div className="overflow-hidden">
              <div className="text-cyber-400 font-bold text-sm tracking-wider whitespace-nowrap">
                STEGANALYSIS
              </div>
              <div className="text-slate-500 text-xs whitespace-nowrap">AI Security Platform</div>
            </div>
          )}
        </div>

        {/* Nav links */}
        <nav className="flex-1 py-4 px-2 space-y-1">
          {NAV_ITEMS.map(({ to, label, Icon, exact }) => (
            <NavLink
              key={to}
              to={to}
              end={exact}
              className={({ isActive }) =>
                clsx(
                  "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150",
                  isActive
                    ? "bg-cyber-600/20 text-cyber-400 border border-cyber-600/30"
                    : "text-slate-400 hover:text-slate-100 hover:bg-navy-800"
                )
              }
            >
              <Icon className="w-5 h-5 shrink-0" />
              {sidebarOpen && <span className="whitespace-nowrap">{label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-4 py-4 border-t border-slate-700/50">
          {sidebarOpen && (
            <div className="text-xs text-slate-600 font-mono">v0.1.0-alpha</div>
          )}
        </div>
      </aside>

      {/* ── Main area ───────────────────────────────────────────────────── */}
      <div className="flex flex-col flex-1 overflow-hidden">
        {/* Top bar */}
        <header className="flex items-center gap-4 px-6 py-4 bg-navy-900 border-b border-slate-700/50 shrink-0">
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            className="text-slate-400 hover:text-slate-100 transition-colors"
            aria-label="Toggle sidebar"
          >
            <MenuIcon className="w-5 h-5" />
          </button>
          <div className="flex-1">
            <Breadcrumb path={location.pathname} />
          </div>
          <div className="flex items-center gap-2 text-xs font-mono text-slate-500">
            <span className="w-2 h-2 rounded-full bg-safe-500 animate-pulse" />
            System Online
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-6">
          {children}
        </main>
      </div>
    </div>
  );
}

// ─── Breadcrumb helper ────────────────────────────────────────────────────────
function Breadcrumb({ path }: { path: string }) {
  const segments = path.split("/").filter(Boolean);
  if (segments.length === 0) return <span className="text-sm text-slate-400">Dashboard</span>;

  return (
    <nav className="flex items-center gap-1 text-sm">
      <span className="text-slate-500">~</span>
      {segments.map((seg, idx) => (
        <span key={idx} className="flex items-center gap-1">
          <span className="text-slate-600">/</span>
          <span className={idx === segments.length - 1 ? "text-slate-200" : "text-slate-500"}>
            {seg}
          </span>
        </span>
      ))}
    </nav>
  );
}

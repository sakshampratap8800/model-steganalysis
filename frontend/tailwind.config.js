/** @type {import("tailwindcss").Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          950: "#0a0f1e",
          900: "#0f172a",
          800: "#1e293b",
          700: "#1e3a5f",
          600: "#253a5e",
        },
        cyber: {
          400: "#22d3ee",
          500: "#06b6d4",
          600: "#0891b2",
          700: "#0e7490",
        },
        danger: {
          400: "#f87171",
          500: "#ef4444",
          600: "#dc2626",
          700: "#b91c1c",
        },
        warning: {
          400: "#fbbf24",
          500: "#f59e0b",
          600: "#d97706",
          700: "#b45309",
        },
        safe: {
          400: "#34d399",
          500: "#10b981",
          600: "#059669",
          700: "#047857",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "Cascadia Code", "ui-monospace", "monospace"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      animation: {
        "spin-slow": "spin 2s linear infinite",
        "pulse-slow": "pulse 3s ease-in-out infinite",
        "scan-line": "scanline 2s ease-in-out infinite",
      },
      keyframes: {
        scanline: {
          "0%, 100%": { opacity: "0.3" },
          "50%": { opacity: "1" },
        },
      },
      boxShadow: {
        cyber: "0 0 20px rgba(6,182,212,0.15), 0 0 40px rgba(6,182,212,0.05)",
        danger: "0 0 20px rgba(239,68,68,0.2), 0 0 40px rgba(239,68,68,0.05)",
        warning: "0 0 20px rgba(245,158,11,0.2)",
        safe: "0 0 20px rgba(16,185,129,0.2)",
      },
    },
  },
  plugins: [],
};

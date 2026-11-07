/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        body: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      colors: {
        forensic: {
          bg: "var(--bg-primary)",
          card: "var(--bg-card)",
          elevated: "var(--bg-elevated)",
          border: "var(--border)",
          text: "var(--text-primary)",
          muted: "var(--text-secondary)",
          accent: "var(--accent-blue)",
          fake: "var(--verdict-fake)",
          real: "var(--verdict-real)",
        },
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(59,130,246,0.2), 0 0 25px rgba(59,130,246,0.18)",
        fake: "0 0 35px rgba(239,68,68,0.18)",
        real: "0 0 35px rgba(34,197,94,0.18)",
      },
      backgroundImage: {
        grid: "radial-gradient(circle at top, rgba(59,130,246,0.12), transparent 45%), linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)",
      },
      backgroundSize: {
        grid: "auto, 42px 42px, 42px 42px",
      },
    },
  },
  plugins: [],
};

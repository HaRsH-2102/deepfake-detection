import PropTypes from "prop-types";
import { motion } from "framer-motion";

function Header({ theme, onToggleTheme }) {
  return (
    <motion.header
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="relative overflow-hidden rounded-[2rem] border border-forensic-border bg-forensic-card/80 px-6 py-8 shadow-glow sm:px-10"
    >
      <div className="scanlines absolute inset-0" />
      <div className="relative flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
        <div className="space-y-4">
          <div className="data-chip">FORENSIC AI ANALYSIS NODE</div>
          <div>
            <h1 className="font-mono text-4xl font-extrabold tracking-[0.25em] text-forensic-text sm:text-6xl">
              DEEPFAKE DETECTOR
            </h1>
            <p className="mt-3 max-w-2xl text-sm text-forensic-muted sm:text-base">
              AI-powered forensic analysis for images and videos, with explainability,
              metadata checks, live frame streaming, and session recall.
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="rounded-full border border-forensic-border bg-forensic-elevated px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] text-forensic-muted">
            localhost telemetry ready
          </div>
          <button
            type="button"
            onClick={onToggleTheme}
            className="rounded-full border border-forensic-border bg-forensic-elevated px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] text-forensic-text transition hover:border-forensic-accent hover:text-forensic-accent"
          >
            {theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          </button>
        </div>
      </div>
    </motion.header>
  );
}

Header.propTypes = {
  theme: PropTypes.string.isRequired,
  onToggleTheme: PropTypes.func.isRequired,
};

export default Header;

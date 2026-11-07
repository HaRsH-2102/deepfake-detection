import PropTypes from "prop-types";

function ConfidenceBar({ confidence }) {
  const percentage = Math.max(0, Math.min(100, confidence * 100));

  return (
    <div className="panel px-5 py-5">
      <div className="mb-4 flex items-center justify-between">
        <p className="panel-heading">Confidence Scale</p>
        <p className="font-mono text-sm text-forensic-muted">{percentage.toFixed(1)}%</p>
      </div>

      <div className="relative h-4 rounded-full bg-white/5">
        <div
          className="h-full rounded-full bg-gradient-to-r from-forensic-real via-yellow-400 to-forensic-fake transition-all duration-700"
          style={{ width: `${percentage}%` }}
        />
        <div
          className="absolute top-1/2 h-7 w-7 -translate-y-1/2 rounded-full border-2 border-white bg-forensic-accent shadow-glow"
          style={{ left: `calc(${percentage}% - 14px)` }}
        />
      </div>

      <div className="mt-3 flex items-center justify-between text-xs uppercase tracking-[0.18em] text-forensic-muted">
        <span>Likely Real</span>
        <span>Borderline</span>
        <span>Likely Fake</span>
      </div>
    </div>
  );
}

ConfidenceBar.propTypes = {
  confidence: PropTypes.number.isRequired,
};

export default ConfidenceBar;

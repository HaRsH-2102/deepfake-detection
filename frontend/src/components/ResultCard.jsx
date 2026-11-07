import PropTypes from "prop-types";
import { motion } from "framer-motion";

function ResultCard({ prediction, confidence, processingTime, mode }) {
  const isFake = prediction === "FAKE";
  const ringClass = isFake ? "shadow-fake" : "shadow-real";

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`panel relative overflow-hidden px-6 py-7 ${ringClass}`}
    >
      <div className="absolute inset-y-0 right-0 w-40 bg-gradient-to-l from-forensic-accent/10 to-transparent" />
      <p className="panel-heading">Verdict</p>
      <div className="mt-6 flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
        <div>
          <div
            className="inline-flex h-28 w-28 items-center justify-center rounded-full border text-xs font-mono uppercase tracking-[0.24em]"
            style={{
              borderColor: isFake ? "var(--verdict-fake)" : "var(--verdict-real)",
              boxShadow: isFake
                ? "0 0 0 12px var(--verdict-fake-glow)"
                : "0 0 0 12px var(--verdict-real-glow)",
            }}
          >
            Scan
          </div>
          <h2
            className="mt-6 font-mono text-5xl font-extrabold tracking-[0.2em]"
            style={{ color: isFake ? "var(--verdict-fake)" : "var(--verdict-real)" }}
          >
            {prediction}
          </h2>
          <p className="mt-2 text-sm uppercase tracking-[0.24em] text-forensic-muted">
            {mode} analysis
          </p>
        </div>

        <div className="space-y-3 text-right">
          <p className="text-sm uppercase tracking-[0.24em] text-forensic-muted">
            Confidence
          </p>
          <p className="font-mono text-4xl font-bold text-forensic-text">
            {(confidence * 100).toFixed(2)}%
          </p>
          {processingTime ? (
            <p className="text-sm text-forensic-muted">{processingTime} ms runtime</p>
          ) : null}
        </div>
      </div>
    </motion.div>
  );
}

ResultCard.propTypes = {
  prediction: PropTypes.string.isRequired,
  confidence: PropTypes.number.isRequired,
  processingTime: PropTypes.number,
  mode: PropTypes.string.isRequired,
};

export default ResultCard;

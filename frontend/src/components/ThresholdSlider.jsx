import PropTypes from "prop-types";

function ThresholdSlider({ label, value, min, max, step, onChange, hint }) {
  return (
    <div className="rounded-2xl border border-forensic-border bg-black/10 p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-forensic-muted">
            {label}
          </p>
          <p className="mt-2 text-sm text-forensic-muted">{hint}</p>
        </div>
        <div className="rounded-full bg-forensic-elevated px-3 py-2 font-mono text-sm text-forensic-text">
          {Number(value).toFixed(2)}
        </div>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-4 h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-[var(--accent-blue)]"
      />
    </div>
  );
}

ThresholdSlider.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.number.isRequired,
  min: PropTypes.number,
  max: PropTypes.number,
  step: PropTypes.number,
  onChange: PropTypes.func.isRequired,
  hint: PropTypes.string,
};

ThresholdSlider.defaultProps = {
  min: 0,
  max: 1,
  step: 0.01,
  hint: "",
};

export default ThresholdSlider;

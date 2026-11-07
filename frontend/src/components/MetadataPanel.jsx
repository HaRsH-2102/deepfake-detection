import PropTypes from "prop-types";

function MetadataPanel({ flags }) {
  const hasFlags = Boolean(flags?.length);

  return (
    <div className="panel p-5">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="panel-heading">Metadata / EXIF Forensics</p>
          <p className="mt-2 text-sm text-forensic-muted">
            Edit traces, missing tags, and metadata inconsistencies.
          </p>
        </div>
        <span
          className="data-chip"
          style={{
            color: hasFlags ? "var(--verdict-fake)" : "var(--verdict-real)",
            borderColor: hasFlags ? "rgba(239,68,68,0.25)" : "rgba(34,197,94,0.25)",
          }}
        >
          {hasFlags ? `${flags.length} flags` : "clean metadata"}
        </span>
      </div>

      <div className="mt-5 flex flex-wrap gap-3">
        {hasFlags ? (
          flags.map((flag) => (
            <span
              key={flag}
              className="rounded-full px-4 py-2 text-sm"
              style={{
                background: flag.toLowerCase().includes("software")
                  ? "rgba(239,68,68,0.14)"
                  : "rgba(251,191,36,0.14)",
                color: flag.toLowerCase().includes("software")
                  ? "var(--verdict-fake)"
                  : "#facc15",
              }}
            >
              {flag}
            </span>
          ))
        ) : (
          <span className="rounded-full bg-forensic-real/10 px-4 py-2 text-sm text-forensic-real">
            No suspicious metadata detected
          </span>
        )}
      </div>
    </div>
  );
}

MetadataPanel.propTypes = {
  flags: PropTypes.arrayOf(PropTypes.string),
};

MetadataPanel.defaultProps = {
  flags: [],
};

export default MetadataPanel;

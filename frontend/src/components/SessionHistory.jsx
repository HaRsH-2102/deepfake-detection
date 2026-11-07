import PropTypes from "prop-types";

function SessionHistory({ history, onRestore, onClear }) {
  return (
    <aside className="panel p-5">
      <div className="flex items-center justify-between">
        <div>
          <p className="panel-heading">Session History</p>
          <p className="mt-2 text-sm text-forensic-muted">Last 10 analyses in this browser.</p>
        </div>
        <button
          type="button"
          onClick={onClear}
          className="rounded-full border border-forensic-border px-3 py-2 font-mono text-xs uppercase tracking-[0.18em] text-forensic-muted transition hover:border-forensic-accent hover:text-forensic-accent"
        >
          Clear
        </button>
      </div>

      <div className="mt-5 space-y-3">
        {history.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-forensic-border px-4 py-8 text-center text-sm text-forensic-muted">
            No saved analysis yet.
          </div>
        ) : (
          history.map((entry, index) => (
            <button
              key={`${entry.timestamp}-${index}`}
              type="button"
              onClick={() => onRestore(entry)}
              className="flex w-full items-center gap-4 rounded-2xl border border-forensic-border bg-black/10 p-3 text-left transition hover:border-forensic-accent"
            >
              {entry.thumbnail ? (
                <img
                  src={`data:image/png;base64,${entry.thumbnail}`}
                  alt={entry.mode}
                  className="h-16 w-16 rounded-xl object-cover"
                />
              ) : (
                <div className="flex h-16 w-16 items-center justify-center rounded-xl bg-forensic-elevated font-mono text-xs uppercase tracking-[0.18em] text-forensic-muted">
                  {entry.mode}
                </div>
              )}
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-3">
                  <p className="font-mono text-xs uppercase tracking-[0.18em] text-forensic-muted">
                    {entry.mode}
                  </p>
                  {entry.prediction ? (
                    <span
                      className="rounded-full px-2 py-1 text-[11px] font-mono"
                      style={{
                        background:
                          entry.prediction === "FAKE"
                            ? "rgba(239,68,68,0.12)"
                            : "rgba(34,197,94,0.12)",
                        color:
                          entry.prediction === "FAKE"
                            ? "var(--verdict-fake)"
                            : "var(--verdict-real)",
                      }}
                    >
                      {entry.prediction}
                    </span>
                  ) : null}
                </div>
                <p className="mt-2 truncate text-sm text-forensic-text">
                  {entry.title || "Untitled analysis"}
                </p>
                <p className="mt-1 text-xs text-forensic-muted">
                  {new Date(entry.timestamp).toLocaleString()}
                </p>
              </div>
            </button>
          ))
        )}
      </div>
    </aside>
  );
}

SessionHistory.propTypes = {
  history: PropTypes.arrayOf(PropTypes.object).isRequired,
  onRestore: PropTypes.func.isRequired,
  onClear: PropTypes.func.isRequired,
};

export default SessionHistory;

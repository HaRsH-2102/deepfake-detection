import PropTypes from "prop-types";

function ProgressStream({ progress }) {
  if (!progress) {
    return null;
  }

  const total = progress.total || 1;
  const percentage = Math.round(((progress.current || 0) / total) * 100);

  return (
    <div className="panel grid gap-5 p-5 lg:grid-cols-[1fr_220px]">
      <div>
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="panel-heading">Live Frame Stream</p>
            <p className="mt-2 text-sm text-forensic-muted">
              Streaming frame-by-frame confidence over WebSocket.
            </p>
          </div>
          <span className="data-chip">{percentage}% complete</span>
        </div>

        <div className="mt-5 h-3 rounded-full bg-white/5">
          <div
            className="h-full rounded-full bg-gradient-to-r from-forensic-accent to-cyan-300"
            style={{ width: `${percentage}%` }}
          />
        </div>

        <div className="mt-4 flex flex-wrap gap-3 text-sm text-forensic-muted">
          <span className="data-chip">
            frame {progress.frame ?? progress.current ?? 0}
          </span>
          <span className="data-chip">
            confidence {(Number(progress.confidence || 0) * 100).toFixed(1)}%
          </span>
          <span className="data-chip">
            {progress.current || 0}/{progress.total || 0} analyzed
          </span>
        </div>
      </div>

      <div className="rounded-2xl border border-forensic-border bg-black/15 p-3">
        {progress.frame_preview ? (
          <img
            src={`data:image/png;base64,${progress.frame_preview}`}
            alt="Live frame preview"
            className="h-44 w-full rounded-xl object-cover"
          />
        ) : (
          <div className="flex h-44 items-center justify-center rounded-xl bg-forensic-elevated font-mono text-xs uppercase tracking-[0.18em] text-forensic-muted">
            Awaiting frame preview
          </div>
        )}
      </div>
    </div>
  );
}

ProgressStream.propTypes = {
  progress: PropTypes.object,
};

export default ProgressStream;

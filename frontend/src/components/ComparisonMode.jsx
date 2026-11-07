import PropTypes from "prop-types";
import UploadZone from "./UploadZone";
import ResultCard from "./ResultCard";
import ConfidenceBar from "./ConfidenceBar";
import GradCamPanel from "./GradCamPanel";

function ComparisonMode({
  leftFile,
  rightFile,
  leftResult,
  rightResult,
  onChangeLeft,
  onChangeRight,
  onAnalyze,
  loading,
}) {
  const differenceReady = leftResult?.gradcam_image && rightResult?.gradcam_image;
  const delta = leftResult && rightResult
    ? Math.abs(leftResult.confidence - rightResult.confidence)
    : null;

  return (
    <div className="space-y-6">
      <div className="grid gap-6 xl:grid-cols-2">
        <UploadZone
          title="Comparison A"
          description="Upload the first face image for side-by-side analysis."
          accept={{ "image/*": [".jpg", ".jpeg", ".png", ".webp"] }}
          files={leftFile}
          onFilesChange={onChangeLeft}
        />
        <UploadZone
          title="Comparison B"
          description="Upload the second face image for direct forensic comparison."
          accept={{ "image/*": [".jpg", ".jpeg", ".png", ".webp"] }}
          files={rightFile}
          onFilesChange={onChangeRight}
        />
      </div>

      <button
        type="button"
        disabled={!leftFile || !rightFile || loading}
        onClick={onAnalyze}
        className="rounded-full bg-forensic-accent px-6 py-3 font-mono text-xs uppercase tracking-[0.2em] text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? "Analyzing both images..." : "Run side-by-side comparison"}
      </button>

      {(leftResult || rightResult) && (
        <div className="grid gap-6 xl:grid-cols-2">
          {leftResult ? (
            <div className="space-y-4">
              <ResultCard
                prediction={leftResult.prediction}
                confidence={leftResult.confidence}
                processingTime={leftResult.processing_time_ms}
                mode="compare-a"
              />
              <ConfidenceBar confidence={leftResult.confidence} />
              <GradCamPanel
                faceImage={leftResult.face_image}
                gradcamImage={leftResult.gradcam_image}
              />
            </div>
          ) : null}

          {rightResult ? (
            <div className="space-y-4">
              <ResultCard
                prediction={rightResult.prediction}
                confidence={rightResult.confidence}
                processingTime={rightResult.processing_time_ms}
                mode="compare-b"
              />
              <ConfidenceBar confidence={rightResult.confidence} />
              <GradCamPanel
                faceImage={rightResult.face_image}
                gradcamImage={rightResult.gradcam_image}
              />
            </div>
          ) : null}
        </div>
      )}

      {differenceReady ? (
        <div className="panel p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="panel-heading">Difference View</p>
              <p className="mt-2 text-sm text-forensic-muted">
                Overlaying Grad-CAM maps with blend difference for quick hotspot contrast.
              </p>
            </div>
            {delta !== null ? (
              <span className="data-chip">{(delta * 100).toFixed(2)}% confidence delta</span>
            ) : null}
          </div>

          <div className="mt-5 grid gap-5 lg:grid-cols-3">
            <img
              src={`data:image/png;base64,${leftResult.gradcam_image}`}
              alt="Left GradCAM"
              className="h-64 w-full rounded-2xl object-cover"
            />
            <img
              src={`data:image/png;base64,${rightResult.gradcam_image}`}
              alt="Right GradCAM"
              className="h-64 w-full rounded-2xl object-cover"
            />
            <div className="relative h-64 overflow-hidden rounded-2xl border border-forensic-border bg-black">
              <img
                src={`data:image/png;base64,${leftResult.gradcam_image}`}
                alt="Difference base"
                className="absolute inset-0 h-full w-full object-cover"
              />
              <img
                src={`data:image/png;base64,${rightResult.gradcam_image}`}
                alt="Difference overlay"
                className="absolute inset-0 h-full w-full object-cover mix-blend-difference opacity-80"
              />
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

ComparisonMode.propTypes = {
  leftFile: PropTypes.object,
  rightFile: PropTypes.object,
  leftResult: PropTypes.object,
  rightResult: PropTypes.object,
  onChangeLeft: PropTypes.func.isRequired,
  onChangeRight: PropTypes.func.isRequired,
  onAnalyze: PropTypes.func.isRequired,
  loading: PropTypes.bool,
};

export default ComparisonMode;

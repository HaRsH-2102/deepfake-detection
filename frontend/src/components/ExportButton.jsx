import PropTypes from "prop-types";
import toast from "react-hot-toast";
import { generateReport } from "../utils/generateReport";

function ExportButton({ result }) {
  if (!result) {
    return null;
  }

  async function handlePdfExport() {
    try {
      const blob = await generateReport({
        prediction: result.prediction,
        confidence: result.confidence,
        mode: result.mode,
        timestamp: result.timestamp ? new Date(result.timestamp) : new Date(),
        gradcamBase64: result.gradcam_image,
        limeBase64: result.lime_image,
        exifFlags: result.exif_flags || [],
        frameConfidences: result.frame_confidences || [],
        topFakeFrames: result.top_fake_frames || [],
      });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `deepfake-report-${result.mode}.pdf`;
      link.click();
      URL.revokeObjectURL(link.href);
      toast.success("PDF report exported.");
    } catch (error) {
      toast.error("Could not generate the PDF report.");
    }
  }

  async function handleCopyJson() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
      toast.success("JSON results copied.");
    } catch (error) {
      toast.error("Clipboard copy failed.");
    }
  }

  return (
    <div className="flex flex-wrap gap-3">
      <button
        type="button"
        onClick={handlePdfExport}
        className="rounded-full bg-forensic-accent px-5 py-3 font-mono text-xs uppercase tracking-[0.18em] text-white transition hover:opacity-90"
      >
        Download PDF Report
      </button>
      <button
        type="button"
        onClick={handleCopyJson}
        className="rounded-full border border-forensic-border px-5 py-3 font-mono text-xs uppercase tracking-[0.18em] text-forensic-text transition hover:border-forensic-accent hover:text-forensic-accent"
      >
        Copy JSON Results
      </button>
    </div>
  );
}

ExportButton.propTypes = {
  result: PropTypes.object,
};

export default ExportButton;

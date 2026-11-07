import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import toast from "react-hot-toast";
import Header from "./components/Header";
import UploadZone from "./components/UploadZone";
import ResultCard from "./components/ResultCard";
import ConfidenceBar from "./components/ConfidenceBar";
import GradCamPanel from "./components/GradCamPanel";
import LimePanel from "./components/LimePanel";
import VideoTimeline from "./components/VideoTimeline";
import FakeFramesGallery from "./components/FakeFramesGallery";
import BatchResultsTable from "./components/BatchResultsTable";
import SessionHistory from "./components/SessionHistory";
import ThresholdSlider from "./components/ThresholdSlider";
import ComparisonMode from "./components/ComparisonMode";
import MetadataPanel from "./components/MetadataPanel";
import ProgressStream from "./components/ProgressStream";
import ExportButton from "./components/ExportButton";
import { requestImageAnalysis, useDetection } from "./hooks/useDetection";
import { useSessionHistory } from "./hooks/useSessionHistory";
import { useWebSocket } from "./hooks/useWebSocket";

const modeOptions = [
  { id: "image", label: "Image" },
  { id: "video", label: "Video" },
  { id: "batch", label: "Batch" },
  { id: "compare", label: "Compare" },
];

const imageAccept = { "image/*": [".jpg", ".jpeg", ".png", ".webp"] };
const videoAccept = { "video/*": [".mp4", ".mov", ".avi", ".webm"] };

function App() {
  const detection = useDetection();
  const { history, addEntry, clearHistory } = useSessionHistory();
  const websocket = useWebSocket(import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/predict-video");

  const [theme, setTheme] = useState(() => localStorage.getItem("deepfake-theme") || "dark");
  const [settingsOpen, setSettingsOpen] = useState(true);
  const [primaryInput, setPrimaryInput] = useState(null);
  const [batchFiles, setBatchFiles] = useState([]);
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [compareFiles, setCompareFiles] = useState({ left: null, right: null });
  const [compareResults, setCompareResults] = useState({ left: null, right: null });
  const [compareLoading, setCompareLoading] = useState(false);
  const [settings, setSettings] = useState({
    imageThreshold: 0.85,
    videoThreshold: 0.4,
    frameInterval: 10,
    showGradcam: true,
    showLime: true,
    showLandmarks: true,
    enableMetadata: true,
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("deepfake-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!websocket.latestMessage) {
      return;
    }

    detection.handleVideoMessage(websocket.latestMessage);
    if (websocket.latestMessage.type === "complete") {
      const message = websocket.latestMessage;
      addEntry({
        mode: "video",
        prediction: message.prediction,
        title: `Video scan | ${(message.confidence * 100).toFixed(1)}% confidence`,
        timestamp: new Date().toISOString(),
        thumbnail: message.top_fake_frames?.[0]?.frame_image || null,
        result: {
          ...message,
          mode: "video",
          timestamp: new Date().toISOString(),
        },
      });
      websocket.disconnect();
      toast.success("Video analysis complete.");
    }

    if (websocket.latestMessage.type === "error") {
      websocket.disconnect();
      toast.error(websocket.latestMessage.detail || "Video analysis failed.");
    }
  }, [websocket.latestMessage]);

  function updateSetting(key, value) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  function resetAll() {
    detection.reset();
    setPrimaryInput(null);
    setBatchFiles([]);
    setSelectedFrame(null);
  }

  function saveHistory(result, mode) {
    const thumbnail =
      result.face_image || result.top_fake_frames?.[0]?.frame_image || result.gradcam_image || null;
    addEntry({
      mode,
      prediction: result.prediction,
      title:
        mode === "batch"
          ? `${result.summary.fake_count} fake / ${result.summary.real_count} real`
          : `${result.prediction} | ${(result.confidence * 100).toFixed(1)}% confidence`,
      timestamp: result.timestamp,
      thumbnail,
      result,
    });
  }

  async function handleAnalyze() {
    try {
      if (detection.mode === "image") {
        if (!primaryInput) {
          toast.error("Upload an image first.");
          return;
        }
        const result = await detection.analyzeImage(primaryInput, {
          threshold: settings.imageThreshold,
          showGradcam: settings.showGradcam,
          showLime: settings.showLime,
          enableMetadata: settings.enableMetadata,
        });
        saveHistory(result, "image");
        toast.success("Image analysis complete.");
        return;
      }

      if (detection.mode === "batch") {
        if (!batchFiles.length) {
          toast.error("Upload at least one image.");
          return;
        }
        const result = await detection.analyzeBatch(batchFiles, {
          threshold: settings.imageThreshold,
        });
        saveHistory(result, "batch");
        toast.success("Batch analysis complete.");
        return;
      }

      if (detection.mode === "video") {
        if (!primaryInput) {
          toast.error("Upload a video first.");
          return;
        }

        detection.beginVideoStream(primaryInput);
        await websocket.connect();
        websocket.sendMessage({
          type: "config",
          frame_interval: settings.frameInterval,
          fake_ratio_threshold: settings.videoThreshold,
          frame_fake_threshold: settings.imageThreshold,
        });

        const chunkSize = 64 * 1024;
        for (let offset = 0; offset < primaryInput.size; offset += chunkSize) {
          websocket.sendMessage(primaryInput.slice(offset, offset + chunkSize));
        }
        websocket.sendMessage({ type: "complete" });
        toast.success("Video upload streamed. Processing frames now.");
      }
    } catch (error) {
      toast.error(error.response?.data?.detail || error.message);
    }
  }

  async function handleCompareAnalyze() {
    if (!compareFiles.left || !compareFiles.right) {
      toast.error("Add both comparison images first.");
      return;
    }

    setCompareLoading(true);
    try {
      const [leftResult, rightResult] = await Promise.all([
        requestImageAnalysis(compareFiles.left, {
          threshold: settings.imageThreshold,
          showGradcam: true,
          showLime: true,
          enableMetadata: settings.enableMetadata,
        }),
        requestImageAnalysis(compareFiles.right, {
          threshold: settings.imageThreshold,
          showGradcam: true,
          showLime: true,
          enableMetadata: settings.enableMetadata,
        }),
      ]);
      setCompareResults({ left: leftResult, right: rightResult });
      toast.success("Comparison analysis complete.");
    } catch (error) {
      toast.error(error.response?.data?.detail || error.message);
    } finally {
      setCompareLoading(false);
    }
  }

  function restoreHistory(entry) {
    detection.setMode(entry.mode);
    detection.setResults(entry.result);
    if (entry.mode === "batch") {
      setBatchFiles([]);
    } else {
      setPrimaryInput(null);
    }
  }

  const result = detection.results;
  const videoFakeFrameIndices = result?.fake_frame_indices || [];
  const selectedFrameIsTrigger = selectedFrame
    ? videoFakeFrameIndices.includes(selectedFrame.frame)
    : false;

  return (
    <div className="min-h-screen bg-grid bg-grid px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto grid max-w-7xl gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
        <main className="space-y-6">
          <Header
            theme={theme}
            onToggleTheme={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
          />

          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="panel px-5 py-5"
          >
            <div className="flex flex-wrap gap-3">
              {modeOptions.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => {
                    detection.setMode(option.id);
                    detection.setResults(null);
                    detection.setFile(null);
                    setSelectedFrame(null);
                  }}
                  className={`rounded-full px-5 py-3 font-mono text-xs uppercase tracking-[0.2em] transition ${
                    detection.mode === option.id
                      ? "bg-forensic-accent text-white shadow-glow"
                      : "border border-forensic-border text-forensic-muted hover:border-forensic-accent hover:text-forensic-accent"
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            className="panel overflow-hidden"
          >
            <button
              type="button"
              onClick={() => setSettingsOpen((current) => !current)}
              className="flex w-full items-center justify-between px-5 py-5 text-left"
            >
              <div>
                <p className="panel-heading">Detection Settings</p>
                <p className="mt-2 text-sm text-forensic-muted">
                  Tune thresholds, explainability outputs, and metadata checks.
                </p>
              </div>
              <span className="data-chip">{settingsOpen ? "collapse" : "expand"}</span>
            </button>

            {settingsOpen ? (
              <div className="grid gap-4 border-t border-forensic-border px-5 py-5 md:grid-cols-2">
                <ThresholdSlider
                  label="Image fake threshold"
                  value={settings.imageThreshold}
                  onChange={(value) => updateSetting("imageThreshold", value)}
                  hint="Default 0.85. Lowering increases fake sensitivity and false positives."
                />
                <ThresholdSlider
                  label="Video fake ratio threshold"
                  value={settings.videoThreshold}
                  onChange={(value) => updateSetting("videoThreshold", value)}
                  hint="Default 0.40. Video is marked fake only if this fraction of analyzed frames is fake."
                />
                <ThresholdSlider
                  label="Frame interval"
                  value={settings.frameInterval}
                  min={1}
                  max={30}
                  step={1}
                  onChange={(value) => updateSetting("frameInterval", value)}
                  hint="How frequently video frames are sampled."
                />
                <div className="rounded-2xl border border-forensic-border bg-black/10 p-4">
                  <p className="font-mono text-xs uppercase tracking-[0.18em] text-forensic-muted">
                    Output toggles
                  </p>
                  <div className="mt-4 grid gap-3">
                    {[
                      ["showGradcam", "Show Grad-CAM"],
                      ["showLime", "Show LIME"],
                      ["showLandmarks", "Show landmarks"],
                      ["enableMetadata", "Enable metadata analysis"],
                    ].map(([key, label]) => (
                      <label
                        key={key}
                        className="flex items-center justify-between rounded-2xl border border-forensic-border px-4 py-3 text-sm text-forensic-text"
                      >
                        <span>{label}</span>
                        <input
                          type="checkbox"
                          checked={settings[key]}
                          onChange={(event) => updateSetting(key, event.target.checked)}
                          className="h-4 w-4 accent-[var(--accent-blue)]"
                        />
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            ) : null}
          </motion.section>

          {detection.mode !== "compare" ? (
            <>
              {detection.mode === "image" ? (
                <UploadZone
                  title="Image Upload"
                  description="Drop a single image for face detection, explainability, and metadata analysis."
                  accept={imageAccept}
                  files={primaryInput}
                  onFilesChange={setPrimaryInput}
                />
              ) : null}

              {detection.mode === "video" ? (
                <UploadZone
                  title="Video Upload"
                  description="Upload a video for frame sampling, fake-frame ratio analysis, and top suspicious frames."
                  accept={videoAccept}
                  files={primaryInput}
                  onFilesChange={setPrimaryInput}
                />
              ) : null}

              {detection.mode === "batch" ? (
                <UploadZone
                  title="Batch Upload"
                  description="Upload up to 10 images for a quick per-file summary."
                  accept={imageAccept}
                  multiple
                  files={batchFiles}
                  onFilesChange={setBatchFiles}
                />
              ) : null}

              <div className="flex flex-wrap gap-3">
                <button
                  type="button"
                  disabled={detection.loading}
                  onClick={handleAnalyze}
                  className="rounded-full bg-forensic-accent px-6 py-3 font-mono text-xs uppercase tracking-[0.2em] text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {detection.loading ? "Analyzing..." : "Analyze media"}
                </button>
                <button
                  type="button"
                  onClick={resetAll}
                  className="rounded-full border border-forensic-border px-6 py-3 font-mono text-xs uppercase tracking-[0.2em] text-forensic-text transition hover:border-forensic-accent hover:text-forensic-accent"
                >
                  Reset
                </button>
              </div>
            </>
          ) : null}

          {detection.mode === "compare" ? (
            <ComparisonMode
              leftFile={compareFiles.left}
              rightFile={compareFiles.right}
              leftResult={compareResults.left}
              rightResult={compareResults.right}
              onChangeLeft={(file) => setCompareFiles((current) => ({ ...current, left: file }))}
              onChangeRight={(file) => setCompareFiles((current) => ({ ...current, right: file }))}
              onAnalyze={handleCompareAnalyze}
              loading={compareLoading}
            />
          ) : null}

          {detection.loading && detection.mode !== "compare" ? (
            <ProgressStream progress={detection.progress} />
          ) : null}

          {detection.error ? (
            <div className="rounded-3xl border border-red-500/20 bg-red-500/10 px-5 py-4 text-sm text-red-300">
              {detection.error}
            </div>
          ) : null}

          {result && detection.mode !== "compare" ? (
            <div className="space-y-6">
              {result.mode === "batch" ? (
                <BatchResultsTable results={result.results} summary={result.summary} />
              ) : (
                <>
                  <ResultCard
                    prediction={result.prediction}
                    confidence={result.confidence}
                    processingTime={result.processing_time_ms}
                    mode={result.mode}
                  />
                  <ConfidenceBar confidence={result.confidence} />
                  {result.exif_flags ? <MetadataPanel flags={result.exif_flags} /> : null}
                  <GradCamPanel
                    faceImage={result.face_image}
                    gradcamImage={settings.showGradcam ? result.gradcam_image : null}
                  />
                  <LimePanel
                    limeImage={settings.showLime ? result.lime_image : null}
                    landmarkImage={settings.showLandmarks ? result.face_landmark_image : null}
                  />
                  <ExportButton result={result} />
                </>
              )}

              {result.mode === "video" ? (
                <>
                  <div className="panel p-5">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                      <div className="space-y-3">
                        <p className="panel-heading">Video Decision Trace</p>
                        <h3
                          className="font-mono text-2xl font-bold"
                          style={{
                            color:
                              result.prediction === "FAKE"
                                ? "var(--verdict-fake)"
                                : "var(--verdict-real)",
                          }}
                        >
                          Video marked {result.prediction}
                        </h3>
                        <p className="max-w-3xl text-sm text-forensic-muted">
                          {result.decision_reason ||
                            "Video verdict based on the analyzed frame-level outputs."}
                        </p>
                      </div>

                      <div className="flex flex-wrap gap-2">
                        <span className="data-chip text-forensic-fake">
                          {result.fake_frame_count || 0} trigger frames
                        </span>
                        <span className="data-chip">
                          fake ratio {(Number(result.fake_ratio || 0) * 100).toFixed(1)}%
                        </span>
                        <span className="data-chip">
                          ratio threshold {(Number(result.fake_ratio_threshold_used ?? settings.videoThreshold) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>

                    {videoFakeFrameIndices.length ? (
                      <div className="mt-5">
                        <p className="font-mono text-xs uppercase tracking-[0.18em] text-forensic-muted">
                          Triggered by frames
                        </p>
                        <div className="mt-3 flex flex-wrap gap-2">
                          {videoFakeFrameIndices.map((frameIndex) => (
                            <button
                              key={frameIndex}
                              type="button"
                              onClick={() =>
                                setSelectedFrame({
                                  frame: frameIndex,
                                  confidence:
                                    result.frame_confidences?.[
                                      result.frame_indices?.indexOf(frameIndex)
                                    ] ?? 0,
                                })
                              }
                              className="rounded-full border border-red-500/25 bg-red-500/10 px-3 py-2 font-mono text-xs text-forensic-fake transition hover:border-forensic-fake"
                            >
                              frame {frameIndex}
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>

                  <VideoTimeline
                    frameConfidences={result.frame_confidences || []}
                    frameIndices={result.frame_indices || []}
                    fakeFrameIndices={videoFakeFrameIndices}
                    threshold={settings.imageThreshold}
                    onSelectFrame={setSelectedFrame}
                  />
                  {selectedFrame ? (
                    <div
                      className="rounded-2xl border px-4 py-3 text-sm"
                      style={{
                        borderColor: selectedFrameIsTrigger
                          ? "rgba(239,68,68,0.25)"
                          : "var(--border)",
                        background: selectedFrameIsTrigger
                          ? "rgba(239,68,68,0.08)"
                          : "rgba(0,0,0,0.1)",
                        color: "var(--text-secondary)",
                      }}
                    >
                      Selected frame {selectedFrame.frame} | confidence{" "}
                      {(selectedFrame.confidence * 100).toFixed(2)}%
                      {selectedFrameIsTrigger ? " | trigger frame" : ""}
                    </div>
                  ) : null}
                  <FakeFramesGallery frames={result.top_fake_frames || []} />
                </>
              ) : null}
            </div>
          ) : null}
        </main>

        <div className="space-y-6">
          <SessionHistory
            history={history}
            onRestore={restoreHistory}
            onClear={clearHistory}
          />
        </div>
      </div>
    </div>
  );
}

export default App;

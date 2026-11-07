import { useState } from "react";
import axios from "axios";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function postForm(path, formData) {
  const response = await axios.post(`${apiBaseUrl}${path}`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
}

export async function requestImageAnalysis(file, options = {}) {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("threshold", options.threshold ?? 0.85);
  formData.append("return_gradcam", options.showGradcam ?? true);
  formData.append("return_lime", options.showLime ?? true);
  formData.append("enable_metadata", options.enableMetadata ?? true);
  return postForm("/predict-image", formData);
}

export async function requestBatchAnalysis(files, options = {}) {
  const formData = new FormData();
  files.forEach((file) => formData.append("images", file));
  formData.append("threshold", options.threshold ?? 0.85);
  return postForm("/predict-image-batch", formData);
}

export async function requestVideoAnalysis(file, options = {}) {
  const formData = new FormData();
  formData.append("video", file);
  formData.append("frame_interval", options.frameInterval ?? 10);
  formData.append("fake_ratio_threshold", options.fakeRatioThreshold ?? 0.4);
  formData.append("frame_fake_threshold", options.frameFakeThreshold ?? 0.85);
  return postForm("/predict-video", formData);
}

export function useDetection() {
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState("image");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(null);

  async function analyzeImage(nextFile, options = {}) {
    setLoading(true);
    setError(null);
    setProgress(null);
    try {
      const payload = await requestImageAnalysis(nextFile, options);
      const result = {
        ...payload,
        mode: "image",
        timestamp: new Date().toISOString(),
      };
      setResults(result);
      return result;
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message);
      throw requestError;
    } finally {
      setLoading(false);
    }
  }

  async function analyzeVideo(nextFile, options = {}) {
    setLoading(true);
    setError(null);
    setProgress(null);
    try {
      const payload = await requestVideoAnalysis(nextFile, options);
      const result = {
        ...payload,
        mode: "video",
        timestamp: new Date().toISOString(),
      };
      setResults(result);
      return result;
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message);
      throw requestError;
    } finally {
      setLoading(false);
    }
  }

  async function analyzeBatch(files, options = {}) {
    setLoading(true);
    setError(null);
    setProgress(null);
    try {
      const payload = await requestBatchAnalysis(files, options);
      const result = {
        ...payload,
        mode: "batch",
        timestamp: new Date().toISOString(),
      };
      setResults(result);
      return result;
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message);
      throw requestError;
    } finally {
      setLoading(false);
    }
  }

  function beginVideoStream(nextFile) {
    setFile(nextFile);
    setLoading(true);
    setError(null);
    setProgress({
      current: 0,
      total: 0,
      confidence: 0,
      frame_preview: null,
    });
  }

  function handleVideoMessage(message) {
    if (!message || typeof message !== "object") {
      return;
    }

    if (message.type === "progress") {
      setProgress(message);
      return;
    }

    if (message.type === "complete") {
      setResults({
        ...message,
        mode: "video",
        timestamp: new Date().toISOString(),
      });
      setProgress(null);
      setLoading(false);
      return;
    }

    if (message.type === "error") {
      setError(message.detail || "WebSocket analysis failed.");
      setLoading(false);
    }
  }

  function reset() {
    setFile(null);
    setResults(null);
    setError(null);
    setLoading(false);
    setProgress(null);
  }

  return {
    file,
    setFile,
    mode,
    setMode,
    results,
    setResults,
    loading,
    error,
    progress,
    analyzeImage,
    analyzeVideo,
    analyzeBatch,
    beginVideoStream,
    handleVideoMessage,
    reset,
  };
}

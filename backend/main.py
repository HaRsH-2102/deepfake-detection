from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from backend.explainability import run_explainability, run_gradcam
from backend.face_landmarks import draw_landmarks
from backend.forensics import extract_exif
from backend.predictor import (
    extract_frames,
    get_default_image_threshold,
    get_default_video_fake_ratio_threshold,
    get_model,
    is_lstm_available,
    initialize_runtime,
    predict_image,
    predict_video,
    preprocess_image,
    resolve_model_path,
    detect_face,
)
from backend.report_generator import generate_pdf_report
from backend.utils import rgb_to_base64_png, safe_unlink


load_dotenv(Path(__file__).with_name(".env"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepfake-backend")

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}

MODEL_STATUS: dict[str, Any] = {"model_loaded": False, "error": None, "model_path": None}


def _load_runtime() -> None:
    if os.getenv("SKIP_MODEL_LOAD") == "1":
        MODEL_STATUS.update({"model_loaded": False, "error": "Skipped by environment flag"})
        return

    try:
        runtime = initialize_runtime(
            model_path=os.getenv("MODEL_PATH"),
            device_name=os.getenv("DEVICE"),
        )
        MODEL_STATUS.update(
            {
                "model_loaded":          True,
                "error":                 None,
                "model_path":            runtime["model_path"],
                "lstm_model_path":       runtime.get("lstm_model_path"),
                "lstm_available":        runtime.get("lstm_available"),
                "device":                runtime["device"],
                "face_detector_backend": runtime.get("face_detector_backend"),
                "image_size":            runtime.get("image_size"),
                "normalize":             runtime.get("normalize"),
            }
        )
    except Exception as exc:  # pragma: no cover - depends on local ML runtime
        fallback_path = os.getenv("MODEL_PATH")
        if not fallback_path:
            try:
                fallback_path = resolve_model_path()
            except Exception:
                fallback_path = None
        MODEL_STATUS.update(
            {
                "model_loaded": False,
                "error": str(exc),
                "model_path": fallback_path,
            }
        )
        logger.exception("Failed to initialize runtime: %s", exc)


def _label_color(prediction: str) -> str:
    return {
        "FAKE": "#ef4444",
        "REAL": "#22c55e",
    }.get(prediction, "#f59e0b")


def _decode_image(upload: UploadFile) -> np.ndarray:
    raw = upload.file.read()
    file_bytes = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return image


def _ensure_runtime_loaded() -> None:
    if not MODEL_STATUS["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. {MODEL_STATUS.get('error') or 'Backend runtime unavailable.'}",
        )


def _save_upload_to_temp(upload: UploadFile, suffix: str) -> str:
    upload.file.seek(0)
    with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(upload.file.read())
        return handle.name


def _analyze_image_file(
    upload: UploadFile,
    threshold: float | None,
    return_gradcam: bool,
    return_lime: bool,
    enable_metadata: bool = True,
) -> dict[str, Any]:
    _ensure_runtime_loaded()
    started_at = time.perf_counter()
    decision_threshold = get_default_image_threshold() if threshold is None else threshold

    if upload.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported image file type.")

    upload.file.seek(0)
    image_bgr = _decode_image(upload)
    detection = detect_face(image_bgr)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = detection["face_rgb"] if detection else image_rgb

    prediction, confidence = predict_image(face_rgb, threshold=decision_threshold)
    gradcam_image = None
    lime_image = None
    if return_gradcam or return_lime:
        explainability = run_explainability(get_model(), face_rgb)
        gradcam_image = explainability["gradcam_image"] if return_gradcam else None
        lime_image = explainability["lime_image"] if return_lime else None

    upload.file.seek(0)
    temp_path = _save_upload_to_temp(upload, suffix=Path(upload.filename or "image.png").suffix or ".png")
    try:
        exif_flags = extract_exif(temp_path) if enable_metadata else []
    finally:
        safe_unlink(temp_path)

    processing_time_ms = int((time.perf_counter() - started_at) * 1000)

    return {
        "filename": upload.filename,
        "prediction": prediction,
        "confidence": confidence,
        "threshold_used": decision_threshold,
        "label_color": _label_color(prediction),
        "face_image": rgb_to_base64_png(face_rgb),
        "gradcam_image": gradcam_image,
        "lime_image": lime_image,
        "face_landmark_image": draw_landmarks(face_rgb, detection["landmarks"] if detection else None),
        "exif_flags": exif_flags,
        "processing_time_ms": processing_time_ms,
    }


async def _process_video_over_websocket(
    websocket: WebSocket,
    video_path: str,
    frame_interval: int,
    fake_ratio_threshold: float | None,
    frame_fake_threshold: float | None,
) -> dict[str, Any]:
    _ensure_runtime_loaded()
    sampled_frames = extract_frames(video_path, frame_interval)
    analyzed_frames: list[dict[str, Any]] = []
    decision_threshold = get_default_image_threshold() if frame_fake_threshold is None else frame_fake_threshold
    ratio_threshold = (
        get_default_video_fake_ratio_threshold()
        if fake_ratio_threshold is None
        else fake_ratio_threshold
    )

    for current, (frame_index, frame_bgr) in enumerate(sampled_frames, start=1):
        detection = detect_face(frame_bgr)
        if detection is None:
            continue

        label, confidence = predict_image(detection["face_rgb"], threshold=decision_threshold)
        analyzed_frames.append(
            {
                "frame_index": frame_index,
                "prediction": label,
                "confidence": confidence,
                "frame_rgb": detection["frame_rgb"],
                "face_rgb": detection["face_rgb"],
            }
        )
        await websocket.send_json(
            {
                "type": "progress",
                "frame": frame_index,
                "current": current,
                "total": len(sampled_frames),
                "confidence": confidence,
                "frame_preview": rgb_to_base64_png(detection["frame_rgb"]),
            }
        )

    if analyzed_frames:
        fake_frames = [item for item in analyzed_frames if item["prediction"] == "FAKE"]
        fake_count = len(fake_frames)
        fake_ratio = fake_count / len(analyzed_frames)
        average_confidence = float(
            sum(item["confidence"] for item in analyzed_frames) / len(analyzed_frames)
        )
        prediction = "FAKE" if fake_ratio >= ratio_threshold else "REAL"
        threshold_based_prediction = prediction
        fake_frame_indices = [int(item["frame_index"]) for item in fake_frames]
        if fake_frame_indices and prediction == "FAKE":
            decision_reason = (
                f"Marked FAKE because {fake_count}/{len(analyzed_frames)} analyzed frames "
                f"({fake_ratio:.2%}) crossed the per-frame fake threshold, meeting the video ratio threshold "
                f"of {ratio_threshold:.2%}. Trigger frames: "
                + ", ".join(str(index) for index in fake_frame_indices)
            )
        elif fake_frame_indices:
            decision_reason = (
                f"Marked REAL because only {fake_count}/{len(analyzed_frames)} analyzed frames "
                f"({fake_ratio:.2%}) crossed the per-frame fake threshold, which is below the video ratio "
                f"threshold of {ratio_threshold:.2%}."
            )
        else:
            decision_reason = "Marked REAL because no analyzed frame crossed the fake threshold."
        top_source_frames = sorted(
            fake_frames,
            key=lambda item: item["confidence"],
            reverse=True,
        )[:3]
    else:
        fake_ratio = 0.0
        average_confidence = 0.0
        prediction = "INCONCLUSIVE"
        threshold_based_prediction = "INCONCLUSIVE"
        fake_frame_indices = []
        decision_reason = "No frames with detectable faces were analyzed."
        top_source_frames = []

    top_frames = []
    for frame in top_source_frames:
        gradcam_image = run_gradcam(
            get_model(),
            preprocess_image(frame["face_rgb"]),
            original_image=frame["face_rgb"],
        )
        top_frames.append(
            {
                "frame_index": frame["frame_index"],
                "confidence": frame["confidence"],
                "frame_image": rgb_to_base64_png(frame["frame_rgb"]),
                "face_image": rgb_to_base64_png(frame["face_rgb"]),
                "gradcam_image": gradcam_image,
            }
        )

    payload = {
        "type": "complete",
        "prediction": prediction,
        "confidence": average_confidence,
        "fake_ratio": fake_ratio,
        "frame_fake_threshold": decision_threshold,
        "fake_ratio_threshold_used": ratio_threshold,
        "fake_frame_count": len(fake_frame_indices),
        "fake_frame_indices": fake_frame_indices,
        "decision_reason": decision_reason,
        "threshold_based_prediction": threshold_based_prediction,
        "total_frames_analyzed": len(analyzed_frames),
        "frame_confidences": [item["confidence"] for item in analyzed_frames],
        "frame_indices": [item["frame_index"] for item in analyzed_frames],
        "top_fake_frames": top_frames,
    }
    await websocket.send_json(payload)
    return payload


app = FastAPI(title="Deepfake Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request, call_next):  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info("%s %s -> %s (%sms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


@app.on_event("startup")
async def startup_event() -> None:
    _load_runtime()


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status":                         "ok",
        "model_loaded":                   MODEL_STATUS["model_loaded"],
        "model_path":                     MODEL_STATUS.get("model_path"),
        "lstm_model_path":                MODEL_STATUS.get("lstm_model_path"),
        "lstm_available":                 MODEL_STATUS.get("lstm_available"),
        "face_detector_backend":          MODEL_STATUS.get("face_detector_backend"),
        "image_size":                     MODEL_STATUS.get("image_size"),
        "normalize":                      MODEL_STATUS.get("normalize"),
        "default_image_threshold":        get_default_image_threshold(),
        "default_video_fake_ratio_threshold": get_default_video_fake_ratio_threshold(),
        "error":                          MODEL_STATUS.get("error"),
    }


@app.post("/predict-image")
async def predict_image_endpoint(
    image: UploadFile = File(...),
    threshold: float | None = Form(None),
    return_gradcam: bool = Form(True),
    return_lime: bool = Form(True),
    enable_metadata: bool = Form(True),
) -> JSONResponse:
    result = _analyze_image_file(image, threshold, return_gradcam, return_lime, enable_metadata)
    return JSONResponse(result)


@app.post("/predict-image-batch")
async def predict_image_batch_endpoint(
    images: list[UploadFile] = File(...),
    threshold: float | None = Form(None),
) -> JSONResponse:
    if len(images) > 10:
        raise HTTPException(status_code=422, detail="Batch mode supports up to 10 images.")

    results = [
        _analyze_image_file(image, threshold, return_gradcam=False, return_lime=False)
        for image in images
    ]
    avg_confidence = (
        float(sum(item["confidence"] for item in results) / len(results)) if results else 0.0
    )
    fake_count = sum(1 for item in results if item["prediction"] == "FAKE")
    summary = {
        "total": len(results),
        "fake_count": fake_count,
        "real_count": len(results) - fake_count,
        "avg_confidence": avg_confidence,
    }

    return JSONResponse(
        {
            "results": [
                {
                    "filename": item["filename"],
                    "prediction": item["prediction"],
                    "confidence": item["confidence"],
                    "label_color": item["label_color"],
                }
                for item in results
            ],
            "summary": summary,
        }
    )


@app.post("/predict-video")
async def predict_video_endpoint(
    video: UploadFile = File(...),
    frame_interval: int = Form(10),
    fake_ratio_threshold: float | None = Form(None),
    frame_fake_threshold: float | None = Form(None),
) -> JSONResponse:
    _ensure_runtime_loaded()
    if video.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported video file type.")

    temp_path = _save_upload_to_temp(video, suffix=Path(video.filename or "video.mp4").suffix or ".mp4")
    try:
        result = predict_video(
            temp_path,
            frame_interval=frame_interval,
            fake_ratio_threshold=fake_ratio_threshold,
            frame_fake_threshold=frame_fake_threshold,
        )
        top_frames = []
        for frame in result["top_fake_frames"]:
            top_frames.append(
                {
                    "frame_index": frame["frame_index"],
                    "confidence": frame["confidence"],
                    "frame_image": rgb_to_base64_png(frame["frame_rgb"]),
                    "face_image": rgb_to_base64_png(frame["face_rgb"]),
                    "gradcam_image": run_gradcam(
                        get_model(),
                        preprocess_image(frame["face_rgb"]),
                        original_image=frame["face_rgb"],
                    ),
                }
            )

        payload = {
            "prediction":                result["prediction"],
            "confidence":                result["confidence"],
            "fake_ratio":                result["fake_ratio"],
            "frame_fake_threshold":      result["frame_fake_threshold"],
            "fake_ratio_threshold_used": result["fake_ratio_threshold_used"],
            "fake_frame_count":          result["fake_frame_count"],
            "fake_frame_indices":        result["fake_frame_indices"],
            "decision_reason":           result["decision_reason"],
            "threshold_based_prediction": result["threshold_based_prediction"],
            "total_frames_analyzed":     result["total_frames_analyzed"],
            "frame_confidences":         result["frame_confidences"],
            "frame_indices":             result.get("analyzed_frame_indices", []),
            "label_color":               _label_color(result["prediction"]),
            "top_fake_frames":           top_frames,
            "model_used":                result.get("model_used", "unknown"),
        }
        return JSONResponse(payload)
    finally:
        safe_unlink(temp_path)


@app.post("/generate-report")
async def generate_report_endpoint(payload: dict[str, Any] = Body(...)) -> Response:
    pdf_bytes = generate_pdf_report(payload)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=deepfake-report.pdf"},
    )


@app.websocket("/ws/predict-video")
async def websocket_predict_video(websocket: WebSocket) -> None:
    await websocket.accept()
    frame_interval = 10
    fake_ratio_threshold: float | None = None
    frame_fake_threshold: float | None = None
    temp_path: str | None = None

    try:
        with NamedTemporaryFile(delete=False, suffix=".mp4") as handle:
            temp_path = handle.name
            while True:
                message = await websocket.receive()
                text = message.get("text")
                binary = message.get("bytes")

                if text is not None:
                    payload = json.loads(text)
                    message_type = payload.get("type")
                    if message_type == "config":
                        frame_interval = int(payload.get("frame_interval", frame_interval))
                        fake_ratio_threshold = float(
                            payload.get("fake_ratio_threshold", fake_ratio_threshold)
                        )
                        if payload.get("frame_fake_threshold") is not None:
                            frame_fake_threshold = float(payload.get("frame_fake_threshold"))
                    elif message_type == "complete":
                        break
                elif binary is not None:
                    handle.write(binary)

        if temp_path is None:
            raise RuntimeError("Failed to prepare temporary video path.")
        await _process_video_over_websocket(
            websocket,
            temp_path,
            frame_interval=frame_interval,
            fake_ratio_threshold=fake_ratio_threshold,
            frame_fake_threshold=frame_fake_threshold,
        )
    except WebSocketDisconnect:
        logger.info("Video websocket disconnected.")
    except Exception as exc:
        logger.exception("WebSocket video processing failed: %s", exc)
        await websocket.send_json({"type": "error", "detail": str(exc)})
    finally:
        safe_unlink(temp_path)

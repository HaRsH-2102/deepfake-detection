from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from backend.explainability import run_explainability
from backend.face_landmarks import draw_landmarks
from backend.forensics import extract_exif
from backend.predictor import (
    detect_face,
    get_model,
    initialize_runtime,
    predict_image,
    predict_video,
)
from backend.report_generator import generate_pdf_report
from backend.utils import base64_to_temp_image, rgb_to_base64_png, safe_unlink


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the backend pipeline end-to-end: model loading, "
            "face detection, prediction, Grad-CAM, LIME, EXIF forensics, "
            "landmarks, PDF generation, and optional video analysis."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a real face image for image-side tests.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Optional path to a video for video pipeline testing.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional model checkpoint path. Defaults to backend/.env or fallback search.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/smoke_test",
        help="Directory to save generated artifacts and the JSON summary.",
    )
    parser.add_argument(
        "--image-threshold",
        type=float,
        default=0.5,
        help="Threshold for image classification.",
    )
    parser.add_argument(
        "--video-threshold",
        type=float,
        default=0.3,
        help="Fake-ratio threshold for video classification.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=10,
        help="Frame sampling interval for video testing.",
    )
    return parser.parse_args()


def ensure_file(path: str | None, label: str) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def write_base64_image(encoded: str | None, output_path: Path) -> str | None:
    if not encoded:
        return None
    temp_path = base64_to_temp_image(encoded)
    try:
        output_path.write_bytes(Path(temp_path).read_bytes())
    finally:
        safe_unlink(temp_path)
    return str(output_path)


def build_report_payload(image_result: dict[str, Any], video_result: dict[str, Any] | None) -> dict[str, Any]:
    payload = {
        "prediction": image_result["prediction"],
        "confidence": image_result["confidence"],
        "mode": "image",
        "timestamp": datetime.now().isoformat(),
        "gradcam_image": image_result.get("gradcam_image"),
        "lime_image": image_result.get("lime_image"),
        "face_landmark_image": image_result.get("landmark_image"),
        "exif_flags": image_result.get("exif_flags", []),
        "processing_time_ms": image_result.get("processing_time_ms", 0),
    }

    if video_result is not None:
        payload.update(
            {
                "mode": "video",
                "frame_confidences": video_result.get("frame_confidences", []),
                "top_fake_frames": video_result.get("top_fake_frames", []),
            }
        )

    return payload


def main() -> int:
    args = parse_args()
    image_path = ensure_file(args.image, "Image")
    video_path = ensure_file(args.video, "Video")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/8] Initializing backend runtime...")
    runtime = initialize_runtime(model_path=args.model_path, device_name=args.device)
    print(f"Loaded model from: {runtime['model_path']}")
    print(f"Using device: {runtime['device']}")

    if image_path is None:
        raise RuntimeError("An input image is required for the smoke test.")

    print("\n[2/8] Loading image and running face detection...")
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    detection = detect_face(image_bgr)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = detection["face_rgb"] if detection else image_rgb
    print("Face detected." if detection else "No face detected; using full image.")

    print("\n[3/8] Running model inference...")
    prediction, confidence = predict_image(face_rgb, threshold=args.image_threshold)
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.6f}")

    print("\n[4/8] Running EXIF / metadata forensics...")
    exif_flags = extract_exif(str(image_path))
    print(f"EXIF flags found: {len(exif_flags)}")

    print("\n[5/8] Running Grad-CAM and LIME...")
    explainability = run_explainability(get_model(), face_rgb)
    print("Explainability outputs generated.")

    print("\n[6/8] Generating landmark overlay and saving artifacts...")
    artifact_paths: dict[str, str | None] = {}
    artifact_paths["face_image"] = str(output_dir / "face.png")
    cv2.imwrite(str(output_dir / "face.png"), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))

    landmark_image = draw_landmarks(face_rgb, detection["landmarks"] if detection else None)
    artifact_paths["landmark_image"] = write_base64_image(
        landmark_image,
        output_dir / "landmarks.png",
    )
    artifact_paths["gradcam_image"] = write_base64_image(
        explainability.get("gradcam_image"),
        output_dir / "gradcam.png",
    )
    artifact_paths["lime_image"] = write_base64_image(
        explainability.get("lime_image"),
        output_dir / "lime.png",
    )

    print("\n[7/8] Generating PDF report...")
    image_result = {
        "prediction": prediction,
        "confidence": confidence,
        "gradcam_image": explainability.get("gradcam_image"),
        "lime_image": explainability.get("lime_image"),
        "landmark_image": landmark_image,
        "exif_flags": exif_flags,
        "processing_time_ms": 0,
    }

    video_result: dict[str, Any] | None = None
    if video_path is not None:
        print("\n[8/8] Running video pipeline...")
        video_result = predict_video(
            str(video_path),
            frame_interval=args.frame_interval,
            fake_ratio_threshold=args.video_threshold,
        )
        print(f"Video prediction: {video_result['prediction']}")
        print(f"Video confidence: {video_result['confidence']:.6f}")
        print(f"Fake ratio: {video_result['fake_ratio']:.6f}")
    else:
        print("\n[8/8] Skipping video pipeline because --video was not provided.")

    report_payload = build_report_payload(image_result, video_result)
    report_bytes = generate_pdf_report(report_payload)
    report_path = output_dir / "smoke_test_report.pdf"
    report_path.write_bytes(report_bytes)

    if video_result is not None:
        top_frames_for_json: list[dict[str, Any]] = []
        for item in video_result.get("top_fake_frames", []):
            top_frames_for_json.append(
                {
                    "frame_index": item["frame_index"],
                    "confidence": item["confidence"],
                }
            )
        video_result = {
            **video_result,
            "top_fake_frames": top_frames_for_json,
        }

    summary = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "inputs": {
            "image": str(image_path),
            "video": str(video_path) if video_path is not None else None,
            "model_path": runtime["model_path"],
            "device": runtime["device"],
        },
        "image_test": {
            "prediction": prediction,
            "confidence": confidence,
            "face_detected": detection is not None,
            "exif_flags": exif_flags,
            "artifacts": artifact_paths,
        },
        "video_test": video_result,
        "report_path": str(report_path),
    }

    summary_path = output_dir / "smoke_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSmoke test completed successfully.")
    print(f"Artifacts saved in: {output_dir}")
    print(f"Summary JSON: {summary_path}")
    print(f"PDF report: {report_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nSmoke test failed: {exc}", file=sys.stderr)
        raise

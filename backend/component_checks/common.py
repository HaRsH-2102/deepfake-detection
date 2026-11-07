from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.predictor import detect_face, initialize_runtime
from backend.utils import base64_to_temp_image, safe_unlink


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--image", default=None, help="Path to an input image.")
    parser.add_argument("--video", default=None, help="Path to an input video.")
    parser.add_argument("--model-path", default=None, help="Optional model checkpoint path.")
    parser.add_argument("--device", default=None, help="Optional device override: cpu or cuda.")
    parser.add_argument(
        "--output-dir",
        default="outputs/component_checks",
        help="Base directory for saved verification artifacts.",
    )
    return parser


def resolve_existing_path(path: str | None, label: str) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def ensure_output_dir(base_output_dir: str | Path, component_name: str) -> Path:
    path = Path(base_output_dir).resolve() / component_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def initialize_backend(model_path: str | None, device: str | None) -> dict[str, str]:
    return initialize_runtime(model_path=model_path, device_name=device)


def load_image_rgb(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_bgr, image_rgb


def detect_face_or_fallback(image_bgr: np.ndarray, image_rgb: np.ndarray) -> dict[str, Any]:
    detection = detect_face(image_bgr)
    face_rgb = detection["face_rgb"] if detection else image_rgb
    return {
        "detection": detection,
        "face_rgb": face_rgb,
    }


def save_rgb_image(image_rgb: np.ndarray, output_path: Path) -> str:
    output_path.write_bytes(cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))[1].tobytes())
    return str(output_path)


def save_base64_png(encoded: str | None, output_path: Path) -> str | None:
    if not encoded:
        return None
    temp_path = base64_to_temp_image(encoded)
    try:
        output_path.write_bytes(Path(temp_path).read_bytes())
    finally:
        safe_unlink(temp_path)
    return str(output_path)


def write_summary(output_dir: Path, summary: dict[str, Any], filename: str = "result.json") -> Path:
    summary = {
        "timestamp": datetime.now().isoformat(),
        **summary,
    }
    path = output_dir / filename
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path

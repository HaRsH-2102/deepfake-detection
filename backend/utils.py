from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image


def ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    return array


def rgb_to_base64_png(image: np.ndarray) -> str:
    rgb = ensure_uint8_rgb(image)
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def bgr_to_base64_png(image: np.ndarray) -> str:
    rgb = cv2.cvtColor(ensure_uint8_rgb(image), cv2.COLOR_BGR2RGB)
    return rgb_to_base64_png(rgb)


def base64_to_temp_image(encoded: str, suffix: str = ".png") -> str:
    raw = base64.b64decode(encoded)
    with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(raw)
        return handle.name


def save_binary_to_temp(stream: BinaryIO, suffix: str) -> str:
    with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(stream.read())
        return handle.name


def safe_unlink(path: str | Path | None) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass

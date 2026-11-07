from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from backend.utils import rgb_to_base64_png


LANDMARK_COLORS = [
    (59, 130, 246),
    (34, 197, 94),
    (251, 191, 36),
    (244, 63, 94),
    (168, 85, 247),
]


def draw_landmarks(
    image_array: np.ndarray, mtcnn_result: Iterable[Iterable[float]] | None
) -> str | None:
    if mtcnn_result is None:
        return None

    canvas = np.asarray(image_array).copy()
    points = np.asarray(list(mtcnn_result), dtype=int)

    if points.ndim != 2 or points.shape[1] != 2:
        return None

    for index, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        color = LANDMARK_COLORS[index % len(LANDMARK_COLORS)]
        cv2.circle(canvas, (x, y), 4, color, thickness=-1)
        cv2.circle(canvas, (x, y), 8, (255, 255, 255), thickness=1)

    return rgb_to_base64_png(canvas)

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import piexif
import pytest
from PIL import Image


@pytest.fixture()
def sample_real_image(tmp_path: Path) -> Path:
    path = tmp_path / "real.jpg"
    image = np.full((128, 128, 3), 210, dtype=np.uint8)
    Image.fromarray(image).save(path)
    return path


@pytest.fixture()
def sample_fake_image(tmp_path: Path) -> Path:
    path = tmp_path / "fake.jpg"
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:, :, 0] = 240
    Image.fromarray(image).save(path)
    return path


@pytest.fixture()
def photoshop_image(tmp_path: Path) -> Path:
    path = tmp_path / "photoshop.jpg"
    image = Image.fromarray(np.full((128, 128, 3), 180, dtype=np.uint8))
    exif_dict = {"0th": {piexif.ImageIFD.Software: "Adobe Photoshop"}}
    image.save(path, exif=piexif.dump(exif_dict))
    return path


@pytest.fixture()
def sample_video(tmp_path: Path) -> Path:
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (128, 128),
    )
    for index in range(12):
        frame = np.full((128, 128, 3), 40 + index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path

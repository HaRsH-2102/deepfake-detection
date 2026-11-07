from __future__ import annotations

from pathlib import Path

import piexif
from PIL import Image


SUSPICIOUS_SOFTWARE = (
    "photoshop",
    "gimp",
    "stable diffusion",
    "comfyui",
    "midjourney",
    "canva",
    "generative fill",
)


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip("\x00").strip()
    return str(value)


def extract_exif(image_path: str) -> list[str]:
    path = Path(image_path)
    if not path.exists():
        return []

    flags: list[str] = []

    try:
        exif_dict = piexif.load(str(path))
    except Exception:
        return ["No EXIF metadata present"]

    zeroth = exif_dict.get("0th", {})
    exif = exif_dict.get("Exif", {})
    gps = exif_dict.get("GPS", {})

    software = _decode(zeroth.get(piexif.ImageIFD.Software))
    if software and any(keyword in software.lower() for keyword in SUSPICIOUS_SOFTWARE):
        flags.append(f"Software: {software}")

    create_date = _decode(exif.get(piexif.ExifIFD.DateTimeOriginal))
    modify_date = _decode(zeroth.get(piexif.ImageIFD.DateTime))
    if create_date and modify_date and create_date != modify_date:
        flags.append(f"Date mismatch: created {create_date}, modified {modify_date}")

    if not gps:
        flags.append("GPS metadata absent")

    if not exif.get(piexif.ExifIFD.MakerNote):
        flags.append("Maker notes missing")

    try:
        with Image.open(path) as image:
            if image.format not in {"JPEG", "TIFF"} and not flags:
                flags.append("Limited EXIF support for this format")
    except Exception:
        pass

    return flags

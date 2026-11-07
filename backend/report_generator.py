from __future__ import annotations

from datetime import datetime
from tempfile import NamedTemporaryFile
from typing import Iterable

from fpdf import FPDF

from backend.utils import base64_to_temp_image, safe_unlink


def _draw_multiline_section(pdf: FPDF, title: str, lines: Iterable[str]) -> None:
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(248, 250, 252)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_font("Helvetica", "", 11)
    for line in lines:
        pdf.multi_cell(0, 7, f"- {line}")
    pdf.ln(2)


def _apply_dark_theme(pdf: FPDF) -> None:
    pdf.set_fill_color(10, 10, 15)
    pdf.rect(0, 0, 210, 297, style="F")
    pdf.set_text_color(248, 250, 252)


def generate_pdf_report(result_dict: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    temp_paths: list[str] = []

    prediction = result_dict.get("prediction", "INCONCLUSIVE")
    confidence = float(result_dict.get("confidence", 0.0))
    timestamp = result_dict.get("timestamp") or datetime.utcnow().isoformat()
    verdict_color = (239, 68, 68) if prediction == "FAKE" else (34, 197, 94)

    try:
        pdf.add_page()
        _apply_dark_theme(pdf)
        pdf.set_font("Helvetica", "B", 24)
        pdf.cell(0, 14, "DEEPFAKE DETECTOR REPORT", ln=True)
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 8, f"Generated: {timestamp}", ln=True)
        pdf.ln(6)

        pdf.set_font("Helvetica", "B", 28)
        pdf.set_text_color(*verdict_color)
        pdf.cell(0, 16, prediction, ln=True)
        pdf.set_font("Helvetica", "", 14)
        pdf.set_text_color(248, 250, 252)
        pdf.cell(0, 10, f"Confidence: {confidence:.2%}", ln=True)
        pdf.cell(0, 10, f"Mode: {result_dict.get('mode', 'image')}", ln=True)
        pdf.ln(6)

        exif_flags = result_dict.get("exif_flags") or ["No suspicious metadata detected"]
        _draw_multiline_section(pdf, "Metadata Signals", exif_flags)

        if result_dict.get("processing_time_ms") is not None:
            _draw_multiline_section(
                pdf,
                "Runtime",
                [f"Processing time: {result_dict['processing_time_ms']} ms"],
            )

        visual_paths: list[tuple[str, str]] = []
        if result_dict.get("gradcam_image"):
            visual_paths.append(("Grad-CAM", result_dict["gradcam_image"]))
        if result_dict.get("lime_image"):
            visual_paths.append(("LIME", result_dict["lime_image"]))
        if result_dict.get("face_landmark_image"):
            visual_paths.append(("Face Landmarks", result_dict["face_landmark_image"]))

        if visual_paths:
            pdf.add_page()
            _apply_dark_theme(pdf)
            pdf.set_font("Helvetica", "B", 20)
            pdf.cell(0, 14, "Explainability Visuals", ln=True)
            y_position = 30
            for title, encoded_image in visual_paths:
                image_path = base64_to_temp_image(encoded_image)
                temp_paths.append(image_path)
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 10, title, ln=True)
                pdf.image(image_path, x=15, y=y_position + 8, w=82)
                y_position += 92
                pdf.set_y(y_position)

        if result_dict.get("frame_confidences"):
            pdf.add_page()
            _apply_dark_theme(pdf)
            pdf.set_font("Helvetica", "B", 20)
            pdf.cell(0, 14, "Video Summary", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(
                0,
                7,
                "Frame confidences: "
                + ", ".join(f"{float(item):.2f}" for item in result_dict["frame_confidences"][:50]),
            )
    finally:
        output = pdf.output(dest="S")
        for path in temp_paths:
            safe_unlink(path)

    if isinstance(output, bytearray):
        return bytes(output)
    if isinstance(output, str):
        return output.encode("latin-1")
    return bytes(output)

from __future__ import annotations

from backend.component_checks.common import (
    base_parser,
    detect_face_or_fallback,
    ensure_output_dir,
    initialize_backend,
    load_image_rgb,
    resolve_existing_path,
    save_base64_png,
    save_rgb_image,
    write_summary,
)
from backend.explainability import run_explainability
from backend.face_landmarks import draw_landmarks
from backend.forensics import extract_exif
from backend.predictor import get_model, predict_image
from backend.report_generator import generate_pdf_report


def main() -> int:
    parser = base_parser("Verify backend PDF report generation.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Image prediction threshold.")
    args = parser.parse_args()

    image_path = resolve_existing_path(args.image, "Image")
    if image_path is None:
        raise ValueError("--image is required for report generation.")

    output_dir = ensure_output_dir(args.output_dir, "report_generator")
    runtime = initialize_backend(args.model_path, args.device)
    image_bgr, image_rgb = load_image_rgb(image_path)
    result = detect_face_or_fallback(image_bgr, image_rgb)
    detection = result["detection"]
    face_rgb = result["face_rgb"]

    prediction, confidence = predict_image(face_rgb, threshold=args.threshold)
    explainability = run_explainability(get_model(), face_rgb)
    landmark_image = draw_landmarks(face_rgb, detection["landmarks"] if detection else None)
    exif_flags = extract_exif(str(image_path))

    report_payload = {
        "prediction": prediction,
        "confidence": confidence,
        "mode": "image",
        "gradcam_image": explainability.get("gradcam_image"),
        "lime_image": explainability.get("lime_image"),
        "face_landmark_image": landmark_image,
        "exif_flags": exif_flags,
    }

    pdf_bytes = generate_pdf_report(report_payload)
    pdf_path = output_dir / "component_report.pdf"
    pdf_path.write_bytes(pdf_bytes)

    face_path = save_rgb_image(face_rgb, output_dir / "input_face.png")
    gradcam_path = save_base64_png(explainability.get("gradcam_image"), output_dir / "gradcam.png")
    lime_path = save_base64_png(explainability.get("lime_image"), output_dir / "lime.png")
    landmark_path = save_base64_png(landmark_image, output_dir / "landmarks.png")

    summary = {
        "status": "ok",
        "model_path": runtime["model_path"],
        "device": runtime["device"],
        "image_path": str(image_path),
        "prediction": prediction,
        "confidence": confidence,
        "outputs": {
            "input_face_image": face_path,
            "gradcam_image": gradcam_path,
            "lime_image": lime_path,
            "landmarks_image": landmark_path,
            "pdf_report": str(pdf_path),
        },
    }
    summary_path = write_summary(output_dir, summary)
    print(f"Report generator check passed. Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

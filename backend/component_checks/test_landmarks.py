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
from backend.face_landmarks import draw_landmarks


def main() -> int:
    parser = base_parser("Verify facial landmark overlay generation.")
    args = parser.parse_args()

    image_path = resolve_existing_path(args.image, "Image")
    if image_path is None:
        raise ValueError("--image is required for landmark testing.")

    output_dir = ensure_output_dir(args.output_dir, "landmarks")
    runtime = initialize_backend(args.model_path, args.device)
    image_bgr, image_rgb = load_image_rgb(image_path)
    result = detect_face_or_fallback(image_bgr, image_rgb)
    detection = result["detection"]
    face_rgb = result["face_rgb"]

    overlay = draw_landmarks(face_rgb, detection["landmarks"] if detection else None)
    face_path = save_rgb_image(face_rgb, output_dir / "input_face.png")
    landmark_path = save_base64_png(overlay, output_dir / "landmarks.png")

    summary = {
        "status": "ok",
        "model_path": runtime["model_path"],
        "device": runtime["device"],
        "image_path": str(image_path),
        "face_detected": detection is not None,
        "landmarks_present": bool(detection and detection.get("landmarks")),
        "landmark_count": len(detection["landmarks"]) if detection and detection.get("landmarks") else 0,
        "outputs": {
            "input_face_image": face_path,
            "landmark_overlay_image": landmark_path,
        },
    }
    summary_path = write_summary(output_dir, summary)
    print(f"Landmark check passed. Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

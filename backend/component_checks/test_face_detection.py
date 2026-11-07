from __future__ import annotations

import cv2

from backend.component_checks.common import (
    base_parser,
    detect_face_or_fallback,
    ensure_output_dir,
    initialize_backend,
    load_image_rgb,
    resolve_existing_path,
    save_rgb_image,
    write_summary,
)


def main() -> int:
    parser = base_parser("Verify face detection and save the crop plus annotated frame.")
    args = parser.parse_args()
    image_path = resolve_existing_path(args.image, "Image")
    if image_path is None:
        raise ValueError("--image is required for face detection.")

    output_dir = ensure_output_dir(args.output_dir, "face_detection")
    runtime = initialize_backend(args.model_path, args.device)
    image_bgr, image_rgb = load_image_rgb(image_path)
    result = detect_face_or_fallback(image_bgr, image_rgb)
    detection = result["detection"]
    face_rgb = result["face_rgb"]

    annotated_bgr = image_bgr.copy()
    if detection:
        x1, y1, x2, y2 = detection["box"]
        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (59, 130, 246), 2)

    original_path = save_rgb_image(image_rgb, output_dir / "original.png")
    face_path = save_rgb_image(face_rgb, output_dir / "face_crop.png")
    annotated_path = str(output_dir / "annotated.png")
    cv2.imwrite(annotated_path, annotated_bgr)

    summary = {
        "status": "ok",
        "model_path": runtime["model_path"],
        "device": runtime["device"],
        "image_path": str(image_path),
        "face_detected": detection is not None,
        "box": detection["box"] if detection else None,
        "detector_probability": detection["probability"] if detection else None,
        "outputs": {
            "original_image": original_path,
            "annotated_image": annotated_path,
            "face_crop_image": face_path,
        },
    }
    summary_path = write_summary(output_dir, summary)
    print(f"Face detection check passed. Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

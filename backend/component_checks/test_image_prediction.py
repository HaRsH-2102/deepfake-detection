from __future__ import annotations

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
from backend.predictor import predict_image


def main() -> int:
    parser = base_parser("Verify image prediction output and save the analyzed face.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Image prediction threshold.")
    args = parser.parse_args()

    image_path = resolve_existing_path(args.image, "Image")
    if image_path is None:
        raise ValueError("--image is required for image prediction.")

    output_dir = ensure_output_dir(args.output_dir, "image_prediction")
    runtime = initialize_backend(args.model_path, args.device)
    image_bgr, image_rgb = load_image_rgb(image_path)
    result = detect_face_or_fallback(image_bgr, image_rgb)
    face_rgb = result["face_rgb"]

    prediction, confidence = predict_image(face_rgb, threshold=args.threshold)
    face_path = save_rgb_image(face_rgb, output_dir / "analyzed_face.png")

    summary = {
        "status": "ok",
        "model_path": runtime["model_path"],
        "device": runtime["device"],
        "image_path": str(image_path),
        "threshold": args.threshold,
        "prediction": prediction,
        "confidence": confidence,
        "face_detected": result["detection"] is not None,
        "outputs": {
            "analyzed_face_image": face_path,
        },
    }
    summary_path = write_summary(output_dir, summary)
    print(f"Image prediction check passed. Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

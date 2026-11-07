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
from backend.predictor import get_model


def main() -> int:
    parser = base_parser("Verify Grad-CAM and LIME generation.")
    args = parser.parse_args()

    image_path = resolve_existing_path(args.image, "Image")
    if image_path is None:
        raise ValueError("--image is required for explainability.")

    output_dir = ensure_output_dir(args.output_dir, "explainability")
    runtime = initialize_backend(args.model_path, args.device)
    image_bgr, image_rgb = load_image_rgb(image_path)
    result = detect_face_or_fallback(image_bgr, image_rgb)
    face_rgb = result["face_rgb"]

    explainability = run_explainability(get_model(), face_rgb)
    face_path = save_rgb_image(face_rgb, output_dir / "input_face.png")
    gradcam_path = save_base64_png(explainability.get("gradcam_image"), output_dir / "gradcam.png")
    lime_path = save_base64_png(explainability.get("lime_image"), output_dir / "lime.png")

    summary = {
        "status": "ok",
        "model_path": runtime["model_path"],
        "device": runtime["device"],
        "image_path": str(image_path),
        "face_detected": result["detection"] is not None,
        "outputs": {
            "input_face_image": face_path,
            "gradcam_image": gradcam_path,
            "lime_image": lime_path,
        },
    }
    summary_path = write_summary(output_dir, summary)
    print(f"Explainability check passed. Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

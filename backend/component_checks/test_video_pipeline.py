from __future__ import annotations

from backend.component_checks.common import (
    base_parser,
    ensure_output_dir,
    initialize_backend,
    resolve_existing_path,
    save_base64_png,
    write_summary,
)
from backend.explainability import run_gradcam
from backend.predictor import get_model, predict_video, preprocess_image
from backend.utils import rgb_to_base64_png


def main() -> int:
    parser = base_parser("Verify the video prediction pipeline and save top suspicious frames.")
    parser.add_argument("--frame-interval", type=int, default=10, help="Frame interval for video sampling.")
    parser.add_argument(
        "--fake-ratio-threshold",
        type=float,
        default=None,
        help="Fake-ratio threshold for final video classification. Defaults to backend DEFAULT_VIDEO_FAKE_RATIO_THRESHOLD.",
    )
    parser.add_argument(
        "--frame-fake-threshold",
        type=float,
        default=None,
        help="Per-frame fake threshold. Defaults to backend DEFAULT_IMAGE_THRESHOLD.",
    )
    args = parser.parse_args()

    video_path = resolve_existing_path(args.video, "Video")
    if video_path is None:
        raise ValueError("--video is required for video pipeline testing.")

    output_dir = ensure_output_dir(args.output_dir, "video_pipeline")
    runtime = initialize_backend(args.model_path, args.device)
    result = predict_video(
        str(video_path),
        frame_interval=args.frame_interval,
        fake_ratio_threshold=args.fake_ratio_threshold,
        frame_fake_threshold=args.frame_fake_threshold,
    )

    saved_top_frames: list[dict[str, object]] = []
    for index, frame in enumerate(result.get("top_fake_frames", []), start=1):
        frame_image_path = save_base64_png(
            rgb_to_base64_png(frame["frame_rgb"]),
            output_dir / f"top_frame_{index}.png",
        )
        face_image_path = save_base64_png(
            rgb_to_base64_png(frame["face_rgb"]),
            output_dir / f"top_face_{index}.png",
        )
        gradcam_image_path = save_base64_png(
            run_gradcam(
                get_model(),
                preprocess_image(frame["face_rgb"]),
                original_image=frame["face_rgb"],
            ),
            output_dir / f"top_gradcam_{index}.png",
        )
        saved_top_frames.append(
            {
                "frame_index": frame["frame_index"],
                "confidence": frame["confidence"],
                "frame_image": frame_image_path,
                "face_image": face_image_path,
                "gradcam_image": gradcam_image_path,
            }
        )

    summary = {
        "status": "ok",
        "model_path": runtime["model_path"],
        "device": runtime["device"],
        "face_detector_backend": runtime.get("face_detector_backend"),
        "video_path": str(video_path),
        "frame_interval": args.frame_interval,
        "fake_ratio_threshold": args.fake_ratio_threshold,
        "frame_fake_threshold": result["frame_fake_threshold"],
        "fake_ratio_threshold_used": result["fake_ratio_threshold_used"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "fake_ratio": result["fake_ratio"],
        "fake_frame_count": result["fake_frame_count"],
        "fake_frame_indices": result["fake_frame_indices"],
        "decision_reason": result["decision_reason"],
        "threshold_based_prediction": result["threshold_based_prediction"],
        "total_frames_analyzed": result["total_frames_analyzed"],
        "frame_confidences": result["frame_confidences"],
        "top_fake_frames": saved_top_frames,
    }
    summary_path = write_summary(output_dir, summary)
    print(f"Video pipeline check passed. Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

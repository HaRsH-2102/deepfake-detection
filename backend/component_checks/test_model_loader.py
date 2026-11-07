from __future__ import annotations

from backend.predictor import get_model
from backend.component_checks.common import (
    base_parser,
    ensure_output_dir,
    initialize_backend,
    write_summary,
)


def main() -> int:
    parser = base_parser("Verify model loading and save model metadata.")
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir, "model_loader")
    runtime = initialize_backend(args.model_path, args.device)
    model = get_model()

    param_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    summary = {
        "status": "ok",
        "model_path": runtime["model_path"],
        "device": runtime["device"],
        "face_detector_backend": runtime.get("face_detector_backend"),
        "model_class": model.__class__.__name__,
        "parameter_count": int(param_count),
        "trainable_parameter_count": int(trainable_count),
        "top_level_modules": list(model._modules.keys()),
    }
    summary_path = write_summary(output_dir, summary)

    (output_dir / "model_repr.txt").write_text(str(model), encoding="utf-8")
    print(f"Model loader check passed. Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

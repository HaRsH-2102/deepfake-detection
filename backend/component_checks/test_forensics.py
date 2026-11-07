from __future__ import annotations

from backend.component_checks.common import (
    base_parser,
    ensure_output_dir,
    initialize_backend,
    resolve_existing_path,
    write_summary,
)
from backend.forensics import extract_exif


def main() -> int:
    parser = base_parser("Verify EXIF and metadata forensics output.")
    args = parser.parse_args()

    image_path = resolve_existing_path(args.image, "Image")
    if image_path is None:
        raise ValueError("--image is required for metadata forensics.")

    output_dir = ensure_output_dir(args.output_dir, "forensics")
    runtime = initialize_backend(args.model_path, args.device)
    exif_flags = extract_exif(str(image_path))

    summary = {
        "status": "ok",
        "model_path": runtime["model_path"],
        "device": runtime["device"],
        "image_path": str(image_path),
        "flag_count": len(exif_flags),
        "flags": exif_flags,
    }
    summary_path = write_summary(output_dir, summary)
    print(f"Forensics check passed. Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

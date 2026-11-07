from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    ("model_loader", "backend.component_checks.test_model_loader"),
    ("face_detection", "backend.component_checks.test_face_detection"),
    ("image_prediction", "backend.component_checks.test_image_prediction"),
    ("explainability", "backend.component_checks.test_explainability"),
    ("forensics", "backend.component_checks.test_forensics"),
    ("landmarks", "backend.component_checks.test_landmarks"),
    ("report_generator", "backend.component_checks.test_report_generator"),
    ("video_pipeline", "backend.component_checks.test_video_pipeline"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all component checks in sequence.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--video", default=None, help="Optional path to an input video.")
    parser.add_argument("--model-path", default=None, help="Optional model checkpoint path.")
    parser.add_argument("--device", default=None, help="Optional device override.")
    parser.add_argument(
        "--output-dir",
        default="outputs/component_checks",
        help="Base output directory for component results.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    failures: list[str] = []

    for label, module_name in SCRIPTS:
        if label == "video_pipeline" and not args.video:
            print(f"Skipping {label}: no --video provided.")
            continue

        command = [
            sys.executable,
            "-m",
            module_name,
            "--output-dir",
            args.output_dir,
        ]
        if args.image:
            command.extend(["--image", args.image])
        if args.video:
            command.extend(["--video", args.video])
        if args.model_path:
            command.extend(["--model-path", args.model_path])
        if args.device:
            command.extend(["--device", args.device])

        print(f"\nRunning {label}...")
        completed = subprocess.run(command, cwd=repo_root, check=False)
        if completed.returncode != 0:
            failures.append(label)

    if failures:
        print(f"\nComponent checks failed: {', '.join(failures)}")
        return 1

    print("\nAll requested component checks completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

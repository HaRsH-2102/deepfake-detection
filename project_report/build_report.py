from __future__ import annotations

import argparse
import html
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "own_train" / "runs" / "20260422_231937_xception"
DEFAULT_COMPONENT_DIR = REPO_ROOT / "outputs" / "component_checks"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "project_report" / "generated"


@dataclass
class ReportContext:
    run_dir: Path
    component_dir: Path
    output_dir: Path
    assets_dir: Path
    artifacts_dir: Path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def fmt_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def safe_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def copy_asset(ctx: ReportContext, src: Path, dest_name: str) -> str:
    dest = ctx.assets_dir / dest_name
    safe_copy(src, dest)
    return f"assets/{dest_name}"


def copy_artifact(ctx: ReportContext, src: Path, dest_name: str) -> str:
    dest = ctx.artifacts_dir / dest_name
    safe_copy(src, dest)
    return f"artifacts/{dest_name}"


def build_asset_map(ctx: ReportContext) -> dict[str, str]:
    plots_dir = ctx.run_dir / "plots"
    assets: dict[str, str] = {}

    figure_files = {
        "learning_curves": plots_dir / "learning_curves.png",
        "val_confusion_matrix": plots_dir / "val_confusion_matrix.png",
        "test_confusion_matrix": plots_dir / "test_confusion_matrix.png",
        "val_roc_curve": plots_dir / "val_roc_curve.png",
        "test_roc_curve": plots_dir / "test_roc_curve.png",
        "val_error_gallery": plots_dir / "val_error_gallery.png",
        "test_error_gallery": plots_dir / "test_error_gallery.png",
        "face_original": ctx.component_dir / "face_detection" / "original.png",
        "face_annotated": ctx.component_dir / "face_detection" / "annotated.png",
        "face_crop": ctx.component_dir / "face_detection" / "face_crop.png",
        "image_prediction_face": ctx.component_dir / "image_prediction" / "analyzed_face.png",
        "explainability_input_face": ctx.component_dir / "explainability" / "input_face.png",
        "gradcam": ctx.component_dir / "explainability" / "gradcam.png",
        "lime": ctx.component_dir / "explainability" / "lime.png",
        "landmarks_image": ctx.component_dir / "landmarks" / "landmarks.png",
        "video_top_frame_1": ctx.component_dir / "video_pipeline" / "top_frame_1.png",
        "video_top_face_1": ctx.component_dir / "video_pipeline" / "top_face_1.png",
        "video_top_gradcam_1": ctx.component_dir / "video_pipeline" / "top_gradcam_1.png",
        "video_top_frame_2": ctx.component_dir / "video_pipeline" / "top_frame_2.png",
        "video_top_face_2": ctx.component_dir / "video_pipeline" / "top_face_2.png",
        "video_top_gradcam_2": ctx.component_dir / "video_pipeline" / "top_gradcam_2.png",
        "video_top_frame_3": ctx.component_dir / "video_pipeline" / "top_frame_3.png",
        "video_top_face_3": ctx.component_dir / "video_pipeline" / "top_face_3.png",
        "video_top_gradcam_3": ctx.component_dir / "video_pipeline" / "top_gradcam_3.png",
    }

    for key, src in figure_files.items():
        if src.exists():
            assets[key] = copy_asset(ctx, src, src.name if key.startswith("video_") else f"{key}{src.suffix}")

    artifact_files = {
        "final_report": ctx.run_dir / "final_report.json",
        "config": ctx.run_dir / "config.json",
        "split_summary": ctx.run_dir / "split_summary.json",
        "model_loader": ctx.component_dir / "model_loader" / "result.json",
        "face_detection": ctx.component_dir / "face_detection" / "result.json",
        "image_prediction": ctx.component_dir / "image_prediction" / "result.json",
        "explainability": ctx.component_dir / "explainability" / "result.json",
        "forensics": ctx.component_dir / "forensics" / "result.json",
        "landmarks_result": ctx.component_dir / "landmarks" / "result.json",
        "video_pipeline": ctx.component_dir / "video_pipeline" / "result.json",
    }

    for key, src in artifact_files.items():
        if src.exists():
            assets[key] = copy_artifact(ctx, src, f"{key}.json")

    return assets


def figure_markdown(title: str, path: str | None, caption: str) -> str:
    if not path:
        return ""
    return f"### {title}\n\n![{title}]({path})\n\n*{caption}*\n"


def figure_html(title: str, path: str | None, caption: str) -> str:
    if not path:
        return ""
    return (
        '<figure class="figure-card">'
        f'<img src="{html.escape(path)}" alt="{html.escape(title)}" />'
        f"<figcaption><strong>{html.escape(title)}</strong><br>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def render_markdown(
    ctx: ReportContext,
    assets: dict[str, str],
    config: dict[str, Any],
    final_report: dict[str, Any],
    split_summary: dict[str, Any],
    model_loader: dict[str, Any],
    face_detection: dict[str, Any],
    image_prediction: dict[str, Any],
    explainability: dict[str, Any],
    forensics: dict[str, Any],
    landmarks: dict[str, Any],
    video_pipeline: dict[str, Any],
) -> str:
    val = final_report["final_val_metrics"]
    test = final_report["final_test_metrics"]
    split = final_report["split_summary"]

    fake_methods = split_summary["fake_method_counts"]["train"]
    method_rows = "\n".join(
        f"| {method} | {count:,} |"
        for method, count in fake_methods.items()
    )

    trigger_frames = ", ".join(str(item) for item in video_pipeline.get("fake_frame_indices", [])[:20])
    if len(video_pipeline.get("fake_frame_indices", [])) > 20:
        trigger_frames += ", ..."

    return f"""# Deepfake Detection System Report

**Project:** Fine-Tuned XceptionNet Deepfake Detection Platform  
**Repository:** `deepfake-detection-XceptionNet-main`  
**Generated:** {datetime.now().strftime("%B %d, %Y %H:%M")}  
**Primary checkpoint:** `{final_report['split_summary']['split_mode']}` split, best model at `own_train/runs/20260422_231937_xception/checkpoints/best_model.pt`

## Abstract

This report documents the end-to-end development of a deepfake detection system built around a fine-tuned XceptionNet classifier, an MTCNN-based face extraction pipeline, explainability modules, and a production-oriented FastAPI plus React deployment stack. The model was re-trained on the local `Dataset` directory using an identity-grouped split to reduce subject leakage, then integrated into the backend with the same preprocessing used during training. The resulting system combines frame-level image analysis, ratio-based video analysis, metadata inspection, face landmark visualization, Grad-CAM, LIME, PDF reporting, and an interactive forensic dashboard.

The final fine-tuned model achieved **{fmt_pct(test['accuracy'])} test accuracy**, **{fmt_pct(test['f1'])} test F1-score**, and **{fmt_float(test['auc'])} test AUC**, while retaining strong validation performance. A representative fake-image verification sample was classified as **{image_prediction['prediction']}** with **{fmt_pct(image_prediction['confidence'])} confidence**, and the representative fake-video verification run was classified as **{video_pipeline['prediction']}** with **{fmt_pct(video_pipeline['fake_ratio'])} fake-frame ratio**.

## Objectives

1. Fine-tune a stronger XceptionNet checkpoint on the provided local dataset.
2. Keep the backend preprocessing aligned with the training pipeline.
3. Build an explainable detection workflow for both images and videos.
4. Provide a professional frontend suitable for demonstrations and forensic interpretation.
5. Generate reproducible results, plots, and assets suitable for publication-style reporting.

## System Overview

The final system consists of four tightly connected layers:

1. **Training layer:** `own_train/` contains the fine-tuning pipeline, grouped splitting logic, metric tracking, checkpointing, and plotting.
2. **Inference layer:** `backend/predictor.py` loads the fine-tuned Xception model, applies the training-consistent preprocessing (`299x299` with ImageNet normalization), detects faces via `from mtcnn import MTCNN`, and produces image/video predictions.
3. **Forensic analysis layer:** `backend/explainability.py`, `backend/forensics.py`, and `backend/face_landmarks.py` generate Grad-CAM, LIME, metadata flags, and landmark overlays.
4. **Application layer:** FastAPI routes in `backend/main.py` expose image, batch, video, WebSocket, and PDF-report endpoints, while the React/Vite frontend presents uploads, thresholds, timelines, trigger frames, comparison mode, and session history.

## Dataset and Split Strategy

The training process used the local `Dataset` folder and applied an **identity-grouped split** to reduce train/validation/test leakage.

### Split Summary

| Split | Identities | Samples | Real Frames | Fake Frames |
| --- | ---: | ---: | ---: | ---: |
| Train | {split['identity_split_counts']['train']:,} | {split['train_sample_count']:,} | {split['real_frame_counts']['train']:,} | {split['fake_frame_counts']['train']:,} |
| Validation | {split['identity_split_counts']['val']:,} | {split['val_sample_count']:,} | {split['real_frame_counts']['val']:,} | {split['fake_frame_counts']['val']:,} |
| Test | {split['identity_split_counts']['test']:,} | {split['test_sample_count']:,} | {split['real_frame_counts']['test']:,} | {split['fake_frame_counts']['test']:,} |

### Training Fake-Method Distribution

| Manipulation Method | Training Frames |
| --- | ---: |
{method_rows}

Additional leakage-control details:

- Cross-split fake frames dropped: **{split['dropped_cross_split_fake_frames']:,}**
- Cross-split fake groups dropped: **{split['dropped_cross_split_fake_groups']:,}**
- Retained fake groups: **{split['retained_fake_groups']:,}**

## Training Configuration

| Parameter | Value |
| --- | --- |
| Backbone | {config['model_name']} |
| Pretrained initialization | {config['pretrained']} |
| Image size | {config['image_size']} |
| Batch size | {config['batch_size']} |
| Epochs | {config['epochs']} |
| Frozen epochs | {config['frozen_epochs']} |
| Learning rate | {config['learning_rate']} |
| Weight decay | {config['weight_decay']} |
| Patience | {config['patience']} |
| Weighted sampler | {config['use_weighted_sampler']} |
| Mixed precision | {config['amp']} |
| Decision threshold during training evaluation | {config['decision_threshold']} |
| Best epoch | {final_report['best_epoch']} |
| Training time | {final_report['training_time_minutes']:.2f} minutes |
| Runtime device | {final_report['device']} |

The fine-tuning pipeline used a warmup phase for the classifier head followed by full-network adaptation. Checkpoint selection was validation-driven, with early stopping and plot generation integrated into the workflow.

## Quantitative Results

### Validation Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | {fmt_pct(val['accuracy'])} |
| Precision | {fmt_pct(val['precision'])} |
| Recall | {fmt_pct(val['recall'])} |
| F1-score | {fmt_pct(val['f1'])} |
| AUC | {fmt_float(val['auc'])} |
| Loss | {fmt_float(val['loss'])} |

### Test Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | {fmt_pct(test['accuracy'])} |
| Precision | {fmt_pct(test['precision'])} |
| Recall | {fmt_pct(test['recall'])} |
| F1-score | {fmt_pct(test['f1'])} |
| AUC | {fmt_float(test['auc'])} |
| Loss | {fmt_float(test['loss'])} |

### Training and Evaluation Figures

{figure_markdown("Learning Curves", assets.get("learning_curves"), "Training and validation loss/metric evolution across epochs.")}

{figure_markdown("Validation Confusion Matrix", assets.get("val_confusion_matrix"), "Validation-set confusion matrix for the selected checkpoint.")}

{figure_markdown("Test Confusion Matrix", assets.get("test_confusion_matrix"), "Test-set confusion matrix showing generalization of the final model.")}

{figure_markdown("Validation ROC Curve", assets.get("val_roc_curve"), "Validation ROC curve with strong class separability.")}

{figure_markdown("Test ROC Curve", assets.get("test_roc_curve"), "Test ROC curve confirming high discriminative power on held-out identities.")}

{figure_markdown("Validation Error Gallery", assets.get("val_error_gallery"), "Representative difficult validation examples to support qualitative analysis.")}

{figure_markdown("Test Error Gallery", assets.get("test_error_gallery"), "Representative held-out failure cases useful for discussion and future work.")}

## Inference Pipeline and Deployment

The deployment pipeline uses the fine-tuned checkpoint directly. The backend now applies the same preprocessing used during training: **`299x299` resize with ImageNet normalization**. Face detection is handled exclusively through **`from mtcnn import MTCNN`**, and the face crop includes a modest margin to preserve the jawline and the area around the head.

### Backend Capabilities

- `/health` for runtime status and threshold verification
- `/predict-image` for single-image analysis
- `/predict-image-batch` for multi-image summary mode
- `/predict-video` for ratio-based video classification
- `/ws/predict-video` for streaming progress updates
- `/generate-report` for PDF export

### Frontend Capabilities

- Image, video, batch, and comparison modes
- Adjustable image and video thresholds
- WebSocket progress stream during video processing
- Video decision trace with trigger-frame chips
- Top-frame gallery with Grad-CAM visualization
- Session history and report export

## Component Verification

### Model Loader

- Model class: **{model_loader['model_class']}**
- Parameter count: **{model_loader['parameter_count']:,}**
- Trainable parameters: **{model_loader['trainable_parameter_count']:,}**
- Face detector backend: **{model_loader['face_detector_backend']}**

### Face Detection and Cropping

- Detection confidence: **{fmt_pct(face_detection['detector_probability'])}**
- Bounding box: **{face_detection['box']}**

{figure_markdown("Original Face-Detection Input", assets.get("face_original"), "Original sample used to verify face localization.")}

{figure_markdown("Annotated Face Detection", assets.get("face_annotated"), "Detected face bounding box overlaid on the original image.")}

{figure_markdown("Expanded Face Crop", assets.get("face_crop"), "Slightly enlarged face crop used for downstream model inference.")}

### Image Prediction

- Verification sample: `{image_prediction['image_path']}`
- Result: **{image_prediction['prediction']}**
- Confidence: **{fmt_pct(image_prediction['confidence'])}**
- Threshold used during check: **{image_prediction['threshold']:.2f}**

{figure_markdown("Analyzed Face", assets.get("image_prediction_face"), "Final face crop passed through the classifier for the sample fake-image check.")}

### Explainability

{figure_markdown("Explainability Input Face", assets.get("explainability_input_face"), "Face crop used for attribution generation.")}

{figure_markdown("Grad-CAM", assets.get("gradcam"), "Heatmap highlighting regions that most influenced the classifier output.")}

{figure_markdown("LIME", assets.get("lime"), "Superpixel-based explanation illustrating local decision evidence.")}

### Face Landmarks and Metadata

- Landmark count: **{landmarks['landmark_count']}**
- Metadata flags: **{'; '.join(forensics['flags'])}**

{figure_markdown("Landmark Overlay", assets.get("landmarks_image"), "Five-point facial landmark overlay on the analyzed face crop.")}

### Video Verification

- Verification clip: `{video_pipeline['video_path']}`
- Final decision: **{video_pipeline['prediction']}**
- Average confidence: **{fmt_pct(video_pipeline['confidence'])}**
- Fake-frame ratio: **{fmt_pct(video_pipeline['fake_ratio'])}**
- Fake-frame count: **{video_pipeline['fake_frame_count']} / {video_pipeline['total_frames_analyzed']}**
- Frame fake threshold: **{video_pipeline['frame_fake_threshold']:.2f}**
- Video fake-ratio threshold: **{video_pipeline['fake_ratio_threshold_used']:.2f}**
- Representative trigger frames: **{trigger_frames or 'None'}**

Decision trace:

> {video_pipeline['decision_reason']}

{figure_markdown("Top Video Frame 1", assets.get("video_top_frame_1"), "Representative high-confidence manipulated frame from the test video.")}

{figure_markdown("Top Video Face Crop 1", assets.get("video_top_face_1"), "Detected face crop corresponding to the first top suspicious frame.")}

{figure_markdown("Top Video Grad-CAM 1", assets.get("video_top_gradcam_1"), "Grad-CAM explanation for the highest-scoring suspicious video frame.")}

{figure_markdown("Top Video Frame 2", assets.get("video_top_frame_2"), "Second representative manipulated frame from the verification run.")}

{figure_markdown("Top Video Face Crop 2", assets.get("video_top_face_2"), "Detected face crop corresponding to the second suspicious frame.")}

{figure_markdown("Top Video Grad-CAM 2", assets.get("video_top_gradcam_2"), "Grad-CAM explanation for the second suspicious frame.")}

{figure_markdown("Top Video Frame 3", assets.get("video_top_frame_3"), "Third representative manipulated frame from the verification run.")}

{figure_markdown("Top Video Face Crop 3", assets.get("video_top_face_3"), "Detected face crop corresponding to the third suspicious frame.")}

{figure_markdown("Top Video Grad-CAM 3", assets.get("video_top_gradcam_3"), "Grad-CAM explanation for the third suspicious frame.")}

## Discussion

The project demonstrates that fine-tuning the XceptionNet backbone on the provided local dataset was more effective than relying on the original pretrained checkpoint. The final model achieves strong validation and test performance while remaining usable in a practical analysis interface. The identity-grouped split materially improves the credibility of the reported numbers by reducing subject leakage between partitions.

The most important engineering change after training was ensuring the backend exactly mirrored the fine-tuning pipeline. Once the deployment code was updated to use the new `best_model.pt`, apply `299x299` resizing with ImageNet normalization, and use MTCNN-only face detection with a better crop margin, the inference behavior became much more consistent with expectations.

## Limitations

1. Reported performance is still dataset-dependent and may drop under unseen manipulations, compression regimes, or demographic shifts.
2. The current video classifier relies on sampled frames rather than full temporal modeling.
3. Metadata forensics is intentionally lightweight and should be treated as supporting evidence rather than a primary decision source.
4. The pipeline currently assumes a single dominant face per frame and uses the top-confidence face detection.

## Future Work

1. Calibrate decision thresholds directly from validation-score distributions for stricter operating-point control.
2. Add temporal consistency modeling for videos to reduce frame-level noise.
3. Expand multi-face handling for crowded scenes or interview-style content.
4. Introduce external benchmark comparisons and ablation studies for explainability and crop-margin sensitivity.

## Reproducibility and Artifact Map

### Key Source Paths

- Training entry point: `own_train/train_xception.py`
- Data split logic: `own_train/data.py`
- Plot generation: `own_train/plotting.py`
- Backend API: `backend/main.py`
- Inference pipeline: `backend/predictor.py`
- Explainability: `backend/explainability.py`
- Metadata analysis: `backend/forensics.py`
- Landmark overlay: `backend/face_landmarks.py`
- PDF export: `backend/report_generator.py`
- Frontend dashboard: `frontend/src/App.jsx`

### Bundled Evidence Files

- Final metrics: `{assets.get('final_report', 'artifacts/final_report.json')}`
- Split summary: `{assets.get('split_summary', 'artifacts/split_summary.json')}`
- Training config: `{assets.get('config', 'artifacts/config.json')}`
- Model loader verification: `{assets.get('model_loader', 'artifacts/model_loader.json')}`
- Face detection verification: `{assets.get('face_detection', 'artifacts/face_detection.json')}`
- Image prediction verification: `{assets.get('image_prediction', 'artifacts/image_prediction.json')}`
- Explainability verification: `{assets.get('explainability', 'artifacts/explainability.json')}`
- Forensics verification: `{assets.get('forensics', 'artifacts/forensics.json')}`
- Landmark verification: `{assets.get('landmarks_result', 'artifacts/landmarks_result.json')}`
- Video verification: `{assets.get('video_pipeline', 'artifacts/video_pipeline.json')}`

## Conclusion

This project now includes a complete, fine-tuned deepfake detection workflow spanning training, evaluation, explainability, API deployment, and frontend presentation. The final XceptionNet model trained on the provided dataset reached **{fmt_pct(test['accuracy'])} test accuracy** and **{fmt_float(test['auc'])} test AUC**, while the surrounding platform supports single-image, batch, and video analysis with interpretable outputs. The report package generated here is intended to serve directly as a publication-ready project document and as a durable artifact bundle for future submission, export, or extension.
"""


def render_html(
    ctx: ReportContext,
    assets: dict[str, str],
    config: dict[str, Any],
    final_report: dict[str, Any],
    split_summary: dict[str, Any],
    model_loader: dict[str, Any],
    face_detection: dict[str, Any],
    image_prediction: dict[str, Any],
    explainability: dict[str, Any],
    forensics: dict[str, Any],
    landmarks: dict[str, Any],
    video_pipeline: dict[str, Any],
) -> str:
    val = final_report["final_val_metrics"]
    test = final_report["final_test_metrics"]
    split = final_report["split_summary"]

    hero_cards = [
        ("Test Accuracy", fmt_pct(test["accuracy"])),
        ("Test F1", fmt_pct(test["f1"])),
        ("Test AUC", fmt_float(test["auc"])),
        ("Best Epoch", str(final_report["best_epoch"])),
    ]

    plot_cards = [
        ("Learning Curves", assets.get("learning_curves"), "Training and validation evolution across the full fine-tuning run."),
        ("Validation Confusion Matrix", assets.get("val_confusion_matrix"), "Held-out validation confusion matrix."),
        ("Test Confusion Matrix", assets.get("test_confusion_matrix"), "Held-out test confusion matrix."),
        ("Validation ROC Curve", assets.get("val_roc_curve"), "Validation ROC curve."),
        ("Test ROC Curve", assets.get("test_roc_curve"), "Test ROC curve."),
        ("Validation Error Gallery", assets.get("val_error_gallery"), "Representative difficult validation examples."),
        ("Test Error Gallery", assets.get("test_error_gallery"), "Representative difficult test examples."),
    ]

    qualitative_cards = [
        ("Original Face-Detection Input", assets.get("face_original"), "Original image used to verify the detector."),
        ("Annotated Face Detection", assets.get("face_annotated"), "Detected face bounding box overlay."),
        ("Expanded Face Crop", assets.get("face_crop"), "Crop passed to the classifier after adding margin."),
        ("Explainability Input Face", assets.get("explainability_input_face"), "Input crop used to generate attribution maps."),
        ("Grad-CAM", assets.get("gradcam"), "Saliency map over the manipulated face."),
        ("LIME", assets.get("lime"), "Superpixel-level explanation of the classifier decision."),
        ("Landmark Overlay", assets.get("landmarks_image"), "Five-point facial landmark overlay."),
        ("Top Video Frame 1", assets.get("video_top_frame_1"), "First high-confidence fake frame from video verification."),
        ("Top Video Face 1", assets.get("video_top_face_1"), "Face crop for the first high-confidence fake frame."),
        ("Top Video Grad-CAM 1", assets.get("video_top_gradcam_1"), "Grad-CAM explanation for the first suspicious frame."),
        ("Top Video Frame 2", assets.get("video_top_frame_2"), "Second high-confidence fake frame from video verification."),
        ("Top Video Face 2", assets.get("video_top_face_2"), "Face crop for the second high-confidence fake frame."),
        ("Top Video Grad-CAM 2", assets.get("video_top_gradcam_2"), "Grad-CAM explanation for the second suspicious frame."),
        ("Top Video Frame 3", assets.get("video_top_frame_3"), "Third high-confidence fake frame from video verification."),
        ("Top Video Face 3", assets.get("video_top_face_3"), "Face crop for the third high-confidence fake frame."),
        ("Top Video Grad-CAM 3", assets.get("video_top_gradcam_3"), "Grad-CAM explanation for the third suspicious frame."),
    ]

    def stat_table(rows: list[tuple[str, str]]) -> str:
        return "".join(
            f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
            for label, value in rows
        )

    def card_grid(cards: list[tuple[str, str | None, str]]) -> str:
        return "".join(figure_html(title, path, caption) for title, path, caption in cards if path)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Deepfake Detection System Report</title>
  <style>
    :root {{
      --bg: #eef4f8;
      --paper: #ffffff;
      --ink: #12212b;
      --muted: #52636f;
      --accent: #0f766e;
      --accent-soft: #dff5f1;
      --border: #d9e6ec;
      --shadow: 0 20px 60px rgba(18, 33, 43, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Georgia", "Times New Roman", serif;
      background: radial-gradient(circle at top, #f5fbfd 0%, var(--bg) 45%, #e7eef4 100%);
      color: var(--ink);
      line-height: 1.65;
    }}
    .page {{
      width: min(1120px, calc(100% - 48px));
      margin: 32px auto 48px;
      background: var(--paper);
      border: 1px solid var(--border);
      border-radius: 28px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .hero {{
      padding: 56px 56px 32px;
      background:
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.18), transparent 28%),
        linear-gradient(135deg, #103d52 0%, #0e665b 100%);
      color: #f8fcfd;
    }}
    .hero h1 {{
      margin: 0;
      font-size: 2.9rem;
      line-height: 1.1;
      letter-spacing: -0.03em;
    }}
    .hero p {{
      margin: 14px 0 0;
      max-width: 860px;
      color: rgba(248, 252, 253, 0.88);
    }}
    .hero-meta {{
      margin-top: 24px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }}
    .hero-card {{
      padding: 16px 18px;
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.18);
      backdrop-filter: blur(10px);
    }}
    .hero-card span {{
      display: block;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: rgba(248, 252, 253, 0.7);
    }}
    .hero-card strong {{
      display: block;
      margin-top: 8px;
      font-size: 1.3rem;
    }}
    .content {{
      padding: 40px 56px 56px;
    }}
    h2 {{
      margin: 34px 0 14px;
      font-size: 1.75rem;
      color: #0f433d;
    }}
    h3 {{
      margin: 26px 0 10px;
      font-size: 1.15rem;
      color: #174e47;
    }}
    p, li {{
      font-size: 1rem;
      color: var(--ink);
    }}
    .lede {{
      font-size: 1.05rem;
      color: var(--muted);
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 22px;
      align-items: start;
    }}
    .panel {{
      padding: 22px;
      border: 1px solid var(--border);
      border-radius: 22px;
      background: linear-gradient(180deg, #ffffff 0%, #f9fcfd 100%);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0 20px;
      font-size: 0.96rem;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{
      text-align: left;
      color: #214b58;
      width: 42%;
      background: #f5fafb;
    }}
    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 18px;
    }}
    .figure-card {{
      margin: 0;
      padding: 16px;
      border: 1px solid var(--border);
      border-radius: 22px;
      background: #fbfeff;
      break-inside: avoid;
    }}
    .figure-card img {{
      width: 100%;
      display: block;
      border-radius: 16px;
      border: 1px solid #e7eef1;
      background: #eef4f6;
    }}
    .figure-card figcaption {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .artifact-list code {{
      font-size: 0.9rem;
    }}
    .quote {{
      margin: 18px 0;
      padding: 18px 20px;
      border-left: 4px solid var(--accent);
      background: var(--accent-soft);
      color: #164b45;
      border-radius: 0 16px 16px 0;
    }}
    @media print {{
      body {{ background: #fff; }}
      .page {{
        width: 100%;
        margin: 0;
        border: none;
        border-radius: 0;
        box-shadow: none;
      }}
      .hero {{
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
      }}
      .figure-card {{
        page-break-inside: avoid;
      }}
    }}
    @media (max-width: 900px) {{
      .page {{ width: calc(100% - 24px); }}
      .hero, .content {{ padding-left: 24px; padding-right: 24px; }}
      .hero-meta, .two-col, .figure-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <article class="page">
    <section class="hero">
      <h1>Deepfake Detection System Report</h1>
      <p>
        Publication-style summary of the fine-tuned XceptionNet training pipeline, deployment stack,
        evaluation metrics, explainability outputs, and verification artifacts generated for the
        <code>deepfake-detection-XceptionNet-main</code> project.
      </p>
      <div class="hero-meta">
        {"".join(f'<div class="hero-card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>' for label, value in hero_cards)}
      </div>
    </section>

    <section class="content">
      <h2>Abstract</h2>
      <p class="lede">
        This system combines a fine-tuned XceptionNet backbone, MTCNN face localization,
        training-consistent preprocessing, Grad-CAM, LIME, metadata inspection, and a FastAPI plus React
        dashboard for image and video deepfake analysis. The final model was trained on the local
        <code>Dataset</code> directory using an identity-grouped split to reduce leakage and integrated back
        into the deployment runtime with the same preprocessing used during fine-tuning.
      </p>

      <div class="two-col">
        <div class="panel">
          <h3>Core Outcome</h3>
          <p>
            The fine-tuned model achieved <strong>{fmt_pct(test['accuracy'])} test accuracy</strong>,
            <strong>{fmt_pct(test['f1'])} test F1-score</strong>, and <strong>{fmt_float(test['auc'])} test AUC</strong>.
            The representative fake-image verification sample was classified as
            <strong>{html.escape(image_prediction['prediction'])}</strong> at
            <strong>{fmt_pct(image_prediction['confidence'])}</strong> confidence, while the representative
            fake-video verification run reached a <strong>{fmt_pct(video_pipeline['fake_ratio'])}</strong> fake-frame ratio.
          </p>
        </div>
        <div class="panel">
          <h3>Deployment Snapshot</h3>
          <table>
            {stat_table([
                ("Backbone", config["model_name"]),
                ("Image size", str(config["image_size"])),
                ("Normalization", "ImageNet statistics"),
                ("Face detector", model_loader["face_detector_backend"]),
                ("Image threshold", "0.85"),
                ("Video ratio threshold", "0.40"),
            ])}
          </table>
        </div>
      </div>

      <h2>Dataset and Split Strategy</h2>
      <p>
        Training used an identity-grouped split to reduce subject overlap across train, validation, and test
        sets. This is especially important in deepfake detection because identity leakage can inflate
        performance estimates without improving generalization.
      </p>
      <table>
        <tr><th>Train identities</th><td>{split['identity_split_counts']['train']:,}</td></tr>
        <tr><th>Validation identities</th><td>{split['identity_split_counts']['val']:,}</td></tr>
        <tr><th>Test identities</th><td>{split['identity_split_counts']['test']:,}</td></tr>
        <tr><th>Train samples</th><td>{split['train_sample_count']:,}</td></tr>
        <tr><th>Validation samples</th><td>{split['val_sample_count']:,}</td></tr>
        <tr><th>Test samples</th><td>{split['test_sample_count']:,}</td></tr>
        <tr><th>Dropped cross-split fake frames</th><td>{split['dropped_cross_split_fake_frames']:,}</td></tr>
        <tr><th>Dropped cross-split fake groups</th><td>{split['dropped_cross_split_fake_groups']:,}</td></tr>
      </table>

      <h2>Training Configuration and Results</h2>
      <div class="two-col">
        <div class="panel">
          <h3>Configuration</h3>
          <table>
            {stat_table([
                ("Batch size", str(config["batch_size"])),
                ("Epochs", str(config["epochs"])),
                ("Frozen epochs", str(config["frozen_epochs"])),
                ("Learning rate", str(config["learning_rate"])),
                ("Weight decay", str(config["weight_decay"])),
                ("Patience", str(config["patience"])),
                ("Weighted sampler", str(config["use_weighted_sampler"])),
                ("AMP", str(config["amp"])),
                ("Training time", f"{final_report['training_time_minutes']:.2f} minutes"),
            ])}
          </table>
        </div>
        <div class="panel">
          <h3>Held-Out Performance</h3>
          <table>
            {stat_table([
                ("Validation accuracy", fmt_pct(val["accuracy"])),
                ("Validation AUC", fmt_float(val["auc"])),
                ("Test accuracy", fmt_pct(test["accuracy"])),
                ("Test precision", fmt_pct(test["precision"])),
                ("Test recall", fmt_pct(test["recall"])),
                ("Test F1-score", fmt_pct(test["f1"])),
                ("Test AUC", fmt_float(test["auc"])),
                ("Test loss", fmt_float(test["loss"])),
            ])}
          </table>
        </div>
      </div>

      <div class="figure-grid">
        {card_grid(plot_cards)}
      </div>

      <h2>Inference Stack and Application Design</h2>
      <div class="two-col">
        <div class="panel">
          <h3>Backend</h3>
          <ul>
            <li>FastAPI service with health, image, batch, video, WebSocket, and PDF-report endpoints.</li>
            <li>Xception model loaded from the fine-tuned <code>best_model.pt</code> checkpoint.</li>
            <li>Training-consistent preprocessing: resize to <code>299x299</code> with ImageNet normalization.</li>
            <li>MTCNN-only face detection with a slightly expanded crop margin for fuller facial coverage.</li>
          </ul>
        </div>
        <div class="panel">
          <h3>Frontend</h3>
          <ul>
            <li>Image, video, batch, and comparison workflows.</li>
            <li>Interactive threshold sliders for image/frame and video ratio decisions.</li>
            <li>Video decision trace, timeline, trigger-frame chips, and suspicious-frame gallery.</li>
            <li>Session history, explainability panels, and PDF export support.</li>
          </ul>
        </div>
      </div>

      <h2>Component Verification</h2>
      <table>
        {stat_table([
            ("Model class", model_loader["model_class"]),
            ("Parameter count", f"{model_loader['parameter_count']:,}"),
            ("Trainable parameters", f"{model_loader['trainable_parameter_count']:,}"),
            ("Detector confidence", fmt_pct(face_detection["detector_probability"])),
            ("Image verification result", f"{image_prediction['prediction']} @ {fmt_pct(image_prediction['confidence'])}"),
            ("Landmark count", str(landmarks["landmark_count"])),
            ("Metadata flags", "; ".join(forensics["flags"])),
            ("Video fake-frame ratio", fmt_pct(video_pipeline["fake_ratio"])),
            ("Video fake-frame count", f"{video_pipeline['fake_frame_count']} / {video_pipeline['total_frames_analyzed']}"),
        ])}
      </table>

      <div class="quote">{html.escape(video_pipeline["decision_reason"])}</div>

      <div class="figure-grid">
        {card_grid(qualitative_cards)}
      </div>

      <h2>Discussion and Future Work</h2>
      <p>
        The most important practical lesson from this project was that deployment fidelity matters almost as much
        as training quality. Once the backend was updated to load the new fine-tuned checkpoint, use the same
        preprocessing as the training pipeline, and rely on MTCNN-only cropping, the model behavior aligned much
        more closely with the expected results. The final stack is now suitable for demos, project reports, and
        further experimentation.
      </p>
      <p>
        Future improvements should focus on threshold calibration from validation distributions, temporal modeling
        for more robust video decisions, multi-face handling, and broader external benchmarking against unseen
        manipulations.
      </p>

      <h2>Bundled Artifact Paths</h2>
      <ul class="artifact-list">
        <li><code>{html.escape(assets.get('final_report', 'artifacts/final_report.json'))}</code></li>
        <li><code>{html.escape(assets.get('split_summary', 'artifacts/split_summary.json'))}</code></li>
        <li><code>{html.escape(assets.get('config', 'artifacts/config.json'))}</code></li>
        <li><code>{html.escape(assets.get('model_loader', 'artifacts/model_loader.json'))}</code></li>
        <li><code>{html.escape(assets.get('face_detection', 'artifacts/face_detection.json'))}</code></li>
        <li><code>{html.escape(assets.get('image_prediction', 'artifacts/image_prediction.json'))}</code></li>
        <li><code>{html.escape(assets.get('explainability', 'artifacts/explainability.json'))}</code></li>
        <li><code>{html.escape(assets.get('forensics', 'artifacts/forensics.json'))}</code></li>
        <li><code>{html.escape(assets.get('landmarks_result', 'artifacts/landmarks_result.json'))}</code></li>
        <li><code>{html.escape(assets.get('video_pipeline', 'artifacts/video_pipeline.json'))}</code></li>
      </ul>
    </section>
  </article>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the publication-style project report.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--component-dir", type=Path, default=DEFAULT_COMPONENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ctx = ReportContext(
        run_dir=args.run_dir.resolve(),
        component_dir=args.component_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        assets_dir=args.output_dir.resolve() / "assets",
        artifacts_dir=args.output_dir.resolve() / "artifacts",
    )
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    ctx.assets_dir.mkdir(parents=True, exist_ok=True)
    ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(ctx.run_dir / "config.json")
    final_report = load_json(ctx.run_dir / "final_report.json")
    split_summary = load_json(ctx.run_dir / "split_summary.json")
    model_loader = load_json(ctx.component_dir / "model_loader" / "result.json")
    face_detection = load_json(ctx.component_dir / "face_detection" / "result.json")
    image_prediction = load_json(ctx.component_dir / "image_prediction" / "result.json")
    explainability = load_json(ctx.component_dir / "explainability" / "result.json")
    forensics = load_json(ctx.component_dir / "forensics" / "result.json")
    landmarks = load_json(ctx.component_dir / "landmarks" / "result.json")
    video_pipeline = load_json(ctx.component_dir / "video_pipeline" / "result.json")

    assets = build_asset_map(ctx)

    markdown = render_markdown(
        ctx,
        assets,
        config,
        final_report,
        split_summary,
        model_loader,
        face_detection,
        image_prediction,
        explainability,
        forensics,
        landmarks,
        video_pipeline,
    )
    html_doc = render_html(
        ctx,
        assets,
        config,
        final_report,
        split_summary,
        model_loader,
        face_detection,
        image_prediction,
        explainability,
        forensics,
        landmarks,
        video_pipeline,
    )

    (ctx.output_dir / "deepfake_detection_project_report.md").write_text(markdown, encoding="utf-8")
    (ctx.output_dir / "deepfake_detection_project_report.html").write_text(html_doc, encoding="utf-8")
    (ctx.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "run_dir": str(ctx.run_dir),
                "component_dir": str(ctx.component_dir),
                "output_dir": str(ctx.output_dir),
                "bundled_assets": assets,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

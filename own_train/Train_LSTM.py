from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import random
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
from mtcnn import MTCNN
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from own_train.plotting import save_confusion_matrix, save_learning_curves, save_roc_curve


matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Hardware-aware global settings (RTX 4060 8 GB VRAM, i7-14700HX, 16 GB RAM)
# cuDNN auto-tuner picks the fastest conv algorithm for fixed input sizes.
# ---------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VideoRecord:
    path: str
    relative_path: str
    label: int
    split: str
    group_id: str
    method: str
    primary_id: str
    secondary_id: str | None
    frame_count: int
    width: int
    height: int
    codec: str
    file_size_mb: float


@dataclass
class TrainConfig:
    dataset_root: str
    metadata_csv: str
    backbone_checkpoint: str
    output_root: str = "own_train/video_runs"
    cache_root: str = "own_train/video_cache"
    image_size: int = 299
    # RTX 4060 has 8 GB VRAM.  batch_size=2 with sequence_length=10 keeps
    # peak VRAM well under 7 GB while AMP is active.  Raise these only if
    # you know your videos are short and the cache is already warm.
    sequence_length: int = 10
    batch_size: int = 2
    epochs: int = 10
    frozen_epochs: int = 2
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42
    # num_workers=0 keeps MTCNN/OpenCV in the main process (safest on
    # Windows).  Set to 2 once the cache is fully warm for a small speed
    # boost, but keep it low to avoid duplicating the model in RAM.
    num_workers: int = 0
    patience: int = 3
    use_weighted_sampler: bool = True
    decision_threshold: float = 0.5
    model_name: str = "xception"
    pretrained: bool = False
    amp: bool = True
    device: str | None = None
    face_margin: float = 0.18
    detector_interval: int = 4
    lstm_hidden_size: int = 256
    lstm_layers: int = 2
    bidirectional: bool = True
    temporal_dropout: float = 0.35
    gradient_clip_norm: float = 1.0
    cache_sequences: bool = True


def parse_args() -> TrainConfig:
    default_dataset_root = str((REPO_ROOT.parent / "FaceForensics++_C23").resolve())
    default_checkpoint = str(
        (REPO_ROOT / "own_train" / "runs" / "20260422_231937_xception" / "checkpoints" / "best_model.pt").resolve()
    )
    parser = argparse.ArgumentParser(
        description="Train an Xception + LSTM temporal deepfake detector on FaceForensics++ videos."
    )
    parser.add_argument("--dataset-root", default=default_dataset_root, help="Path to FaceForensics++_C23.")
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Optional explicit metadata CSV. Defaults to <dataset-root>/csv/FF++_Metadata.csv.",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        default=default_checkpoint,
        help="Path to the fine-tuned Xception checkpoint used to initialize the frame encoder.",
    )
    parser.add_argument("--output-root", default="own_train/video_runs", help="Directory where LSTM runs are stored.")
    parser.add_argument("--cache-root", default="own_train/video_cache", help="Directory for cached face sequences.")
    parser.add_argument("--image-size", type=int, default=299, help="Input image size for Xception frames.")
    parser.add_argument("--sequence-length", type=int, default=10, help="Number of sampled face frames per video.")
    parser.add_argument("--batch-size", type=int, default=2, help="Video batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum number of epochs.")
    parser.add_argument("--frozen-epochs", type=int, default=2, help="Backbone freeze warmup epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Temporal head learning rate.")
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate used once the Xception backbone is unfrozen.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Identity-level train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Identity-level validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader workers. Default is 0 because MTCNN-based video decoding is most reliable on Windows in the main process.",
    )
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience.")
    parser.add_argument("--disable-weighted-sampler", action="store_true", help="Disable weighted sampling.")
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Probability threshold for binary predictions.")
    parser.add_argument("--device", default=None, help="Force a device such as cpu or cuda.")
    parser.add_argument("--face-margin", type=float, default=0.18, help="Extra crop margin around detected faces.")
    parser.add_argument(
        "--detector-interval",
        type=int,
        default=4,
        help="Run MTCNN once every N sampled sequence steps and reuse the last valid box in between.",
    )
    parser.add_argument("--lstm-hidden-size", type=int, default=256, help="LSTM hidden size.")
    parser.add_argument("--lstm-layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--temporal-dropout", type=float, default=0.35, help="Dropout used in temporal head.")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--disable-cache", action="store_true", help="Disable on-disk sequence caching.")

    args = parser.parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    metadata_csv = Path(args.metadata_csv).resolve() if args.metadata_csv else (dataset_root / "csv" / "FF++_Metadata.csv")
    return TrainConfig(
        dataset_root=str(dataset_root),
        metadata_csv=str(metadata_csv),
        backbone_checkpoint=str(Path(args.backbone_checkpoint).resolve()),
        output_root=args.output_root,
        cache_root=args.cache_root,
        image_size=args.image_size,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        frozen_epochs=args.frozen_epochs,
        learning_rate=args.learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        weight_decay=args.weight_decay,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        patience=args.patience,
        use_weighted_sampler=not args.disable_weighted_sampler,
        decision_threshold=args.decision_threshold,
        device=args.device,
        face_margin=args.face_margin,
        detector_interval=max(1, args.detector_interval),
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        temporal_dropout=args.temporal_dropout,
        gradient_clip_norm=args.gradient_clip_norm,
        cache_sequences=not args.disable_cache,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_identities(
    identity_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    if not identity_ids:
        raise ValueError("No identity IDs were found for splitting.")

    rng = random.Random(seed)
    ids = list(identity_ids)
    rng.shuffle(ids)

    train_count = int(len(ids) * train_ratio)
    val_count = int(len(ids) * val_ratio)
    test_count = len(ids) - train_count - val_count
    if test_count <= 0:
        raise ValueError("Split ratios leave no identities for the test split.")

    split_map: dict[str, str] = {}
    for index, identity_id in enumerate(ids):
        if index < train_count:
            split_map[identity_id] = "train"
        elif index < train_count + val_count:
            split_map[identity_id] = "val"
        else:
            split_map[identity_id] = "test"
    return split_map


def _register_timm_compat_aliases() -> None:
    module_aliases = {
        "timm.models.layers": "timm.layers",
        "timm.models.layers.adaptive_avgmax_pool": "timm.layers.adaptive_avgmax_pool",
        "timm.models.layers.activations": "timm.layers.activations",
        "timm.models.layers.activations_jit": "timm.layers.activations_jit",
        "timm.models.layers.blur_pool": "timm.layers.blur_pool",
        "timm.models.layers.classifier": "timm.layers.classifier",
        "timm.models.layers.config": "timm.layers.config",
        "timm.models.layers.conv2d_same": "timm.layers.conv2d_same",
        "timm.models.layers.create_act": "timm.layers.create_act",
        "timm.models.layers.create_attn": "timm.layers.create_attn",
        "timm.models.layers.create_conv2d": "timm.layers.create_conv2d",
        "timm.models.layers.create_norm": "timm.layers.create_norm",
        "timm.models.layers.drop": "timm.layers.drop",
        "timm.models.layers.fast_norm": "timm.layers.fast_norm",
        "timm.models.layers.helpers": "timm.layers.helpers",
        "timm.models.layers.mixed_conv2d": "timm.layers.mixed_conv2d",
        "timm.models.layers.norm": "timm.layers.norm",
        "timm.models.layers.norm_act": "timm.layers.norm_act",
        "timm.models.layers.padding": "timm.layers.padding",
        "timm.models.layers.pool2d_same": "timm.layers.pool2d_same",
        "timm.models.layers.separable_conv": "timm.layers.separable_conv",
        "timm.models.layers.space_to_depth": "timm.layers.space_to_depth",
        "timm.models.layers.split_attn": "timm.layers.split_attn",
        "timm.models.layers.std_conv": "timm.layers.std_conv",
        "timm.models.layers.test_time_pool": "timm.layers.test_time_pool",
        "timm.models.layers.trace_utils": "timm.layers.trace_utils",
        "timm.models.layers.weight_init": "timm.layers.weight_init",
    }

    for old_name, new_name in module_aliases.items():
        if old_name in sys.modules:
            continue
        try:
            sys.modules[old_name] = importlib.import_module(new_name)
        except ModuleNotFoundError:
            continue


def torch_load_any(path: str | Path, device: torch.device) -> Any:
    _register_timm_compat_aliases()
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_video_identities(relative_path: str) -> tuple[str, str | None, str, str]:
    path_obj = Path(relative_path)
    method = path_obj.parts[0]
    stem = path_obj.stem
    if method.lower() == "original":
        return stem, None, method, f"{method}/{stem}"

    pair_prefix = stem.split("__", maxsplit=1)[0]
    pair_tokens = pair_prefix.split("_")
    if len(pair_tokens) >= 2:
        primary_id, secondary_id = pair_tokens[0], pair_tokens[1]
    else:
        primary_id = pair_prefix
        secondary_id = None
    return primary_id, secondary_id, method, f"{method}/{pair_prefix}"


def _build_existing_paths_set(root: Path) -> set[str]:
    """Walk the dataset root once with os.scandir (fast) and return a set of
    lowercased absolute path strings.  This is orders of magnitude faster than
    calling Path.exists() on each CSV row individually on Windows."""
    import os
    existing: set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            existing.add(os.path.join(dirpath, fname).lower())
    return existing


def scan_faceforensics_metadata(dataset_root: str | Path, metadata_csv: str | Path) -> list[dict[str, Any]]:
    root = Path(dataset_root).resolve()
    csv_path = Path(metadata_csv).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    # Build the set of existing files once (one fast os.walk) instead of
    # calling path.exists() for every CSV row — critical for Windows perf.
    print("Scanning dataset directory for existing video files… (one-time, fast)")
    existing_paths = _build_existing_paths_set(root)
    print(f"  Found {len(existing_paths):,} files on disk.")

    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            relative_path = row.get("File Path")
            label_text = (row.get("Label") or "").strip().upper()
            if not relative_path or label_text not in {"REAL", "FAKE"}:
                continue

            resolved_path = (root / relative_path).resolve()
            # Fast set lookup instead of a syscall per row.
            if resolved_path.__str__().lower() not in existing_paths:
                continue

            primary_id, secondary_id, method, group_id = parse_video_identities(relative_path)
            rows.append(
                {
                    "path": str(resolved_path),
                    "relative_path": relative_path,
                    "label": 1 if label_text == "FAKE" else 0,
                    "primary_id": primary_id,
                    "secondary_id": secondary_id,
                    "method": method,
                    "group_id": group_id,
                    "frame_count": int(float(row.get("Frame Count") or 0)),
                    "width": int(float(row.get("Width") or 0)),
                    "height": int(float(row.get("Height") or 0)),
                    "codec": row.get("Codec") or "",
                    "file_size_mb": float(row.get("File Size(MB)") or 0.0),
                }
            )
    if not rows:
        raise RuntimeError("No valid video rows were loaded from the FaceForensics++ metadata CSV.")
    return rows



def build_grouped_video_splits(
    dataset_root: str | Path,
    metadata_csv: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[dict[str, list[VideoRecord]], dict[str, Any]]:
    scanned = scan_faceforensics_metadata(dataset_root, metadata_csv)
    real_records = [row for row in scanned if row["label"] == 0]
    fake_records = [row for row in scanned if row["label"] == 1]

    identity_ids = sorted(
        {
            identity
            for row in scanned
            for identity in (row["primary_id"], row["secondary_id"])
            if identity
        }
    )
    identity_split_map = split_identities(identity_ids, train_ratio, val_ratio, seed)

    split_records: dict[str, list[VideoRecord]] = {"train": [], "val": [], "test": []}
    split_summary: dict[str, Any] = {
        "split_mode": "identity_grouped_video",
        "identity_split_counts": Counter(identity_split_map.values()),
        "real_video_counts": {},
        "fake_video_counts": {},
        "fake_method_counts": {},
        "dropped_cross_split_fake_videos": 0,
        "retained_fake_videos": 0,
        "avg_frame_count": {},
    }

    for row in real_records:
        split = identity_split_map[row["primary_id"]]
        split_records[split].append(VideoRecord(split=split, **row))

    for row in fake_records:
        left_split = identity_split_map.get(row["primary_id"])
        right_split = identity_split_map.get(row["secondary_id"] or "")
        if left_split is None or right_split is None or left_split != right_split:
            split_summary["dropped_cross_split_fake_videos"] += 1
            continue
        split = left_split
        split_records[split].append(VideoRecord(split=split, **row))
        split_summary["retained_fake_videos"] += 1

    for split_name, records in split_records.items():
        label_counts = Counter(record.label for record in records)
        method_counts = Counter(record.method for record in records if record.label == 1)
        frame_counts = [record.frame_count for record in records if record.frame_count > 0]
        split_summary["real_video_counts"][split_name] = int(label_counts.get(0, 0))
        split_summary["fake_video_counts"][split_name] = int(label_counts.get(1, 0))
        split_summary["fake_method_counts"][split_name] = dict(sorted(method_counts.items()))
        split_summary[f"{split_name}_sample_count"] = len(records)
        split_summary["avg_frame_count"][split_name] = float(np.mean(frame_counts)) if frame_counts else 0.0

    return split_records, split_summary


def build_frame_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def create_weighted_sampler(records: list[VideoRecord]) -> WeightedRandomSampler:
    label_counts = np.bincount([record.label for record in records], minlength=2)
    class_weights = np.array([0.0, 0.0], dtype=np.float64)
    for label, count in enumerate(label_counts):
        if count > 0:
            class_weights[label] = 1.0 / count
    sample_weights = [class_weights[record.label] for record in records]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


class FaceSequenceDataset(Dataset):
    def __init__(
        self,
        records: list[VideoRecord],
        config: TrainConfig,
        transform: transforms.Compose,
        cache_root: str | Path,
    ) -> None:
        self.records = records
        self.config = config
        self.transform = transform
        self.cache_root = Path(cache_root).resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._detector: MTCNN | None = None
        self._record_lookup = {record.path: record for record in records}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        frames = self.get_sequence_array(record)  # shape: (T, H, W, 3)  uint8
        # Build tensors one frame at a time to avoid allocating a large
        # intermediate list of float32 tensors before stacking.
        frame_tensors = []
        for frame in frames:
            frame_tensors.append(self.transform(Image.fromarray(frame)))
        sequence_tensor = torch.stack(frame_tensors, dim=0)  # (T, C, H, W)
        del frame_tensors  # free list before returning
        return {
            "sequence": sequence_tensor,
            "label": record.label,
            "path": record.path,
            "relative_path": record.relative_path,
            "group_id": record.group_id,
            "method": record.method,
            "split": record.split,
            "frame_count": record.frame_count,
        }

    def get_record(self, path: str) -> VideoRecord:
        return self._record_lookup[path]

    def get_cache_path(self, record: VideoRecord) -> Path:
        cache_key = hashlib.sha1(
            (
                f"{record.relative_path}|{self.config.sequence_length}|{self.config.face_margin:.3f}|"
                f"{self.config.detector_interval}|v3"
            ).encode("utf-8")
        ).hexdigest()
        return self.cache_root / f"{cache_key}.npz"

    def get_sequence_array(self, record: VideoRecord) -> np.ndarray:
        cache_path = self.get_cache_path(record)
        if self.config.cache_sequences and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            return cached["frames"]

        frames = self._extract_face_sequence(record)
        if self.config.cache_sequences:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npz", dir=cache_path.parent) as temp_handle:
                np.savez_compressed(temp_handle.name, frames=frames)
                temp_path = Path(temp_handle.name)
            temp_path.replace(cache_path)
        return frames

    def _get_detector(self) -> MTCNN:
        if self._detector is None:
            self._detector = MTCNN()
        return self._detector

    def _sample_frame_indices(self, total_frames: int) -> list[int]:
        if total_frames <= 0:
            return [0] * self.config.sequence_length
        return np.linspace(0, max(total_frames - 1, 0), num=self.config.sequence_length, dtype=np.int32).tolist()

    def _extract_face_sequence(self, record: VideoRecord) -> np.ndarray:
        capture = cv2.VideoCapture(record.path)
        total_frames = int(record.frame_count or capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_indices = self._sample_frame_indices(total_frames)
        faces: list[np.ndarray] = []
        last_box: tuple[int, int, int, int] | None = None

        try:
            for step_index, frame_index in enumerate(frame_indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame_bgr = capture.read()
                if not success or frame_bgr is None:
                    if faces:
                        faces.append(faces[-1].copy())
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                should_run_detector = last_box is None or (step_index % self.config.detector_interval == 0)
                face_rgb, last_box = self._extract_face_from_frame(
                    frame_rgb,
                    last_box,
                    run_detector=should_run_detector,
                )
                faces.append(self._normalize_face_size(face_rgb))
        finally:
            capture.release()

        if not faces:
            raise RuntimeError(f"Failed to extract any frames from video: {record.path}")

        while len(faces) < self.config.sequence_length:
            faces.append(faces[-1].copy())
        return np.stack(faces[: self.config.sequence_length], axis=0).astype(np.uint8)

    def _normalize_face_size(self, face_rgb: np.ndarray) -> np.ndarray:
        target_size = self.config.image_size
        height, width = face_rgb.shape[:2]
        if height <= 0 or width <= 0:
            return np.zeros((target_size, target_size, 3), dtype=np.uint8)

        scale = min(target_size / max(width, 1), target_size / max(height, 1))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(face_rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        offset_x = (target_size - resized_width) // 2
        offset_y = (target_size - resized_height) // 2
        canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
        return canvas

    def _extract_face_from_frame(
        self,
        frame_rgb: np.ndarray,
        last_box: tuple[int, int, int, int] | None,
        run_detector: bool = True,
    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        if run_detector:
            detector = self._get_detector()
            detections = detector.detect_faces(frame_rgb)
            if detections:
                detections = sorted(detections, key=lambda item: item.get("confidence", 0.0), reverse=True)
                best = detections[0]
                x, y, width, height = best["box"]
                x = max(0, int(x))
                y = max(0, int(y))
                width = max(1, int(width))
                height = max(1, int(height))
                end_x = min(frame_rgb.shape[1], x + width)
                end_y = min(frame_rgb.shape[0], y + height)

                margin_x = int(width * self.config.face_margin)
                margin_y = int(height * self.config.face_margin)
                start_x = max(0, x - margin_x)
                start_y = max(0, y - margin_y)
                crop_end_x = min(frame_rgb.shape[1], end_x + margin_x)
                crop_end_y = min(frame_rgb.shape[0], end_y + margin_y)
                crop = frame_rgb[start_y:crop_end_y, start_x:crop_end_x]
                if crop.size > 0:
                    return crop, (start_x, start_y, crop_end_x, crop_end_y)

        if last_box is not None:
            start_x, start_y, end_x, end_y = last_box
            start_x = max(0, min(frame_rgb.shape[1] - 1, start_x))
            start_y = max(0, min(frame_rgb.shape[0] - 1, start_y))
            end_x = max(start_x + 1, min(frame_rgb.shape[1], end_x))
            end_y = max(start_y + 1, min(frame_rgb.shape[0], end_y))
            crop = frame_rgb[start_y:end_y, start_x:end_x]
            if crop.size > 0:
                return crop, last_box

        height, width = frame_rgb.shape[:2]
        side = min(height, width)
        start_x = max(0, (width - side) // 2)
        start_y = max(0, (height - side) // 2)
        crop = frame_rgb[start_y : start_y + side, start_x : start_x + side]
        return crop, (start_x, start_y, start_x + side, start_y + side)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, sequence_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.attention(sequence_features)
        weights = torch.softmax(weights, dim=1)
        pooled = torch.sum(sequence_features * weights, dim=1)
        return pooled, weights.squeeze(-1)


class XceptionLSTMDetector(nn.Module):
    def __init__(self, config: TrainConfig, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.device_ref = device
        self.backbone = timm.create_model(config.model_name, pretrained=config.pretrained, num_classes=1)
        self.backbone = self.backbone.to(device)
        self.backbone_feature_dim = self._load_backbone_checkpoint(config.backbone_checkpoint, device)
        lstm_input_dim = self.backbone_feature_dim + 1
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.temporal_dropout if config.lstm_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        temporal_dim = config.lstm_hidden_size * (2 if config.bidirectional else 1)
        self.attention_pool = TemporalAttentionPooling(temporal_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Dropout(config.temporal_dropout),
            nn.Linear(temporal_dim, temporal_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.temporal_dropout),
            nn.Linear(temporal_dim // 2, 1),
        )
        self.to(device)

    def _load_backbone_checkpoint(self, checkpoint_path: str, device: torch.device) -> int:
        checkpoint = torch_load_any(checkpoint_path, device)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        elif isinstance(checkpoint, nn.Module):
            state_dict = checkpoint.state_dict()
        else:
            state_dict = None
        if not state_dict:
            raise RuntimeError("Backbone checkpoint does not contain a usable state_dict.")

        cleaned_state = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
        missing_keys, unexpected_keys = self.backbone.load_state_dict(cleaned_state, strict=False)
        if unexpected_keys:
            print(f"Warning: unexpected keys when loading backbone checkpoint: {unexpected_keys[:10]}")
        allowed_missing = {"fc.weight", "fc.bias"}
        disallowed_missing = [key for key in missing_keys if key not in allowed_missing]
        if disallowed_missing:
            print(f"Warning: missing backbone keys after checkpoint load: {disallowed_missing[:10]}")

        self.backbone = self.backbone.to(device)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.config.image_size, self.config.image_size, device=device)
            features, _ = self._extract_backbone_outputs(dummy)
        return int(features.shape[1])

    def _extract_backbone_outputs(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone.forward_features(frames)
        if hasattr(self.backbone, "global_pool"):
            features = self.backbone.global_pool(features)
        if features.ndim > 2:
            features = torch.flatten(features, 1)

        classifier = getattr(self.backbone, "fc", None)
        if classifier is None:
            raise AttributeError("Expected the Xception backbone to expose classifier head as `fc`.")
        frame_logits = classifier(features)
        return features, frame_logits

    def forward(self, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, time_steps, channels, height, width = sequences.shape
        # Flatten (B, T, C, H, W) -> (B*T, C, H, W) for the backbone.
        # We process all frames in one shot; AMP keeps peak VRAM manageable.
        flattened = sequences.view(batch_size * time_steps, channels, height, width)
        frame_features, frame_logits = self._extract_backbone_outputs(flattened)
        temporal_input = torch.cat([frame_features, frame_logits], dim=1)
        # Reshape back to (B, T, features) before the LSTM.
        temporal_input = temporal_input.view(batch_size, time_steps, -1)
        lstm_output, _ = self.lstm(temporal_input)
        pooled, attention_weights = self.attention_pool(lstm_output)
        logits = self.classifier(pooled)
        # Free intermediate tensors explicitly to reduce peak VRAM.
        del flattened, frame_features, frame_logits, temporal_input, lstm_output
        return logits, attention_weights


def set_backbone_trainability(model: XceptionLSTMDetector, freeze_backbone: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = not freeze_backbone


def make_optimizer(model: XceptionLSTMDetector, config: TrainConfig) -> torch.optim.Optimizer:
    backbone_params = list(model.backbone.parameters())
    temporal_params = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    ]
    return AdamW(
        [
            {"params": backbone_params, "lr": config.backbone_learning_rate},
            {"params": temporal_params, "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )


def make_loss(records: list[VideoRecord], device: torch.device) -> nn.Module:
    labels = np.array([record.label for record in records], dtype=np.int64)
    real_count = max(int((labels == 0).sum()), 1)
    fake_count = max(int((labels == 1).sum()), 1)
    pos_weight = torch.tensor([real_count / fake_count], device=device, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def create_dataloaders(
    split_records: dict[str, list[VideoRecord]],
    config: TrainConfig,
) -> tuple[dict[str, DataLoader], dict[str, FaceSequenceDataset]]:
    train_transform, eval_transform = build_frame_transforms(config.image_size)
    datasets = {
        "train": FaceSequenceDataset(split_records["train"], config, train_transform, config.cache_root),
        "val": FaceSequenceDataset(split_records["val"], config, eval_transform, config.cache_root),
        "test": FaceSequenceDataset(split_records["test"], config, eval_transform, config.cache_root),
    }

    # pin_memory speeds up host→GPU transfers but costs extra RAM.
    # With 16 GB system RAM and large video sequences we keep it enabled
    # only when workers > 0 (i.e. data is prepared in background threads).
    use_pin_memory = torch.cuda.is_available() and config.num_workers > 0
    # persistent_workers keeps worker processes alive between epochs,
    # avoiding the overhead of re-spawning them every epoch.
    use_persistent = config.num_workers > 0

    train_loader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": use_pin_memory,
        "persistent_workers": use_persistent,
    }
    if config.use_weighted_sampler:
        train_loader_kwargs["sampler"] = create_weighted_sampler(split_records["train"])
    else:
        train_loader_kwargs["shuffle"] = True

    dataloaders = {
        "train": DataLoader(datasets["train"], **train_loader_kwargs),
        "val": DataLoader(
            datasets["val"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent,
        ),
    }
    return dataloaders, datasets


def batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = batch["sequence"].to(device, non_blocking=True)
    labels = batch["label"].float().unsqueeze(1).to(device, non_blocking=True)
    return sequences, labels


def compute_metrics(labels: list[int], probabilities: list[float], predictions: list[int]) -> dict[str, float]:
    if not labels:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.0,
        }

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(labels, probabilities))
    except ValueError:
        metrics["auc"] = 0.0
    return metrics


def train_one_epoch(
    model: XceptionLSTMDetector,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    threshold: float,
    amp_enabled: bool,
    clip_norm: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_items = 0
    probabilities: list[float] = []
    labels: list[int] = []

    progress = tqdm(loader, leave=False)
    for batch in progress:
        sequences, target = batch_to_device(batch, device)
        # zero_grad(set_to_none=True) frees gradient buffers instead of
        # zeroing them in-place, reducing peak VRAM by up to 1×param-size.
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            logits, _ = model(sequences)
            loss = criterion(logits, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        scaler.step(optimizer)
        scaler.update()

        probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
        batch_labels = target.squeeze(1).detach().cpu().numpy().astype(int)
        loss_val = float(loss.item())
        # Explicitly delete GPU tensors we no longer need so the allocator
        # can reclaim VRAM before the next batch arrives.
        del sequences, target, logits
        total_loss += loss_val * len(batch_labels)
        total_items += len(batch_labels)
        probabilities.extend(probs.tolist())
        labels.extend(batch_labels.tolist())
        progress.set_postfix(loss=f"{loss_val:.4f}")

    predictions = [int(probability >= threshold) for probability in probabilities]
    metrics = compute_metrics(labels, probabilities, predictions)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def evaluate_model(
    model: XceptionLSTMDetector,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    amp_enabled: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    probabilities: list[float] = []
    labels: list[int] = []
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            sequences, target = batch_to_device(batch, device)
            with autocast(enabled=amp_enabled):
                logits, attention_weights = model(sequences)
                loss = criterion(logits, target)

            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            batch_labels = target.squeeze(1).detach().cpu().numpy().astype(int)
            attention = attention_weights.detach().cpu().numpy()
            # Free GPU tensors immediately after pulling data to CPU.
            del sequences, target, logits, attention_weights

            total_loss += float(loss.item()) * len(batch_labels)
            total_items += len(batch_labels)
            probabilities.extend(probs.tolist())
            labels.extend(batch_labels.tolist())

            for path, relative_path, label, probability, method, frame_count, attention_map in zip(
                batch["path"],
                batch["relative_path"],
                batch_labels.tolist(),
                probs.tolist(),
                batch["method"],
                batch["frame_count"].tolist(),
                attention.tolist(),
            ):
                prediction = int(probability >= threshold)
                peak_index = int(np.argmax(attention_map)) if attention_map else 0
                records.append(
                    {
                        "path": path,
                        "relative_path": relative_path,
                        "label": int(label),
                        "probability": float(probability),
                        "prediction": prediction,
                        "method": method,
                        "frame_count": int(frame_count),
                        "peak_attention_index": peak_index,
                    }
                )

    predictions = [int(probability >= threshold) for probability in probabilities]
    metrics = compute_metrics(labels, probabilities, predictions)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics, records


def save_csv_records(records: list[dict[str, Any]], output_path: str | Path) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_json(data: Any, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def serialize_video_records(records: list[VideoRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]


def save_video_error_gallery(
    records: list[dict[str, Any]],
    dataset: FaceSequenceDataset,
    output_path: str | Path,
    max_items: int = 12,
) -> None:
    if not records:
        return

    selected = records[:max_items]
    columns = 3
    rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, 4.2 * rows))
    axes = np.atleast_1d(axes).flatten()

    for axis in axes:
        axis.axis("off")

    for axis, record in zip(axes, selected):
        video_record = dataset.get_record(record["path"])
        frames = dataset.get_sequence_array(video_record)
        preview_index = min(len(frames) - 1, record.get("peak_attention_index", len(frames) // 2))
        axis.imshow(frames[preview_index])
        axis.set_title(
            f"{video_record.method}\n"
            f"GT: {'FAKE' if record['label'] else 'REAL'} | Pred: {'FAKE' if record['prediction'] else 'REAL'}\n"
            f"P(fake)={record['probability']:.3f} | Peak step={preview_index}",
            fontsize=9,
        )
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def train_model(config: TrainConfig) -> Path:
    seed_everything(config.seed)
    device = resolve_device(config.device)

    run_dir = Path(config.output_root).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_xception_lstm")
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    checkpoints_dir = run_dir / "checkpoints"
    reports_dir = run_dir / "reports"
    for path in (plots_dir, checkpoints_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    split_records, split_summary = build_grouped_video_splits(
        dataset_root=config.dataset_root,
        metadata_csv=config.metadata_csv,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    dataloaders, datasets = create_dataloaders(split_records, config)

    model = XceptionLSTMDetector(config, device)
    criterion = make_loss(split_records["train"], device)
    optimizer = make_optimizer(model, config)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6)
    scaler = GradScaler(enabled=config.amp and device.type == "cuda")

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_score = (-1.0, -1.0, float("inf"))
    patience_counter = 0

    save_json(asdict(config), run_dir / "config.json")
    save_json(split_summary, run_dir / "split_summary.json")
    save_json(
        {split: serialize_video_records(records[:100]) for split, records in split_records.items()},
        run_dir / "sample_manifest_preview.json",
    )

    # Warm up the CUDA allocator and clear any fragmentation from model init.
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(device)} | "
              f"VRAM allocated after model init: "
              f"{torch.cuda.memory_allocated(device) / 1024**3:.2f} GB / "
              f"{torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")

    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        freeze_backbone = epoch <= config.frozen_epochs
        set_backbone_trainability(model, freeze_backbone=freeze_backbone)

        train_metrics = train_one_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            scaler,
            device,
            config.decision_threshold,
            amp_enabled=config.amp and device.type == "cuda",
            clip_norm=config.gradient_clip_norm,
        )
        val_metrics, val_records = evaluate_model(
            model,
            dataloaders["val"],
            criterion,
            device,
            config.decision_threshold,
            amp_enabled=config.amp and device.type == "cuda",
        )
        scheduler.step(val_metrics["loss"])

        # Release any cached VRAM fragments between epochs so the next
        # epoch starts with a clean allocator state.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        epoch_record = {
            "epoch": epoch,
            "backbone_learning_rate": optimizer.param_groups[0]["lr"],
            "temporal_learning_rate": optimizer.param_groups[1]["lr"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "backbone_frozen": freeze_backbone,
        }
        history.append(epoch_record)
        save_json(history, run_dir / "history.json")

        current_score = (val_metrics["auc"], val_metrics["f1"], -val_metrics["loss"])
        if current_score > best_score:
            best_score = current_score
            patience_counter = 0
            # Save model weights on CPU to avoid doubling VRAM during
            # serialisation (torch.save streams from the tensor's device).
            best_state = {
                "epoch": epoch,
                "config": asdict(config),
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "history": history,
                "val_metrics": val_metrics,
                "backbone_feature_dim": model.backbone_feature_dim,
                "model_family": "xception_lstm_temporal",
            }
            torch.save(best_state, checkpoints_dir / "best_lstm_model.pt")
            save_csv_records(val_records, reports_dir / "best_val_predictions.csv")
        else:
            patience_counter += 1

        # Last checkpoint also stores weights on CPU to keep save() cheap.
        torch.save(
            {
                "epoch": epoch,
                "config": asdict(config),
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "backbone_feature_dim": model.backbone_feature_dim,
                "model_family": "xception_lstm_temporal",
            },
            checkpoints_dir / "last_lstm_model.pt",
        )

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        if patience_counter >= config.patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training completed without producing a best checkpoint.")

    model.load_state_dict(best_state["model_state_dict"])

    val_metrics, val_records = evaluate_model(
        model,
        dataloaders["val"],
        criterion,
        device,
        config.decision_threshold,
        amp_enabled=config.amp and device.type == "cuda",
    )
    test_metrics, test_records = evaluate_model(
        model,
        dataloaders["test"],
        criterion,
        device,
        config.decision_threshold,
        amp_enabled=config.amp and device.type == "cuda",
    )

    save_csv_records(val_records, reports_dir / "val_predictions.csv")
    save_csv_records(test_records, reports_dir / "test_predictions.csv")

    val_labels = [record["label"] for record in val_records]
    val_probs = [record["probability"] for record in val_records]
    val_preds = [record["prediction"] for record in val_records]
    test_labels = [record["label"] for record in test_records]
    test_probs = [record["probability"] for record in test_records]
    test_preds = [record["prediction"] for record in test_records]

    save_learning_curves(history, plots_dir / "learning_curves.png")
    save_confusion_matrix(val_labels, val_preds, plots_dir / "val_confusion_matrix.png", "Validation Confusion Matrix")
    save_confusion_matrix(test_labels, test_preds, plots_dir / "test_confusion_matrix.png", "Test Confusion Matrix")
    save_roc_curve(val_labels, val_probs, plots_dir / "val_roc_curve.png", "Validation ROC Curve")
    save_roc_curve(test_labels, test_probs, plots_dir / "test_roc_curve.png", "Test ROC Curve")

    val_errors = [record for record in val_records if record["label"] != record["prediction"]]
    val_errors.sort(key=lambda item: abs(item["probability"] - item["label"]), reverse=True)
    save_video_error_gallery(val_errors, datasets["val"], plots_dir / "val_error_gallery.png")

    test_errors = [record for record in test_records if record["label"] != record["prediction"]]
    test_errors.sort(key=lambda item: abs(item["probability"] - item["label"]), reverse=True)
    save_video_error_gallery(test_errors, datasets["test"], plots_dir / "test_error_gallery.png")

    final_report = {
        "best_epoch": best_state["epoch"],
        "training_time_minutes": round((time.time() - start_time) / 60.0, 2),
        "device": str(device),
        "split_summary": split_summary,
        "backbone_checkpoint": config.backbone_checkpoint,
        "sequence_length": config.sequence_length,
        "backbone_feature_dim": model.backbone_feature_dim,
        "final_val_metrics": val_metrics,
        "final_test_metrics": test_metrics,
        "classification_report_val": classification_report(
            val_labels,
            val_preds,
            target_names=["REAL", "FAKE"],
            output_dict=True,
            zero_division=0,
        ),
        "classification_report_test": classification_report(
            test_labels,
            test_preds,
            target_names=["REAL", "FAKE"],
            output_dict=True,
            zero_division=0,
        ),
    }
    save_json(final_report, run_dir / "final_report.json")

    print("\nTemporal training complete.")
    print(f"Best checkpoint: {checkpoints_dir / 'best_lstm_model.pt'}")
    print(f"Final report: {run_dir / 'final_report.json'}")
    print(f"Plots directory: {plots_dir}")
    print(
        "Summary metrics | "
        f"val_acc={val_metrics['accuracy']:.4f}, val_auc={val_metrics['auc']:.4f}, "
        f"test_acc={test_metrics['accuracy']:.4f}, test_auc={test_metrics['auc']:.4f}"
    )
    return run_dir


def main() -> int:
    config = parse_args()
    train_model(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
predictor.py — Deepfake Detection Runtime
==========================================
Primary model  : Xception + LSTM temporal detector (video-level)
Fallback model : Fine-tuned Xception (frame-level, used for images and
                 when the LSTM model is unavailable)

Loading priority
----------------
1. LSTM checkpoint is attempted first for video inference.
2. If the LSTM model is missing or fails to load, video inference
   transparently falls back to the Xception frame-level model.
3. Image inference always uses the Xception frame-level model because
   a single image provides no temporal sequence.
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None

try:
    from mtcnn import MTCNN
except ImportError:  # pragma: no cover
    MTCNN = None


# ---------------------------------------------------------------------------
# Global runtime state
# ---------------------------------------------------------------------------
MODEL: torch.nn.Module | None = None           # Xception frame-level model
LSTM_MODEL: torch.nn.Module | None = None      # Xception+LSTM temporal model
LSTM_CONFIG: dict[str, Any] | None = None      # Config stored in LSTM checkpoint
DEVICE: torch.device | None = None
FACE_DETECTOR: Any | None = None
FACE_DETECTOR_BACKEND: str | None = None
MODEL_PATH: str | None = None
MODEL_CONFIG: dict[str, Any] | None = None
LSTM_MODEL_PATH: str | None = None

# ---------------------------------------------------------------------------
# Checkpoint discovery order
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

# Xception frame-level candidates (fallback)
MODEL_CANDIDATES = [
    "own_train/runs/20260422_231937_xception/checkpoints/best_model.pt",
    "Models/FFPP_model_20_epochs_99acc.pt",
]

# LSTM temporal candidates (primary for video)
LSTM_MODEL_CANDIDATES = [
    "own_train/video_runs/20260527_204127_xception_lstm/checkpoints/best_lstm_model.pt",
]

# ---------------------------------------------------------------------------
# Image preprocessing constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
PREPROCESS_IMAGE_SIZE  = 128
PREPROCESS_NORMALIZE   = False
PREPROCESS_TRANSFORM   = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((PREPROCESS_IMAGE_SIZE, PREPROCESS_IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)
FACE_CROP_MARGIN = 0.18


# ---------------------------------------------------------------------------
# Helpers — thresholds from env
# ---------------------------------------------------------------------------
def get_default_image_threshold() -> float:
    try:
        return float(os.getenv("DEFAULT_IMAGE_THRESHOLD", "0.85"))
    except ValueError:
        return 0.85


def get_default_video_fake_ratio_threshold() -> float:
    try:
        return float(os.getenv("DEFAULT_VIDEO_FAKE_RATIO_THRESHOLD", "0.4"))
    except ValueError:
        return 0.4


# ---------------------------------------------------------------------------
# Transform builder
# ---------------------------------------------------------------------------
def _build_preprocess_transform(image_size: int, normalize: bool) -> transforms.Compose:
    steps: list[Any] = [
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if normalize:
        steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(steps)


def _apply_checkpoint_preprocessing_config(config: dict[str, Any] | None) -> None:
    global PREPROCESS_IMAGE_SIZE, PREPROCESS_NORMALIZE, PREPROCESS_TRANSFORM, MODEL_CONFIG

    checkpoint_config = dict(config or {})
    image_size = int(checkpoint_config.get("image_size", 128))
    normalize  = bool(checkpoint_config)

    PREPROCESS_IMAGE_SIZE = image_size
    PREPROCESS_NORMALIZE  = normalize
    PREPROCESS_TRANSFORM  = _build_preprocess_transform(image_size=image_size, normalize=normalize)
    MODEL_CONFIG          = checkpoint_config or None


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------
def _resolve_candidates(candidates: list[str], explicit_path: str | None = None) -> str:
    raw: list[Path] = []
    if explicit_path:
        raw.append(Path(explicit_path))
    for c in candidates:
        raw.append(Path(c))

    expanded: list[Path] = []
    for p in raw:
        if p.is_absolute():
            expanded.append(p)
        else:
            expanded.append(Path.cwd() / p)
            expanded.append(REPO_ROOT / p)
            expanded.append(p)

    seen: set[str] = set()
    unique: list[Path] = []
    for p in expanded:
        key = str(p.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    for p in unique:
        if p.exists():
            return str(p.resolve())
    raise FileNotFoundError(
        "Could not find checkpoint. Looked for: "
        + ", ".join(str(p.resolve(strict=False)) for p in unique)
    )


def resolve_model_path(explicit_path: str | None = None) -> str:
    return _resolve_candidates(MODEL_CANDIDATES, explicit_path)


def resolve_lstm_model_path(explicit_path: str | None = None) -> str:
    return _resolve_candidates(LSTM_MODEL_CANDIDATES, explicit_path)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# timm compatibility shim (old pickled checkpoints reference timm.models.layers)
# ---------------------------------------------------------------------------
def _register_timm_compat_aliases() -> None:
    if timm is None:
        return
    module_aliases = {
        "timm.models.layers": "timm.layers",
        "timm.models.layers.adaptive_avgmax_pool":  "timm.layers.adaptive_avgmax_pool",
        "timm.models.layers.activations":           "timm.layers.activations",
        "timm.models.layers.activations_jit":       "timm.layers.activations_jit",
        "timm.models.layers.blur_pool":             "timm.layers.blur_pool",
        "timm.models.layers.classifier":            "timm.layers.classifier",
        "timm.models.layers.config":                "timm.layers.config",
        "timm.models.layers.conv2d_same":           "timm.layers.conv2d_same",
        "timm.models.layers.create_act":            "timm.layers.create_act",
        "timm.models.layers.create_attn":           "timm.layers.create_attn",
        "timm.models.layers.create_conv2d":         "timm.layers.create_conv2d",
        "timm.models.layers.create_norm":           "timm.layers.create_norm",
        "timm.models.layers.drop":                  "timm.layers.drop",
        "timm.models.layers.fast_norm":             "timm.layers.fast_norm",
        "timm.models.layers.helpers":               "timm.layers.helpers",
        "timm.models.layers.mixed_conv2d":          "timm.layers.mixed_conv2d",
        "timm.models.layers.norm":                  "timm.layers.norm",
        "timm.models.layers.norm_act":              "timm.layers.norm_act",
        "timm.models.layers.padding":               "timm.layers.padding",
        "timm.models.layers.pool2d_same":           "timm.layers.pool2d_same",
        "timm.models.layers.separable_conv":        "timm.layers.separable_conv",
        "timm.models.layers.space_to_depth":        "timm.layers.space_to_depth",
        "timm.models.layers.split_attn":            "timm.layers.split_attn",
        "timm.models.layers.std_conv":              "timm.layers.std_conv",
        "timm.models.layers.test_time_pool":        "timm.layers.test_time_pool",
        "timm.models.layers.trace_utils":           "timm.layers.trace_utils",
        "timm.models.layers.weight_init":           "timm.layers.weight_init",
    }
    for old, new in module_aliases.items():
        if old in sys.modules:
            continue
        try:
            sys.modules[old] = importlib.import_module(new)
        except ModuleNotFoundError:
            continue


def _torch_load(path: str, device: torch.device) -> Any:
    _register_timm_compat_aliases()
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# ---------------------------------------------------------------------------
# Xception frame-level model
# ---------------------------------------------------------------------------
def _build_fresh_xception(device: torch.device, model_name: str = "xception") -> torch.nn.Module:
    if timm is None:
        raise RuntimeError("timm is required to construct the Xception model.")
    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    return model.to(device)


def load_model(path: str, device: torch.device | None = None) -> tuple[torch.nn.Module, dict[str, Any]]:
    runtime_device = device or get_device()
    checkpoint = _torch_load(path, runtime_device)
    checkpoint_config: dict[str, Any] = {}

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        checkpoint_config = dict(checkpoint.get("config") or {})
        state_dict      = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        checkpoint_model = checkpoint.get("model")

        if isinstance(checkpoint_model, torch.nn.Module):
            model = checkpoint_model
            if state_dict:
                model.load_state_dict(state_dict)
        elif state_dict:
            model_name = checkpoint_config.get("model_name", "xception")
            model = _build_fresh_xception(runtime_device, model_name=model_name)
            cleaned = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned, strict=False)
        else:
            raise RuntimeError("Unsupported checkpoint structure; no model weights found.")
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    model = model.to(runtime_device)
    model.eval()
    _apply_checkpoint_preprocessing_config(checkpoint_config)
    return model, checkpoint_config


# ---------------------------------------------------------------------------
# Xception + LSTM temporal model
# ---------------------------------------------------------------------------
class _TemporalAttentionPooling(nn.Module):
    """Attention pooling over the LSTM output sequence."""
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.attention(x), dim=1)
        return torch.sum(x * w, dim=1), w.squeeze(-1)


class LSTMPredictor(nn.Module):
    """
    Xception backbone + bidirectional LSTM + attention pooling classifier.
    Mirrors the architecture in Train_LSTM.py::XceptionLSTMDetector exactly
    so that the saved state_dict loads without key mismatches.
    """
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__()
        if timm is None:
            raise RuntimeError("timm is required for the LSTM model.")

        model_name  = config.get("model_name", "xception")
        hidden_size = int(config.get("lstm_hidden_size", 256))
        num_layers  = int(config.get("lstm_layers", 2))
        bidirectional = bool(config.get("bidirectional", True))
        dropout     = float(config.get("temporal_dropout", 0.35))
        image_size  = int(config.get("image_size", 299))

        self.image_size = image_size
        self.config     = config

        # Backbone (no pretrained weights — we load from checkpoint)
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=1)
        self.backbone = self.backbone.to(device)

        # Probe backbone feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size, device=device)
            feats = self.backbone.forward_features(dummy)
            if hasattr(self.backbone, "global_pool"):
                feats = self.backbone.global_pool(feats)
            if feats.ndim > 2:
                feats = torch.flatten(feats, 1)
            self.backbone_feature_dim = int(feats.shape[1])

        lstm_input_dim = self.backbone_feature_dim + 1  # features + frame logit
        temporal_dim   = hidden_size * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(
            input_size  = lstm_input_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )
        self.attention_pool = _TemporalAttentionPooling(temporal_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, temporal_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim // 2, 1),
        )
        self.to(device)

    def _backbone_outputs(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone.forward_features(frames)
        if hasattr(self.backbone, "global_pool"):
            features = self.backbone.global_pool(features)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        classifier = getattr(self.backbone, "fc", None)
        if classifier is None:
            raise AttributeError("Xception backbone must expose classifier head as `fc`.")
        return features, classifier(features)

    def forward(self, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = sequences.shape
        flat = sequences.view(B * T, C, H, W)
        feats, logits = self._backbone_outputs(flat)
        temporal_in = torch.cat([feats, logits], dim=1).view(B, T, -1)
        lstm_out, _ = self.lstm(temporal_in)
        pooled, attn = self.attention_pool(lstm_out)
        return self.classifier(pooled), attn


def load_lstm_model(
    path: str,
    device: torch.device | None = None,
) -> tuple[LSTMPredictor, dict[str, Any]]:
    """Load the Xception+LSTM checkpoint and return (model, config)."""
    runtime_device = device or get_device()
    checkpoint = _torch_load(path, runtime_device)

    if not isinstance(checkpoint, dict):
        raise RuntimeError("LSTM checkpoint must be a dict.")

    config     = dict(checkpoint.get("config") or {})
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if not state_dict:
        raise RuntimeError("LSTM checkpoint has no model_state_dict.")

    model = LSTMPredictor(config, runtime_device)
    cleaned = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if unexpected:
        print(f"[LSTM loader] unexpected keys: {unexpected[:5]}")
    allowed_missing = {"fc.weight", "fc.bias"}
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing:
        print(f"[LSTM loader] missing keys: {bad_missing[:5]}")

    model.eval()
    return model, config


# ---------------------------------------------------------------------------
# Runtime initialization
# ---------------------------------------------------------------------------
def initialize_runtime(
    model_path: str | None = None,
    device_name: str | None = None,
    lstm_model_path: str | None = None,
) -> dict[str, str]:
    global MODEL, LSTM_MODEL, LSTM_CONFIG, DEVICE
    global FACE_DETECTOR, FACE_DETECTOR_BACKEND
    global MODEL_PATH, MODEL_CONFIG, LSTM_MODEL_PATH

    DEVICE = get_device(device_name)

    # --- Primary: Xception frame-level (always required) ---
    MODEL_PATH = resolve_model_path(model_path)
    MODEL, MODEL_CONFIG = load_model(MODEL_PATH, DEVICE)
    print(f"[predictor] Xception model loaded: {MODEL_PATH}")

    # --- Optional: Xception+LSTM temporal (primary for video) ---
    LSTM_MODEL = None
    LSTM_CONFIG = None
    LSTM_MODEL_PATH = None
    try:
        lstm_path = resolve_lstm_model_path(
            lstm_model_path or os.getenv("LSTM_MODEL_PATH")
        )
        LSTM_MODEL, LSTM_CONFIG = load_lstm_model(lstm_path, DEVICE)
        LSTM_MODEL_PATH = lstm_path
        print(f"[predictor] LSTM model loaded: {lstm_path}")
    except FileNotFoundError:
        print("[predictor] LSTM checkpoint not found — video will use Xception fallback.")
    except Exception as exc:
        print(f"[predictor] LSTM model failed to load ({exc}) — video will use Xception fallback.")

    # --- Face detector ---
    if MTCNN is None:
        raise RuntimeError("mtcnn is required for face detection but is not installed.")
    FACE_DETECTOR = MTCNN()
    FACE_DETECTOR_BACKEND = "mtcnn"

    return {
        "model_path":            MODEL_PATH,
        "lstm_model_path":       LSTM_MODEL_PATH or "not loaded",
        "device":                str(DEVICE),
        "face_detector_backend": FACE_DETECTOR_BACKEND,
        "image_size":            str(PREPROCESS_IMAGE_SIZE),
        "normalize":             str(PREPROCESS_NORMALIZE),
        "lstm_available":        str(LSTM_MODEL is not None),
    }


# ---------------------------------------------------------------------------
# Getters
# ---------------------------------------------------------------------------
def get_model() -> torch.nn.Module:
    if MODEL is None:
        raise RuntimeError("Model runtime has not been initialized.")
    return MODEL


def get_lstm_model() -> torch.nn.Module | None:
    return LSTM_MODEL


def get_face_detector() -> Any:
    if FACE_DETECTOR is None:
        raise RuntimeError("Face detector runtime has not been initialized.")
    return FACE_DETECTOR


def get_face_detector_backend() -> str:
    if FACE_DETECTOR_BACKEND is None:
        raise RuntimeError("Face detector runtime has not been initialized.")
    return FACE_DETECTOR_BACKEND


def is_lstm_available() -> bool:
    return LSTM_MODEL is not None


# ---------------------------------------------------------------------------
# Image preprocessing (used by Xception frame-level path)
# ---------------------------------------------------------------------------
def preprocess_image(image_rgb: np.ndarray) -> torch.Tensor:
    return PREPROCESS_TRANSFORM(image_rgb).unsqueeze(0).to(DEVICE or get_device())


# ---------------------------------------------------------------------------
# Video frame utilities
# ---------------------------------------------------------------------------
def extract_frames(video_path: str, interval: int) -> list[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(video_path)
    frames: list[tuple[int, np.ndarray]] = []
    index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if index % max(interval, 1) == 0:
                frames.append((index, frame))
            index += 1
    finally:
        capture.release()
    return frames


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------
def detect_face(frame_bgr: np.ndarray) -> dict[str, Any] | None:
    detector  = get_face_detector()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(frame_rgb)
    if not detections:
        return None

    detections = sorted(detections, key=lambda item: item.get("confidence", 0.0), reverse=True)
    best = detections[0]

    x, y, width, height = best["box"]
    x, y = max(0, int(x)), max(0, int(y))
    width, height = max(0, int(width)), max(0, int(height))
    end_x = min(frame_rgb.shape[1], x + width)
    end_y = min(frame_rgb.shape[0], y + height)

    mx = int(width  * FACE_CROP_MARGIN)
    my = int(height * FACE_CROP_MARGIN)
    sx = max(0, x - mx);       sy = max(0, y - my)
    ex = min(frame_rgb.shape[1], end_x + mx)
    ey = min(frame_rgb.shape[0], end_y + my)

    crop = frame_rgb[sy:ey, sx:ex]
    if crop.size == 0:
        return None

    keypoints = best.get("keypoints") or {}
    ordered_points = []
    for key in ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]:
        point = keypoints.get(key)
        if point is not None:
            ordered_points.append([int(point[0] - sx), int(point[1] - sy)])

    return {
        "face_rgb":     crop,
        "frame_rgb":    frame_rgb,
        "box":          [x, y, end_x, end_y],
        "expanded_box": [sx, sy, ex, ey],
        "probability":  float(best.get("confidence", 0.0)),
        "landmarks":    ordered_points or None,
    }


# ---------------------------------------------------------------------------
# Image prediction (Xception frame-level, always)
# ---------------------------------------------------------------------------
def predict_image(
    image_array: np.ndarray,
    threshold: float | None = None,
) -> tuple[str, float]:
    model = get_model()
    decision_threshold = get_default_image_threshold() if threshold is None else threshold
    image_tensor = preprocess_image(image_array)
    with torch.no_grad():
        output = model(image_tensor.float())
        probability = torch.sigmoid(output).item()
    label = "FAKE" if probability > decision_threshold else "REAL"
    return label, float(probability)


# ---------------------------------------------------------------------------
# LSTM video sequence extraction helpers
# ---------------------------------------------------------------------------
def _sample_frame_indices(total_frames: int, sequence_length: int) -> list[int]:
    if total_frames <= 0:
        return [0] * sequence_length
    return np.linspace(
        0, max(total_frames - 1, 0), num=sequence_length, dtype=np.int32
    ).tolist()


def _extract_face_sequence_for_lstm(
    video_path: str,
    sequence_length: int,
    image_size: int,
    face_margin: float = 0.18,
    detector_interval: int = 4,
) -> np.ndarray:
    """Extract a fixed-length sequence of face-cropped frames as uint8 numpy array (T,H,W,3)."""
    from PIL import Image as _PIL

    capture     = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices      = _sample_frame_indices(total_frames, sequence_length)
    faces: list[np.ndarray] = []
    last_box: tuple[int, int, int, int] | None = None
    detector = get_face_detector()

    try:
        for step, frame_idx in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = capture.read()
            if not ok or bgr is None:
                if faces:
                    faces.append(faces[-1].copy())
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            run_det = (last_box is None) or (step % detector_interval == 0)

            if run_det:
                dets = detector.detect_faces(rgb)
                if dets:
                    dets = sorted(dets, key=lambda d: d.get("confidence", 0.0), reverse=True)
                    bx, by, bw, bh = dets[0]["box"]
                    bx, by = max(0, int(bx)), max(0, int(by))
                    bw, bh = max(1, int(bw)), max(1, int(bh))
                    ex, ey = min(rgb.shape[1], bx + bw), min(rgb.shape[0], by + bh)
                    mx = int(bw * face_margin); my = int(bh * face_margin)
                    sx = max(0, bx - mx); sy = max(0, by - my)
                    cex = min(rgb.shape[1], ex + mx); cey = min(rgb.shape[0], ey + my)
                    crop = rgb[sy:cey, sx:cex]
                    if crop.size > 0:
                        last_box = (sx, sy, cex, cey)
                        face = _normalize_face(crop, image_size)
                        faces.append(face)
                        continue

            if last_box is not None:
                sx, sy, ex, ey = last_box
                sx = max(0, min(rgb.shape[1] - 1, sx))
                sy = max(0, min(rgb.shape[0] - 1, sy))
                ex = max(sx + 1, min(rgb.shape[1], ex))
                ey = max(sy + 1, min(rgb.shape[0], ey))
                crop = rgb[sy:ey, sx:ex]
                if crop.size > 0:
                    faces.append(_normalize_face(crop, image_size))
                    continue

            # Skip frame if no face is found
            pass
    finally:
        capture.release()

    if not faces:
        return np.array([], dtype=np.uint8)

    while len(faces) < sequence_length:
        faces.append(faces[-1].copy())

    return np.stack(faces[:sequence_length], axis=0).astype(np.uint8)


def _normalize_face(face_rgb: np.ndarray, target_size: int) -> np.ndarray:
    h, w = face_rgb.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    scale = min(target_size / max(w, 1), target_size / max(h, 1))
    rw = max(1, int(round(w * scale)))
    rh = max(1, int(round(h * scale)))
    resized = cv2.resize(face_rgb, (rw, rh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    ox, oy = (target_size - rw) // 2, (target_size - rh) // 2
    canvas[oy:oy + rh, ox:ox + rw] = resized
    return canvas


# ---------------------------------------------------------------------------
# LSTM video prediction
# ---------------------------------------------------------------------------
def predict_video_lstm(
    video_path: str,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Run the Xception+LSTM model on a video and return a result dict compatible
    with the existing predict_video() output shape.
    """
    lstm     = LSTM_MODEL
    cfg      = LSTM_CONFIG or {}
    device   = DEVICE or get_device()
    seq_len  = int(cfg.get("sequence_length", 10))
    img_size = int(cfg.get("image_size", 299))
    face_margin    = float(cfg.get("face_margin", 0.18))
    det_interval   = int(cfg.get("detector_interval", 4))
    decision_threshold = get_default_image_threshold() if threshold is None else threshold

    if lstm is None:
        raise RuntimeError("LSTM model is not loaded.")

    # Build eval transform matching training
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    frames_np = _extract_face_sequence_for_lstm(
        video_path, seq_len, img_size, face_margin, det_interval
    )  # (T, H, W, 3)  uint8

    if frames_np.size == 0:
        return {
            "prediction":                "INCONCLUSIVE",
            "confidence":                0.0,
            "fake_ratio":                0.0,
            "total_frames_analyzed":     0,
            "frame_confidences":         [],
            "fake_frame_count":          0,
            "fake_frame_indices":        [],
            "frame_fake_threshold":      decision_threshold,
            "fake_ratio_threshold_used": decision_threshold,
            "decision_reason":           "No frames with detectable faces were found.",
            "threshold_based_prediction": "INCONCLUSIVE",
            "analyzed_frame_indices":    [],
            "top_fake_frames":           [],
            "model_used":                "xception_lstm",
            "attention_weights":         [],
        }

    from PIL import Image as _PIL
    tensors = [eval_transform(_PIL.fromarray(f)) for f in frames_np]
    sequence = torch.stack(tensors, dim=0).unsqueeze(0).to(device)  # (1,T,C,H,W)

    with torch.no_grad():
        logits, attn_weights = lstm(sequence)
        probability = float(torch.sigmoid(logits).item())

    prediction = "FAKE" if probability >= decision_threshold else "REAL"
    attn_list  = attn_weights.squeeze(0).cpu().tolist()
    peak_step  = int(np.argmax(attn_list))

    return {
        "prediction":                prediction,
        "confidence":                probability,
        "fake_ratio":                float(probability),
        "total_frames_analyzed":     seq_len,
        "frame_confidences":         attn_list,
        "fake_frame_count":          1 if prediction == "FAKE" else 0,
        "fake_frame_indices":        [peak_step] if prediction == "FAKE" else [],
        "frame_fake_threshold":      decision_threshold,
        "fake_ratio_threshold_used": decision_threshold,
        "decision_reason": (
            f"Xception+LSTM temporal model scored P(fake)={probability:.4f} "
            f"(threshold={decision_threshold}). Peak attention at frame step {peak_step}."
        ),
        "threshold_based_prediction": prediction,
        "analyzed_frame_indices":    list(range(seq_len)),
        "top_fake_frames":           [],
        "model_used":                "xception_lstm",
        "attention_weights":         attn_list,
    }


# ---------------------------------------------------------------------------
# Frame-level video prediction (Xception fallback)
# ---------------------------------------------------------------------------
def predict_video(
    video_path: str,
    frame_interval: int = 10,
    fake_ratio_threshold: float | None = None,
    frame_fake_threshold: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Predict on a video.
    - Uses Xception+LSTM if available (primary, best accuracy).
    - Falls back to Xception frame-by-frame if LSTM is unavailable.
    """
    if is_lstm_available():
        try:
            result = predict_video_lstm(video_path, threshold=frame_fake_threshold)
            result["model_used"] = "xception_lstm"
            return result
        except Exception as exc:
            print(f"[predictor] LSTM inference failed ({exc}), falling back to Xception.")

    # ---- Xception frame-level fallback ----
    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()

    analyzed_frames: list[dict[str, Any]] = []
    sampled_frames   = extract_frames(video_path, frame_interval)
    decision_threshold = get_default_image_threshold() if frame_fake_threshold is None else frame_fake_threshold
    ratio_threshold    = (
        get_default_video_fake_ratio_threshold()
        if fake_ratio_threshold is None else fake_ratio_threshold
    )

    for processed_index, (frame_index, frame_bgr) in enumerate(sampled_frames, start=1):
        detection = detect_face(frame_bgr)
        if detection is None:
            continue

        label, confidence = predict_image(detection["face_rgb"], threshold=decision_threshold)
        analyzed_frames.append(
            {
                "frame_index": frame_index,
                "prediction":  label,
                "confidence":  confidence,
                "frame_rgb":   detection["frame_rgb"],
                "face_rgb":    detection["face_rgb"],
                "landmarks":   detection["landmarks"],
            }
        )

        if progress_callback is not None:
            progress_callback(
                {
                    "frame":      frame_index,
                    "current":    processed_index,
                    "total":      len(sampled_frames) or total_frames,
                    "confidence": confidence,
                }
            )

    if not analyzed_frames:
        return {
            "prediction":                "INCONCLUSIVE",
            "confidence":                0.0,
            "fake_ratio":                0.0,
            "total_frames_analyzed":     0,
            "frame_confidences":         [],
            "fake_frame_count":          0,
            "fake_frame_indices":        [],
            "frame_fake_threshold":      decision_threshold,
            "fake_ratio_threshold_used": ratio_threshold,
            "decision_reason":           "No frames with detectable faces were analyzed.",
            "threshold_based_prediction": "INCONCLUSIVE",
            "top_fake_frames":           [],
            "model_used":                "xception_fallback",
        }

    fake_frames       = [item for item in analyzed_frames if item["prediction"] == "FAKE"]
    fake_count        = len(fake_frames)
    fake_ratio        = fake_count / len(analyzed_frames)
    average_confidence = float(np.mean([item["confidence"] for item in analyzed_frames]))
    fake_frame_indices = [int(item["frame_index"]) for item in fake_frames]
    top_fake_frames    = sorted(fake_frames, key=lambda i: i["confidence"], reverse=True)[:3]
    prediction         = "FAKE" if fake_ratio >= ratio_threshold else "REAL"

    if fake_frame_indices and prediction == "FAKE":
        decision_reason = (
            f"[Xception fallback] FAKE: {fake_count}/{len(analyzed_frames)} frames "
            f"({fake_ratio:.2%}) above threshold. Frames: "
            + ", ".join(str(i) for i in fake_frame_indices)
        )
    elif fake_frame_indices:
        decision_reason = (
            f"[Xception fallback] REAL: only {fake_count}/{len(analyzed_frames)} frames "
            f"({fake_ratio:.2%}) above threshold (< {ratio_threshold:.2%})."
        )
    else:
        decision_reason = "[Xception fallback] REAL: no analyzed frame crossed the fake threshold."

    return {
        "prediction":                prediction,
        "confidence":                average_confidence,
        "fake_ratio":                float(fake_ratio),
        "total_frames_analyzed":     len(analyzed_frames),
        "frame_confidences":         [float(item["confidence"]) for item in analyzed_frames],
        "fake_frame_count":          fake_count,
        "fake_frame_indices":        fake_frame_indices,
        "frame_fake_threshold":      decision_threshold,
        "fake_ratio_threshold_used": ratio_threshold,
        "decision_reason":           decision_reason,
        "threshold_based_prediction": prediction,
        "analyzed_frame_indices":    [int(item["frame_index"]) for item in analyzed_frames],
        "top_fake_frames":           top_fake_frames,
        "model_used":                "xception_fallback",
    }

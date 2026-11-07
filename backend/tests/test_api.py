from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import main


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setitem(main.MODEL_STATUS, "model_loaded", True)
    monkeypatch.setitem(main.MODEL_STATUS, "error", None)
    monkeypatch.setitem(main.MODEL_STATUS, "model_path", "Models/FFPP_model_20_epochs_99acc.pt")

    def fake_analyze(upload, threshold, return_gradcam, return_lime, enable_metadata=True):
        filename = upload.filename or "image.jpg"
        confidence = 0.92 if "fake" in filename else 0.12
        prediction = "FAKE" if confidence > threshold else "REAL"
        return {
            "filename": filename,
            "prediction": prediction,
            "confidence": confidence,
            "label_color": "#ef4444" if prediction == "FAKE" else "#22c55e",
            "gradcam_image": "gradcam-b64" if return_gradcam else None,
            "lime_image": "lime-b64" if return_lime else None,
            "face_landmark_image": "landmarks-b64",
            "face_image": "face-b64",
            "exif_flags": ["Software: Adobe Photoshop"] if "photoshop" in filename else [],
            "processing_time_ms": 12,
        }

    def fake_predict_video(*_args, **_kwargs):
        return {
            "prediction": "FAKE",
            "confidence": 0.74,
            "fake_ratio": 0.52,
            "total_frames_analyzed": 4,
            "frame_confidences": [0.1, 0.8, 0.3, 0.9],
            "analyzed_frame_indices": [0, 10, 20, 30],
            "top_fake_frames": [
                {
                    "frame_index": 30,
                    "confidence": 0.9,
                    "frame_rgb": __import__("numpy").zeros((16, 16, 3), dtype="uint8"),
                    "face_rgb": __import__("numpy").zeros((16, 16, 3), dtype="uint8"),
                }
            ],
        }

    monkeypatch.setattr(main, "_analyze_image_file", fake_analyze)
    monkeypatch.setattr(main, "predict_video", fake_predict_video)
    monkeypatch.setattr(main, "run_gradcam", lambda *_args, **_kwargs: "gradcam-b64")
    monkeypatch.setattr(main, "rgb_to_base64_png", lambda *_args, **_kwargs: "frame-b64")
    monkeypatch.setattr(main, "get_model", lambda: object())
    monkeypatch.setattr(main, "preprocess_image", lambda *_args, **_kwargs: None)

    return TestClient(main.app)


def _open_file(path: Path, content_type: str) -> tuple[str, bytes, str]:
    return (path.name, path.read_bytes(), content_type)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_image_real(client: TestClient, sample_real_image: Path) -> None:
    response = client.post("/predict-image", files={"image": _open_file(sample_real_image, "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["prediction"] == "REAL"


def test_predict_image_fake(client: TestClient, sample_fake_image: Path) -> None:
    response = client.post("/predict-image", files={"image": _open_file(sample_fake_image, "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["prediction"] == "FAKE"


def test_predict_image_returns_gradcam(client: TestClient, sample_fake_image: Path) -> None:
    response = client.post("/predict-image", files={"image": _open_file(sample_fake_image, "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["gradcam_image"] == "gradcam-b64"


def test_predict_image_returns_lime(client: TestClient, sample_fake_image: Path) -> None:
    response = client.post("/predict-image", files={"image": _open_file(sample_fake_image, "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["lime_image"] == "lime-b64"


def test_predict_image_custom_threshold(client: TestClient, sample_real_image: Path) -> None:
    response = client.post(
        "/predict-image",
        files={"image": _open_file(sample_real_image, "image/jpeg")},
        data={"threshold": "0.1"},
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == "FAKE"


def test_batch_predict(client: TestClient, sample_real_image: Path, sample_fake_image: Path) -> None:
    files = [
        ("images", _open_file(sample_real_image, "image/jpeg")),
        ("images", _open_file(sample_fake_image, "image/jpeg")),
        ("images", ("fake-2.jpg", sample_fake_image.read_bytes(), "image/jpeg")),
    ]
    response = client.post("/predict-image-batch", files=files)
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total"] == 3
    assert payload["summary"]["fake_count"] + payload["summary"]["real_count"] == 3


def test_predict_video(client: TestClient, sample_video: Path) -> None:
    response = client.post("/predict-video", files={"video": _open_file(sample_video, "video/mp4")})
    assert response.status_code == 200
    assert 0 <= response.json()["fake_ratio"] <= 1


def test_exif_flags(client: TestClient, photoshop_image: Path) -> None:
    response = client.post("/predict-image", files={"image": _open_file(photoshop_image, "image/jpeg")})
    assert response.status_code == 200
    assert any("Photoshop" in flag for flag in response.json()["exif_flags"])


def test_invalid_file_type(client: TestClient, tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not an image", encoding="utf-8")
    response = client.post("/predict-image", files={"image": _open_file(text_file, "text/plain")})
    assert response.status_code == 422

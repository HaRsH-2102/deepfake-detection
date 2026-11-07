# Deepfake Detection System

This repository now includes a full-stack deepfake detection application built around the existing Xception checkpoints already present in `Models/`.

The implementation adds:

- FastAPI backend for image, batch-image, video, and report endpoints
- Grad-CAM and LIME explainability
- EXIF / metadata forensics
- Face landmark overlays
- WebSocket-driven video progress streaming
- React + Vite forensic dashboard with image, video, batch, and compare modes
- Session history, threshold controls, CSV export, and PDF export
- Pytest API scaffolding and Docker support

## Project Structure

```text
deepfake-detection-XceptionNet-main/
|-- backend/
|   |-- main.py
|   |-- predictor.py
|   |-- explainability.py
|   |-- forensics.py
|   |-- face_landmarks.py
|   |-- report_generator.py
|   |-- requirements.txt
|   `-- tests/
|-- frontend/
|   |-- src/
|   |-- package.json
|   `-- Dockerfile
|-- Models/
|-- docker-compose.yml
`-- run.sh
```

## Backend API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | backend health and model-load status |
| `POST` | `/predict-image` | single-image analysis with Grad-CAM, LIME, landmarks, metadata |
| `POST` | `/predict-image-batch` | batch analysis for up to 10 images |
| `POST` | `/predict-video` | sampled-frame video analysis |
| `POST` | `/generate-report` | PDF export from a result payload |
| `WS` | `/ws/predict-video` | streamed video upload + live progress events |

## Tech Stack

| Layer | Tools |
| --- | --- |
| Model inference | PyTorch, timm, facenet-pytorch, OpenCV |
| Backend | FastAPI, uvicorn, python-multipart |
| Explainability | custom Grad-CAM, LIME |
| Forensics | piexif, Pillow |
| Reports | fpdf2, jsPDF, jsPDF-AutoTable |
| Frontend | React 18, Vite, Tailwind CSS, Framer Motion, Recharts |
| Testing | pytest, httpx, FastAPI TestClient |

## Local Setup

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

On Windows PowerShell, use:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

### 3. Start both with one command

```bash
./run.sh
```

## Environment Files

### `backend/.env`

```env
MODEL_PATH=Models/FFPP_model_20_epochs_99acc.pt
DEVICE=cpu
MAX_VIDEO_SIZE_MB=200
```

### `frontend/.env`

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws/predict-video
```

## Notes About The Model

- The backend keeps the required preprocessing from your notebooks: resize to `128x128`, convert to tensor, and no normalization.
- The loader is resilient to the current checkpoint format and falls back across the existing model files in `Models/`.
- The current default model path points to `Models/FFPP_model_20_epochs_99acc.pt` because this repo does not currently include `Models/deepfake_model.pt`.

## Testing

```bash
cd backend
pytest
```

The included tests focus on API contracts and use deterministic monkeypatching so the suite can run without requiring a full GPU-backed inference environment.

## Docker

```bash
docker compose up --build
```

This starts:

- backend on [http://localhost:8000](http://localhost:8000)
- frontend on [http://localhost:3000](http://localhost:3000)

## Screenshots

Add screenshots of:

- the dashboard landing view
- a completed image analysis
- a video timeline result
- the comparison mode

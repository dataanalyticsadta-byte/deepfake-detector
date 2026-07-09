# deepfake-detector

A prototype realtime deepfake detection app with webcam streaming, image/video upload support, and heuristic report generation.

## Install

```bash
pip install -r requirements.txt
```

## Run

On Windows, use Python to run Uvicorn directly:

```powershell
cd c:\Users\kilbi\.vscode\deepfake-detector
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

On Linux/macOS with `uvicorn` installed in PATH:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Use

- Realtime webcam detection: `http://localhost:8000/static/index.html`
- Upload image/video file: `http://localhost:8000/static/upload.html`

Supported uploads: JPG, JPEG, PNG, WEBP, BMP, TIFF, GIF, MP4, MOV, AVI, MKV, WEBM, FLV, MPEG.

## API

### POST /detect
Upload a single image file and receive a prediction.

### POST /analyze
Upload an image or video file for a prediction and aggressive heuristics report.

### WebSocket /ws
Send JSON messages containing `{"frame": "data:image/jpeg;base64,..."}` to receive realtime predictions.

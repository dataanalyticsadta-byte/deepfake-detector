# deepfake-detector

A prototype realtime deepfake detection app with webcam streaming, image/video upload support, and heuristic report generation.

## Install

### Backend
```bash
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm.cmd install
npm.cmd run build
```

## Run

After building the frontend, start the FastAPI server:

```powershell
cd c:\Users\kilbi\.vscode\deepfake-detector
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open the site at `http://localhost:8000/`.

If you want to run the frontend in development mode instead of the built site:

```powershell
cd frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

## Use

- Web app homepage: `http://localhost:8000/`
- Realtime webcam detection: `http://localhost:8000/webcam`
- Upload image/video file: `http://localhost:8000/upload`
- Auth endpoints: `/api/auth/register` and `/api/auth/login`

Supported uploads: JPG, JPEG, PNG, WEBP, BMP, TIFF, GIF, MP4, MOV, AVI, MKV, WEBM, FLV, MPEG.

## API

### POST /detect
Upload a single image file and receive a prediction.

### POST /analyze
Upload an image or video file for a prediction and aggressive heuristics report.

### WebSocket /ws
Send JSON messages containing `{"frame": "data:image/jpeg;base64,..."}` to receive realtime predictions.

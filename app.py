from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import base64
import json
from io import BytesIO
from PIL import Image

from detector import predict, predict_pil
from reporter import DeepfakeReporter, analyze_video_file

app = FastAPI(title="Deepfake Detector Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif'}
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.mpeg', '.mpg'}

os.makedirs(UPLOAD_DIR, exist_ok=True)
from fastapi.staticfiles import StaticFiles

FRONTEND_DIST = os.path.join('frontend', 'dist')
FRONTEND_INDEX = os.path.join(FRONTEND_DIST, 'index.html')

# Serve legacy static assets from /static
if os.path.isdir('static'):
    app.mount('/static', StaticFiles(directory='static'), name='static')

# Serve built React assets from /assets
if os.path.isdir(FRONTEND_DIST):
    assets_dir = os.path.join(FRONTEND_DIST, 'assets')
    if os.path.isdir(assets_dir):
        app.mount('/assets', StaticFiles(directory=assets_dir), name='frontend_assets')

# Auth router (simple SQLite + JWT scaffolding)
try:
    from auth import router as auth_router
    app.include_router(auth_router, prefix="/api/auth")
except Exception:
    # auth optional if dependencies are missing during initial dev
    pass


def _serve_frontend_index():
    if os.path.exists(FRONTEND_INDEX):
        return FileResponse(FRONTEND_INDEX)
    if os.path.exists(os.path.join('static', 'index.html')):
        return FileResponse(os.path.join('static', 'index.html'))
    raise HTTPException(status_code=404, detail='Frontend not found')


@app.get("/")
def home():
    return _serve_frontend_index()


@app.get('/webcam')
def webcam_page():
    return _serve_frontend_index()


@app.get('/upload')
def upload_page():
    return _serve_frontend_index()


@app.post("/detect")
async def detect(file: UploadFile):
    filepath = os.path.join(UPLOAD_DIR, file.filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict(filepath)

    return result


def _get_upload_type(filename: str):
    lower = os.path.splitext(filename)[1].lower()
    if lower in SUPPORTED_IMAGE_EXTENSIONS:
        return 'image'
    if lower in SUPPORTED_VIDEO_EXTENSIONS:
        return 'video'
    return None


@app.post("/analyze")
async def analyze(file: UploadFile):
    filename = os.path.basename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    upload_type = _get_upload_type(filename)

    if upload_type == 'image' or upload_type is None:
        try:
            image = Image.open(filepath).convert('RGB')
            prediction = predict_pil(image)
            return {
                'type': 'image',
                'filename': filename,
                **prediction,
            }
        except Exception as e:
            if upload_type == 'image':
                return {
                    'error': 'Uploaded image could not be parsed. Supported formats: JPG, PNG, WEBP, BMP, TIFF, GIF.',
                    'detail': str(e)
                }

    if upload_type == 'video' or upload_type is None:
        video_report = analyze_video_file(filepath)
        if video_report.get('summary'):
            return {
                'type': 'video',
                'filename': filename,
                'report': video_report,
            }
        if upload_type == 'video':
            return {'error': 'Uploaded video could not be parsed. Supported formats: MP4, MOV, AVI, MKV, WEBM, FLV, MPEG.'}

    return {'error': 'Unsupported file type. Supported uploads are image and video files, not plain text.'}


@app.get('/{full_path:path}')
def catch_all(full_path: str):
    if full_path.startswith('api/') or full_path.startswith('assets/') or full_path.startswith('static/'):
        raise HTTPException(status_code=404, detail='Not found')
    return _serve_frontend_index()


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    print('WebSocket attempt to connect')
    await websocket.accept()
    print('WebSocket accepted')
    try:
        frame_buffer = []
        MAX_BUFFER = 8  # use last N frames for smoother heuristics
        MIN_FRAMES = 3
        while True:
            data = await websocket.receive_text()
            # Expect JSON: {"frame": "<base64jpeg>"}
            payload = json.loads(data)
            b64 = payload.get('frame')
            if not b64:
                await websocket.send_text(json.dumps({'error': 'no frame provided'}))
                continue

            # strip prefix if present
            if b64.startswith('data:'):
                b64 = b64.split(',', 1)[1]

            try:
                img_bytes = base64.b64decode(b64)
                img = Image.open(BytesIO(img_bytes)).convert('RGB')
            except Exception as e:
                print('WebSocket received invalid image:', e)
                await websocket.send_text(json.dumps({'error': 'invalid image', 'detail': str(e)}))
                continue

            # maintain rolling buffer
            frame_buffer.append(img)
            if len(frame_buffer) > MAX_BUFFER:
                frame_buffer.pop(0)

            # only analyze when we have a few frames for temporal heuristics
            try:
                if len(frame_buffer) >= MIN_FRAMES:
                    from detector import predict_frames
                    result = predict_frames(frame_buffer)
                else:
                    # fallback to single-frame prediction
                    result = predict_pil(img)
            except Exception as e:
                print('Prediction error:', e)
                result = {'error': 'prediction_failed', 'detail': str(e)}

            await websocket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        print('WebSocket disconnected')
        return
    except Exception as e:
        print('WebSocket error:', e)
        try:
            await websocket.close()
        except Exception:
            pass
        return
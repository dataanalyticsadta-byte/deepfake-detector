from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
import shutil
import os
import base64
import json
from io import BytesIO
from PIL import Image

from detector import predict, predict_pil
from reporter import DeepfakeReporter, analyze_video_file

app = FastAPI(title="Deepfake Detector Prototype")

UPLOAD_DIR = "uploads"
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif'}
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.mpeg', '.mpg'}

os.makedirs(UPLOAD_DIR, exist_ok=True)
from fastapi.staticfiles import StaticFiles

# Serve frontend files from /static
if os.path.isdir('static'):
    app.mount('/static', StaticFiles(directory='static'), name='static')


@app.get("/")
def home():
    return {"message": "Deepfake Detector Prototype"}


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
            report = DeepfakeReporter().analyze_frames([image])
            prediction = predict_pil(image)
            return {
                'type': 'image',
                'filename': filename,
                'prediction': prediction,
                'report': report,
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


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
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
                await websocket.send_text(json.dumps({'error': 'invalid image', 'detail': str(e)}))
                continue

            result = predict_pil(img)
            await websocket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        return
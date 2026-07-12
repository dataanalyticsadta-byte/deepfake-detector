Place a pretrained deepfake classifier in this folder to enable learned-model scoring.

Supported filenames:
- deepfake_classifier.h5  (Keras / TensorFlow SavedModel or HDF5)
- deepfake_classifier.onnx (ONNX format)

Recommended setup:
1. Install TensorFlow (for .h5 models) or onnxruntime (for .onnx):
   - TensorFlow: python -m pip install "tensorflow>=2.11"
   - ONNX runtime: python -m pip install onnxruntime

2. Put the model file in the `models/` folder with one of the supported names above.

Notes:
- The loader uses a generic preprocessing (resize to 224x224, scale to [0,1] and ImageNet preprocess when using Keras). If your model expects different preprocessing, either wrap it into a new model that handles preprocessing, or modify `detector._classifier_predict_on_frames` to match your model's expected input.
- If no model is present or required packages are missing, the system falls back to heuristic-only scoring.

Training and download helper
----------------------------
- Train a small MobileNetV2-based classifier (expects dataset/real and dataset/fake):

```bash
python scripts/train.py --data_dir path/to/dataset --epochs 5 --batch_size 32
```

- Download a model from a URL into `models/`:

```bash
python scripts/download_model.py --url https://example.com/model.onnx --out models/deepfake_classifier.onnx
```

Notes:
- Training on a small dataset may produce a weak model; for production accuracy use a larger curated dataset (FaceForensics++, Celeb-DF, etc.) and longer training.
- After placing a model in `models/`, restart the server; `detector` will automatically detect and load the model.

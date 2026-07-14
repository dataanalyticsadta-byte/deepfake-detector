from PIL import Image
import os
import numpy as np

from reporter import DeepfakeReporter

reporter = DeepfakeReporter()

# --- Learned classifier integration (optional) ---
CLASSIFIER = None
CLASSIFIER_TYPE = None
CLASSIFIER_PATH = None
CLASSIFIER_MIX_WEIGHT = 0.65  # weight given to learned model vs heuristics (0-1)


def _try_load_classifier():
    global CLASSIFIER, CLASSIFIER_TYPE, CLASSIFIER_PATH

    def try_transformers_dir(model_dir):
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            processor = AutoImageProcessor.from_pretrained(model_dir)
            model = AutoModelForImageClassification.from_pretrained(model_dir)
            return (model, processor)
        except Exception:
            return None

    # search for a Hugging Face transformer model directory
    if os.path.isdir('models'):
        hf_dir = os.path.join('models', 'hf_model')
        if os.path.isdir(hf_dir):
            cls = try_transformers_dir(hf_dir)
            if cls is not None:
                CLASSIFIER = cls
                CLASSIFIER_TYPE = 'transformers'
                CLASSIFIER_PATH = hf_dir
                return

        for entry in sorted(os.listdir('models')):
            candidate = os.path.join('models', entry)
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, 'config.json')):
                cls = try_transformers_dir(candidate)
                if cls is not None:
                    CLASSIFIER = cls
                    CLASSIFIER_TYPE = 'transformers'
                    CLASSIFIER_PATH = candidate
                    return

    # look for common model filenames
    cand_h5 = os.path.join('models', 'deepfake_classifier.h5')
    cand_onnx = os.path.join('models', 'deepfake_classifier.onnx')
    if os.path.exists(cand_h5):
        try:
            from tensorflow import keras
            model = keras.models.load_model(cand_h5)
            CLASSIFIER = model
            CLASSIFIER_TYPE = 'keras'
            CLASSIFIER_PATH = cand_h5
            return
        except Exception:
            CLASSIFIER = None
    if os.path.exists(cand_onnx):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(cand_onnx)
            CLASSIFIER = sess
            CLASSIFIER_TYPE = 'onnx'
            CLASSIFIER_PATH = cand_onnx
            return
        except Exception:
            CLASSIFIER = None


# try load at import time (best-effort; silent fallback)
try:
    _try_load_classifier()
except Exception:
    CLASSIFIER = None
    CLASSIFIER_TYPE = None


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    return predict_pil(image)


def predict_pil(image):
    image = image.convert("RGB")
    return predict_frames([image])


def _score_from_report(report):
    if report.get('summary') == 'No face detected':
        return 0.0

    values = [v for v in report.get('scores', {}).values() if isinstance(v, (int, float))]
    if not values:
        return 0.0

    values = np.asarray(values, dtype=np.float32)
    max_score = float(np.max(values))
    avg_score = float(np.mean(values))
    strong_scores = values[values >= 0.5]
    strong_mean = float(np.mean(strong_scores)) if strong_scores.size else 0.0
    flagged_ratio = float(np.mean(values >= 0.5))
    or_score = float(1.0 - np.prod(1.0 - values))

    combined = (
        0.35 * max_score +
        0.25 * strong_mean +
        0.15 * avg_score +
        0.15 * flagged_ratio +
        0.10 * or_score
    )

    # Slightly boost moderate suspicious signals so the detector isn't too conservative
    combined = float(np.clip(combined + 0.03 * np.mean(values), 0.0, 1.0))
    return combined


def _classifier_predict_on_frames(frames):
    """Return a classifier score between 0..1 averaged across frames, or None if no classifier."""
    if CLASSIFIER is None:
        return None
    import numpy as np
    try:
        if CLASSIFIER_TYPE == 'transformers':
            model, processor = CLASSIFIER
            inputs = processor(images=frames, return_tensors='pt')
            outputs = model(**inputs)
            logits = outputs.logits
            if logits is None:
                return None
            import torch
            if logits.ndim == 2 and logits.shape[1] == 2:
                probs = torch.softmax(logits, dim=-1)[:, 1]
            else:
                probs = torch.sigmoid(logits.reshape(-1))
            return float(np.clip(float(probs.mean().item()), 0.0, 1.0))
        elif CLASSIFIER_TYPE == 'keras':
            inputs = []
            for p in frames:
                img = p.convert('RGB').resize((224, 224))
                arr = np.array(img).astype('float32') / 255.0
                inputs.append(arr)
            X = np.stack(inputs, axis=0)
            from tensorflow.keras.applications.imagenet_utils import preprocess_input
            Xp = preprocess_input(X.copy())
            preds = CLASSIFIER.predict(Xp)
            if preds.ndim == 2 and preds.shape[1] == 2:
                scores = preds[:, 1]
            else:
                scores = preds.reshape((-1,))
            return float(np.clip(float(scores.mean()), 0.0, 1.0))
        elif CLASSIFIER_TYPE == 'onnx':
            inputs = []
            for p in frames:
                img = p.convert('RGB').resize((224, 224))
                arr = np.array(img).astype('float32') / 255.0
                inputs.append(arr)
            X = np.stack(inputs, axis=0)
            import onnxruntime as ort
            inp_name = CLASSIFIER.get_inputs()[0].name
            ort_inputs = {inp_name: X.astype('float32')}
            out = CLASSIFIER.run(None, ort_inputs)
            preds = out[0]
            if preds.ndim == 2 and preds.shape[1] == 2:
                scores = preds[:, 1]
            else:
                scores = preds.reshape((-1,))
            return float(np.clip(float(scores.mean()), 0.0, 1.0))
    except Exception:
        return None


def predict_frames(pil_frames):
    # Ensure PIL images
    frames = [p.convert('RGB') for p in pil_frames]
    report = reporter.analyze_frames(frames)
    fake_score = _score_from_report(report)

    # get learned-model score if available
    clf_score = _classifier_predict_on_frames(frames)
    if clf_score is not None:
        combined_score = float(np.clip(CLASSIFIER_MIX_WEIGHT * clf_score + (1.0 - CLASSIFIER_MIX_WEIGHT) * fake_score, 0.0, 1.0))
        if clf_score > 0.8:
            combined_score = max(combined_score, float(clf_score))
        elif clf_score > 0.55 and fake_score > 0.35:
            combined_score = max(combined_score, float(fake_score))
    else:
        combined_score = float(fake_score)

    # Use combined score for final verdict when classifier available
    score_for_verdict = combined_score
    if score_for_verdict < 0.35:
        prediction = "Looks Real"
    elif score_for_verdict < 0.58:
        prediction = "Potentially Altered"
    else:
        prediction = "Potential Fake"

    # Overall per-reason percentage scores (0-100)
    reason_scores = {label: round(score * 100, 1) for label, score in report.get('scores', {}).items()}
    reasons = [label for label, score in reason_scores.items() if score > 50]
    reason_details = {
        label: report.get('verdicts', {}).get(label, {}).get('explanation', '')
        for label in reasons
    }

    # Simple overall breakdown between AI (fake) and Real — use combined score when available
    ai_percent = round(combined_score * 100, 1)
    real_percent = round((1.0 - combined_score) * 100, 1)

    return {
        "prediction": prediction,
        "confidence": round(combined_score * 100, 2),
        "fake_score": round(fake_score * 100, 2),
        "classifier_score": None if clf_score is None else round(clf_score * 100, 2),
        "combined_score": round(combined_score * 100, 2),
        "model_loaded": CLASSIFIER is not None,
        "classifier_type": CLASSIFIER_TYPE,
        "breakdown": {"AI": ai_percent, "Real": real_percent},
        "reason_scores": reason_scores,
        "reasons": reasons,
        "reason_details": reason_details,
        "report": report
    }

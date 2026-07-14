from PIL import Image
import numpy as np

from reporter import DeepfakeReporter

reporter = DeepfakeReporter()


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    return predict_pil(image)


def _score_from_report(report):
    if report.get('summary') == 'No face detected':
        return 0.0

    values = [v for v in report.get('scores', {}).values() if isinstance(v, (int, float))]
    if not values:
        return 0.0

    values = np.asarray(values, dtype=np.float32)
    max_score = float(np.max(values))
    avg_score = float(np.mean(values))
    flagged_ratio = float(np.mean(values >= 0.5))
    or_score = float(1.0 - np.prod(1.0 - values))

    combined = (
        0.35 * max_score +
        0.25 * avg_score +
        0.20 * flagged_ratio +
        0.20 * or_score
    )
    return float(np.clip(combined, 0.0, 1.0))


def predict_pil(image):
    image = image.convert("RGB")
    report = reporter.analyze_frames([image])
    fake_score = _score_from_report(report)

    if fake_score < 0.30:
        prediction = "Looks Real"
    elif fake_score < 0.60:
        prediction = "Potentially Altered"
    else:
        prediction = "Potential Fake"

    return {
        "prediction": prediction,
        "confidence": round(fake_score * 100, 2),
        "fake_score": round(fake_score * 100, 2),
        "report": report
    }

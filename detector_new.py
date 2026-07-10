from PIL import Image

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

    return float(sum(values) / len(values))


def predict_pil(image):
    image = image.convert("RGB")
    report = reporter.analyze_frames([image])
    fake_score = _score_from_report(report)

    if fake_score < 0.25:
        prediction = "Looks Real"
    elif fake_score < 0.60:
        prediction = "Potentially Altered"
    else:
        prediction = "Potential Fake"

    return {
        "prediction": prediction,
        "confidence": round(fake_score * 100, 2),
        "report": report
    }

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
    elif fake_score < 0.65:
        prediction = "Potentially Altered"
    else:
        prediction = "Potential Fake"

    reason_scores = {label: round(score * 100, 1) for label, score in report.get('scores', {}).items()}
    reasons = [label for label, score in reason_scores.items() if score > 50]
    reason_details = {
        label: report.get('verdicts', {}).get(label, {}).get('explanation', '')
        for label in reasons
    }

    return {
        "prediction": prediction,
        "confidence": round(fake_score * 100, 2),
        "reason_scores": reason_scores,
        "reasons": reasons,
        "reason_details": reason_details,
        "report": report
    }

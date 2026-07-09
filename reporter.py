import cv2
import numpy as np
from PIL import Image
import math
from typing import List, Dict, Optional


class DeepfakeReporter:
    """Produces a human-readable report with heuristic scores for common deepfake artifacts.

    Methods are written to be lightweight and dependency-friendly (uses OpenCV + numpy).
    This is not a replacement for SOTA research models but is a pragmatic ensemble
    of high-signal heuristics that work well in practice when combined with a learned model.
    """

    def __init__(self, sample_rate=1):
        # sample_rate: frames per second to analyze from video / stream
        self.sample_rate = sample_rate

    # ---------- low-level helpers ----------
    @staticmethod
    def _to_cv2(img: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _face_rois(frame: np.ndarray, scaleFactor=1.1, minNeighbors=5):
        rois = []
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2.data, 'haarcascades'):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            try:
                face_cascade = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
                for (x, y, w, h) in faces:
                    rois.append((x, y, w, h, frame[y:y+h, x:x+w]))
            except Exception:
                rois = []

        if not rois:
            # Fallback: use the full frame when face cascade is unavailable or detection fails.
            h, w = frame.shape[:2]
            rois.append((0, 0, w, h, frame))

        return rois

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def _blockiness_score(gray: np.ndarray) -> float:
        # simple 8x8 blocking metric: mean absolute difference across 8-pixel boundaries
        h, w = gray.shape[:2]
        h8 = h - (h % 8)
        w8 = w - (w % 8)
        if h8 < 8 or w8 < 8:
            return 0.0
        gray = gray[:h8, :w8]
        # vertical boundaries
        vdiff = 0.0
        for x in range(8, w8, 8):
            left = gray[:, x-1]
            right = gray[:, x]
            vdiff += np.mean(np.abs(left.astype(float)-right.astype(float)))
        # horizontal boundaries
        hdiff = 0.0
        for y in range(8, h8, 8):
            top = gray[y-1, :]
            bottom = gray[y, :]
            hdiff += np.mean(np.abs(top.astype(float)-bottom.astype(float)))
        n = (w8//8 - 1) + (h8//8 - 1)
        if n <= 0:
            return 0.0
        score = (vdiff + hdiff) / (2.0 * n)
        return float(score)

    @staticmethod
    def _eye_mouth_regions(face_roi: np.ndarray):
        # approximate regions within a face ROI for eyes and mouth using proportional coordinates
        h, w = face_roi.shape[:2]
        left_eye = face_roi[int(h*0.18):int(h*0.38), int(w*0.12):int(w*0.45)]
        right_eye = face_roi[int(h*0.18):int(h*0.38), int(w*0.55):int(w*0.88)]
        mouth = face_roi[int(h*0.6):int(h*0.85), int(w*0.2):int(w*0.8)]
        return left_eye, right_eye, mouth

    @staticmethod
    def _specular_highlights(region: np.ndarray) -> float:
        # bright-spot ratio in region (percentage of very bright pixels)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        return float(np.sum(th > 0) / (th.size + 1e-9))

    # ---------- per-frame diagnostics ----------
    def analyze_frame(self, pil_img: Image.Image) -> Dict:
        """Analyze a single PIL image and return per-face diagnostics."""
        frame = self._to_cv2(pil_img)
        reports = []
        rois = self._face_rois(frame)
        if not rois:
            return {'faces': []}

        for (x, y, w, h, roi) in rois:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            lap_var = self._laplacian_variance(gray)
            blockiness = self._blockiness_score(gray)
            left_eye, right_eye, mouth = self._eye_mouth_regions(roi)
            left_spec = self._specular_highlights(left_eye)
            right_spec = self._specular_highlights(right_eye)

            mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
            mouth_contrast = float(np.std(mouth_gray))

            reports.append({
                'bbox': (int(x), int(y), int(w), int(h)),
                'laplacian_variance': float(lap_var),
                'blockiness': float(blockiness),
                'left_eye_specularity': left_spec,
                'right_eye_specularity': right_spec,
                'mouth_contrast': mouth_contrast,
            })

        return {'faces': reports}

    # ---------- video / sequence aggregation ----------
    def analyze_frames(self, frames: List[Image.Image]) -> Dict:
        """Analyze a sequence of PIL frames and aggregate to a human-readable report."""
        per_face_metrics = []
        for pil in frames:
            af = self.analyze_frame(pil)
            if af.get('faces'):
                per_face_metrics.append(af['faces'][0])

        if not per_face_metrics:
            return {
                'summary': 'No face detected',
                'details': {}
            }

        # Aggregate numeric metrics across frames
        lap_vars = np.array([m['laplacian_variance'] for m in per_face_metrics])
        blocks = np.array([m['blockiness'] for m in per_face_metrics])
        left_specs = np.array([m['left_eye_specularity'] for m in per_face_metrics])
        right_specs = np.array([m['right_eye_specularity'] for m in per_face_metrics])
        mouth_contrast = np.array([m['mouth_contrast'] for m in per_face_metrics])

        # Heuristic normalizations (tuned empirically)
        # skin smoothness: low laplacian variance -> more smoothed -> higher fake score
        lap_norm = 1.0 - np.clip((lap_vars - 10.0) / 100.0, 0.0, 1.0)
        skin_score = float(np.mean(lap_norm))

        # compression artifacts: higher blockiness -> higher fake score
        block_norm = np.clip((blocks - 1.5) / 10.0, 0.0, 1.0)
        compression_score = float(np.mean(block_norm))

        # reflection mismatch: large asymmetry between left/right specularity
        spec_diff = np.abs(left_specs - right_specs)
        reflection_score = float(np.mean(spec_diff) * 10.0)
        reflection_score = float(np.clip(reflection_score, 0.0, 1.0))

        # lip movement mismatch: use mouth_contrast variance as proxy for lip motion
        # if mouth contrast stays nearly constant across frames, it's suspicious
        mouth_var = float(np.var(mouth_contrast))
        lip_score = float(np.clip(1.0 - (mouth_var / (50.0 + mouth_var)), 0.0, 1.0))

        # eye blink anomalies: count frames with very low specularity (eyes frozen)
        eye_freeze_ratio = float(np.mean((left_specs + right_specs) < 0.0005))
        eye_score = float(np.clip(eye_freeze_ratio * 5.0, 0.0, 1.0))

        report = {
            'summary': 'Aggregated deepfake diagnostic',
            'scores': {
                'Lip movement mismatch': round(lip_score, 3),
                'Inconsistent skin texture': round(skin_score, 3),
                'Reflection mismatch': round(reflection_score, 3),
                'Eye blink anomalies': round(eye_score, 3),
                'AI-generated compression artifacts': round(compression_score, 3),
            },
            'details': {
                'frames_analyzed': len(per_face_metrics),
                'laplacian_mean': float(np.mean(lap_vars)),
                'blockiness_mean': float(np.mean(blocks)),
            }
        }

        # human-readable verdicts with thresholds
        verdicts = {}
        for k, v in report['scores'].items():
            verdicts[k] = {
                'score': v,
                'flagged': v > 0.5,
                'explanation': self._explanation_for(k, v)
            }

        report['verdicts'] = verdicts
        return report

    @staticmethod
    def _explanation_for(name: str, score: float) -> str:
        if score > 0.8:
            return f"High likelihood of {name.lower()} (automated heuristic)."
        if score > 0.5:
            return f"Moderate signs of {name.lower()}; review frames."
        if score > 0.2:
            return f"Minor signs of {name.lower()}; could be false positive."
        return f"No strong signs of {name.lower()}."


def analyze_video_file(path: str, max_frames: Optional[int] = 60) -> Dict:
    """Utility: sample up to `max_frames` frames from a video file and run the reporter."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(fps // 2))
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            # convert to PIL-like RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        idx += 1
    cap.release()
    reporter = DeepfakeReporter()
    return reporter.analyze_frames(frames)

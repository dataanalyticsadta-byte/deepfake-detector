import cv2
import numpy as np
from PIL import Image
import math
from typing import List, Dict, Optional

# Try to use MediaPipe Face Mesh for robust face localization and landmarks
try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_face_mesh = mp.solutions.face_mesh
except Exception:
    MP_AVAILABLE = False


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
        # Prefer MediaPipe Face Mesh if available for accurate landmarks and bounding boxes
        if MP_AVAILABLE:
            try:
                h, w = frame.shape[:2]
                with mp_face_mesh.FaceMesh(static_image_mode=True) as fm:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = fm.process(img_rgb)
                    if results and results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            # compute bounding box from landmarks
                            xs = [lm.x for lm in face_landmarks.landmark]
                            ys = [lm.y for lm in face_landmarks.landmark]
                            x_min = int(max(0, min(xs) * w))
                            x_max = int(min(w, max(xs) * w))
                            y_min = int(max(0, min(ys) * h))
                            y_max = int(min(h, max(ys) * h))
                            # expand slightly
                            pad_x = int((x_max - x_min) * 0.12)
                            pad_y = int((y_max - y_min) * 0.16)
                            x1 = max(0, x_min - pad_x)
                            y1 = max(0, y_min - pad_y)
                            x2 = min(w, x_max + pad_x)
                            y2 = min(h, y_max + pad_y)
                            roi = frame[y1:y2, x1:x2]
                            rois.append((x1, y1, x2 - x1, y2 - y1, roi, face_landmarks))
            except Exception:
                rois = []

        # Fallback to Haar cascade if MediaPipe missing or failed
        if not rois and hasattr(cv2, 'CascadeClassifier') and hasattr(cv2.data, 'haarcascades'):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            try:
                face_cascade = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
                for (x, y, w, h) in faces:
                    rois.append((x, y, w, h, frame[y:y+h, x:x+w], None))
            except Exception:
                rois = []

        if not rois:
            # Fallback: use the full frame when face detection fails.
            h, w = frame.shape[:2]
            rois.append((0, 0, w, h, frame, None))

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
    def _landmark_eye_aspect_ratio(landmarks, indices, bbox):
        # landmarks: mediapipe face_landmarks, indices: list of idx for eye contour
        # bbox: (x,y,w,h) of ROI in original image for converting normalized coords
        h, w = bbox[3], bbox[2]
        # convert normalized MP coords to pixel coords within bbox
        pts = []
        for idx in indices:
            lm = landmarks.landmark[idx]
            px = int((lm.x * w))
            py = int((lm.y * h))
            pts.append((px, py))
        if len(pts) < 6:
            return 0.0
        # EAR formula: (|p2-p6| + |p3-p5|) / (2*|p1-p4|)
        p1, p2, p3, p4, p5, p6 = pts[0:6]
        def dist(a,b):
            return math.hypot(a[0]-b[0], a[1]-b[1])
        ear = (dist(p2,p6) + dist(p3,p5)) / (2.0 * (dist(p1,p4) + 1e-9))
        return float(ear)

    @staticmethod
    def _landmark_mouth_opening(landmarks, top_idx, bottom_idx, bbox):
        h, w = bbox[3], bbox[2]
        lt = landmarks.landmark[top_idx]
        lb = landmarks.landmark[bottom_idx]
        top = (int(lt.x * w), int(lt.y * h))
        bot = (int(lb.x * w), int(lb.y * h))
        return float(math.hypot(top[0]-bot[0], top[1]-bot[1]))

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

        for (x, y, w, h, roi, landmarks) in rois:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            lap_var = self._laplacian_variance(gray)
            blockiness = self._blockiness_score(gray)
            left_eye, right_eye, mouth = self._eye_mouth_regions(roi)
            left_spec = self._specular_highlights(left_eye)
            right_spec = self._specular_highlights(right_eye)

            mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
            mouth_contrast = float(np.std(mouth_gray))

            face_report = {
                'bbox': (int(x), int(y), int(w), int(h)),
                'laplacian_variance': float(lap_var),
                'blockiness': float(blockiness),
                'left_eye_specularity': left_spec,
                'right_eye_specularity': right_spec,
                'mouth_contrast': mouth_contrast,
            }

            # If we have MediaPipe landmarks, compute EAR and mouth opening
            if landmarks is not None and MP_AVAILABLE:
                try:
                    # indices for key eye and mouth landmarks (MediaPipe face mesh)
                    left_eye_idx = [33, 160, 158, 133, 153, 144]
                    right_eye_idx = [263, 387, 385, 362, 380, 373]
                    mouth_top = 13
                    mouth_bottom = 14

                    ear_l = self._landmark_eye_aspect_ratio(landmarks, left_eye_idx, (x, y, w, h))
                    ear_r = self._landmark_eye_aspect_ratio(landmarks, right_eye_idx, (x, y, w, h))
                    mouth_open = self._landmark_mouth_opening(landmarks, mouth_top, mouth_bottom, (x, y, w, h))

                    face_report.update({
                        'landmark_ear_left': ear_l,
                        'landmark_ear_right': ear_r,
                        'landmark_mouth_open': mouth_open,
                        'landmarks_present': True,
                    })
                except Exception:
                    face_report['landmarks_present'] = False
            else:
                face_report['landmarks_present'] = False

            reports.append(face_report)

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
        # If landmarks provided, prefer landmark mouth variance across frames
        try:
            mouth_opens = np.array([m.get('landmark_mouth_open', np.nan) for m in per_face_metrics])
            if np.isfinite(mouth_opens).sum() >= 2:
                mvar = float(np.nanvar(mouth_opens))
                lip_score = float(np.clip(1.0 - (mvar / (25.0 + mvar)), 0.0, 1.0))
        except Exception:
            pass

        # eye blink anomalies: count frames with very low specularity (eyes frozen)
        eye_freeze_ratio = float(np.mean((left_specs + right_specs) < 0.0005))
        eye_score = float(np.clip(eye_freeze_ratio * 5.0, 0.0, 1.0))
        # If landmark EARs exist, use variance in EAR across frames as a blink proxy
        try:
            ears_l = np.array([m.get('landmark_ear_left', np.nan) for m in per_face_metrics])
            ears_r = np.array([m.get('landmark_ear_right', np.nan) for m in per_face_metrics])
            valid = np.isfinite(ears_l) & np.isfinite(ears_r)
            if np.sum(valid) >= 2:
                ear_var = float(np.nanvar((ears_l[valid] + ears_r[valid]) / 2.0))
                # higher variance in EAR -> natural blinking -> lower fake score
                eye_score = float(np.clip(1.0 - (ear_var * 50.0), 0.0, 1.0))
        except Exception:
            pass

        frame_count = len(per_face_metrics)
        if frame_count < 2:
            lip_score = 0.0
            eye_score = 0.0

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
                'frames_analyzed': frame_count,
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

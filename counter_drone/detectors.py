"""Per-sensor target detectors.

Two classical detectors are provided that need no training data and work for
small (a few to a few dozen pixel) aerial targets against sky:

* ``RGBSmallTargetDetector`` - morphological top-hat / black-hat saliency
  (small bright *or* dark blobs vs. a smooth sky), adaptive CFAR-style
  threshold on robust noise, plus a running-background motion cue.
* ``ThermalHotspotDetector`` - white top-hat on the AGC-normalised thermal
  image: drones' motors, ESCs and battery are warmer than the sky.

``UltralyticsDetector`` wraps a YOLO model (optional dependency) for either
sensor when you have trained weights; it returns the same ``Detection``
objects so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import RGBDetectorConfig, ThermalDetectorConfig
from .types import Detection


def _noise_sigma(img: np.ndarray) -> float:
    """Sensor noise from the high-pass residual (MAD -> sigma); robust to targets and clouds."""
    res = img - cv2.GaussianBlur(img, (0, 0), 2.0)
    s = res[::2, ::2].ravel()
    return float(1.4826 * np.median(np.abs(s - np.median(s)))) + 1e-6


def _cfar_threshold(sal: np.ndarray, noise: float, k_sigma: float, k_local: float, win: int,
                    floor: float) -> np.ndarray:
    """Per-pixel threshold = max(global noise floor, local mean + k_local * local std).

    The local term raises the bar inside cloud edges / textured clutter while
    leaving clean sky at the sensor-noise limit (a censored cell-averaging CFAR).
    """
    glob = max(float(np.median(sal)) + k_sigma * noise, floor)
    # censored CFAR: clip strong returns before estimating local clutter so a
    # bright target does not mask a neighbouring one (two drones flying close)
    cens = np.minimum(sal, 2.0 * glob)
    mu = cv2.blur(cens, (win, win))
    var = np.maximum(cv2.blur(cens * cens, (win, win)) - mu * mu, 0.0)
    return np.maximum(glob, mu + k_local * np.sqrt(var))


def _roi_mask(shape, roi) -> np.ndarray | None:
    if list(roi) == [0.0, 0.0, 1.0, 1.0]:
        return None
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
    m[y1:y2, x1:x2] = 1
    return m


def _blobs(saliency, threshold, min_area, max_area, max_aspect, sigma, merge_px=1, border_px=0):
    """Connected components above threshold (scalar or per-pixel map) -> list of (box, peak_sigmas).

    Pixels within ``merge_px`` of each other are grouped into one target, so a
    drone whose motors show up as separate hot spots (or whose thin arms vanish
    at low resolution) yields one detection, not several.
    """
    binary = (saliency > threshold).astype(np.uint8)
    k = 2 * merge_px + 1
    grouped = cv2.dilate(binary, np.ones((k, k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        sub = (labels[y:y + h, x:x + w] == i) & (binary[y:y + h, x:x + w] > 0)
        area = int(sub.sum())
        if not (min_area <= area <= max_area):
            continue
        ys, xs = np.nonzero(sub)
        x1, x2, y1, y2 = x + xs.min(), x + xs.max() + 1, y + ys.min(), y + ys.max() + 1
        H, W = saliency.shape
        if x1 < border_px or y1 < border_px or x2 > W - border_px or y2 > H - border_px:
            continue  # morphology is unreliable at the image edge
        bw, bh = x2 - x1, y2 - y1
        if max(bw, bh) / max(1, min(bw, bh)) > max_aspect:
            continue
        peak = float(saliency[y:y + h, x:x + w][sub].max())
        out.append((np.array([x1, y1, x2, y2], float), peak / sigma))
    return out


def _score_from_contrast(c_sigma: float, k_sigma: float) -> float:
    # logistic: k_sigma -> ~0.5, 2*k_sigma -> ~0.95
    return float(1.0 / (1.0 + np.exp(-3.0 * (c_sigma / k_sigma - 1.0))))


def rgb_usable(frame_bgr: np.ndarray | None, min_mean: float = 25.0) -> bool:
    """Is the RGB image bright enough for silhouette detection (vs. night / heavy dusk)?"""
    if frame_bgr is None:
        return False
    return float(frame_bgr[::8, ::8].mean()) >= min_mean


class RGBSmallTargetDetector:
    def __init__(self, cfg: RGBDetectorConfig):
        self.cfg = cfg
        k = cfg.kernel_px | 1
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.bg: np.ndarray | None = None

    def reset(self):
        self.bg = None

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        cfg = self.cfg
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        gray = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)

        # small dark targets on bright sky (black-hat) and small bright ones (top-hat)
        sal = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.kernel)
        if cfg.polarity == "both":
            sal = np.maximum(sal, cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self.kernel))
        elif cfg.polarity == "bright":
            sal = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self.kernel)
        mask = _roi_mask(sal.shape, cfg.roi)
        if mask is not None:
            sal *= mask

        # motion cue from a running background (helps rank, not required: drones hover)
        if self.bg is None or self.bg.shape != gray.shape:
            self.bg = gray.copy()
        motion = np.abs(gray - self.bg)
        cv2.accumulateWeighted(gray, self.bg, cfg.bg_alpha)

        sigma = _noise_sigma(gray)
        thr = _cfar_threshold(sal, sigma, cfg.k_sigma, cfg.k_local, cfg.cfar_win, cfg.min_abs_contrast)
        msigma = sigma
        dets = []
        for box, c in _blobs(sal, thr, cfg.min_area, cfg.max_area, cfg.max_aspect, sigma, cfg.merge_px,
                              cfg.border_px):
            x1, y1, x2, y2 = box.astype(int)
            moving = float(motion[y1:y2, x1:x2].max()) > 6 * msigma + 3
            score = min(1.0, _score_from_contrast(c, cfg.k_sigma) + (cfg.motion_bonus if moving else 0.0))
            dets.append(Detection(box=box, score=score, source="rgb", rgb_score=score,
                                  contrast=c, moving=moving))
        return dets


def normalize_thermal(frame: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.9) -> np.ndarray:
    """Percentile AGC: any thermal input (8-bit, 16-bit radiometric Y16, float) -> float32 [0,1]."""
    f = frame.astype(np.float32)
    if f.ndim == 3:
        f = f.mean(axis=2)
    lo, hi = np.percentile(f[::2, ::2], [low_pct, high_pct])
    if hi - lo < 1e-6:
        return np.zeros_like(f)
    return np.clip((f - lo) / (hi - lo), 0.0, 1.0)


class ThermalHotspotDetector:
    def __init__(self, cfg: ThermalDetectorConfig):
        self.cfg = cfg
        k = cfg.kernel_px | 1
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def detect(self, thermal_norm: np.ndarray) -> list[Detection]:
        """``thermal_norm`` is the output of :func:`normalize_thermal`."""
        cfg = self.cfg
        t = cv2.GaussianBlur(thermal_norm, (3, 3), 0)
        sal = cv2.morphologyEx(t, cv2.MORPH_TOPHAT, self.kernel)
        if cfg.detect_cold:
            sal = np.maximum(sal, cv2.morphologyEx(t, cv2.MORPH_BLACKHAT, self.kernel))
        mask = _roi_mask(sal.shape, cfg.roi)
        if mask is not None:
            sal *= mask
        sigma = _noise_sigma(t)
        thr = _cfar_threshold(sal, sigma, cfg.k_sigma, cfg.k_local, cfg.cfar_win, cfg.min_abs_contrast)
        dets = []
        for box, c in _blobs(sal, thr, cfg.min_area, cfg.max_area, 6.0, sigma, cfg.merge_px, cfg.border_px):
            score = _score_from_contrast(c, cfg.k_sigma)
            dets.append(Detection(box=box, score=score, source="thermal", thermal_score=score,
                                  contrast=c, thermal_contrast=c))
        return dets


class UltralyticsDetector:
    """YOLO (ultralytics) backend for either sensor. Optional dependency."""

    def __init__(self, weights: str, conf: float, drone_classes: list[str], source: str, imgsz: int = 640,
                 device: str = "cpu"):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:  # pragma: no cover - optional
            raise ImportError("backend 'yolo' needs `pip install ultralytics`") from e
        if not weights:
            raise ValueError("yolo backend requires yolo_weights")
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.drone_classes = {c.lower() for c in drone_classes}
        self.source = source

    def detect(self, image: np.ndarray) -> list[Detection]:  # pragma: no cover - optional
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        res = self.model.predict(image, conf=self.conf, imgsz=self.imgsz, device=self.device, max_det=50,
                                 verbose=False)[0]
        names = res.names
        dets = []
        for xyxy, conf, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                   res.boxes.cls.cpu().numpy().astype(int)):
            label = str(names[cls]).lower()
            label = "drone" if label in self.drone_classes else label
            score = float(conf)
            d = Detection(box=xyxy.astype(float), score=score, source=self.source, label=label,
                          contrast=10.0 * score)
            if self.source == "rgb":
                d.rgb_score = score
            else:
                d.thermal_score, d.thermal_contrast = score, 10.0 * score
            dets.append(d)
        return dets


class HybridDetector:
    """Classical detector for recall on tiny targets + YOLO for class identity.

    Classical blobs that coincide with a YOLO box inherit its class label (and the
    higher score); YOLO boxes with no classical counterpart are added (e.g. targets
    against ground clutter or too large for the morphology kernel). Classical blobs
    YOLO does not recognise are kept unlabeled and left to the tracker/classifier.
    """

    def __init__(self, classical, yolo: UltralyticsDetector, gate_px: float = 10.0, standalone_conf: float = 0.5,
                 label_conf: float = 0.4):
        self.classical, self.yolo, self.gate_px = classical, yolo, gate_px
        self.label_conf = label_conf            # min YOLO confidence to hand its class to a classical blob
        self.standalone_conf = standalone_conf  # YOLO-only boxes need more confidence than corroborated ones

    def detect(self, image) -> list[Detection]:
        cls_d, yolo_d = self.classical.detect(image), self.yolo.detect(image)
        used = set()
        for d in cls_d:
            best, bd = None, 1e9
            for j, y in enumerate(yolo_d):
                dist = float(np.linalg.norm(d.center - y.center))
                if j not in used and dist < max(self.gate_px, 0.75 * max(y.wh)) and dist < bd:
                    best, bd = j, dist
            if best is not None:
                y = yolo_d[best]
                used.add(best)
                if y.score < self.label_conf:
                    continue
                d.label = y.label
                d.score = max(d.score, y.score)
                if y.source == "rgb":
                    d.rgb_score = d.score
                else:
                    d.thermal_score = d.score
        return cls_d + [y for j, y in enumerate(yolo_d) if j not in used and y.score >= self.standalone_conf]


def _build(cfg, classical_cls, source):
    if cfg.backend == "classical":
        return classical_cls(cfg)
    y = UltralyticsDetector(cfg.yolo_weights, cfg.yolo_conf, cfg.yolo_drone_classes, source, cfg.yolo_imgsz,
                            cfg.yolo_device)
    if cfg.backend == "yolo":
        return y
    if cfg.backend == "hybrid":
        return HybridDetector(classical_cls(cfg), y, standalone_conf=cfg.yolo_standalone_conf,
                              label_conf=cfg.yolo_label_conf)
    raise ValueError(f"unknown detector backend {cfg.backend!r}")


def build_rgb_detector(cfg: RGBDetectorConfig):
    return _build(cfg, RGBSmallTargetDetector, "rgb")


def build_thermal_detector(cfg: ThermalDetectorConfig):
    return _build(cfg, ThermalHotspotDetector, "thermal")

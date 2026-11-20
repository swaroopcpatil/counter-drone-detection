"""Late (detection-level) fusion of RGB and thermal detections.

Thermal detections are mapped into the RGB frame through the calibrated
homography and paired with RGB detections by optimal assignment on centre
distance. Paired detections get a noisy-OR combined score and the sharper RGB
box; single-sensor detections are kept but down-weighted (an RGB-only blob
*inside* the thermal FOV is suspicious - most drones are warm - while a
thermal-only blob is how we still see drones at night / in glare).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .calibration import map_boxes
from .config import FusionConfig
from .types import Detection


class Fuser:
    def __init__(self, cfg: FusionConfig, H_thermal_to_rgb: np.ndarray, thermal_size: tuple[int, int]):
        self.cfg = cfg
        self.H = np.asarray(H_thermal_to_rgb, float)
        w, h = thermal_size
        c = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
        self.fov = cv2.perspectiveTransform(c.astype(np.float64), self.H).astype(np.float32)

    def in_thermal_fov(self, pt) -> bool:
        return cv2.pointPolygonTest(self.fov, (float(pt[0]), float(pt[1])), False) >= 0

    def fuse(self, rgb: list[Detection], thermal: list[Detection], rgb_ok: bool = True,
             thermal_ok: bool = True) -> list[Detection]:
        """``thermal_ok=False`` (no thermal frame this cycle): RGB detections are not penalised
        or marked as 'inside the thermal FOV', since thermal could not have confirmed them."""
        cfg = self.cfg
        mapped = map_boxes(np.array([d.box for d in thermal]), self.H) if thermal else np.zeros((0, 4))
        th = []
        for d, box in zip(thermal, mapped):
            th.append(Detection(box=box, score=d.score, source="thermal", thermal_score=d.thermal_score,
                                contrast=d.contrast, thermal_contrast=d.thermal_contrast, label=d.label))

        pairs = []
        if rgb and th:
            rc = np.array([d.center for d in rgb])
            tc = np.array([d.center for d in th])
            dist = np.linalg.norm(rc[:, None] - tc[None], axis=2)
            # gate grows with target size: big close targets can be offset more (parallax, sync)
            size = np.array([max(d.wh) for d in rgb])[:, None]
            gate = np.maximum(cfg.gate_px, 0.75 * size)
            cost = np.where(dist <= gate, dist / gate, 1e6)
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e5:
                    pairs.append((i, j))

        out: list[Detection] = []
        used_r = {i for i, _ in pairs}
        used_t = {j for _, j in pairs}
        for i, j in pairs:
            r, t = rgb[i], th[j]
            score = 1.0 - (1.0 - r.rgb_score) * (1.0 - t.thermal_score)
            out.append(Detection(box=r.box.copy(), score=score, source="fused", rgb_score=r.rgb_score,
                                 thermal_score=t.thermal_score, contrast=r.contrast,
                                 thermal_contrast=t.thermal_contrast, in_thermal_fov=True,
                                 moving=r.moving, label=r.label or t.label))
        for i, r in enumerate(rgb):
            if i in used_r:
                continue
            inside = thermal_ok and self.in_thermal_fov(r.center)
            f = cfg.rgb_only_in_thermal_fov_factor if inside else 1.0
            out.append(Detection(box=r.box.copy(), score=r.score * f, source="rgb", rgb_score=r.rgb_score,
                                 contrast=r.contrast, in_thermal_fov=inside, moving=r.moving, label=r.label))
        for j, t in enumerate(th):
            if j in used_t:
                continue
            t.score *= cfg.thermal_only_factor
            out.append(t)
        for d in out:
            d.rgb_ok = rgb_ok
            if d.source == "thermal":
                d.in_thermal_fov = True
        return [d for d in out if d.score >= cfg.min_score]

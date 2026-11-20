"""Operator display: annotated RGB view + false-colour thermal view side by side."""

from __future__ import annotations

import cv2
import numpy as np

from .calibration import thermal_fov_polygon
from .pipeline import FrameResult

THREAT_COLORS = {"NONE": (160, 160, 160), "LOW": (0, 220, 255), "MEDIUM": (0, 140, 255), "HIGH": (0, 0, 255)}


def _put(img, text, org, color, scale=0.45, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render(rgb: np.ndarray, res: FrameResult, H_t2r: np.ndarray, thermal_size, show_dets=True,
           fps: float | None = None, has_thermal: bool = True) -> np.ndarray:
    """Annotated RGB view, plus the false-colour thermal panel when a thermal camera is used."""
    view = rgb.copy() if rgb is not None else np.zeros((540, 960, 3), np.uint8)
    if has_thermal:
        poly = thermal_fov_polygon(H_t2r, thermal_size).astype(np.int32)
        cv2.polylines(view, [poly], True, (255, 0, 255), 1, cv2.LINE_AA)
        _put(view, "thermal FOV", tuple(poly[0] + [4, 14]), (255, 0, 255), 0.4)

    if show_dets:
        for d in res.fused_dets:
            x1, y1, x2, y2 = d.box.astype(int)
            col = {"fused": (0, 255, 0), "rgb": (255, 200, 0), "thermal": (255, 0, 255)}[d.source]
            cv2.rectangle(view, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), col, 1)

    for s in res.tracks:
        if not s.confirmed:
            continue
        col = THREAT_COLORS[s.threat] if s.drone_prob >= 0.45 else (200, 200, 200)
        pts = np.array(s.trail, np.int32)
        if len(pts) > 1:
            cv2.polylines(view, [pts], False, col, 1, cv2.LINE_AA)
        x1, y1, x2, y2 = s.box.astype(int)
        pad = 6
        cv2.rectangle(view, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), col, 2 if s.threat == "HIGH" else 1)
        kind = "DRONE" if s.drone_prob >= 0.7 else ("drone?" if s.drone_prob >= 0.45 else "not-drone")
        yolo = f" yolo:{s.label}" if s.label else ""
        _put(view, f"T{s.track_id} {kind} {s.drone_prob:.2f} [{s.sources}]{yolo}", (x1 - pad, y1 - pad - 16), col)
        rng = f"{s.range_m:.0f}m" if s.range_m else "-"
        _put(view, f"{s.threat} az{s.azimuth_deg:.1f} el{s.elevation_deg:.1f} {rng}",
             (x1 - pad, y1 - pad - 4), col, 0.38)

    n_drone = sum(1 for s in res.tracks if s.confirmed and s.drone_prob >= 0.7)
    hud = f"t={res.t:6.2f}s  dets R{len(res.rgb_dets)} T{len(res.thermal_dets)} F{len(res.fused_dets)}  " \
          f"tracks {sum(s.confirmed for s in res.tracks)}  drones {n_drone}"
    if fps:
        hud += f"  {fps:.0f} fps"
    cv2.rectangle(view, (0, 0), (view.shape[1], 22), (0, 0, 0), -1)
    _put(view, hud, (6, 16), (255, 255, 255), 0.5)

    if not has_thermal:
        return view
    # thermal panel
    h = view.shape[0]
    if res.thermal_norm is not None:
        th = cv2.applyColorMap((res.thermal_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        sc = h / th.shape[0]
        th = cv2.resize(th, (int(th.shape[1] * sc), h), interpolation=cv2.INTER_NEAREST)
        for d in res.thermal_dets:
            x1, y1, x2, y2 = (d.box * sc).astype(int)
            cv2.rectangle(th, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (255, 255, 0), 1)
        _put(th, "THERMAL", (6, 16), (255, 255, 255), 0.5)
    else:
        th = np.zeros((h, h * 5 // 4, 3), np.uint8)
        _put(th, "THERMAL: no signal", (6, 16), (0, 0, 255), 0.5)
    return np.hstack([view, th])

"""Geolocation (bearing / elevation / rough range) and threat levelling."""

from __future__ import annotations

import math

import numpy as np

from .config import CameraConfig, ThreatConfig
from .tracking import Track
from .types import ThreatEvent

LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH"]


class CameraGeometry:
    def __init__(self, cfg: CameraConfig, image_size: tuple[int, int]):
        self.cfg = cfg
        self.w, self.h = image_size
        self.f = (self.w / 2) / math.tan(math.radians(cfg.hfov_deg) / 2)  # focal length, px

    def angles(self, cx: float, cy: float) -> tuple[float, float]:
        az = self.cfg.pan_deg + math.degrees(math.atan2(cx - self.w / 2, self.f))
        el = self.cfg.tilt_deg + math.degrees(math.atan2(self.h / 2 - cy, self.f))
        return (az + 360.0) % 360.0, el

    def range_m(self, size_px: float) -> float | None:
        """Monocular pinhole estimate from a nominal drone span. Rough (+-50 %)."""
        if size_px < 2:
            return None
        return self.f * self.cfg.drone_span_m / size_px


class ThreatAssessor:
    def __init__(self, cfg: ThreatConfig, geom: CameraGeometry, debounce: int = 3):
        self.cfg = cfg
        self.geom = geom
        self.debounce = debounce

    def kinematics(self, tr: Track):
        cx, cy = tr.kf.x[:2]
        az, el = self.geom.angles(cx, cy)
        hist = list(tr.history)[-30:]
        rng = self.geom.range_m(max(hist[-1]["w"], hist[-1]["h"])) if hist else None
        closing = None
        if len(hist) >= 8:
            tt = np.array([e["t"] for e in hist])
            size = np.array([max(e["w"], e["h"]) for e in hist])
            if np.ptp(tt) > 1.0 and np.all(size >= 4):
                r = self.geom.f * self.geom.cfg.drone_span_m / size
                closing = -float(np.polyfit(tt, r, 1)[0])  # +ve = approaching
        if hist:
            # range from the smoothed size, not the latest noisy box
            rng = self.geom.range_m(float(np.median([max(e["w"], e["h"]) for e in hist[-8:]])))
        return az, el, rng, closing

    def level_for(self, tr: Track, rng, closing) -> str:
        c = self.cfg
        if not tr.extra.get("evidence"):
            return "NONE"  # not classified yet: p is still the 0.5 prior
        if not tr.confirmed or tr.drone_prob < c.drone_prob_low - (0.1 if tr.threat != "NONE" else 0.0):
            return "NONE"
        hp = 0.1 if tr.threat in ("MEDIUM", "HIGH") else 0.0  # probability hysteresis
        if tr.drone_prob < c.drone_prob_high - hp:
            return "LOW"
        # hysteresis: once HIGH, stay HIGH until clearly outside the alert envelope
        h = 1.25 if tr.threat == "HIGH" else 1.0
        near = rng is not None and rng < c.alert_range_m * h
        closing_fast = closing is not None and closing > c.closing_speed_mps / h
        return "HIGH" if near or closing_fast else "MEDIUM"

    def assess(self, tr: Track, t: float) -> ThreatEvent | None:
        az, el, rng, closing = self.kinematics(tr)
        tr.extra.update(az=az, el=el, range_m=rng, closing=closing)
        new = self.level_for(tr, rng, closing)
        if new == tr.threat:
            tr.threat_pending = (new, 0)
            return None
        pend, n = tr.threat_pending
        n = n + 1 if pend == new else 1
        # escalate quickly, de-escalate slowly
        need = 1 if LEVELS.index(new) > LEVELS.index(tr.threat) else self.debounce * 3
        if n < need:
            tr.threat_pending = (new, n)
            return None
        prev, tr.threat, tr.threat_pending = tr.threat, new, (new, 0)
        rtxt = f"{rng:.0f} m" if rng else "?"
        return ThreatEvent(t=t, track_id=tr.id, level=new, previous=prev, drone_prob=tr.drone_prob,
                           azimuth_deg=az, elevation_deg=el, range_m=rng,
                           message=f"Track {tr.id}: {prev}->{new} az {az:.1f} el {el:.1f} rng~{rtxt} "
                                   f"p(drone)={tr.drone_prob:.2f}")

    def on_track_lost(self, tr: Track, t: float) -> ThreatEvent | None:
        if tr.threat == "NONE":
            return None
        az, el = tr.extra.get("az", 0.0), tr.extra.get("el", 0.0)
        return ThreatEvent(t=t, track_id=tr.id, level="LOST", previous=tr.threat, drone_prob=tr.drone_prob,
                           azimuth_deg=az, elevation_deg=el, range_m=tr.extra.get("range_m"),
                           message=f"Track {tr.id} lost (was {tr.threat})")

"""End-to-end per-frame processing: detect (RGB, thermal) -> fuse -> track ->
classify -> assess threat."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .detectors import build_rgb_detector, build_thermal_detector, normalize_thermal, rgb_usable
from .fusion import Fuser
from .threat import CameraGeometry, ThreatAssessor
from .tracking import DroneClassifier, MultiTargetTracker
from .types import Detection, ThreatEvent, TrackSnapshot


@dataclass
class FrameResult:
    t: float
    rgb_dets: list[Detection]
    thermal_dets: list[Detection]        # in thermal pixel coords
    fused_dets: list[Detection]          # in RGB pixel coords
    tracks: list[TrackSnapshot]
    events: list[ThreatEvent]
    thermal_norm: np.ndarray | None = None
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "t": round(self.t, 3),
            "n_rgb": len(self.rgb_dets), "n_thermal": len(self.thermal_dets), "n_fused": len(self.fused_dets),
            "tracks": [s.to_dict() for s in self.tracks if s.confirmed],
            "events": [e.to_dict() for e in self.events],
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
        }


class CounterDronePipeline:
    def __init__(self, cfg: Config, H_thermal_to_rgb: np.ndarray,
                 rgb_size: tuple[int, int], thermal_size: tuple[int, int]):
        """Sizes are (width, height)."""
        self.cfg = cfg
        self.rgb_size, self.thermal_size = rgb_size, thermal_size
        self.H = np.asarray(H_thermal_to_rgb, float)
        self.rgb_det = build_rgb_detector(cfg.rgb)
        self.th_det = build_thermal_detector(cfg.thermal)
        self.fuser = Fuser(cfg.fusion, self.H, thermal_size)
        self.tracker = MultiTargetTracker(cfg.tracker, DroneClassifier(cfg.classifier))
        self.threat = ThreatAssessor(cfg.threat, CameraGeometry(cfg.camera, rgb_size))

    def process(self, rgb: np.ndarray | None, thermal: np.ndarray | None, t: float) -> FrameResult:
        """Either frame may be None (sensor dropout); the other sensor keeps tracks alive."""
        tm = {}
        t0 = time.perf_counter()
        rgb_dets = self.rgb_det.detect(rgb) if rgb is not None else []
        tm["rgb_det"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        th_norm = None
        th_dets: list[Detection] = []
        if thermal is not None:
            th_norm = normalize_thermal(thermal, self.cfg.thermal.agc_low_pct, self.cfg.thermal.agc_high_pct)
            th_dets = self.th_det.detect(th_norm)  # every backend gets the AGC'd image used in training
        tm["thermal_det"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        fused = self.fuser.fuse(rgb_dets, th_dets, rgb_usable(rgb, self.cfg.rgb.min_mean_brightness),
                                thermal_ok=thermal is not None)
        tm["fusion"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        tracks = self.tracker.step(fused, t)
        events = []
        for tr in tracks:
            ev = self.threat.assess(tr, t)
            if ev:
                events.append(ev)
        for tr in self.tracker.removed:
            ev = self.threat.on_track_lost(tr, t)
            if ev:
                events.append(ev)
        tm["track"] = (time.perf_counter() - t0) * 1e3

        snaps = [TrackSnapshot(
            track_id=tr.id, box=tr.box, velocity=tr.velocity, confirmed=tr.confirmed,
            drone_prob=tr.drone_prob, threat=tr.threat,
            azimuth_deg=tr.extra.get("az", 0.0), elevation_deg=tr.extra.get("el", 0.0),
            range_m=tr.extra.get("range_m"), closing_mps=tr.extra.get("closing"),
            age=tr.age, misses=tr.misses, sources=tr.recent_sources(), trail=list(tr.trail)[-40:],
            label=tr.label(), score=tr.recent_score(), hits=tr.hits,
            det_score=getattr(tr, "det_score", 0.0) if tr.misses == 0 else 0.0,
        ) for tr in tracks]
        return FrameResult(t, rgb_dets, th_dets, fused, snaps, events, th_norm, tm)

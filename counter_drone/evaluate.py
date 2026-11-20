"""Scoring against synthetic (or annotated) ground truth."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .pipeline import FrameResult


class Evaluator:
    """Per-frame matching of confirmed tracks to GT objects by centre distance.

    A drone is *declared* in a frame when a confirmed track with
    p(drone) >= ``declare_prob`` is matched to it. Reports declaration rate,
    false declarations (on birds or nothing), ID switches and latency.
    """

    def __init__(self, match_px: float = 20.0, declare_prob: float = 0.7):
        self.match_px = match_px
        self.declare_prob = declare_prob
        self.frames = 0
        self.gt_drone_frames = 0
        self.declared_frames = 0
        self.tracked_frames = 0
        self.false_decl = 0
        self.bird_decl = 0
        self.det_tp = self.det_fp = self.det_fn = 0
        self.last_id: dict[int, int] = {}
        self.id_switches = 0
        self.first_seen: dict[int, float] = {}
        self.first_declared: dict[int, float] = {}
        self.by_phase = defaultdict(lambda: [0, 0])  # phase -> [declared, gt]

    @staticmethod
    def _match(a, b, gate):
        if len(a) == 0 or len(b) == 0:
            return []
        d = np.linalg.norm(np.asarray(a)[:, None] - np.asarray(b)[None], axis=2)
        r, c = linear_sum_assignment(np.where(d < gate, d, 1e6))
        return [(i, j) for i, j in zip(r, c) if d[i, j] < gate]

    def add(self, res: FrameResult, gts, phase: str = "day"):
        self.frames += 1
        center = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)  # noqa: E731
        gt_c = [center(g.box) for g in gts]

        # detection level (fused detections vs all drone GTs)
        drones = [i for i, g in enumerate(gts) if g.cls == "drone"]
        dm = self._match([gt_c[i] for i in drones], [d.center for d in res.fused_dets], self.match_px)
        self.det_tp += len(dm)
        self.det_fn += len(drones) - len(dm)

        tracks = [s for s in res.tracks if s.confirmed]
        m = self._match(gt_c, [center(s.box) for s in tracks], self.match_px)
        matched_tracks = set()
        for gi, ti in m:
            g, s = gts[gi], tracks[ti]
            matched_tracks.add(ti)
            if g.cls == "drone":
                self.tracked_frames += 1
                if self.last_id.get(g.obj_id, s.track_id) != s.track_id:
                    self.id_switches += 1
                self.last_id[g.obj_id] = s.track_id
                if s.drone_prob >= self.declare_prob:
                    self.declared_frames += 1
                    self.by_phase[phase][0] += 1
                    self.first_declared.setdefault(g.obj_id, res.t)
            elif s.drone_prob >= self.declare_prob:
                self.bird_decl += 1
        for ti, s in enumerate(tracks):
            if ti not in matched_tracks and s.drone_prob >= self.declare_prob:
                self.false_decl += 1
        for g in gts:
            if g.cls == "drone":
                self.gt_drone_frames += 1
                self.by_phase[phase][1] += 1
                self.first_seen.setdefault(g.obj_id, res.t)

    def summary(self) -> dict:
        lat = {k: round(self.first_declared[k] - self.first_seen[k], 2) for k in self.first_declared}
        return {
            "frames": self.frames,
            "detection_recall": round(self.det_tp / max(1, self.det_tp + self.det_fn), 3),
            "track_coverage": round(self.tracked_frames / max(1, self.gt_drone_frames), 3),
            "declaration_rate": round(self.declared_frames / max(1, self.gt_drone_frames), 3),
            "declaration_rate_by_phase": {k: round(v[0] / max(1, v[1]), 3) for k, v in self.by_phase.items()},
            "false_declarations_on_birds": self.bird_decl,
            "false_declarations_on_nothing": self.false_decl,
            "id_switches": self.id_switches,
            "declare_latency_s": lat,
            "drones_never_declared": sorted(set(self.first_seen) - set(self.first_declared)),
        }

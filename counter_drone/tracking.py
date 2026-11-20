"""Multi-target tracking: constant-velocity Kalman filters + global nearest
neighbour (Hungarian) association with Mahalanobis gating, M-of-N track
confirmation, coasting through dropouts, and a per-track drone/clutter
classifier that accumulates kinematic and thermal evidence over time.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import ClassifierConfig, TrackerConfig
from .types import Detection


class KalmanCV:
    """State [cx, cy, vx, vy, w, h]; measurement [cx, cy, w, h]."""

    Hm = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1]], float)

    def __init__(self, z: np.ndarray, cfg: TrackerConfig):
        self.cfg = cfg
        self.x = np.array([z[0], z[1], 0.0, 0.0, z[2], z[3]], float)
        self.P = np.diag([cfg.meas_pos_std ** 2] * 2 + [80.0 ** 2] * 2 + [cfg.meas_size_std ** 2] * 2)

    def predict(self, dt: float):
        F = np.eye(6)
        F[0, 2] = F[1, 3] = dt
        q = self.cfg.process_accel ** 2
        Q = np.zeros((6, 6))
        for p, v in ((0, 2), (1, 3)):
            Q[p, p] = q * dt ** 4 / 4
            Q[p, v] = Q[v, p] = q * dt ** 3 / 2
            Q[v, v] = q * dt ** 2
        Q[4, 4] = Q[5, 5] = (10.0 ** 2) * dt  # size random walk
        self.x = F @ self.x
        self.x[4:] = np.maximum(self.x[4:], 1.0)
        self.P = F @ self.P @ F.T + Q

    @property
    def size(self) -> float:
        return float(max(self.x[4], self.x[5], 1.0))

    def pos_std(self) -> float:
        """Measurement jitter grows with target size: a 100 px drone's box centre moves
        far more between frames (detector box noise, camera shake) than a 6 px one's."""
        c = self.cfg
        return max(c.meas_pos_std, c.meas_rel_std * float(max(self.x[4], self.x[5])))

    def R(self, scale: float = 1.0) -> np.ndarray:
        c = self.cfg
        size = float(max(self.x[4], self.x[5]))
        ps, ss = self.pos_std(), max(c.meas_size_std, c.meas_rel_std * size)
        return np.diag([(ps * scale) ** 2] * 2 + [(ss * scale) ** 2] * 2)

    def innovation(self, z, scale=1.0):
        S = self.Hm @ self.P @ self.Hm.T + self.R(scale)
        return z - self.Hm @ self.x, S

    def mahalanobis_pos(self, z, scale=1.0) -> float:
        y, S = self.innovation(z, scale)
        return float(y[:2] @ np.linalg.solve(S[:2, :2], y[:2]))

    def update(self, z, scale=1.0):
        y, S = self.innovation(z, scale)
        K = self.P @ self.Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ self.Hm
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R(scale) @ K.T  # Joseph form


def _z(det: Detection) -> np.ndarray:
    c, wh = det.center, det.wh
    return np.array([c[0], c[1], max(wh[0], 1.0), max(wh[1], 1.0)])


def _meas_scale(det: Detection) -> float:
    # thermal-only boxes come from a lower-resolution sensor
    return 2.0 if det.source == "thermal" else 1.0


class Track:
    _next_id = 1

    def __init__(self, det: Detection, t: float, cfg: TrackerConfig):
        self.id = Track._next_id
        Track._next_id += 1
        self.cfg = cfg
        self.kf = KalmanCV(_z(det), cfg)
        self.t_last = t
        self.t_start = t
        self.age = 1
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.hit_window = deque([True], maxlen=cfg.confirm_window)
        self.history = deque(maxlen=cfg.history_len)   # per-update evidence
        self.trail = deque(maxlen=cfg.history_len)     # (cx, cy) for drawing
        self.drone_prob = 0.5
        self.threat = "NONE"
        self.threat_pending = ("NONE", 0)
        self.extra: dict = {}
        self._record(det, t)

    # ---- lifecycle ---------------------------------------------------------
    def predict(self, t: float):
        dt = max(1e-3, t - self.t_last)
        self.kf.predict(dt)
        self.t_last = t
        self.age += 1

    def update(self, det: Detection, t: float):
        self.kf.update(_z(det), _meas_scale(det))
        self.hits += 1
        self.misses = 0
        self.hit_window.append(True)
        if not self.confirmed and sum(self.hit_window) >= self.cfg.min_hits_confirm:
            self.confirmed = True
        self._record(det, t)

    def mark_missed(self):
        self.misses += 1
        self.hit_window.append(False)
        self.trail.append(tuple(self.kf.x[:2]))

    @property
    def dead(self) -> bool:
        limit = self.cfg.max_misses_confirmed if self.confirmed else self.cfg.max_misses_tentative
        return self.misses > limit

    def _record(self, det: Detection, t: float):
        x = self.kf.x
        self.history.append({
            "t": t, "cx": x[0], "cy": x[1], "vx": x[2], "vy": x[3],
            "w": float(det.wh[0]), "h": float(det.wh[1]),
            "src": det.source, "in_tfov": det.in_thermal_fov, "rgb_ok": det.rgb_ok,
            "tc": det.thermal_contrast, "label": det.label, "score": float(det.score),
        })
        self.last_det_box = det.box.copy()
        self.det_score = float(det.score)
        self.trail.append((x[0], x[1]))

    @property
    def box(self) -> np.ndarray:
        """Where the target is now: this frame's detection box when the track was just
        matched (no smoothing lag on jerky motion), else the Kalman prediction."""
        if self.misses == 0 and getattr(self, "last_det_box", None) is not None:
            return self.last_det_box.copy()
        return self.filtered_box

    @property
    def filtered_box(self) -> np.ndarray:
        cx, cy, _, _, w, h = self.kf.x
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def recent_score(self, n: int = 10) -> float:
        """Mean detection confidence over the last n matched detections."""
        sc = [h["score"] for h in list(self.history)[-n:]]
        return float(np.mean(sc)) if sc else 0.0

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.x[2:4].copy()

    def label(self, n: int = 30) -> str | None:
        """Majority class from a learned detector over recent updates (None if never labelled)."""
        labels = [h["label"] for h in list(self.history)[-n:] if h["label"]]
        return max(set(labels), key=labels.count) if labels else None

    def recent_sources(self, n: int = 10) -> str:
        srcs = [h["src"] for h in list(self.history)[-n:]]
        r = any(s in ("rgb", "fused") for s in srcs)
        th = any(s in ("thermal", "fused") for s in srcs)
        return ("R" if r else "") + ("T" if th else "")


class MultiTargetTracker:
    def __init__(self, cfg: TrackerConfig, classifier: "DroneClassifier | None" = None):
        self.cfg = cfg
        self.tracks: list[Track] = []
        self.removed: list[Track] = []
        self.classifier = classifier

    def step(self, dets: list[Detection], t: float) -> list[Track]:
        for tr in self.tracks:
            tr.predict(t)

        # Confirmed tracks get first pick of detections (cascade), then tentative.
        unmatched = list(range(len(dets)))
        for group in ([tr for tr in self.tracks if tr.confirmed],
                      [tr for tr in self.tracks if not tr.confirmed]):
            matched_tracks, unmatched = self._associate(group, dets, unmatched, t)
            for tr in group:
                if tr not in matched_tracks:
                    tr.mark_missed()

        for j in unmatched:
            if dets[j].score >= self.cfg.init_min_score:
                self.tracks.append(Track(dets[j], t, self.cfg))

        self.removed = [tr for tr in self.tracks if tr.dead]
        self.tracks = [tr for tr in self.tracks if not tr.dead]
        if self.classifier is not None:
            for tr in self.tracks:
                self.classifier.update(tr)
        return self.tracks

    def _associate(self, tracks, dets, det_idx, t):
        if not tracks or not det_idx:
            return set(), det_idx
        big = 1e6
        Z = np.array([_z(dets[j])[:2] for j in det_idx])                 # D,2
        scale = np.array([_meas_scale(dets[j]) for j in det_idx])       # D
        score = np.array([dets[j].score for j in det_idx])
        cost = np.full((len(tracks), len(det_idx)), big)
        for i, tr in enumerate(tracks):
            r2 = tr.kf.pos_std() ** 2
            P = tr.kf.P[:2, :2]
            y = Z - tr.kf.x[:2]
            # S differs per detection only through the (isotropic) measurement-noise scale
            a, b, d = P[0, 0], P[0, 1], P[1, 1]
            sa, sd = a + r2 * scale ** 2, d + r2 * scale ** 2
            det_s = sa * sd - b * b
            d2 = (sd * y[:, 0] ** 2 - 2 * b * y[:, 0] * y[:, 1] + sa * y[:, 1] ** 2) / det_s
            ok = d2 <= self.cfg.gate_chi2
            cost[i, ok] = d2[ok] - 0.5 * score[ok]  # prefer confident detections on ties
            # Size gate: a detection within ~1.5 box-widths of the prediction is the same
            # target even when it jumped more than the motion model expects (a camera that
            # pans after the drone makes its image motion jerky). Ranked after statistical matches.
            near = (~ok) & (np.linalg.norm(y, axis=1) <= self.cfg.gate_size_frac * tr.kf.size)
            cost[i, near] = self.cfg.gate_chi2 + np.linalg.norm(y[near], axis=1) / tr.kf.size - 0.5 * score[near]
        matched = set()
        used = set()
        for i, k in zip(*linear_sum_assignment(cost)):
            if cost[i, k] >= big:
                continue
            tracks[i].update(dets[det_idx[k]], t)
            matched.add(tracks[i])
            used.add(k)
        return matched, [j for k, j in enumerate(det_idx) if k not in used]


class DroneClassifier:
    """Explainable log-odds classifier: drone vs. clutter (birds, cloud edges, noise).

    Evidence terms (each clipped so no single cue dominates); the per-track
    breakdown is kept in ``track.extra["evidence"]`` for operator display/logs.

      * consistency - fraction of frames the target was detected by BOTH
        sensors while both could see it. Real aircraft are; cloud edges, glints
        and thermal sky texture usually only trip one sensor. (Birds are warm
        too, so thermal *brightness* is deliberately not used as drone evidence.)
      * flap  - wing-beat makes a bird's box aspect ratio oscillate; it is
        scale-invariant, so an approaching drone's growth does not trigger it
      * turn  - birds change heading erratically; drones fly smooth paths
      * hover - drones can hold position; birds cannot
      * label - class from a learned detector, when one is used
    """

    def __init__(self, cfg: ClassifierConfig):
        self.cfg = cfg

    def evidence(self, tr: Track) -> dict[str, float]:
        c = self.cfg
        h = list(tr.history)[-c.window:]
        ev: dict[str, float] = {}
        if len(h) < c.min_track_len:
            return ev

        both = [e for e in h if e["in_tfov"] and e["rgb_ok"]]
        if len(both) >= c.min_consistency_frames:
            x = float(np.clip(2 * np.mean([e["src"] == "fused" for e in both]) - 1, -1, 1))
            ev["consistency"] = (c.w_consistency_pos if x > 0 else c.w_consistency_neg) * x

        rgb = [e for e in h if e["src"] in ("rgb", "fused") and e["rgb_ok"]]
        if len(rgb) >= c.min_flap_frames:
            asp = float(np.std([np.log(e["h"] / e["w"]) for e in rgb]))
            ev["flap"] = float(np.clip(c.w_flap * (asp - c.flap_ref), -3.0, 1.5))

        tt = np.array([e["t"] for e in h])
        pos = np.array([[e["cx"], e["cy"]] for e in h])
        v = np.array([[e["vx"], e["vy"]] for e in h])
        ev["hover"] = c.w_hover * float(np.mean(np.linalg.norm(v, axis=1) < c.hover_speed_px))
        rate = self._turn_rate(tt, pos, c.hover_speed_px * 2)
        if rate is not None:
            ev["turn"] = float(np.clip(c.w_turn * (rate - c.turn_ref), -2.5, 1.0))

        labels = [e["label"] for e in h if e["label"]]
        if labels:
            frac = labels.count("drone") / len(labels)
            # weigh the label by how confident the detector was: a 0.85 'drone' is strong
            # evidence, a 0.3 'drone' (typical of a false box on a building edge) counts against
            conf = float(np.mean([e["score"] for e in h if e["label"]]))
            trust = float(np.clip((conf - c.label_conf_lo) / (c.label_conf_hi - c.label_conf_lo), 0.0, 1.0))
            ev["label"] = c.w_label * (2 * frac - 1) * (2 * trust - 1)
            # A trained detector recognises the target from its appearance. The hand-made
            # flight cues assume a fixed camera; on a panning/tilting camera (or a drone
            # turning, which changes its box shape) they misfire, so when the detector
            # labels the track they only nudge the decision.
            if len(labels) >= c.label_min_share * len(h) and ev["label"] > 0:
                for k in ("flap", "turn", "hover"):
                    if k in ev:
                        ev[k] *= c.kinematic_scale_with_label
        return ev

    @staticmethod
    def _turn_rate(tt, pos, min_speed, k: int = 5) -> float | None:
        """Mean |heading change| (rad/s) from displacements over k frames.

        Using displacement over a short baseline (not instantaneous velocity)
        and only contiguous *moving* segments keeps hover jitter - where heading
        is undefined - from looking like erratic flight.
        """
        if len(tt) <= k + 2:
            return None
        d = pos[k:] - pos[:-k]
        dt = tt[k:] - tt[:-k]
        speed = np.linalg.norm(d, axis=1) / np.maximum(dt, 1e-3)
        mov = speed > min_speed
        ang = np.arctan2(d[:, 1], d[:, 0])
        both = mov[1:] & mov[:-1]
        if both.sum() < 5:
            return None
        dang = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)[both]
        step = np.diff(tt[k:])[both]
        return float(np.mean(dang / np.maximum(step, 1e-3)))

    def update(self, tr: Track):
        ev = self.evidence(tr)
        if not ev:
            tr.extra["evidence"] = ev
            return
        # cues that are unavailable right now (e.g. RGB blind at dusk) keep their
        # last value, slowly decaying, instead of vanishing from the decision
        mem = tr.extra.setdefault("ev_mem", {})
        for k in list(mem):
            if k not in ev:
                mem[k] *= self.cfg.memory_decay
                if abs(mem[k]) > 0.05:
                    ev[k] = mem[k]
        mem.update(ev)
        if "consistency" not in ev and ev.get("label", 0.0) <= 0:
            # no cross-check and no learned identity: unconfirmed single-sensor blob
            ev["single_sensor"] = self.cfg.w_single_sensor
        tr.extra["evidence"] = ev
        logit = self.cfg.bias + sum(ev.values())
        p = 1.0 / (1.0 + math.exp(-logit))
        tr.drone_prob = (1 - self.cfg.ema) * tr.drone_prob + self.cfg.ema * p

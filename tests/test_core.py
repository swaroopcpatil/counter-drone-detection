import json

import cv2
import numpy as np
import pytest

from counter_drone.calibration import homography_from_points, map_boxes, save_homography, load_homography
from counter_drone.config import Config, TrackerConfig, ClassifierConfig
from counter_drone.detectors import (RGBSmallTargetDetector, ThermalHotspotDetector, normalize_thermal,
                                     rgb_usable)
from counter_drone.fusion import Fuser
from counter_drone.tracking import DroneClassifier, KalmanCV, MultiTargetTracker, Track
from counter_drone.types import Detection


def _sky(h=360, w=640, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (230, 190, 140), np.float32) + rng.normal(0, 2, (h, w, 3))
    return img


def _det(cx, cy, s=8, score=0.9, src="fused"):
    return Detection(box=np.array([cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2], float), score=score,
                     source=src, rgb_score=score, thermal_score=score)


# ---------------------------------------------------------------- detectors
def test_rgb_detector_finds_small_dark_target_and_ignores_clean_sky():
    det = RGBSmallTargetDetector(Config().rgb)
    img = _sky()
    assert det.detect(np.clip(img, 0, 255).astype(np.uint8)) == []
    cv2.circle(img, (300, 150), 4, (40, 40, 40), -1)
    out = det.detect(np.clip(img, 0, 255).astype(np.uint8))
    assert len(out) == 1
    assert np.allclose(out[0].center, (300, 150), atol=1.5)


def test_thermal_cfar_does_not_mask_close_weaker_target():
    rng = np.random.default_rng(1)
    t = 0.2 + rng.normal(0, 0.006, (256, 320))
    yy, xx = np.mgrid[:256, :320]
    t += 0.6 * np.exp(-((xx - 150) ** 2 + (yy - 120) ** 2) / 2.0)   # strong target
    t += 0.12 * np.exp(-((xx - 165) ** 2 + (yy - 128) ** 2) / 2.0)  # weak neighbour ~17 px away
    raw = (7000 + t * 4000).astype(np.uint16)
    out = ThermalHotspotDetector(Config().thermal).detect(normalize_thermal(raw))
    centers = sorted(tuple(np.round(d.center).astype(int)) for d in out)
    assert len(out) == 2, centers


def test_rgb_usable():
    assert rgb_usable(np.full((10, 10, 3), 120, np.uint8))
    assert not rgb_usable(np.full((10, 10, 3), 8, np.uint8))
    assert not rgb_usable(None)


# ---------------------------------------------------------------- calibration + fusion
def test_homography_from_points_roundtrip(tmp_path):
    H_true = np.array([[2.0, 0.05, 160], [-0.03, 2.0, 14], [0, 0, 1]])
    pt = np.array([[10, 10], [300, 20], [310, 240], [15, 250], [160, 128], [80, 200]], float)
    pr = cv2.perspectiveTransform(pt.reshape(-1, 1, 2), H_true).reshape(-1, 2)
    H, rms = homography_from_points(pt, pr)
    assert rms < 1e-3
    save_homography(tmp_path / "c.json", H)
    assert np.allclose(load_homography(tmp_path / "c.json"), H)


def test_map_boxes_scale_offset():
    H = np.array([[2.0, 0, 160], [0, 2.0, 14], [0, 0, 1]])
    assert np.allclose(map_boxes(np.array([[10, 10, 20, 30]]), H), [[180, 34, 200, 74]])


def test_fusion_pairs_and_penalises_single_sensor():
    H = np.array([[2.0, 0, 160], [0, 2.0, 14], [0, 0, 1]])
    fuser = Fuser(Config().fusion, H, (320, 256))
    rgb = [Detection(np.array([396, 196, 404, 204.]), 0.8, "rgb", rgb_score=0.8),     # pairs with thermal
           Detection(np.array([500, 300, 508, 308.]), 0.8, "rgb", rgb_score=0.8),     # rgb-only, inside FOV
           Detection(np.array([20, 300, 28, 308.]), 0.8, "rgb", rgb_score=0.8)]       # rgb-only, outside FOV
    th = [Detection(np.array([118, 91, 122, 95.]), 0.7, "thermal", thermal_score=0.7, thermal_contrast=9)]
    out = {d.source + str(int(d.box[0])): d for d in fuser.fuse(rgb, th)}
    fused = out["fused396"]
    assert fused.score == pytest.approx(1 - 0.2 * 0.3)
    assert out["rgb500"].score == pytest.approx(0.8 * 0.75) and out["rgb500"].in_thermal_fov
    assert out["rgb20"].score == pytest.approx(0.8) and not out["rgb20"].in_thermal_fov


# ---------------------------------------------------------------- tracking
def test_kalman_learns_constant_velocity():
    cfg = TrackerConfig()
    kf = KalmanCV(np.array([100, 100, 8, 8.]), cfg)
    for k in range(1, 30):
        kf.predict(0.1)
        kf.update(np.array([100 + 5 * k, 100 - 2 * k, 8, 8.]))
    assert np.allclose(kf.x[2:4], [50, -20], atol=2)


def test_tracker_keeps_identity_through_dropout_and_separates_targets():
    tr = MultiTargetTracker(TrackerConfig())
    ids_a = set()
    for k in range(60):
        t = k / 15
        dets = [_det(100 + 40 * t, 200), _det(400, 100 + 20 * t)]
        if 25 <= k < 33:            # target A missing for ~0.5 s
            dets = dets[1:]
        tracks = tr.step(dets, t)
        for x in tracks:
            if x.confirmed and abs(x.kf.x[1] - 200) < 10:
                ids_a.add(x.id)
    assert len(ids_a) == 1
    assert sum(x.confirmed for x in tr.tracks) == 2


def test_tracker_does_not_confirm_one_off_noise():
    tr = MultiTargetTracker(TrackerConfig())
    rng = np.random.default_rng(0)
    for k in range(30):
        tr.step([_det(*rng.uniform(50, 600, 2), score=0.6)], k / 15)
    assert not any(x.confirmed for x in tr.tracks)


# ---------------------------------------------------------------- classifier
def _run_track(path_fn, box_fn, n=45, src="fused"):
    cfg = TrackerConfig()
    clf = DroneClassifier(ClassifierConfig())
    t0 = 0.0
    x, y = path_fn(0)
    w, h = box_fn(0)
    tr = Track(Detection(np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]), 0.9, src), t0, cfg)
    for k in range(1, n):
        t = k / 15
        tr.predict(t)
        x, y = path_fn(t)
        w, h = box_fn(t)
        tr.update(Detection(np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]), 0.9, src), t)
        clf.update(tr)
    return tr


def test_classifier_separates_drone_from_bird():
    drone = _run_track(lambda t: (100 + 30 * t, 200 + 5 * t), lambda t: (10, 10))
    rng = np.random.default_rng(3)
    heading = np.cumsum(rng.normal(0, 0.8, 200))
    pos = np.cumsum(np.stack([np.cos(heading), np.sin(heading)], 1) * 70 / 15, 0) + 300
    bird = _run_track(lambda t: tuple(pos[int(round(t * 15))]),
                      lambda t: (12, 3 + 4 * abs(np.sin(2 * np.pi * 5 * t))))
    assert drone.drone_prob > 0.75, drone.extra
    assert bird.drone_prob < 0.35, bird.extra


def test_single_sensor_clutter_is_not_declared():
    clutter = _run_track(lambda t: (300 + 10 * t, 200), lambda t: (10, 8 + 3 * (int(t * 15) % 2)), src="rgb")
    assert clutter.drone_prob < 0.5, clutter.extra


# ---------------------------------------------------------------- config
def test_config_roundtrip(tmp_path):
    c = Config()
    c.rgb.k_sigma = 9.5
    c.save(tmp_path / "c.json")
    c2 = Config.load(tmp_path / "c.json")
    assert c2.rgb.k_sigma == 9.5 and c2.to_dict() == c.to_dict()
    with pytest.raises(KeyError):
        Config.from_dict({"rgb": {"nope": 1}})


def test_labelled_drone_on_panning_camera_is_declared():
    """Real footage (Anti-UAV): the camera pans/tilts after the drone and the drone's box
    changes shape as it turns, so its image motion looks erratic. A track the trained
    detector consistently labels 'drone' must still be declared, from one camera alone."""
    cfg = TrackerConfig()
    clf = DroneClassifier(ClassifierConfig())
    rng = np.random.default_rng(7)
    heading = np.cumsum(rng.normal(0, 0.9, 120))
    pos = np.cumsum(np.stack([np.cos(heading), np.sin(heading)], 1) * 90 / 25, 0) + 500

    def det(k):
        x, y = pos[k]
        w, h = 60 * (1 + 0.25 * np.sin(k / 3)), 40 * (1 + 0.3 * np.cos(k / 2))
        return Detection(np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]), 0.9, "rgb", label="drone",
                         in_thermal_fov=False)

    tr = Track(det(0), 0.0, cfg)
    for k in range(1, 60):
        t = k / 25
        tr.predict(t)
        tr.update(det(k), t)
        clf.update(tr)
    assert tr.drone_prob > 0.7, tr.extra["evidence"]


def test_large_jittery_target_keeps_one_track():
    """A near, 120 px drone: detector boxes jitter by several pixels frame to frame.
    The gate must scale with target size so the track is not lost and restarted."""
    tr = MultiTargetTracker(TrackerConfig())
    rng = np.random.default_rng(1)
    ids, covered = set(), 0
    for k in range(75):
        t = k / 25
        cx, cy = 900 + 60 * t + rng.normal(0, 15), 500 + rng.normal(0, 15)
        confirmed = [x for x in tr.step([_det(cx, cy, s=120)], t) if x.confirmed]
        ids |= {x.id for x in confirmed}
        covered += bool(confirmed)
    assert len(ids) == 1, ids
    assert covered >= 65, covered


def _anti_uav_like_run(seed, n=300, fps=25, hard=False):
    """One YOLO-labelled drone whose image motion is jerky (camera chasing it), plus a
    persistent weak false box on a 'building' and occasional random weak boxes."""
    from counter_drone.anti_uav import best_target, iou_xyxy
    from counter_drone.pipeline import CounterDronePipeline

    class Stub:  # stands in for the YOLO detector: returns this frame's boxes
        dets = []

        def detect(self, _img):
            return self.dets

    rng = np.random.default_rng(seed)
    cfg = Config()
    pipe = CounterDronePipeline(cfg, np.eye(3), (1920, 1080), (1920, 1080))
    pipe.rgb_det = Stub()
    frame = np.full((8, 8, 3), 120, np.uint8)
    p, v, size = np.array([960.0, 540.0]), np.zeros(2), 60.0
    building = rng.uniform([200, 700], [1700, 1000])
    hits = 0
    for k in range(n):
        v = 0.9 * v + rng.normal(0, 60, 2)
        if rng.random() < (0.05 if hard else 0.02):
            v += rng.normal(0, 1200 if hard else 600, 2)
        p = np.clip(p + v / fps, 100, [1820, 980])
        gt = np.array([p[0] - 30, p[1] - 21, p[0] + 30, p[1] + 21])
        dets = []
        if rng.random() < 0.95:
            dets.append(Detection(gt + rng.normal(0, 3.6, 4), 0.85, "rgb", rgb_score=0.85, label="drone"))
        if rng.random() < 0.6:
            dets.append(Detection(np.r_[building - 15, building + 15], 0.33, "rgb", rgb_score=0.33, label="drone"))
        if hard:  # medium-confidence distractor drifting across (bird / plane)
            q = np.array([300 + 40 * k / fps, 300 + 15 * np.sin(k / fps)])
            sc = float(rng.uniform(0.5, 0.65))
            dets.append(Detection(np.r_[q - 20, q + 20], sc, "rgb", rgb_score=sc, label="drone"))
        Stub.dets = dets
        best = best_target(pipe.process(frame, None, k / fps).tracks, 0.3)
        hits += best is not None and iou_xyxy(best.box, gt) > 0.3
    return hits / n


def test_tracker_keeps_detector_hits_on_chasing_camera():
    """Regression for the Anti-UAV per-camera gap (detector 76% vs tracked 35%): the tracker
    must not fragment on jerky motion nor prefer a weak false box."""
    rates = [_anti_uav_like_run(s) for s in range(3)]   # previous tracker: 0.28 / 0.56 / 0.43
    assert min(rates) > 0.7 and np.mean(rates) > 0.8, rates


def test_selection_follows_confident_detector_under_distractor():
    """Regression for the remaining RGB gap (detector 76% vs tracked 58%): with fast pans
    and a medium-confidence distractor, the reported target must follow the detector's
    confident box. Previous selection rule: about 0.44 to 0.57 in this scenario."""
    rates = [_anti_uav_like_run(s, hard=True) for s in range(3)]
    assert min(rates) > 0.75 and np.mean(rates) > 0.8, rates


def _gimbal_run(n=750, fps=25, seed=0):
    """A gimbal that lags the drone and snaps it back to centre every 1.3 s (as in Anti-UAV
    sequence 20190925_124000_1_3): returns (confirmed track ids on the drone, LOST alerts)."""
    from counter_drone.anti_uav import iou_xyxy
    from counter_drone.pipeline import CounterDronePipeline

    class Stub:
        dets = []

        def detect(self, _img):
            return self.dets

    rng = np.random.default_rng(seed)
    pipe = CounterDronePipeline(Config(), np.eye(3), (1920, 1080), (1920, 1080))
    pipe.rgb_det = Stub()
    frame = np.full((8, 8, 3), 120, np.uint8)
    x, size = 1100.0, 136.0
    ids, lost = set(), 0
    for k in range(n):
        x -= 250 / fps
        if k % int(1.3 * fps) == 0 and k:
            x += 450                       # gimbal snaps the drone back toward centre
        gt = np.array([x - size / 2, 500 - size * 0.35, x + size / 2, 500 + size * 0.35])
        Stub.dets = [Detection(gt + rng.normal(0, 4, 4), 0.85, "rgb", rgb_score=0.85, label="drone")] \
            if rng.random() < 0.95 else []
        res = pipe.process(frame, None, k / fps)
        ids |= {s.track_id for s in res.tracks if s.confirmed and iou_xyxy(s.box, gt) > 0.3}
        lost += sum(e.level == "LOST" for e in res.events)
    return ids, lost


def test_track_identity_survives_gimbal_recentring_jumps():
    """Live run on Anti-UAV video showed a new track + a LOST alert every ~1.3 s.
    One drone must stay one track, with no LOST alerts while it is in view."""
    ids, lost = _gimbal_run()
    assert len(ids) == 1, ids
    assert lost == 0, lost

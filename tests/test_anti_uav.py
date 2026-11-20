import numpy as np
import pytest

from counter_drone.anti_uav import anti_uav_accuracy, evaluate_sequence, iou_xyxy, make_fake_anti_uav
from counter_drone.calibration import homography_from_box_tracks
from counter_drone.synthetic import SyntheticScene
from counter_drone.yolo_data import convert_anti_uav, load_anti_uav_labels


def test_accuracy_metric_matches_definition():
    # 4 frames: 2 visible (IoU .8 and missed), 2 absent (correctly empty, wrongly reported)
    acc = anti_uav_accuracy([0.8, 0.0, 0.0, 0.0], [False, True, True, False], [True, True, False, False])
    expected = (0.8 + 0 + 1 + 0) / 4 - 0.2 * (1 / 2) ** 0.3
    assert acc == pytest.approx(expected)
    assert iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_homography_from_box_tracks_recovers_mapping():
    H = SyntheticScene.H_thermal_to_rgb
    rng = np.random.default_rng(0)
    ir, rgb = [], []
    for _ in range(50):
        x, y = rng.uniform(10, 300), rng.uniform(10, 240)
        ir.append([x, y, 4, 4])
        rgb.append([2 * x + 160 + rng.normal(0, 0.3), 2 * y + 14 + rng.normal(0, 0.3), 8, 8])
    Hf, rms, _ = homography_from_box_tracks(rgb, ir)
    assert rms < 1.0
    assert np.allclose(Hf / Hf[2, 2], H, atol=0.05)


@pytest.fixture(scope="module")
def fake_auv(tmp_path_factory):
    return make_fake_anti_uav(tmp_path_factory.mktemp("auv") / "Anti-UAV-RGBT", 2, 1, 1, seconds=4.0)


def test_convert_respects_benchmark_splits(fake_auv, tmp_path):
    convert_anti_uav(fake_auv.parent, tmp_path / "rgb", "rgb", stride=10, crop=640)
    names = {s: {p.stem.rsplit("_", 1)[0] for p in (tmp_path / "rgb" / "images" / s).glob("*")}
             for s in ("train", "val")}
    assert names["train"] == {"seq_000", "seq_001"} and names["val"] == {"seq_002"}  # test seq_003 held out
    img = next((tmp_path / "rgb" / "images" / "train").glob("*.jpg"))
    import cv2
    assert cv2.imread(str(img)).shape[:2] == (540, 640)


def test_evaluate_sequence_classical(fake_auv):
    seq = fake_auv / "test" / "seq_003"
    e, r = load_anti_uav_labels(seq, "ir")
    assert len(e) == 100
    res = evaluate_sequence(seq, min_prob=0.3)
    assert res["calib_rms_px"] < 1.0
    assert res["hit_rate_rgb"] > 0.6
    assert res["acc_rgb"] > 0.2

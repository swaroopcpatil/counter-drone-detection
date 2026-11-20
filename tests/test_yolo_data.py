import json

import cv2
import numpy as np

from counter_drone import yolo_data as yd
from counter_drone.config import Config
from counter_drone.detectors import HybridDetector, RGBSmallTargetDetector
from counter_drone.types import Detection


def _labels(d, split="train"):
    out = []
    for f in sorted((d / "labels" / split).glob("*.txt")):
        out += [list(map(float, ln.split())) for ln in f.read_text().splitlines() if ln.strip()]
    return out


def test_yolo_line_normalises_and_pads_tiny_boxes():
    ln = yd._yolo_line(0, (100, 50, 101, 51), 200, 100, min_px=4)
    c, cx, cy, w, h = map(float, ln.split())
    assert (c, cx, cy) == (0, 100.5 / 200, 50.5 / 100)
    assert w == 4 / 200 and h == 4 / 100
    assert yd._yolo_line(0, (-10, -10, -5, -5), 200, 100) is None


def test_export_synthetic_both_modalities(tmp_path):
    for mod in ("rgb", "thermal"):
        y = yd.export_synthetic(tmp_path / mod, mod, n_scenes=3, scene_seconds=1.0, stride=5)
        assert "names:" in y.read_text() and "drone" in y.read_text()
        imgs = list((tmp_path / mod / "images" / "train").glob("*.png"))
        assert imgs
        img = cv2.imread(str(imgs[0]), cv2.IMREAD_UNCHANGED)
        assert img.shape[:2] == ((540, 960) if mod == "rgb" else (256, 320))
        labs = _labels(tmp_path / mod) + _labels(tmp_path / mod, "val")
        assert labs and all(0 <= v <= 1 for lab in labs for v in lab[1:])


def test_convert_anti_uav_image_sequences(tmp_path):
    root = tmp_path / "anti"
    for s in range(3):
        seq = root / f"seq{s}"
        seq.mkdir(parents=True)
        exist, rects = [], []
        for i in range(10):
            img = np.full((120, 160), 60, np.uint8)
            if i < 8:
                cv2.circle(img, (40 + i * 5, 60), 3, 255, -1)
                exist.append(1)
                rects.append([37 + i * 5, 57, 7, 7])
            else:
                exist.append(0)
                rects.append([])
            cv2.imwrite(str(seq / f"{i:04d}.jpg"), img)
        (seq / "IR_label.json").write_text(json.dumps({"exist": exist, "gt_rect": rects}))
    yd.convert_anti_uav(root, tmp_path / "out", "ir", stride=2, neg_frac=1.0)
    labs = _labels(tmp_path / "out") + _labels(tmp_path / "out", "val")
    assert len(labs) == 3 * 4                      # frames 0,2,4,6 have a drone
    assert all(lab[0] == 0 for lab in labs)
    n_imgs = sum(1 for _ in (tmp_path / "out" / "images").rglob("*.*"))
    assert n_imgs == 3 * 5                         # + the negative frame 8


def test_convert_coco_maps_category_names(tmp_path):
    (tmp_path / "imgs").mkdir()
    cv2.imwrite(str(tmp_path / "imgs" / "a.png"), np.zeros((100, 200, 3), np.uint8))
    coco = {"images": [{"id": 1, "file_name": "a.png", "width": 200, "height": 100}],
            "categories": [{"id": 7, "name": "UAV"}, {"id": 8, "name": "Bird"}, {"id": 9, "name": "car"}],
            "annotations": [{"image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 10]},
                            {"image_id": 1, "category_id": 8, "bbox": [50, 50, 10, 10]},
                            {"image_id": 1, "category_id": 9, "bbox": [90, 50, 10, 10]}]}
    (tmp_path / "c.json").write_text(json.dumps(coco))
    yd.convert_coco(tmp_path / "c.json", tmp_path / "imgs", tmp_path / "out")
    labs = sorted(lab[0] for lab in _labels(tmp_path / "out", "val"))
    assert labs == [0, 1]                          # drone, bird; car dropped


def test_crop_dataset_keeps_native_scale(tmp_path):
    src = tmp_path / "src"
    img = np.zeros((540, 960, 3), np.uint8)
    yd._write(src, "train", "x", img, [yd._yolo_line(0, (470, 260, 490, 280), 960, 540)])
    yd._write(src, "val", "y", img, [])
    yd.crop_dataset(src, tmp_path / "dst", 640, seed=1)
    out = cv2.imread(str(tmp_path / "dst" / "images" / "train" / "x.png"))
    assert out.shape[:2] == (540, 640)
    labs = _labels(tmp_path / "dst")
    if labs:  # object inside the crop: same 20 px size in a 640 px image
        assert abs(labs[0][3] * 640 - 20) < 1e-3


class _FakeYolo:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, image):
        return [Detection(d.box.copy(), d.score, d.source, label=d.label) for d in self.dets]


def test_hybrid_detector_labels_classical_blobs_and_adds_yolo_only():
    img = np.full((300, 400, 3), (230, 190, 140), np.uint8)
    cv2.circle(img, (100, 100), 4, (40, 40, 40), -1)
    cv2.circle(img, (300, 200), 4, (40, 40, 40), -1)
    yolo = _FakeYolo([Detection(np.array([95, 95, 105, 105.]), 0.9, "rgb", label="drone"),
                      Detection(np.array([200, 250, 230, 280.]), 0.6, "rgb", label="bird"),
                      Detection(np.array([20, 250, 40, 270.]), 0.3, "rgb", label="drone")])  # weak, unsupported
    out = HybridDetector(RGBSmallTargetDetector(Config().rgb), yolo).detect(img)
    by = {(round(d.center[0] / 50), round(d.center[1] / 50)): d for d in out}
    assert by[(2, 2)].label == "drone" and by[(2, 2)].score >= 0.9
    assert by[(6, 4)].label is None                 # classical only
    assert by[(4, 5)].label == "bird"               # yolo only
    assert len(out) == 3                            # weak YOLO-only box rejected

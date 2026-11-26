"""Fit the drone-vs-clutter classifier on labelled real tracks (Anti-UAV300).

    python scripts/fit_classifier.py --root datasets/Anti-UAV300 --rgb-yolo runs/yolo/anti_uav_rgb_neg2/weights/best.pt --out runs/classifier_fit

Fits on the train and val sequences, scores on the held-out test sequences, and writes
tracks.jsonl, fitted_weights.json and summary.json to --out. Nothing in the live system
changes until you choose to apply the weights.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from counter_drone.classifier_fit import run  # noqa: E402
from counter_drone.config import Config  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", required=True)
ap.add_argument("--rgb-yolo", required=True)
ap.add_argument("--thermal-yolo", help="optional: fit on infrared tracks instead of RGB")
ap.add_argument("--out", default="runs/classifier_fit")
ap.add_argument("--max-frames", type=int, default=600)
ap.add_argument("--max-seqs", type=int, help="limit sequences for a quick run")
ap.add_argument("--device", default="0")
ap.add_argument("--imgsz", type=int, default=1920)
a = ap.parse_args()

cfg = Config()
cfg.rgb.backend, cfg.rgb.yolo_weights = "yolo", a.rgb_yolo
cfg.rgb.yolo_device, cfg.rgb.yolo_imgsz = a.device, a.imgsz
modality = "rgb"
if a.thermal_yolo:
    cfg.thermal.backend, cfg.thermal.yolo_weights, cfg.thermal.yolo_device = "yolo", a.thermal_yolo, a.device
    modality = "ir"
summary = run(a.root, cfg, modality, a.out, a.max_frames, a.max_seqs)
print(json.dumps(summary, indent=2))

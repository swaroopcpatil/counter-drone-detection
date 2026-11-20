# Counter-drone detection & tracking (RGB + thermal)

Detects, tracks, classifies and geolocates small UAVs using a co-located visible
(RGB) camera and a thermal (LWIR) camera. It runs on live cameras, RTSP streams or
recorded files, and ships with a synthetic sky scene with ground truth so the whole
pipeline can be run, scored and tuned without hardware.

```
 RGB frame ──► small-target detector ─┐
                                      ├─► fusion ─► multi-target ─► drone/clutter ─► threat level,
 thermal ───► hot-spot detector ──────┘  (homography)   tracker      classifier      az/el/range, alerts
```

| Stage | What it does |
|---|---|
| **RGB detector** (`detectors.py`) | Morphological black-hat saliency (dark silhouettes against sky) with a **censored CFAR** threshold: global sensor-noise floor plus local clutter statistics, so cloud edges need more contrast while clean sky is detected at the noise limit. Strong returns are clipped before computing local stats, so one drone can't mask a nearby one. Running-background motion cue. Optional YOLO backend. |
| **Thermal detector** | Percentile AGC (handles 8-bit and 16-bit radiometric Y16) followed by a white top-hat and the same CFAR. Nearby hot spots (motors, ESC, battery) are merged into one target. |
| **Fusion** (`fusion.py`) | Thermal boxes are mapped into the RGB frame by a calibrated homography, then paired by Hungarian assignment. Paired detections get a noisy-OR score. Unpaired RGB blobs inside the thermal FOV are down-weighted. Thermal-only detections are kept, so tracking continues at night or in glare. |
| **Tracker** (`tracking.py`) | Constant-velocity Kalman filter per track (state: position, velocity, size). Mahalanobis gating, Hungarian assignment with confirmed tracks matched first, M-of-N confirmation, and about 1.3 s of coasting through dropouts. |
| **Classifier** | Explainable log-odds model. Each track carries a per-cue breakdown in `track.extra["evidence"]`: **cross-sensor consistency** (real object vs. cloud edge or glint), **wing-beat** (aspect-ratio oscillation; scale-invariant, so a drone approaching and growing isn't penalised), **turn rate** (birds fly erratically), **hover**, and the optional detector **label**. Cues that are temporarily unavailable (RGB blind at dusk) decay from memory instead of vanishing. Thermal *brightness* is deliberately not drone evidence, because birds are warm too. |
| **Threat** (`threat.py`) | Azimuth and elevation from camera intrinsics plus pan/tilt. Monocular range from a nominal drone span (rough, ±50 %). Closing speed. Levels NONE / LOW / MEDIUM / HIGH with hysteresis, debouncing and LOST events, emitted as JSONL. |

## Quick start

```bash
pip install -r requirements.txt   # + `pip install ultralytics` for the YOLO features
python -m counter_drone demo --out runs/demo        # synthetic scene, scored against ground truth
python -m pytest -q tests
```

`runs/demo/` then contains `demo.mp4` (operator view: annotated RGB plus false-colour
thermal), snapshot PNGs, `tracks.jsonl` (per-frame confirmed tracks), `events.jsonl`
(alerts) and `metrics.json`.

### Synthetic benchmark (seed 0, 30 s, 15 fps, 3 drones, 3 birds, clouds, 6 s dusk)

| Metric | Value |
|---|---|
| Detection recall (fused detections on drones) | 1.00 |
| Track coverage of drones | 0.99 |
| Drone declared (p ≥ 0.7): day / dusk (RGB blind) | 0.95 / 1.00 |
| Time to declare | 1.1–1.4 s |
| Bird frames wrongly declared drone | 1 of ~1350 |
| ID switches | 2 |
| Processing time | ~64 ms/frame (960×540 + 320×256, single CPU core, pure NumPy/OpenCV) |

The synthetic scene is a development aid, not evidence of field performance. Tune and
validate on recordings from your own site (see below).

## Running on real cameras

```bash
# 1. Calibrate the thermal->RGB mapping (once per rig, or after re-mounting)
python -m counter_drone calibrate --rgb-image rgb.png --thermal-image th.png --pattern 9x6 --out calib.json
#    ...or from >=4 hand-picked correspondences on distant features:
python -m counter_drone calibrate --points pairs.json --out calib.json   # {"thermal": [[x,y],..], "rgb": [[x,y],..]}

# 2. Dump the default config and edit it for your site / optics
python -m counter_drone config --out site.json

# 3. Run: device indices, files, image sequences (16-bit OK) or rtsp:// URLs
python -m counter_drone run --rgb rtsp://cam1/stream --thermal 1 --homography calib.json \
       --config site.json --out runs/live --record
```

Alerts are printed to stdout as JSON lines, ready to pipe into a C2/alerting system.
Add `--display` for a live window (needs `opencv-python`, not the headless build).

**Notes for real hardware**

- **Calibration:** a heated checkerboard (aluminium squares on a dark board, or a board warmed by the sun) is visible to both sensors. Check the RMS reported by `calibrate` (< 2 px is good). A single homography assumes the cameras are boresighted and the targets are far away compared with the camera baseline, which holds for anything beyond a few tens of metres.
- **Thermal cores:** UVC cores (FLIR Boson/Lepton via PureThermal, etc.) are opened with `CAP_PROP_CONVERT_RGB=0` to get raw Y16. 16-bit PNG/TIFF sequences are read natively. Flat-field corrections (FFC) freeze the image briefly; the tracker coasts through them.
- **`camera` config:** set `hfov_deg`, `pan_deg` and `tilt_deg` for correct azimuth and elevation (on a pan-tilt unit, update pan/tilt every frame). `drone_span_m` sets the scale of the range estimate.
- **Masking:** set `rgb.roi` / `thermal.roi` to exclude ground, trees and buildings below the horizon. The classical detectors are designed for sky backgrounds.
- **Night:** set `rgb.polarity` to `"both"` (lit drones show as bright points). The pipeline already flags the RGB channel as unusable when the image is dark, so missing RGB detections don't count against a track.
- **Learned detectors:** set `rgb.backend` / `thermal.backend` to `"yolo"` and point `yolo_weights` at a model trained on drone data (e.g. Anti-UAV style RGB-T datasets). Detections flow through the same fusion, tracking and classification, and the class label becomes one more evidence term.

## Train on real data: Anti-UAV300 + your GPU (one command)

[Anti-UAV](https://github.com/ZhaoJ9014/Anti-UAV) **Anti-UAV300** is a real-world benchmark of
paired RGB + infrared videos with per-frame drone boxes and visibility flags (train / val / test
splits). `scripts/train_anti_uav.py` uses it end to end on an NVIDIA GPU:

1. checks the GPU and PyTorch build,
2. downloads and extracts Anti-UAV300 (Google Drive, about 10 GB),
3. converts the train and val sequences to YOLO (IR full frames, RGB 960 px native-resolution crops),
4. fine-tunes YOLO11 on IR and on RGB,
5. evaluates the **full RGB+IR pipeline** on the held-out **test** split with the official
   Anti-UAV accuracy. It compares `classical` (no model), `yolo` and `hybrid`, and renders annotated
   videos with the ground truth drawn in yellow.

The RGB and IR cameras in Anti-UAV are not pixel-aligned. The thermal->RGB mapping for each
sequence is fitted from the two label tracks, so the labels are used for calibration only.

**Setup on Windows with an RTX 50-series card (e.g. RTX 5050)**. These GPUs need a PyTorch build
with CUDA 12.8 or newer:

```bash
# latest NVIDIA driver first (check with: nvidia-smi), Python 3.10-3.12 recommended
python -m venv .venv && .venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt ultralytics gdown
python -c "import torch; print(torch.cuda.get_device_name(0))"   # should print your RTX card
```

**Train**

```bash
python scripts/train_anti_uav.py --download                       # downloads, converts, trains, evaluates
# already downloaded (Google Drive quota errors happen: then download in a browser or from Baidu):
python scripts/train_anti_uav.py --data-dir D:/datasets/Anti-UAV300
```

Useful options: `--model yolo11n.pt|yolo11s.pt|yolo11m.pt` (speed vs. accuracy), `--epochs 60`,
`--stride 5` (lower = more frames), `--rgb-batch 4` if you run out of GPU memory (8 GB cards),
`--only ir`. Each stage is skipped if its output exists, so you can rerun after an interruption.
Expect roughly 2-5 h on an RTX 5050 with the defaults.

Outputs: `runs/yolo/anti_uav_{ir,rgb}/weights/best.pt`, training curves in the same folders,
`runs/anti_uav_eval/summary.json` (accuracy per mode) and `runs/anti_uav_eval/yolo/videos/*.mp4`.
At the end the script prints the exact command to run the trained models **in real time on your own
cameras**, for example:

```bash
python -m counter_drone run --rgb 0 --thermal 1 --homography calib.json \
  --rgb-yolo runs/yolo/anti_uav_rgb/weights/best.pt --thermal-yolo runs/yolo/anti_uav_ir/weights/best.pt \
  --yolo-mode hybrid --yolo-device 0 --rgb-imgsz 1280 --thermal-imgsz 640 --display
```

Score any weights on the test split yourself with `python -m counter_drone anti-uav-eval --root
<Anti-UAV300>/test --rgb-yolo ... --thermal-yolo ... --yolo-device 0 --render 3`.

Anti-UAV contains only drones (no labelled birds). The trained models learn "drone vs. background",
and bird rejection still comes from the tracker's kinematic cues. For the best results on your site,
add your own footage: draft labels with `yolo-autolabel`, correct them, and fine-tune from the Anti-UAV
`best.pt` (`yolo-train --model runs/yolo/anti_uav_rgb/weights/best.pt ...`).

## Training YOLO models (drone / bird identification)

The classical detectors find *something small in the sky*; a YOLO model trained on drone
imagery says *what it is*. The default way to combine them is the **hybrid** backend: the classical
detector keeps recall on tiny targets, YOLO adds class labels (and any targets the classical
detector missed). Labels feed the classifier as strong evidence and are shown in the video
(`yolo:drone`, `yolo:bird`). One model per sensor; classes are `0 = drone, 1 = bird`.

```bash
pip install ultralytics            # pulls PyTorch; use a CUDA machine for real training

# 1a. Real data: Anti-UAV style RGB-T benchmark (per-sequence infrared.json/visible.json or IR_label.json)
python -m counter_drone yolo-dataset anti-uav --src /data/Anti-UAV410 --modality thermal --out datasets/auv_ir
python -m counter_drone yolo-dataset anti-uav --src /data/Anti-UAV410 --modality rgb     --out datasets/auv_rgb
# 1b. Real data: anything exported as COCO JSON (CVAT, Label Studio, Roboflow)
python -m counter_drone yolo-dataset coco --src ann.json --images imgs/ --out datasets/site_rgb
# 1c. Your own footage, no labels yet: draft labels from the tracker, then review/correct them
python -m counter_drone yolo-autolabel --rgb day1_rgb.mp4 --thermal day1_ir.mp4 --homography calib.json --out datasets/site_draft
# 1d. Synthetic bootstrap data (perfect labels, not real imagery)
python -m counter_drone yolo-dataset synthetic --modality rgb --out datasets/syn_rgb

# 2. Small targets: train on native-resolution crops instead of downscaled frames
python -m counter_drone yolo-dataset crop --src datasets/auv_rgb --out datasets/auv_rgb_crops --size 640

# 3. Train (GPU: --device 0). Start from COCO-pretrained yolo11n/s/m.pt or fine-tune an earlier best.pt
python -m counter_drone yolo-train --data datasets/auv_rgb_crops/data.yaml --imgsz 640 --epochs 100 --device 0 --name rgb
python -m counter_drone yolo-train --data datasets/auv_ir/data.yaml       --imgsz 640 --epochs 100 --device 0 --name thermal

# 4. Use the weights (hybrid backend); also works with `run` on live cameras
python -m counter_drone demo --rgb-yolo runs/yolo/rgb/weights/best.pt --thermal-yolo runs/yolo/thermal/weights/best.pt
```

Set `yolo_imgsz` to the full frame width at inference (the model was trained on crops at native
scale) and `yolo_device` to `0` on a GPU. Mix real and synthetic data by listing several image
folders under `train:` in a `data.yaml`. Always evaluate on footage from your own site and
cameras: a model trained on one sensor/lens/background does not automatically transfer.

## Layout

```
counter_drone/
  config.py        all tunables (dataclasses <-> JSON)
  detectors.py     RGB small-target, thermal hot-spot, YOLO, hybrid
  yolo_data.py     YOLO datasets (synthetic, Anti-UAV, COCO, auto-label, crops) + training
  calibration.py   homography from points / checkerboard, box mapping
  fusion.py        RGB-thermal late fusion
  tracking.py      Kalman CV filter, multi-target tracker, drone classifier
  threat.py        az/el/range, closing speed, threat levels + events
  pipeline.py      per-frame orchestration
  sources.py       cameras / files / streams, threaded capture, pairing
  synthetic.py     synthetic RGB+thermal scenes with ground truth
  evaluate.py      metrics vs ground truth
  visualize.py     operator display
  anti_uav.py      Anti-UAV benchmark evaluation (official accuracy), fake-dataset generator
scripts/
  train_anti_uav.py  one-command GPU training + evaluation on Anti-UAV300
tests/             unit tests + synthetic end-to-end regression
```

## Limitations / next steps

- Range is monocular and relies on an assumed drone size. A second sensor (radar, RF, or a stereo baseline) gives real range.
- Classifier weights are hand-set and checked against the synthetic scene. With labelled site recordings, fit them (they are logistic-regression coefficients) or replace the classifier with a learned one.
- No pan-tilt control loop yet: `TrackSnapshot.azimuth_deg/elevation_deg` of the highest-threat track is the natural slew target.
- Frame pairing for live sources uses the latest frame from each sensor. For fast targets, use hardware-triggered or timestamped cameras.

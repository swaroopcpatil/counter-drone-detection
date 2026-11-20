"""YOLO dataset tooling: build, convert and auto-label training data.

Classes (shared by every dataset this module writes):  0 = drone, 1 = bird

* ``export_synthetic``  - randomized synthetic RGB / thermal scenes, perfectly labelled.
                          Good for bootstrapping and regression tests, NOT a substitute
                          for real imagery.
* ``convert_anti_uav``  - Anti-UAV style RGB-T benchmarks (per-sequence ``infrared.json`` /
                          ``visible.json`` or ``IR_label.json`` with ``exist`` + ``gt_rect``).
* ``convert_coco``      - any COCO-JSON annotated dataset (CVAT, Label Studio, Roboflow exports).
* ``autolabel``         - run the classical tracker on your own footage and write draft
                          YOLO labels for confirmed drone tracks, to correct in a labelling
                          tool instead of drawing every box by hand.

Every writer produces the standard ultralytics layout::

    <out>/images/{train,val}/*.png   <out>/labels/{train,val}/*.txt   <out>/data.yaml
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["drone", "bird"]


# --------------------------------------------------------------------------- common
def _yolo_line(cls: int, box, img_w: int, img_h: int, min_px: float = 3.0) -> str | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = max(0.0, x1), min(img_w, x2)
    y1, y2 = max(0.0, y1), min(img_h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max(x2 - x1, min_px), max(y2 - y1, min_px)  # tiny targets: give the box a minimum size
    return f"{cls} {cx / img_w:.6f} {cy / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def _write(out: Path, split: str, name: str, img: np.ndarray, lines: list[str], ext: str = "png"):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == "jpg" else []
    cv2.imwrite(str(out / "images" / split / f"{name}.{ext}"), img, params)
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def write_data_yaml(out: Path, names=CLASSES) -> Path:
    out = Path(out).resolve()
    y = out / "data.yaml"
    y.write_text(f"path: {out}\ntrain: images/train\nval: images/val\n"
                 f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    return y


def thermal_to_8bit(raw: np.ndarray) -> np.ndarray:
    """Same AGC the pipeline uses, so train and inference images match."""
    from .detectors import normalize_thermal
    return (normalize_thermal(raw) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- synthetic
def export_synthetic(out: str | Path, modality: str = "rgb", n_scenes: int = 60, scene_seconds: float = 8.0,
                     stride: int = 6, val_frac: float = 0.2, seed0: int = 1000) -> Path:
    """Randomized synthetic scenes -> YOLO dataset. Split is by scene (no near-duplicate leakage)."""
    from .calibration import map_boxes
    from .synthetic import SyntheticScene

    out = Path(out)
    n_val = max(1, int(n_scenes * val_frac))
    for k in range(n_scenes):
        split = "val" if k < n_val else "train"
        dusk = (-1, -1) if modality == "rgb" else (scene_seconds * 0.4, scene_seconds * 0.7)
        sc = SyntheticScene(seed=seed0 + k, duration=scene_seconds, dusk=dusk, randomize=True)
        for i in range(0, len(sc), stride):
            _, rgb, thermal, gts = sc.frame(i)
            lines = []
            if modality == "rgb":
                img, (w, h) = rgb, sc.rgb_size
                for g in gts:
                    ln = _yolo_line(CLASSES.index(g.cls), g.box, w, h, 4.0)
                    lines += [ln] if ln else []
            else:
                img, (w, h) = thermal_to_8bit(thermal), sc.thermal_size
                for g in gts:
                    if not g.in_thermal_fov:
                        continue
                    ln = _yolo_line(CLASSES.index(g.cls), map_boxes(g.box[None], sc.H_rgb_to_thermal)[0], w, h, 3.0)
                    lines += [ln] if ln else []
            _write(out, split, f"s{seed0 + k:05d}_{i:04d}", img, lines)
    return write_data_yaml(out)


# --------------------------------------------------------------------------- real datasets
def _frames_from(seq: Path, modality: str):
    """Yields (index, image) for an Anti-UAV sequence: video file or image folder."""
    stem = "infrared" if modality == "ir" else "visible"
    for ext in (".mp4", ".avi"):
        v = seq / f"{stem}{ext}"
        if v.exists():
            cap = cv2.VideoCapture(str(v))
            i = 0
            while True:
                ok, f = cap.read()
                if not ok:
                    return
                yield i, f
                i += 1
    imgs = sorted([p for p in seq.iterdir() if p.suffix.lower() in (".jpg", ".png", ".bmp")])
    for i, p in enumerate(imgs):
        yield i, cv2.imread(str(p), cv2.IMREAD_UNCHANGED)


def anti_uav_sequences(root: str | Path, modality: str = "ir") -> list[Path]:
    names = ["infrared.json", "IR_label.json"] if modality == "ir" else ["visible.json", "RGB_label.json"]
    return sorted({p.parent for n in names for p in Path(root).rglob(n)})


def load_anti_uav_labels(seq: Path, modality: str) -> tuple[list[int], list[list[float]]]:
    """Returns (exist flags, xywh boxes) with absent frames normalised to exist=0 / []."""
    names = ["infrared.json", "IR_label.json"] if modality == "ir" else ["visible.json", "RGB_label.json"]
    lab = next(seq / n for n in names if (seq / n).exists())
    ann = json.loads(lab.read_text())
    rects = ann.get("gt_rect", [])
    exist = ann.get("exist", [1] * len(rects))
    out_e, out_r = [], []
    for i, r in enumerate(rects):
        ok = bool(exist[i] if i < len(exist) else 1) and len(r) == 4 and r[2] > 0 and r[3] > 0
        out_e.append(int(ok))
        out_r.append(list(map(float, r)) if ok else [])
    return out_e, out_r


def _split_of(seq: Path, root: Path) -> str | None:
    """Use the benchmark's own split when the folder tree has one (train/val/test)."""
    for part in seq.relative_to(root).parts[:-1]:
        p = part.lower()
        if p in ("train", "training"):
            return "train"
        if p in ("val", "valid", "validation"):
            return "val"
        if p in ("test", "testing", "test-dev", "test-challenge"):
            return "test"
    return None


def convert_anti_uav(root: str | Path, out: str | Path, modality: str = "ir", stride: int = 5,
                     neg_frac: float = 0.1, val_frac: float = 0.15, seed: int = 0, crop: int | None = None,
                     ext: str = "jpg") -> Path:
    """Anti-UAV (300/410/600-style) -> YOLO.

    Each sequence directory contains ``infrared.json``/``visible.json`` (or ``IR_label.json``)
    with ``{"exist": [0|1, ...], "gt_rect": [[x, y, w, h] | [] | [0,0,0,0], ...]}`` and either a
    video (``infrared.mp4``/``visible.mp4``) or the frames as images. The benchmark's own
    train/val split is kept when the tree has one (test sequences are skipped - keep them for
    ``anti-uav-eval``); otherwise sequences are split randomly. Every ``stride``-th labelled
    frame is kept, plus a ``neg_frac`` share of drone-free frames as negatives.

    ``crop`` (e.g. 960 for 1080p RGB) saves one native-resolution square-ish crop per frame,
    placed randomly but containing the drone: small targets keep their pixel size and the
    dataset stays a fraction of the size of full frames.
    """
    root, out = Path(root), Path(out)
    rnd = random.Random(seed)
    seqs = anti_uav_sequences(root, modality)
    if not seqs:
        raise FileNotFoundError(f"no Anti-UAV label files for modality {modality!r} under {root}")
    rnd.shuffle(seqs)
    n_val = max(1, int(len(seqs) * val_frac))
    has_splits = any(_split_of(s, root) for s in seqs)
    for k, seq in enumerate(seqs):
        split = _split_of(seq, root) if has_splits else ("val" if k < n_val else "train")
        if split not in ("train", "val"):
            continue
        exist, rects = load_anti_uav_labels(seq, modality)
        for i, img in _frames_from(seq, modality):
            if img is None or i >= len(rects) or i % stride:
                continue
            if not exist[i] and rnd.random() > neg_frac:
                continue
            if img.dtype != np.uint8:
                img = thermal_to_8bit(img)
            h, w = img.shape[:2]
            box = None
            if exist[i]:
                x, y, bw, bh = rects[i]
                box = np.array([x, y, x + bw, y + bh])
            if crop and (w > crop or h > crop):
                cw, ch = min(crop, w), min(crop, h)
                if box is not None:   # window containing the target (if it fits)
                    lo_x, hi_x = int(max(0, box[2] - cw)), int(min(w - cw, box[0]))
                    lo_y, hi_y = int(max(0, box[3] - ch)), int(min(h - ch, box[1]))
                else:
                    lo_x, hi_x, lo_y, hi_y = 0, w - cw, 0, h - ch
                x0 = rnd.randint(lo_x, max(lo_x, hi_x))
                y0 = rnd.randint(lo_y, max(lo_y, hi_y))
                img = img[y0:y0 + ch, x0:x0 + cw]
                if box is not None:
                    box = box - [x0, y0, x0, y0]
                h, w = ch, cw
            lines = []
            if box is not None:
                ln = _yolo_line(0, box, w, h)
                lines += [ln] if ln else []
            _write(out, split, f"{seq.name}_{i:05d}", img, lines, ext)
    if not (out / "images" / "val").exists():  # benchmark had no val split: carve one from train
        tr = sorted((out / "images" / "train").glob(f"*.{ext}"))
        seq_names = sorted({p.stem.rsplit("_", 1)[0] for p in tr})
        for name in seq_names[: max(1, int(len(seq_names) * val_frac))]:
            for p in (out / "images" / "train").glob(f"{name}_*.{ext}"):
                for src, dst in ((p, out / "images" / "val" / p.name),
                                 (out / "labels" / "train" / f"{p.stem}.txt", out / "labels" / "val" / f"{p.stem}.txt")):
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    src.rename(dst)
    return write_data_yaml(out)


def convert_coco(coco_json: str | Path, images_dir: str | Path, out: str | Path,
                 class_map: dict[str, str] | None = None, val_frac: float = 0.15, seed: int = 0) -> Path:
    """COCO JSON -> YOLO. ``class_map`` maps source category names to 'drone'/'bird'
    (default: names containing drone/uav/uas/quad -> drone, bird -> bird, others dropped)."""
    d = json.loads(Path(coco_json).read_text())
    images_dir, out = Path(images_dir), Path(out)

    def target(name: str):
        n = name.lower()
        if class_map:
            return class_map.get(name)
        if any(k in n for k in ("drone", "uav", "uas", "quad", "copter")):
            return "drone"
        return "bird" if "bird" in n else None

    cat = {c["id"]: target(c["name"]) for c in d["categories"]}
    anns: dict[int, list] = {}
    for a in d["annotations"]:
        if cat.get(a["category_id"]):
            anns.setdefault(a["image_id"], []).append(a)
    imgs = list(d["images"])
    random.Random(seed).shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_frac))
    for k, im in enumerate(imgs):
        img = cv2.imread(str(images_dir / im["file_name"]), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.dtype != np.uint8:
            img = thermal_to_8bit(img)
        h, w = img.shape[:2]
        lines = []
        for a in anns.get(im["id"], []):
            x, y, bw, bh = a["bbox"]
            ln = _yolo_line(CLASSES.index(cat[a["category_id"]]), (x, y, x + bw, y + bh), w, h)
            lines += [ln] if ln else []
        _write(out, "val" if k < n_val else "train", Path(im["file_name"]).stem, img, lines)
    return write_data_yaml(out)


# --------------------------------------------------------------------------- auto-label
def autolabel(rgb_spec: str | None, thermal_spec: str | None, out: str | Path, homography=None,
              config=None, stride: int = 10, min_prob: float = 0.7, val_frac: float = 0.15) -> dict:
    """Draft labels from the classical tracker on real footage (review before training!).

    Confirmed tracks with p(drone) >= ``min_prob`` -> class drone. Labels are written for
    whichever sensor(s) are given; thermal boxes are mapped back through the homography.
    """
    from .calibration import map_boxes
    from .config import Config
    from .pipeline import CounterDronePipeline
    from .sources import PairedSource

    out = Path(out)
    src = PairedSource(rgb_spec, thermal_spec)
    frames = iter(src)
    first = next(frames)
    _, r0, t0 = first
    rgb_size = (r0.shape[1], r0.shape[0]) if r0 is not None else None
    th_size = (t0.shape[1], t0.shape[0]) if t0 is not None else None
    H = np.asarray(homography, float) if homography is not None else (
        np.diag([rgb_size[0] / th_size[0], rgb_size[1] / th_size[1], 1.0]) if rgb_size and th_size else np.eye(3))
    pipe = CounterDronePipeline(config or Config(), H, rgb_size or th_size, th_size or rgb_size)
    Hinv = np.linalg.inv(H)
    stats = {"frames": 0, "labelled_frames": 0, "boxes": 0}
    import itertools
    for i, (t, rgb, th) in enumerate(itertools.chain([first], frames)):
        res = pipe.process(rgb, th, t)
        if i % stride:
            continue
        boxes = [s.box for s in res.tracks if s.confirmed and s.drone_prob >= min_prob]
        split = "val" if (i // stride) % int(round(1 / val_frac)) == 0 else "train"
        if rgb is not None:
            lines = [ln for b in boxes if (ln := _yolo_line(0, b, rgb.shape[1], rgb.shape[0], 4.0))]
            _write(out / "rgb", split, f"f{i:06d}", rgb, lines)
        if th is not None:
            tb = map_boxes(np.array(boxes), Hinv) if boxes else []
            lines = [ln for b in tb if (ln := _yolo_line(0, b, th.shape[1], th.shape[0], 3.0))]
            _write(out / "thermal", split, f"f{i:06d}", thermal_to_8bit(th), lines)
        stats["frames"] += 1
        stats["labelled_frames"] += bool(boxes)
        stats["boxes"] += len(boxes)
    src.release()
    for sub in ("rgb", "thermal"):
        if (out / sub).exists():
            write_data_yaml(out / sub)
    return stats


# --------------------------------------------------------------------------- training
def train(data: str, model: str = "yolo11n.pt", imgsz: int = 960, epochs: int = 50, batch: int = 8,
          device: str = "cpu", project: str = "runs/yolo", name: str = "train", workers: int = 2,
          **overrides) -> Path:
    """Fine-tune a COCO-pretrained YOLO on a drone dataset. Returns path to best.pt.

    Defaults are tuned for small aerial targets: train at (near) native resolution so a
    6 px drone is not shrunk to 3 px, no vertical flips (sky is up), mild colour/scale
    jitter, mosaic for context variety, closed for the last epochs.
    """
    from ultralytics import YOLO

    args = dict(data=str(data), imgsz=imgsz, epochs=epochs, batch=batch, device=device, workers=workers,
                project=str(Path(project).resolve()), name=name, exist_ok=True, patience=max(10, epochs // 4),
                fliplr=0.5, flipud=0.0, degrees=0.0, scale=0.3, translate=0.1, mosaic=1.0,
                close_mosaic=max(1, epochs // 5), hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, plots=True, verbose=False)
    args.update(overrides)
    YOLO(model).train(**args)
    return Path(args["project"]) / name / "weights" / "best.pt"


def training_state(run_dir: str | Path) -> tuple[str, int | None]:
    """('none' | 'partial' | 'done', last finished epoch) for an ultralytics run folder.

    ultralytics writes ``weights/last.pt`` after every epoch with the epoch number, and sets
    it to -1 when training ends normally (all epochs or early stop).
    """
    last = Path(run_dir) / "weights" / "last.pt"
    if not last.exists():
        return ("done" if (Path(run_dir) / "weights" / "best.pt").exists() else "none"), None
    import torch
    ep = torch.load(last, map_location="cpu", weights_only=False).get("epoch", -1)
    return ("done", None) if ep == -1 else ("partial", int(ep) + 1)


def resume(run_dir: str | Path, **overrides) -> Path:
    """Continue an interrupted training run from ``weights/last.pt`` (optimizer, LR schedule, epoch).

    ``overrides`` such as batch / workers / device / cache may be changed on resume.
    """
    from ultralytics import YOLO

    last = Path(run_dir) / "weights" / "last.pt"
    YOLO(str(last)).train(resume=True, **overrides)
    return Path(run_dir) / "weights" / "best.pt"


def capture_negatives(source: str, out: str | Path, frames: int = 300, every: int = 5,
                      val_frac: float = 0.1) -> Path:
    """Record drone-FREE frames (faces, rooms, buildings, birds ...) as background images.

    Each image gets an empty label file, which tells YOLO 'nothing to detect here'. Adding
    such hard negatives from the deployment camera and fine-tuning removes false alarms the
    original training data never covered (e.g. eyes in a close-up face). Make sure no drone
    is visible while recording. ``source`` = camera index ("0") or a video file.
    """
    out = Path(out)
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {source!r}")
    n_val = max(1, int(frames * val_frac))
    kept = i = 0
    print(f"recording {frames} drone-free frames from {source} (every {every}th) - keep drones out of view; "
          "move around, show faces, room, windows", flush=True)
    while kept < frames:
        ok, img = cap.read()
        if not ok:
            break
        if i % every == 0:
            split = "val" if kept % max(1, frames // n_val) == 0 else "train"
            _write(out, split, f"neg_{kept:05d}", img, [], "jpg")
            kept += 1
            if kept % 50 == 0:
                print(f"  {kept}/{frames}", flush=True)
        i += 1
    cap.release()
    print(f"kept {kept} negative frames in {out}")
    return write_data_yaml(out)


def merge_datasets(base: str | Path, extras: list, out_yaml: str | Path) -> Path:
    """One data.yaml whose train/val lists span several dataset folders (same class list)."""
    dirs = [Path(base).resolve()] + [Path(e).resolve() for e in extras]
    def split(name):
        return [str(d / "images" / name) for d in dirs if (d / "images" / name).exists()]
    out_yaml = Path(out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    lines = ["train:"] + [f"  - {p}" for p in split("train")] + ["val:"] + [f"  - {p}" for p in split("val")]
    lines += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(CLASSES)]
    out_yaml.write_text("\n".join(lines) + "\n")
    return out_yaml


def crop_dataset(src: str | Path, dst: str | Path, size: int = 640, seed: int = 0, min_keep: float = 0.5) -> Path:
    """One random native-resolution ``size`` crop per image (small-object training).

    Training on crops instead of downscaled full frames keeps tiny targets at their real
    pixel size and cuts CPU/GPU cost; YOLO is fully convolutional, so the model is then run
    on full frames at inference (``yolo_imgsz`` = full frame width).
    """
    src, dst = Path(src), Path(dst)
    rnd = random.Random(seed)
    for split in ("train", "val"):
        for ip in sorted((src / "images" / split).glob("*.png")):
            img = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
            h, w = img.shape[:2]
            cw, ch = min(size, w), min(size, h)
            x0, y0 = rnd.randint(0, w - cw), rnd.randint(0, h - ch)
            lines = []
            for ln in (src / "labels" / split / f"{ip.stem}.txt").read_text().split("\n"):
                if not ln.strip():
                    continue
                c, cx, cy, bw, bh = ln.split()
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                b = np.array([cx - bw / 2 - x0, cy - bh / 2 - y0, cx + bw / 2 - x0, cy + bh / 2 - y0])
                cb = np.clip(b, 0, [cw, ch, cw, ch])
                if (cb[2] - cb[0]) * (cb[3] - cb[1]) < min_keep * bw * bh:
                    continue  # mostly cut off
                out = _yolo_line(int(c), cb, cw, ch)
                lines += [out] if out else []
            _write(dst, split, ip.stem, img[y0:y0 + ch, x0:x0 + cw], lines)
    return write_data_yaml(dst)

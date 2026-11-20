"""Evaluate the full RGB+thermal pipeline on Anti-UAV benchmark sequences.

Each Anti-UAV300 sequence holds a synchronised ``visible.mp4`` / ``infrared.mp4`` pair and
per-frame labels (``visible.json`` / ``infrared.json``: ``exist`` flag + ``gt_rect`` xywh).
The two cameras are not pixel-aligned, so the thermal->RGB mapping is fitted from the two
label tracks (``homography_from_box_tracks``) - labels are used for *calibration only*,
never for detection.

Scoring uses the Anti-UAV challenge accuracy (single target per frame)::

    acc = sum_t [ IoU_t * d(v_t>0) + p_t * (1 - d(v_t>0)) ] / T  -  0.2 * (sum_t p_t * d(v_t>0) / T*) ** 0.3

with v_t the ground-truth visibility, p_t = 1 when the tracker reports "no target". The
pipeline's prediction for a frame is its confirmed track with the highest p(drone) (if
>= ``min_prob``). Results are reported for both the IR and RGB views.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from .calibration import homography_from_box_tracks, map_boxes
from .config import Config
from .pipeline import CounterDronePipeline
from .sources import PairedSource
from .yolo_data import anti_uav_sequences, load_anti_uav_labels


def iou_xyxy(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def anti_uav_accuracy(ious, pred_absent, gt_visible) -> float:
    ious, p, v = np.asarray(ious, float), np.asarray(pred_absent, float), np.asarray(gt_visible, bool)
    T = len(v)
    if T == 0:
        return float("nan")
    acc = float(np.sum(np.where(v, ious, p)) / T)
    if v.any():
        acc -= 0.2 * float(np.sum(p * v) / v.sum()) ** 0.3
    return acc


def _xywh_to_xyxy(r):
    return np.array([r[0], r[1], r[0] + r[2], r[1] + r[3]], float)


def _find_video(seq: Path, stem: str) -> str | None:
    for ext in (".mp4", ".avi", ".mkv"):
        if (seq / f"{stem}{ext}").exists():
            return str(seq / f"{stem}{ext}")
    return None


def evaluate_sequence(seq: str | Path, cfg: Config | None = None, min_prob: float = 0.5,
                      render_path: str | Path | None = None, max_frames: int | None = None) -> dict:
    seq = Path(seq)
    cfg = cfg or Config()
    ve, vr = load_anti_uav_labels(seq, "rgb")
    ie, ir = load_anti_uav_labels(seq, "ir")
    H, rms, model = homography_from_box_tracks(vr, ir, ve, ie)
    Hinv = np.linalg.inv(H)

    src = PairedSource(_find_video(seq, "visible"), _find_video(seq, "infrared"))
    frames = iter(src)
    first = next(frames)
    _, r0, t0 = first
    rgb_size, th_size = (r0.shape[1], r0.shape[0]), (t0.shape[1], t0.shape[0])
    pipe = CounterDronePipeline(cfg, H, rgb_size, th_size)

    vw = None
    rows = {"rgb": ([], [], []), "ir": ([], [], [])}   # ious, pred_absent, gt_visible
    n = min(len(vr), len(ir))
    for i, (t, rgb, th) in enumerate(itertools.chain([first], frames)):
        if i >= n or (max_frames and i >= max_frames):
            break
        res = pipe.process(rgb, th, t)
        cands = [s for s in res.tracks if s.confirmed and s.drone_prob >= min_prob]
        best = best_target(res.tracks, min_prob)

        pred_rgb = best.box if best is not None else None
        pred_ir = None
        if best is not None:
            pred_ir = map_boxes(best.box[None], Hinv)[0]
            # prefer the thermal detector's own box (IR resolution) when one supports the track
            c = (pred_ir[:2] + pred_ir[2:]) / 2
            near = [d for d in res.thermal_dets if np.linalg.norm(d.center - c) < max(6.0, max(d.wh))]
            if near:
                pred_ir = min(near, key=lambda d: np.linalg.norm(d.center - c)).box
            elif not (0 <= c[0] < th_size[0] and 0 <= c[1] < th_size[1]):
                pred_ir = None  # target is outside the IR camera's field of view

        for key, pred, ex, rects in (("rgb", pred_rgb, ve, vr), ("ir", pred_ir, ie, ir)):
            vis = bool(ex[i])
            iou = iou_xyxy(pred, _xywh_to_xyxy(rects[i])) if (vis and pred is not None) else 0.0
            rows[key][0].append(iou)
            rows[key][1].append(pred is None)
            rows[key][2].append(vis)

        if render_path:
            from .visualize import render
            frame = render(rgb, res, H, th_size, show_dets=False)
            sc = frame.shape[0] / th_size[1]
            if ve[i]:
                x1, y1, x2, y2 = _xywh_to_xyxy(vr[i]).astype(int)
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 255), 1)
            if ie[i]:
                x1, y1, x2, y2 = (_xywh_to_xyxy(ir[i]) * sc).astype(int) + [rgb_size[0], 0, rgb_size[0], 0]
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 255), 1)
            cv2.putText(frame, "yellow = ground truth", (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
            if vw is None:
                Path(render_path).parent.mkdir(parents=True, exist_ok=True)
                vw = cv2.VideoWriter(str(render_path), cv2.VideoWriter_fourcc(*"mp4v"), 25,
                                     (frame.shape[1], frame.shape[0]))
            vw.write(frame)
    src.release()
    if vw is not None:
        vw.release()

    out = {"sequence": seq.name, "frames": len(rows["ir"][0]), "calib_model": model, "calib_rms_px": round(rms, 2)}
    for key, (ious, absent, vis) in rows.items():
        vis_a = np.asarray(vis, bool)
        ious_a = np.asarray(ious)
        out[f"acc_{key}"] = round(anti_uav_accuracy(ious, absent, vis), 4)
        out[f"hit_rate_{key}"] = round(float(np.mean(ious_a[vis_a] > 0.3)) if vis_a.any() else float("nan"), 4)
        out[f"mean_iou_{key}"] = round(float(np.mean(ious_a[vis_a])) if vis_a.any() else float("nan"), 4)
    return out


def evaluate_dataset(root: str | Path, cfg: Config | None = None, out_dir: str | Path | None = None,
                     render: int = 0, max_frames: int | None = None, min_prob: float = 0.5) -> dict:
    """Evaluate every sequence under ``root`` (e.g. ``Anti-UAV-RGBT/test``)."""
    seqs = [s for s in anti_uav_sequences(root, "ir") if _find_video(s, "visible") and _find_video(s, "infrared")]
    if not seqs:
        raise FileNotFoundError(f"no sequences with visible+infrared videos under {root}")
    out_dir = Path(out_dir) if out_dir else None
    results = []
    for k, seq in enumerate(seqs):
        rp = out_dir / "videos" / f"{seq.name}.mp4" if (out_dir and k < render) else None
        try:
            r = evaluate_sequence(seq, cfg, min_prob, rp, max_frames)
        except (ValueError, RuntimeError) as e:  # e.g. too few co-labelled frames to calibrate
            r = {"sequence": seq.name, "error": str(e)}
        results.append(r)
        print(json.dumps(r), flush=True)
    ok = [r for r in results if "error" not in r]
    summary = {"sequences": len(results), "evaluated": len(ok)}
    for key in ("acc_ir", "acc_rgb", "hit_rate_ir", "hit_rate_rgb", "mean_iou_ir", "mean_iou_rgb"):
        vals = [r[key] for r in ok if not np.isnan(r[key])]
        summary[key] = round(float(np.mean(vals)), 4) if vals else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "per_sequence.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def best_target(snapshots, min_prob: float, strong_det: float = 0.5, min_hits: int = 2):
    """Single-target choice.

    1. A track matched THIS frame to a confident detection (>= ``strong_det``) wins, the most
       confident first: when the detector is sure, follow it (even a just-started track,
       so a fragmented track cannot hand the pick to an older false one).
    2. Otherwise the confirmed track the system is most sure is a drone, weighting its
       classification by recent detection confidence - this is where the tracker earns
       its keep, bridging frames the detector misses.
    """
    strong = [s for s in snapshots if s.det_score >= strong_det and s.hits >= min_hits]
    if strong:
        return max(strong, key=lambda s: (s.det_score, s.drone_prob))
    cands = [s for s in snapshots if s.confirmed and s.drone_prob >= min_prob]
    return max(cands, key=lambda s: (s.drone_prob * max(s.score, 0.05), s.age)) if cands else None


def _single_sensor_cfg(cfg: Config, modality: str) -> Config:
    """A pipeline that runs ONE camera through its YOLO model (no fusion, no calibration)."""
    src = cfg.rgb if modality == "rgb" else cfg.thermal
    if not src.yolo_weights:
        raise ValueError(f"no YOLO weights given for {modality}")
    c = Config()
    c.rgb.backend = "yolo"
    c.rgb.yolo_weights, c.rgb.yolo_imgsz = src.yolo_weights, src.yolo_imgsz
    c.rgb.yolo_device, c.rgb.yolo_conf = src.yolo_device, src.yolo_conf
    c.rgb.min_mean_brightness = 0.0  # infrared frames are often dark; still a usable image
    return c


def evaluate_sequence_single(seq: str | Path, cfg: Config, modality: str, min_prob: float = 0.3,
                             max_frames: int | None = None, det_conf: float = 0.25) -> dict:
    """Score one camera of one sequence on its own, two ways:

    * ``det``: the detector's single most confident drone box per frame (no tracking) -
      what the trained model itself achieves;
    * ``trk``: the same detections through the tracker + classifier (best confirmed track).
    """
    seq = Path(seq)
    exist, rects = load_anti_uav_labels(seq, modality)
    cap = cv2.VideoCapture(_find_video(seq, "visible" if modality == "rgb" else "infrared"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    c = _single_sensor_cfg(cfg, modality)
    pipe = None
    rows = {"det": ([], [], []), "trk": ([], [], [])}
    i = 0
    while True:
        if i >= len(rects) or (max_frames and i >= max_frames):
            break
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if pipe is None:
            size = (frame.shape[1], frame.shape[0])
            pipe = CounterDronePipeline(c, np.eye(3), size, size)
        res = pipe.process(frame, None, i / fps)
        dets = [d for d in res.rgb_dets if d.label in (None, "drone") and d.score >= det_conf]
        det_box = max(dets, key=lambda d: d.score).box if dets else None
        cands = [s for s in res.tracks if s.confirmed and s.drone_prob >= min_prob]
        best = best_target(res.tracks, min_prob)
        trk_box = best.box if best is not None else None
        vis = bool(exist[i])
        gt = _xywh_to_xyxy(rects[i]) if vis else None
        for key, pred in (("det", det_box), ("trk", trk_box)):
            rows[key][0].append(iou_xyxy(pred, gt) if (vis and pred is not None) else 0.0)
            rows[key][1].append(pred is None)
            rows[key][2].append(vis)
        i += 1
    cap.release()
    out = {"sequence": seq.name, "modality": modality, "frames": i}
    for key, (ious, absent, vis) in rows.items():
        v, io = np.asarray(vis, bool), np.asarray(ious)
        out[f"{key}_acc"] = round(anti_uav_accuracy(ious, absent, vis), 4)
        out[f"{key}_hit_rate"] = round(float(np.mean(io[v] > 0.3)) if v.any() else float("nan"), 4)
        out[f"{key}_mean_iou"] = round(float(np.mean(io[v])) if v.any() else float("nan"), 4)
    return out


def evaluate_dataset_per_sensor(root: str | Path, cfg: Config, out_dir: str | Path | None = None,
                                max_frames: int | None = None, min_prob: float = 0.3,
                                max_seqs: int | None = None) -> dict:
    """Per-camera evaluation (see ``evaluate_sequence_single``) for every sensor that has weights."""
    mods = [m for m, s in (("rgb", cfg.rgb), ("ir", cfg.thermal)) if s.yolo_weights]
    if not mods:
        raise ValueError("give --rgb-yolo and/or --thermal-yolo weights")
    seqs = anti_uav_sequences(root, "ir")[: max_seqs or None]
    results = []
    for seq in seqs:
        for m in mods:
            if not _find_video(seq, "visible" if m == "rgb" else "infrared"):
                continue
            r = evaluate_sequence_single(seq, cfg, m, min_prob, max_frames)
            results.append(r)
            print(json.dumps(r), flush=True)
    summary = {"sequences": len(seqs)}
    for m in mods:
        mine = [r for r in results if r["modality"] == m]
        for k in ("det_acc", "det_hit_rate", "det_mean_iou", "trk_acc", "trk_hit_rate", "trk_mean_iou"):
            vals = [r[k] for r in mine if not np.isnan(r[k])]
            summary[f"{m}_{k}"] = round(float(np.mean(vals)), 4) if vals else None
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "per_sensor.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
        (out_dir / "per_sensor_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def make_fake_anti_uav(root: str | Path, n_train: int = 3, n_val: int = 1, n_test: int = 1,
                       seconds: float = 6.0, seed: int = 500) -> Path:
    """Write synthetic sequences in Anti-UAV300 layout (for tests / dry runs without the dataset)."""
    from .synthetic import SyntheticScene
    from .yolo_data import thermal_to_8bit

    root = Path(root)
    k = 0
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        for _ in range(n):
            sc = SyntheticScene(seed=seed + k, duration=seconds, dusk=(-1, -1), randomize=True, n_drones=1,
                                n_birds=1, fps=25.0)
            seq = root / split / f"seq_{k:03d}"
            seq.mkdir(parents=True, exist_ok=True)
            vw_r = cv2.VideoWriter(str(seq / "visible.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, sc.rgb_size)
            vw_t = cv2.VideoWriter(str(seq / "infrared.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, sc.thermal_size)
            lab = {"rgb": {"exist": [], "gt_rect": []}, "ir": {"exist": [], "gt_rect": []}}
            for i in range(len(sc)):
                _, rgb, th, gts = sc.frame(i)
                vw_r.write(rgb)
                vw_t.write(cv2.cvtColor(thermal_to_8bit(th), cv2.COLOR_GRAY2BGR))
                d = [g for g in gts if g.cls == "drone"]
                if d:
                    b = d[0].box
                    lab["rgb"]["exist"].append(1)
                    lab["rgb"]["gt_rect"].append([float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])])
                    if d[0].in_thermal_fov:
                        tb = map_boxes(b[None], sc.H_rgb_to_thermal)[0]
                        lab["ir"]["exist"].append(1)
                        lab["ir"]["gt_rect"].append([float(tb[0]), float(tb[1]), float(tb[2] - tb[0]),
                                                     float(tb[3] - tb[1])])
                    else:
                        lab["ir"]["exist"].append(0)
                        lab["ir"]["gt_rect"].append([0, 0, 0, 0])
                else:
                    for m in ("rgb", "ir"):
                        lab[m]["exist"].append(0)
                        lab[m]["gt_rect"].append([])
            vw_r.release()
            vw_t.release()
            (seq / "visible.json").write_text(json.dumps(lab["rgb"]))
            (seq / "infrared.json").write_text(json.dumps(lab["ir"]))
            k += 1
    return root

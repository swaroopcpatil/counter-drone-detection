"""Fit the drone-vs-clutter classifier on labelled real tracks.

The classifier adds up weighted cues (its log-odds). Here each cue's contribution is kept
as a feature, and a logistic regression learns one scale factor per cue plus an intercept:

    logit = intercept + sum_k beta_k * contribution_k

beta = 1 for all cues and intercept = 0 is exactly the hand-set classifier. Fitting is
regularised towards that starting point, so the result is a correction, not a new model.

A track is labelled a drone when at least half of its confirmed frames overlap the
ground-truth drone box (IoU > 0.3); otherwise it is clutter (birds, noise, false boxes).
Each track contributes one sample: the mean of its per-frame cue contributions.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from .anti_uav import _find_video, _single_sensor_cfg, iou_xyxy
from .config import Config
from .pipeline import CounterDronePipeline
from .yolo_data import _split_of, anti_uav_sequences, load_anti_uav_labels

CUES = ["consistency", "single_sensor", "flap", "turn", "hover", "label"]


def _contributions(tr) -> list[float]:
    ev = tr.extra.get("evidence") or {}
    return [float(ev.get(k, 0.0)) for k in CUES]


def collect_tracks(seq: Path, cfg: Config, modality: str, max_frames: int | None = None,
                   min_frames: int = 15) -> list[dict]:
    """Run one sequence through the single-camera pipeline and return one row per track."""
    exist, rects = load_anti_uav_labels(seq, modality)
    video = _find_video(seq, "visible" if modality == "rgb" else "infrared")
    if video is None:
        return []
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ok, first = cap.read()
    if not ok:
        return []
    if first.ndim == 2:
        first = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
    size = (first.shape[1], first.shape[0])
    pipe = CounterDronePipeline(_single_sensor_cfg(cfg, modality), np.eye(3), size, size)

    stats: dict[int, dict] = {}
    def frames():
        yield first
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield f

    for i, frame in enumerate(frames()):
        if i >= len(rects) or (max_frames and i >= max_frames):
            break
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        pipe.process(frame, None, i / fps)
        gt = np.array([rects[i][0], rects[i][1], rects[i][0] + rects[i][2], rects[i][1] + rects[i][3]]) \
            if exist[i] else None
        for tr in pipe.tracker.tracks:
            if not tr.confirmed:
                continue
            d = stats.setdefault(tr.id, {"n": 0, "hit": 0, "feat": []})
            d["n"] += 1
            if gt is not None and iou_xyxy(tr.box, gt) > 0.3:
                d["hit"] += 1
            d["feat"].append(_contributions(tr))
    cap.release()

    rows = []
    for tid, d in stats.items():
        if d["n"] < min_frames:
            continue
        rows.append({"seq": seq.name, "track": tid, "n": d["n"],
                     "label": int(d["hit"] / d["n"] >= 0.5),
                     "x": np.mean(d["feat"], axis=0).tolist()})
    return rows


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Probability that a random drone track scores above a random clutter track."""
    y = np.asarray(y, int)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit(rows: list[dict], prior_strength: float = 1.0) -> dict:
    """Logistic regression on cue contributions, regularised towards the hand-set model."""
    X = np.array([r["x"] for r in rows], float)
    y = np.array([r["label"] for r in rows], float)

    def loss(p):
        b0, beta = p[0], p[1:]
        z = b0 + X @ beta
        nll = np.sum(np.logaddexp(0.0, z) - y * z)
        return nll + 0.5 * prior_strength * np.sum((beta - 1.0) ** 2)

    p0 = np.r_[0.0, np.ones(len(CUES))]
    res = minimize(loss, p0, method="L-BFGS-B")
    return {"intercept": float(res.x[0]), "beta": dict(zip(CUES, map(float, res.x[1:]))),
            "converged": bool(res.success), "n_tracks": int(len(rows)),
            "n_drone": int(y.sum())}


def score(rows: list[dict], model: dict | None) -> dict:
    """AUC and drone-track detection rate at p >= 0.7. model=None means the hand-set classifier."""
    X = np.array([r["x"] for r in rows], float)
    y = np.array([r["label"] for r in rows], int)
    if model is None:
        z = X.sum(axis=1)
    else:
        beta = np.array([model["beta"][k] for k in CUES])
        z = model["intercept"] + X @ beta
    p = 1.0 / (1.0 + np.exp(-z))
    return {"auc": round(auc(y, p), 4), "drone_tracks": int(y.sum()), "clutter_tracks": int((1 - y).sum()),
            "drone_p_ge_0.7": round(float(np.mean(p[y == 1] >= 0.7)), 4) if y.sum() else None,
            "clutter_p_ge_0.7": round(float(np.mean(p[y == 0] >= 0.7)), 4) if (1 - y).sum() else None}


def run(root: str | Path, cfg: Config, modality: str, out_dir: str | Path, max_frames: int | None = None,
        max_seqs: int | None = None) -> dict:
    """Collect tracks from train/val (fit) and test (held out) sequences, fit, and score."""
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = [s for s in anti_uav_sequences(root, "ir") if _find_video(s, "visible") and _find_video(s, "infrared")]
    if max_seqs:
        seqs = seqs[:max_seqs]
    rows_path = out_dir / "tracks.jsonl"
    rows = []
    with rows_path.open("w") as f:
        for k, seq in enumerate(seqs):
            split = _split_of(seq, root) or "train"
            for r in collect_tracks(seq, cfg, modality, max_frames):
                r["split"] = split
                rows.append(r)
                f.write(json.dumps(r) + "\n")
            print(f"[{k + 1}/{len(seqs)}] {seq.name} ({split}): {len(rows)} tracks so far", flush=True)

    train = [r for r in rows if r["split"] in ("train", "val")]
    test = [r for r in rows if r["split"] == "test"]
    model = fit(train)
    n_drone = int(sum(r["label"] for r in train))
    warning = None
    if n_drone < 100 or len(train) - n_drone < 100:
        warning = (f"only {n_drone} drone and {len(train) - n_drone} clutter tracks: too few to fit reliably; "
                   "run on the full train/val split before using these weights")
        print("WARNING:", warning, flush=True)
    summary = {"fit_on": {"tracks": len(train), "drone": n_drone}, "warning": warning,
               "fitted": model,
               "hand_set_train": score(train, None), "fitted_train": score(train, model)}
    if test:
        summary["hand_set_test"] = score(test, None)
        summary["fitted_test"] = score(test, model)
    (out_dir / "fitted_weights.json").write_text(json.dumps(model, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary

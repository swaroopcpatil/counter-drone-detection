"""Background detector for the dashboard.

One worker thread runs the counter_drone pipeline on a *named* source and shares the latest
annotated JPEG frame, the recent alerts and a few counters with the web routes. Exactly one
detector runs per server process: the model is built inside the worker thread and never called
from two threads at once. Run the server with a single worker process.
"""
from __future__ import annotations

import inspect
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

MAX_ALERTS = 500
STREAM_WIDTH = 960
JPEG_QUALITY = 80
STREAM_MAX_FPS = 15
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
SENSITIVITIES = ("standard", "high")
REPO = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO / "runs" / "yolo" / "anti_uav_rgb" / "weights" / "best.pt"


def default_sources() -> dict:
    """Server-side allowlist of source names. The browser may only pick one of these names.

    Override with DASHBOARD_SOURCES='{"name": "path-or-url", ...}'.
    """
    env = os.environ.get("DASHBOARD_SOURCES")
    if env:
        return {str(k): str(v) for k, v in json.loads(env).items()}
    found = sorted((REPO / "datasets" / "Anti-UAV300" / "test").glob("*/visible.mp4"))[:3]
    return {f"Anti-UAV test clip {i + 1}": str(p) for i, p in enumerate(found)}


def default_config(sensitivity: str = "standard"):
    """YOLO on the GPU when weights and CUDA are available, otherwise the classical detector.

    standard: YOLO only, confidence 0.5 (few false alarms).
    high:     classical saliency detector plus YOLO labels, confidence 0.25, larger input
              (finds more small or unfamiliar drones, at the cost of more false alarms).
    """
    from counter_drone.config import Config
    cfg = Config()
    weights = os.environ.get("DASHBOARD_YOLO_WEIGHTS") or str(DEFAULT_WEIGHTS)
    if os.environ.get("DASHBOARD_BACKEND") != "classical" and Path(weights).exists():
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        cfg.rgb.yolo_weights, cfg.rgb.yolo_device = weights, device
        if sensitivity == "high":
            cfg.rgb.backend, cfg.rgb.yolo_conf, cfg.rgb.yolo_imgsz = "hybrid", 0.25, 1280
        else:
            cfg.rgb.backend, cfg.rgb.yolo_conf, cfg.rgb.yolo_imgsz = "yolo", 0.5, 960
    return cfg


class DetectorService:
    def __init__(self, sources: dict | None = None, make_config=None, loop: bool = True):
        self.sources = dict(default_sources() if sources is None else sources)
        self._make_config = make_config or default_config
        self.loop = loop
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._reset()

    # ---- state -------------------------------------------------------------------------
    def _reset(self):
        self._running = False
        self._source = None
        self._error = None
        self._fps = 0.0
        self._ms = 0.0
        self._frames = 0
        self._detected = 0
        self._sens = "standard"
        self._seq = 0
        self._jpeg: bytes | None = None
        self._alerts: deque = deque(maxlen=MAX_ALERTS)

    def status(self) -> dict:
        with self._cond:
            return {"running": self._running, "source": self._source, "fps": round(self._fps, 1),
                    "ms_per_frame": round(self._ms, 1), "frames": self._frames, "error": self._error,
                    "detections": self._detected, "sensitivity": self._sens,
                    "sources": sorted(self.sources)}

    def alerts(self, limit: int = 50) -> list:
        """Newest first."""
        with self._cond:
            items = list(self._alerts)
        return items[::-1][:limit]

    def add_source(self, name: str, spec: str) -> None:
        with self._cond:
            self.sources[name] = spec

    def remove_source_path(self, spec: str) -> None:
        """Forget every source name that points at this path (used when an upload is deleted)."""
        with self._cond:
            if self._running and self._source and self.sources.get(self._source) == spec:
                return
            for k in [k for k, v in self.sources.items() if v == spec]:
                del self.sources[k]

    def has_alerts(self) -> bool:
        with self._cond:
            return bool(self._alerts) or self._running

    # ---- control -----------------------------------------------------------------------
    def scan_local(self, folder) -> None:
        """List the videos in a server-side folder as sources named "Local: <file>"."""
        folder = Path(folder)
        found = {}
        if folder.is_dir():
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    found[f"Local: {p.stem}"] = str(p)
        with self._cond:
            for k in [k for k in self.sources if k.startswith("Local: ") and k not in found
                      and not (self._running and self._source == k)]:
                del self.sources[k]
            self.sources.update(found)

    def start(self, name: str, sensitivity: str = "standard") -> None:
        if name not in self.sources:
            raise KeyError(name)
        if sensitivity not in SENSITIVITIES:
            raise ValueError(sensitivity)
        with self._cond:
            if self._running:
                raise RuntimeError("the detector is already running")
        if self._thread is not None:
            self._thread.join(timeout=10)       # let a finished/failed worker release its resources
        with self._cond:
            self._reset()
            self._running, self._source, self._sens = True, name, sensitivity
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, args=(name, sensitivity), name="detector", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        with self._cond:
            self._running = False
            self._cond.notify_all()

    # ---- worker ------------------------------------------------------------------------
    def _run(self, name: str, sensitivity: str = "standard"):
        src = None
        try:
            import cv2
            import numpy as np
            from counter_drone.runtime import build_pipeline
            from counter_drone.sources import PairedSource
            from counter_drone.visualize import render

            spec = self.sources[name]
            if "sensitivity" in inspect.signature(self._make_config).parameters:
                cfg = self._make_config(sensitivity=sensitivity)
            else:
                cfg = self._make_config()
            prep = build_pipeline(cfg, spec)
            pipe, H, th_size, rgb_size = prep.pipe, prep.H, prep.thermal_size, prep.rgb_size
            src, frames = prep.src, iter(prep)
            live = bool(getattr(src, "live", False))
            period = 0.0 if live else 1.0 / max(1.0, float(getattr(src, "file_fps", 30.0)))
            base, t_off, last_t = None, 0.0, 0.0
            t_prev = next_due = time.perf_counter()
            while not self._stop.is_set():
                for t, rgb, thermal in frames:
                    if self._stop.is_set():
                        break
                    tick = time.perf_counter()
                    base = t if (base is None and live) else (base or 0.0)
                    tn = t_off + t - base
                    res = pipe.process(rgb, thermal, tn)
                    last_t = t - base
                    img = rgb if rgb is not None else np.zeros((rgb_size[1], rgb_size[0], 3), np.uint8)
                    frame = render(img, res, H, th_size, fps=self._fps, has_thermal=False)
                    if frame.shape[1] > STREAM_WIDTH:
                        s = STREAM_WIDTH / frame.shape[1]
                        frame = cv2.resize(frame, (STREAM_WIDTH, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    done = time.perf_counter()
                    with self._cond:
                        self._frames += 1
                        self._detected += 1 if res.fused_dets else 0
                        self._ms = (done - tick) * 1e3
                        self._fps = 0.9 * self._fps + 0.1 / max(1e-3, done - t_prev) if self._fps else 1.0 / max(1e-3, done - t_prev)
                        if ok:
                            self._jpeg, self._seq = buf.tobytes(), self._seq + 1
                        for e in res.events:
                            self._alerts.append(e.to_dict())
                        self._cond.notify_all()
                    t_prev = done
                    if period:                        # play recorded files at their own frame rate
                        next_due += period                # deadline pacing: coarse sleeps do not accumulate drift
                        wait = next_due - time.perf_counter()
                        if wait > 0:
                            self._stop.wait(wait)
                        else:
                            next_due = time.perf_counter()  # running behind real time: do not try to catch up
                if self._stop.is_set() or live or not self.loop:
                    break
                src.release()                         # restart the clip; the pipeline keeps running
                t_off += last_t + period
                src = PairedSource(spec, None)
                frames = iter(src)
        except Exception as exc:                      # surfaced in /api/detector/status
            with self._cond:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            if src is not None:
                try:
                    src.release()
                except Exception:
                    pass
            with self._cond:
                self._running = False
                self._cond.notify_all()

    # ---- MJPEG -------------------------------------------------------------------------
    def latest_jpeg(self) -> bytes | None:
        with self._cond:
            return self._jpeg

    def stream(self):
        """Yield multipart/x-mixed-replace chunks until the detector stops."""
        seq, last = -1, 0.0
        min_gap = 1.0 / STREAM_MAX_FPS
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._seq != seq or not self._running, timeout=1.0)
                if self._seq == seq:
                    if not self._running:
                        return
                    continue
                seq, data = self._seq, self._jpeg
            if not data:
                continue
            gap = min_gap - (time.perf_counter() - last)
            if gap > 0:
                time.sleep(gap)
            last = time.perf_counter()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode()
                   + b"\r\n\r\n" + data + b"\r\n")

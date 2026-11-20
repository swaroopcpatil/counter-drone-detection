"""Camera / file inputs and RGB-thermal frame pairing.

A source spec may be a device index ("0"), a video file, an image-sequence
pattern ("frames/%05d.png") or a stream URL (rtsp://, http://). Live sources
are read on background threads so the slower sensor never stalls the other;
the pairer then hands the pipeline the latest frame of each with timestamps
and drops a stale sensor (-> None) instead of blocking.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class ImageSequence:
    """printf-style pattern (frames/%05d.png) or directory, read with IMREAD_UNCHANGED.

    cv2.VideoCapture would silently convert 16-bit radiometric PNG/TIFF frames
    to 8-bit, destroying most of the thermal dynamic range.
    """

    def __init__(self, spec: str, fps: float = 30.0):
        import glob
        import os
        if os.path.isdir(spec):
            self.files = sorted(f for ext in ("png", "tif", "tiff", "jpg", "bmp")
                                for f in glob.glob(os.path.join(spec, f"*.{ext}")))
        else:
            self.files = []
            i = 0
            while os.path.exists(spec % i) or (i == 0 and os.path.exists(spec % 1)):
                if os.path.exists(spec % i):
                    self.files.append(spec % i)
                i += 1
        self.i, self.fps = 0, fps

    def isOpened(self):
        return bool(self.files)

    def read(self):
        if self.i >= len(self.files):
            return False, None
        f = cv2.imread(self.files[self.i], cv2.IMREAD_UNCHANGED)
        self.i += 1
        return f is not None, f

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0

    def set(self, *_):
        return False

    def release(self):
        pass


def _open(spec: str, thermal: bool):
    import os
    if "%" in spec or os.path.isdir(spec):
        seq = ImageSequence(spec)
        if not seq.isOpened():
            raise RuntimeError(f"no frames found for {spec!r}")
        return seq
    src = int(spec) if spec.isdigit() else spec
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source {spec!r}")
    if thermal and isinstance(src, int):
        # UVC thermal cores (FLIR Boson/Lepton via PureThermal, etc.) expose Y16; ask for raw.
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    return cap


def _is_live(spec: str) -> bool:
    return spec.isdigit() or spec.startswith(("rtsp://", "http://", "https://", "udp://"))


class ThreadedCapture:
    def __init__(self, spec: str, thermal: bool = False):
        self.cap = _open(spec, thermal)
        self.frame: np.ndarray | None = None
        self.t = 0.0
        self.seq = 0
        self.ok = True
        self._lock = threading.Lock()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while self.ok:
            ok, f = self.cap.read()
            if not ok:
                self.ok = False
                break
            with self._lock:
                self.frame, self.t, self.seq = f, time.monotonic(), self.seq + 1

    def latest(self):
        with self._lock:
            return self.frame, self.t, self.seq

    def release(self):
        self.ok = False
        self.cap.release()


class PairedSource:
    """Iterates (t, rgb, thermal). Either frame may be None if that sensor is stale/absent."""

    def __init__(self, rgb_spec: str | None, thermal_spec: str | None, max_skew_s: float = 0.1,
                 file_fps: float = 30.0):
        self.live = any(s and _is_live(s) for s in (rgb_spec, thermal_spec))
        self.max_skew = max_skew_s
        self.file_fps = file_fps
        if self.live:
            self.rgb = ThreadedCapture(rgb_spec) if rgb_spec else None
            self.th = ThreadedCapture(thermal_spec, thermal=True) if thermal_spec else None
        else:  # files: read in lock-step, timestamps from the frame counter
            self.rgb = _open(rgb_spec, False) if rgb_spec else None
            self.th = _open(thermal_spec, True) if thermal_spec else None
            fps = (self.rgb or self.th).get(cv2.CAP_PROP_FPS)
            self.file_fps = fps if fps and fps > 1 else file_fps
        if self.rgb is None and self.th is None:
            raise ValueError("need at least one of rgb / thermal")

    def __iter__(self):
        return self._iter_live() if self.live else self._iter_files()

    def _iter_files(self):
        i = 0
        while True:
            r = t = None
            if self.rgb is not None:
                ok, r = self.rgb.read()
                if not ok:
                    return
            if self.th is not None:
                ok, t = self.th.read()
                if not ok:
                    return
            yield i / self.file_fps, r, _thermal_raw(t)
            i += 1

    def _iter_live(self):
        last = (-1, -1)
        while True:
            alive = [s for s in (self.rgb, self.th) if s is not None]
            if not any(s.ok for s in alive):
                return
            r, tr, sr = self.rgb.latest() if self.rgb else (None, 0.0, -1)
            th, tt, st = self.th.latest() if self.th else (None, 0.0, -1)
            if (sr, st) == last or (r is None and th is None):
                time.sleep(0.002)
                continue
            last = (sr, st)
            now = time.monotonic()
            # drop a sensor whose newest frame is too old (cable pulled, core FFC freeze, ...)
            if r is not None and now - tr > 1.0:
                r = None
            if th is not None and now - tt > 1.0:
                th = None
            yield now, r, _thermal_raw(th)

    def release(self):
        for s in (self.rgb, self.th):
            if s is not None:
                s.release()


def _thermal_raw(f):
    """Normalise whatever the capture backend delivered for thermal into a 2-D array."""
    if f is None:
        return None
    if f.ndim == 3 and f.shape[2] == 3:
        return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    if f.ndim == 2 and f.dtype == np.uint8 and f.shape[0] == 1:
        return None  # undecodable raw buffer
    return f.squeeze()

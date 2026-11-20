"""Synthetic paired RGB + thermal sky scenes with ground truth.

Used for the offline demo, regression tests and tuning when no hardware is
attached. The scene contains drones (hovering/patrolling, approaching,
crossing in and out of the thermal FOV), flapping birds with erratic flight,
drifting clouds, sensor noise, and a "dusk" interval where the RGB camera is
nearly blind so the thermal channel must carry the tracks.

Thermal frames are emitted as uint16 counts, like a radiometric Y16 core.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GTObject:
    obj_id: int
    cls: str                 # "drone" | "bird"
    box: np.ndarray          # RGB frame xyxy
    in_thermal_fov: bool


def _smoothstep(a):
    return a * a * (3 - 2 * a)


class _Keyframes:
    """Piecewise smooth motion through (t, x, y, size) keys; equal keys = hover."""

    def __init__(self, keys):
        self.k = np.array(keys, float)

    def __call__(self, t):
        k = self.k
        if t <= k[0, 0]:
            return k[0, 1:]
        if t >= k[-1, 0]:
            return k[-1, 1:]
        i = np.searchsorted(k[:, 0], t) - 1
        a = _smoothstep((t - k[i, 0]) / (k[i + 1, 0] - k[i, 0]))
        return k[i, 1:] * (1 - a) + k[i + 1, 1:] * a


class _Bird:
    def __init__(self, rng, size, bounds, duration, fps, t_start=0.0):
        self.size = size
        self.flap_hz = rng.uniform(3.5, 6.0)
        self.phase0 = rng.uniform(0, 2 * np.pi)
        x0, y0, x1, y1 = bounds
        p = np.array([rng.uniform(x0 + 50, x1 - 50), rng.uniform(y0 + 50, y1 - 50)])
        heading = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(55, 85)
        dt = 1.0 / fps
        self.t_start = t_start
        self.path = []
        for _ in range(int(duration * fps) + 2):
            heading += rng.normal(0, 2.2) * dt * 3 + 0.6 * np.sin(len(self.path) * 0.07)  * dt
            p = p + speed * dt * np.array([np.cos(heading), np.sin(heading)])
            for ax, lo, hi in ((0, x0 + 20, x1 - 20), (1, y0 + 20, y1 - 20)):  # bounce
                if p[ax] < lo or p[ax] > hi:
                    p[ax] = np.clip(p[ax], lo, hi)
                    heading = np.pi - heading if ax == 0 else -heading
            self.path.append(p.copy())
        self.fps = fps

    def state(self, t):
        i = min(int(max(0.0, t - self.t_start) * self.fps), len(self.path) - 1)
        x, y = self.path[i]
        wing = np.sin(2 * np.pi * self.flap_hz * t + self.phase0)  # -1..1
        return x, y, wing


class SyntheticScene:
    H_thermal_to_rgb = np.array([[2.0, 0.0, 160.0], [0.0, 2.0, 14.0], [0.0, 0.0, 1.0]])

    def __init__(self, seed: int = 0, rgb_size=(960, 540), thermal_size=(320, 256), fps: float = 15.0,
                 duration: float = 30.0, dusk=(17.0, 23.0), randomize: bool = False,
                 n_drones: int | None = None, n_birds: int | None = None):
        """``randomize=True`` draws random drone paths/sizes, bird count/sizes and
        sky colour from ``seed`` - used to generate varied YOLO training data."""
        self.rng = np.random.default_rng(seed)
        self.rgb_size, self.thermal_size, self.fps = rgb_size, thermal_size, fps
        self.duration, self.dusk = duration, dusk
        self.H_rgb_to_thermal = np.linalg.inv(self.H_thermal_to_rgb)
        W, H = rgb_size
        # wide low-frequency cloud field that drifts across the view
        coarse = self.rng.random((H // 48 + 4, (W + 600) // 48 + 4)).astype(np.float32)
        fine = self.rng.random((H // 12 + 4, (W + 600) // 12 + 4)).astype(np.float32)
        size = ((W + 600) + 96, H + 96)
        field = cv2.resize(coarse, size, interpolation=cv2.INTER_CUBIC)
        field += 0.25 * cv2.resize(fine, size, interpolation=cv2.INTER_CUBIC)  # ragged cloud edges
        field = cv2.GaussianBlur(field, (0, 0), 10)
        field = (field - field.min()) / (np.ptp(field) + 1e-6)
        self.cloud_field = np.clip((field - 0.45) * 2.5, 0, 1).astype(np.float32)
        grad = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        top, horizon = np.array([215, 160, 95], np.float32), np.array([245, 225, 200], np.float32)  # BGR
        if randomize:  # vary sky tint / brightness (haze, time of day)
            top = np.clip(top * self.rng.uniform(0.75, 1.1) + self.rng.normal(0, 12, 3), 60, 255).astype(np.float32)
            horizon = np.clip(horizon * self.rng.uniform(0.8, 1.05), 80, 255).astype(np.float32)
        self.sky = np.broadcast_to(top * (1 - grad) + horizon * grad, (H, W, 3)).copy()

        self.drones = {
            # patrols, hovers twice
            1: _Keyframes([(0, 420, 170, 8), (5, 520, 150, 8), (8, 520, 150, 8), (13, 640, 230, 8),
                           (16, 560, 300, 8), (20, 560, 300, 8), (26, 400, 220, 8), (30, 380, 200, 8)]),
            # inbound: far and small -> close and large
            2: _Keyframes([(0, 700, 120, 4.5), (10, 640, 190, 8), (20, 560, 260, 15), (30, 500, 330, 30)]),
            # crosser: starts outside the thermal FOV (x<160), crosses it, leaves at right
            3: _Keyframes([(2, 40, 420, 7), (16, 500, 380, 7), (28, 930, 330, 7)]),
        }
        self.drone_window = {1: (0, 30), 2: (0, 30), 3: (2, 28)}
        bounds = (0, 0, W, int(H * 0.9))
        bird_sizes = [12, 10, 14]
        if randomize:
            self.drones, self.drone_window = {}, {}
            for oid in range(1, (n_drones if n_drones is not None else int(self.rng.integers(1, 5))) + 1):
                self.drones[oid] = self._random_path(W, H, duration)
                self.drone_window[oid] = (0, duration)
            bird_sizes = list(self.rng.uniform(7, 22, n_birds if n_birds is not None else int(self.rng.integers(0, 5))))
        self.birds = {10 + i: _Bird(self.rng, s, bounds, duration, fps, 0.0)
                      for i, s in enumerate(bird_sizes)}

    def _random_path(self, W, H, duration):
        rng = self.rng
        keys, t = [], 0.0
        x, y = rng.uniform(40, W - 40), rng.uniform(30, H * 0.85)
        size = rng.uniform(4, 28)
        while t <= duration + 3:
            keys.append((t, x, y, size))
            if rng.random() < 0.3:                     # hover
                t += rng.uniform(1, 4)
                keys.append((t, x, y, size))
            t += rng.uniform(2, 6)
            x = float(np.clip(x + rng.normal(0, 180), 20, W - 20))
            y = float(np.clip(y + rng.normal(0, 90), 20, H * 0.9))
            size = float(np.clip(size * rng.uniform(0.7, 1.4), 4, 36))
        return _Keyframes(keys)

    # ------------------------------------------------------------------
    def __len__(self):
        return int(self.duration * self.fps)

    def _in_tfov(self, x, y):
        p = self.H_rgb_to_thermal @ np.array([x, y, 1.0])
        u, v = p[:2] / p[2]
        return 0 <= u < self.thermal_size[0] and 0 <= v < self.thermal_size[1]

    def objects(self, t: float):
        objs = []
        for oid, kf in self.drones.items():
            t0, t1 = self.drone_window[oid]
            if t0 <= t <= t1:
                x, y, s = kf(t)
                objs.append(("drone", oid, x, y, s, 0.0))
        for oid, b in self.birds.items():
            x, y, wing = b.state(t)
            objs.append(("bird", oid, x, y, b.size, wing))
        return objs

    def frame(self, i: int):
        t = i / self.fps
        W, H = self.rgb_size
        rng = self.rng

        # ---------------- RGB ----------------
        off = int(t * 12) % 600
        clouds = self.cloud_field[48:48 + H, 48 + off:48 + off + W]
        img = self.sky * (1 - 0.85 * clouds[..., None]) + 248 * 0.85 * clouds[..., None]

        gts: list[GTObject] = []
        for cls, oid, x, y, s, wing in self.objects(t):
            if cls == "drone":
                self._draw_drone(img, x, y, s, t)
                box = np.array([x - s / 2, y - s / 2, x + s / 2, y + s / 2])
            else:
                hh = self._draw_bird(img, x, y, s, wing)
                box = np.array([x - s / 2, y - hh, x + s / 2, y + hh])
            if 0 <= x < W and 0 <= y < H:
                gts.append(GTObject(oid, cls, box, self._in_tfov(x, y)))

        dusk = self.dusk[0] <= t <= self.dusk[1]
        if dusk:
            img = img * 0.06 + 4
        img += rng.normal(0, 3.0 if dusk else 2.0, img.shape)
        rgb = np.clip(img, 0, 255).astype(np.uint8)

        # ---------------- thermal ----------------
        tw, th = self.thermal_size
        tclouds = cv2.warpPerspective(clouds, self.H_rgb_to_thermal, (tw, th), flags=cv2.INTER_LINEAR)
        therm = 0.20 + 0.06 * tclouds + 0.04 * np.linspace(0, 1, th)[:, None]  # sky colder at zenith
        for cls, oid, x, y, s, wing in self.objects(t):
            p = self.H_rgb_to_thermal @ np.array([x, y, 1.0])
            u, v = p[:2] / p[2]
            ts = s / 2.0  # thermal is half the RGB resolution
            if cls == "drone":
                arm = ts * 0.35
                for dx, dy in ((-arm, -arm), (arm, -arm), (-arm, arm), (arm, arm)):
                    self._splat(therm, u + dx, v + dy, 0.16, max(0.6, ts * 0.12))   # motors
                self._splat(therm, u, v, 0.22, max(0.8, ts * 0.18))                 # battery/ESC
            else:
                self._splat(therm, u, v, 0.10, max(0.7, ts * 0.15))                 # body
                self._splat(therm, u - ts * 0.3, v, 0.03, max(0.6, ts * 0.12))
                self._splat(therm, u + ts * 0.3, v, 0.03, max(0.6, ts * 0.12))
        therm += rng.normal(0, 0.006, therm.shape)
        thermal = np.clip(7000 + therm * 4000, 0, 65535).astype(np.uint16)
        return t, rgb, thermal, gts

    def __iter__(self):
        for i in range(len(self)):
            yield self.frame(i)

    # ---------------- drawing helpers ----------------
    @staticmethod
    def _draw_drone(img, x, y, s, t):
        col = (38, 38, 42)
        sh = 4  # sub-pixel precision for cv2 drawing
        P = lambda a, b: (int(round(a * 2 ** sh)), int(round(b * 2 ** sh)))  # noqa: E731
        arm = s * 0.35
        for dx, dy in ((-arm, -arm), (arm, -arm), (-arm, arm), (arm, arm)):
            cv2.line(img, P(x, y), P(x + dx, y + dy), col, max(1, int(s / 10)), cv2.LINE_AA, sh)
            cv2.circle(img, P(x + dx, y + dy), int(max(0.8, s * 0.15) * 2 ** sh), col, -1, cv2.LINE_AA, sh)
        cv2.ellipse(img, P(x, y), (int(s * 0.18 * 2 ** sh), int(s * 0.12 * 2 ** sh)), 0, 0, 360, col, -1,
                    cv2.LINE_AA, sh)

    @staticmethod
    def _draw_bird(img, x, y, s, wing):
        col = (30, 32, 34)
        sh = 4
        P = lambda a, b: (int(round(a * 2 ** sh)), int(round(b * 2 ** sh)))  # noqa: E731
        lift = wing * s * 0.35
        half = s / 2 * (0.75 + 0.25 * abs(np.cos(np.arcsin(np.clip(wing, -1, 1)))))
        pts = np.array([P(x - half, y - lift), P(x - half * 0.4, y - lift * 0.3), P(x, y),
                        P(x + half * 0.4, y - lift * 0.3), P(x + half, y - lift)], np.int32)
        cv2.polylines(img, [pts], False, col, max(1, int(s / 7)), cv2.LINE_AA, sh)
        cv2.ellipse(img, P(x, y + 0.5), (int(s * 0.12 * 2 ** sh), int(s * 0.08 * 2 ** sh)), 0, 0, 360, col, -1,
                    cv2.LINE_AA, sh)
        return max(abs(lift), s * 0.12) + 1

    @staticmethod
    def _splat(img, u, v, amp, sigma):
        h, w = img.shape
        r = int(3 * sigma) + 1
        x0, x1 = max(0, int(u) - r), min(w, int(u) + r + 1)
        y0, y1 = max(0, int(v) - r), min(h, int(v) + r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        img[y0:y1, x0:x1] += amp * np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * sigma ** 2))

"""Configuration for the whole pipeline.

Everything is a plain dataclass so it can be loaded from / dumped to JSON and
tuned per site without touching code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class RGBDetectorConfig:
    backend: str = "classical"          # "classical" | "yolo" | "hybrid" (classical recall + YOLO labels)
    yolo_weights: str = ""              # path to ultralytics weights (see `yolo-train`)
    yolo_conf: float = 0.25
    yolo_imgsz: int = 960               # inference size: use the full frame width for small targets
    yolo_standalone_conf: float = 0.5   # hybrid: YOLO boxes with no classical blob need this confidence
    yolo_label_conf: float = 0.4        # hybrid: min confidence for a YOLO class label to be trusted
    yolo_device: str = "cpu"            # "cpu", "0" (CUDA GPU), "mps"
    yolo_drone_classes: list[str] = field(default_factory=lambda: ["drone", "uav"])
    # classical small-target detector
    polarity: str = "dark"              # "dark" (silhouettes vs sky, day) | "bright" | "both"
    kernel_px: int = 25                 # morphological kernel, > largest expected target
    k_sigma: float = 8.0                # global threshold in sensor-noise sigmas
    k_local: float = 5.0                # CFAR: local clutter std multiplier
    cfar_win: int = 61                  # CFAR window (px), several x kernel
    min_abs_contrast: float = 6.0       # grey levels, guards against flat frames
    min_area: int = 4
    max_area: int = 2500
    merge_px: int = 2                   # group blobs closer than this into one target
    border_px: int = 4                  # ignore blobs touching the image edge
    min_mean_brightness: float = 25.0   # below this the RGB channel is considered blind
    max_aspect: float = 5.0
    bg_alpha: float = 0.05              # running background learning rate (motion cue)
    motion_bonus: float = 0.15
    roi: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])  # x1,y1,x2,y2 fractions


@dataclass
class ThermalDetectorConfig:
    backend: str = "classical"          # "classical" | "yolo" | "hybrid"
    yolo_weights: str = ""
    yolo_conf: float = 0.25
    yolo_imgsz: int = 320
    yolo_standalone_conf: float = 0.5
    yolo_label_conf: float = 0.4
    yolo_device: str = "cpu"
    yolo_drone_classes: list[str] = field(default_factory=lambda: ["drone", "uav"])
    agc_low_pct: float = 1.0            # percentile AGC for 16-bit radiometric input
    agc_high_pct: float = 99.9
    kernel_px: int = 15
    k_sigma: float = 8.0
    k_local: float = 5.0
    cfar_win: int = 41
    min_abs_contrast: float = 0.02      # in normalised [0,1] units
    min_area: int = 2
    max_area: int = 800
    merge_px: int = 2
    border_px: int = 2
    detect_cold: bool = False           # also detect targets colder than background
    roi: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])


@dataclass
class FusionConfig:
    gate_px: float = 25.0               # max centre distance (RGB px) to fuse a pair
    rgb_only_in_thermal_fov_factor: float = 0.75  # thermal should usually see a drone too
    thermal_only_factor: float = 0.9
    min_score: float = 0.25


@dataclass
class TrackerConfig:
    process_accel: float = 150.0        # px/s^2 white-noise acceleration
    meas_pos_std: float = 2.5           # px
    meas_size_std: float = 3.0          # px
    meas_rel_std: float = 0.1           # jitter as a fraction of box size (big/near targets)
    gate_chi2: float = 13.8             # chi-square gate, 2 dof, 99.9 %
    gate_size_frac: float = 1.5         # also match a detection within this many box sizes
    min_hits_confirm: int = 4
    confirm_window: int = 6             # M-of-N: min_hits_confirm hits in last N frames
    max_misses_tentative: int = 2
    max_misses_confirmed: int = 20      # coast through short dropouts
    init_min_score: float = 0.35
    history_len: int = 90


@dataclass
class ClassifierConfig:
    """Weights of the explainable log-odds drone-vs-clutter classifier."""
    min_track_len: int = 15             # ~1 s at 15 fps: a few wing-beats
    window: int = 45                    # frames of history used
    bias: float = 0.0
    w_consistency_pos: float = 1.0      # seen by BOTH sensors when both could see it -> real object
    w_consistency_neg: float = 2.5      # ...but only ever by one of them -> clutter (cloud edge, glint)
    min_consistency_frames: int = 8
    w_single_sensor: float = -0.75      # no cross-sensor check possible yet (outside thermal FOV, ...)
    memory_decay: float = 0.985         # per-frame decay of evidence that is temporarily unavailable
    w_flap: float = -12.0               # wing-beat: oscillating box aspect ratio (std of log h/w)
    flap_ref: float = 0.2
    min_flap_frames: int = 15
    w_turn: float = -1.2                # erratic heading changes (rad/s) -> bird
    turn_ref: float = 1.2
    w_hover: float = 1.0                # fraction of time (near-)stationary -> drone
    hover_speed_px: float = 6.0
    w_label: float = 3.0                # class label from a learned detector, if any
    kinematic_scale_with_label: float = 0.25  # weight of flap/turn/hover when a detector labels the track
    label_min_share: float = 0.3        # ... in at least this share of the window's frames
    label_conf_lo: float = 0.25         # mean detector confidence at which a label counts fully against
    label_conf_hi: float = 0.55         # ... and fully for (0.40 = neutral)
    ema: float = 0.15                   # smoothing of the posterior


@dataclass
class CameraConfig:
    hfov_deg: float = 20.0              # RGB camera horizontal field of view (tele lens)
    pan_deg: float = 0.0                # azimuth of optical axis (true north = 0)
    tilt_deg: float = 10.0              # elevation of optical axis
    drone_span_m: float = 0.35          # nominal span for monocular range estimate


@dataclass
class ThreatConfig:
    drone_prob_low: float = 0.45
    drone_prob_high: float = 0.7
    alert_range_m: float = 100.0
    closing_speed_mps: float = 2.0


@dataclass
class Config:
    rgb: RGBDetectorConfig = field(default_factory=RGBDetectorConfig)
    thermal: ThermalDetectorConfig = field(default_factory=ThermalDetectorConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    threat: ThreatConfig = field(default_factory=ThreatConfig)

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return _fill(cls, d)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _fill(dc_type, d: dict):
    kwargs = {}
    names = {f.name: f for f in fields(dc_type)}
    for key, value in d.items():
        if key not in names:
            raise KeyError(f"Unknown config key '{key}' for {dc_type.__name__}")
        default = names[key].default_factory() if callable(names[key].default_factory) else None
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _fill(type(default), value)
        else:
            kwargs[key] = value
    return dc_type(**kwargs)

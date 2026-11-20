from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """A single target candidate.

    ``box`` is (x1, y1, x2, y2). Detectors emit boxes in their own sensor's
    pixel frame; after fusion all boxes are in the RGB (reference) frame.
    """

    box: np.ndarray
    score: float
    source: str                      # "rgb" | "thermal" | "fused"
    rgb_score: float = 0.0
    thermal_score: float = 0.0
    contrast: float = 0.0            # peak contrast in noise sigmas (sensor-specific)
    thermal_contrast: float = 0.0    # thermal contrast in sigmas, 0 if not seen in thermal
    in_thermal_fov: bool = True
    rgb_ok: bool = True              # RGB channel usable this frame (not blind / missing)
    moving: bool = False
    label: str | None = None         # class name from a learned detector, if any

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    @property
    def wh(self) -> np.ndarray:
        return np.array([self.box[2] - self.box[0], self.box[3] - self.box[1]])


@dataclass
class ThreatEvent:
    t: float
    track_id: int
    level: str
    previous: str
    drone_prob: float
    azimuth_deg: float
    elevation_deg: float
    range_m: float | None
    message: str = ""

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class TrackSnapshot:
    """Immutable view of a track for output/visualisation."""

    track_id: int
    box: np.ndarray
    velocity: np.ndarray
    confirmed: bool
    drone_prob: float
    threat: str
    azimuth_deg: float
    elevation_deg: float
    range_m: float | None
    closing_mps: float | None
    age: int
    misses: int
    sources: str                     # e.g. "RT" = recently seen by RGB and Thermal
    trail: list = field(default_factory=list)
    label: str | None = None         # majority class from the learned detector (YOLO), if used
    score: float = 0.0               # mean detection confidence over recent matches

    def to_dict(self) -> dict:
        r = lambda v: None if v is None else round(float(v), 2)  # noqa: E731
        return {
            "id": self.track_id,
            "box": [r(v) for v in self.box],
            "vel": [r(v) for v in self.velocity],
            "confirmed": self.confirmed,
            "drone_prob": r(self.drone_prob),
            "threat": self.threat,
            "az": r(self.azimuth_deg),
            "el": r(self.elevation_deg),
            "range_m": r(self.range_m),
            "closing_mps": r(self.closing_mps),
            "age": self.age,
            "misses": self.misses,
            "sources": self.sources,
            "label": self.label,
        }

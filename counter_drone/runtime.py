"""Build a ready-to-run pipeline from source specs. Shared by the CLI and the dashboard."""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass

import numpy as np

from .calibration import load_homography
from .config import Config
from .pipeline import CounterDronePipeline
from .sources import PairedSource


@dataclass
class Prepared:
    """A pipeline plus the opened source. Iterate it for (t, rgb, thermal) frames."""
    src: PairedSource
    pipe: CounterDronePipeline
    H: np.ndarray
    rgb_size: tuple
    thermal_size: tuple
    first: tuple
    frames: object

    def __iter__(self):
        return itertools.chain([self.first], self.frames)

    def release(self) -> None:
        self.src.release()


def build_pipeline(cfg: Config, rgb_spec, thermal_spec=None, homography=None,
                   rgb_size=(1920, 1080), thermal_size=(640, 512)) -> Prepared:
    """Open the source, read the first frame to learn the image sizes, and build the pipeline.

    rgb_size / thermal_size are only the fallback when a sensor's first frame is missing.
    """
    src = PairedSource(rgb_spec, thermal_spec)
    try:
        frames = iter(src)
        first = next(frames)
    except StopIteration:
        src.release()
        raise RuntimeError("the source produced no frames") from None
    except Exception:
        src.release()
        raise
    _, r0, t0 = first
    rgb_size = (r0.shape[1], r0.shape[0]) if r0 is not None else tuple(rgb_size)
    th_size = (t0.shape[1], t0.shape[0]) if t0 is not None else tuple(thermal_size)
    if homography:
        H = load_homography(homography)
    else:
        if thermal_spec:
            # no calibration: assume the two cameras are boresighted with the thermal image filling the RGB image
            print("WARNING: no --homography given; assuming thermal FOV == RGB FOV. Calibrate for real use.",
                  file=sys.stderr)
        # with no thermal camera the matrix is unused
        H = np.diag([rgb_size[0] / th_size[0], rgb_size[1] / th_size[1], 1.0])
    pipe = CounterDronePipeline(cfg, H, rgb_size, th_size)
    return Prepared(src, pipe, H, rgb_size, th_size, first, frames)

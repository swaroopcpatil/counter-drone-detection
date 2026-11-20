"""RGB <-> thermal spatial registration.

For a co-located (side-by-side, boresighted) camera pair looking at targets
tens to hundreds of metres away, parallax is negligible and a single
homography maps thermal pixels into the RGB image. Calibrate it with either

* hand-picked point correspondences (e.g. distant corners/lamps/a heated
  target visible in both), or
* a heated / aluminium-on-black checkerboard seen by both cameras.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def homography_from_points(pts_thermal, pts_rgb) -> tuple[np.ndarray, float]:
    """Returns (H mapping thermal->RGB, RMS reprojection error in RGB px)."""
    pt = np.asarray(pts_thermal, np.float64).reshape(-1, 2)
    pr = np.asarray(pts_rgb, np.float64).reshape(-1, 2)
    if len(pt) < 4 or len(pt) != len(pr):
        raise ValueError("need >= 4 matching point pairs")
    H, _ = cv2.findHomography(pt, pr, cv2.RANSAC if len(pt) > 4 else 0, 3.0)
    if H is None:
        raise RuntimeError("homography estimation failed (degenerate points?)")
    return H, reprojection_rms(H, pt, pr)


def reprojection_rms(H, pts_src, pts_dst) -> float:
    proj = cv2.perspectiveTransform(np.asarray(pts_src, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((proj - np.asarray(pts_dst).reshape(-1, 2)) ** 2, axis=1))))


def _find_board(img: np.ndarray, pattern: tuple[int, int]):
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    for candidate in (gray, 255 - gray):  # thermal boards often have inverted polarity
        ok, corners = cv2.findChessboardCorners(candidate, pattern, flags)
        if ok:
            corners = cv2.cornerSubPix(candidate, corners, (5, 5), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            return corners.reshape(-1, 2)
    return None


def homography_from_checkerboard(rgb_img, thermal_img, pattern=(9, 6)) -> tuple[np.ndarray, float]:
    """``pattern`` = inner corners (cols, rows)."""
    pr = _find_board(rgb_img, pattern)
    pt = _find_board(thermal_img, pattern)
    if pr is None or pt is None:
        raise RuntimeError(f"checkerboard not found in {'RGB' if pr is None else 'thermal'} image")
    return homography_from_points(pt, pr)


def save_homography(path, H, thermal_size=None, rgb_size=None, rms=None):
    Path(path).write_text(json.dumps({
        "H_thermal_to_rgb": np.asarray(H).tolist(),
        "thermal_size": thermal_size, "rgb_size": rgb_size, "rms_px": rms,
    }, indent=2))


def load_homography(path) -> np.ndarray:
    return np.array(json.loads(Path(path).read_text())["H_thermal_to_rgb"], float)


def map_boxes(boxes: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Map Nx4 xyxy boxes through H; returns the axis-aligned bounds of the warped corners."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if len(boxes) == 0:
        return boxes
    x1, y1, x2, y2 = boxes.T
    corners = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1),
                        np.stack([x2, y2], 1), np.stack([x1, y2], 1)], 1)  # N,4,2
    warped = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 4, 2)
    return np.concatenate([warped.min(1), warped.max(1)], axis=1)


def thermal_fov_polygon(H: np.ndarray, thermal_size: tuple[int, int]) -> np.ndarray:
    """Outline of the thermal image in RGB pixel coordinates (4x2)."""
    w, h = thermal_size
    c = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(c, H).reshape(-1, 2)


def homography_from_box_tracks(rgb_boxes, ir_boxes, rgb_exist=None, ir_exist=None,
                               max_rms_px: float = 15.0) -> tuple[np.ndarray, float, str]:
    """Thermal->RGB mapping from the same target annotated in both synchronised videos.

    Datasets like Anti-UAV label the drone independently in the RGB and IR videos; the box
    centres in frames where both see it are point correspondences. A full homography is
    fitted when the points are well spread, otherwise a similarity transform (scale +
    rotation + shift), which is all a boresighted pair needs at long range.
    Returns (H, rms_px, model).
    """
    pr, pt = [], []
    for i in range(min(len(rgb_boxes), len(ir_boxes))):
        if (rgb_exist and not rgb_exist[i]) or (ir_exist and not ir_exist[i]):
            continue
        r, t = rgb_boxes[i], ir_boxes[i]
        if len(r) != 4 or len(t) != 4 or r[2] <= 0 or t[2] <= 0:
            continue
        pr.append((r[0] + r[2] / 2, r[1] + r[3] / 2))
        pt.append((t[0] + t[2] / 2, t[1] + t[3] / 2))
    pr, pt = np.array(pr, np.float64), np.array(pt, np.float64)
    if len(pr) < 4:
        raise ValueError("fewer than 4 frames with the target labelled in both videos")
    spread = np.linalg.svd(pt - pt.mean(0), compute_uv=False)
    if len(pr) >= 20 and spread.min() > 0.15 * spread.max():
        H, _ = cv2.findHomography(pt, pr, cv2.RANSAC, 8.0)
        if H is not None:
            rms = reprojection_rms(H, pt, pr)
            if rms <= max_rms_px:
                return H, rms, "homography"
    A, _ = cv2.estimateAffinePartial2D(pt, pr, method=cv2.RANSAC, ransacReprojThreshold=8.0)
    if A is None:
        raise RuntimeError("could not fit RGB<->IR mapping from the box tracks")
    H = np.vstack([A, [0, 0, 1]])
    return H, reprojection_rms(H, pt, pr), "similarity"

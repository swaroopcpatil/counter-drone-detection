"""Turn a recorded run (e.g. runs/live_demo/run.mp4) into README media.

    python scripts/make_media.py runs/live_demo/run.mp4 --name anti_uav_demo --start 2 --seconds 10

Writes docs/media/<name>.gif (looping, sized for a README) and docs/media/<name>.jpg
(a full-quality still from the middle of the clip). Needs only OpenCV and Pillow
(installed with ultralytics) - no ffmpeg.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "docs" / "media"


def make(video: str, name: str, start: float, seconds: float, width: int, fps: float, max_mb: float) -> None:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * src_fps))
    step = max(1, round(src_fps / fps))
    frames, i = [], 0
    while len(frames) < int(seconds * fps):
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(f)
        i += 1
    cap.release()
    if not frames:
        raise SystemExit("no frames read - check --start against the video length")

    OUT.mkdir(parents=True, exist_ok=True)
    mid = frames[len(frames) // 2]
    cv2.imwrite(str(OUT / f"{name}.jpg"), mid, [cv2.IMWRITE_JPEG_QUALITY, 90])

    gif = OUT / f"{name}.gif"
    for w in (width, int(width * 0.85), int(width * 0.7), int(width * 0.6)):   # shrink until under max_mb
        imgs = []
        for f in frames:
            h = int(f.shape[0] * w / f.shape[1])
            rgb = cv2.cvtColor(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            imgs.append(Image.fromarray(rgb).quantize(colors=96, method=Image.Quantize.FASTOCTREE))
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0,
                     optimize=True)
        size = gif.stat().st_size / 2**20
        if size <= max_mb:
            break
    print(f"{gif} ({size:.1f} MB, {len(frames)} frames, {w} px wide)\n{OUT / (name + '.jpg')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--name", default="anti_uav_demo")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the video")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--max-mb", type=float, default=5.0)
    a = ap.parse_args()
    make(a.video, a.name, a.start, a.seconds, a.width, a.fps, a.max_mb)

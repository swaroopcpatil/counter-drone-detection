"""Compare YOLO models live on the same camera frames, showing raw confidences.

    python scripts/yolo_live_view.py --weights runs/yolo/anti_uav_rgb/weights/best.pt runs/yolo/anti_uav_rgb_neg2/weights/best.pt

Each model gets its own panel on the SAME frame, with every box drawn above --conf and its
confidence. Once per second the terminal prints each model's highest confidence, so you can
compare, e.g., a drone image on a phone vs. your hair. Press q in the window or Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def open_source(src: str):
    if src.isdigit():
        idx = int(src)
        if sys.platform == "win32":  # DirectShow is more reliable than MSMF for many webcams
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(idx)
    return cv2.VideoCapture(src)


def short_name(w: str) -> str:
    p = Path(w)
    return p.parent.parent.name if p.parent.name == "weights" else p.stem


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="+", required=True, help="one or more YOLO .pt files to compare")
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--conf", type=float, default=0.05, help="lowest confidence to draw")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--height", type=int, default=480, help="panel height on screen")
    ap.add_argument("--no-show", action="store_true", help="print only (no window)")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    models = [(short_name(w), YOLO(w)) for w in a.weights]
    cap = open_source(a.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.source!r} - is another program using the camera?")
    print("models:", ", ".join(n for n, _ in models), "| q or Ctrl+C to stop", flush=True)

    best = {n: 0.0 for n, _ in models}
    t_print, n = time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if a.source.isdigit():
                    time.sleep(0.05)
                    continue
                break
            panels = []
            for name, m in models:
                r = m.predict(frame, conf=a.conf, imgsz=a.imgsz, device=a.device, verbose=False)[0]
                view = frame.copy()
                confs = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
                for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy().astype(int), confs):
                    col = (0, 0, 255) if c >= 0.5 else (0, 200, 255) if c >= 0.25 else (200, 200, 200)
                    cv2.rectangle(view, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(view, f"{c:.2f}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                top = float(confs.max()) if len(confs) else 0.0
                best[name] = max(best[name], top)
                cv2.rectangle(view, (0, 0), (view.shape[1], 30), (0, 0, 0), -1)
                cv2.putText(view, f"{name}  top {top:.2f}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (255, 255, 255), 2)
                s = a.height / view.shape[0]
                panels.append(cv2.resize(view, (int(view.shape[1] * s), a.height)))
            if not a.no_show:
                cv2.imshow("YOLO live comparison (q to quit)", np.hstack(panels))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            n += 1
            if time.time() - t_print >= 1.0:
                print("  ".join(f"{k}: top {v:.2f}" for k, v in best.items()), flush=True)
                best = {k: 0.0 for k in best}
                t_print = time.time()
            if a.max_frames and n >= a.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not a.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

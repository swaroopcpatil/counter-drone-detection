"""Command line entry point.

    python -m counter_drone demo  --out runs/demo
    python -m counter_drone run   --rgb 0 --thermal 1 --homography calib.json --display
    python -m counter_drone calibrate --points pairs.json --out calib.json
    python -m counter_drone calibrate --rgb-image rgb.png --thermal-image th.png --pattern 9x6 --out calib.json
    python -m counter_drone config --out my_site.json      # dump defaults to edit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .calibration import (homography_from_checkerboard, homography_from_points, load_homography,
                          save_homography)
from .config import Config
from .pipeline import CounterDronePipeline


def _writer(path: Path, frame_shape, fps):
    h, w = frame_shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    return vw if vw.isOpened() else None


def _apply_yolo_args(cfg: Config, a) -> Config:
    """--rgb-yolo / --thermal-yolo weights switch that sensor to the hybrid backend."""
    for sensor in ("rgb", "thermal"):
        w = getattr(a, f"{sensor}_yolo", None)
        if w:
            c = getattr(cfg, sensor)
            c.yolo_weights, c.backend = w, getattr(a, "yolo_mode", "hybrid")
            if getattr(a, "yolo_device", None):
                c.yolo_device = a.yolo_device
            if getattr(a, f"{sensor}_imgsz", None):
                c.yolo_imgsz = getattr(a, f"{sensor}_imgsz")
    return cfg


def _add_yolo_args(p):
    p.add_argument("--rgb-yolo", help="YOLO weights for the RGB camera (enables hybrid backend)")
    p.add_argument("--thermal-yolo", help="YOLO weights for the thermal camera (enables hybrid backend)")
    p.add_argument("--yolo-mode", default="hybrid", choices=["hybrid", "yolo"],
                   help="hybrid = classical recall + YOLO labels (default); yolo = YOLO only")
    p.add_argument("--yolo-device", help="cpu | 0 (first CUDA GPU) | mps")
    p.add_argument("--rgb-imgsz", type=int, help="YOLO inference size for RGB (default: config, 960)")
    p.add_argument("--thermal-imgsz", type=int, help="YOLO inference size for thermal (default: config, 320)")


def cmd_yolo_dataset(a):
    from . import yolo_data as yd
    if a.kind == "synthetic":
        y = yd.export_synthetic(a.out, a.modality, n_scenes=a.scenes)
    elif a.kind == "anti-uav":
        y = yd.convert_anti_uav(a.src, a.out, "ir" if a.modality == "thermal" else "rgb", stride=a.stride,
                                crop=a.size if a.modality == "rgb" else None)
    elif a.kind == "coco":
        y = yd.convert_coco(a.src, a.images, a.out)
    else:  # crop
        y = yd.crop_dataset(a.src, a.out, a.size)
    print(f"dataset written: {y}")


def cmd_yolo_autolabel(a):
    from .yolo_data import autolabel
    H = load_homography(a.homography) if a.homography else None
    cfg = Config.load(a.config) if a.config else None
    print(json.dumps(autolabel(a.rgb, a.thermal, a.out, H, cfg, a.stride, a.min_prob), indent=2))
    print(f"draft labels in {a.out}/(rgb|thermal) - review/correct them (CVAT, Label Studio) before training")


def cmd_anti_uav_eval(a):
    from .anti_uav import evaluate_dataset, evaluate_dataset_per_sensor
    cfg = _apply_yolo_args(Config.load(a.config) if a.config else Config(), a)
    if a.per_sensor:
        s = evaluate_dataset_per_sensor(a.root, cfg, a.out, a.max_frames, a.min_prob, a.max_seqs)
    else:
        s = evaluate_dataset(a.root, cfg, a.out, a.render, a.max_frames, a.min_prob)
    print(json.dumps(s, indent=2))


def cmd_yolo_train(a):
    from .yolo_data import train
    best = train(a.data, a.model, a.imgsz, a.epochs, a.batch, a.device, a.project, a.name)
    print(f"best weights: {best}")


def cmd_demo(a):
    from .evaluate import Evaluator
    from .synthetic import SyntheticScene
    from .visualize import render

    cfg = _apply_yolo_args(Config.load(a.config) if a.config else Config(), a)
    scene = SyntheticScene(seed=a.seed, duration=a.duration)
    pipe = CounterDronePipeline(cfg, scene.H_thermal_to_rgb, scene.rgb_size, scene.thermal_size)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ev = Evaluator()
    vw = None
    log = (out / "tracks.jsonl").open("w")
    events = (out / "events.jsonl").open("w")
    t_proc = []
    snap_at = {int(x * scene.fps) for x in (6, 12, 20, 27)}
    for i, (t, rgb, thermal, gts) in enumerate(scene):
        t0 = time.perf_counter()
        res = pipe.process(rgb, thermal, t)
        t_proc.append(time.perf_counter() - t0)
        phase = "dusk" if scene.dusk[0] <= t <= scene.dusk[1] else "day"
        ev.add(res, gts, phase)
        log.write(json.dumps(res.to_dict()) + "\n")
        for e in res.events:
            events.write(json.dumps(e.to_dict()) + "\n")
            print(f"[{t:6.2f}s] {e.message}")
        if not a.no_video or i in snap_at:
            frame = render(rgb, res, scene.H_thermal_to_rgb, scene.thermal_size)
            if i in snap_at:
                cv2.imwrite(str(out / f"frame_{i:04d}.png"), frame)
            if not a.no_video:
                vw = vw or _writer(out / "demo.mp4", frame.shape, scene.fps)
                if vw:
                    vw.write(frame)
    if vw:
        vw.release()
    log.close()
    events.close()
    summ = ev.summary()
    summ["mean_ms_per_frame"] = round(1e3 * float(np.mean(t_proc)), 1)
    (out / "metrics.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    print(f"outputs in {out}/")


def cmd_run(a):
    from .sources import PairedSource
    from .visualize import render

    cfg = _apply_yolo_args(Config.load(a.config) if a.config else Config(), a)
    src = PairedSource(a.rgb, a.thermal)
    frames = iter(src)
    first = next(frames)
    _, r0, t0_ = first
    rgb_size = (r0.shape[1], r0.shape[0]) if r0 is not None else tuple(a.rgb_size)
    th_size = (t0_.shape[1], t0_.shape[0]) if t0_ is not None else tuple(a.thermal_size)
    if a.homography:
        H = load_homography(a.homography)
    else:
        # no calibration: assume the two cameras are boresighted with the thermal image filling the RGB image
        print("WARNING: no --homography given; assuming thermal FOV == RGB FOV. Calibrate for real use.",
              file=sys.stderr)
        H = np.diag([rgb_size[0] / th_size[0], rgb_size[1] / th_size[1], 1.0])
    pipe = CounterDronePipeline(cfg, H, rgb_size, th_size)
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        log = (out / "tracks.jsonl").open("w")
    vw = None
    t_last = time.perf_counter()
    fps = 0.0
    try:
        import itertools
        for t, rgb, thermal in itertools.chain([first], frames):
            res = pipe.process(rgb, thermal, t)
            for e in res.events:
                print(json.dumps(e.to_dict()), flush=True)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(1e-3, now - t_last)
            t_last = now
            if out:
                log.write(json.dumps(res.to_dict()) + "\n")
            if a.display or (out and a.record):
                frame = render(rgb if rgb is not None else np.zeros((rgb_size[1], rgb_size[0], 3), np.uint8),
                               res, H, th_size, fps=fps)
                if out and a.record:
                    vw = vw or _writer(out / "run.mp4", frame.shape, 15)
                    if vw:
                        vw.write(frame)
                if a.display:
                    cv2.imshow("counter-drone", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        src.release()
        if vw:
            vw.release()


def cmd_calibrate(a):
    if a.points:
        d = json.loads(Path(a.points).read_text())
        H, rms = homography_from_points(d["thermal"], d["rgb"])
        tsz, rsz = d.get("thermal_size"), d.get("rgb_size")
    else:
        rgb = cv2.imread(a.rgb_image, cv2.IMREAD_COLOR)
        th = cv2.imread(a.thermal_image, cv2.IMREAD_UNCHANGED)
        cols, rows = (int(v) for v in a.pattern.lower().split("x"))
        H, rms = homography_from_checkerboard(rgb, th, (cols, rows))
        tsz, rsz = [th.shape[1], th.shape[0]], [rgb.shape[1], rgb.shape[0]]
    save_homography(a.out, H, tsz, rsz, rms)
    print(f"H (thermal->RGB) saved to {a.out}, RMS reprojection error {rms:.2f} px")
    if rms > 3:
        print("WARNING: RMS > 3 px - check correspondences / board detection", file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(prog="counter_drone", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run on a synthetic RGB+thermal scene and score against ground truth")
    d.add_argument("--out", default="runs/demo")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--duration", type=float, default=30.0)
    d.add_argument("--config")
    d.add_argument("--no-video", action="store_true")
    _add_yolo_args(d)
    d.set_defaults(fn=cmd_demo)

    r = sub.add_parser("run", help="run on cameras, streams or recorded files")
    r.add_argument("--rgb", help="device index, file, or URL")
    r.add_argument("--thermal", help="device index, file, or URL")
    r.add_argument("--homography", help="JSON from `calibrate`")
    r.add_argument("--config")
    _add_yolo_args(r)
    r.add_argument("--out", help="directory for tracks.jsonl (and run.mp4 with --record)")
    r.add_argument("--record", action="store_true")
    r.add_argument("--display", action="store_true", help="needs opencv-python (non-headless)")
    r.add_argument("--rgb-size", type=int, nargs=2, default=[1920, 1080])
    r.add_argument("--thermal-size", type=int, nargs=2, default=[640, 512])
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("calibrate", help="estimate the thermal->RGB homography")
    c.add_argument("--points", help='JSON {"thermal": [[x,y],...], "rgb": [[x,y],...]}')
    c.add_argument("--rgb-image")
    c.add_argument("--thermal-image")
    c.add_argument("--pattern", default="9x6", help="inner corners, cols x rows")
    c.add_argument("--out", default="calib.json")
    c.set_defaults(fn=cmd_calibrate)

    yd = sub.add_parser("yolo-dataset", help="build / convert a YOLO training dataset")
    yd.add_argument("kind", choices=["synthetic", "anti-uav", "coco", "crop"])
    yd.add_argument("--out", required=True)
    yd.add_argument("--modality", choices=["rgb", "thermal"], default="rgb")
    yd.add_argument("--src", help="anti-uav: dataset root; coco: annotations json; crop: dataset dir")
    yd.add_argument("--images", help="coco: image directory")
    yd.add_argument("--scenes", type=int, default=60, help="synthetic: number of random scenes")
    yd.add_argument("--stride", type=int, default=5, help="anti-uav: keep every Nth frame")
    yd.add_argument("--size", type=int, default=960, help="crop size (crop, and anti-uav RGB)")
    yd.set_defaults(fn=cmd_yolo_dataset)

    al = sub.add_parser("yolo-autolabel", help="draft YOLO labels from your own footage via the tracker")
    al.add_argument("--rgb")
    al.add_argument("--thermal")
    al.add_argument("--homography")
    al.add_argument("--config")
    al.add_argument("--out", required=True)
    al.add_argument("--stride", type=int, default=10)
    al.add_argument("--min-prob", type=float, default=0.7)
    al.set_defaults(fn=cmd_yolo_autolabel)

    ae = sub.add_parser("anti-uav-eval", help="score the pipeline on Anti-UAV RGB+IR sequences (official accuracy)")
    ae.add_argument("--root", required=True, help="e.g. Anti-UAV-RGBT/test")
    ae.add_argument("--out", default="runs/anti_uav_eval")
    ae.add_argument("--config")
    ae.add_argument("--render", type=int, default=0, help="write annotated videos for the first N sequences")
    ae.add_argument("--max-frames", type=int, help="limit frames per sequence (quick runs)")
    ae.add_argument("--min-prob", type=float, default=0.3, help="min p(drone) to report a target")
    ae.add_argument("--per-sensor", action="store_true",
                    help="score each camera on its own (no fusion/calibration): detector-only and tracked")
    ae.add_argument("--max-seqs", type=int, help="per-sensor: only the first N sequences")
    _add_yolo_args(ae)
    ae.set_defaults(fn=cmd_anti_uav_eval)

    yt = sub.add_parser("yolo-train", help="fine-tune a YOLO model on a dataset (data.yaml)")
    yt.add_argument("--data", required=True)
    yt.add_argument("--model", default="yolo11n.pt", help="pretrained start point (yolo11n/s/m.pt or your best.pt)")
    yt.add_argument("--imgsz", type=int, default=640)
    yt.add_argument("--epochs", type=int, default=50)
    yt.add_argument("--batch", type=int, default=8)
    yt.add_argument("--device", default="cpu", help="cpu | 0 (GPU) | mps")
    yt.add_argument("--project", default="runs/yolo")
    yt.add_argument("--name", default="train")
    yt.set_defaults(fn=cmd_yolo_train)

    g = sub.add_parser("config", help="write the default config to a JSON file for editing")
    g.add_argument("--out", default="config.json")
    g.set_defaults(fn=lambda a: (Config().save(a.out), print(f"wrote {a.out}")))

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

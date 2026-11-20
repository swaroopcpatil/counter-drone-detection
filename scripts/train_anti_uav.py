"""Train drone detectors on the real Anti-UAV300 RGB+IR dataset, on your GPU.

Run from the project root on the GPU machine:

    python scripts/train_anti_uav.py --download                 # fetch Anti-UAV300 from Google Drive (~10 GB)
    python scripts/train_anti_uav.py --data-dir D:/Anti-UAV300  # or use an already-downloaded copy

Steps (each is skipped if its output already exists; an interrupted training run is
resumed from its last finished epoch - stop any time with Ctrl+C and rerun the same command):
  1. check GPU / PyTorch (RTX 50-series needs a CUDA 12.8+ PyTorch build)
  2. download + extract Anti-UAV300 (optional)
  3. convert train/val sequences to YOLO datasets (IR full frames, RGB native-res crops)
  4. fine-tune YOLO on IR and on RGB
  5. evaluate the full RGB+IR pipeline on the test split (official Anti-UAV accuracy) and
     render a few annotated videos (yellow = ground truth)

Dataset: https://github.com/ZhaoJ9014/Anti-UAV (Anti-UAV300 is the RGB+IR version).
"""

from __future__ import annotations

import argparse
import os
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ANTI_UAV300_GDRIVE_ID = "1NPYaop35ocVTYWHOYQQHn8YHsM9jmLGr"


def step(msg):
    print(f"\n=== {msg} ===", flush=True)


def check_gpu(device: str):
    step("1/5 environment")
    try:
        import torch
        import ultralytics
    except ImportError as e:
        sys.exit(f"missing package: {e.name}. Install with:\n"
                 "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
                 "  pip install ultralytics gdown")
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), ultralytics {ultralytics.__version__}")
    if device == "cpu":
        print("WARNING: training on CPU - this will take a very long time")
        return
    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not visible to PyTorch. Install the CUDA build:\n"
                 "  pip uninstall -y torch torchvision\n"
                 "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
                 "and make sure the NVIDIA driver is up to date (nvidia-smi).")
    idx = int(device.split(",")[0]) if device[0].isdigit() else 0
    name = torch.cuda.get_device_name(idx)
    cap = torch.cuda.get_device_capability(idx)
    mem = torch.cuda.get_device_properties(idx).total_memory / 2**30
    print(f"GPU {idx}: {name}, compute capability {cap[0]}.{cap[1]}, {mem:.1f} GB")
    if cap[0] >= 12:  # Blackwell (RTX 50xx): kernels only in CUDA >= 12.8 builds
        cuda = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
        if cuda < (12, 8):
            sys.exit(f"{name} needs a PyTorch build with CUDA >= 12.8 (you have {torch.version.cuda}):\n"
                     "  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    x = torch.randn(256, 256, device=f"cuda:{idx}")
    (x @ x).sum().item()  # fails fast if the kernels are missing for this GPU
    print("GPU self-test OK")


def download(dest: Path) -> Path:
    step("2/5 download Anti-UAV300")
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.rglob("infrared.json")):
        print(f"already extracted under {dest}")
        return dest
    try:
        import gdown
    except ImportError:
        sys.exit("pip install gdown   (or download manually from the Anti-UAV README and pass --data-dir)")
    archive = dest / "Anti-UAV300.zip"
    if not archive.exists():
        gdown.download(id=ANTI_UAV300_GDRIVE_ID, output=str(archive), quiet=False)
    if not archive.exists() or archive.stat().st_size < 10_000_000:
        sys.exit("download failed (Google Drive quota?). Download it in a browser from the Anti-UAV README "
                 "(Google Drive or Baidu link), extract it, and re-run with --data-dir <folder>")
    print("extracting ...")
    shutil.unpack_archive(str(archive), str(dest))
    return dest


def find_split_root(data: Path) -> Path:
    """Folder holding train/ val/ test/ (layout varies between releases)."""
    from counter_drone.yolo_data import anti_uav_sequences
    seqs = anti_uav_sequences(data, "ir")
    if not seqs:
        sys.exit(f"no Anti-UAV sequences (infrared.json) found under {data}")
    for s in seqs:
        for parent in s.parents:
            if parent.name.lower() in ("train", "val", "test"):
                return parent.parent
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="datasets/Anti-UAV300", help="extracted dataset (or download target)")
    ap.add_argument("--download", action="store_true", help="download Anti-UAV300 from Google Drive")
    ap.add_argument("--work", default="datasets/anti_uav_yolo", help="converted YOLO datasets")
    ap.add_argument("--device", default="0", help="0 = first CUDA GPU; cpu for a dry run")
    ap.add_argument("--model", default="yolo11s.pt", help="yolo11n.pt (fastest) / yolo11s.pt / yolo11m.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--stride", type=int, default=5, help="keep every Nth video frame (consecutive frames are near-duplicates)")
    ap.add_argument("--rgb-crop", type=int, default=960, help="native-resolution RGB crop size")
    ap.add_argument("--ir-batch", type=int, default=16)
    ap.add_argument("--rgb-batch", type=int, default=8, help="lower to 4 if you run out of GPU memory")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                    help="data-loading processes (raise if the GPU sits idle between batches)")
    ap.add_argument("--cache", choices=["ram", "disk"], help="cache decoded images: much faster epochs "
                    "(ram needs ~1 MB per image of RAM, disk the same in free disk space)")
    ap.add_argument("--fraction", type=float, default=1.0, help="train on this share of the images (e.g. 0.5)")
    ap.add_argument("--hours", type=float, help="time budget per model in hours: ultralytics re-plans the "
                    "remaining epochs and learning-rate schedule to finish in this time (also works on resume)")
    ap.add_argument("--retrain", action="store_true", help="discard existing runs and train from scratch")
    ap.add_argument("--eval-seqs", type=int, default=20, help="test sequences to evaluate (0 = all)")
    ap.add_argument("--eval-frames", type=int, default=600, help="max frames per test sequence")
    ap.add_argument("--render", type=int, default=3, help="annotated videos to write")
    ap.add_argument("--only", choices=["ir", "rgb"], help="train just one modality")
    a = ap.parse_args()

    from counter_drone.yolo_data import convert_anti_uav, train

    t0 = time.time()
    check_gpu(a.device)
    data = download(Path(a.data_dir)) if a.download else Path(a.data_dir)
    root = find_split_root(data)
    print(f"dataset root: {root}  (splits: {sorted(p.name for p in root.iterdir() if p.is_dir())})")

    step("3/5 convert to YOLO")
    work = Path(a.work)
    mods = [a.only] if a.only else ["ir", "rgb"]
    for m in mods:
        out = work / m
        if (out / "data.yaml").exists():
            print(f"{m}: already converted ({out})")
            continue
        convert_anti_uav(root, out, m, stride=a.stride, crop=a.rgb_crop if m == "rgb" else None)
        n = {s: len(list((out / "images" / s).glob("*"))) for s in ("train", "val")}
        print(f"{m}: {n} images -> {out}")

    step("4/5 train")
    from counter_drone.yolo_data import resume, training_state
    weights = {}
    for m in mods:
        run_dir = ROOT / "runs" / "yolo" / f"anti_uav_{m}"
        state, ep = training_state(run_dir)
        extra = {"workers": a.workers}
        if a.cache:
            extra["cache"] = a.cache
        if a.hours:
            extra["time"] = a.hours
        if state == "done" and not a.retrain:
            best = run_dir / "weights" / "best.pt"
            print(f"{m}: training already finished -> {best}  (--retrain to start over)")
        elif state == "partial" and not a.retrain:
            print(f"{m}: resuming interrupted training after epoch {ep}", flush=True)
            best = resume(run_dir, batch=a.ir_batch if m == "ir" else a.rgb_batch, device=a.device, **extra)
        else:
            if run_dir.exists() and a.retrain:
                shutil.rmtree(run_dir)
            n_train = len(list((work / m / "images" / "train").glob("*")))
            print(f"{m}: training from {a.model} on {n_train} images for {a.epochs} epochs", flush=True)
            best = train(str(work / m / "data.yaml"), a.model, imgsz=640 if m == "ir" else a.rgb_crop,
                         epochs=a.epochs, batch=a.ir_batch if m == "ir" else a.rgb_batch, device=a.device,
                         project=str(ROOT / "runs" / "yolo"), name=f"anti_uav_{m}",
                         fraction=a.fraction, **extra)
        weights[m] = best
        print(f"{m}: {best}")

    step("5/5 evaluate on the Anti-UAV test split (full RGB+IR pipeline)")
    from counter_drone.anti_uav import evaluate_dataset
    from counter_drone.config import Config
    test = root / "test" if (root / "test").exists() else root / "val"
    results = {}
    for mode in ("classical", "yolo", "hybrid"):   # classical = baseline without the trained models
        cfg = Config()
        for m, sensor in (("rgb", cfg.rgb), ("ir", cfg.thermal)):
            if m in weights and mode != "classical":
                sensor.backend, sensor.yolo_weights, sensor.yolo_device = mode, str(weights[m]), a.device
                sensor.yolo_imgsz = 1920 if m == "rgb" else 640
        out = ROOT / "runs" / "anti_uav_eval" / mode
        sub = test
        if a.eval_seqs:
            from counter_drone.yolo_data import anti_uav_sequences
            pick = anti_uav_sequences(test, "ir")[: a.eval_seqs]
            sub = out / "_subset"
            if sub.exists():
                shutil.rmtree(sub)
            sub.mkdir(parents=True)
            for s in pick:  # junction-free "view": copy only the small label files, link videos
                d = sub / s.name
                d.mkdir()
                for f in s.iterdir():
                    try:
                        (d / f.name).symlink_to(f.resolve())
                    except OSError:  # Windows without symlink rights
                        shutil.copy2(f, d / f.name)
        results[mode] = evaluate_dataset(sub, cfg, out, render=a.render if mode == "yolo" else 0,
                                         max_frames=a.eval_frames or None, min_prob=0.3)
        print(mode, json.dumps(results[mode]))
    (ROOT / "runs" / "anti_uav_eval" / "summary.json").write_text(json.dumps(results, indent=2))

    score = {k: (v.get("acc_ir") or 0) + (v.get("acc_rgb") or 0) for k, v in results.items()}
    print("\nAnti-UAV accuracy (IR + RGB) by mode: " + ", ".join(f"{k} {v:.3f}" for k, v in score.items()))
    best_mode = max(("yolo", "hybrid"), key=score.get)
    if score["classical"] >= score[best_mode]:
        print("WARNING: the trained models do not beat the classical baseline - train longer / a larger model "
              "(--model yolo11m.pt), lower --stride for more data, and check runs/yolo/anti_uav_*/results.png")
    print(f"\nDone in {(time.time() - t0) / 3600:.1f} h. Results: runs/anti_uav_eval/summary.json, "
          f"videos: runs/anti_uav_eval/yolo/videos/")
    w_rgb, w_ir = weights.get("rgb", ""), weights.get("ir", "")
    print("\nRun on your own cameras in real time (best mode on the test split: %s):" % best_mode)
    print(f"  python -m counter_drone run --rgb 0 --thermal 1 --homography calib.json "
          f"{'--rgb-yolo ' + str(w_rgb) if w_rgb else ''} {'--thermal-yolo ' + str(w_ir) if w_ir else ''} "
          f"--yolo-mode {best_mode} --yolo-device {a.device} --rgb-imgsz 1280 --thermal-imgsz 640 --display")


if __name__ == "__main__":  # required on Windows (DataLoader workers spawn processes)
    main()

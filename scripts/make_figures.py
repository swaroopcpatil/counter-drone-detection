"""Regenerate the README figures in docs/media from recorded results.

    python scripts/make_figures.py

Data sources (copied from the run logs, not re-measured here):
  * training curves - ultralytics validation mAP50-95 per epoch from the RTX 5050 runs
    (runs/yolo/anti_uav_ir/results.csv, runs/yolo/anti_uav_rgb/results.csv)
  * tracker progress - `anti-uav-eval --per-sensor` summaries, 20 Anti-UAV300 test sequences,
    up to 600 frames each (runs/anti_uav_eval/per_sensor*)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "media"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RGB_C, IR_C = "#2a78d6", "#eb6834"      # categorical slots 1-2 (validated: CVD dE 24.7, normal 33.6)

IR_MAP = [0.566, 0.572, 0.543, 0.582, 0.599, 0.606, 0.609, 0.605, 0.608, 0.609, 0.615, 0.619, 0.619,
          0.617, 0.618, 0.621, 0.62, 0.619, 0.619, 0.619, 0.62, 0.619, 0.622, 0.62, 0.619, 0.62, 0.62,
          0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.619, 0.619, 0.619]
RGB_MAP = [0.61, 0.603, 0.626, 0.639, 0.647, 0.657, 0.657, 0.666, 0.667, 0.671, 0.671, 0.679, 0.676,
           0.678, 0.68, 0.682, 0.682, 0.685, 0.683, 0.681, 0.683, 0.683, 0.683, 0.683, 0.685, 0.686,
           0.686, 0.686, 0.685, 0.686, 0.686, 0.687, 0.687, 0.687, 0.688, 0.688, 0.688, 0.688, 0.688,
           0.688, 0.689, 0.689, 0.688, 0.688, 0.688, 0.688, 0.688, 0.688, 0.688, 0.689, 0.689, 0.689,
           0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689]

VERSIONS = ["Original", "Fix 1", "Fix 2", "Fix 3"]
RGB_FOUND = [35.2, 37.5, 57.6, 76.9]    # tracked hit rate (IoU > 0.3), %
IR_FOUND = [32.6, 33.8, 58.5, 63.2]
RGB_DET, IR_DET = 75.7, 62.0            # detector alone, same sequences


def _style(ax, fig):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def training_curves():
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    _style(ax, fig)
    ax.plot(range(1, len(RGB_MAP) + 1), RGB_MAP, color=RGB_C, lw=2, label="RGB model (960 px crops)")
    ax.plot(range(1, len(IR_MAP) + 1), IR_MAP, color=IR_C, lw=2, label="Infrared model (640 px)")
    ax.annotate(f"RGB {RGB_MAP[-1]:.3f}", (len(RGB_MAP), RGB_MAP[-1]), xytext=(-4, 8),
                textcoords="offset points", ha="right", color=INK, fontsize=10, fontweight="bold")
    best = max(range(len(IR_MAP)), key=lambda i: IR_MAP[i])
    ax.annotate(f"Infrared best {IR_MAP[best]:.3f} (epoch {best + 1})\nearly stop at epoch {len(IR_MAP)}",
                (len(IR_MAP), IR_MAP[-1]),
                xytext=(8, -4), textcoords="offset points", va="top", color=INK, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 61)
    ax.set_ylim(0.52, 0.71)
    ax.set_xlabel("Epoch", color=INK2, fontsize=10)
    ax.set_ylabel("Validation mAP50-95", color=INK2, fontsize=10)
    fig.suptitle("Both YOLO11s detectors converge within ~20 epochs on Anti-UAV300", x=0.06, ha="left",
                 color=INK, fontsize=13, fontweight="bold")
    ax.set_title("mAP50 reached 0.995 (RGB) and 0.993 (infrared) on the validation split", loc="left",
                 color=INK2, fontsize=10)
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", facecolor=SURFACE)
    plt.close(fig)


def tracker_progress():
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    _style(ax, fig)
    x = range(len(VERSIONS))
    w = 0.36
    gap = 0.02
    rb = ax.bar([i - w / 2 - gap / 2 for i in x], RGB_FOUND, w, color=RGB_C, label="RGB tracked")
    ib = ax.bar([i + w / 2 + gap / 2 for i in x], IR_FOUND, w, color=IR_C, label="Infrared tracked")
    for bars in (rb, ib):
        for k in (0, len(VERSIONS) - 1):
            b = bars[k]
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.1f}%",
                    ha="center", va="bottom", color=INK, fontsize=9)
    for ref, name in ((RGB_DET, "RGB detector alone"), (IR_DET, "Infrared detector alone")):
        ax.axhline(ref, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(-0.62, ref + 1.4, f"{name} {ref:.1f}%", color=INK2, fontsize=9, va="bottom")
    ax.set_xticks(list(x), VERSIONS)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.65, len(VERSIONS) - 0.4)
    ax.set_ylabel("Frames with drone found (%)", color=INK2, fontsize=10)
    fig.suptitle("Three tracker fixes took tracking from a third of frames to above the detector",
                 x=0.06, ha="left", color=INK, fontsize=13, fontweight="bold")
    ax.set_title("Per-camera evaluation, 20 held-out Anti-UAV300 test videos (IoU > 0.3)", loc="left",
                 color=INK2, fontsize=10)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "tracker_progress.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    training_curves()
    tracker_progress()
    print(f"figures written to {OUT}")

"""Render the PCBA AOI execution-plan flow diagram to docs/flow_diagram.png.

Detection-first flow (phase 1 scope: missing / wrong parts only):

  top    — GPU training station (offline): P0 scope → P1 hardware → P2 software
           → P3 capture & label components → P4 train/validate PP-YOLOE+
           → P5 export, with phase gates
  bottom — shop-floor non-GPU PC (online): deploy → capture → align → detect
           components → rule engine verdict → operator feedback (dashed arrow
           back to training)

The anomaly branch (PatchCore) is deferred and intentionally not shown.

Regenerate:  python scripts/render_flow_diagram.py
Output:      docs/flow_diagram.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from daimon_runtime import setup_plot

setup_plot()

OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "flow_diagram.png"

# --- stage colors (fill, edge) ------------------------------------------------
SETUP = ("#dbeafe", "#2563eb")    # blue   — P0–P2 setup
TRAIN = ("#ffedd5", "#ea580c")    # orange — P3–P4 capture/train
DEPLOY = ("#f3e8ff", "#9333ea")   # purple — P5 export
OPERATE = ("#dcfce7", "#16a34a")  # green  — shop-floor operations
GATE = "#f59e0b"                  # amber  — gate diamonds

BW, BH = 2.4, 1.55  # block width / height
TOP_Y, BOT_Y = 7.6, 2.1  # block center y per lane


def box(ax, xc, yc, title, sub, colors, title_size=10):
    fill, edge = colors
    ax.add_patch(FancyBboxPatch(
        (xc - BW / 2, yc - BH / 2), BW, BH,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        facecolor=fill, edgecolor=edge, linewidth=1.8, zorder=3,
    ))
    ax.text(xc, yc + 0.32, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color="#111827", zorder=4)
    ax.text(xc, yc - 0.32, sub, ha="center", va="center",
            fontsize=8.2, color="#374151", zorder=4)


def arrow(ax, p1, p2, color="#4b5563", dashed=False, lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=16,
        linewidth=lw, color=color, zorder=2,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
    ))


def gate(ax, xc, yc, label, label_side="above", dy=None):
    s = 0.20
    ax.add_patch(plt.Polygon(
        [(xc, yc + s), (xc + s, yc), (xc, yc - s), (xc - s, yc)],
        facecolor=GATE, edgecolor="#b45309", linewidth=1.4, zorder=4,
    ))
    if dy is None:
        dy = 0.52 if label_side == "above" else -0.52
    ax.text(xc, yc + dy, label, ha="center",
            va="bottom" if label_side == "above" else "top",
            fontsize=7.6, color="#92400e", zorder=4)


def main() -> None:
    fig, ax = plt.subplots(figsize=(17, 9.6))
    ax.set_xlim(0, 16.9)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # --- swimlane backgrounds ---------------------------------------------------
    ax.add_patch(plt.Rectangle((0.15, 5.6), 16.6, 4.25,
                 facecolor="#eff6ff", edgecolor="none", zorder=0))
    ax.add_patch(plt.Rectangle((0.15, 0.15), 16.6, 4.25,
                 facecolor="#f0fdf4", edgecolor="none", zorder=0))
    ax.text(0.45, 9.55, "GPU TRAINING STATION  (offline: setup → capture & label → train → export)",
            fontsize=11.5, fontweight="bold", color="#1e3a8a", va="center")
    ax.text(0.45, 4.1, "SHOP-FLOOR PC — NO GPU  (online: capture → align → detect → rule engine → verdict)",
            fontsize=11.5, fontweight="bold", color="#14532d", va="center")

    xs = [1.55, 4.25, 6.95, 9.65, 12.35, 15.05]

    # --- top lane: P0–P5 ---------------------------------------------------------
    box(ax, xs[0], TOP_Y, "P0 · Scope lock",
        "1 PTH PCBA, component side\nmissing / wrong parts only", SETUP)
    box(ax, xs[1], TOP_Y, "P1 · Hardware setup",
        "rig · camera (manual!)\n2× LED bars · fiducials", SETUP)
    box(ax, xs[2], TOP_Y, "P2 · Software setup",
        "CUDA · aoi-detect env\nPaddleDetection clone", SETUP)
    box(ax, xs[3], TOP_Y, "P3 · Capture & label",
        "50–100 boards, COCO\n+ expected_components.json", TRAIN)
    box(ax, xs[4], TOP_Y, "P4 · Train & validate",
        "PP-YOLOE+ fine-tune\nmissing-part recall ≈100%", TRAIN)
    box(ax, xs[5], TOP_Y, "P5 · Export",
        "Paddle inference model\nCPU (or ONNX)", DEPLOY)

    for i in range(5):
        arrow(ax, (xs[i] + BW / 2, TOP_Y), (xs[i + 1] - BW / 2, TOP_Y))

    # gates on top-lane transitions (labels below the diamond — empty lane space,
    # pushed clear of the block bottom edges)
    gate(ax, (xs[1] + xs[2]) / 2, TOP_Y, "G1 · rig stable\nfixed focus/exposure",
         label_side="below", dy=-1.05)
    gate(ax, (xs[3] + xs[4]) / 2, TOP_Y, "G2 · ≥50 labeled boards\nevery class ≥100 inst",
         label_side="below", dy=-1.05)
    gate(ax, (xs[4] + xs[5]) / 2, TOP_Y, "G3 · recall ≈100%\nFR within target",
         label_side="below", dy=-1.05)

    # --- deploy transition: P5 down into bottom lane ------------------------------
    box(ax, xs[5], BOT_Y, "P6 · Deploy & go-live",
        "copy models/ + app\nautostart · operator SOP", DEPLOY)
    arrow(ax, (xs[5], TOP_Y - BH / 2), (xs[5], BOT_Y + BH / 2), color="#9333ea", lw=2.2)
    gate(ax, xs[5], (TOP_Y + BOT_Y) / 2, "G4 · CPU latency\nwithin tact time", label_side="above")

    # --- bottom lane: flows right → left -------------------------------------------
    bot_titles = [
        ("Capture board", "jig + fiducials\ncomponent side up"),
        ("Align", "src/align_board.py\nhomography to golden"),
        ("Detect components", "PP-YOLOE+ on CPU\n(paddle.inference)"),
        ("Rule engine → verdict", "vs expected_components.json\nOK / NG banner"),
        ("NG handling", "confirm / override\n→ results/feedback.jsonl"),
    ]
    bot_xs = [xs[4], xs[3], xs[2], xs[1], xs[0]]  # right → left
    for xc, (title, sub) in zip(bot_xs, bot_titles):
        box(ax, xc, BOT_Y, title, sub, OPERATE, title_size=10)
    arrow(ax, (xs[5] - BW / 2, BOT_Y), (xs[4] + BW / 2, BOT_Y))
    for i in range(4):
        arrow(ax, (bot_xs[i] - BW / 2, BOT_Y), (bot_xs[i + 1] + BW / 2, BOT_Y),
              color="#16a34a")

    # --- feedback loop: NG handling back up to P4 (retrain) -----------------------
    # Route: left edge of "NG handling" box -> far-left margin -> above the top
    # lane -> drop into the top of the P4 block. Kept in the left margin so the
    # dashed line does not cross the lane header text or the G2 gate.
    fb_x = 0.35
    fb_top_y = 9.15
    fb_start = (xs[0] - BW / 2, BOT_Y)
    ax.plot([fb_start[0], fb_x], [fb_start[1], fb_start[1]],
            color="#16a34a", lw=1.8, ls="--", zorder=1)
    ax.plot([fb_x, fb_x], [fb_start[1], fb_top_y],
            color="#16a34a", lw=1.8, ls="--", zorder=1)
    ax.plot([fb_x, xs[4]], [fb_top_y, fb_top_y],
            color="#16a34a", lw=1.8, ls="--", zorder=1)
    arrow(ax, (xs[4], fb_top_y), (xs[4], TOP_Y + BH / 2),
          color="#16a34a", dashed=True)
    ax.text(0.55, fb_top_y + 0.12,
            "feedback loop: false rejects / new component lots → label more boards + retrain (weekly review)",
            fontsize=8.6, color="#15803d", va="bottom", style="italic")

    # --- legend -------------------------------------------------------------------
    legend_items = [
        ("Setup (P0–P2)", SETUP), ("Capture / label / train (P3–P4)", TRAIN),
        ("Export / deploy (P5–P6)", DEPLOY), ("Operate (shop floor)", OPERATE),
    ]
    lx = 0.55
    for label, (fill, edge) in legend_items:
        ax.add_patch(plt.Rectangle((lx, 0.32), 0.32, 0.32,
                     facecolor=fill, edgecolor=edge, linewidth=1.5))
        ax.text(lx + 0.42, 0.48, label, fontsize=8.6, va="center", color="#111827")
        lx += 0.42 + len(label) * 0.115 + 0.35
    ax.add_patch(plt.Polygon([(lx + 0.16, 0.64), (lx + 0.32, 0.48),
                              (lx + 0.16, 0.32), (lx, 0.48)],
                 facecolor=GATE, edgecolor="#b45309"))
    ax.text(lx + 0.42, 0.48, "Phase gate (proceed-when checklist)", fontsize=8.6,
            va="center", color="#111827")

    ax.set_title("PCBA AOI — executable plan: GPU training station → non-GPU inspection station",
                 fontsize=13.5, fontweight="bold", color="#111827", pad=14)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"written: {OUT_PATH}")


if __name__ == "__main__":
    main()

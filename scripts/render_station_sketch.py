"""Render the PTH station AOI-retrofit annotation overlay.

Loads docs/station/pth_station_original.png (165x360, low-res — all geometry
below is in ORIGINAL pixel coordinates, using the relative positions described
for the photo) and draws the retrofit layout on top:

  1. overhead camera on a new lean-tube crossbar + field-of-view cone
  2. board jig with fiducials on the ESD mat
  3. two angled LED bar lights on the rack uprights
  4. mini PC (no GPU) on the right side, under the table
  5. operator monitor at eye level, upper-left
  6. capture trigger button on the table front edge
  7. camera-to-board working-distance dimension arrow (~40–60 cm)

Regenerate:  python scripts/render_station_sketch.py
Output:      docs/station/pth_station_modification.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
from PIL import Image

from daimon_runtime import setup_plot

setup_plot()

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "station" / "pth_station_original.png"
OUT = ROOT / "docs" / "station" / "pth_station_modification.png"

# Original photo: 165 x 360 px. Reference zones (fraction of height):
#   "PTH 1" sign ~2–16% | shelves/bins ~30–55% | work surface ~55–65%
#   blue ESD mat ~65–85% | ESD chair bottom | neighboring stations at edges
W, H = 165, 360

# Colors (high contrast over the photo)
C_CAMERA = "#facc15"   # yellow
C_LIGHT = "#fb923c"    # orange
C_JIG = "#e879f9"      # magenta
C_PC = "#22d3ee"       # cyan
C_MON = "#a3e635"      # lime
C_TRIG = "#ef4444"     # red
C_DIM = "#f8fafc"      # white-ish


def callout(ax, text, xy_text, xy_target, color, ha="left"):
    """Label box with a leader line to the annotated element."""
    ax.annotate(
        text, xy=xy_target, xytext=xy_text,
        fontsize=8.6, fontweight="bold", color="#111827", ha=ha, va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=color, alpha=0.92,
                  edgecolor="white", linewidth=1.2),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                        mutation_scale=12),
        zorder=8,
    )


def main() -> None:
    photo = Image.open(SRC).convert("RGB")
    # Upscale 4x nearest so the low-res photo stays crisp as a backdrop;
    # annotation coordinates stay in original-pixel space via extent.
    photo_big = photo.resize((W * 4, H * 4), Image.NEAREST)

    fig, ax = plt.subplots(figsize=(9.5, 15.5))
    ax.imshow(photo_big, extent=(0, W, H, 0))  # y axis: top-down like the photo
    ax.set_xlim(-2, W + 2)
    ax.set_ylim(H + 2, -14)
    ax.axis("off")

    # --- 1. Camera crossbar + camera + FOV cone --------------------------------
    crossbar_y = 200  # just above the work-surface zone (~55–65% of height)
    ax.plot([14, 151], [crossbar_y, crossbar_y], color=C_CAMERA, lw=4,
            solid_capstyle="round", zorder=5)
    for jx in (14, 151):  # lean-tube joints on the uprights
        ax.add_patch(Circle((jx, crossbar_y), 3.2, facecolor=C_CAMERA,
                            edgecolor="white", lw=1.2, zorder=6))
    cam_x = W / 2  # ~82
    ax.add_patch(Rectangle((cam_x - 9, crossbar_y - 9), 18, 11,
                           facecolor="#713f12", edgecolor=C_CAMERA, lw=2.2,
                           zorder=6))
    ax.add_patch(Circle((cam_x, crossbar_y - 3), 3.4, facecolor="#111827",
                        edgecolor=C_CAMERA, lw=1.5, zorder=7))
    # FOV cone down to the jig on the mat (mat surface ≈ y 216–250 in the photo;
    # below that the chair occludes the table front edge)
    jig_cx, jig_cy = W / 2, 232
    ax.add_patch(Polygon([(cam_x, crossbar_y + 2),
                          (jig_cx - 30, jig_cy + 8), (jig_cx + 30, jig_cy + 8)],
                         closed=True, facecolor=C_CAMERA, alpha=0.14,
                         edgecolor=C_CAMERA, lw=1.2, ls="--", zorder=4))
    callout(ax, "Camera (manual focus,\nfixed exposure)", (66, 158),
            (cam_x + 9, crossbar_y - 4), C_CAMERA)

    # --- 2. Board jig + fiducials on the blue ESD mat ---------------------------
    ax.add_patch(Rectangle((jig_cx - 27, jig_cy - 10), 54, 20,
                           facecolor="none", edgecolor=C_JIG, lw=2.4, zorder=6))
    for fx in (jig_cx - 30, jig_cx + 30):
        for fy in (jig_cy - 13, jig_cy + 13):
            ax.add_patch(Circle((fx, fy), 1.9, facecolor=C_JIG,
                                edgecolor="white", lw=0.8, zorder=7))
    callout(ax, "Board jig + fiducials\n(fixed position)", (92, 292),
            (jig_cx + 27, jig_cy + 9), C_JIG)

    # --- 3. Two LED bar lights on the uprights, ~45° ----------------------------
    # Left bar
    ax.plot([20, 34], [138, 192], color=C_LIGHT, lw=5, solid_capstyle="round",
            zorder=5)
    # Right bar
    ax.plot([145, 131], [138, 192], color=C_LIGHT, lw=5, solid_capstyle="round",
            zorder=5)
    # light rays toward the work surface
    for (x0, y0, x1, y1) in [(27, 165, 62, 228), (30, 178, 72, 236),
                             (138, 165, 103, 228), (135, 178, 93, 236)]:
        ax.plot([x0, x1], [y0, y1], color=C_LIGHT, lw=0.9, alpha=0.8, zorder=4)
    callout(ax, "LED bar light ×2\ndiffuse, ~45°", (48, 108), (27, 160),
            C_LIGHT)
    ax.annotate("", xy=(136, 165), xytext=(97, 108),
                arrowprops=dict(arrowstyle="-|>", color=C_LIGHT, lw=1.6,
                                mutation_scale=12), zorder=8)

    # --- 4. Mini PC (no GPU), right side under the table ------------------------
    pc_x, pc_y = 138, 316
    ax.add_patch(Rectangle((pc_x, pc_y), 22, 15, facecolor="#164e63",
                           edgecolor=C_PC, lw=2.2, zorder=6))
    ax.plot([pc_x + 3, pc_x + 19], [pc_y + 4, pc_y + 4], color=C_PC, lw=1.2,
            zorder=7)
    callout(ax, "Mini PC — no GPU\n(runs inspection app)", (58, 336),
            (pc_x + 4, pc_y + 10), C_PC)

    # --- 5. Operator monitor at eye level, upper-left ---------------------------
    mon_x, mon_y = 10, 92
    ax.add_patch(Rectangle((mon_x, mon_y), 26, 17, facecolor="#1a2e05",
                           edgecolor=C_MON, lw=2.2, zorder=6))
    ax.add_patch(Rectangle((mon_x + 3, mon_y + 3), 20, 9, facecolor="#22c55e",
                           edgecolor="none", zorder=7))
    ax.plot([mon_x + 13, mon_x + 13], [mon_y + 17, mon_y + 24], color=C_MON,
            lw=2.4, zorder=6)  # mount arm to upright
    callout(ax, "Operator monitor\n(OK/NG verdict)", (44, 66),
            (mon_x + 26, mon_y + 6), C_MON)

    # --- 6. Capture trigger on the table front edge ------------------------------
    trig_x, trig_y = 50, 257
    ax.add_patch(Circle((trig_x, trig_y), 4.2, facecolor=C_TRIG,
                        edgecolor="white", lw=1.2, zorder=7))
    callout(ax, "Capture trigger\n(button / pedal)", (2, 292),
            (trig_x - 2, trig_y + 3), C_TRIG)

    # --- 7. Working-distance dimension arrow (right edge) ------------------------
    dim_x = 159
    ax.add_patch(FancyArrowPatch((dim_x, crossbar_y), (dim_x, jig_cy),
                                 arrowstyle="<|-|>", mutation_scale=11,
                                 color=C_DIM, lw=1.6, zorder=6))
    ax.plot([151, dim_x + 2], [crossbar_y, crossbar_y], color=C_DIM, lw=0.8,
            ls=":", zorder=5)
    ax.plot([jig_cx + 30, dim_x + 2], [jig_cy, jig_cy], color=C_DIM, lw=0.8,
            ls=":", zorder=5)
    ax.text(dim_x + 2.5, (crossbar_y + jig_cy) / 2, "WD\n40–60\ncm",
            rotation=0, fontsize=7.4, color=C_DIM, ha="left", va="center",
            fontweight="bold", zorder=7,
            bbox=dict(facecolor="#111827", alpha=0.65, edgecolor="none",
                      boxstyle="round,pad=0.2"))

    # --- title + legend ------------------------------------------------------------
    fig.suptitle("PTH 1 station — AOI retrofit layout",
                 fontsize=15, fontweight="bold", color="#111827", y=0.985)
    legend = [
        (C_CAMERA, "Camera + crossbar + FOV"),
        (C_LIGHT, "LED bar lights ×2, ~45°"),
        (C_JIG, "Board jig + fiducials"),
        (C_PC, "Mini PC (no GPU)"),
        (C_MON, "Operator monitor"),
        (C_TRIG, "Capture trigger"),
    ]
    lx, ly = 0.03, 0.035
    for i, (color, label) in enumerate(legend):
        col = i % 2
        row = i // 2
        fig.patches.append(Rectangle((lx + col * 0.30, ly - row * 0.024),
                                     0.018, 0.012, transform=fig.transFigure,
                                     facecolor=color, edgecolor="#111827"))
        fig.text(lx + col * 0.30 + 0.024, ly - row * 0.024 + 0.006, label,
                 fontsize=9, va="center", color="#111827")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160, facecolor="white")
    plt.close(fig)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()

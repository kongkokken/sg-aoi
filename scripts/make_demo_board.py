#!/usr/bin/env python
"""Generate a synthetic demo PCBA board + scripted detections for the demo mode.

Renders a plausible top-down PCBA image (Pillow + numpy only — no opencv) and
writes the golden reference plus three scripted inspection scenarios:

  * OK board                — every expected component present
  * NG: missing part (R7)   — bare pads where R7 should be
  * NG: wrong part (C3)     — an electrolytic cap where a ceramic cap is expected

Outputs (relative to the project root):

  data/golden/golden_board.jpg
  data/golden/expected_components.json
  data/demo/board_ok.jpg          + detections_ok.json
  data/demo/board_ng_missing.jpg  + detections_ng_missing.json
  data/demo/board_ng_wrong.jpg    + detections_ng_wrong.json

Bounding boxes are COCO-style ``[x, y, width, height]`` in the golden image's
pixels (the canonical aligned view), exactly as documented in data/README.md
and consumed by src/infer_pipeline.py. Class name strings must match between
expected_components.json and the detections — the rule engine compares exact
strings.

The generator is deterministic (seeded) and safe to re-run:

    python scripts/make_demo_board.py [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BOARD_W, BOARD_H = 1600, 1200
GREEN = (26, 94, 56)          # solder-mask green
TRACE_DARK = (18, 76, 44)
TRACE_LIGHT = (38, 112, 70)
COPPER = (184, 132, 55)
COPPER_EDGE = (120, 80, 30)
HOLE = (35, 30, 25)
LEAD = (170, 170, 178)
SILK = (232, 232, 232)

# Class names — must stay in sync with the detection dataset's COCO categories
# and with expected_components.json (rule engine compares exact strings).
CLS_RESISTOR = "resistor_axial"
CLS_CERAMIC = "capacitor_ceramic"
CLS_ELECTROLYTIC = "capacitor_electrolytic"
CLS_DIP = "ic_dip"
CLS_HEADER = "pin_header"
CLS_DIODE = "diode_axial"


@dataclass(frozen=True)
class Component:
    """One placed through-hole component on the demo board."""

    cid: str
    cls: str
    cx: int
    cy: int
    length: int   # major-axis extent in px
    width: int    # minor-axis extent in px
    angle: int    # 0 (horizontal) or 90 (vertical)

    @property
    def bbox(self) -> list[int]:
        """COCO-style [x, y, w, h] in board pixels."""
        w, h = (self.length, self.width) if self.angle == 0 else (self.width, self.length)
        return [self.cx - w // 2, self.cy - h // 2, w, h]


# Fixed layout: 16 through-hole components on a 1600x1200 board.
COMPONENTS: list[Component] = [
    Component("R1", CLS_RESISTOR, 200, 200, 150, 40, 0),
    Component("R2", CLS_RESISTOR, 430, 190, 150, 40, 90),
    Component("R3", CLS_RESISTOR, 640, 200, 150, 40, 0),
    Component("R4", CLS_RESISTOR, 860, 195, 150, 40, 0),
    Component("R5", CLS_RESISTOR, 1090, 200, 150, 40, 0),
    Component("R6", CLS_RESISTOR, 1320, 195, 150, 40, 0),
    Component("R7", CLS_RESISTOR, 1420, 500, 150, 40, 90),   # removed in NG-missing
    Component("C1", CLS_CERAMIC, 190, 470, 64, 64, 0),
    Component("C2", CLS_CERAMIC, 330, 480, 64, 64, 0),
    Component("C3", CLS_CERAMIC, 470, 470, 64, 64, 0),        # swapped in NG-wrong
    Component("C4", CLS_ELECTROLYTIC, 660, 480, 92, 92, 0),
    Component("U1", CLS_DIP, 900, 480, 200, 70, 90),
    Component("J1", CLS_HEADER, 1180, 480, 240, 52, 0),
    Component("D1", CLS_DIODE, 260, 760, 140, 38, 0),
    Component("C5", CLS_ELECTROLYTIC, 560, 770, 92, 92, 0),
    Component("U2", CLS_DIP, 830, 770, 200, 70, 0),
    Component("J2", CLS_HEADER, 1230, 780, 240, 52, 90),
]

RESISTOR_BANDS = [
    (160, 40, 30), (40, 120, 50), (60, 60, 65), (230, 150, 40),
    (120, 70, 30), (200, 60, 90), (90, 90, 200),
]


# ---------------------------------------------------------------------------
# Fonts / small helpers
# ---------------------------------------------------------------------------

def _font(size: int) -> ImageFont.ImageFont:
    """Best-effort TrueType font; falls back to PIL's bitmap default."""
    candidates: list[Path | str] = []
    try:
        import matplotlib  # noqa: PLC0415 - optional, ships DejaVuSans

        candidates.append(
            Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
        )
    except ImportError:
        pass
    candidates.append("arial.ttf")  # present on Windows
    for cand in candidates:
        try:
            return ImageFont.truetype(str(cand), size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _pad(draw: ImageDraw.ImageDraw, px: float, py: float, r: int = 9) -> None:
    """Through-hole annular ring: copper pad with a dark drilled hole."""
    draw.ellipse([px - r, py - r, px + r, py + r], fill=COPPER + (255,),
                 outline=COPPER_EDGE + (255,))
    draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=HOLE + (255,))


def _tile_rng(seed: int, cid: str) -> np.random.Generator:
    """Per-component RNG so removing/swapping one part never perturbs others."""
    return np.random.default_rng([seed, zlib.crc32(cid.encode("utf-8"))])


# ---------------------------------------------------------------------------
# Component tiles (drawn horizontally; rotated by the caller for angle=90)
# ---------------------------------------------------------------------------

def _tile_axial(length: int, width: int, rng: np.random.Generator,
                diode: bool) -> Image.Image:
    """Axial resistor / diode: cylindrical body, color bands, leads, pads."""
    tile = Image.new("RGBA", (length, width), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    cy = width // 2
    for px in (10, length - 10):
        _pad(d, px, cy)
    d.line([(0, cy), (length, cy)], fill=LEAD + (255,), width=4)

    bx0, bx1 = int(length * 0.22), int(length * 0.78)
    by0, by1 = int(width * 0.12), int(width * 0.88)
    radius = max(4, width // 5)
    if diode:
        body, edge = (178, 58, 42), (90, 28, 18)
    else:
        body, edge = (214, 183, 132), (122, 92, 56)
    d.rounded_rectangle([bx0, by0, bx1, by1], radius=radius,
                        fill=body + (255,), outline=edge + (255,))
    # top highlight stripe for a cylindrical look
    d.rounded_rectangle([bx0 + 3, by0 + 2, bx1 - 3, by0 + max(4, (by1 - by0) // 3)],
                        radius=radius, fill=(255, 255, 255, 55))

    span = bx1 - bx0
    if diode:
        band_x = bx0 + int(span * 0.72)  # single cathode band
        d.rectangle([band_x, by0 + 1, band_x + 7, by1 - 1], fill=(30, 30, 34, 255))
    else:
        fractions = (0.22, 0.38, 0.54, 0.72)
        picks = rng.integers(0, len(RESISTOR_BANDS), size=len(fractions))
        for frac, pick in zip(fractions, picks):
            band_x = bx0 + int(span * frac)
            d.rectangle([band_x, by0 + 1, band_x + 7, by1 - 1],
                        fill=RESISTOR_BANDS[int(pick)] + (255,))
    return tile


def _tile_ceramic(length: int, width: int) -> Image.Image:
    """Radial ceramic cap: orange disc body with two leads + pads below."""
    tile = Image.new("RGBA", (length, width), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    bx0, bx1 = int(length * 0.12), int(length * 0.88)
    by0, by1 = int(width * 0.14), int(width * 0.78)
    d.rounded_rectangle([bx0, by0, bx1, by1], radius=8,
                        fill=(206, 126, 44, 255), outline=(130, 74, 22, 255))
    d.rounded_rectangle([bx0 + 3, by0 + 2, bx1 - 3, by0 + max(4, (by1 - by0) // 3)],
                        radius=6, fill=(255, 255, 255, 45))
    for px in (length // 2 - 10, length // 2 + 10):
        d.line([(px, by1), (px, width - 6)], fill=LEAD + (255,), width=3)
        _pad(d, px, width - 8, r=6)
    return tile


def _tile_electrolytic(length: int, width: int) -> Image.Image:
    """Electrolytic cap seen top-down: dark can, metallic scored top, polarity."""
    tile = Image.new("RGBA", (length, width), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    cx, cy = length // 2, width // 2
    r = min(length, width) // 2 - 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(28, 38, 62, 255),
              outline=(10, 14, 24, 255), width=2)
    r2 = int(r * 0.78)
    d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=(198, 202, 210, 255),
              outline=(120, 126, 138, 255), width=2)
    d.line([(cx - r2 + 6, cy), (cx + r2 - 6, cy)], fill=(150, 155, 165, 255), width=2)
    d.line([(cx, cy - r2 + 6), (cx, cy + r2 - 6)], fill=(150, 155, 165, 255), width=2)
    d.arc([cx - r + 3, cy - r + 3, cx + r - 3, cy + r - 3],
          start=-40, end=40, fill=(235, 235, 240, 255), width=4)
    return tile


def _tile_dip(length: int, width: int) -> Image.Image:
    """DIP IC: black body, two pin rows, pin-1 notch and dot."""
    tile = Image.new("RGBA", (length, width), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    n_pins, pin_w, pin_h = 8, 10, 8
    for i in range(n_pins):
        x = int(14 + i * (length - 28 - pin_w) / (n_pins - 1))
        d.rectangle([x, 0, x + pin_w, pin_h], fill=(176, 180, 188, 255))
        d.rectangle([x, width - pin_h, x + pin_w, width], fill=(176, 180, 188, 255))
    d.rounded_rectangle([6, pin_h - 2, length - 6, width - pin_h + 2], radius=6,
                        fill=(26, 26, 32, 255), outline=(8, 8, 10, 255))
    d.ellipse([12, width // 2 - 6, 24, width // 2 + 6], fill=(60, 60, 68, 255))
    d.ellipse([32, pin_h + 6, 40, pin_h + 14], fill=(200, 200, 205, 255))
    return tile


def _tile_header(length: int, width: int) -> Image.Image:
    """Pin-header connector: dark plastic base with two rows of gold pins."""
    tile = Image.new("RGBA", (length, width), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    d.rounded_rectangle([0, 0, length - 1, width - 1], radius=4,
                        fill=(36, 36, 42, 255), outline=(12, 12, 16, 255))
    n, pin = 11, 14
    for i in range(n):
        x = int(10 + i * (length - 20 - pin) / (n - 1))
        for y in (6, width - 6 - pin):
            d.rectangle([x, y, x + pin, y + pin], fill=(210, 172, 66, 255),
                        outline=(140, 110, 40, 255))
            d.rectangle([x + 4, y + 4, x + pin - 4, y + pin - 4],
                        fill=(120, 95, 35, 255))
    return tile


def _render_tile(cls: str, comp: Component, seed: int) -> Image.Image:
    rng = _tile_rng(seed, comp.cid)
    if cls == CLS_RESISTOR:
        return _tile_axial(comp.length, comp.width, rng, diode=False)
    if cls == CLS_DIODE:
        return _tile_axial(comp.length, comp.width, rng, diode=True)
    if cls == CLS_CERAMIC:
        return _tile_ceramic(comp.length, comp.width)
    if cls == CLS_ELECTROLYTIC:
        return _tile_electrolytic(comp.length, comp.width)
    if cls == CLS_DIP:
        return _tile_dip(comp.length, comp.width)
    if cls == CLS_HEADER:
        return _tile_header(comp.length, comp.width)
    raise ValueError(f"Unknown component class: {cls}")


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------

def _background(rng: np.random.Generator) -> Image.Image:
    """Green solder mask with fine noise + low-frequency mottling."""
    base = np.empty((BOARD_H, BOARD_W, 3), dtype=np.float32)
    base[:] = GREEN
    noise = rng.normal(0.0, 3.5, (BOARD_H, BOARD_W, 1)).astype(np.float32)
    low = rng.normal(0.0, 1.0, (BOARD_H // 10, BOARD_W // 10)).astype(np.float32)
    low_img = Image.fromarray(low).resize((BOARD_W, BOARD_H), Image.BILINEAR)
    mottle = np.asarray(low_img, dtype=np.float32)[..., None] * 7.0
    arr = np.clip(base + noise + mottle, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _draw_traces(draw: ImageDraw.ImageDraw, rng: np.random.Generator) -> None:
    """Copper-look traces (drawn under the components) and scattered vias."""
    for _ in range(28):
        color = TRACE_DARK if rng.integers(0, 2) else TRACE_LIGHT
        width = int(rng.integers(3, 6))
        length = int(rng.integers(120, 420))
        if rng.integers(0, 2):
            x = int(rng.integers(60, BOARD_W - 460))
            y = int(rng.integers(60, BOARD_H - 60))
            pts = [(x, y), (x + length, y)]
        else:
            x = int(rng.integers(60, BOARD_W - 60))
            y = int(rng.integers(60, BOARD_H - 460))
            pts = [(x, y), (x, y + length)]
        draw.line(pts, fill=color + (255,), width=width)
        ex, ey = pts[-1]
        draw.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=COPPER + (255,))
    for _ in range(40):
        x = int(rng.integers(40, BOARD_W - 40))
        y = int(rng.integers(40, BOARD_H - 40))
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(14, 60, 34, 255))


def _draw_frame(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> None:
    draw.rectangle([18, 18, BOARD_W - 19, BOARD_H - 19], outline=(220, 220, 220, 255),
                   width=2)
    draw.text((96, 30), "DEMO BOARD REV A", font=font, fill=(235, 235, 235, 255))
    for cx, cy in [(48, 48), (BOARD_W - 48, 48), (48, BOARD_H - 48),
                   (BOARD_W - 48, BOARD_H - 48)]:
        draw.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=(120, 90, 45, 255),
                     outline=(220, 220, 220, 255), width=2)
        draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=(60, 45, 25, 255))


def _draw_silkscreen(draw: ImageDraw.ImageDraw, comp: Component,
                     font: ImageFont.ImageFont) -> None:
    """White placement outline + reference designator next to each part."""
    x, y, w, h = comp.bbox
    m = 10
    draw.rounded_rectangle([x - m, y - m, x + w + m, y + h + m], radius=6,
                           outline=SILK + (255,), width=2)
    draw.text((x - m, max(28, y - m - 26)), comp.cid, font=font,
              fill=(240, 240, 240, 255))


def _draw_bare_pads(draw: ImageDraw.ImageDraw, comp: Component) -> None:
    """Empty pads where a through-hole part should have been inserted."""
    half = comp.length / 2 - 12
    if comp.angle == 0:
        pts = [(comp.cx - half, comp.cy), (comp.cx + half, comp.cy)]
    else:
        pts = [(comp.cx, comp.cy - half), (comp.cx, comp.cy + half)]
    for px, py in pts:
        _pad(draw, px, py)


def render_board(seed: int, missing: set[str] | None = None,
                 wrong: dict[str, str] | None = None) -> Image.Image:
    """Render the full board; ``missing`` parts become bare pads, ``wrong``
    maps a designator to a substitute class drawn in its place."""
    missing = missing or set()
    wrong = wrong or {}
    rng = np.random.default_rng(seed)
    img = _background(rng)
    draw = ImageDraw.Draw(img, "RGBA")
    _draw_traces(draw, rng)
    _draw_frame(draw, _font(26))
    label_font = _font(22)
    for comp in COMPONENTS:
        _draw_silkscreen(draw, comp, label_font)
        if comp.cid in missing:
            _draw_bare_pads(draw, comp)
            continue
        tile = _render_tile(wrong.get(comp.cid, comp.cls), comp, seed)
        if comp.angle == 90:
            tile = tile.rotate(90, expand=True)
        img.paste(tile, (comp.bbox[0], comp.bbox[1]), tile)
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# JSON artifacts
# ---------------------------------------------------------------------------

def make_expected_components() -> dict:
    """Golden expected-components list (format per data/README.md)."""
    return {
        "board_name": "demo_board_revA",
        "reference_image": "golden_board.jpg",
        "components": [
            {
                "id": c.cid,
                "class": c.cls,
                "bbox": c.bbox,
                "expected_angle_deg": c.angle,
            }
            for c in COMPONENTS
        ],
    }


def make_detections(seed: int, missing: set[str] | None = None,
                    wrong: dict[str, str] | None = None) -> list[dict]:
    """Scripted detector output: one detection per rendered component.

    Scores are uniform in [0.85, 0.97] — above the config's 0.5 confidence
    threshold — and seeded so re-runs are byte-identical.
    """
    missing = missing or set()
    wrong = wrong or {}
    rng = np.random.default_rng(seed)
    detections = []
    for comp in COMPONENTS:
        if comp.cid in missing:
            continue
        detections.append({
            "class": wrong.get(comp.cid, comp.cls),
            "bbox": comp.bbox,
            "score": round(float(rng.uniform(0.85, 0.97)), 2),
        })
    return detections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for rendering + detection scores.")
    args = parser.parse_args()

    golden_dir = PROJECT_ROOT / "data" / "golden"
    demo_dir = PROJECT_ROOT / "data" / "demo"
    golden_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    missing_spec = {"R7"}
    wrong_spec = {"C3": CLS_ELECTROLYTIC}

    # Golden reference + OK demo board (same render — the golden board IS OK).
    golden = render_board(args.seed)
    golden.save(golden_dir / "golden_board.jpg", quality=92)
    golden.save(demo_dir / "board_ok.jpg", quality=92)
    (golden_dir / "expected_components.json").write_text(
        json.dumps(make_expected_components(), indent=2) + "\n", encoding="utf-8"
    )

    # NG scenario images.
    render_board(args.seed, missing=missing_spec).save(
        demo_dir / "board_ng_missing.jpg", quality=92
    )
    render_board(args.seed, wrong=wrong_spec).save(
        demo_dir / "board_ng_wrong.jpg", quality=92
    )

    # Scripted detections (what a trained detector would return on each image).
    det_seed = args.seed + 1
    for name, kwargs in (
        ("detections_ok.json", {}),
        ("detections_ng_missing.json", {"missing": missing_spec}),
        ("detections_ng_wrong.json", {"wrong": wrong_spec}),
    ):
        (demo_dir / name).write_text(
            json.dumps(make_detections(det_seed, **kwargs), indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"Golden board + expected components -> {golden_dir}")
    print(f"Demo scenarios (ok / ng_missing / ng_wrong) -> {demo_dir}")
    print(f"Components: {len(COMPONENTS)} "
          f"({len({c.cls for c in COMPONENTS})} classes); seed={args.seed}")


if __name__ == "__main__":
    main()

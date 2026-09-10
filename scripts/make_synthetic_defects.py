"""Synthesize NG board images from aligned OK images.

Real NG boards are scarce early in a project, but you still need NG samples to
validate the rule engine, tune thresholds, and demo the UI. This utility
injects the project's six defect types into OK images with pure OpenCV/numpy
operations — no heavy dependencies.

Injected defects and which branch should catch them:
* wrong_part       — paste a component patch over a different component (A)
* wrong_orientation— rotate a component patch in place (A)
* scratch          — thin dark/bright line with alpha blending (B)
* solder_ball      — small bright blob near solder joints (B)
* missing_part     — erase a component, fill with estimated background (A)

Every injected defect is recorded in a JSON manifest next to the images so the
pipeline's verdicts can be scored against ground truth.

Deterministic: same --seed -> same output.

CLI
---
    python scripts/make_synthetic_defects.py \
        --boards data/boards_ok \
        --components data/golden/expected_components.json \
        --output data/synthetic_ng \
        --per-image 2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger("make_synthetic_defects")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Component-patch-based defects need expected_components.json bboxes.
PATCH_DEFECTS = ("wrong_part", "wrong_orientation", "missing_part")
ALL_DEFECTS = PATCH_DEFECTS + ("scratch", "solder_ball")


# ---------------------------------------------------------------------------
# Individual injectors — each returns the injected-defect record
# ---------------------------------------------------------------------------

def _extract_patch(image: np.ndarray, bbox: list[float]) -> tuple[np.ndarray, list[int]]:
    x, y, w, h = (int(round(v)) for v in bbox)
    ih, iw = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(iw, x + w), min(ih, y + h)
    return image[y1:y2, x1:x2].copy(), [x1, y1, x2 - x1, y2 - y1]


def inject_missing_part(image: np.ndarray, comp: dict[str, Any]) -> dict[str, Any]:
    """Erase a component by filling its bbox with surrounding background color."""
    _, (x, y, w, h) = _extract_patch(image, comp["bbox"])
    # Background estimate: median of a ring around the bbox. PCBA substrates
    # are fairly uniform, so this erases convincingly without blending seams.
    ring = image[max(0, y - 6) : y + h + 6, max(0, x - 6) : x + w + 6]
    fill = np.median(ring.reshape(-1, 3), axis=0).astype(np.uint8)
    image[y : y + h, x : x + w] = fill
    return {"type": "missing_part", "location": [x, y, w, h],
            "detail": f"Erased {comp.get('id')} ({comp.get('class')})."}


def inject_wrong_part(
    image: np.ndarray, comp: dict[str, Any], donor: dict[str, Any]
) -> dict[str, Any]:
    """Replace a component with a patch of a *different-class* component."""
    patch, _ = _extract_patch(image, donor["bbox"])
    _, (x, y, w, h) = _extract_patch(image, comp["bbox"])
    if patch.size == 0 or w == 0 or h == 0:
        return {}
    patch = cv2.resize(patch, (w, h))
    image[y : y + h, x : x + w] = patch
    return {"type": "wrong_part", "location": [x, y, w, h],
            "detail": f"{comp.get('id')}: replaced {comp.get('class')} "
                      f"with {donor.get('class')} from {donor.get('id')}."}


def inject_wrong_orientation(image: np.ndarray, comp: dict[str, Any]) -> dict[str, Any]:
    """Rotate a component patch 90 degrees in place.

    The patch is re-centered on the bbox; corners that fall outside the bbox
    after rotation are simply clipped, which mimics a real rotated part
    overlapping its footprint.
    """
    patch, (x, y, w, h) = _extract_patch(image, comp["bbox"])
    if patch.size == 0:
        return {}
    rotated = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    rh, rw = rotated.shape[:2]
    # Fill the original footprint with background first, then stamp the
    # rotated patch centered on it.
    ring = image[max(0, y - 6) : y + h + 6, max(0, x - 6) : x + w + 6]
    fill = np.median(ring.reshape(-1, 3), axis=0).astype(np.uint8)
    image[y : y + h, x : x + w] = fill
    cx, cy = x + w // 2, y + h // 2
    px1, py1 = max(0, cx - rw // 2), max(0, cy - rh // 2)
    px2, py2 = min(image.shape[1], px1 + rw), min(image.shape[0], py1 + rh)
    image[py1:py2, px1:px2] = rotated[: py2 - py1, : px2 - px1]
    return {"type": "wrong_orientation", "location": [x, y, w, h],
            "detail": f"Rotated {comp.get('id')} by 90 degrees."}


def inject_scratch(image: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    """Draw a thin pseudo-random scratch line with alpha blending."""
    h, w = image.shape[:2]
    p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
    angle = rng.uniform(0, 2 * np.pi)
    length = int(rng.integers(w // 10, w // 3))
    p2 = (int(p1[0] + length * np.cos(angle)), int(p1[1] + length * np.sin(angle)))
    # Real scratches are usually darker than the substrate, occasionally bright.
    dark = bool(rng.integers(0, 2))
    color = (30, 30, 30) if dark else (220, 220, 220)
    thickness = int(rng.integers(1, 3))
    overlay = image.copy()
    cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
    return {"type": "scratch", "location": [x1, y1, abs(p2[0] - p1[0]) + 1,
            abs(p2[1] - p1[1]) + 1], "detail": "Synthetic scratch line."}


def inject_solder_ball(image: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    """Add a small bright blob resembling a solder sphere / debris speck."""
    h, w = image.shape[:2]
    cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
    radius = int(rng.integers(2, 6))
    overlay = image.copy()
    cv2.circle(overlay, (cx, cy), radius, (210, 215, 220), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
    return {"type": "solder_ball", "location": [cx - radius, cy - radius,
            2 * radius, 2 * radius], "detail": "Synthetic solder ball."}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def synthesize(
    boards_dir: Path,
    components_path: Path | None,
    output_dir: Path,
    per_image: int,
    max_images: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    board_paths = sorted(
        p for p in boards_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )[:max_images]
    if not board_paths:
        raise FileNotFoundError(f"No images found in {boards_dir}")

    components: list[dict[str, Any]] = []
    if components_path and components_path.is_file():
        components = json.loads(components_path.read_text(encoding="utf-8")).get(
            "components", []
        )
    else:
        LOGGER.warning(
            "No expected_components.json — patch-based defects (%s) will be skipped.",
            ", ".join(PATCH_DEFECTS),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for board_path in board_paths:
        image = cv2.imread(str(board_path))
        if image is None:
            LOGGER.warning("Unreadable image skipped: %s", board_path)
            continue

        # Pick defect types: patch defects need components, the rest always work.
        available = list(ALL_DEFECTS if components else ("scratch", "solder_ball"))
        chosen = rng.choice(available, size=min(per_image, len(available)), replace=False)

        defects: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for defect_type in chosen:
            record: dict[str, Any] = {}
            if defect_type in PATCH_DEFECTS:
                # Don't touch the same component twice in one image.
                pool = [c for c in components if c.get("id") not in used_ids]
                if not pool:
                    continue
                comp = pool[int(rng.integers(0, len(pool)))]
                used_ids.add(comp.get("id"))
                if defect_type == "missing_part":
                    record = inject_missing_part(image, comp)
                elif defect_type == "wrong_orientation":
                    record = inject_wrong_orientation(image, comp)
                else:  # wrong_part needs a donor of a different class
                    donors = [c for c in components
                              if c.get("class") != comp.get("class")
                              and c.get("id") != comp.get("id")]
                    if donors:
                        donor = donors[int(rng.integers(0, len(donors)))]
                        used_ids.add(donor.get("id"))
                        record = inject_wrong_part(image, comp, donor)
            elif defect_type == "scratch":
                record = inject_scratch(image, rng)
            else:
                record = inject_solder_ball(image, rng)

            if record:
                defects.append(record)

        if not defects:
            continue
        out_name = f"{board_path.stem}_syn{board_path.suffix}"
        cv2.imwrite(str(output_dir / out_name), image)
        manifest.append({"image": out_name, "source": board_path.name, "defects": defects})

    (output_dir / "manifest.json").write_text(
        json.dumps({"seed": seed, "boards": manifest}, indent=2), encoding="utf-8"
    )
    LOGGER.info("Wrote %d synthetic NG images + manifest to %s", len(manifest), output_dir)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize NG board images from OK images.")
    parser.add_argument("--boards", type=Path, default=Path("data/boards_ok"),
                        help="Directory of aligned OK images.")
    parser.add_argument("--components", type=Path,
                        default=Path("data/golden/expected_components.json"),
                        help="Expected components JSON (for patch-based defects).")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_ng"),
                        help="Output directory for synthetic NG images + manifest.")
    parser.add_argument("--per-image", type=int, default=2,
                        help="Number of defects to inject per image.")
    parser.add_argument("--max-images", type=int, default=50,
                        help="Cap on how many OK images to process.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed — same seed reproduces the same output.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    synthesize(args.boards, args.components, args.output,
               args.per_image, args.max_images, args.seed)


if __name__ == "__main__":
    main()

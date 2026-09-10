"""Align a captured board image to the canonical (golden) view.

Why alignment is critical
-------------------------
Both AI branches assume the board always appears in the same position, scale
and rotation:

* The anomaly branch (PatchCore) compares local appearance patches against a
  memory bank built from OK boards. A few pixels of position jitter or a
  degree of rotation shifts every patch, inflating anomaly scores everywhere
  and drowning real defects (scratches, solder balls) in noise.
* The rule engine compares detector output against ``expected_components.json``
  coordinates in golden-image space. Without alignment the coordinates are
  meaningless.

This script computes a homography (perspective transform) from the captured
image to the golden reference and warps the image into the canonical view.

Alignment strategy, in order of preference
------------------------------------------
1. **Fiducial markers** — high-contrast filled circles placed on the jig or
   the board corners. Most reliable: detect dark round blobs, take their
   centroids, and solve the homography against the same fiducials found in
   the golden image.
2. **Board corners** — fall back to the largest quadrilateral contour if the
   board has a distinct outline against the background.
3. **ORB feature matching** — last resort: match ORB keypoints between the
   capture and the golden image and estimate the homography with RANSAC.
   Works surprisingly well on feature-rich PCBs, but slower and less exact.

CLI
---
    python src/align_board.py --input <img_or_dir> --golden <golden_img> --output <dir>

Windows / Git Bash note: pass paths with forward slashes; everything here is
pathlib-based and OS-agnostic.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger("align_board")

# A homography needs at least 4 point pairs.
MIN_HOMOGRAPHY_POINTS = 4

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class AlignmentResult:
    """Outcome of aligning one image."""

    warped: np.ndarray | None
    method: str  # "fiducials" | "corners" | "orb" | "failed"
    homography: np.ndarray | None = None
    messages: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fiducial detection
# ---------------------------------------------------------------------------

def detect_fiducials(image: np.ndarray, n_fiducials: int = 4) -> np.ndarray | None:
    """Detect round high-contrast fiducial markers and return their centroids.

    Fiducials are filled dark circles on the jig/board border. We threshold
    the grayscale image for very dark regions, keep blobs whose contour is
    approximately circular, and return the ``n_fiducials`` largest ones.

    Returns an (N, 2) array of (x, y) centroids, or None if fewer than
    ``MIN_HOMOGRAPHY_POINTS`` fiducials are found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dark blobs: invert so markers become bright peaks for thresholding.
    # Otsu adapts to the session's lighting; we bias toward the darkest tail.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[float, float]]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:  # reject specks
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        # Circularity == 1.0 for a perfect circle; fiducials are circles by
        # design so this is a strong filter against board features.
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.75:
            continue
        (cx, cy), _radius = cv2.minEnclosingCircle(cnt)
        candidates.append((area, (cx, cy)))

    if len(candidates) < MIN_HOMOGRAPHY_POINTS:
        return None

    candidates.sort(key=lambda c: c[0], reverse=True)
    centers = np.array([c[1] for c in candidates[:n_fiducials]], dtype=np.float32)
    return _order_points(centers)


def _order_points(points: np.ndarray) -> np.ndarray:
    """Order points as top-left, top-right, bottom-right, bottom-left.

    Both the capture and the golden image are ordered the same way, so the
    point correspondence needed for the homography is consistent even though
    the camera and board may be rotated relative to each other.
    """
    pts = np.asarray(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]      # top-left
    ordered[2] = pts[np.argmax(s)]      # bottom-right
    ordered[1] = pts[np.argmin(diff)]   # top-right
    ordered[3] = pts[np.argmax(diff)]   # bottom-left
    return ordered


# ---------------------------------------------------------------------------
# Fallback 1: board-corner contour
# ---------------------------------------------------------------------------

def detect_board_corners(image: np.ndarray) -> np.ndarray | None:
    """Find the board outline as the largest 4-sided contour.

    Requires the board to contrast with the jig background. Returns ordered
    corner points (TL, TR, BR, BL) or None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.05 * image.shape[0] * image.shape[1]:
        return None  # too small to plausibly be the board

    approx = cv2.approxPolyDP(largest, 0.02 * cv2.arcLength(largest, True), True)
    if len(approx) != 4:
        return None
    return _order_points(approx.reshape(4, 2))


# ---------------------------------------------------------------------------
# Fallback 2: ORB feature matching against the golden image
# ---------------------------------------------------------------------------

def homography_from_orb(image: np.ndarray, golden: np.ndarray) -> np.ndarray | None:
    """Estimate the homography by matching ORB features with RANSAC.

    PCBs are feature-rich (silkscreen, pads, traces), so this is a decent
    last resort, but it is less accurate than fiducials — use it only when
    marker detection fails, and check the log for the inlier ratio.
    """
    orb = cv2.ORB_create(nfeatures=5000)
    kp_img, des_img = orb.detectAndCompute(image, None)
    kp_gold, des_gold = orb.detectAndCompute(golden, None)
    if des_img is None or des_gold is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(des_img, des_gold, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]  # Lowe ratio
    if len(good) < MIN_HOMOGRAPHY_POINTS:
        return None

    src = np.float32([kp_img[m.queryIdx].pt for m in good])
    dst = np.float32([kp_gold[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=5.0)
    if H is None:
        return None
    inlier_ratio = float(mask.sum()) / len(mask) if mask is not None else 0.0
    LOGGER.info("ORB homography inlier ratio: %.2f (%d matches)", inlier_ratio, len(good))
    if inlier_ratio < 0.3:
        LOGGER.warning("ORB homography has a low inlier ratio — alignment may be off.")
    return H


# ---------------------------------------------------------------------------
# Main alignment entry points
# ---------------------------------------------------------------------------

def align_image(image: np.ndarray, golden: np.ndarray) -> AlignmentResult:
    """Warp ``image`` into the golden reference's canonical view."""
    golden_h, golden_w = golden.shape[:2]

    # Strategy 1: fiducials in both images -> direct point-pair homography.
    src_pts = detect_fiducials(image)
    if src_pts is not None:
        dst_pts = detect_fiducials(golden)
        if dst_pts is not None and len(src_pts) == len(dst_pts):
            H, _ = cv2.findHomography(src_pts, dst_pts)
            if H is not None:
                warped = cv2.warpPerspective(image, H, (golden_w, golden_h))
                return AlignmentResult(warped=warped, method="fiducials", homography=H)
        LOGGER.warning("Fiducials found in input but not matched in golden; falling back.")

    # Strategy 2: board corners -> rectangle in golden space.
    corners = detect_board_corners(image)
    if corners is not None:
        golden_corners = detect_board_corners(golden)
        if golden_corners is not None:
            H, _ = cv2.findHomography(corners, golden_corners)
            if H is not None:
                warped = cv2.warpPerspective(image, H, (golden_w, golden_h))
                return AlignmentResult(warped=warped, method="corners", homography=H)
        LOGGER.warning("Board corners found in input but not in golden; falling back.")

    # Strategy 3: ORB feature matching.
    H = homography_from_orb(image, golden)
    if H is not None:
        warped = cv2.warpPerspective(image, H, (golden_w, golden_h))
        return AlignmentResult(warped=warped, method="orb", homography=H)

    return AlignmentResult(
        warped=None,
        method="failed",
        messages=["All alignment strategies failed; image skipped."],
    )


def align_path(input_path: Path, golden_path: Path, output_dir: Path) -> list[Path]:
    """Align one image or every image in a directory; write results to ``output_dir``.

    Returns the list of written file paths. Failures are logged, not raised,
    so one bad capture does not abort a batch.
    """
    golden = cv2.imread(str(golden_path))
    if golden is None:
        raise FileNotFoundError(f"Could not read golden image: {golden_path}")

    if input_path.is_dir():
        inputs = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        inputs = [input_path]
    if not inputs:
        LOGGER.warning("No images found at %s", input_path)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for img_path in inputs:
        image = cv2.imread(str(img_path))
        if image is None:
            LOGGER.warning("Unreadable image skipped: %s", img_path)
            continue
        result = align_image(image, golden)
        if result.warped is None:
            LOGGER.error("Alignment FAILED for %s", img_path)
            continue
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), result.warped)
        LOGGER.info("Aligned %s via %s -> %s", img_path.name, result.method, out_path)
        written.append(out_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align board captures to the golden reference view."
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Input image file or directory of images.")
    parser.add_argument("--golden", required=True, type=Path,
                        help="Golden reference image (defines the canonical view/size).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory for aligned images.")
    parser.add_argument("--verbose", action="store_true", help="Debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    written = align_path(args.input, args.golden, args.output)
    LOGGER.info("Done: %d image(s) aligned.", len(written))


if __name__ == "__main__":
    main()

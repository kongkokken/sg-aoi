#!/usr/bin/env python
"""
capture_webcam.py — multi-shot webcam capture tool for PCBA-AOI debug/development.

PURPOSE
    Use the laptop's built-in camera to grab one or many frames into
    ``data/raw/captures/`` so the AOI pipeline (align -> detect -> rules) can be
    exercised and debugged with real images before the production camera rig
    exists.

    NOTE: laptop-webcam images are for DEBUG/DEVELOPMENT ONLY. They have
    auto exposure/focus/white-balance and no fixed geometry, so they are NOT
    suitable as training data. The production rig requirements (fixed mount,
    fixed focus/exposure/WB, jig, fiducials) are in
    ``docs/data_collection_guide.md``.

USAGE (from the project root)
    python scripts/capture_webcam.py --shots 5 --interval 2
    python scripts/capture_webcam.py --shots 1                       # single shot
    python scripts/capture_webcam.py --shots 0 --interval 5          # timelapse until 'q'
    python scripts/capture_webcam.py --shots 3 --no-preview          # headless
    python scripts/capture_webcam.py --camera 1 --width 1920 --height 1080

OUTPUT
    Each run creates a timestamped session folder:

        data/raw/captures/YYYY-MM-DD_HHMMSS/
            shot_001.jpg
            shot_002.jpg
            ...
            session_log.json

    ``session_log.json`` records start/end time, camera index, requested vs
    actual resolution, interval, and the list of saved files — it is the
    manifest later debug/dataset tooling should read.

KEYS (preview window)
    SPACE   capture a frame immediately
    q/ESC   quit the session early (already-saved shots are kept)

EXIT CODES
    0  session finished (at least one shot saved, or clean early quit)
    1  camera problem (not found / busy / permission denied)
    2  output problem (cannot create or write the output directory)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Silence OpenCV's own backend probing chatter (WARN/ERROR lines from failed
# backend attempts); our friendly error messages carry the real information.
os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")

try:
    import cv2
except ImportError:
    print(
        "ERROR: OpenCV is not installed in this Python environment.\n"
        "Install it with:  pip install opencv-python numpy\n"
        "(Use the project venv, e.g. .venv-demo — see docs/webcam_capture.md.)"
    )
    sys.exit(1)

# Project root is the parent of this script's folder, so the tool works no
# matter which directory it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw" / "captures"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture one or more webcam shots for AOI pipeline debug/development.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--shots", type=int, default=1, metavar="N",
                   help="number of shots to capture; 0 = unlimited until you quit")
    p.add_argument("--interval", type=float, default=1.0, metavar="SECONDS",
                   help="seconds between automatic captures (burst/timelapse)")
    p.add_argument("--countdown", type=float, default=0.0, metavar="SECONDS",
                   help="countdown delay before the first shot")
    p.add_argument("--camera", type=int, default=0, metavar="INDEX",
                   help="camera index (0 = default/built-in)")
    p.add_argument("--width", type=int, default=0,
                   help="requested capture width in pixels (0 = camera default)")
    p.add_argument("--height", type=int, default=0,
                   help="requested capture height in pixels (0 = camera default)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, metavar="DIR",
                   help="base output directory; a timestamped session folder is created inside")
    p.add_argument("--prefix", default="shot",
                   help="filename prefix, e.g. 'board_a' -> board_a_001.jpg")
    p.add_argument("--no-preview", action="store_true",
                   help="headless mode: no preview window (for remote/SSH runs)")
    args = p.parse_args(argv)
    if args.shots < 0:
        p.error("--shots must be >= 0 (0 means unlimited)")
    if args.interval <= 0:
        p.error("--interval must be > 0")
    if args.countdown < 0:
        p.error("--countdown must be >= 0")
    return args


def friendly_camera_error(index: int, exc: Exception | None = None) -> str:
    """Plain-English camera failure message for Windows laptops."""
    lines = [
        f"ERROR: could not open camera index {index}.",
        "",
        "Things to check:",
        "  1. Another app may be using the camera — close Teams, Zoom,",
        "     the Windows Camera app, browser tabs with video, then retry.",
        "  2. Windows may be blocking camera access — check",
        "     Settings -> Privacy & security -> Camera and make sure camera",
        "     access and 'Let desktop apps access your camera' are ON.",
        "  3. If the laptop has more than one camera, try --camera 1.",
        "  4. Check the physical camera shutter/privacy switch or Fn key",
        "     (many laptops have one) and Device Manager for a disabled device.",
    ]
    if exc is not None:
        lines += ["", f"(technical detail: {exc})"]
    return "\n".join(lines)


def open_camera(index: int) -> cv2.VideoCapture:
    """Open the camera, preferring DirectShow on Windows (faster, clearer errors)."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
    last_exc: Exception | None = None
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        try:
            if cap.isOpened():
                return cap
        except cv2.error as e:  # e.g. permission errors raised by some backends
            last_exc = e
        cap.release()
    print(friendly_camera_error(index, last_exc))
    sys.exit(1)


def make_session_dir(base: Path) -> Path:
    """Create <base>/YYYY-MM-DD_HHMMSS/ (suffix _2, _3... on the rare collision)."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    session = base / stamp
    n = 2
    while session.exists():
        session = base / f"{stamp}_{n}"
        n += 1
    try:
        session.mkdir(parents=True, exist_ok=False)
    except PermissionError:
        print(
            f"ERROR: cannot create output folder:\n  {session}\n"
            "Windows says permission denied. Check that the folder is not\n"
            "read-only / synced-locked, or pass a different --out DIR."
        )
        sys.exit(2)
    except OSError as e:
        print(
            f"ERROR: cannot create output folder:\n  {session}\n"
            f"({e}). Check the path and free disk space, or pass --out DIR."
        )
        sys.exit(2)
    return session


def draw_hud(frame, lines) -> None:
    """Overlay a semi-transparent HUD banner with the given text lines."""
    h, w = frame.shape[:2]
    bar_h = 22 + 24 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    y = 22
    for text, color in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 24


def main(argv=None) -> int:
    args = parse_args(argv)

    cap = open_camera(args.camera)

    # Apply requested resolution, then read back what the camera ACTUALLY gave —
    # webcams silently fall back to the nearest supported mode.
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    requested = f"{args.width}x{args.height}" if args.width and args.height else "camera default"

    # Warm-up: webcams need a few frames for auto-exposure to settle.
    for _ in range(5):
        cap.read()

    session_dir = make_session_dir(args.out)
    rel_session = session_dir.relative_to(PROJECT_ROOT) if session_dir.is_relative_to(PROJECT_ROOT) else session_dir

    print(f"Camera {args.camera} opened: actual resolution {actual_w}x{actual_h} "
          f"(requested: {requested})")
    print(f"Session folder: {session_dir}")
    if args.shots == 0:
        print(f"Unlimited mode: capturing every {args.interval}s until you quit "
              f"({'q/ESC' if not args.no_preview else 'Ctrl+C'}).")
    else:
        print(f"Capturing {args.shots} shot(s), every {args.interval}s.")

    log = {
        "session_dir": str(session_dir),
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "end_time": None,
        "camera_index": args.camera,
        "requested_resolution": {"width": args.width or None, "height": args.height or None},
        "actual_resolution": {"width": actual_w, "height": actual_h},
        "interval_seconds": args.interval,
        "countdown_seconds": args.countdown,
        "shots_requested": args.shots,
        "files": [],
    }

    window = "AOI webcam capture (SPACE=shot, q/ESC=quit)"
    if not args.no_preview:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    shots_taken = 0
    quit_early = False
    start = time.monotonic()
    next_shot_at = start + args.countdown  # countdown delays the FIRST shot only
    unlimited = args.shots == 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("\nERROR: camera stopped returning frames (unplugged? taken over "
                      "by another app?). Saving what we have so far.")
                break

            now = time.monotonic()
            remaining_cd = max(0.0, next_shot_at - now) if shots_taken == 0 else 0.0

            key = -1
            if not args.no_preview:
                hud = [
                    (f"Shots: {shots_taken}"
                     + ("" if unlimited else f" / {args.shots}")
                     + f"   interval {args.interval}s", (0, 255, 0)),
                    (f"Session: {rel_session}", (200, 200, 200)),
                ]
                if remaining_cd > 0:
                    hud.append((f"First shot in {remaining_cd:.1f}s  (SPACE to shoot now)",
                                (0, 200, 255)))
                else:
                    next_in = max(0.0, next_shot_at - now)
                    hud.append((f"Next auto-shot in {next_in:.1f}s   SPACE=shoot now, q=quit",
                                (0, 200, 255)))
                draw_hud(frame, hud)
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF

            take = False
            if key == 32:                      # SPACE — capture immediately
                take = True
                next_shot_at = now + args.interval
            elif key in (27, ord("q")):        # ESC / q — quit early
                quit_early = True
                break
            elif now >= next_shot_at:          # scheduled auto-capture
                take = True
                next_shot_at = now + args.interval

            if take:
                shots_taken += 1
                path = session_dir / f"{args.prefix}_{shots_taken:03d}.jpg"
                try:
                    if not cv2.imwrite(str(path), frame):
                        raise IOError("cv2.imwrite returned False")
                except (cv2.error, IOError, OSError) as e:
                    print(f"\nERROR: failed to write {path} ({e}). "
                          "Check free disk space and folder permissions. Stopping.")
                    shots_taken -= 1
                    break
                log["files"].append(path.name)
                print(f"  saved {path.name} ({shots_taken}"
                      + ("" if unlimited else f"/{args.shots}") + ")")

            if not unlimited and shots_taken >= args.shots:
                break

            if args.no_preview:
                # Headless: poll at ~50 Hz so Ctrl+C stays responsive.
                time.sleep(0.02)
    except KeyboardInterrupt:
        quit_early = True
        print("\nInterrupted by user (Ctrl+C).")
    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
        log["end_time"] = datetime.now().isoformat(timespec="seconds")
        log["elapsed_seconds"] = round(time.monotonic() - start, 2)
        log["quit_early"] = quit_early
        log_path = session_dir / "session_log.json"
        try:
            log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        except OSError as e:
            print(f"WARNING: could not write session log {log_path} ({e}).")

    print(f"Done: {shots_taken} shot(s) saved in {session_dir}"
          + (" (quit early)" if quit_early else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

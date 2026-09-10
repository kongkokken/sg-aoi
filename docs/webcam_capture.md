# Webcam capture for debug/development (`scripts/capture_webcam.py`)

Use the laptop's built-in camera to grab real frames into `data/raw/captures/`
so the AOI pipeline can be exercised **before the production camera rig exists**.

> ⚠️ **Debug/development only.** Laptop-webcam shots use auto exposure, auto
> focus, and auto white balance, with no fixed mount or jig — exactly what the
> production rig must NOT have (see `docs/data_collection_guide.md`: fixed
> mount, fixed focus/exposure/WB, jig, fiducials). Use these captures to debug
> and develop the pipeline and to eyeball early image quality. Do **not** mix
> them into the labeled training dataset.

## Install (once)

Use the project demo venv (do not install into the managed/system Python):

```bash
cd pcba-aoi
python -m venv .venv-demo           # skip if it already exists
.venv-demo/Scripts/python -m pip install opencv-python numpy
```

(The `aoi-app` environment works the same way: `pip install opencv-python`
into whichever env you run the scripts with.)

## Typical commands

Run from the project root (`pcba-aoi/`):

```bash
# Single shot with live preview (SPACE also shoots, q/ESC quits)
.venv-demo/Scripts/python scripts/capture_webcam.py --shots 1

# 10-shot burst, one frame per second
.venv-demo/Scripts/python scripts/capture_webcam.py --shots 10 --interval 1

# Timelapse: one frame every 5 s until you press q
.venv-demo/Scripts/python scripts/capture_webcam.py --shots 0 --interval 5

# Headless (no preview window) — useful over SSH / from CI
.venv-demo/Scripts/python scripts/capture_webcam.py --shots 3 --interval 1 --no-preview

# 5-second countdown before the first shot (time to step away from the keyboard)
.venv-demo/Scripts/python scripts/capture_webcam.py --shots 5 --interval 2 --countdown 5

# Request a specific resolution (the tool reports what the camera ACTUALLY gave)
.venv-demo/Scripts/python scripts/capture_webcam.py --shots 5 --width 1920 --height 1080
```

Each run creates a timestamped session folder:

```
data/raw/captures/2026-09-10_153012/
    shot_001.jpg
    shot_002.jpg
    ...
    session_log.json     # start/end time, camera index, requested vs actual
                         # resolution, interval, list of saved files
```

`session_log.json` is the manifest for later debug/dataset tooling — read it
rather than globbing the folder.

## Using the captures for pipeline debugging

Point the laptop camera at a board (or a printed board photo) on the desk,
capture a few shots, then run the inference pipeline on one:

```bash
python src/infer_pipeline.py \
    --image data/raw/captures/2026-09-10_153012/shot_001.jpg \
    --config configs/pipeline.yaml
```

Webcam frames usually lack the fiducial markers, so `src/align_board.py` will
fall back to board corners / ORB matching — expect alignment quality to vary;
that is fine for pipeline smoke-testing.

## Windows camera-permission gotchas

If the tool prints its "could not open camera" message:

1. **Camera busy** — close Teams, Zoom, the Windows Camera app, and browser
   tabs using video, then retry. Only one app can hold the webcam.
2. **Privacy setting** — Settings → Privacy & security → Camera: enable
   "Camera access" **and** "Let desktop apps access your camera".
3. **Physical shutter / Fn key** — many laptops have a privacy shutter or an
   Fn+F-key camera toggle; also check Device Manager for a disabled device.
4. **Wrong index** — try `--camera 1` if the machine has multiple cameras.

## Tips for less-terrible debug shots

- Desk lamp on, no daylight from the side; keep lighting identical between
  shots (mimics the "one lighting recipe, always" rule).
- Prop the laptop so the camera looks down at the board as steadily as
  possible; don't hand-hold it.
- Use `--countdown` so pressing keys doesn't shake the frame.

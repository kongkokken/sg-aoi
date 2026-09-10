# Dataset layout

All image data lives under this directory. Keep raw captures untouched —
every processed artifact must be regenerable from `raw/`.

**Phase 1 (detection only) needs just three things here:**
`detection_dataset/` (labeled COCO component dataset), `golden/` (golden
board + `expected_components.json`), and optionally `synthetic_ng/` for
rule-engine validation. `boards_ok/` / `boards_ng/` belong to the **deferred
anomaly phase** and may stay empty.

```
data/
  raw/                    # untouched captures from the camera rig
  boards_ok/              # [DEFERRED — anomaly phase] ALIGNED OK images
  boards_ng/              # [DEFERRED — anomaly phase] ALIGNED NG images
    scratches/
    solder_balls/
    foreign_particles/
    wrong_part/           # (optional, for evaluating the fused verdict)
    wrong_orientation/
    missing_part/
  synthetic_ng/           # output of scripts/make_synthetic_defects.py
  detection_dataset/      # [PHASE 1 CRITICAL] component-detector dataset, COCO format
    images/
      train/
      val/
    annotations/
      instances_train.json
      instances_val.json
  golden/
    golden_board.jpg      # [PHASE 1 CRITICAL] reference board, canonical aligned view
    expected_components.json
```

## Folder purposes

| Folder | Used by | Notes |
|--------|---------|-------|
| `raw/` | — | Straight from the camera. Never edit; alignment reads from here. |
| `boards_ok/` | Branch B (anomaly training) — **deferred** | Output of `src/align_board.py`. 200–500 images of known-good boards. Not needed for phase 1. |
| `boards_ng/` | Branch B (validation) — **deferred** | Aligned NG images, grouped by defect type. **Never** used for training PatchCore. |
| `synthetic_ng/` | rule-engine validation | Generated from OK/golden images; has a JSON manifest of injected missing/wrong-part defects. |
| `detection_dataset/` | **Branch A (PP-YOLOE+ fine-tune) — phase 1** | COCO format. Label with X-AnyLabeling or CVAT, export COCO, split train/val. |
| `golden/` | **rule engine — phase 1** | One canonical reference board + the expected component list. |

## expected_components.json format

One entry per component that must be present on the assembled board.
Coordinates are in the **canonical aligned view** (i.e. pixel coordinates in
`golden_board.jpg`), so they can be compared directly against detections on
aligned images.

```json
{
  "board_name": "demo_board_revA",
  "reference_image": "golden_board.jpg",
  "components": [
    {
      "id": "R12",
      "class": "resistor_0603",
      "bbox": [x_min, y_min, width, height],
      "expected_angle_deg": 0
    },
    {
      "id": "U3",
      "class": "ic_sop8",
      "bbox": [x_min, y_min, width, height],
      "expected_angle_deg": 90
    }
  ]
}
```

- `id`: silkscreen reference designator — what a human operator would look for.
- `class`: must match the class names used in the detection dataset's COCO
  categories exactly.
- `bbox`: COCO-style `[x, y, width, height]` in the golden image's pixels.
- `expected_angle_deg`: expected orientation in degrees, multiples of 90 for
  rectangular passives; arbitrary for keyed/polarized parts. The rule engine
  compares a rotation estimate against this value (see `src/infer_pipeline.py`).

Build this file once by hand (or with the help of the detector's output on the
golden board + manual cleanup). It is the single source of truth for
wrong/missing/misoriented-part checks.

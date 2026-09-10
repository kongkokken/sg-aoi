# Data collection guide (one page) — phase 1: component detection

Phase 1 trains a **component detector** (PP-YOLOE+) for missing/wrong-part
checks on the component side of a PTH board. The critical artifact is a
**labeled component dataset**, not image volume — 60 well-labeled boards beat
500 unlabeled ones.

## Camera rig

- **Fixed mount**: camera rigidly above the board, top-down on the
  **component side**. A copy stand, tripod arm, or a crossbar on the
  existing lean-tube rack all work (see `docs/station/` for a real-bench
  retrofit) — it must not move between sessions.
- **Fixed board position**: a jig (corner stops, rails, or a recess) so every
  board lands within a few millimeters of the same spot.
- **Manual everything**: fixed focus, fixed exposure, fixed white balance.
  Auto-anything changes appearance between sessions and the detector sees
  that as noise it must learn around.
- Resolution: enough that your smallest component covers ~15+ pixels. For
  axial resistors and small ceramics on a typical PTH board that usually
  means ≥ 12 MP at a 40–60 cm working distance.
- **Tall parts shadow small ones**: connectors/electrolytics next to small
  resistors need light from two sides (see below), and shadowed instances
  belong *in* the training data.

## Lighting

- Two diffuse LED bars mounted on the rack uprights at ~30–45°, aimed at the
  work surface — this balances shadows from tall PTH components.
- Avoid hotspots on glossy component bodies and the solder mask; matte,
  even light is the goal.
- One lighting recipe, always. Never mix sessions with daylight leaking in.

## Fiducials / alignment reference

- Place 4 dark circular markers (printed stickers work) on the **jig**, near
  the board corners — not on the board, so they are identical for every unit.
- `src/align_board.py` uses them to compute the homography into the canonical
  view. Without fiducials it falls back to board corners, then ORB matching —
  both less reliable.

## The labeled component dataset (phase 1 critical path)

- Volume: **50–100 boards**, captured across multiple days/batches so the
  detector sees supplier-lot and lighting variation.
- Tools: **X-AnyLabeling** (has model-assisted pre-labeling, exports COCO) or
  **CVAT**. Export **COCO** format.
- Split ~85/15 into `images/train` + `images/val` with
  `annotations/instances_train.json` / `instances_val.json` under
  `data/detection_dataset/`.
- Every class should reach ≥ ~100 instances across the dataset; capture more
  boards containing rare classes.

### Class list definition (PTH example)

Define classes once, up front, and never rename them — the rule engine
compares class names as exact strings against `expected_components.json`:

| Class name | Notes |
|------------|-------|
| `resistor_axial` | All axial resistors one class; value-level differences belong in expected_components, not the class |
| `capacitor_ceramic` | Small ceramic discs |
| `capacitor_electrolytic` | Can-type; **polarity/orientation checking is deferred future work** — phase 1 only verifies presence and identity |
| `diode` | Axial diodes; polarity bands are future orientation work |
| `ic_dip` | DIP packages |
| `connector_pinheader` | Pin headers / connectors |
| `transistor_to92` | Adjust to your actual BOM |

Rules of thumb: one class per visually distinguishable component *type*; if
two classes are visually indistinguishable at your capture resolution, merge
them or raise resolution — otherwise they will confuse the detector forever.

## expected_components.json

Build it from the golden board (format in `data/README.md`): one entry per
component with reference designator (`R12`), class (must match the dataset
class names), bbox in golden-image pixels, and expected angle (kept for the
deferred orientation check). This file is what makes a detection "wrong" or
"missing" — verify it once by drawing its bboxes over the golden image.

## Augmentation do's and don'ts

| Augmentation | Verdict | Why |
|--------------|---------|-----|
| Horizontal / vertical flip | ❌ **Never** | A mirrored board is a *wrong* board; orientation is semantics, not noise. |
| Free rotation (90/180/270) | ❌ **Never** | Same reason. |
| Small translation (± few px) | ⚠️ Only if alignment is already applied | Position is canonical after alignment. |
| Small brightness / contrast jitter (±5–10%) | ✅ | Models session-to-session lighting drift. |
| Gaussian noise (small) | ✅ | Models sensor noise. |
| Blur (small) | ⚠️ Sparingly | Overuse hides small-part features. |
| Mosaic / MixUp (detector) | ⚠️ | PP-YOLOE+'s default recipe uses Mosaic; safe for component appearance, disable if the detector invents parts at tile boundaries. |

When in doubt: collect and label more real boards instead of augmenting. For
NG cases, prefer `scripts/make_synthetic_defects.py`, which injects missing/
wrong parts with known ground truth you can score the rule engine against.

## Deferred: anomaly-phase data (not needed for phase 1)

If the anomaly branch (scratches, solder balls, foreign particles) is
activated later, it will need **200–500 aligned OK board images** in
`data/boards_ok/` (PatchCore trains on good boards only, across many sessions
and boards) plus a small NG validation set in `data/boards_ng/`. Skip this
for phase 1 — the folders can stay empty.

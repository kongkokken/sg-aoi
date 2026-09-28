# PCBA AOI with AI (scaled-down starter)

A hobbyist-scale Automated Optical Inspection (AOI) pipeline for classifying
assembled PCBs as **OK / NG**.

**Phase 1 scope (current): detect MISSING PARTS and WRONG PARTS on the
component side of a through-hole (PTH) PCBA** — nothing else. The pipeline is
built around one branch:

- **Branch A — Object detection** (PaddleDetection **PP-YOLOE+**, Apache 2.0)
  finds every component (resistor, capacitor, diode, IC, connector, ...) with
  its class and location. A **rule engine** compares detections against the
  expected component list (`data/golden/expected_components.json`) to catch
  *missing* and *wrong* parts. **This is the primary and only required
  branch for phase 1.**
- **Branch B — Anomaly detection** (Intel **anomalib**, PatchCore, Apache 2.0)
  is **DEFERRED — not needed for phase 1**. The code (`src/train_anomaly.py`)
  and docs are kept for a later phase targeting appearance defects
  (scratches, solder balls/debris, foreign particles). Do not set it up now.
- **Decision engine** fuses the active branch(es) into one OK/NG verdict with
  an annotated output image.

The two branches use **separate Python environments** (PaddlePaddle and
anomalib/PyTorch have conflicting dependency chains) — another reason the
deferred branch stays out of the phase-1 install.

## Architecture

```
                       +---------------------------+
  camera capture ----> |  Board alignment          |
  (top-down, fixed jig)|  (fiducials -> homography)|
                       +-------------+-------------+
                                     |  aligned image
              +----------------------+----------------------+
              |                                             |
   +----------v----------+                       +----------v----------+
   | Branch A  [PHASE 1] |                       | Branch B [DEFERRED] |
   | PP-YOLOE+ detection |                       | PatchCore anomaly   |
   | -> components       |                       | -> heatmap + score  |
   +----------+----------+                       +----------+----------+
              | detections (class, bbox)          | anomaly map
              v                                   v
   +---------------------+             +-----------------------+
   | Rule engine         |             | Heatmap thresholding  |
   | vs expected list:   |             | -> appearance regions |
   | MISSING / WRONG part|             |    [DEFERRED]         |
   | (orientation: off)  |             +-----------+-----------+
   +----------+----------+                         |
              | defect records                     | defect records
              v                                    v
                    +-------------------------------+
                    |  Decision engine (fuse)        |
                    |  -> verdict OK / NG            |
                    |  -> verdict JSON + annotated   |
                    |     board image                |
                    +-------------------------------+
```

## Defect types -> branch mapping (phase 1 scope marked)

| # | Defect type                    | Phase 1?          | Branch | How it is caught                                  |
|---|--------------------------------|-------------------|--------|---------------------------------------------------|
| 1 | Wrong parts                    | ✅ **IN SCOPE**   | A      | Detected class != expected class at that location |
| 6 | Missing parts                  | ✅ **IN SCOPE**   | A      | Expected component with no matching detection     |
| 2 | Wrong orientation              | ⏸ deferred        | A      | Bbox aspect / rotation heuristic vs expected angle (disabled in `configs/pipeline.yaml`; too coarse for PTH phase 1) |
| 3 | PCBA scratches                 | ⏸ deferred        | B      | Anomaly heatmap region (needs Branch B)           |
| 4 | Solder balls / solder debris   | ⏸ deferred        | B      | Anomaly heatmap region (needs Branch B)           |
| 5 | Foreign particles              | ⏸ deferred        | B      | Anomaly heatmap region (needs Branch B)           |

Deferred checks are switched off via the `checks:` flags in
`configs/pipeline.yaml`; enable them when the corresponding branch/phase is
activated.

## Repository layout

```
pcba-aoi/
  README.md
  configs/
    pipeline.yaml              # decision-engine config (thresholds, model paths)
    detection/
      ppyoloe_plus_custom.yml  # PP-YOLOE+ fine-tuning config
      README.md                # how to plug it into a cloned PaddleDetection repo
  data/                        # datasets (gitignore the images, keep the README)
    README.md                  # dataset layout + expected_components.json format
  docs/
    execution_plan.md          # gated go-live plan (detection-first)
    flow_diagram.png           # rendered flow diagram
    ui_design.md               # operator UI design spec
    data_collection_guide.md   # camera rig / lighting / labeling guide
    station/                   # PTH station retrofit sketch + guide
  requirements/                # separate env per branch (see below)
  scripts/
    make_synthetic_defects.py  # synthesize NG images from OK images
    render_flow_diagram.py     # regenerate docs/flow_diagram.png
  src/
    align_board.py             # fiducial/homography board alignment
    train_anomaly.py           # anomalib PatchCore training + export [DEFERRED]
    infer_pipeline.py          # combined decision engine
    app.py                     # Streamlit operator UI
```

## Setup (phase 1 needs only TWO environments)

Phase 1 (detection only) requires the **`aoi-detect`** env (training/export)
and the **`aoi-app`** env (inference + UI). The `aoi-anomaly` env is needed
**only when the deferred anomaly branch is activated** — skip it for now.
Everything below works on Windows with Git Bash; conda is recommended.

```bash
# Branch A: detection (train/export PP-YOLOE+) — PHASE 1
conda create -n aoi-detect python=3.10 -y
conda activate aoi-detect
pip install -r requirements/requirements-detection.txt

# App / decision engine (inference + UI) — PHASE 1
# If you run detection inference via paddle.inference inside this env,
# install paddlepaddle here too (CPU build is fine for the shop-floor PC).
conda create -n aoi-app python=3.10 -y
conda activate aoi-app
pip install -r requirements/requirements-app.txt

# Branch B: anomaly — DEFERRED, only when the anomaly branch is activated
# conda create -n aoi-anomaly python=3.10 -y
# conda activate aoi-anomaly
# pip install -r requirements/requirements-anomaly.txt
```

Also clone PaddleDetection somewhere outside this repo (Branch A env):

```bash
git clone https://github.com/PaddlePaddle/PaddleDetection.git
# see configs/detection/README.md for how to register the custom dataset
```

## Data collection checklist (phase 1)

1. **Fixed camera jig**: rigid mount, top-down on the **component side**,
   fixed focus and exposure (manual exposure — auto anything adds noise the
   detector did not train on).
2. **Diffuse lighting**: LED bars at an angle, no specular hotspots; keep it
   constant across capture sessions. See `docs/station/` for the retrofit
   layout of a real PTH bench.
3. **Fiducials / reference markers** on the jig (not on the board) or rely
   on board corners — `src/align_board.py` needs something stable to
   compute the homography from.
4. **50–100 labeled boards** for the detector (THE critical phase-1
   artifact): annotate every component with its class, export COCO
   (see `docs/data_collection_guide.md`).
5. Capture a **golden reference** board image + `expected_components.json`
   (format in `data/README.md`) — the rule engine's source of truth.
6. **Do NOT use horizontal/vertical flips or free rotation augmentation** —
   PCBA orientation is meaningful; a "flipped" board changes what is
   correct. Small brightness/translation jitter is OK.
7. *(Deferred — anomaly phase only)* 200–500 aligned OK board images for
   PatchCore. Not needed for phase 1.

## Training

### Branch A — PP-YOLOE+ (detection) — PHASE 1

```bash
# in the aoi-detect env, inside the cloned PaddleDetection repo
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_custom.yml --eval
python tools/eval.py  -c configs/ppyoloe/ppyoloe_plus_custom.yml
python tools/export_model.py -c configs/ppyoloe/ppyoloe_plus_custom.yml \
    -o weights=output/ppyoloe_plus_crn_s_80e_coco/best_model.pdparams \
       output_dir=output_inference/pcba_ppyoloe
```

### Branch B — PatchCore (anomaly) — DEFERRED (future phase)

```bash
# in the aoi-anomaly env — only when the anomaly branch is activated
python src/train_anomaly.py \
    --data-root data \
    --model patchcore \
    --image-size 512 \
    --output-dir models/anomaly
```

## Inference / decision engine

```bash
# in the aoi-app env (detector trained and exported first)
python src/infer_pipeline.py \
    --image data/boards_ok/board_0001.jpg \
    --config configs/pipeline.yaml

# operator UI
streamlit run src/app.py
```

## User interface

The operator UI is a multi-page Streamlit app (`src/app.py`) with Season
Group branding and a sectioned, role-filtered sidebar (Operator / Engineer /
Admin simulation). Designed in detail in
[docs/ui_design.md](docs/ui_design.md) (personas, wireframes, interaction
flows, visual rules, state handling). Pages:

- **RUN · Inspection & Training** — mode toggle: *Inspection* is an
  explicit two-step flow — **"📷 Take snapshot"** (browser webcam, works
  locally and on Streamlit Cloud; upload as fallback) holds the current
  snapshot, then **"🔍 Inspection"** runs the pipeline → verdict banner
  (OK green / NG red, readable from a distance) with a verdict-source
  indicator (trained model / demo / precomputed JSON), annotated image,
  defect table, NG override with logged reason; *Training* is capture-time
  OK/NG labeling from the browser
  webcam or an upload (NG defect type + reference designator, session/variant
  tracking, golden-board capture, dataset export). Phase 1 shows Branch A
  findings (missing/wrong part) only.
- **RUN · Review & Repair** — filterable past verdicts from `results/`,
  annotated image drill-down, false-reject/false-accept marking into
  `feedback.jsonl`.
- **MONITOR · Dashboard** — boards inspected / OK / NG / FPY metric strip,
  daily FPY trend, top-defects Pareto, recent-NG feed with thumbnails,
  station status.
- **MONITOR · SPC** — date-range filter, FPY trend, defect Pareto by type and
  designator, p-chart with center line and UCL/LCL.
- **BUILD · ➕ Create New** — item-onboarding wizard: name the product,
  capture a few verified-good boards (camera or upload), pick the golden
  board, and register the item under `data/items/<item_id>/` with an honest
  empty `expected_components.json` template. Registered items become
  selectable as the sidebar **Active item**, which repoints the golden
  reference in memory (pipeline.yaml untouched); until components are
  annotated, inspections for the item show an amber "setup pending" state.
- **BUILD · Dataset Review** (was "Labeling") — bulk import, relabel, and
  export: mark images OK/NG into `boards_ok`/`boards_ng` with a
  `labels.jsonl` ledger and dataset zip export. Capture-time labeling lives
  in RUN · Inspection & Training.
- **BUILD · Dataset & Training** — image counts per data folder, golden board
  and expected-components status, model file presence, retrain commands.
- **ADMINISTRATION · Audit Trail** — read-only merged timeline of labeling
  actions and inspection feedback.
- **ADMINISTRATION · Settings** — threshold sliders and per-defect-type
  toggles over `configs/pipeline.yaml`, with plain-language explanations;
  deferred checks are marked as such.
- **MAINTENANCE · System Check** — one-click green/red station
  self-diagnosis (config, golden files, models, disk, writable results).
- **Setup Wizard** (last) — live checklist computed from the filesystem,
  mapping 1:1 to the roadmap below.

The app loads and shows guidance even before any model is trained; missing
models degrade the verdict explicitly (amber "PARTIAL"/"NOT READY" states),
never silently.

## Executable plan

For a concrete, gated go-live plan — one wave-soldered PTH PCBA inspected on
its **component side**, from camera/GPU setup to a trained PP-YOLOE+ detector
running on a non-GPU shop-floor PC — see
[docs/execution_plan.md](docs/execution_plan.md) and the rendered flow
diagram [docs/flow_diagram.png](docs/flow_diagram.png) (regenerate with
`python scripts/render_flow_diagram.py`). The plan is **detection-first**:
Branch A is the phase-1 deliverable; the anomaly branch is a deferred,
optional Phase 7.

For retrofitting an actual PTH manual assembly bench (camera crossbar,
lights, jig, mini PC, monitor), see the annotated layout
[docs/station/pth_station_modification.png](docs/station/pth_station_modification.png)
and the step-by-step
[docs/station/station_modification_guide.md](docs/station/station_modification_guide.md).

## Roadmap (phase-1 order)

- [ ] Retrofit camera rig + lighting at the PTH station (`docs/station/`)
- [ ] Capture boards, label components (50–100 boards, COCO), build
      `expected_components.json` from the golden board
- [ ] Fine-tune PP-YOLOE+, validate (missing-part recall ≈ 100%), export
      inference model
- [ ] Deploy to the non-GPU station PC, verify CPU latency, go live with the
      operator app
- [ ] False-reject feedback loop + periodic retraining
- [ ] Synthetic NG augmentation (`scripts/make_synthetic_defects.py`) for
      rule-engine validation
- [ ] *(Deferred)* Branch B anomaly model for scratches / solder debris /
      foreign particles (`src/train_anomaly.py`)
- [ ] *(Deferred)* Orientation check — likely needs a rotate-aware detector
      (oriented bboxes); the aspect-ratio heuristic is too coarse alone

# PCBA AOI — Operator UI Design

Design spec for the operator-facing application. Current implementation target
is the Streamlit app (`src/app.py`); the layout and state rules are written so
the same design can later be ported to a desktop/embedded HMI (touch panel at
the inspection station).

**Phase 1 scope: the UI serves a detection-only system** — it reports
`missing_part` and `wrong_part` findings from Branch A (PP-YOLOE+ + rule
engine). All Branch B (anomaly) UI elements are shown but marked **deferred**
until that branch is activated.

Guiding principle: **the inspection screen is the product.** Everything else
exists to keep that screen fast, honest, and unmissable.

---

## 1. Personas & context

| Persona | What they do all day | What the UI must give them |
|---------|----------------------|----------------------------|
| **Line operator** | Feeds boards, runs inspections back-to-back | Instant OK/NG readable from 2 m away; the NG defect table names the reference designator (e.g. `R12 — missing`) so rework starts immediately; one-click re-inspect; zero configuration surface |
| **Process engineer** | Reviews false rejects, tunes thresholds, watches for systematic wrong-part runs (bin mixing) | Filterable inspection history with annotated images; false-reject/false-accept marking; threshold sliders with plain-language meaning |
| **Admin** | Owns models, datasets, retraining | Dataset counts, model version/file status, setup wizard progress, retrain commands |

Context: shop floor, possibly gloves, possibly touchscreen, ambient noise,
lighting varies. The operator should never have to read a paragraph.

---

## 2. Screen inventory & wireframes

Navigation: five pages in a sidebar selector (Streamlit: `st.sidebar.radio`;
future HMI: bottom tab bar). The Inspection page is the default and the only
one an operator needs.

### 2.1 Inspection screen (90% of usage)

```
┌────────────────────────────────────────────────────────────────────┐
│ PCBA AOI                                            [EN ▾]  page 1 │
├────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                                                              │  │
│  │                        ✅  OK                                │  │
│  │            (or ❌ NG — 2 defects found)                      │  │
│  │        64 px+ bold white text on solid green/red             │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────┐ ┌────────────────────────┐  │
│  │                                   │ │ Defects (Branch A)     │  │
│  │   ANNOTATED BOARD IMAGE           │ │ ┌────────────────────┐ │  │
│  │   - component bboxes (green ok)   │ │ │type      conf      │ │  │
│  │   - defect boxes (red)            │ │ │missing…  1.00      │ │  │
│  │   - labels name the designator    │ │ │wrong_pa… 0.92      │ │  │
│  │     (R12, C7, U3 …)               │ │ │  detail: R12 …     │ │  │
│  │                                   │ └────────────────────┘ │  │
│  └───────────────────────────────────┘ └────────────────────────┘  │
│  [ 📷 Capture / Upload ]  [ ▶ Run inspection ]                     │
│  [ 🔁 Re-inspect ]        [ ✔ Confirm & next board ]               │
│  NG only: override reason [select ▾] [ ⚠ Override to OK ]          │
└────────────────────────────────────────────────────────────────────┘
```

Rules:
- Verdict banner is full-width, minimum ~64 px equivalent font, solid
  green/red background. Readable from 2 meters. No fine text on this screen.
- Phase 1: the annotated image shows component boxes and red defect boxes
  (missing/wrong part). The verdict derives **only** from Branch A rule-
  engine findings; deferred checks (orientation, appearance anomalies) are
  disabled in `configs/pipeline.yaml` and produce no UI noise.
- When the deferred anomaly branch is activated later, its findings appear
  as amber boxes over a heatmap overlay — same table, `branch` column = B.
- Capture source: camera snapshot or file upload. Optional alignment step.
- "Confirm & next board" clears state for the next board — one click, no
  modal.

### 2.2 Defect review / history screen

```
┌────────────────────────────────────────────────────────────────────┐
│ Review History                                                     │
├────────────────────────────────────────────────────────────────────┤
│ Filters: verdict [All ▾]  defect type [multiselect]  search [____] │
│ ┌──────────────────────────────────────────────────────────────┐   │
│ │ time       board_id   verdict  defects  types                │   │
│ │ 09:14:02   board_012  NG       2        missing_part, wrong… │   │
│ │ 09:11:47   board_011  OK       0        —                    │   │
│ │ ...                                                          │   │
│ └──────────────────────────────────────────────────────────────┘   │
│ ⚠ A run of identical wrong_part findings = probable bin mixing at  │
│   the PTH station — treat as a process alarm, not an AI problem.   │
│ Selected: board_012                                                │
│ ┌───────────────────────────────┐  [ Mark FALSE REJECT ]           │
│ │ annotated image               │  [ Mark FALSE ACCEPT  ]          │
│ │ (component boxes + defects)   │  → appends to results/feedback.  │
│ └───────────────────────────────┘    jsonl (feeds model tuning)    │
└────────────────────────────────────────────────────────────────────┘
```

### 2.3 Model & threshold settings screen

```
┌────────────────────────────────────────────────────────────────────┐
│ Settings                                                           │
├────────────────────────────────────────────────────────────────────┤
│ Detection confidence      [─────●──────] 0.50        ← PHASE 1     │
│   ℹ Below this, a detected component is ignored. Lower it if small │
│     parts are missed; raise it if phantom components appear.       │
│ Component match IoU       [────●───────] 0.40        ← PHASE 1     │
│   ℹ Overlap at which a detection counts as the expected part. Keep │
│     lenient for hand-inserted PTH parts.                           │
│ Orientation tolerance     [───●────────] 30°   (deferred — check   │
│   is off in phase 1; aspect-ratio heuristic is coarse)             │
│ Anomaly score threshold   [───────●────] 0.50  (DEFERRED — has no  │
│   effect until the anomaly branch is trained and enabled)          │
│ Min anomaly region area   [ 64 ] px            (DEFERRED — same)   │
│ Checks: [x] missing part [x] wrong part                            │
│         [ ] orientation (deferred)  [ ] scratches (deferred)       │
│         [ ] solder balls (deferred) [ ] foreign particles (deferred)│
│ Model files: detector ✅ (2024-…, model.pdmodel)  anomaly —        │
│              (deferred, not required)                              │
│ [ ] I understand this overwrites configs/pipeline.yaml             │
│ [ 💾 Save settings ]                                               │
└────────────────────────────────────────────────────────────────────┘
```

Phase 1 primary controls are detection confidence + match IoU. The anomaly
controls stay visible (disabled semantics explained inline) so the settings
page does not change shape when Branch B is activated later.

### 2.4 Dataset & training status screen

```
┌────────────────────────────────────────────────────────────────────┐
│ Dataset & Training                                                 │
├────────────────────────────────────────────────────────────────────┤
│ detection_dataset    68 train / 12 val  (target 50–100 boards)     │
│   ← PHASE 1 critical artifact                                      │
│ Golden board: [preview]   expected components: 23 defined          │
│   ← PHASE 1 critical artifact (rule engine source of truth)        │
│ synthetic_ng          50 imgs  (rule-engine validation)            │
│ boards_ok / boards_ng — shown as "deferred: anomaly phase only"    │
│ Models: detector ✅ exported   anomaly — deferred, not required    │
│ Retrain: `python tools/train.py -c configs/ppyoloe/…` (Paddle repo)│
│          `python src/train_anomaly.py …` (deferred)                │
└────────────────────────────────────────────────────────────────────┘
```

### 2.5 First-run / setup wizard

```
┌────────────────────────────────────────────────────────────────────┐
│ Setup Wizard                                  progress: 4 / 8      │
├────────────────────────────────────────────────────────────────────┤
│ ✅ 1. Camera rig + captures in data/raw                            │
│ ✅ 2. Golden board captured (data/golden/golden_board.jpg)         │
│ ✅ 3. expected_components.json defined (23 components)             │
│ ✅ 4. Detection dataset labeled (68 images, COCO annotations)      │
│ ❌ 5. Detector trained & exported (models/detection/pcba_ppyoloe)  │
│        → python tools/train.py / tools/export_model.py …           │
│ ⏸ 6. Anomaly model (PatchCore) — DEFERRED, optional, not required  │
│ ❌ 7. Go live (needs 1–5 only)                                     │
└────────────────────────────────────────────────────────────────────┘
```

Each item is computed live from the filesystem — the wizard is a status
dashboard, not a form. Items map 1:1 to the README roadmap; deferred steps
are shown as ⏸ so a new user understands they are intentionally skipped.

---

## 3. Interaction flows

### 3.1 Happy-path inspection (target < 3 s after capture)

```
capture/upload → (optional) align → run pipeline
  → Branch A detections → rule engine vs expected_components.json
  → verdict banner + annotated image + defect table
  → verdict JSON + annotated image saved to results/  (automatic, no click)
  → operator presses "Confirm & next board" → state cleared
```

### 3.2 NG handling

```
NG banner → defect table names the designator (e.g. "R12: missing_part")
  → operator inspects annotated image
  ├─ real defect      → board diverted to rework; "Confirm & next board"
  └─ operator disagrees → select override reason (select box:
                          "false alarm - lighting", "false alarm - position",
                          "defect not real", "other")
                        → "Override to OK" logs the override to
                          results/feedback.jsonl with board id + reason
```

The override never silently flips the record — it appends a feedback entry so
the process engineer can audit.

### 3.3 False-reject feedback loop

```
NG in history → engineer opens annotated image → judges it a false reject
  → "Mark FALSE REJECT" (or FALSE ACCEPT for missed defects on OK boards)
  → record appended to results/feedback.jsonl
  → periodic review: thresholds adjusted (Settings) and/or the affected
    boards are labeled and added to detection_dataset for retraining
    (a new component lot/color is the usual cause)
```

---

## 4. Visual design rules

- **Color semantics** (never encode meaning by color alone — always pair with
  text/icon):
  - OK green `#22c55e`, NG red `#ef4444`, warning/partial amber `#f59e0b`,
    neutral grays `#374151` / `#9ca3af`. (Anomaly-overlay amber-orange is
    reserved for the deferred Branch B.)
- **Verdict typography**: ≥ 64 px equivalent, bold, white on the solid
  verdict color, full-width banner. Defect table text ≥ 14 px. No fine print
  on the inspection screen.
- **Defect-type conventions**: active phase-1 types are `missing_part` and
  `wrong_part` (drawn red). `wrong_orientation` and the Branch-B
  `appearance_anomaly` types exist in code but are disabled in phase-1
  configs; when activated later they render red (Branch A) and amber
  (Branch B) respectively. Type strings are shown verbatim so history,
  feedback and code stay greppable.
- **Theme**: dark-friendly industrial theme (Streamlit dark theme works;
  verdict colors chosen to keep contrast on dark backgrounds). Minimum text
  contrast WCAG AA (4.5:1).
- Icons supplement, never replace, text (✅/❌/⚠️/⏸ for deferred).

---

## 5. State handling (explicit degradation, never silent)

| State | UI behavior |
|-------|-------------|
| Detection model not trained/exported | Amber info banner on Inspection: "Branch A checks skipped". In phase 1 this means **no verdict is meaningful** — banner reads **NOT READY** in amber, never OK/NG. |
| Anomaly model absent | **Expected phase-1 state.** A single quiet info note ("anomaly branch deferred — not required"), not a warning. It must not escalate to PARTIAL while phase-1 config has all Branch-B checks disabled. |
| Alignment failure | Red error state: "Board not detected / fiducials not found — reseat board, check lighting, retry." No verdict is produced. |
| Detector errors mid-run | Pipeline catches it, logs a warning; banner shows **NOT READY / PARTIAL** (amber) rather than a bare OK. |
| Empty history / no data folders | Placeholder copy ("No inspections yet — run your first board"), never an empty frame or traceback. |

---

## 6. Accessibility & shop-floor realities

- Touch targets ≥ 44 px (Streamlit buttons meet this; the future HMI must
  keep it with gloves on).
- All primary actions are single clicks; no drag gestures, no hover-only
  affordances, no keyboard required on the Inspection screen.
- Audible alert on NG (buzzer/beep) is **designed but not implementable in
  Streamlit** — browsers block autoplay audio. Noted as a TODO in code and a
  requirement for the future HMI.
- Language toggle placeholder (EN/中文) in the sidebar; actual translation
  tables deferred (i18n is a future-HMI task).
- Works on a 1280×800 panel; the inspection layout is two-column with the
  banner on top so it degrades to a single column on narrow screens.

---

## 7. Mapping to implementation

| Screen / element | Streamlit implementation | Backing code |
|------------------|--------------------------|--------------|
| Page navigation | `st.sidebar.radio` (5 pages) | `main()` in `src/app.py` |
| Inspection page | "Inspection" page | `src/infer_pipeline.py::inspect_board()` → `run_detection()`, `run_rule_engine()`, `decide()` (`run_anomaly()` present but inert while Branch B is absent) |
| Alignment toggle | checkbox + golden path input | `src/align_board.py::align_image()` |
| Verdict banner | `st.markdown` HTML block (64 px, solid color) | `Verdict` from `decide()` |
| Annotated image | `st.image` (BGR→RGB) | `annotate()` in `infer_pipeline.py` |
| Defect table | `st.dataframe` | `Defect` dataclass rows |
| Result persistence | automatic after run | `save_outputs()` → `results/*_verdict.json`, `*_annotated.jpg` |
| History table + filters | dataframe + sidebar filters | reads `results/*_verdict.json` |
| False reject/accept buttons | buttons appending JSONL | `results/feedback.jsonl` |
| Settings sliders/toggles | `st.slider` / `st.checkbox` + save | reads/writes `configs/pipeline.yaml` (`checks:` flags drive phase-1 scope) |
| Dataset counts | filesystem scan | `data/` subfolders, `models/` existence |
| Setup wizard | computed checklist | filesystem checks vs. README roadmap steps |

**Known Streamlit limitations (doc-vs-app gaps), deferred to the future HMI:**
audible NG alert (browser autoplay policy), live camera stream (Streamlit
offers snapshot input only), true i18n, and comment preservation when the
Settings page rewrites `pipeline.yaml` (yaml round-trip drops comments — the
canonical commented copy should be kept in version control).

---

## 8. Labeling workflow (OK/NG)

The **Labeling** page (sidebar, between Settings and Dataset & Training) is the
dataset-building entry point — the first step of the train-the-model workflow.

- **Sources**: upload a batch of images (saved to `data/raw/uploads/<session>/`)
  or import a webcam capture session from `data/raw/captures/*/` (see
  `docs/webcam_capture.md`).
- **Workflow**: one large image at a time ("3 of 12"), two big buttons —
  **✅ OK** (green `#22c55e`) and **❌ NG** (red `#ef4444`) — plus **Skip** and
  **Back** for relabeling the previous image. Labeling **copies** the image
  into `data/boards_ok/` / `data/boards_ng/` (collision-safe, session-prefixed
  filenames); raw captures stay untouched.
- **Ledger**: every action appends one JSON line to `data/labels.jsonl`
  (`timestamp`, `action` = label/relabel, `source_path`, `saved_path`,
  `label`). Corrections are appended as new lines — history is never rewritten.
- **Review & fix**: expander lists everything labeled in the current working
  set; a per-row dropdown moves the file between `boards_ok`/`boards_ng` and
  appends a `relabel` record.
- **Progress**: bar + counters (OK/NG/remaining in the set, plus running totals
  in `boards_ok`/`boards_ng` via the same `_count_images` helper the Dataset &
  Training page uses, so both pages always agree).
- **Export**: "Download labeled dataset (.zip)" bundles `boards_ok/` +
  `boards_ng/` + `labels.jsonl`. Required on Streamlit Cloud, whose filesystem
  is ephemeral — a caption on the page says exactly that.
- Labeling is a real feature and works identically in demo mode; the demo
  banner logic is untouched.

---

## 9. Professional UI (Phase A, implemented 2026-09-11)

Season Group branding + industry-standard AOI information architecture, per
`docs/ui_proposal_sg_aoi.md` (approved Phase A scope).

**Theme & chrome**

- `.streamlit/config.toml`: dark theme — primary `#FB6362` (SG coral),
  background `#1D252D` (SG charcoal), secondary background `#343741`,
  text `#FFFFFF`, sans serif. Verdict colors (`#22c55e` / `#ef4444` /
  `#f59e0b`) remain code-driven and are intentionally NOT part of the theme.
- Sidebar header: white logo (`brand/logo.webp`, committed), title "SG-AOI",
  caption "Season Group · PCBA Line 1", hairline divider; page title
  "SG-AOI · Season Group" with the logo as favicon.
- Global CSS injection: Noto Sans (Google Fonts, system fallback), small
  uppercase letter-spaced coral section labels in the sidebar, card styling
  for metric containers.

**Navigation — sectioned, role-filtered (simulated, no authentication)**

| Section | Pages | Operator | Engineer | Admin |
|---|---|---|---|---|
| RUN | Inspection · Review & Repair (renamed from "Review History") | ✅ | ✅ | ✅ |
| MONITOR | Dashboard · SPC | — | ✅ | ✅ |
| BUILD | Labeling · Dataset & Training | — | ✅ | ✅ |
| ADMINISTRATION | Audit Trail · Settings | — | — | ✅ |
| MAINTENANCE | System Check | — | — | ✅ |
| (last) | Setup Wizard | ✅ | ✅ | ✅ |

**New pages**

- **Dashboard** — metric strip (boards inspected / OK / NG / FPY % / last
  inspection), daily FPY trend, top-defects Pareto (type + designator parsed
  from defect detail), recent-NG feed with annotated thumbnails, station
  status card, coral "DEMO — simulated detections" chip in demo mode.
- **SPC** — date-range filter, FPY trend, defect Pareto by type AND by
  designator, and a p-chart (daily NG proportion with p̄ center line and
  UCL/LCL = p̄ ± 3·√(p̄(1−p̄)/n)); a caption notes limits are approximate
  when daily n is small/variable.
- **System Check** — one-click green/red checklist: config parses, golden
  image, expected_components.json (with component count), detection model or
  demo precomputed, anomaly model (deferred = OK), results dir writable,
  ≥ 1 GB free disk, data folders, labels ledger line count; ALL GREEN /
  ISSUES FOUND banner, per-row fix hints.
- **Audit Trail** — read-only merged timeline of `labels.jsonl` and
  `feedback.jsonl`, newest first, filterable by action type; caption notes
  operator identity is not tracked yet (Users & Roles is a later phase).

All analytics are pure functions over `results/*_verdict.json` +
`data/labels.jsonl` (headlessly testable, no schema changes, no new
dependencies); verdict timestamps come from file mtimes, matching the Review
& Repair page convention. Charts use Streamlit-native `st.line_chart` /
`st.bar_chart` (pandas is a Streamlit hard dependency).

---

## 10. Inspection & Training page (training mode, implemented 2026-09-13)

The former **Inspection** page becomes **Inspection & Training** (RUN section,
still first in the sidebar). A horizontal `st.radio` at the page top switches
between two modes:

- **Inspection** — exactly the previous behavior: demo banner + scenario
  selector (demo mode only; it appears ONLY in this mode), upload + camera
  snapshot, run pipeline, verdict banner. Nothing about the inspection flow,
  demo logic, or verdict colors changed.
- **Training** — capture-time OK/NG labeling straight from the browser webcam
  (`st.camera_input`, works on cloud and local) or a single-image upload.

### Training-mode flow

1. Session fields: **Board variant** text input (default `DEMO-REV-A`) and a
   read-only **session id** (timestamp, generated once per training-mode
   entry, held in `session_state`).
2. Capture: `st.camera_input("Capture board")` or the file-uploader fallback.
   The image is written immediately to `data/raw/training/<session_id>/` and
   held as a **pending preview** in `session_state`, so it survives reruns.
3. The pending image is shown large with three actions: **✅ OK** (green
   `#22c55e`), **❌ NG** (red `#ef4444` — same CSS marker-span pattern as the
   Dataset Review page), **↺ Retake** (discards the pending capture).
4. While an image is pending, an NG-detail row offers **Defect type (for NG)**
   (missing part / wrong part / other / unspecified) and an optional
   **Reference designator (e.g. R7)**. These are recorded with NG labels only
   and ignored for OK.
5. OK/NG saves through the SAME labeling helpers as the Dataset Review page:
   collision-safe copy into `data/boards_ok|boards_ng/` (session+variant
   prefixed filename), one append-only line in `data/labels.jsonl`. Pending
   state clears and the camera is ready for the next shot immediately.
6. A progress row shows OK/NG counts **for this session** (derived from the
   ledger, so they survive restarts) plus dataset totals via `_count_images`.

### Ledger extension (backward compatible)

Training-mode label records add optional fields to the existing JSONL schema:
`session`, `variant`, `defect_type`, `refdes` (NG only), and
`origin: "training_mode"`. The ledger stays append-only; old lines simply lack
the new fields and every reader (`_audit_rows`, `load_label_state`, Dataset
Review) parses the extended lines unchanged.

### Golden-board capture

A **"Capture as golden board"** expander carries a prominent warning: saving
overwrites `data/golden/golden_board.jpg`, and a REAL golden board also
requires rebuilding `data/golden/expected_components.json` — the rule engine's
source of truth — otherwise inspections compare the new image against the old
component list. The save button stays disabled until the "I understand"
checkbox is ticked and an image is pending; the success message repeats the
`expected_components.json` reminder.

### Honesty captions

- "OK/NG board labels triage the dataset — detector training still needs
  component-level box annotation (see docs/data_collection_guide.md)."
- "On the cloud demo, labeled images vanish on redeploy — export the zip
  below; run the local app for real data collection."

Export reuses `build_dataset_zip` + `st.download_button` (same as Dataset
Review).

### Labeling → Dataset Review

The **Labeling** page is renamed **Dataset Review** (BUILD section, before
Dataset & Training) with a one-line intro clarifying the split of duties:
bulk import / relabel / export lives here; capture-time labeling lives in
Inspection & Training → Training mode. Its logic is unchanged.

# Concept revision — item-agnostic golden-based learning (2026-09-28)

**Supersedes the PCBA-only framing as the default concept.** The app is
**item-agnostic**: it learns what *any* item looks like from golden pictures
plus the operator's accept/reject feedback, finds the item's unique
pixels/patterns, and **auto-judges new captures** — including auto-REJECTING
images similar to previously rejected ones. PCBA is the first item, not the
only one. The PP-YOLOE+ / rule-engine pipeline stays as an **optional
advanced path** for when reference-designator-level proof is required.

## The loop

```
 golden board + captures          operator feedback
 (verified-good examples)         (accept / reject)
         │                                │
         └──────────►  galleries  ◄───────┘
                  accepted / rejected
                     prototypes
                          │
              new capture ─┼─► judge() ─► ACCEPT / REJECT / REVIEW
                          │        │
                          │        └─ "Was this judgment correct?"
                          │            ✅ Correct / ❌ Wrong  ─► learn()
                          ▼
              every confirm/override and every Training-mode
              OK/NG label grows the galleries
```

## How the engine works (`src/similarity_engine.py`)

- **Galleries per item.** Accepted (OK) and rejected (NG) prototypes. For a
  registered item: `data/items/<id>/golden_board.jpg` + `captures/` are
  accepted by default; everything the user labels is tracked in
  `data/items/<id>/learned.jsonl` (append-only; a learned label overrides
  the folder default). For the default/demo scope: `data/golden/` +
  `data/boards_ok/` (accepted), `data/boards_ng/` (rejected), ledger in
  `data/.sim_cache/`.
- **Embedding (item identity only).** Canonical 256×256 resize, then four
  L2-normalized weighted blocks: HSV color histogram (global color), raw 8×8
  grayscale grid moments (position-sensitive layout), **sorted** grid
  moments + grayscale quantiles (position-*invariant* texture signature),
  and a HOG-style Sobel gradient-orientation grid (structure).
  Deterministic, numpy + cv2 only, ~9 ms/image on CPU — no
  torch/anomalib/paddle, cloud-safe. The embedding answers "is this the
  item?" — measured on the demo board, same-item captures (with or without
  defects) sit at cosine distances of ~3e-5–1e-2, alien images at
  ~0.15–0.20. It is **not** the defect detector: a defect covering ~0.2% of
  the pixels moves it by only ~3e-5, and position-sensitive blocks push
  *different* defects *apart* (two different defects measured 3× farther
  from each other than each is from the golden board), so embedding-only
  scoring false-accepts unseen defects.
- **Cache.** `data/items/<id>/features.npz` (default scope:
  `data/.sim_cache/default.npz`) stores a format `version` plus paths +
  mtimes + labels + vectors + per-entry learned-anomaly values
  (`anomalies`, `anomaly_refs`). Only new/changed files are re-embedded
  (mtime check); stale-version caches rebuild from scratch; anomaly values
  are recomputed gallery-wide whenever the entry set changes.
- **Scoring — two stages.** *Stage 1 (identity gate):* cosine distance to
  the nearest accepted prototype must be ≤ `IDENTITY_GATE = 0.05`, else
  REVIEW ("capture does not resemble this item"). *Stage 2 (aligned local
  anomaly — the defect mechanism):* both images go to 512×512 blurred
  grayscale, small camera translation is compensated via
  `cv2.phaseCorrelate` (> 0.5 px → `warpAffine`), absdiff is pooled into
  per-cell means over a 24×24 grid, and the capture's `anomaly` is the
  hottest cell's mean (0–255 scale) with its `hot_region` reported. The
  reject threshold is **learned from the gallery**: accepted entries
  score leave-one-out against their nearest *other* accepted entry,
  rejected entries against their nearest accepted entry, and
  `T = max(12, (max_accepted + min_rejected) / 2)` — the floor keeps
  sensor/JPEG noise from ever triggering REJECT. Verdict: `A ≥ T×1.15` →
  REJECT, `A ≤ T/1.15` → ACCEPT, between → REVIEW. Galleries with < 1
  accepted or < 1 rejected example **always REVIEW** — the engine refuses
  to guess before it has seen both sides. Confidence is a deterministic
  separation-from-boundary value; reasons state the why ("Localized
  difference vs accepted reference '…' (strength 25 at region row 9 col 7;
  learned reject threshold 17 from 2 accepted / 1 rejected…)").

## Where each piece lives

| Piece | Code |
|---|---|
| Learning engine (galleries, embedding, cache, `learn()`, `judge()`) | `src/similarity_engine.py` |
| On-page item selector (synced with sidebar Active item) | `src/app.py` `_page_item_selector` |
| Inspection auto-judge + confirm/override learning loop | `src/app.py` `_similarity_inspection`, `_sim_learn_and_log` |
| Training-mode OK/NG → `learn()` + gallery line | `src/app.py` `_training_mode` |
| Dashboard learned-counts stat | `src/app.py` `page_dashboard` station status |
| Feedback ledger (append-only) | `results/feedback.jsonl` (`_append_feedback`) |
| Label ledger (append-only) | `data/labels.jsonl` |
| Engine tests (synthetic item, 27 checks) | `scripts/test_similarity_engine.py` |

Precedence: the **default (demo board)** keeps its existing flows untouched —
demo scenarios / precomputed JSON / trained PP-YOLOE+ model still drive it.
For **registered items**, similarity learning is the judge as soon as any
example is learned; the amber "setup pending" state only remains while the
galleries are completely empty.

## Honest limitations

- **Appearance-level similarity + local differencing only.** The engine
  answers "is this the item, and does any small region differ from the
  golden references more than learned rejects did?" It cannot name a
  missing reference designator, cannot prove *which* part is wrong, and
  explains a defect only as "strength N at region row R col C, similar in
  spirit to this earlier rejected capture".
- **Controlled rig still required.** Fixed camera, fixed lighting, fixed
  seating. Stage 2 compensates **small translations** via phase
  correlation, but rotation, scale, and viewpoint changes are NOT
  corrected — the item must sit in a fixed mount. A lighting shift still
  looks like a defect.
- **Cold start is honest, not magic.** With < 1 accepted or < 1 rejected
  example every verdict is REVIEW; judgments inside the ±15% dead band
  around the learned threshold are REVIEW by design. The operator's first
  corrections are the training data.
- **Not a replacement for component-level inspection.** When ref-des-level
  proof is needed ("R7 missing", "C3 wrong part"), the PP-YOLOE+ detector +
  rule engine + `expected_components.json` remains the right tool — it is
  kept as the advanced path for the default item and annotated items.
- **Upgrade path.** If appearance similarity plateaus, the embedding can be
  swapped for a deep backbone (torch + PatchCore/anomalib, already deferred
  in the repo) behind the same `learn()`/`judge()` API — galleries, ledger,
  and UI wiring stay unchanged.

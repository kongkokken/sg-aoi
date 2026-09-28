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
- **Embedding.** Canonical 256×256 resize, then four L2-normalized weighted
  blocks: HSV color histogram (global color), raw 8×8 grayscale grid moments
  (position-sensitive layout), **sorted** grid moments + grayscale quantiles
  (position-*invariant* defect signature — a dark blob/missing part changes
  the distribution tails wherever it sits; heaviest weight), and a HOG-style
  Sobel gradient-orientation grid (structure). Deterministic, numpy + cv2
  only, ~9 ms/image on CPU — no torch/anomalib/paddle, cloud-safe.
- **Cache.** `data/items/<id>/features.npz` (default scope:
  `data/.sim_cache/default.npz`) stores paths + mtimes + labels + vectors;
  only new/changed files are re-embedded (mtime check).
- **Scoring.** Cosine distance to the nearest accepted prototype (`d_ok`)
  vs nearest rejected (`d_ng`); margin score `(d_ng − d_ok)/(d_ng + d_ok)`.
  ACCEPT ≥ +0.10, REJECT ≤ −0.10, else REVIEW. Galleries with < 1 accepted
  or < 1 rejected example **always REVIEW** — the engine refuses to guess
  before it has seen both sides. Confidence = similarity to the nearest
  winning prototype. Reasons name the prototype ("Very similar to a
  rejected capture '…' from 2026-09-28 (distance 0.03)").

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

- **Appearance-level similarity only.** The engine answers "does this look
  like things you accepted or things you rejected?" It cannot name a missing
  reference designator, cannot prove *which* part is wrong, and cannot
  explain a defect beyond "similar to this earlier capture".
- **Controlled rig still required.** Fixed camera, fixed lighting, fixed
  seating. The embedding tolerates noise and small variations, not a
  different viewpoint or illumination change — a lighting shift looks like a
  defect.
- **Cold start is honest, not magic.** With < 1 accepted or < 1 rejected
  example every verdict is REVIEW; early judgments near the boundary are
  REVIEW by design. The operator's first corrections are the training data.
- **Not a replacement for component-level inspection.** When ref-des-level
  proof is needed ("R7 missing", "C3 wrong part"), the PP-YOLOE+ detector +
  rule engine + `expected_components.json` remains the right tool — it is
  kept as the advanced path for the default item and annotated items.
- **Upgrade path.** If appearance similarity plateaus, the embedding can be
  swapped for a deep backbone (torch + PatchCore/anomalib, already deferred
  in the repo) behind the same `learn()`/`judge()` API — galleries, ledger,
  and UI wiring stay unchanged.

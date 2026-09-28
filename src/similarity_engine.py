"""Per-item similarity learning engine — the item-agnostic default brain.

Concept (docs/concept_revision.md): the app learns what an item looks like
from GOLDEN pictures plus the operator's accept/reject feedback, and then
AUTO-JUDGES new captures by similarity — no component annotation, no trained
detector, no GPU, no deep-learning framework.

Gallery model
-------------
Per item there are two galleries: ACCEPTED examples (OK prototypes) and
REJECTED examples (NG prototypes). Sources:

* a registered item (``item_id``): ``data/items/<id>/golden_board.jpg`` and
  ``data/items/<id>/captures/*`` count as accepted; everything the user ever
  labeled accept/reject is tracked in ``data/items/<id>/learned.jsonl`` and
  layered on top (a learned label wins over the folder default for the same
  file).
* the default/demo scope (``item_id=None``): ``data/golden/golden_board.jpg``
  + ``data/boards_ok/*`` (accepted), ``data/boards_ng/*`` (rejected), plus
  the learned ledger ``data/.sim_cache/default_learned.jsonl``.

Embedding design (deterministic, CPU-only, a few ms per image)
--------------------------------------------------------------
Each image is resized to a canonical 256x256 and described by four
complementary blocks, each L2-normalized and weighted before concatenation:

* **HSV color histogram** (16x8x4 bins) — global color signature. Cheap and
  stable under the controlled rig; catches wrong-part / wrong-material
  changes that shift the color mix.
* **Raw 8x8 grayscale grid moments** (mean + std per cell) — position-
  SENSITIVE layout signature: "is this the same item, seated the same way,
  with the same local brightness pattern?".
* **SORTED 8x8 grid moments + grayscale quantiles** — position-INVARIANT
  texture signature. This block is what lets a defect match previously
  rejected captures even when it sits at a different location: a dark blob /
  missing part / contamination changes the sorted tails of the cell-mean
  distribution and the low intensity quantiles the same way wherever it
  appears. It carries the largest weight.
* **Gradient-orientation grid** — a HOG-style descriptor computed manually
  (Sobel gradients on a 128x128 grayscale thumbnail, orientation histograms
  per 8x8 cell, 9 bins): edge and structure signature, sensitive to shape
  changes a histogram misses. Manual because OpenCV 5.0 dropped
  ``cv2.HOGDescriptor``; Sobel + binning is deterministic and just as fast.

The concatenated vector (~4.3k floats) is L2-normalized once more. The
combination is deliberately simple: deterministic, dependency-free (numpy +
cv2 only), fast on CPU, and discriminative enough to separate *items* from
each other and gate capture identity. It is NOT the defect detector — small
localized defects move it by less than its own noise floor (see "Scoring"),
so defect judgement lives in the aligned-local-anomaly stage.

Scoring — two-stage judgement (identity gate + aligned local anomaly)
---------------------------------------------------------------------
The global appearance embedding alone cannot see small localized defects:
a missing/wrong part that changes ~0.2% of the pixels moves the ~4.3k-float
embedding by a cosine distance of ~0.00003, and position-sensitive blocks
push *different* defects *apart* in different directions (measured: two
different defects can be 3x farther from each other than each is from the
golden board), so no re-weighting of the embedding can avoid false accepts.
``judge()`` therefore scores in two stages, the way real AOI works:

* **Stage 1 — identity gate (embedding).** Cosine distance to the nearest
  accepted prototype must be <= ``IDENTITY_GATE`` (0.05, deliberately loose:
  same-item captures measure ~3e-5..1e-2, alien images ~0.15-0.20). Beyond
  that the capture simply is not this item (wrong product, wrong framing)
  and the verdict is REVIEW with a "does not resemble this item" reason.
  The embedding keeps its item-identity role; it no longer carries the
  defect decision.
* **Stage 2 — aligned local anomaly vs the nearest accepted prototype
  (the defect mechanism).** Both images are converted to grayscale,
  resized to 512x512 and Gaussian-blurred (5x5, JPEG/sensor-noise
  suppression). Small camera translation is compensated with
  ``cv2.phaseCorrelate`` — when the measured shift is > 0.5 px the capture
  is warped back by the compensating shift (``cv2.warpAffine``);
  rotation/scale correction is deliberately out of scope (controlled-rig
  assumption). The absdiff is pooled into per-cell means over a 24x24
  grid; ``anomaly`` is the hottest cell's mean (0-255 blurred-gray scale)
  and the hot cell's ``(row, col)`` is reported as ``hot_region``.

The reject threshold is LEARNED from the gallery, recomputed whenever the
cache rebuilds: each accepted entry's anomaly vs its nearest OTHER accepted
entry (leave-one-out; a single accepted entry contributes 0), and each
rejected entry's anomaly vs ITS nearest accepted entry (by embedding).
``T = max(ANOMALY_FLOOR, (max_accepted + min_rejected) / 2)`` with
``ANOMALY_FLOOR = 12`` so sensor/JPEG noise alone can never trigger REJECT.
Verdict from capture anomaly ``A`` with a +/-15% dead band:
``A >= T*1.15`` -> REJECT, ``A <= T/1.15`` -> ACCEPT, between -> REVIEW.
Galleries with fewer than 1 accepted or 1 rejected example ALWAYS return
REVIEW: the engine refuses to guess before it has seen both sides.

``confidence`` is a deterministic separation-from-boundary value: for
REJECT, ``min(1, (A - T*1.15) / (2 * T*1.15))`` (0 at the reject boundary,
1 at three times it — a barely-over-threshold reject is honestly
low-confidence); for ACCEPT, ``(T/1.15 - A) / (T/1.15)`` (1 at zero
anomaly, 0 at the accept boundary); 0.5 inside the dead band and 0.0 when
the galleries are one-sided. The old cosine margin score is still reported
as ``margin_score`` for diagnostics but no longer drives the verdict.

Cache
-----
Per item, ``data/items/<id>/features.npz`` (default scope:
``data/.sim_cache/default.npz`` stores paths + mtimes + labels + vectors,
plus ``version`` and the per-entry learned-anomaly values
(``anomalies`` + ``anomaly_refs``, the accepted-reference pairing used for
each value). On every load, entries whose file mtime is unchanged are
reused; only new or changed files are re-embedded, and vanished files are
dropped. Caches without the current ``version`` are rebuilt from scratch,
so format upgrades never read stale arrays. Anomaly values are recomputed
for ALL entries whenever the entry set changes (they depend on the gallery
composition) and reused from the cache while nothing changed.

All heavy imports (cv2, numpy) are lazy inside functions so this module
imports anywhere (cloud-safe, test-safe). No torch / anomalib / paddle.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Scope ids / on-disk layout -------------------------------------------------
DEFAULT_SCOPE_CACHE_DIR = ".sim_cache"  # under data_root for the default item
DEFAULT_CACHE_NAME = "default.npz"
DEFAULT_LEDGER_NAME = "default_learned.jsonl"
ITEM_CACHE_NAME = "features.npz"
ITEM_LEDGER_NAME = "learned.jsonl"

# Embedding shape ------------------------------------------------------------
_EMBED_SIZE = 256        # canonical resize (square)
_GRID = 8                # grid moments: _GRID x _GRID cells
_HOG_SIZE = 128          # HOG thumbnail

# Block weights (pre-final-normalization). The sorted grid moments carry the
# most weight because they are the position-invariant defect signature; the
# raw grid keeps the engine honest about layout, and the gradient-orientation
# block stays light — it is informative but has the highest noise floor.
_W_HIST = 1.0
_W_GRID = 0.3
_W_SGRID = 2.5
_W_HOG = 0.15

# Decision threshold on the normalized margin score (see module docstring).
# 0.10 = the nearer side must be ~10% of the distance scale closer, else REVIEW.
# Kept for the diagnostic ``margin_score`` field only — the verdict is now
# driven by the two-stage scheme below.
REVIEW_MARGIN = 0.10

# Stage 1 — identity gate: cosine distance to the nearest accepted prototype
# beyond this means "this capture is not the item" -> REVIEW. Measured scale:
# same-item captures (even unseen ones, different defects) sit at ~3e-5..1e-2,
# while alien images (solid color, pure noise) land at ~0.15-0.20. 0.05 is
# therefore still a very loose identity-only gate — the embedding gates item
# identity, never the defect decision.
IDENTITY_GATE = 0.05

# Stage 2 — aligned local anomaly (see module docstring).
_ANOM_SIZE = 512         # canonical anomaly working size (square)
_ANOM_GRID = 24          # per-cell mean pooling grid: _ANOM_GRID x _ANOM_GRID
_ANOM_SHIFT_MIN = 0.5    # px; translations below this are not compensated
ANOMALY_FLOOR = 12.0     # 0-255 blurred-gray scale; noise can never reject
_ANOM_DEADBAND = 1.15    # +/-15% dead band around the learned threshold

# npz cache format version. Bump when stored arrays change meaning/layout;
# older caches are discarded and rebuilt from scratch.
_CACHE_VERSION = 2

_LABELS = {"accept", "reject"}
_LABEL_ALIASES = {"ok": "accept", "ng": "reject",
                  "accept": "accept", "reject": "reject"}


# ---------------------------------------------------------------------------
# Paths / scope resolution
# ---------------------------------------------------------------------------

def _data_root(data_root: str | Path | None) -> Path:
    return Path(data_root) if data_root else DEFAULT_DATA_ROOT


def _item_dir(item_id: str, data_root: Path) -> Path:
    return data_root / "items" / item_id


def cache_path(item_id: str | None, data_root: str | Path | None = None) -> Path:
    root = _data_root(data_root)
    if item_id:
        return _item_dir(item_id, root) / ITEM_CACHE_NAME
    return root / DEFAULT_SCOPE_CACHE_DIR / DEFAULT_CACHE_NAME


def learned_ledger_path(item_id: str | None,
                        data_root: str | Path | None = None) -> Path:
    root = _data_root(data_root)
    if item_id:
        return _item_dir(item_id, root) / ITEM_LEDGER_NAME
    return root / DEFAULT_SCOPE_CACHE_DIR / DEFAULT_LEDGER_NAME


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("label") in _LABELS and rec.get("path"):
            out.append(rec)
    return out


def _normalize_label(label: str) -> str:
    norm = _LABEL_ALIASES.get(str(label).strip().lower())
    if norm is None:
        raise ValueError(f"invalid label {label!r} — use 'accept' or 'reject'")
    return norm


# ---------------------------------------------------------------------------
# Gallery sources (pure filesystem scan — no embeddings computed here)
# ---------------------------------------------------------------------------

def gallery_sources(item_id: str | None = None,
                    data_root: str | Path | None = None) -> list[tuple[Path, str]]:
    """(path, label) pairs that make up the item's galleries right now.

    Folder defaults are applied first; learned-ledger entries override them
    for the same file (a user correction beats the folder the file sits in).
    Missing files are dropped. Sorted for determinism.
    """
    root = _data_root(data_root)
    sources: dict[str, str] = {}

    def add_file(path: Path, label: str) -> None:
        if path.is_file():
            sources[str(path.resolve())] = label

    def add_folder(folder: Path, label: str) -> None:
        if not folder.is_dir():
            return
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                sources[str(p.resolve())] = label

    if item_id:
        idir = _item_dir(item_id, root)
        add_file(idir / "golden_board.jpg", "accept")
        add_folder(idir / "captures", "accept")
    else:
        add_file(root / "golden" / "golden_board.jpg", "accept")
        add_folder(root / "boards_ok", "accept")
        add_folder(root / "boards_ng", "reject")

    for rec in _read_ledger(learned_ledger_path(item_id, root)):
        p = Path(rec["path"])
        if p.is_file():
            sources[str(p.resolve())] = rec["label"]

    return [(Path(p), lbl) for p, lbl in sorted(sources.items())]


def gallery_counts(item_id: str | None = None,
                   data_root: str | Path | None = None) -> dict[str, int]:
    """Cheap accepted/rejected example counts (no embeddings computed)."""
    counts = {"ok": 0, "ng": 0}
    for _, label in gallery_sources(item_id, data_root):
        counts["ok" if label == "accept" else "ng"] += 1
    return counts


# ---------------------------------------------------------------------------
# Embedding (lazy cv2/numpy)
# ---------------------------------------------------------------------------

def _read_image(image: str | Path | Any):
    """BGR uint8 image from a path or an existing ndarray (BGR or RGB)."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"could not read image: {image}")
        return img
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected an HxWx3 image array")
    return arr


def _l2norm(vec):
    import numpy as np  # noqa: PLC0415

    return vec / (np.linalg.norm(vec) + 1e-12)


def _embedding(img_bgr):
    """The weighted multi-block feature vector (see module docstring)."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    img = cv2.resize(img_bgr, (_EMBED_SIZE, _EMBED_SIZE),
                     interpolation=cv2.INTER_AREA)

    # 1. HSV color histogram (16x8x4 = 512-d, L2-normalized)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 4],
                        [0, 180, 0, 256, 0, 256]).flatten().astype(np.float64)
    hist = _l2norm(hist)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    # 2./3. 8x8 grid moments — raw (position-sensitive) and sorted
    # (position-invariant defect signature), plus grayscale quantiles: a dark
    # blob / contamination shows up in the low quantiles wherever it sits.
    cell = _EMBED_SIZE // _GRID
    cells = gray.reshape(_GRID, cell, _GRID, cell)
    means = cells.mean(axis=(1, 3)).ravel()
    stds = cells.std(axis=(1, 3)).ravel()
    grid = _l2norm(np.concatenate([means, stds]))
    quantiles = np.quantile(
        gray, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    sgrid = _l2norm(np.concatenate(
        [np.sort(means), np.sort(stds), quantiles * 8.0]))

    # 4. HOG-style gradient-orientation grid on a 128x128 thumbnail:
    # Sobel gradients, 9 unsigned-orientation bins per 8x8 cell (576-d).
    # (cv2 5.0 has no HOGDescriptor; this manual version is deterministic.)
    thumb = cv2.resize(gray, (_HOG_SIZE, _HOG_SIZE),
                       interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(thumb, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(thumb, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) % np.pi) * (9.0 / np.pi)  # 9 unsigned bins
    hcell = _HOG_SIZE // _GRID
    hog = np.zeros((_GRID, _GRID, 9), dtype=np.float64)
    for cy in range(_GRID):
        for cx in range(_GRID):
            ys, xs = cy * hcell, cx * hcell
            cell_mag = mag[ys:ys + hcell, xs:xs + hcell].ravel()
            cell_ang = ang[ys:ys + hcell, xs:xs + hcell].ravel().astype(int)
            hog[cy, cx] = np.bincount(np.clip(cell_ang, 0, 8),
                                      weights=cell_mag, minlength=9)
    hvec = _l2norm(hog.ravel())

    vec = np.concatenate([hist * _W_HIST, grid * _W_GRID,
                          sgrid * _W_SGRID, hvec * _W_HOG])
    return _l2norm(vec).astype(np.float32)


def embed_image(image: str | Path | Any):
    """Public embedding helper (path or HxWx3 array -> float32 vector)."""
    return _embedding(_read_image(image))


# ---------------------------------------------------------------------------
# Aligned local anomaly (stage 2 — lazy cv2/numpy)
# ---------------------------------------------------------------------------

def _anomaly_gray(img_bgr):
    """512x512 blurred float32 grayscale — the canonical anomaly working form.

    Blurring with a 5x5 Gaussian suppresses JPEG/sensor pixel noise so the
    per-cell absdiff means below reflect structure, not grain.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    img = cv2.resize(img_bgr, (_ANOM_SIZE, _ANOM_SIZE),
                     interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray.astype(np.float32)


def _compensate_translation(ref_gray, cap_gray):
    """Warp ``cap_gray`` onto ``ref_gray`` when translated > _ANOM_SHIFT_MIN px.

    ``cv2.phaseCorrelate`` measures the global translation between the two
    blurred float32 images (deterministic, CPU-only, sub-ms). Rotation and
    scale are deliberately NOT corrected — the controlled-rig assumption
    (fixed mount, fixed lighting) stays; see module docstring.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    (dx, dy), _response = cv2.phaseCorrelate(ref_gray, cap_gray)
    if abs(dx) > _ANOM_SHIFT_MIN or abs(dy) > _ANOM_SHIFT_MIN:
        mat = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
        cap_gray = cv2.warpAffine(
            cap_gray, mat, (_ANOM_SIZE, _ANOM_SIZE),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cap_gray, (float(dx), float(dy))


def _local_anomaly(ref_gray, cap_gray) -> tuple[float, tuple[int, int]]:
    """Aligned per-cell absdiff anomaly of a capture vs an accepted reference.

    Returns (score, (row, col)): the capture is translation-compensated onto
    the reference, absdiff is pooled into per-cell means over a 24x24 grid,
    and the score is the hottest cell's mean on the 0-255 blurred-gray
    scale. The max cell (not a robust rank) is used on purpose: real defects
    can be tiny enough to light up essentially one cell, so a 2nd-highest
    statistic would mask exactly the defects this stage exists to catch.
    """
    import numpy as np  # noqa: PLC0415

    cap_aligned, _shift = _compensate_translation(ref_gray, cap_gray)
    diff = np.abs(ref_gray - cap_aligned)
    edges = np.linspace(0, _ANOM_SIZE, _ANOM_GRID + 1).astype(int)
    best = -1.0
    hot = (0, 0)
    for r in range(_ANOM_GRID):
        for c in range(_ANOM_GRID):
            v = float(diff[edges[r]:edges[r + 1],
                           edges[c]:edges[c + 1]].mean())
            if v > best:
                best, hot = v, (r, c)
    return best, hot


def _anomaly_vs_entry(entry_gray, ref_gray) -> float:
    """Score-only anomaly helper (gallery self-calibration)."""
    return _local_anomaly(ref_gray, entry_gray)[0]


# ---------------------------------------------------------------------------
# Feature cache (paths + mtimes + labels + vectors, incremental rebuild)
# ---------------------------------------------------------------------------

def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    import numpy as np  # noqa: PLC0415

    try:
        with np.load(path, allow_pickle=False) as data:
            version = int(data["version"]) if "version" in data else 0
            if version < _CACHE_VERSION:
                return {}  # stale format — rebuild from scratch
            paths = [str(p) for p in data["paths"]]
            labels = [str(l) for l in data["labels"]]
            mtimes = data["mtimes"].astype(float)
            vectors = data["vectors"]
            anomalies = data["anomalies"].astype(float)
            anomaly_refs = [str(r) for r in data["anomaly_refs"]]
    except Exception:  # noqa: BLE001 - corrupt cache rebuilds from scratch
        return {}
    return {
        p: {"label": labels[i], "mtime": float(mtimes[i]),
            "vector": vectors[i], "anomaly": float(anomalies[i]),
            "anomaly_ref": anomaly_refs[i]}
        for i, p in enumerate(paths)
    }


def _save_cache(path: Path, entries: list[dict[str, Any]]) -> None:
    import numpy as np  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        version=np.int64(_CACHE_VERSION),
        paths=np.array([e["path"] for e in entries]),
        labels=np.array([e["label"] for e in entries]),
        mtimes=np.array([e["mtime"] for e in entries], dtype=float),
        vectors=np.stack([e["vector"] for e in entries])
        if entries else np.zeros((0, 0), dtype=np.float32),
        anomalies=np.array([e.get("anomaly", 0.0) for e in entries],
                           dtype=float),
        anomaly_refs=np.array([e.get("anomaly_ref", "") for e in entries]),
    )


def _calibrate_anomalies(entries: list[dict[str, Any]]) -> None:
    """Learn per-entry anomaly values from the gallery itself (in place).

    * accepted side: each accepted entry's anomaly vs its nearest OTHER
      accepted entry (leave-one-out, nearest by embedding); with only one
      accepted entry the side contributes 0.
    * rejected side: each rejected entry's anomaly vs ITS nearest accepted
      entry (nearest by embedding); with no accepted entries there is no
      reference, so the value stays 0.

    ``anomaly_ref`` records the accepted-reference path used ("" when none).
    Values are recomputed for ALL entries whenever the entry set changed —
    they depend on the gallery composition, not on the entry alone.
    """
    ok = [e for e in entries if e["label"] == "accept"]
    grays: dict[str, Any] = {}

    def gray_of(entry: dict[str, Any]):
        g = grays.get(entry["path"])
        if g is None:
            g = _anomaly_gray(_read_image(entry["path"]))
            grays[entry["path"]] = g
        return g

    for e in entries:
        e["anomaly"] = 0.0
        e["anomaly_ref"] = ""
        if e["label"] == "accept":
            others = [o for o in ok if o["path"] != e["path"]]
            if not others:
                continue  # single accepted entry -> side contributes 0
            _, ref = _nearest(e["vector"], others)
        else:
            if not ok:
                continue
            _, ref = _nearest(e["vector"], ok)
        e["anomaly"] = _anomaly_vs_entry(gray_of(e), gray_of(ref))
        e["anomaly_ref"] = ref["path"]


def _learned_threshold(entries: list[dict[str, Any]]) -> float | None:
    """Reject threshold T from the calibrated gallery (None if one-sided).

    T = max(ANOMALY_FLOOR, (max_accepted + min_rejected) / 2) — the midpoint
    between the worst accepted self-anomaly and the easiest rejected
    anomaly, floored so sensor/JPEG noise alone can never trigger REJECT.
    """
    acc = [e["anomaly"] for e in entries if e["label"] == "accept"]
    rej = [e["anomaly"] for e in entries if e["label"] == "reject"]
    if not acc or not rej:
        return None
    return float(max(ANOMALY_FLOOR, (max(acc) + min(rej)) / 2.0))


def load_gallery(item_id: str | None = None,
                 data_root: str | Path | None = None) -> dict[str, Any]:
    """Gallery entries with embedding vectors, using the mtime cache.

    Returns {"entries": [{path, name, label, mtime, date, vector, anomaly,
             anomaly_ref}, ...], "n_ok": int, "n_ng": int,
             "threshold": float | None}. Only new/changed files are
    re-embedded; per-entry anomaly values are recomputed for the whole
    gallery whenever the entry set changed (they depend on the gallery
    composition); the cache file is rewritten when anything changed.
    """
    root = _data_root(data_root)
    sources = gallery_sources(item_id, root)
    cached = _load_cache(cache_path(item_id, root))

    entries: list[dict[str, Any]] = []
    dirty = len(cached) != 0  # dropped files must be purged from the cache
    for path, label in sources:
        key = str(path.resolve())
        mtime = path.stat().st_mtime
        hit = cached.get(key)
        if hit is not None and hit["mtime"] == mtime and hit["label"] == label:
            vector = hit["vector"]
            anomaly = hit["anomaly"]
            anomaly_ref = hit["anomaly_ref"]
        else:
            vector = _embedding(_read_image(path))
            anomaly, anomaly_ref = 0.0, ""
            dirty = True
        entries.append({
            "path": key, "name": path.name, "label": label, "mtime": mtime,
            "date": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d"),
            "vector": vector, "anomaly": anomaly, "anomaly_ref": anomaly_ref,
        })

    if dirty or len(cached) != len(entries):
        _calibrate_anomalies(entries)
        _save_cache(cache_path(item_id, root), entries)

    return {
        "entries": entries,
        "n_ok": sum(1 for e in entries if e["label"] == "accept"),
        "n_ng": sum(1 for e in entries if e["label"] == "reject"),
        "threshold": _learned_threshold(entries),
    }


# ---------------------------------------------------------------------------
# Learning + judging
# ---------------------------------------------------------------------------

def learn(image_path: str | Path, label: str, item_id: str | None = None,
          data_root: str | Path | None = None) -> dict[str, Any]:
    """Add a user-labeled image to the item's galleries and refresh the cache.

    Appends one record to the learned ledger (append-only JSONL) and returns
    the updated gallery counts. Exact duplicate (same file, same label) is a
    no-op for the ledger. Raises ValueError on an unreadable image or bad
    label.
    """
    root = _data_root(data_root)
    label = _normalize_label(label)
    path = Path(image_path)
    _read_image(path)  # raises ValueError if unreadable
    key = str(path.resolve())

    ledger = learned_ledger_path(item_id, root)
    existing = _read_ledger(ledger)
    if not any(str(Path(r["path"]).resolve()) == key and r["label"] == label
               for r in existing):
        ledger.parent.mkdir(parents=True, exist_ok=True)
        record = {"path": str(path), "label": label,
                  "timestamp": datetime.now().isoformat(timespec="seconds")}
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    gal = load_gallery(item_id, root)
    return {"path": str(path), "label": label,
            "ok": gal["n_ok"], "ng": gal["n_ng"]}


def _nearest(vec, entries: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    import numpy as np  # noqa: PLC0415

    best_d, best = float("inf"), entries[0]
    for e in entries:
        d = 1.0 - float(np.dot(vec, e["vector"]))
        if d < best_d:
            best_d, best = d, e
    return best_d, best


def _proto_info(entry: dict[str, Any] | None, distance: float | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    return {"path": entry["path"], "name": entry["name"],
            "date": entry["date"], "distance": round(float(distance), 5)}


def judge(image: str | Path | Any, item_id: str | None = None,
          data_root: str | Path | None = None) -> dict[str, Any]:
    """Auto-judge one capture against the item's learned galleries.

    Two-stage scheme (see module docstring): stage 1 gates item identity
    with the global embedding (cosine distance to the nearest accepted
    prototype must be <= IDENTITY_GATE, else REVIEW "does not resemble this
    item"); stage 2 measures the aligned local anomaly of the capture vs the
    nearest accepted prototype and compares it with the threshold learned
    from the gallery (REJECT at A >= T*1.15, ACCEPT at A <= T/1.15, REVIEW
    inside the dead band).

    Returns {verdict: ACCEPT|REJECT|REVIEW, confidence: 0..1,
             nearest_ok: {...}|None, nearest_ng: {...}|None, reason: str,
             n_ok, n_ng, anomaly: float|None, threshold: float|None,
             hot_region: (row, col)|None, margin_score: float (diagnostic)}.
    Galleries with <1 accepted or <1 rejected example always REVIEW ("not
    enough learned examples"). Confidence is a deterministic
    separation-from-boundary value (module docstring, "Scoring").
    """
    gal = load_gallery(item_id, data_root)
    ok_entries = [e for e in gal["entries"] if e["label"] == "accept"]
    ng_entries = [e for e in gal["entries"] if e["label"] == "reject"]
    threshold = gal["threshold"]

    base: dict[str, Any] = {
        "n_ok": len(ok_entries), "n_ng": len(ng_entries),
        "anomaly": None, "threshold": threshold, "hot_region": None,
    }
    if not ok_entries or not ng_entries:
        return {
            **base, "verdict": "REVIEW", "confidence": 0.0,
            "nearest_ok": None, "nearest_ng": None, "margin_score": 0.0,
            "reason": ("Not enough learned examples — need at least "
                       "1 accepted and 1 rejected "
                       f"(have {len(ok_entries)} accepted / "
                       f"{len(ng_entries)} rejected)."),
        }

    img = _read_image(image)
    vec = _embedding(img)
    d_ok, e_ok = _nearest(vec, ok_entries)
    d_ng, e_ng = _nearest(vec, ng_entries)
    score = (d_ng - d_ok) / (d_ng + d_ok + 1e-12)

    result = {
        **base,
        "nearest_ok": _proto_info(e_ok, d_ok),
        "nearest_ng": _proto_info(e_ng, d_ng),
        "margin_score": round(score, 4),
    }

    # --- stage 1: identity gate ------------------------------------------
    if d_ok > IDENTITY_GATE:
        result.update(
            verdict="REVIEW", confidence=0.0,
            reason=("Capture does not resemble this item — distance "
                    f"{d_ok:.5f} to the nearest accepted example "
                    f"'{e_ok['name']}' exceeds the identity gate "
                    f"{IDENTITY_GATE:.2f}. Wrong item, wrong framing, or a "
                    "viewpoint/lighting change; human review needed."),
        )
        return result

    # --- stage 2: aligned local anomaly vs nearest accepted prototype -----
    cap_gray = _anomaly_gray(img)
    ref_gray = _anomaly_gray(_read_image(e_ok["path"]))
    anomaly, hot = _local_anomaly(ref_gray, cap_gray)
    assert threshold is not None  # both sides present -> threshold learned
    reject_at = threshold * _ANOM_DEADBAND
    accept_at = threshold / _ANOM_DEADBAND
    result.update(anomaly=round(anomaly, 2), hot_region=hot)

    if anomaly >= reject_at:
        # 0 at the reject boundary, 1 at 3x the boundary — a barely-over-
        # threshold reject is honestly low-confidence.
        conf = min(1.0, (anomaly - reject_at) / (2.0 * max(reject_at, 1e-9)))
        result.update(
            verdict="REJECT", confidence=round(conf, 4),
            reason=("Localized difference vs accepted reference "
                    f"'{e_ok['name']}' (strength {anomaly:.0f} at region "
                    f"row {hot[0]} col {hot[1]}; learned reject threshold "
                    f"{threshold:.0f} from {len(ok_entries)} accepted / "
                    f"{len(ng_entries)} rejected; nearest rejected "
                    f"'{e_ng['name']}' at embedding distance {d_ng:.5f})."),
        )
    elif anomaly <= accept_at:
        conf = max(0.0, min(1.0, (accept_at - anomaly)
                            / max(accept_at, 1e-9)))
        result.update(
            verdict="ACCEPT", confidence=round(conf, 4),
            reason=(f"Matches accepted reference '{e_ok['name']}' within "
                    f"tolerance (strength {anomaly:.0f} vs learned "
                    f"threshold {threshold:.0f} from {len(ok_entries)} "
                    f"accepted / {len(ng_entries)} rejected; nearest "
                    f"rejected '{e_ng['name']}' at embedding distance "
                    f"{d_ng:.5f})."),
        )
    else:
        result.update(
            verdict="REVIEW", confidence=0.5,
            reason=("Ambiguous — localized difference strength "
                    f"{anomaly:.0f} at region row {hot[0]} col {hot[1]} "
                    f"sits inside the dead band around the learned "
                    f"threshold {threshold:.0f} (accept <= "
                    f"{accept_at:.0f}, reject >= {reject_at:.0f}); "
                    "human review needed."),
        )
    return result

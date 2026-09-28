"""Headless tests for src/similarity_engine.py — the per-item learning brain.

Run with the conda env:  D:\\miniforge3\\envs\\aoi-app\\python.exe scripts/test_similarity_engine.py

Builds a TEMP item with synthetic PIL images (good = same pattern + noise,
reject = same pattern + a black-box defect at varying locations) and checks
the full learn/judge/cache loop. The real data/ tree is never touched.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import similarity_engine as se  # noqa: E402

PASS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    PASS.append(cond)
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


# --- synthetic image factory -------------------------------------------------

def base_pattern(size: int = 512, seed: int = 7) -> Image.Image:
    """A fixed pseudo-PCBA pattern: colored rectangles + grid lines."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), (40, 90, 55))
    draw = ImageDraw.Draw(img)
    for _ in range(28):
        x, y = int(rng.integers(0, size - 60)), int(rng.integers(0, size - 60))
        w, h = int(rng.integers(18, 60)), int(rng.integers(10, 44))
        color = tuple(int(c) for c in rng.integers(60, 230, 3))
        draw.rectangle([x, y, x + w, y + h], fill=color, outline=(20, 20, 20))
    for gx in range(0, size, 32):
        draw.line([(gx, 0), (gx, size)], fill=(30, 70, 45))
        draw.line([(0, gx), (size, gx)], fill=(30, 70, 45))
    return img


def noisy_variant(base: Image.Image, seed: int, sigma: float = 6.0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(base).astype(np.float64)
    arr += rng.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def with_defect(base: Image.Image, seed: int, box: int = 64) -> Image.Image:
    """Same pattern + noise + a solid black box (the 'defect') somewhere."""
    rng = np.random.default_rng(seed)
    img = noisy_variant(base, seed)
    draw = ImageDraw.Draw(img)
    size = base.size[0]
    x = int(rng.integers(0, size - box))
    y = int(rng.integers(0, size - box))
    draw.rectangle([x, y, x + box, y + box], fill=(5, 5, 5))
    return img


def save(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="aoi_sim_test_"))
    try:
        data_root = tmp / "data"
        item_id = "widget-x"
        idir = data_root / "items" / item_id
        work = tmp / "work"
        base = base_pattern()

        goods = [save(noisy_variant(base, 100 + i), work / f"good_{i}.jpg")
                 for i in range(5)]
        defects = [save(with_defect(base, 200 + i), work / f"defect_{i}.jpg")
                   for i in range(4)]

        # --- (a) empty galleries -> REVIEW ------------------------------------
        r0 = se.judge(goods[0], item_id, data_root=data_root)
        check("(a) empty gallery -> REVIEW", r0["verdict"] == "REVIEW",
              r0["verdict"])
        check("(a) reason says not enough learned examples",
              "not enough learned examples" in r0["reason"].lower(),
              r0["reason"])
        check("(a) result shape complete",
              set(("verdict", "confidence", "nearest_ok", "nearest_ng",
                   "reason", "n_ok", "n_ng")) <= set(r0))
        check("(a) zero confidence on REVIEW-empty", r0["confidence"] == 0.0)
        check("(a) default scope empty too -> REVIEW",
              se.judge(goods[0], None, data_root=data_root)["verdict"] == "REVIEW")

        # --- learn 3 OK + 2 NG --------------------------------------------------
        for p in goods[:3]:
            se.learn(p, "accept", item_id, data_root=data_root)
        out = se.learn(defects[0], "reject", item_id, data_root=data_root)
        out = se.learn(defects[1], "reject", item_id, data_root=data_root)
        check("learn returns updated counts",
              out["ok"] == 3 and out["ng"] == 2, str(out))
        check("learned ledger written",
              se.learned_ledger_path(item_id, data_root).is_file())
        counts = se.gallery_counts(item_id, data_root)
        check("gallery_counts matches", counts == {"ok": 3, "ng": 2}, str(counts))
        # duplicate learn is a ledger no-op
        se.learn(defects[1], "reject", item_id, data_root=data_root)
        lines = [l for l in se.learned_ledger_path(item_id, data_root)
                 .read_text(encoding="utf-8").splitlines() if l.strip()]
        check("duplicate learn does not double-append", len(lines) == 5,
              f"{len(lines)} ledger lines")

        # --- (b) good capture -> ACCEPT, confidence ordering --------------------
        r_good = se.judge(goods[3], item_id, data_root=data_root)
        r_def = se.judge(defects[2], item_id, data_root=data_root)
        check("(b) unseen good -> ACCEPT", r_good["verdict"] == "ACCEPT",
              f"{r_good['verdict']} ({r_good['reason']})")
        check("(b) good confidence > defect confidence",
              r_good["confidence"] > r_def["confidence"],
              f"{r_good['confidence']} vs {r_def['confidence']}")

        # --- (c) defect capture -> REJECT ---------------------------------------
        check("(c) unseen defect -> REJECT", r_def["verdict"] == "REJECT",
              f"{r_def['verdict']} ({r_def['reason']})")
        r_def2 = se.judge(defects[3], item_id, data_root=data_root)
        check("(c) second unseen defect -> REJECT",
              r_def2["verdict"] == "REJECT",
              f"{r_def2['verdict']} ({r_def2['reason']})")

        # --- (d) reasons name the nearest prototype -----------------------------
        check("(d) REJECT reason names rejected prototype",
              r_def["nearest_ng"]["name"] in r_def["reason"],
              r_def["reason"])
        check("(d) ACCEPT reason names accepted prototype",
              r_good["nearest_ok"]["name"] in r_good["reason"],
              r_good["reason"])
        check("(d) nearest dicts carry path/date/distance",
              all(k in r_def["nearest_ng"]
                  for k in ("path", "name", "date", "distance"))
              and all(k in r_good["nearest_ok"]
                      for k in ("path", "name", "date", "distance")))
        check("(d) nearest_ok/ng point at the right labels",
              Path(r_def["nearest_ng"]["path"]).name.startswith("defect")
              and Path(r_good["nearest_ok"]["path"]).name.startswith("good"))

        # --- (e) cache persists + mtime invalidation ------------------------------
        cache = se.cache_path(item_id, data_root)
        check("(e) cache file exists", cache.is_file(), str(cache))
        gal1 = se.load_gallery(item_id, data_root)
        vec_before = {e["path"]: e["vector"].copy() for e in gal1["entries"]}
        # rewrite one gallery image with different pixels + bumped mtime
        target = goods[0]
        save(noisy_variant(base, 999, sigma=40.0), target)
        os.utime(target, (target.stat().st_atime + 5,
                          target.stat().st_mtime + 5))
        gal2 = se.load_gallery(item_id, data_root)
        vec_after = {e["path"]: e["vector"] for e in gal2["entries"]}
        key = str(target.resolve())
        check("(e) changed file re-embedded",
              not np.allclose(vec_before[key], vec_after[key]))
        others_changed = [p for p in vec_after
                          if p != key
                          and not np.allclose(vec_before[p], vec_after[p])]
        check("(e) unchanged files reuse cached vectors", not others_changed,
              str(others_changed))
        check("(e) entry count stable across reload",
              len(gal2["entries"]) == len(gal1["entries"]) == 5,
              f"{len(gal2['entries'])} entries")

        # --- (f) correction flips a verdict ---------------------------------------
        # The operator says the rejected capture is actually acceptable:
        pre = se.judge(defects[2], item_id, data_root=data_root)
        check("(f) pre-correction verdict is REJECT",
              pre["verdict"] == "REJECT", pre["verdict"])
        se.learn(defects[2], "accept", item_id, data_root=data_root)
        post = se.judge(defects[2], item_id, data_root=data_root)
        check("(f) verdict flips to ACCEPT after correction",
              post["verdict"] == "ACCEPT",
              f"{post['verdict']} ({post['reason']})")
        check("(f) corrected example now in accepted gallery",
              post["nearest_ok"]["path"] == str(defects[2].resolve()),
              post["nearest_ok"]["path"])
        counts = se.gallery_counts(item_id, data_root)
        check("(f) counts updated after correction",
              counts == {"ok": 4, "ng": 2}, str(counts))

        # --- label validation -----------------------------------------------------
        try:
            se.learn(goods[0], "maybe", item_id, data_root=data_root)
            check("invalid label rejected", False)
        except ValueError:
            check("invalid label rejected", True)
        # OK/NG aliases map onto accept/reject
        alias = se.learn(goods[4], "OK", item_id, data_root=data_root)
        check("OK alias maps to accept", alias["label"] == "accept"
              and alias["ok"] == 5, str(alias))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())

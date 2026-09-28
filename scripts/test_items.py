"""Headless tests for the item registry + config override + item maintenance.

Run with the conda env:  D:\\miniforge3\\envs\\aoi-app\\python.exe scripts/test_items.py

No Streamlit context needed — app.py is imported as a module (st is imported
but no st.* calls are made by the helpers under test). All registry operations
run in a TEMP data dir; the real data/items/ is never touched. Covers:
  * slugify_item_id — spaces, unicode, empty, invalid characters.
  * create_item — validation (empty/invalid/duplicate ids) + registry
    create/read round-trip (list_items / get_item) + defect-type seeding.
  * item_status — transitions: no golden -> setup pending; golden + 0
    components -> setup pending; golden + >0 components -> ready.
  * _item_cfg_override — produces the item's golden paths and NEVER mutates
    the base config dict.
  * item_defect_types — normalization of missing/empty catalogs to
    DEFAULT_DEFECT_TYPES (legacy items keep working).
  * update_item — name/description/revision/defect_types updates, unknown id.
  * delete_item — registry entry + files removed; missing dir tolerated.
  * _rewrite_learned_defect_type — renames defect types in learned.jsonl.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import app  # noqa: E402

PASS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    PASS.append(cond)
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="aoi_items_test_"))
    items_root = tmp / "items"
    try:
        # --- 1. slugify -------------------------------------------------------
        check("slugify spaces -> dashes",
              app.slugify_item_id("Power Supply Rev C") == "power-supply-rev-c",
              app.slugify_item_id("Power Supply Rev C"))
        check("slugify unicode transliterates/drops",
              app.slugify_item_id("Café 板 #2") == "cafe-2",
              app.slugify_item_id("Café 板 #2"))
        check("slugify empty -> ''", app.slugify_item_id("") == "")
        check("slugify punctuation-only -> ''",
              app.slugify_item_id("板#*!") == "")
        check("slugify collapses runs + strips",
              app.slugify_item_id("--A__B--") == "a-b",
              app.slugify_item_id("--A__B--"))

        # --- 2. create_item validation ----------------------------------------
        try:
            app.create_item("", "No Id", items_root=items_root)
            check("empty id rejected", False)
        except ValueError:
            check("empty id rejected", True)
        try:
            app.create_item("Bad Id!", "Bad", items_root=items_root)
            check("invalid id rejected", False)
        except ValueError:
            check("invalid id rejected", True)

        entry = app.create_item("widget-a", "Widget A", description="demo",
                                revision="Rev A", items_root=items_root)
        check("create_item returns id", entry["id"] == "widget-a", str(entry))
        check("captures dir created",
              (items_root / "widget-a" / "captures").is_dir())
        try:
            app.create_item("widget-a", "Widget A again", items_root=items_root)
            check("duplicate id rejected", False)
        except ValueError as exc:
            check("duplicate id rejected", True, str(exc))

        # --- 3. registry create/read round-trip --------------------------------
        app.create_item("widget-b", "Widget B", items_root=items_root)
        items = app.list_items(items_root)
        check("list_items returns both, oldest first",
              [it["id"] for it in items] == ["widget-a", "widget-b"],
              str([it["id"] for it in items]))
        got = app.get_item("widget-a", items_root)
        check("get_item round-trip fields",
              bool(got) and got["name"] == "Widget A"
              and got["revision"] == "Rev A"
              and got["golden_set"] is False
              and got["components_count"] == 0
              and got["annotation_status"] == "pending",
              json.dumps(got, default=str))
        check("get_item unknown -> None",
              app.get_item("nope", items_root) is None)
        check("index.json persisted on disk",
              (items_root / "index.json").is_file()
              and "widget-a" in json.loads(
                  (items_root / "index.json").read_text(encoding="utf-8")))

        # --- 4. item_status transitions -----------------------------------------
        item = app.get_item("widget-a", items_root)
        check("no golden -> setup pending",
              app.item_status(item, items_root) == "setup pending",
              app.item_status(item, items_root))

        idir = items_root / "widget-a"
        app.write_item_expected_components("widget-a", items_root)
        (idir / "golden_board.jpg").write_bytes(b"fake-jpg")
        check("golden + 0 comps -> setup pending",
              app.item_status(item, items_root) == "setup pending",
              app.item_status(item, items_root))
        template = json.loads(
            (idir / "expected_components.json").read_text(encoding="utf-8"))
        check("expected template honest (empty comps, pending)",
              template.get("components") == []
              and template.get("annotation_status") == "pending"
              and template.get("item") == "widget-a",
              json.dumps(template))

        template["components"] = [{"id": "R1", "class": "resistor_0603",
                                   "bbox": [0, 0, 10, 5]}]
        (idir / "expected_components.json").write_text(
            json.dumps(template), encoding="utf-8")
        check("golden + comps>0 -> ready",
              app.item_status(item, items_root) == "ready",
              app.item_status(item, items_root))

        app.refresh_item_registry_entry("widget-a", items_root)
        got = app.get_item("widget-a", items_root)
        check("registry re-sync caches golden_set/components_count",
              got["golden_set"] is True and got["components_count"] == 1
              and got["annotation_status"] == "annotated",
              json.dumps(got, default=str))

        # --- 5. config override ---------------------------------------------------
        base_cfg = {
            "golden": {"image": "data/golden/golden_board.jpg",
                       "expected_components": "data/golden/expected_components.json",
                       "match_iou": 0.4},
            "detection": {"model_dir": "models/detection/pcba_ppyoloe"},
        }
        base_snapshot = json.dumps(base_cfg, sort_keys=True)
        over = app._item_cfg_override(base_cfg, "widget-a", items_root)
        check("override points golden.image at the item",
              over["golden"]["image"] == str(idir / "golden_board.jpg"),
              over["golden"]["image"])
        check("override points expected_components at the item",
              over["golden"]["expected_components"]
              == str(idir / "expected_components.json"))
        check("override keeps unrelated keys",
              over["golden"]["match_iou"] == 0.4
              and over["detection"]["model_dir"]
              == "models/detection/pcba_ppyoloe")
        check("base config NOT mutated",
              json.dumps(base_cfg, sort_keys=True) == base_snapshot)
        check("override is an independent copy",
              over["golden"] is not base_cfg["golden"])

        # --- 6. empty/corrupt registry reads as empty ------------------------------
        check("missing registry -> no items",
              app.list_items(tmp / "no_such_dir") == [])
        corrupt = tmp / "corrupt" / "items"
        corrupt.mkdir(parents=True)
        (corrupt / "index.json").write_text("not json{", encoding="utf-8")
        check("corrupt registry -> no items (no crash)",
              app.list_items(corrupt) == [])

        # --- 7. defect-type catalog: seeding + normalization ----------------------
        check("create_item seeds default defect types",
              app.get_item("widget-a", items_root)["defect_types"]
              == app.DEFAULT_DEFECT_TYPES)
        check("item_defect_types returns the item's catalog",
              app.item_defect_types("widget-a", items_root)
              == app.DEFAULT_DEFECT_TYPES)
        check("item_defect_types default scope -> defaults copy",
              app.item_defect_types(None, items_root)
              == app.DEFAULT_DEFECT_TYPES
              and app.item_defect_types(None, items_root)
              is not app.DEFAULT_DEFECT_TYPES)
        check("item_defect_types unknown id -> defaults",
              app.item_defect_types("nope", items_root)
              == app.DEFAULT_DEFECT_TYPES)
        # Legacy entries without the field normalize to the defaults.
        index = json.loads(
            (items_root / "index.json").read_text(encoding="utf-8"))
        del index["widget-a"]["defect_types"]
        index["widget-b"]["defect_types"] = []
        (items_root / "index.json").write_text(json.dumps(index),
                                               encoding="utf-8")
        check("legacy entry without defect_types -> defaults",
              app.item_defect_types("widget-a", items_root)
              == app.DEFAULT_DEFECT_TYPES)
        check("empty defect_types -> defaults",
              app.item_defect_types("widget-b", items_root)
              == app.DEFAULT_DEFECT_TYPES)

        # --- 8. update_item ---------------------------------------------------------
        updated = app.update_item("widget-a", name="Widget A+", revision="Rev B",
                                  defect_types=["Scratch", "Missing part"],
                                  items_root=items_root)
        check("update_item applies fields",
              updated["name"] == "Widget A+"
              and updated["revision"] == "Rev B"
              and updated["defect_types"] == ["Scratch", "Missing part"],
              json.dumps(updated, default=str))
        check("update_item persisted + keeps unrelated fields",
              app.get_item("widget-a", items_root)["description"] == "demo"
              and app.item_defect_types("widget-a", items_root)
              == ["Scratch", "Missing part"])
        try:
            app.update_item("nope", name="X", items_root=items_root)
            check("update_item unknown id rejected", False)
        except ValueError:
            check("update_item unknown id rejected", True)

        # --- 9. _rewrite_learned_defect_type -----------------------------------------
        ledger = items_root / "widget-a" / "learned.jsonl"
        ledger.write_text(
            json.dumps({"path": "a.jpg", "label": "reject",
                        "defect_type": "Scratch"}) + "\n"
            + json.dumps({"path": "b.jpg", "label": "reject",
                          "defect_type": "Missing part"}) + "\n"
            + json.dumps({"path": "c.jpg", "label": "accept"}) + "\n",
            encoding="utf-8")
        rewritten = app._rewrite_learned_defect_type(
            "widget-a", "Scratch", "Scratch / cosmetic", items_root)
        recs = [json.loads(ln) for ln in
                ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
        check("learned.jsonl rewrite count", rewritten == 1, str(rewritten))
        check("learned.jsonl old type renamed, others untouched",
              recs[0]["defect_type"] == "Scratch / cosmetic"
              and recs[1]["defect_type"] == "Missing part"
              and "defect_type" not in recs[2],
              json.dumps(recs))
        check("rewrite with no matches -> 0",
              app._rewrite_learned_defect_type(
                  "widget-a", "No such type", "X", items_root) == 0)

        # --- 10. delete_item ----------------------------------------------------------
        app.delete_item("widget-b", items_root=items_root)
        check("delete_item removes registry entry + dir",
              app.get_item("widget-b", items_root) is None
              and not (items_root / "widget-b").exists())
        check("delete_item keeps unrelated entries",
              app.get_item("widget-a", items_root) is not None)
        app.delete_item("ghost", items_root=items_root)  # must not raise
        check("delete_item tolerates missing entry/dir", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())

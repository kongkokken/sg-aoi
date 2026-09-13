"""Headless tests for the Inspection mode snapshot flow's pure logic.

Run with the conda env:  D:\\miniforge3\\envs\\aoi-app\\python.exe scripts/test_inspection_flow.py

No Streamlit context needed — app.py is imported as a module (st is imported
but no st.* calls are made by the helpers under test). Covers:
  * _detection_source — the verdict-source resolver behind the status line
    ("model" | "demo" | "precomputed" | "none"), incl. the precedence that
    mirrors infer_pipeline.run_detection (precomputed_json wins over model).
  * _upload_ident — stable (name, size) identity for UploadedFile-likes.
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
    tmp = Path(tempfile.mkdtemp(prefix="aoi_inspection_test_"))
    try:
        # --- 1. demo mode wins over everything --------------------------------
        kind, label = app._detection_source({}, demo_active=True)
        check("demo mode -> demo", kind == "demo", kind)
        check("demo label honest", "DEMO" in label and "simulated" in label, label)

        # --- 2. exported model present -> trained model ------------------------
        model_dir = tmp / "models" / "detection" / "pcba_ppyoloe"
        model_dir.mkdir(parents=True)
        (model_dir / "model.pdmodel").write_bytes(b"fake")
        (model_dir / "model.pdiparams").write_bytes(b"fake")
        cfg_model = {"detection": {"model_dir": str(model_dir)}}
        kind, label = app._detection_source(cfg_model, demo_active=False)
        check("model files -> model", kind == "model", kind)
        check("model label names the dir", str(model_dir) in label, label)

        # --- 3. precomputed JSON -> precomputed (debug) ------------------------
        pre = tmp / "detections.json"
        pre.write_text(json.dumps([]), encoding="utf-8")
        cfg_pre = {"detection": {"precomputed_json": str(pre)}}
        kind, label = app._detection_source(cfg_pre, demo_active=False)
        check("precomputed json -> precomputed", kind == "precomputed", kind)
        check("precomputed label says debug", "precomputed JSON" in label
              and "debug" in label, label)

        # --- 4. precedence mirrors run_detection: precomputed beats model ------
        cfg_both = {"detection": {"model_dir": str(model_dir),
                                  "precomputed_json": str(pre)}}
        kind, _ = app._detection_source(cfg_both, demo_active=False)
        check("precomputed wins over model (run_detection precedence)",
              kind == "precomputed", kind)

        # --- 5. dangling precomputed path falls through to model ---------------
        cfg_dangling = {"detection": {"model_dir": str(model_dir),
                                      "precomputed_json": str(tmp / "gone.json")}}
        kind, _ = app._detection_source(cfg_dangling, demo_active=False)
        check("missing precomputed file falls through to model",
              kind == "model", kind)

        # --- 6. nothing configured -> none -------------------------------------
        kind, label = app._detection_source({"detection": {}}, demo_active=False)
        check("empty detection config -> none", kind == "none", kind)
        check("none label explains itself", "no trained model" in label, label)
        kind, _ = app._detection_source(
            {"detection": {"model_dir": str(tmp / "not_exported")}},
            demo_active=False)
        check("model_dir without model.pdmodel -> none", kind == "none", kind)

        # --- 7. _upload_ident ---------------------------------------------------
        check("upload ident None -> None", app._upload_ident(None) is None)

        class FakeUpload:
            def __init__(self, name: str, size: int) -> None:
                self.name, self.size = name, size

        ident = app._upload_ident(FakeUpload("board.jpg", 12345))
        check("upload ident (name, size)", ident == ("board.jpg", 12345),
              str(ident))
        check("same file same ident",
              app._upload_ident(FakeUpload("board.jpg", 12345)) == ident)
        check("different size -> different ident",
              app._upload_ident(FakeUpload("board.jpg", 999)) != ident)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Headless tests for the Production page's pure logic (training mode).

Run with the conda env:  D:\\miniforge3\\envs\\aoi-app\\python.exe scripts/test_training_mode.py

No Streamlit context needed — app.py is imported as a module (st is imported
but no st.* calls are made by the helpers under test).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime
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
    tmp = Path(tempfile.mkdtemp(prefix="aoi_training_test_"))
    try:
        data_root = tmp / "data"
        results_dir = tmp / "results"
        results_dir.mkdir(parents=True)

        # --- 1. session id generation ---------------------------------------
        sid = app._new_session_id(datetime(2026, 9, 13, 14, 30, 22))
        check("session id format", sid == "20260913_143022", sid)
        sid_now = app._new_session_id()
        check("session id default now", len(sid_now) == 15 and "_" in sid_now, sid_now)

        # --- 2. pending capture save ------------------------------------------
        img_bytes = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
        cap = app.save_pending_capture(img_bytes, ".jpg", data_root, sid)
        check("pending capture written", cap.is_file()
              and cap.read_bytes() == img_bytes
              and cap.parent == data_root / "raw" / "training" / sid, str(cap))
        cap2 = app.save_pending_capture(img_bytes, ".jpg", data_root, sid)
        check("pending capture collision-safe", cap2.is_file() and cap2 != cap,
              f"{cap.name} vs {cap2.name}")
        cap3 = app.save_pending_capture(img_bytes, ".gif", data_root, sid)
        check("non-image suffix coerced to .jpg", cap3.suffix == ".jpg", cap3.name)

        # --- 3. NG ledger record carries the extended fields -------------------
        dest_ng = app.label_image(
            cap, "NG", data_root, session_tag=f"{sid}_DEMO-REV-A",
            ledger_extra={"session": sid, "variant": "DEMO-REV-A",
                          "defect_type": "missing part", "refdes": "R7",
                          "origin": "training_mode"},
        )
        ledger = app._labels_ledger_path(data_root)
        rec = json.loads(ledger.read_text(encoding="utf-8").strip().splitlines()[-1])
        check("NG dest in boards_ng",
              dest_ng.parent == data_root / "boards_ng" and dest_ng.is_file(),
              dest_ng.name)
        check("NG record defect_type", rec.get("defect_type") == "missing part", str(rec.get("defect_type")))
        check("NG record refdes", rec.get("refdes") == "R7", str(rec.get("refdes")))
        check("NG record variant", rec.get("variant") == "DEMO-REV-A")
        check("NG record session", rec.get("session") == sid)
        check("NG record origin", rec.get("origin") == "training_mode")
        check("NG record base fields",
              rec.get("action") == "label" and rec.get("label") == "NG"
              and "timestamp" in rec and rec.get("source_path") == str(cap)
              and rec.get("saved_path") == str(dest_ng))
        check("NG filename session-prefixed",
              dest_ng.name.startswith(f"{sid}_DEMO-REV-A_"), dest_ng.name)

        # --- 4. OK label without extras stays old-format ----------------------
        dest_ok = app.label_image(cap2, "OK", data_root, session_tag="uploads")
        rec_ok = json.loads(ledger.read_text(encoding="utf-8").strip().splitlines()[-1])
        check("OK dest in boards_ok", dest_ok.parent == data_root / "boards_ok")
        check("old-format record has no new fields",
              all(k not in rec_ok for k in ("session", "variant", "defect_type",
                                            "refdes", "origin")))

        # --- 5. collision-safe naming on labeling ------------------------------
        dest_ok2 = app.label_image(cap2, "OK", data_root, session_tag="uploads")
        check("label copy collision-safe", dest_ok2 != dest_ok and dest_ok2.is_file(),
              f"{dest_ok.name} vs {dest_ok2.name}")

        # --- 6. session counts derive from the ledger --------------------------
        counts = app.training_session_counts(ledger, sid)
        check("session counts", counts == {"OK": 0, "NG": 1}, str(counts))

        # --- 7. audit-merge helper reads old + new lines ------------------------
        old_line = {"timestamp": "2026-09-01T09:00:00", "action": "label",
                    "source_path": "/old/cap.jpg", "saved_path": "/old/ok.jpg",
                    "label": "OK"}
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(old_line) + "\n")
        rows = app._audit_rows(data_root, results_dir)
        label_rows = [r for r in rows if r["source"] == "Labeling"]
        # 3 labels written above (1 NG + 2 OK) + 1 hand-appended old-format line
        check("audit rows include old + new records", len(label_rows) == 4,
              f"{len(label_rows)} rows")
        check("audit rows parse extended record fine",
              any(r["detail"] == "label: NG" for r in label_rows))
        check("load_label_state handles mixed ledger",
              app.load_label_state(ledger, [str(cap)])[str(cap)]["label"] == "NG")

        # --- 8. golden save writes the file; original restored -------------------
        real_golden = PROJECT_ROOT / "data" / "golden" / "golden_board.jpg"
        backup = tmp / "golden_backup.jpg"
        shutil.copy2(real_golden, backup)
        try:
            app.save_golden_image(cap, real_golden)
            check("golden save writes the file",
                  real_golden.read_bytes() == img_bytes)
        finally:
            shutil.copy2(backup, real_golden)  # restore the original golden
        check("original golden restored",
              real_golden.read_bytes() == backup.read_bytes())
        # temp-dir golden path resolution (fresh dir created on demand)
        golden_tmp = tmp / "fresh" / "golden" / "golden_board.jpg"
        app.save_golden_image(cap, golden_tmp)
        check("golden save creates parent dirs", golden_tmp.is_file())

        # --- 9. export zip still builds with extended ledger --------------------
        blob = app.build_dataset_zip(data_root)
        check("dataset zip builds", len(blob) > 100 and blob[:2] == b"PK",
              f"{len(blob)} bytes")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())

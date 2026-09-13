"""SG-AOI operator UI (Streamlit) — implements docs/ui_design.md + Phase A
professional UI (docs/ui_proposal_sg_aoi.md §9, Season Group branding).

Ten pages in a sectioned, role-filtered sidebar navigation:
  RUN:            Inspection & Training (mode toggle: Inspection / Training),
                  Review & Repair (was "Review History")
  MONITOR:        Dashboard (FPY / Pareto / NG feed / station status), SPC
  BUILD:          Dataset Review (was "Labeling"), Dataset & Training
  ADMINISTRATION: Audit Trail, Settings
  MAINTENANCE:    System Check
  (last)          Setup Wizard

Role simulation (no authentication — demo only): Operator sees RUN,
Engineer sees RUN+MONITOR+BUILD, Admin sees everything; Setup Wizard always.

Design rules honored here:
* All heavy imports (cv2, yaml, the pipeline module) are lazy inside functions
  so the app loads — and shows guidance — before any model is trained.
* Every state degrades explicitly: missing models, failed alignment and
  partial-branch runs are surfaced as banners, never as tracebacks.
* Verdict styling follows docs/ui_design.md: #22c55e OK / #ef4444 NG /
  #f59e0b partial-or-not-ready, large full-width banner.

TODO (future HMI, cannot be done in Streamlit — see ui_design.md §6/§7):
* Audible alert on NG (browsers block autoplay audio).
* Live camera stream (Streamlit offers snapshot input only).
* Real i18n — the EN/中文 selector below is a placeholder only.

Run:  streamlit run src/app.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

# Make sibling modules (infer_pipeline, align_board) importable no matter how
# Streamlit was launched.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Default pipeline config; override with the AOI_CONFIG env var (absolute or
# project-root-relative path) — e.g. AOI_CONFIG=configs/pipeline.demo.yaml on
# cloud hosts so the management demo boots on the demo config by default.
_config_env = os.environ.get("AOI_CONFIG", "configs/pipeline.yaml")
DEFAULT_CONFIG = Path(_config_env)
if not DEFAULT_CONFIG.is_absolute():
    DEFAULT_CONFIG = PROJECT_ROOT / DEFAULT_CONFIG

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

COLOR_OK = "#22c55e"
COLOR_NG = "#ef4444"
COLOR_WARN = "#f59e0b"

# --- Season Group brand chrome (ui_proposal_sg_aoi.md §2/§8) -----------------
# Brand colors are for chrome only (sidebar, accents); verdict semantics stay
# on the industry-standard green/red/amber above.
SG_CORAL = "#FB6362"
SG_SURFACE = "#343741"
BRAND_DIR = PROJECT_ROOT / "brand"
LOGO_LIGHT = BRAND_DIR / "logo.webp"  # white variant — for the dark UI
LINE_NAME = "Season Group · PCBA Line 1"

# --- Sectioned navigation + role simulation (demo only, no authentication) ---
NAV_SECTIONS: list[tuple[str, list[str]]] = [
    ("RUN", ["Inspection & Training", "Review & Repair"]),
    ("MONITOR", ["Dashboard", "SPC"]),
    ("BUILD", ["Dataset Review", "Dataset & Training"]),
    ("ADMINISTRATION", ["Audit Trail", "Settings"]),
    ("MAINTENANCE", ["System Check"]),
]
SETUP_PAGE = "Setup Wizard"
ROLES = ["Engineer", "Operator", "Admin"]
ROLE_SECTIONS = {
    "Operator": {"RUN"},
    "Engineer": {"RUN", "MONITOR", "BUILD"},
    "Admin": {name for name, _ in NAV_SECTIONS},
}

# Placeholder only — no translation tables yet (see module docstring TODO).
LANGUAGES = {"EN": "English", "中文": "Chinese (placeholder)"}

# --- Demo mode ---------------------------------------------------------------
# Self-contained management demo: synthetic board images + scripted detections
# under data/demo/ (regenerate with scripts/make_demo_board.py) drive the REAL
# rule engine — no trained model required. Activated by AOI_DEMO=1 or
# auto-detected whenever the demo assets exist. Demo scenario selection only
# overrides the config in memory; configs/pipeline.yaml is never rewritten.
DEMO_DIR = PROJECT_ROOT / "data" / "demo"
DEMO_SCENARIOS = {
    "OK board": ("board_ok.jpg", "detections_ok.json"),
    "NG — missing part (R7)": ("board_ng_missing.jpg", "detections_ng_missing.json"),
    "NG — wrong part (C3)": ("board_ng_wrong.jpg", "detections_ng_wrong.json"),
}


def _demo_active() -> bool:
    return os.environ.get("AOI_DEMO") == "1" or (
        DEMO_DIR / "detections_ok.json"
    ).is_file()


def _demo_cfg(cfg: dict[str, Any], detections_name: str) -> dict[str, Any]:
    """In-memory copy of the config repointed at a scenario's detections."""
    import copy  # noqa: PLC0415

    demo_cfg = copy.deepcopy(cfg)
    demo_cfg.setdefault("detection", {})["precomputed_json"] = str(
        DEMO_DIR / detections_name
    )
    return demo_cfg


# ---------------------------------------------------------------------------
# Generic helpers (no heavy imports at module level)
# ---------------------------------------------------------------------------

def _resolve(path_str: str | Path | None) -> Path | None:
    """Resolve a config-relative path against the project root."""
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _load_cfg(config_path: Path) -> dict[str, Any] | None:
    """Load pipeline.yaml; returns None (with a warning shown) on failure."""
    try:
        import yaml  # noqa: PLC0415 - lazy, keeps app importable without pyyaml
    except ImportError:
        st.error("PyYAML is not installed in this environment (`pip install pyyaml`).")
        return None
    try:
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - surface parse errors, don't crash
        st.warning(f"Could not parse {config_path.name}: {exc}")
        return None


def _model_status(cfg: dict[str, Any]) -> dict[str, Any]:
    """Which branches have usable artifacts — pure filesystem check."""
    det = cfg.get("detection", {})
    ano = cfg.get("anomaly", {})
    det_dir = _resolve(det.get("model_dir"))
    ano_path = _resolve(ano.get("model_path"))
    det_model = (det_dir / "model.pdmodel") if det_dir else None
    return {
        "detection_ready": bool(det.get("precomputed_json"))
        or bool(det_model and det_model.is_file()),
        "anomaly_ready": bool(ano_path and ano_path.exists()),
        "detection_path": det_model,
        "anomaly_path": ano_path,
    }


def _detection_source(cfg: dict[str, Any],
                      demo_active: bool | None = None) -> tuple[str, str]:
    """Which detection backend would actually drive the verdict right now.

    Returns (kind, label) with kind in {"model", "demo", "precomputed", "none"}
    and label a one-line status string shown next to the verdict so operators
    can see whether OK/NG comes from the trained model or from simulated data.
    Precedence mirrors infer_pipeline.run_detection: a configured
    precomputed_json wins over an exported model; demo mode repoints
    precomputed_json at the scripted scenario detections.

    Pure function (no st.*) — exercised headlessly by
    scripts/test_inspection_flow.py.
    """
    if demo_active is None:
        demo_active = _demo_active()
    det = cfg.get("detection", {})
    if demo_active:
        return "demo", "Detection source: DEMO — simulated detections"
    pre = _resolve(det.get("precomputed_json"))
    if det.get("precomputed_json") and pre is not None and pre.is_file():
        return "precomputed", "Detection source: precomputed JSON (debug)"
    det_dir = _resolve(det.get("model_dir"))
    if det_dir and (det_dir / "model.pdmodel").is_file():
        model_ref = det.get("model_dir") or str(det_dir)
        return "model", f"Detection source: trained model ({model_ref})"
    return "none", ("Detection source: none — no trained model and no "
                    "detections configured")


def _upload_ident(uploaded: Any) -> tuple[str, int] | None:
    """Stable identity (name, size) for an UploadedFile across reruns.

    st.camera_input / st.file_uploader re-return the same file on every rerun;
    comparing identities lets the inspection page tell "a NEW snapshot
    arrived" apart from "the same widget value again".
    """
    if uploaded is None:
        return None
    return (str(getattr(uploaded, "name", "?")), int(getattr(uploaded, "size", 0)))


def _count_images(folder: Path | None, recursive: bool = True) -> int:
    if not folder or not folder.is_dir():
        return 0
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sum(1 for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _append_feedback(results_dir: Path, record: dict[str, Any]) -> Path:
    """Append one feedback record to results/feedback.jsonl."""
    results_dir.mkdir(parents=True, exist_ok=True)
    record = {**record, "timestamp": datetime.now().isoformat(timespec="seconds")}
    path = results_dir / "feedback.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return path


def _results_dir(cfg: dict[str, Any]) -> Path:
    return _resolve(cfg.get("output", {}).get("results_dir", "results")) or (
        PROJECT_ROOT / "results"
    )


def _results_writable(results_dir: Path) -> tuple[bool, str]:
    """Write+delete a probe file; returns (ok, error-message)."""
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        probe = results_dir / ".write_probe_tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _inject_brand_css() -> None:
    """Season Group chrome: Noto Sans, coral sidebar section labels, metric
    cards. Verdict colors are NOT touched — they stay per docs/ui_design.md."""
    st.markdown(
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?family=Noto+Sans:"
        "wght@400;600;700;800&display=swap');"
        "html, body, .stApp, [class*='css'] {"
        "font-family:'Noto Sans',-apple-system,'Segoe UI',Roboto,sans-serif;}"
        f".sg-section-label{{color:{SG_CORAL};font-size:11px;font-weight:700;"
        "letter-spacing:2px;text-transform:uppercase;margin:14px 0 2px 0;"
        "opacity:0.9;}"
        "[data-testid='stMetric']{"
        f"background-color:{SG_SURFACE};border:1px solid rgba(255,255,255,0.10);"
        "border-radius:12px;padding:12px 16px;}"
        "hr{border-color:rgba(255,255,255,0.10);}"
        "</style>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Analytics helpers — pure functions over results/ + data/ (no st.*), so they
# can be exercised headlessly. Verdict JSONs carry no timestamp field; the
# file mtime is the inspection time (same convention as Review & Repair).
# ---------------------------------------------------------------------------

def _load_verdict_records(results_dir: Path) -> list[dict[str, Any]]:
    """All *_verdict.json files as records sorted by inspection time (mtime)."""
    records: list[dict[str, Any]] = []
    if not results_dir.is_dir():
        return records
    for vf in sorted(results_dir.glob("*_verdict.json")):
        try:
            data = json.loads(vf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - skip corrupt records, keep going
            continue
        records.append({
            "file": vf,
            "time": datetime.fromtimestamp(vf.stat().st_mtime),
            "board_id": data.get("board_id", vf.stem.replace("_verdict", "")),
            "verdict": data.get("verdict", "?"),
            "defects": data.get("defects", []) or [],
        })
    records.sort(key=lambda r: r["time"])
    return records


def _daily_fpy(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per calendar day: boards / ok / ng / FPY %."""
    days: dict[str, dict[str, Any]] = {}
    for r in records:
        key = r["time"].strftime("%Y-%m-%d")
        d = days.setdefault(key, {"date": key, "boards": 0, "ok": 0, "ng": 0})
        d["boards"] += 1
        if r["verdict"] == "OK":
            d["ok"] += 1
        elif r["verdict"] == "NG":
            d["ng"] += 1
    rows = [days[k] for k in sorted(days)]
    for d in rows:
        d["fpy"] = round(100.0 * d["ok"] / d["boards"], 1) if d["boards"] else 0.0
    return rows


# Reference designators look like R7, C3, U12 — extracted from defect detail
# text (e.g. "Expected R7 (resistor_axial) not detected.").
_DESIGNATOR_RE = re.compile(r"\b[A-Z]{1,3}\d+\b")


def _extract_designator(detail: str) -> str | None:
    m = _DESIGNATOR_RE.search(detail or "")
    return m.group(0) if m else None


def _defect_pareto(records: list[dict[str, Any]],
                   by_designator: bool = False) -> list[dict[str, Any]]:
    """Defect counts sorted desc; label is the type, or 'type — designator'."""
    counts: dict[str, int] = {}
    for r in records:
        for d in r["defects"]:
            dtype = d.get("type", "?")
            if by_designator:
                des = _extract_designator(d.get("detail", ""))
                label = f"{dtype} — {des}" if des else dtype
            else:
                label = dtype
            counts[label] = counts.get(label, 0) + 1
    return [{"defect": k, "count": v}
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]


def _pchart(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Daily NG proportion with p-chart center line and 3-sigma limits.

    p_i = NG boards / boards that day; p̄ = total NG / total boards;
    UCL/LCL = p̄ ± 3·sqrt(p̄(1−p̄)/n), n = boards that day (LCL clamped ≥ 0).
    """
    import math  # noqa: PLC0415

    days = _daily_fpy(records)
    total = sum(d["boards"] for d in days)
    total_ng = sum(d["ng"] for d in days)
    p_bar = total_ng / total if total else 0.0
    out: dict[str, Any] = {"dates": [], "p": [], "cl": [], "ucl": [], "lcl": [],
                           "n": [], "p_bar": p_bar}
    for d in days:
        n = d["boards"]
        sigma = math.sqrt(p_bar * (1 - p_bar) / n) if n else 0.0
        out["dates"].append(d["date"])
        out["n"].append(n)
        out["p"].append(d["ng"] / n if n else 0.0)
        out["cl"].append(p_bar)
        out["ucl"].append(min(1.0, p_bar + 3 * sigma))
        out["lcl"].append(max(0.0, p_bar - 3 * sigma))
    return out


def collect_system_checks(cfg: dict[str, Any] | None,
                          config_path: Path) -> list[dict[str, Any]]:
    """One-click self-diagnosis rows — pure filesystem checks, no st.* calls.

    Each row: {check, ok, detail, fix} where fix is a one-line "what to do if
    red" hint. Shares its logic with the Setup Wizard where both compute the
    same thing (golden image, expected components, model status).
    """
    rows: list[dict[str, Any]] = []

    def add(check: str, ok: bool, detail: str, fix: str) -> None:
        rows.append({"check": check, "ok": bool(ok), "detail": detail, "fix": fix})

    # 1. config parses
    eff_cfg: dict[str, Any] = cfg or {}
    try:
        import yaml  # noqa: PLC0415

        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not eff_cfg and isinstance(parsed, dict):
            eff_cfg = parsed
        add("Pipeline config parses", isinstance(parsed, dict),
            f"{config_path.name} loaded",
            "Fix the YAML syntax in configs/pipeline.yaml.")
    except Exception as exc:  # noqa: BLE001
        add("Pipeline config parses", False, f"{config_path.name}: {exc}",
            "Fix the YAML syntax in configs/pipeline.yaml.")

    # 2. golden image exists
    golden_img = _resolve(
        eff_cfg.get("golden", {}).get("image", "data/golden/golden_board.jpg"))
    add("Golden board image exists", bool(golden_img and golden_img.is_file()),
        str(golden_img),
        "Capture one known-good board into data/golden/ "
        "(see docs/data_collection_guide.md).")

    # 3. expected_components.json parses (count components)
    exp_path = _resolve(
        eff_cfg.get("golden", {}).get("expected_components",
                                      "data/golden/expected_components.json"))
    n_comp, exp_ok = 0, False
    if exp_path and exp_path.is_file():
        try:
            n_comp = len(json.loads(exp_path.read_text(encoding="utf-8"))
                         .get("components", []))
            exp_ok = True
        except Exception:  # noqa: BLE001
            exp_ok = False
    add("expected_components.json parses", exp_ok and n_comp > 0,
        f"{n_comp} components defined" if exp_ok else "missing or invalid JSON",
        "Define the expected component list (format in data/README.md).")

    # 4. detection model OR demo precomputed detections
    status = _model_status(eff_cfg)
    if _demo_active():
        add("Detection available", True,
            "demo mode — scripted detections (no trained model required)",
            "Train & export the PP-YOLOE+ detector for production use.")
    else:
        add("Detection model exported", status["detection_ready"],
            str(status["detection_path"]),
            "Run tools/export_model.py in the PaddleDetection repo "
            "(configs/detection/README.md).")

    # 5. anomaly model — deferred in phase 1, never a failure
    add("Anomaly model (Branch B)", True,
        "present" if status["anomaly_ready"] else "deferred (phase 1 — OK)",
        "Not required in phase 1; train only when Branch B is activated.")

    # 6. results dir writable (write + delete a temp probe file)
    rdir = _results_dir(eff_cfg)
    w_ok, w_err = _results_writable(rdir)
    add("Results directory writable", w_ok, str(rdir) if w_ok else f"{rdir}: {w_err}",
        "Check folder permissions on results/.")

    # 7. free disk space (warn below 1 GB)
    try:
        free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024 ** 3)
        add("Free disk space ≥ 1 GB", free_gb >= 1.0, f"{free_gb:.1f} GB free",
            "Free up disk space on this station.")
    except Exception as exc:  # noqa: BLE001
        add("Free disk space ≥ 1 GB", False, str(exc), "Could not read disk usage.")

    # 8. data folders present
    data_root = PROJECT_ROOT / "data"
    missing = [name for name in ("boards_ok", "boards_ng", "golden", "raw")
               if not (data_root / name).is_dir()]
    add("Data folders present", not missing,
        "all present" if not missing else "missing: " + ", ".join(missing),
        "Create the missing folders under data/.")

    # 9. labels ledger line count (informational — never red)
    ledger = _labels_ledger_path(data_root)
    if ledger.is_file():
        n_lines = sum(1 for line in ledger.read_text(encoding="utf-8").splitlines()
                      if line.strip())
        add("Labels ledger", True, f"{n_lines} entries in data/labels.jsonl",
            "Nothing to do.")
    else:
        add("Labels ledger", True,
            "no labels.jsonl yet — normal before the first labeling session",
            "Nothing to do — the ledger is created by Dataset Review or training mode.")

    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL ledger, skipping blank/corrupt lines."""
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _audit_rows(data_root: Path, results_dir: Path) -> list[dict[str, Any]]:
    """Merge labels.jsonl + feedback.jsonl into one timeline, newest first."""
    rows: list[dict[str, Any]] = []
    for rec in _read_jsonl(_labels_ledger_path(data_root)):
        target = (rec.get("saved_path") or rec.get("saved_path_to")
                  or rec.get("source_path", ""))
        rows.append({
            "timestamp": rec.get("timestamp", ""),
            "source": "Labeling",
            "action": rec.get("action", "label"),
            "target": Path(target).name if target else "—",
            "detail": f"label: {rec.get('label', '?')}",
        })
    for rec in _read_jsonl(results_dir / "feedback.jsonl"):
        rows.append({
            "timestamp": rec.get("timestamp", ""),
            "source": "Inspection feedback",
            "action": rec.get("mark", "?"),
            "target": rec.get("board_id", "—"),
            "detail": rec.get("reason", "—"),
        })
    rows.sort(key=lambda r: r["timestamp"], reverse=True)
    return rows


def _verdict_banner(verdict: str, subtitle: str, color: str) -> None:
    """Full-width, 2-meter-readable verdict banner (ui_design.md §2.1/§4)."""
    st.markdown(
        f'<div style="background-color:{color};padding:28px 16px;border-radius:12px;'
        f'text-align:center;color:#ffffff;margin-bottom:12px;">'
        f'<span style="font-size:64px;font-weight:800;line-height:1.1;">{verdict}</span><br/>'
        f'<span style="font-size:20px;">{subtitle}</span></div>',
        unsafe_allow_html=True,
    )


def _defect_rows(defects: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": d.type,
            "branch": d.branch,
            "confidence": round(d.confidence, 3),
            "location (x,y,w,h)": [round(v) for v in d.location],
            "detail": d.detail,
        }
        for d in defects
    ]


# ---------------------------------------------------------------------------
# Labeling helpers — pure filesystem/ledger operations. They take explicit
# paths and never touch `st`, so they can be exercised headlessly (tests,
# scripts) without a running Streamlit context. Only `page_labeling` calls
# them from the UI.
# ---------------------------------------------------------------------------

LABEL_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _labels_ledger_path(data_root: Path) -> Path:
    return data_root / "labels.jsonl"


def _append_label_ledger(record: dict[str, Any], ledger_path: Path) -> Path:
    """Append one JSON line to the label ledger. History is never rewritten."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **record}
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return ledger_path


def _label_target_dir(label: str, data_root: Path) -> Path:
    return data_root / ("boards_ok" if label == "OK" else "boards_ng")


def _collision_safe_name(dest_dir: Path, name: str) -> str:
    """A filename that does not yet exist in dest_dir (name, name_2, ...)."""
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate, n = name, 2
    while (dest_dir / candidate).exists():
        candidate = f"{stem}_{n}{suffix}"
        n += 1
    return candidate


def _sanitize_tag(text: str) -> str:
    """Filesystem-safe session tag for filename prefixes."""
    tag = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in text)
    return tag.strip("_") or "session"


def save_uploaded_images(files: list[Any], data_root: Path) -> Path:
    """Persist st.file_uploader files into data/raw/uploads/<timestamped-session>/.

    Returns the session directory; raw uploads stay untouched by labeling.
    """
    session_dir = data_root / "raw" / "uploads" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = _collision_safe_name(session_dir, Path(f.name).name)
        (session_dir / name).write_bytes(f.getbuffer())
    return session_dir


def label_image(src_path: Path, label: str, data_root: Path, session_tag: str = "",
                ledger_extra: dict[str, Any] | None = None) -> Path:
    """COPY (not move) an image into boards_ok/boards_ng and append a ledger line.

    `ledger_extra` merges optional fields into the ledger record (training mode
    adds session / variant / defect_type / refdes / origin). The ledger is
    append-only JSONL, so records with extra fields stay backward compatible —
    older lines simply lack them.
    """
    src_path = Path(src_path)
    dest_dir = _label_target_dir(label, data_root)
    dest_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_sanitize_tag(session_tag)}_" if session_tag else ""
    dest = dest_dir / _collision_safe_name(dest_dir, prefix + src_path.name)
    shutil.copy2(src_path, dest)
    record: dict[str, Any] = {"action": "label", "source_path": str(src_path),
                              "saved_path": str(dest), "label": label}
    if ledger_extra:
        record.update(ledger_extra)
    _append_label_ledger(record, _labels_ledger_path(data_root))
    return dest


def relabel_image(saved_path: Path, new_label: str, data_root: Path,
                  source_path: str = "") -> Path:
    """Move a labeled image between boards_ok/boards_ng; append a correction line."""
    saved_path = Path(saved_path)
    dest_dir = _label_target_dir(new_label, data_root)
    if saved_path.parent.resolve() == dest_dir.resolve():
        return saved_path  # already in the right folder — nothing to do
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / _collision_safe_name(dest_dir, saved_path.name)
    shutil.move(str(saved_path), str(dest))
    _append_label_ledger(
        {"action": "relabel", "source_path": source_path,
         "saved_path_from": str(saved_path), "saved_path_to": str(dest),
         "label": new_label},
        _labels_ledger_path(data_root),
    )
    return dest


def load_label_state(ledger_path: Path, source_paths: list[str]) -> dict[str, dict[str, str]]:
    """Reconstruct {source_path: {label, saved_path}} for a working set from the
    ledger, applying relabel corrections in order."""
    sources = {str(p) for p in source_paths}
    state: dict[str, dict[str, str]] = {}
    if not Path(ledger_path).is_file():
        return state
    for line in Path(ledger_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip corrupt lines, keep browsing
        sp = rec.get("source_path", "")
        if rec.get("action") == "label" and sp in sources:
            state[sp] = {"label": rec.get("label", ""),
                         "saved_path": rec.get("saved_path", "")}
        elif rec.get("action") == "relabel" and sp in state:
            state[sp] = {"label": rec.get("label", ""),
                         "saved_path": rec.get("saved_path_to", "")}
    return state


def build_dataset_zip(data_root: Path) -> bytes:
    """Zip boards_ok/ + boards_ng/ + labels.jsonl into an in-memory archive."""
    import io  # noqa: PLC0415
    import zipfile  # noqa: PLC0415

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in ("boards_ok", "boards_ng"):
            base = data_root / folder
            if base.is_dir():
                for p in sorted(base.rglob("*")):
                    if p.is_file() and p.name != ".gitkeep":
                        zf.write(p, f"{folder}/{p.relative_to(base).as_posix()}")
        ledger = _labels_ledger_path(data_root)
        if ledger.is_file():
            zf.write(ledger, "labels.jsonl")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Training-mode helpers (Inspection & Training page) — pure functions with no
# st.* calls, so they can be exercised headlessly (scripts/test_training_mode.py).
# ---------------------------------------------------------------------------

def _new_session_id(now: datetime | None = None) -> str:
    """Timestamp id for one training-mode entry (e.g. '20260913_143022')."""
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def save_pending_capture(data: bytes, suffix: str, data_root: Path,
                         session_id: str) -> Path:
    """Persist one captured/uploaded image under data/raw/training/<session>/.

    Writing immediately (instead of holding bytes only in session_state) keeps
    the pending preview stable across reruns and gives the ledger a real
    source_path. Raw captures stay untouched by labeling — labeling copies.
    """
    session_dir = data_root / "raw" / "training" / _sanitize_tag(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    if suffix.lower() not in LABEL_EXTENSIONS:
        suffix = ".jpg"
    name = _collision_safe_name(session_dir, f"capture_{datetime.now():%H%M%S}{suffix}")
    path = session_dir / name
    path.write_bytes(data)
    return path


def training_session_counts(ledger_path: Path, session_id: str) -> dict[str, int]:
    """OK/NG label counts recorded in the ledger for one training session."""
    counts = {"OK": 0, "NG": 0}
    for rec in _read_jsonl(ledger_path):
        if rec.get("action") == "label" and rec.get("session") == session_id:
            if rec.get("label") in counts:
                counts[rec["label"]] += 1
    return counts


def save_golden_image(src_path: Path, golden_path: Path) -> Path:
    """Overwrite the golden board image with a new capture.

    The caller (training mode UI) is responsible for the warning that a REAL
    golden board also requires rebuilding expected_components.json — the rule
    engine's source of truth.
    """
    golden_path = Path(golden_path)
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, golden_path)
    return golden_path


# ---------------------------------------------------------------------------
# Page 1: Inspection & Training — mode toggle: Inspection / Training
# ---------------------------------------------------------------------------

def page_inspection_training(cfg: dict[str, Any]) -> None:
    st.header("Inspection & Training")
    mode = st.radio("Mode", ["Inspection", "Training"], horizontal=True,
                    key="inspection_training_mode")

    # Entering Training mode starts a fresh capture session: new session id,
    # no leftover pending image from a previous entry.
    prev_mode = st.session_state.get("_it_prev_mode")
    if mode == "Training" and prev_mode != "Training":
        st.session_state["training_session_id"] = _new_session_id()
        st.session_state.pop("training_pending", None)
    st.session_state["_it_prev_mode"] = mode

    if mode == "Training":
        _training_mode(cfg)
    else:
        _inspection_mode(cfg)


def _inspection_mode(cfg: dict[str, Any]) -> None:
    """Two-step inspection flow (docs/ui_design.md §11):

    Step 1 "Board image" — 📷 Take snapshot (st.camera_input, works locally
    and on Streamlit Cloud) with the file uploader as fallback; the most
    recent image from either source is held in session_state as the CURRENT
    SNAPSHOT (survives reruns) until "🗑 Clear".
    Step 2 "🔍 Inspection" — runs the pipeline on the current snapshot and
    renders the verdict banner + annotated image + defect table, always with
    a verdict-source indicator (trained model / demo / precomputed JSON).
    """

    # --- demo mode (simulated detections, no trained model) -------------------
    demo_image: Path | None = None
    demo_on = _demo_active()
    if demo_on:
        st.info(
            "🧪 **DEMO MODE — simulated detections (no trained model).** "
            "The verdict below is produced by the real rule engine from "
            "scripted detections and the golden expected-components list."
        )
        scenario = st.radio("Demo scenario", list(DEMO_SCENARIOS), horizontal=True)
        # A scenario switch invalidates any verdict from the previous scenario.
        if st.session_state.get("demo_scenario") != scenario:
            st.session_state.pop("inspection", None)
            st.session_state["demo_scenario"] = scenario
        image_name, detections_name = DEMO_SCENARIOS[scenario]
        cfg = _demo_cfg(cfg, detections_name)  # in-memory override only
        demo_image = DEMO_DIR / image_name

    status = _model_status(cfg)
    det_ready, ano_ready = status["detection_ready"], status["anomaly_ready"]
    source_kind, source_label = _detection_source(cfg, demo_active=demo_on)

    # Explicit degradation banners (ui_design.md §5) — amber, never silent.
    if not det_ready:
        st.info(
            "⚠️ Detection model not trained/exported — Branch A checks "
            "(missing / wrong part) will be skipped and the verdict is NOT "
            "meaningful. Set `detection.precomputed_json` to test the rule "
            "engine without a model."
        )
    if not ano_ready:
        # Expected phase-1 state: the anomaly branch is deferred — a quiet
        # note, not a warning (ui_design.md §5).
        st.caption(
            "ℹ️ Anomaly branch (scratches / solder balls / foreign particles) "
            "is deferred and not active — this is expected in phase 1."
        )

    # --- Step 1: board image ---------------------------------------------------
    st.subheader("1. Board image")
    snapshot = st.camera_input(
        "📷 Take snapshot",
        help="Uses this device's camera through the browser — works on a "
             "local station and on Streamlit Cloud alike.",
    )
    uploaded = st.file_uploader(
        "…or upload a board image (fallback)",
        type=[e.lstrip(".") for e in sorted(IMAGE_EXTENSIONS)],
        key="inspection_upload",
    )
    # Whichever source delivered the most recent image becomes the current
    # snapshot. Identity markers keep the same widget value from re-winning
    # on every rerun.
    for origin, capture in (("upload", uploaded), ("camera", snapshot)):
        ident = _upload_ident(capture)
        if ident is not None and ident != st.session_state.get(f"_insp_seen_{origin}"):
            st.session_state[f"_insp_seen_{origin}"] = ident
            st.session_state["inspection_snapshot"] = {
                "name": str(getattr(capture, "name", f"{origin}.jpg")),
                "bytes": capture.getbuffer().tobytes(),
                "origin": origin,
            }
            st.session_state.pop("inspection", None)  # new image, old verdict stale

    current = st.session_state.get("inspection_snapshot")
    demo_fallback = demo_image is not None and demo_image.is_file()

    align_first = False
    golden_path: str | None = None
    if current:
        st.image(
            current["bytes"],
            caption=f"Current snapshot — {current['name']} "
                    f"(from {current['origin']})",
            use_container_width=True,
        )
        if demo_on:
            st.caption(
                "Demo detections are simulated for the demo board layout; "
                "the boxes shown correspond to the demo scenario, not to "
                "objects in your photo."
            )
        if st.button("🗑 Clear", key="inspection_clear"):
            for key in ("inspection_snapshot", "_insp_seen_camera",
                        "_insp_seen_upload", "inspection"):
                st.session_state.pop(key, None)
            st.rerun()
        align_first = st.checkbox(
            "Run board alignment first (raw capture, not yet aligned)", value=False
        )
        if align_first:
            golden_default = cfg.get("golden", {}).get(
                "image", "data/golden/golden_board.jpg")
            golden_path = st.text_input("Golden reference image", str(golden_default))
    elif demo_fallback:
        # Demo boards are already in the canonical aligned view — the capture
        # and alignment steps are replaced by the scripted scenario image.
        st.image(str(demo_image), caption=f"Demo board: {demo_image.name}",
                 use_container_width=True)
    else:
        st.caption("Waiting for a board image…")

    # --- Step 2: run -----------------------------------------------------------
    st.subheader("2. Inspection")
    have_image = bool(current) or demo_fallback
    if not have_image:
        st.info("Take a snapshot (or upload a board image) first.")
    elif st.button("🔍 Inspection", type="primary", use_container_width=True):
        with st.spinner("Running detection + anomaly branches…"):
            try:
                if current:
                    suffix = Path(current["name"]).suffix.lower()
                    if suffix not in IMAGE_EXTENSIONS:
                        suffix = ".jpg"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(current["bytes"])
                        input_path = Path(tmp.name)
                    if align_first:
                        if not _align_snapshot_in_place(input_path, golden_path, current):
                            return  # error banner already shown; no verdict
                else:
                    input_path = demo_image

                from infer_pipeline import inspect_board  # noqa: PLC0415

                run_cfg = _resolved_run_cfg(cfg)
                verdict, annotated = inspect_board(input_path, run_cfg)
                st.session_state["inspection"] = {
                    "verdict": verdict,
                    "annotated": annotated,
                    "cfg": run_cfg,
                    "source_label": source_label,
                }
                # Persist immediately so Review History sees it (no extra click).
                from infer_pipeline import save_outputs  # noqa: PLC0415

                save_outputs(verdict, annotated, run_cfg)
            except Exception as exc:  # noqa: BLE001 - operator UI must not crash
                st.error(f"Pipeline failed: {exc}")
                return

    state = st.session_state.get("inspection")
    if not state:
        if have_image:
            st.caption("Press **🔍 Inspection** to get a verdict.")
        return

    verdict = state["verdict"]
    n_defects = len(verdict.defects)

    # Phase-1 semantics: the anomaly branch is deferred. Branch B only counts
    # as "missing" if its checks are actually enabled in the pipeline config —
    # otherwise its absence is the expected state, not a degradation.
    checks = cfg.get("checks", {})
    branch_b_active = any(
        checks.get(k, False)
        for k in ("scratches", "solder_balls", "foreign_particles")
    )
    branch_b_missing = branch_b_active and not ano_ready

    # Verdict banner — degrade explicitly when no required branch was available.
    if not det_ready and not (ano_ready and branch_b_active):
        _verdict_banner(
            "NOT READY",
            "Detection model missing — verdict is not meaningful. Train/export "
            "the PP-YOLOE+ detector first.",
            COLOR_WARN,
        )
    elif verdict.verdict == "OK":
        note = " (partial: anomaly checks enabled but model missing)" if branch_b_missing else ""
        _verdict_banner("✅ OK", f"Board {verdict.board_id} passed{note}.", COLOR_OK)
    else:
        _verdict_banner(
            "❌ NG", f"Board {verdict.board_id} — {n_defects} defect(s) found.", COLOR_NG
        )
    if branch_b_missing:
        st.warning(
            "⚠️ PARTIAL verdict — anomaly checks are enabled in the config but "
            "the anomaly model is unavailable; findings reflect Branch A only."
        )

    # Verdict-source indicator (ui_design.md §11): the operator's mental model
    # is "the verdict comes from the pictures we trained the system with" —
    # until a trained model exists, say so explicitly next to every verdict.
    st.caption(state.get("source_label") or source_label)

    # --- annotated image + defect table ----------------------------------------
    import cv2  # noqa: PLC0415

    col_img, col_table = st.columns([3, 2])
    with col_img:
        st.image(
            cv2.cvtColor(state["annotated"], cv2.COLOR_BGR2RGB),
            caption="Annotated result (red = Branch A defects, amber = anomaly regions)",
            use_container_width=True,
        )
    with col_table:
        st.markdown("**Defects**")
        if verdict.defects:
            st.dataframe(_defect_rows(verdict.defects), use_container_width=True)
        else:
            st.caption("No defects found.")

    # --- NG override + next-board actions ---------------------------------------
    if verdict.verdict == "NG" and (det_ready or ano_ready):
        st.markdown("**NG handling** — only override if you have visually confirmed:")
        reason = st.selectbox(
            "Override reason",
            ["false alarm - lighting", "false alarm - position",
             "defect not real", "other"],
        )
        if st.button("⚠ Override to OK (logged)"):
            path = _append_feedback(
                _results_dir(cfg),
                {"board_id": verdict.board_id, "mark": "override_ok", "reason": reason},
            )
            st.success(f"Override logged to {path.name}.")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔁 Re-inspect", use_container_width=True):
            st.session_state.pop("inspection", None)
            st.rerun()
    with col_b:
        if st.button("✔ Confirm & next board", type="primary", use_container_width=True):
            st.session_state.pop("inspection", None)
            st.rerun()

    with st.expander("Raw verdict JSON"):
        st.json(json.loads(json.dumps(asdict(verdict))))


def _align_snapshot_in_place(input_path: Path, golden_path: str | None,
                             current: dict[str, Any]) -> bool:
    """Align the snapshot temp file against the golden reference.

    On success the aligned pixels are written back to input_path AND into the
    session-state snapshot (so the "Current snapshot" preview and any re-run
    use the aligned image). Returns False after showing an explicit error
    banner when alignment is impossible — no verdict is produced then
    (ui_design.md §5).
    """
    import cv2  # noqa: PLC0415

    from align_board import align_image  # noqa: PLC0415

    golden_file = _resolve(golden_path)
    image = cv2.imread(str(input_path))
    golden = cv2.imread(str(golden_file)) if golden_file and golden_file.is_file() else None
    if image is None or golden is None:
        st.error(
            "❌ Alignment not possible — could not read the snapshot or the "
            f"golden reference ({golden_file}). Check the path and retry."
        )
        return False
    result = align_image(image, golden)
    if result.warped is None:
        st.error(
            "❌ Board not detected / fiducials not found — reseat the board, "
            "check lighting, and retry. No verdict was produced."
        )
        return False
    cv2.imwrite(str(input_path), result.warped)
    current["bytes"] = input_path.read_bytes()
    st.session_state["inspection_snapshot"] = current
    st.success(f"Aligned via {result.method}.")
    return True


def _resolved_run_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Copy of the pipeline config with paths resolved against the project root.

    infer_pipeline resolves paths relative to the current working directory;
    the app resolves them against the project root so it works from anywhere.
    """
    import copy  # noqa: PLC0415

    run_cfg = copy.deepcopy(cfg)
    for section, key in (
        ("golden", "expected_components"),
        ("golden", "image"),
        ("detection", "model_dir"),
        ("anomaly", "model_path"),
    ):
        value = run_cfg.get(section, {}).get(key)
        if value:
            run_cfg[section][key] = str(_resolve(value))
    pre = run_cfg.get("detection", {}).get("precomputed_json")
    if pre:
        run_cfg["detection"]["precomputed_json"] = str(_resolve(pre))
    run_cfg.setdefault("output", {})["results_dir"] = str(_results_dir(cfg))
    return run_cfg


def _training_mode(cfg: dict[str, Any]) -> None:
    """Capture-time OK/NG labeling from the browser webcam (st.camera_input —
    works on cloud and local) or a single-image upload. Uses the SAME labeling
    helpers/ledger as the Dataset Review page, with extended ledger fields."""
    data_root = PROJECT_ROOT / "data"
    ledger_path = _labels_ledger_path(data_root)
    session_id = st.session_state["training_session_id"]
    _inject_labeling_css()  # shared OK-green / NG-red button styling

    # --- session fields -----------------------------------------------------
    col_variant, col_session = st.columns([2, 1])
    with col_variant:
        variant = st.text_input("Board variant", value="DEMO-REV-A",
                                key="training_variant")
    with col_session:
        st.text_input("Session id", value=session_id, disabled=True,
                      help="Generated once per training-mode entry; recorded on "
                           "every label saved in this session.")

    st.caption(
        "OK/NG board labels triage the dataset — detector training still needs "
        "component-level box annotation (see docs/data_collection_guide.md)."
    )
    st.caption(
        "On the cloud demo, labeled images vanish on redeploy — export the zip "
        "below; run the local app for real data collection."
    )

    # --- pending image: capture → preview → label -----------------------------
    pending = st.session_state.get("training_pending")
    if pending and not Path(pending).is_file():
        st.session_state.pop("training_pending", None)  # file vanished externally
        pending = None

    if pending is None:
        snapshot = st.camera_input("Capture board")
        uploaded = st.file_uploader(
            "…or upload a board image",
            type=sorted(e.lstrip(".") for e in LABEL_EXTENSIONS),
            key="training_upload",
        )
        capture = snapshot or uploaded
        if capture is not None:
            suffix = Path(getattr(capture, "name", "capture.jpg")).suffix or ".jpg"
            path = save_pending_capture(capture.getbuffer(), suffix, data_root,
                                        session_id)
            st.session_state["training_pending"] = str(path)
            st.rerun()
        st.caption("Waiting for a board image…")
    else:
        pending_path = Path(pending)
        st.image(str(pending_path), caption=f"Pending — {pending_path.name}",
                 use_container_width=True)

        defect_type = st.selectbox(
            "Defect type (for NG)",
            ["missing part", "wrong part", "other", "unspecified"],
            index=3,
            help="Recorded with NG labels only; ignored for OK.",
        )
        refdes = st.text_input(
            "Reference designator (e.g. R7)", key="training_refdes",
            help="Optional — recorded with NG labels only.",
        )

        def _apply_training_label(label: str) -> None:
            extra: dict[str, Any] = {"session": session_id, "variant": variant,
                                     "origin": "training_mode"}
            if label == "NG":
                extra["defect_type"] = defect_type
                if refdes.strip():
                    extra["refdes"] = refdes.strip()
            dest = label_image(pending_path, label, data_root,
                               session_tag=f"{session_id}_{variant}",
                               ledger_extra=extra)
            st.session_state.pop("training_pending", None)
            st.session_state["training_last_save"] = (
                f"{label} → {dest.name} — ready for the next board.")

        col_ok, col_ng, col_retake = st.columns([3, 3, 2])
        with col_ok:
            st.markdown('<span class="aoi-ok-btn"></span>', unsafe_allow_html=True)
            if st.button("✅ OK", use_container_width=True, key="training_ok"):
                _apply_training_label("OK")
                st.rerun()
        with col_ng:
            st.markdown('<span class="aoi-ng-btn"></span>', unsafe_allow_html=True)
            if st.button("❌ NG", use_container_width=True, key="training_ng"):
                _apply_training_label("NG")
                st.rerun()
        with col_retake:
            if st.button("↺ Retake", use_container_width=True, key="training_retake"):
                pending_path.unlink(missing_ok=True)  # discard the unlabeled capture
                st.session_state.pop("training_pending", None)
                st.rerun()

    last_save = st.session_state.pop("training_last_save", None)
    if last_save:
        st.success(last_save)

    # --- progress ---------------------------------------------------------------
    counts = training_session_counts(ledger_path, session_id)
    cols = st.columns(4)
    cols[0].metric("✅ OK (this session)", counts["OK"])
    cols[1].metric("❌ NG (this session)", counts["NG"])
    cols[2].metric("boards_ok total", _count_images(data_root / "boards_ok"))
    cols[3].metric("boards_ng total", _count_images(data_root / "boards_ng"))

    # --- golden board capture -----------------------------------------------------
    with st.expander("⚠ Capture as golden board"):
        st.warning(
            "**This overwrites `data/golden/golden_board.jpg`.** The golden "
            "image is only half of the golden reference — the rule engine's "
            "source of truth is `data/golden/expected_components.json`. A REAL "
            "golden board therefore requires REBUILDING "
            "`expected_components.json` (reference designators, classes, "
            "bboxes) for the new board; otherwise every inspection compares "
            "the new image against the OLD component list."
        )
        golden_confirm = st.checkbox(
            "I understand this overwrites the golden board image and that "
            "expected_components.json must be rebuilt for a new board.",
            key="golden_confirm",
        )
        if st.button("Save pending image as golden board",
                     disabled=not (golden_confirm and pending),
                     use_container_width=True, key="golden_save"):
            golden_path = _resolve(
                cfg.get("golden", {}).get("image", "data/golden/golden_board.jpg")
            ) or (PROJECT_ROOT / "data" / "golden" / "golden_board.jpg")
            save_golden_image(Path(pending), golden_path)
            st.success(
                f"Golden board image saved to {golden_path}. Reminder: rebuild "
                "`expected_components.json` for this board before trusting "
                "inspection verdicts (format in data/README.md)."
            )

    # --- export ----------------------------------------------------------------------
    st.subheader("Export")
    st.download_button(
        "⬇ Download labeled dataset (.zip)",
        data=build_dataset_zip(data_root),
        file_name=f"pcba_aoi_labeled_dataset_{datetime.now():%Y%m%d_%H%M%S}.zip",
        mime="application/zip",
        use_container_width=True,
        key="training_export",
    )


# ---------------------------------------------------------------------------
# Page 2: Review & Repair (was "Review History")
# ---------------------------------------------------------------------------

def page_history(cfg: dict[str, Any]) -> None:
    st.header("Review & Repair")
    results_dir = _results_dir(cfg)
    verdict_files = sorted(
        results_dir.glob("*_verdict.json"), key=lambda p: p.stat().st_mtime, reverse=True
    ) if results_dir.is_dir() else []

    if not verdict_files:
        st.caption("No inspections yet — run your first board on the Inspection & Training page.")
        return

    records: list[dict[str, Any]] = []
    for vf in verdict_files:
        try:
            data = json.loads(vf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - skip corrupt records, keep browsing
            continue
        defects = data.get("defects", [])
        records.append({
            "file": vf,
            "time": datetime.fromtimestamp(vf.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "board_id": data.get("board_id", vf.stem.replace("_verdict", "")),
            "verdict": data.get("verdict", "?"),
            "defects": len(defects),
            "types": ", ".join(sorted({d.get("type", "?") for d in defects})) or "—",
        })

    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        verdict_filter = st.selectbox("Verdict", ["All", "OK", "NG"])
    with col2:
        all_types = sorted({t for r in records for t in r["types"].split(", ") if t != "—"})
        type_filter = st.multiselect("Defect type", all_types)
    with col3:
        search = st.text_input("Search board id")

    filtered = [
        r for r in records
        if (verdict_filter == "All" or r["verdict"] == verdict_filter)
        and (not type_filter or any(t in r["types"] for t in type_filter))
        and (not search or search.lower() in r["board_id"].lower())
    ]
    st.dataframe(
        [{k: v for k, v in r.items() if k != "file"} for r in filtered],
        use_container_width=True,
    )
    if not filtered:
        return

    # --- detail + feedback -------------------------------------------------------
    choice = st.selectbox(
        "Open inspection",
        filtered,
        format_func=lambda r: f"{r['time']} — {r['board_id']} ({r['verdict']})",
    )
    annotated_path = results_dir / f"{choice['board_id']}_annotated.jpg"
    if annotated_path.is_file():
        st.image(str(annotated_path), caption=f"Annotated: {choice['board_id']}",
                 use_container_width=True)
    else:
        st.caption("No annotated image stored for this inspection.")

    st.markdown("**Mark for model improvement** — appends to `results/feedback.jsonl`:")
    col_fr, col_fa = st.columns(2)
    with col_fr:
        if st.button("Mark FALSE REJECT (was NG, actually OK)", use_container_width=True):
            _append_feedback(results_dir,
                             {"board_id": choice["board_id"], "mark": "false_reject"})
            st.success("Logged as false reject.")
    with col_fa:
        if st.button("Mark FALSE ACCEPT (was OK, actually NG)", use_container_width=True):
            _append_feedback(results_dir,
                             {"board_id": choice["board_id"], "mark": "false_accept"})
            st.success("Logged as false accept.")


# ---------------------------------------------------------------------------
# Page 3: Settings
# ---------------------------------------------------------------------------

def page_settings(config_path: Path, cfg: dict[str, Any]) -> None:
    st.header("Model & Threshold Settings")

    det = cfg.get("detection", {})
    ano = cfg.get("anomaly", {})
    golden = cfg.get("golden", {})
    checks = cfg.get("checks", {})

    # --- model file status --------------------------------------------------------
    status = _model_status(cfg)
    st.subheader("Model versions")
    for label, path, ready, required in (
        ("Detector (PP-YOLOE+ export)", status["detection_path"],
         status["detection_ready"], True),
        ("Anomaly model (PatchCore) — deferred", status["anomaly_path"],
         status["anomaly_ready"], False),
    ):
        if ready and path:
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            st.markdown(f"- ✅ **{label}** — `{path.name}`, modified {mtime}")
        elif required:
            st.markdown(f"- ❌ **{label}** — not found (expected at `{path}`)")
        else:
            st.markdown(f"- ⏸ **{label}** — not trained; not required for phase 1")

    # --- thresholds with plain-language explanations (ui_design.md §2.3) -----------
    st.subheader("Thresholds")
    ano_threshold = st.slider(
        "Anomaly score threshold", 0.0, 1.0, float(ano.get("score_threshold", 0.5)), 0.01,
        help="Above this, a heatmap region is reported as scratch / solder debris / "
             "foreign particle. Higher = fewer false alarms but may miss subtle "
             "defects. Tune on the boards_ng validation set.",
    )
    det_threshold = st.slider(
        "Detection confidence threshold", 0.0, 1.0,
        float(det.get("confidence_threshold", 0.5)), 0.01,
        help="Below this, a detected component is ignored. Lower it if small "
             "passives are missed; raise it if phantom components appear.",
    )
    match_iou = st.slider(
        "Component match IoU", 0.1, 0.9, float(golden.get("match_iou", 0.4)), 0.05,
        help="Overlap at which a detection counts as 'the same part' as an expected "
             "component. Lower tolerates more placement jitter.",
    )
    angle_tol = st.slider(
        "Orientation tolerance (degrees)", 5, 90,
        int(golden.get("orientation_tolerance_deg", 30)), 5,
        help="Allowed error between estimated and expected component angle. The "
             "aspect-ratio heuristic is coarse — keep this loose.",
    )
    min_area = st.number_input(
        "Min anomaly region area (px)", 0, 10000, int(ano.get("min_region_area", 64)),
        help="Anomaly regions smaller than this are ignored (speck noise filter).",
    )

    st.subheader("Active defect checks")
    check_labels = {
        "missing_part": "Missing parts (Branch A)",
        "wrong_part": "Wrong parts (Branch A)",
        "wrong_orientation": "Wrong orientation (Branch A)",
        "scratches": "Scratches (Branch B)",
        "solder_balls": "Solder balls / debris (Branch B)",
        "foreign_particles": "Foreign particles (Branch B)",
    }
    new_checks = {
        key: st.checkbox(label, value=bool(checks.get(key, True)))
        for key, label in check_labels.items()
    }

    # --- save with explicit confirmation ---------------------------------------------
    st.warning(
        "Saving rewrites configs/pipeline.yaml (YAML comments are not preserved — "
        "keep the commented original in version control)."
    )
    confirm = st.checkbox("I understand this overwrites configs/pipeline.yaml")
    if st.button("💾 Save settings", disabled=not confirm):
        cfg.setdefault("anomaly", {})["score_threshold"] = ano_threshold
        cfg.setdefault("anomaly", {})["min_region_area"] = int(min_area)
        cfg.setdefault("detection", {})["confidence_threshold"] = det_threshold
        cfg.setdefault("golden", {})["match_iou"] = match_iou
        cfg.setdefault("golden", {})["orientation_tolerance_deg"] = int(angle_tol)
        cfg.setdefault("checks", {}).update(new_checks)
        try:
            import yaml  # noqa: PLC0415

            config_path.write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            st.success(f"Saved to {config_path}. New values apply to the next inspection.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Save failed: {exc}")

    st.caption(
        "Sample-image test: use the Inspection & Training page with a known NG board from "
        "data/boards_ng to preview threshold effects before saving."
    )


# ---------------------------------------------------------------------------
# Page 4: Labeling
# ---------------------------------------------------------------------------

def _inject_labeling_css() -> None:
    """Page-local color for the two primary labeling buttons (ui_design.md §4:
    green #22c55e OK / red #ef4444 NG). Uses :has() marker spans so ONLY the
    OK/NG buttons on this page are recolored; icons + text still carry the
    meaning if a browser ignores the CSS."""
    st.markdown(
        "<style>"
        ".stElementContainer:has(.aoi-ok-btn) + .stElementContainer button{"
        f"background-color:{COLOR_OK};border-color:{COLOR_OK};color:#fff;"
        "height:56px;font-size:20px;font-weight:700;}"
        ".stElementContainer:has(.aoi-ng-btn) + .stElementContainer button{"
        f"background-color:{COLOR_NG};border-color:{COLOR_NG};color:#fff;"
        "height:56px;font-size:20px;font-weight:700;}"
        "</style>",
        unsafe_allow_html=True,
    )


def _session_images(session_dir: Path) -> list[Path]:
    return sorted(p for p in session_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in LABEL_EXTENSIONS)


def _labeling_start(paths: list[Path], origin: str) -> None:
    st.session_state["label_set"] = [str(p) for p in paths]
    st.session_state["label_origin"] = origin
    st.session_state["label_idx"] = 0


def _next_unlabeled(paths: list[str], labels: dict[str, Any], after: int) -> int:
    """First unlabeled index after `after`, wrapping; len(paths) when all done."""
    for j in range(after + 1, len(paths)):
        if paths[j] not in labels:
            return j
    for j in range(0, after + 1):
        if paths[j] not in labels:
            return j
    return len(paths)


def page_labeling() -> None:
    st.header("Dataset Review — bulk import, relabel & export")
    st.caption(
        "Bulk dataset maintenance: import image batches or webcam sessions, fix "
        "labels, and export the labeled dataset. Capture-time labeling (camera "
        "→ OK/NG in one click) lives on the **Inspection & Training** page in "
        "Training mode."
    )
    st.caption(
        "First step of training the model: build the `boards_ok` / `boards_ng` "
        "dataset one image at a time. Labeled images are **copied** into "
        "`data/boards_ok` / `data/boards_ng` — the raw session stays untouched."
    )

    data_root = PROJECT_ROOT / "data"
    ledger_path = _labels_ledger_path(data_root)
    _inject_labeling_css()

    # --- image source -----------------------------------------------------------
    source = st.radio(
        "Image source", ["Upload images", "Import webcam captures"],
        horizontal=True, key="labeling_source",
    )

    if source == "Upload images":
        uploads = st.file_uploader(
            "Upload board images (jpg / jpeg / png / bmp)",
            type=sorted(e.lstrip(".") for e in LABEL_EXTENSIONS),
            accept_multiple_files=True,
        )
        if uploads:
            st.caption(f"{len(uploads)} file(s) selected — saved under "
                       "`data/raw/uploads/` when you start.")
            if st.button("📥 Save & start labeling", use_container_width=True):
                session_dir = save_uploaded_images(uploads, data_root)
                _labeling_start(_session_images(session_dir), session_dir.name)
                st.rerun()
        else:
            st.info("No images yet — upload a batch of board photos to start labeling.")
    else:
        captures_root = data_root / "raw" / "captures"
        sessions = []
        if captures_root.is_dir():
            sessions = [d for d in sorted(captures_root.iterdir(), reverse=True)
                        if d.is_dir() and _session_images(d)]
        if not sessions:
            st.info(
                "No webcam capture sessions yet — run "
                "`scripts/capture_webcam.py` first (see docs/webcam_capture.md), "
                "then pick the session here."
            )
        else:
            pick = st.selectbox(
                "Capture session", sessions,
                format_func=lambda d: f"{d.name} ({len(_session_images(d))} shots)",
            )
            if st.button("📥 Import session & start labeling", use_container_width=True):
                _labeling_start(_session_images(pick), pick.name)
                st.rerun()

    st.divider()

    # --- working set --------------------------------------------------------------
    paths: list[str] = st.session_state.get("label_set", [])
    if not paths:
        st.caption("Pick a source above to start a labeling session.")
    else:
        origin = st.session_state.get("label_origin", "")
        labels = load_label_state(ledger_path, paths)
        n_ok = sum(1 for v in labels.values() if v["label"] == "OK")
        n_ng = sum(1 for v in labels.values() if v["label"] == "NG")
        remaining = len(paths) - len(labels)

        cols = st.columns(5)
        cols[0].metric("✅ OK (this set)", n_ok)
        cols[1].metric("❌ NG (this set)", n_ng)
        cols[2].metric("Remaining", remaining)
        cols[3].metric("boards_ok total", _count_images(data_root / "boards_ok"))
        cols[4].metric("boards_ng total", _count_images(data_root / "boards_ng"))
        st.progress(len(labels) / len(paths) if paths else 0.0,
                    text=f"{len(labels)} / {len(paths)} labeled — source: {origin}")

        idx = min(st.session_state.get("label_idx", 0), len(paths))

        def _apply_label(label: str) -> None:
            """Label or relabel the current image, then auto-advance."""
            src = Path(paths[idx])
            existing = labels.get(paths[idx])
            if existing and existing["label"] == label:
                pass  # same label again — just move on
            elif existing:  # relabel via the big buttons
                relabel_image(Path(existing["saved_path"]), label, data_root,
                              source_path=paths[idx])
            else:
                label_image(src, label, data_root, session_tag=origin)
            fresh = load_label_state(ledger_path, paths)
            st.session_state["label_idx"] = _next_unlabeled(paths, fresh, idx)

        if idx >= len(paths):
            st.success(f"🎉 All {len(paths)} images in this set are labeled "
                       f"({n_ok} OK / {n_ng} NG). Load another source above, "
                       "or fix labels in the review section below.")
            if st.button("← Back to last image"):
                st.session_state["label_idx"] = len(paths) - 1
                st.rerun()
        else:
            current = Path(paths[idx])
            already = labels.get(paths[idx])
            badge = f" — currently labeled **{already['label']}**" if already else ""
            st.subheader(f"Image {idx + 1} of {len(paths)} — `{current.name}`{badge}")
            st.image(str(current), use_container_width=True)

            col_ok, col_ng, col_skip, col_back = st.columns([3, 3, 1, 1])
            with col_ok:
                st.markdown('<span class="aoi-ok-btn"></span>', unsafe_allow_html=True)
                if st.button("✅ OK", use_container_width=True):
                    _apply_label("OK")
                    st.rerun()
            with col_ng:
                st.markdown('<span class="aoi-ng-btn"></span>', unsafe_allow_html=True)
                if st.button("❌ NG", use_container_width=True):
                    _apply_label("NG")
                    st.rerun()
            with col_skip:
                if st.button("Skip →", use_container_width=True):
                    st.session_state["label_idx"] = (idx + 1) % len(paths)
                    st.rerun()
            with col_back:
                if st.button("← Back", use_container_width=True, disabled=idx == 0):
                    st.session_state["label_idx"] = idx - 1
                    st.rerun()

        # --- review & fix -------------------------------------------------------------
        labels = load_label_state(ledger_path, paths)  # refresh after any action
        with st.expander(f"Review & fix — {len(labels)} labeled in this set"):
            if not labels:
                st.caption("Nothing labeled yet in this working set.")
            for i, sp in enumerate(paths):
                entry = labels.get(sp)
                if not entry:
                    continue
                saved = Path(entry["saved_path"])
                col_img, col_name, col_fix = st.columns([1, 3, 2])
                with col_img:
                    if saved.is_file():
                        st.image(str(saved), width=96)
                    else:
                        st.caption("file missing")
                with col_name:
                    st.markdown(f"`{Path(sp).name}` → **{entry['label']}**")
                    st.caption(saved.name)
                with col_fix:
                    choice = st.selectbox(
                        "Change label", ["OK", "NG"],
                        index=0 if entry["label"] == "OK" else 1,
                        key=f"relabel_{i}", label_visibility="collapsed",
                    )
                    if choice != entry["label"]:
                        if saved.is_file():
                            relabel_image(saved, choice, data_root, source_path=sp)
                            st.rerun()
                        else:
                            st.warning("Saved file not found — cannot move it.")

    # --- export ---------------------------------------------------------------------
    st.subheader("Export")
    st.download_button(
        "⬇ Download labeled dataset (.zip)",
        data=build_dataset_zip(data_root),
        file_name=f"pcba_aoi_labeled_dataset_{datetime.now():%Y%m%d_%H%M%S}.zip",
        mime="application/zip",
        use_container_width=True,
    )
    st.caption(
        "Streamlit Cloud's filesystem is ephemeral — anything labeled there is "
        "lost on restart/redeploy, so download the labeled dataset before you leave."
    )


# ---------------------------------------------------------------------------
# Page 5: Dataset & Training
# ---------------------------------------------------------------------------

def page_dataset(cfg: dict[str, Any]) -> None:
    st.header("Dataset & Training Status")

    data_root = PROJECT_ROOT / "data"
    det_dir = data_root / "detection_dataset"

    st.subheader("Image counts")
    rows = [
        {"folder": "boards_ok", "images": _count_images(data_root / "boards_ok"),
         "purpose": "PatchCore training (OK only)", "target": "200–500"},
        {"folder": "boards_ng", "images": _count_images(data_root / "boards_ng"),
         "purpose": "Validation only — never trained on", "target": "10–20 per defect type"},
        {"folder": "detection_dataset/images/train",
         "images": _count_images(det_dir / "images" / "train", recursive=False),
         "purpose": "PP-YOLOE+ fine-tune", "target": "50–100 boards (train+val)"},
        {"folder": "detection_dataset/images/val",
         "images": _count_images(det_dir / "images" / "val", recursive=False),
         "purpose": "Detector validation", "target": "~15% of labeled set"},
        {"folder": "synthetic_ng", "images": _count_images(data_root / "synthetic_ng"),
         "purpose": "Rule-engine validation", "target": "as needed"},
        {"folder": "raw", "images": _count_images(data_root / "raw"),
         "purpose": "Untouched captures", "target": "—"},
    ]
    st.dataframe(rows, use_container_width=True)

    st.subheader("Golden board & expected components")
    golden_img = _resolve(cfg.get("golden", {}).get("image", "data/golden/golden_board.jpg"))
    exp_path = _resolve(
        cfg.get("golden", {}).get("expected_components",
                                  "data/golden/expected_components.json")
    )
    col_g, col_e = st.columns(2)
    with col_g:
        if golden_img and golden_img.is_file():
            st.image(str(golden_img), caption=f"Golden: {golden_img.name}",
                     use_container_width=True)
        else:
            st.info(f"No golden board image yet (expected at {golden_img}).")
    with col_e:
        if exp_path and exp_path.is_file():
            try:
                data = json.loads(exp_path.read_text(encoding="utf-8"))
                comps = data.get("components", [])
                classes = sorted({c.get("class", "?") for c in comps})
                st.markdown(f"✅ **{len(comps)} expected components** defined "
                            f"across **{len(classes)} classes**.")
                st.markdown("Classes: " + ", ".join(f"`{c}`" for c in classes))
            except Exception as exc:  # noqa: BLE001
                st.warning(f"expected_components.json is not valid JSON: {exc}")
        else:
            st.info(
                "No expected_components.json yet — the rule engine (missing / wrong "
                "part / orientation) is disabled until it exists. Edit it by hand or "
                "generate from detector output on the golden board; format is "
                "documented in data/README.md."
            )

    st.subheader("Models")
    status = _model_status(cfg)
    st.markdown(f"- Detector export (phase 1, required): "
                f"{'✅ found' if status['detection_ready'] else '❌ not exported'} "
                f"(`{status['detection_path']}`)")
    st.markdown(f"- Anomaly model (deferred, optional): "
                f"{'✅ found' if status['anomaly_ready'] else '⏸ not trained — not required'} "
                f"(`{status['anomaly_path']}`)")

    st.subheader("Retrain commands")
    st.code("# in the cloned PaddleDetection repo (phase 1):\n"
            "python tools/train.py -c configs/ppyoloe/ppyoloe_plus_custom.yml --eval\n"
            "python tools/export_model.py -c configs/ppyoloe/ppyoloe_plus_custom.yml "
            "-o weights=... output_dir=output_inference/pcba_ppyoloe", language="bash")
    st.code("# DEFERRED — only when the anomaly branch is activated:\n"
            "python src/train_anomaly.py --data-root data --model patchcore "
            "--image-size 512 --output-dir models/anomaly", language="bash")


# ---------------------------------------------------------------------------
# Page: Dashboard (MONITOR) — the management landing page
# ---------------------------------------------------------------------------

def page_dashboard(cfg: dict[str, Any]) -> None:
    st.header("Dashboard")
    if _demo_active():
        st.markdown(
            f'<span style="background-color:{SG_CORAL};color:#ffffff;'
            'font-size:12px;font-weight:700;letter-spacing:1px;'
            'padding:4px 10px;border-radius:999px;">'
            "DEMO — simulated detections</span>",
            unsafe_allow_html=True,
        )

    results_dir = _results_dir(cfg)
    records = _load_verdict_records(results_dir)
    if not records:
        st.info(
            "No inspections yet — run your first board on the **Inspection & Training** "
            "page and this dashboard will light up."
        )
        return

    import pandas as pd  # noqa: PLC0415 - lazy; streamlit hard-depends on it

    # --- metric strip ---------------------------------------------------------
    total = len(records)
    n_ok = sum(1 for r in records if r["verdict"] == "OK")
    n_ng = sum(1 for r in records if r["verdict"] == "NG")
    fpy = 100.0 * n_ok / total if total else 0.0
    last = records[-1]["time"].strftime("%Y-%m-%d %H:%M")
    cols = st.columns(5)
    cols[0].metric("Boards inspected", total)
    cols[1].metric("OK", n_ok)
    cols[2].metric("NG", n_ng)
    cols[3].metric("FPY", f"{fpy:.1f}%")
    cols[4].metric("Last inspection", last)

    # --- FPY trend ------------------------------------------------------------
    st.subheader("FPY trend (daily)")
    daily = _daily_fpy(records)
    fpy_df = pd.DataFrame(daily).set_index("date")[["fpy"]]
    fpy_df.columns = ["FPY %"]
    st.line_chart(fpy_df)

    # --- top defects Pareto ---------------------------------------------------
    st.subheader("Top defects (Pareto)")
    pareto = _defect_pareto(records, by_designator=True)
    if pareto:
        st.bar_chart(pd.DataFrame(pareto).set_index("defect"))
    else:
        st.caption("No defects recorded yet — all boards passed.")

    # --- recent NG feed ---------------------------------------------------------
    st.subheader("Recent NG boards")
    ngs = [r for r in records if r["verdict"] == "NG"][-5:][::-1]
    if not ngs:
        st.caption("No NG boards — nothing to review. 🎉")
    for r in ngs:
        col_img, col_txt = st.columns([1, 3])
        with col_img:
            annotated = results_dir / f"{r['board_id']}_annotated.jpg"
            if annotated.is_file():
                st.image(str(annotated), use_container_width=True)
            else:
                st.caption("no annotated image stored")
        with col_txt:
            st.markdown(f"**{r['board_id']}** — {r['time']:%Y-%m-%d %H:%M}")
            for d in r["defects"]:
                st.markdown(f"- `{d.get('type', '?')}` — {d.get('detail', '')}")

    # --- station status ---------------------------------------------------------
    st.subheader("Station status")
    status = _model_status(cfg)
    w_ok, _ = _results_writable(results_dir)
    st.dataframe(
        [
            {"item": "Pipeline config", "status": "✅ loaded"},
            {"item": "Demo mode",
             "status": "🧪 ON — simulated detections" if _demo_active() else "off"},
            {"item": "Detection model",
             "status": ("✅ ready" if status["detection_ready"]
                        else "❌ not exported")},
            {"item": "Anomaly model (Branch B)",
             "status": ("✅ present" if status["anomaly_ready"]
                        else "⏸ deferred (phase 1 — OK)")},
            {"item": "Results directory",
             "status": "✅ writable" if w_ok else "❌ not writable"},
        ],
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Page: SPC (MONITOR) — statistical process control
# ---------------------------------------------------------------------------

def page_spc(cfg: dict[str, Any]) -> None:
    st.header("SPC — Statistical Process Control")

    results_dir = _results_dir(cfg)
    records = _load_verdict_records(results_dir)
    if not records:
        st.info("No inspection data yet — run boards on the **Inspection & Training** page first.")
        return

    import pandas as pd  # noqa: PLC0415

    # --- date-range filter ------------------------------------------------------
    min_d = records[0]["time"].date()
    max_d = records[-1]["time"].date()
    picked = st.date_input("Date range", value=(min_d, max_d),
                           min_value=min_d, max_value=max_d)
    start, end = ((picked[0], picked[1])
                  if isinstance(picked, tuple) and len(picked) == 2
                  else (min_d, max_d))
    filtered = [r for r in records if start <= r["time"].date() <= end]
    if not filtered:
        st.warning("No inspections in the selected date range.")
        return

    # --- FPY trend --------------------------------------------------------------
    st.subheader("FPY trend (daily)")
    daily = _daily_fpy(filtered)
    fpy_df = pd.DataFrame(daily).set_index("date")[["fpy"]]
    fpy_df.columns = ["FPY %"]
    st.line_chart(fpy_df)

    # --- defect Pareto: by type AND by designator -------------------------------
    st.subheader("Defect Pareto")
    col_t, col_d = st.columns(2)
    with col_t:
        st.markdown("**By defect type**")
        pareto_t = _defect_pareto(filtered)
        if pareto_t:
            st.bar_chart(pd.DataFrame(pareto_t).set_index("defect"))
        else:
            st.caption("No defects in range.")
    with col_d:
        st.markdown("**By type + designator**")
        pareto_d = _defect_pareto(filtered, by_designator=True)
        if pareto_d:
            st.bar_chart(pd.DataFrame(pareto_d).set_index("defect"))
        else:
            st.caption("No defects in range.")

    # --- p-chart ------------------------------------------------------------------
    st.subheader("Defect-rate control chart (p-chart)")
    pc = _pchart(filtered)
    chart_df = pd.DataFrame(
        {"daily NG rate": pc["p"], "center line (p̄)": pc["cl"],
         "UCL": pc["ucl"], "LCL": pc["lcl"]},
        index=pc["dates"],
    )
    st.line_chart(chart_df)
    st.caption(
        f"p̄ = {pc['p_bar']:.3f}. Limits = p̄ ± 3·√(p̄(1−p̄)/n), "
        f"n = boards that day (daily n here: {pc['n']}). When n is small or "
        "varies day to day, the limits are approximate — read them as a "
        "screening aid, not a formal alarm."
    )


# ---------------------------------------------------------------------------
# Page: System Check (MAINTENANCE) — one-click self-diagnosis
# ---------------------------------------------------------------------------

def page_system_check(cfg: dict[str, Any], config_path: Path) -> None:
    st.header("System Check")
    st.caption(
        "One-click station self-diagnosis, computed live from the filesystem — "
        "extends the Setup Wizard checks with runtime health (writable results, "
        "disk space, ledgers)."
    )
    if not st.button("▶ Run checks", type="primary"):
        st.caption("Press **Run checks** to test the station.")
        return

    rows = collect_system_checks(cfg, config_path)
    n_bad = sum(1 for r in rows if not r["ok"])
    if n_bad == 0:
        st.success(f"✅ ALL GREEN — {len(rows)} checks passed.")
    else:
        st.error(f"❌ ISSUES FOUND — {n_bad} of {len(rows)} checks failed.")
    st.dataframe(
        [{"": "✅" if r["ok"] else "❌", "Check": r["check"],
          "Detail": r["detail"], "If red →": r["fix"]} for r in rows],
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Page: Audit Trail (ADMINISTRATION) — read-only human-decision history
# ---------------------------------------------------------------------------

def page_audit_trail(cfg: dict[str, Any]) -> None:
    st.header("Audit Trail")
    st.caption(
        "Read-only history of human decisions: labeling actions "
        "(`data/labels.jsonl`) and inspection feedback / NG overrides "
        "(`results/feedback.jsonl`). Operator identity is not tracked yet — "
        "Users & Roles is a later phase."
    )
    rows = _audit_rows(PROJECT_ROOT / "data", _results_dir(cfg))
    if not rows:
        st.info(
            "No audit entries yet — labeling actions and NG overrides/marks "
            "will appear here as they happen."
        )
        return
    actions = sorted({r["action"] for r in rows})
    picked = st.multiselect("Filter by action type", actions, default=actions)
    st.dataframe([r for r in rows if r["action"] in picked],
                 use_container_width=True)


# ---------------------------------------------------------------------------
# Page 6: Setup Wizard
# ---------------------------------------------------------------------------

def page_wizard(cfg: dict[str, Any]) -> None:
    st.header("Setup Wizard")
    st.caption("Live checklist computed from the filesystem — maps to the README roadmap.")

    data_root = PROJECT_ROOT / "data"
    status = _model_status(cfg)
    golden_img = _resolve(cfg.get("golden", {}).get("image", "data/golden/golden_board.jpg"))
    exp_path = _resolve(
        cfg.get("golden", {}).get("expected_components",
                                  "data/golden/expected_components.json")
    )
    det_train = _count_images(data_root / "detection_dataset" / "images" / "train",
                              recursive=False)
    anno_train = data_root / "detection_dataset" / "annotations" / "instances_train.json"

    n_components = 0
    if exp_path and exp_path.is_file():
        try:
            n_components = len(json.loads(exp_path.read_text(encoding="utf-8"))
                               .get("components", []))
        except Exception:  # noqa: BLE001 - invalid JSON counts as not done
            n_components = 0

    items: list[dict[str, Any]] = [
        {"done": _count_images(data_root / "raw") > 0,
         "title": "Camera rig set up, captures in data/raw",
         "hint": "Fixed mount, diffuse lighting, fiducials on the jig.",
         "cmd": "# see docs/data_collection_guide.md"},
        {"done": bool(golden_img and golden_img.is_file()),
         "title": "Golden board captured (data/golden)",
         "hint": "One known-good board, aligned canonical view.",
         "cmd": "python src/align_board.py --input data/raw --golden <ref> --output data/boards_ok"},
        {"done": n_components > 0,
         "title": f"expected_components.json defined ({n_components} components)",
         "hint": "Reference designator, class, bbox, expected angle per part.",
         "cmd": "# format documented in data/README.md"},
        {"done": anno_train.is_file() and det_train >= 50,
         "title": f"Detection dataset labeled ({det_train} train images, COCO)",
         "hint": "COCO export from X-AnyLabeling / CVAT; every class ≥ ~100 instances.",
         "cmd": "# label, export COCO into data/detection_dataset, then see "
                "configs/detection/README.md"},
        {"done": status["detection_ready"],
         "title": "Detector trained & exported (PP-YOLOE+)",
         "hint": "Run inside the cloned PaddleDetection repo.",
         "cmd": "python tools/export_model.py -c configs/ppyoloe/ppyoloe_plus_custom.yml ..."},
        # Deferred — phase 1 is detection-only; this step is informational and
        # never blocks go-live. ui_design.md §2.5 shows it as ⏸.
        {"done": status["anomaly_ready"],
         "deferred": True,
         "title": "Anomaly model (PatchCore) — DEFERRED, optional",
         "hint": "Only needed for scratches / solder balls / foreign particles "
                 "(a later phase). Not required for go-live.",
         "cmd": "python src/train_anomaly.py --data-root data --output-dir models/anomaly"},
    ]
    go_live = items[3]["done"] and items[4]["done"]
    items.append({"done": go_live, "title": "Go live",
                  "hint": "Needs the labeled dataset AND the exported detector "
                          "(the anomaly model is deferred and not required).",
                  "cmd": "streamlit run src/app.py"})

    active = [i for i in items if not i.get("deferred")]
    done_count = sum(1 for i in active if i["done"])
    st.progress(done_count / len(active),
                text=f"{done_count} / {len(active)} required steps complete")

    for idx, item in enumerate(items, start=1):
        if item.get("deferred"):
            icon = "⏸"
        else:
            icon = "✅" if item["done"] else "❌"
        st.markdown(f"{icon} **{idx}. {item['title']}**")
        if not item["done"] or item.get("deferred"):
            st.caption(f"↳ {item['hint']}")
            st.code(item["cmd"], language="bash")


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="SG-AOI · Season Group",
        page_icon=str(LOGO_LIGHT) if LOGO_LIGHT.is_file() else "🔍",
        layout="wide",
    )
    _inject_brand_css()

    # --- branded sidebar header -------------------------------------------------
    if LOGO_LIGHT.is_file():
        st.sidebar.image(str(LOGO_LIGHT), width=180)
    st.sidebar.markdown("## SG-AOI")
    st.sidebar.caption(LINE_NAME)
    st.sidebar.divider()

    # --- role simulation + sectioned navigation ---------------------------------
    role = st.sidebar.selectbox(
        "Role", ROLES,
        help="Simulated role gating for the demo (no authentication): "
             "Operator sees RUN only, Engineer adds MONITOR + BUILD, "
             "Admin sees everything.",
    )
    allowed = ROLE_SECTIONS[role]
    options: list[str] = []
    for section, pages in NAV_SECTIONS:
        if section not in allowed:
            continue
        st.sidebar.markdown(
            f'<div class="sg-section-label">{section}</div>',
            unsafe_allow_html=True,
        )
        options.extend(pages)
    st.sidebar.markdown('<div class="sg-section-label">SETUP</div>',
                        unsafe_allow_html=True)
    options.append(SETUP_PAGE)
    page = st.sidebar.radio("Page", options, label_visibility="collapsed")

    st.sidebar.divider()
    # Language toggle placeholder — selection is stored but not applied yet.
    st.sidebar.selectbox("Language / 语言", list(LANGUAGES), key="language",
                         help="Placeholder — translation tables are a future-HMI task.")
    config_input = st.sidebar.text_input("Pipeline config", str(DEFAULT_CONFIG))
    config_path = Path(config_input)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.is_file():
        st.error(f"Config not found: {config_path}")
        st.stop()
    cfg = _load_cfg(config_path)
    if cfg is None:
        st.stop()

    if page == "Inspection & Training":
        page_inspection_training(cfg)
    elif page == "Review & Repair":
        page_history(cfg)
    elif page == "Dashboard":
        page_dashboard(cfg)
    elif page == "SPC":
        page_spc(cfg)
    elif page == "Dataset Review":
        page_labeling()
    elif page == "Dataset & Training":
        page_dataset(cfg)
    elif page == "Audit Trail":
        page_audit_trail(cfg)
    elif page == "Settings":
        page_settings(config_path, cfg)
    elif page == "System Check":
        page_system_check(cfg, config_path)
    else:
        page_wizard(cfg)


if __name__ == "__main__":
    main()

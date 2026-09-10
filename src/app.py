"""PCBA AOI operator UI (Streamlit) — implements docs/ui_design.md.

Six pages (sidebar navigation):
  1. Inspection        — capture/upload -> align (optional) -> pipeline -> verdict
  2. Review History    — past verdicts, filters, false-reject/false-accept marks
  3. Settings          — threshold sliders/toggles over configs/pipeline.yaml
  4. Labeling          — mark images OK/NG into boards_ok/boards_ng + labels.jsonl
  5. Dataset & Training— image counts, golden board, model file status
  6. Setup Wizard      — live checklist from filesystem state vs. README roadmap

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


def label_image(src_path: Path, label: str, data_root: Path, session_tag: str = "") -> Path:
    """COPY (not move) an image into boards_ok/boards_ng and append a ledger line."""
    src_path = Path(src_path)
    dest_dir = _label_target_dir(label, data_root)
    dest_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_sanitize_tag(session_tag)}_" if session_tag else ""
    dest = dest_dir / _collision_safe_name(dest_dir, prefix + src_path.name)
    shutil.copy2(src_path, dest)
    _append_label_ledger(
        {"action": "label", "source_path": str(src_path),
         "saved_path": str(dest), "label": label},
        _labels_ledger_path(data_root),
    )
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
# Page 1: Inspection
# ---------------------------------------------------------------------------

def page_inspection(cfg: dict[str, Any]) -> None:
    st.header("Inspection")

    # --- demo mode (simulated detections, no trained model) -------------------
    demo_image: Path | None = None
    if _demo_active():
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

    # --- capture -------------------------------------------------------------
    st.subheader("1. Board image")
    input_path: Path | None = None
    align_first = False
    if demo_image is not None and demo_image.is_file():
        # Demo boards are already in the canonical aligned view — the capture
        # and alignment steps are replaced by the scripted scenario image.
        st.image(str(demo_image), caption=f"Demo board: {demo_image.name}",
                 use_container_width=True)
        input_path = demo_image
    else:
        src_tab_up, src_tab_cam = st.tabs(["Upload", "Camera snapshot"])
        with src_tab_up:
            uploaded = st.file_uploader(
                "Upload a board image", type=[e.lstrip(".") for e in sorted(IMAGE_EXTENSIONS)]
            )
        with src_tab_cam:
            snapshot = st.camera_input("Or take a snapshot")
        capture = uploaded or snapshot

        align_first = st.checkbox(
            "Run board alignment first (raw capture, not yet aligned)", value=False
        )
        golden_default = cfg.get("golden", {}).get("image", "data/golden/golden_board.jpg")
        golden_path = None
        if align_first:
            golden_path = st.text_input("Golden reference image", str(golden_default))

        if capture is None:
            st.caption("Waiting for a board image…")
            return

        suffix = Path(getattr(capture, "name", "snapshot.jpg")).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(capture.getbuffer())
            input_path = Path(tmp.name)

    # --- alignment -----------------------------------------------------------
    if align_first:
        import cv2  # noqa: PLC0415

        from align_board import align_image  # noqa: PLC0415

        golden_file = _resolve(golden_path)
        image = cv2.imread(str(input_path))
        golden = cv2.imread(str(golden_file)) if golden_file and golden_file.is_file() else None
        if image is None or golden is None:
            st.error(
                "❌ Alignment not possible — could not read the uploaded image or the "
                f"golden reference ({golden_file}). Check the path and retry."
            )
            return
        result = align_image(image, golden)
        if result.warped is None:
            # Alignment-failure state (ui_design.md §5): no verdict is produced.
            st.error(
                "❌ Board not detected / fiducials not found — reseat the board, "
                "check lighting, and retry. No verdict was produced."
            )
            return
        cv2.imwrite(str(input_path), result.warped)
        st.success(f"Aligned via {result.method}.")

    # --- run ------------------------------------------------------------------
    st.subheader("2. Inspection")
    if st.button("▶ Run inspection", type="primary", use_container_width=True):
        with st.spinner("Running detection + anomaly branches…"):
            try:
                from infer_pipeline import inspect_board  # noqa: PLC0415

                run_cfg = _resolved_run_cfg(cfg)
                verdict, annotated = inspect_board(input_path, run_cfg)
                st.session_state["inspection"] = {
                    "verdict": verdict,
                    "annotated": annotated,
                    "cfg": run_cfg,
                }
                # Persist immediately so Review History sees it (no extra click).
                from infer_pipeline import save_outputs  # noqa: PLC0415

                save_outputs(verdict, annotated, run_cfg)
            except Exception as exc:  # noqa: BLE001 - operator UI must not crash
                st.error(f"Pipeline failed: {exc}")
                return

    state = st.session_state.get("inspection")
    if not state:
        st.caption("Press **Run inspection** to get a verdict.")
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


# ---------------------------------------------------------------------------
# Page 2: Review History
# ---------------------------------------------------------------------------

def page_history(cfg: dict[str, Any]) -> None:
    st.header("Review History")
    results_dir = _results_dir(cfg)
    verdict_files = sorted(
        results_dir.glob("*_verdict.json"), key=lambda p: p.stat().st_mtime, reverse=True
    ) if results_dir.is_dir() else []

    if not verdict_files:
        st.caption("No inspections yet — run your first board on the Inspection page.")
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
        "Sample-image test: use the Inspection page with a known NG board from "
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
    st.header("Labeling — mark boards OK / NG")
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
    st.set_page_config(page_title="PCBA AOI", page_icon="🔍", layout="wide")
    st.title("PCBA AOI — OK / NG inspection")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Page",
        ["Inspection", "Review History", "Settings", "Labeling",
         "Dataset & Training", "Setup Wizard"],
        label_visibility="collapsed",
    )
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

    if page == "Inspection":
        page_inspection(cfg)
    elif page == "Review History":
        page_history(cfg)
    elif page == "Settings":
        page_settings(config_path, cfg)
    elif page == "Labeling":
        page_labeling()
    elif page == "Dataset & Training":
        page_dataset(cfg)
    else:
        page_wizard(cfg)


if __name__ == "__main__":
    main()

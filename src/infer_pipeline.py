"""Combined AOI decision engine: detection branch + anomaly branch -> verdict.

Pipeline for one aligned board image:

1. Branch A (PP-YOLOE+): detect components -> class + bbox + score.
2. Rule engine: match detections against expected_components.json:
   - expected component with no matching detection        -> missing_part
   - matched detection with wrong class                   -> wrong_part
   - matched detection with implausible orientation       -> wrong_orientation
3. Branch B (PatchCore): anomaly heatmap -> threshold -> defect regions
   (scratches / solder balls / foreign particles share this branch because
   their appearance is unbounded; the defect type label for Branch-B findings
   is reported as "appearance_anomaly" unless classification is added later).
4. Fuse: any finding makes the board NG; else OK. Write verdict JSON and an
   annotated visualization.

Both model backends are pluggable and degrade gracefully:
* If Paddle inference is unavailable or no model is exported yet, detection
  can be read from a precomputed JSON (see configs/pipeline.yaml) or returns
  empty with a warning.
* If the anomaly model is missing, Branch B is skipped with a warning.

CLI
---
    python src/infer_pipeline.py --image <path> --config configs/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger("infer_pipeline")

# BGR colors for the annotated visualization.
COLOR_OK = (60, 180, 75)
COLOR_NG = (40, 40, 230)
COLOR_EXPECTED = (200, 160, 40)
COLOR_ANOMALY = (0, 200, 255)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """One detected component from Branch A."""

    class_name: str
    bbox: tuple[float, float, float, float]  # x, y, w, h (golden-image pixels)
    score: float


@dataclass
class Defect:
    """One fused finding that contributes to an NG verdict."""

    type: str            # missing_part | wrong_part | wrong_orientation | appearance_anomaly
    location: list[float]  # x, y, w, h
    confidence: float
    branch: str          # "A" | "B"
    detail: str = ""


@dataclass
class Verdict:
    """Per-board result written to JSON."""

    board_id: str
    verdict: str  # "OK" | "NG"
    defects: list[Defect] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Branch A backend
# ---------------------------------------------------------------------------

def run_detection(image: np.ndarray, cfg: dict[str, Any]) -> list[Detection]:
    """Run PP-YOLOE+ on the image, or load precomputed detections.

    The Paddle inference runtime is optional: a precomputed JSON keeps the
    rest of the pipeline (rules, fusion, UI) testable before the detector
    is trained or on a machine without PaddlePaddle installed.
    """
    pre = cfg["detection"].get("precomputed_json")
    if pre:
        path = Path(pre)
        if not path.is_file():
            LOGGER.warning("precomputed_json not found: %s", path)
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [
            Detection(
                class_name=d["class"],
                bbox=tuple(float(v) for v in d["bbox"]),
                score=float(d.get("score", 1.0)),
            )
            for d in raw
            if float(d.get("score", 1.0)) >= cfg["detection"]["confidence_threshold"]
        ]

    model_dir = Path(cfg["detection"]["model_dir"])
    if not (model_dir / "model.pdmodel").is_file():
        LOGGER.warning(
            "Detection model not exported yet (%s missing). Returning no detections — "
            "train/export PP-YOLOE+ or set detection.precomputed_json.", model_dir,
        )
        return []

    try:
        return _run_paddle_inference(image, model_dir, cfg)
    except ImportError:
        LOGGER.warning(
            "paddle is not installed in this environment. Returning no detections — "
            "install paddlepaddle here, or set detection.precomputed_json."
        )
        return []


def _run_paddle_inference(
    image: np.ndarray, model_dir: Path, cfg: dict[str, Any]
) -> list[Detection]:
    """Paddle inference for an exported PP-YOLOE+ model.

    TODO: this mirrors PaddleDetection's deploy/python/infer.py preprocessing
    (Resize -> Normalize -> Permute, params read from infer_cfg.yml). If you
    have PaddleDetection cloned, prefer calling its Detector class directly —
    it handles letterbox/scaling edge cases this compact version glosses over.
    """
    from paddle import inference  # noqa: PLC0415 - optional heavy import

    deploy_cfg = _read_deploy_cfg(model_dir / "infer_cfg.yml")
    preprocess_ops = deploy_cfg.get("Preprocess", [])
    label_list = deploy_cfg.get("label_list", [])
    arch = deploy_cfg.get("arch", "YOLO")

    config = inference.Config(
        str(model_dir / "model.pdmodel"), str(model_dir / "model.pdiparams")
    )
    if cfg["detection"].get("device", "cpu") == "gpu":
        config.enable_use_gpu(100, 0)
    else:
        config.disable_gpu()
    predictor = inference.create_predictor(config)

    # Minimal YOLO-style preprocessing: resize to the deploy eval size and
    # normalize. PP-YOLOE+ uses mean [0,0,0], std [1,1,1] (i.e. /255 only).
    eval_size = _find_eval_size(preprocess_ops) or (640, 640)
    h, w = image.shape[:2]
    im = cv2.resize(image, tuple(eval_size))
    im = im[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, scale
    im = im.transpose(2, 0, 1)[np.newaxis, ...]

    input_names = predictor.get_input_names()
    predictor.get_input_handle(input_names[0]).copy_from_cpu(im)
    # YOLO-family models take an im_shape / scale_factor input; names vary
    # by export version, so feed whatever extra shape inputs exist.
    for name in input_names[1:]:
        handle = predictor.get_input_handle(name)
        if "scale" in name:
            handle.copy_from_cpu(
                np.array([[eval_size[1] / h, eval_size[0] / w]], dtype=np.float32)
            )
        else:
            handle.copy_from_cpu(np.array([[h, w]], dtype=np.float32))

    predictor.run()
    output = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()
    # Output rows: [label_id, score, xmin, ymin, xmax, ymax] (deploy format).
    detections: list[Detection] = []
    sx, sy = w / eval_size[0], h / eval_size[1]  # map back to input-image pixels
    for row in output:
        label_id, score, xmin, ymin, xmax, ymax = row[:6]
        if score < cfg["detection"]["confidence_threshold"]:
            continue
        class_name = label_list[int(label_id)] if int(label_id) < len(label_list) else str(int(label_id))
        detections.append(
            Detection(
                class_name=class_name,
                bbox=(float(xmin * sx), float(ymin * sy),
                      float((xmax - xmin) * sx), float((ymax - ymin) * sy)),
                score=float(score),
            )
        )
    LOGGER.info("Branch A (%s): %d detections above threshold.", arch, len(detections))
    return detections


def _read_deploy_cfg(path: Path) -> dict[str, Any]:
    """Read PaddleDetection's infer_cfg.yml (PyYAML or a tiny fallback)."""
    if not path.is_file():
        return {}
    try:
        import yaml  # noqa: PLC0415

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ImportError:
        LOGGER.warning("PyYAML not installed; using deploy defaults.")
        return {}


def _find_eval_size(preprocess_ops: list[dict[str, Any]]) -> tuple[int, int] | None:
    """Pull target_size from the Resize op in infer_cfg.yml, if present."""
    for op in preprocess_ops or []:
        if op.get("type") == "Resize" and op.get("target_size"):
            size = op["target_size"]
            return (int(size[0]), int(size[1]))
    return None


# ---------------------------------------------------------------------------
# Rule engine (Branch A findings)
# ---------------------------------------------------------------------------

def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """IoU of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _aspect_angle(bbox: tuple[float, ...]) -> float:
    """Coarse orientation estimate from bbox aspect ratio: 0 or 90 degrees.

    Rectangular passives (resistors/capacitors/ICs) are placed at multiples of
    90 deg, so a wide box means 0 deg and a tall box means 90 deg. This
    catches the common "part rotated 90 deg" failure. It cannot detect 180 deg
    flips of symmetric parts — that needs template matching or an oriented
    detector (see roadmap in README).
    """
    _, _, w, h = bbox
    if w == 0 or h == 0:
        return 0.0
    return 0.0 if w >= h else 90.0


def run_rule_engine(
    detections: list[Detection],
    expected_components: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> list[Defect]:
    """Compare detections against the expected placement list."""
    checks = cfg.get("checks", {})
    match_iou = float(cfg["golden"].get("match_iou", 0.4))
    angle_tol = float(cfg["golden"].get("orientation_tolerance_deg", 30))

    defects: list[Defect] = []
    unmatched = list(detections)  # detections not yet claimed by an expected part

    for exp in expected_components:
        exp_bbox = tuple(float(v) for v in exp["bbox"])
        # Claim the best-overlapping detection, regardless of class first —
        # a wrong part at the right place must be flagged as wrong_part, not
        # reported as missing + unexpected.
        best, best_iou = None, 0.0
        for det in unmatched:
            iou = _iou(det.bbox, exp_bbox)
            if iou > best_iou:
                best, best_iou = det, iou

        if best is None or best_iou < match_iou:
            if checks.get("missing_part", True):
                defects.append(Defect(
                    type="missing_part",
                    location=list(exp_bbox),
                    confidence=1.0,
                    branch="A",
                    detail=f"Expected {exp.get('id')} ({exp.get('class')}) not detected.",
                ))
            continue

        unmatched.remove(best)

        if checks.get("wrong_part", True) and best.class_name != exp.get("class"):
            defects.append(Defect(
                type="wrong_part",
                location=list(best.bbox),
                confidence=best.score,
                branch="A",
                detail=(f"{exp.get('id')}: expected {exp.get('class')}, "
                        f"detected {best.class_name}."),
            ))
            continue  # orientation of the wrong part is not meaningful

        if checks.get("wrong_orientation", True):
            expected_angle = float(exp.get("expected_angle_deg", 0))
            estimated = _aspect_angle(best.bbox)
            # Angles are modular at 180 for the aspect heuristic.
            diff = abs((estimated - expected_angle) % 180)
            diff = min(diff, 180 - diff)
            if diff > angle_tol:
                defects.append(Defect(
                    type="wrong_orientation",
                    location=list(best.bbox),
                    confidence=best.score,
                    branch="A",
                    detail=(f"{exp.get('id')}: expected ~{expected_angle:.0f} deg, "
                            f"estimated ~{estimated:.0f} deg."),
                ))
    return defects


# ---------------------------------------------------------------------------
# Branch B backend
# ---------------------------------------------------------------------------

def run_anomaly(image: np.ndarray, cfg: dict[str, Any]) -> tuple[float, np.ndarray] | None:
    """Run the anomaly model; return (image-level score, heatmap) or None.

    Supports anomalib's OpenVINO export (directory) and torch checkpoints.
    Returns None when the model/runtime is unavailable so the pipeline can
    still produce Branch-A-only verdicts.
    """
    model_path = Path(cfg["anomaly"]["model_path"])
    if not model_path.exists():
        LOGGER.warning("Anomaly model not found at %s — Branch B skipped.", model_path)
        return None

    try:
        return _run_anomalib_inference(image, model_path, cfg)
    except ImportError:
        LOGGER.warning("anomalib not installed here — Branch B skipped.")
        return None
    except Exception as exc:  # noqa: BLE001 - model/API drift should not kill the run
        LOGGER.warning("Anomaly inference failed (%s) — Branch B skipped.", exc)
        return None


def _run_anomalib_inference(
    image: np.ndarray, model_path: Path, cfg: dict[str, Any]
) -> tuple[float, np.ndarray]:
    """anomalib inferencer: OpenVINO export dir, or a torch .ckpt."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if model_path.is_dir():
        # OpenVINO export layout from anomalib (v1/v2 both expose
        # OpenVINOInferencer, though the import path moved between versions).
        try:
            from anomalib.deploy import OpenVINOInferencer  # noqa: PLC0415
        except ImportError:
            from anomalib.deploy.inferencers import OpenVINOInferencer  # type: ignore  # noqa: PLC0415
        inferencer = OpenVINOInferencer(path=str(model_path), device=cfg["anomaly"].get("device", "CPU"))
        result = inferencer.predict(image=rgb)
        score = float(getattr(result, "pred_score", 0.0) or 0.0)
        heatmap = np.asarray(getattr(result, "anomaly_map", None))
        return score, _normalize_heatmap(heatmap, image.shape[:2])

    # Torch checkpoint path: load via anomalib's TorchInferencer.
    try:
        from anomalib.deploy import TorchInferencer  # noqa: PLC0415
    except ImportError:
        from anomalib.deploy.inferencers import TorchInferencer  # type: ignore  # noqa: PLC0415
    inferencer = TorchInferencer(path=str(model_path), device=cfg["anomaly"].get("device", "cpu"))
    result = inferencer.predict(image=rgb)
    score = float(getattr(result, "pred_score", 0.0) or 0.0)
    heatmap = np.asarray(getattr(result, "anomaly_map", None))
    return score, _normalize_heatmap(heatmap, image.shape[:2])


def _normalize_heatmap(heatmap: np.ndarray | None, out_hw: tuple[int, int]) -> np.ndarray:
    """Resize heatmap to board-image size and scale to [0, 1]."""
    if heatmap is None or heatmap.size == 0:
        return np.zeros(out_hw, dtype=np.float32)
    heatmap = heatmap.astype(np.float32).squeeze()
    heatmap = cv2.resize(heatmap, (out_hw[1], out_hw[0]))
    lo, hi = float(heatmap.min()), float(heatmap.max())
    if hi - lo < 1e-8:
        return np.zeros(out_hw, dtype=np.float32)
    return (heatmap - lo) / (hi - lo)


def extract_anomaly_regions(
    score: float, heatmap: np.ndarray, cfg: dict[str, Any]
) -> list[Defect]:
    """Threshold the heatmap into reported defect regions."""
    threshold = float(cfg["anomaly"]["score_threshold"])
    min_area = int(cfg["anomaly"].get("min_region_area", 64))
    if score < threshold and float(heatmap.max()) < threshold:
        return []

    binary = (heatmap >= threshold).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects: list[Defect] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        region_peak = float(heatmap[y : y + h, x : x + w].max())
        defects.append(Defect(
            type="appearance_anomaly",
            location=[float(x), float(y), float(w), float(h)],
            confidence=region_peak,
            branch="B",
            detail="Scratch / solder debris / foreign particle candidate.",
        ))
    return defects


# ---------------------------------------------------------------------------
# Fusion, annotation, IO
# ---------------------------------------------------------------------------

def decide(board_id: str, defects: list[Defect]) -> Verdict:
    """Fuse all findings into the board verdict. Any defect -> NG."""
    return Verdict(board_id=board_id, verdict="NG" if defects else "OK", defects=defects)


def annotate(
    image: np.ndarray,
    detections: list[Detection],
    defects: list[Defect],
    heatmap: np.ndarray | None,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Draw the operator-facing visualization: heatmap + boxes + verdict."""
    out = image.copy()

    if heatmap is not None:
        heat_vis = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        out = cv2.addWeighted(out, 0.7, heat_vis, 0.3, 0)

    for det in detections:
        x, y, w, h = (int(v) for v in det.bbox)
        cv2.rectangle(out, (x, y), (x + w, y + h), COLOR_OK, 1)
        cv2.putText(out, f"{det.class_name} {det.score:.2f}", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_OK, 1)

    for defect in defects:
        x, y, w, h = (int(v) for v in defect.location)
        color = COLOR_ANOMALY if defect.branch == "B" else COLOR_NG
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, defect.type, (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    verdict_text = "NG" if defects else "OK"
    cv2.putText(out, verdict_text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                COLOR_NG if defects else COLOR_OK, 3)
    return out


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read the pipeline config (pip install pyyaml)."
        ) from exc
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def inspect_board(image_path: Path, cfg: dict[str, Any]) -> tuple[Verdict, np.ndarray]:
    """Full pipeline for one aligned board image. Also used by src/app.py."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    expected_path = Path(cfg["golden"]["expected_components"])
    expected = []
    if expected_path.is_file():
        expected = json.loads(expected_path.read_text(encoding="utf-8")).get("components", [])
    else:
        LOGGER.warning("expected_components.json not found at %s — rule engine disabled.",
                       expected_path)

    # Branch A + rules.
    detections = run_detection(image, cfg)
    defects = run_rule_engine(detections, expected, cfg)

    # Branch B.
    heatmap: np.ndarray | None = None
    anomaly_result = run_anomaly(image, cfg)
    if anomaly_result is not None:
        score, heatmap = anomaly_result
        defects.extend(extract_anomaly_regions(score, heatmap, cfg))

    verdict = decide(image_path.stem, defects)
    annotated = annotate(image, detections, defects, heatmap, cfg)
    return verdict, annotated


def save_outputs(verdict: Verdict, annotated: np.ndarray, cfg: dict[str, Any],
                 extra: dict[str, Any] | None = None) -> None:
    """Persist the verdict JSON + annotated image.

    ``extra`` keys (e.g. item_id / item_name / source tagging from the app)
    are merged into the serialized verdict dict; existing keys are never
    renamed, so legacy records keep parsing unchanged.
    """
    out_dir = Path(cfg["output"].get("results_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg["output"].get("save_json", True):
        (out_dir / f"{verdict.board_id}_verdict.json").write_text(
            json.dumps({**asdict(verdict), **(extra or {})}, indent=2),
            encoding="utf-8"
        )
    if cfg["output"].get("save_annotated", True):
        cv2.imwrite(str(out_dir / f"{verdict.board_id}_annotated.jpg"), annotated)
    LOGGER.info("Results written to %s", out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combined AOI decision engine.")
    parser.add_argument("--image", required=True, type=Path,
                        help="Aligned board image to inspect.")
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"),
                        help="Pipeline configuration YAML.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    verdict, annotated = inspect_board(args.image, cfg)
    save_outputs(verdict, annotated, cfg)
    print(json.dumps(asdict(verdict), indent=2))


if __name__ == "__main__":
    main()

"""Train the anomaly-detection branch (PatchCore by default) with anomalib.

The anomaly branch learns what an OK board looks like — it never sees
defective images during training. Boards with scratches, solder balls or
foreign particles produce high anomaly scores and hot regions in the heatmap.

anomalib API note
-----------------
The Python API changed between anomalib v1.x and v2.x (datamodule and
export signatures in particular). This script targets the modern layout:

    from anomalib.data import Folder
    from anomalib.models import Patchcore, Padim, EfficientAd
    from anomalib.engine import Engine

and uses defensive getattr/try-except at the two spots known to differ.
If your anomalib version raises an ImportError/TypeError, check the installed
version's docs — the model quality is the same, only the plumbing moved.

CLI
---
    python src/train_anomaly.py \
        --data-root data --model patchcore --image-size 512 \
        --output-dir models/anomaly
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

LOGGER = logging.getLogger("train_anomaly")

MODEL_REGISTRY = {
    # PatchCore: memory-bank of patch features from OK images; the standard
    # choice for texture/appearance defects like scratches and solder debris.
    "patchcore": {"layers": ["layer2", "layer3"], "coreset_sampling_ratio": 0.1},
    # PaDiM: Gaussian per patch position; lighter, decent fallback.
    "padim": {"layers": ["layer1", "layer2", "layer3"]},
    # EfficientAD: fast student-teacher; better if you later need real-time.
    "efficient_ad": {},
}


def build_model(model_name: str, backbone: str):
    """Instantiate the anomalib model, tolerating import-path drift."""
    if model_name == "patchcore":
        try:
            from anomalib.models import Patchcore  # anomalib >= 1.1
        except ImportError:  # very old layout (<=1.0)
            from anomalib.models.patchcore import Patchcore  # type: ignore
        return Patchcore(
            backbone=backbone,
            layers=MODEL_REGISTRY["patchcore"]["layers"],
            coreset_sampling_ratio=MODEL_REGISTRY["patchcore"]["coreset_sampling_ratio"],
            num_neighbors=9,
        )
    if model_name == "padim":
        from anomalib.models import Padim
        return Padim(backbone=backbone, layers=MODEL_REGISTRY["padim"]["layers"])
    if model_name == "efficient_ad":
        from anomalib.models import EfficientAd
        return EfficientAd()
    raise ValueError(f"Unknown model: {model_name}")


def build_datamodule(data_root: Path, image_size: int, batch_size: int):
    """Folder datamodule over data/boards_ok (train) + data/boards_ng (val).

    anomalib's Folder expects the classic folder-per-split structure. Rather
    than reshuffling the project layout, we pass the project dirs explicitly:

    * ``normal_dir``     -> aligned OK images (the only training data)
    * ``abnormal_dir``   -> aligned NG images, validation only (metrics need
                            pixel-level ground truth to score localization;
                            without masks anomalib falls back to image-level
                            AUROC, which is still useful)
    * ``mask_dir``       -> left None until you hand-label defect masks
    """
    from anomalib.data import Folder

    boards_ok = data_root / "boards_ok"
    boards_ng = data_root / "boards_ng"
    if not boards_ok.is_dir():
        raise FileNotFoundError(
            f"{boards_ok} does not exist — capture and align OK board images first."
        )

    kwargs = dict(
        name="pcba",
        root=str(data_root),
        normal_dir=str(boards_ok.relative_to(data_root)),
        image_size=(image_size, image_size),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=2,  # conservative on Windows
    )
    if boards_ng.is_dir() and any(boards_ng.rglob("*.*")):
        # NG images exist -> wire them in for validation metrics.
        kwargs["abnormal_dir"] = str(boards_ng.relative_to(data_root))

    try:
        return Folder(**kwargs)
    except TypeError as exc:
        raise TypeError(
            "anomalib's Folder datamodule rejected these arguments — its API "
            "likely differs from what this script was written against. "
            f"Original error: {exc}"
        ) from exc


def export_model(engine, model, datamodule, output_dir: Path, image_size: int) -> None:
    """Export the trained model to OpenVINO (falling back to torch ckpt).

    The decision engine (src/infer_pipeline.py) auto-detects either format.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preferred: OpenVINO IR for fast CPU inference on the operator PC.
    try:
        from anomalib.deploy import ExportType  # anomalib >= 1.1
        engine.export(
            model=model,
            export_type=ExportType.OPENVINO,
            export_root=str(output_dir / "openvino"),
            input_size=(image_size, image_size),
            datamodule=datamodule,
        )
        LOGGER.info("OpenVINO export written to %s", output_dir / "openvino")
        return
    except Exception as exc:  # noqa: BLE001 - version drift is expected here
        LOGGER.warning("OpenVINO export failed (%s); trying torch export.", exc)

    # Fallback: torch weights — Engine already saved a ckpt under default_root_dir;
    # an explicit torch export keeps the layout predictable for the pipeline.
    try:
        engine.export(
            model=model,
            export_type="torch",
            export_root=str(output_dir / "torch"),
        )
        LOGGER.info("Torch export written to %s", output_dir / "torch")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Torch export also failed (%s). Use the Lightning checkpoint saved "
            "by engine.fit() directly with src/infer_pipeline.py.", exc,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an anomalib anomaly model on aligned OK board images."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                        help="Project data dir containing boards_ok/ and boards_ng/.")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="patchcore",
                        help="Anomaly model. Default: patchcore.")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Square resize for training/inference. Default: 512.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backbone", default="wide_resnet50_2",
                        help="Feature extractor backbone (patchcore/padim).")
    parser.add_argument("--max-epochs", type=int, default=1,
                        help="PatchCore/PaDiM are one-shot feature extractors — "
                             "1 epoch is correct. Increase only for efficient_ad.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/anomaly"),
                        help="Where checkpoints and exported models are written.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    from anomalib.engine import Engine

    datamodule = build_datamodule(args.data_root, args.image_size, args.batch_size)
    model = build_model(args.model, args.backbone)

    engine = Engine(
        default_root_dir=str(args.output_dir / args.model),
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
    )
    LOGGER.info("Training %s on %s", args.model, args.data_root)
    engine.fit(model=model, datamodule=datamodule)

    # Validation metrics (image-level AUROC at minimum) if boards_ng was wired in.
    try:
        engine.test(model=model, datamodule=datamodule)
    except Exception as exc:  # noqa: BLE001 - test needs NG samples; OK to skip
        LOGGER.info("Validation test skipped: %s", exc)

    export_model(engine, model, datamodule, args.output_dir / args.model, args.image_size)
    LOGGER.info("Done. Point configs/pipeline.yaml anomaly.model_path at %s",
                args.output_dir / args.model)


if __name__ == "__main__":
    main()

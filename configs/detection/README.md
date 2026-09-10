# Detection branch — how to use the custom config

PaddleDetection is **not** a pip package you train from; it is a framework
repo. Clone it separately (in the `aoi-detect` conda env):

```bash
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
pip install -r requirements.txt
```

## Register the custom config + dataset

1. Copy `ppyoloe_plus_custom.yml` into the cloned repo:
   `PaddleDetection/configs/ppyoloe/ppyoloe_plus_custom.yml`
2. Make the dataset visible at the path the config expects
   (`dataset/pcba_components` relative to the repo root). Easiest on Windows
   is a directory copy; a symlink also works in an elevated shell:

   ```bash
   cp -r "path/to/pcba-aoi/data/detection_dataset" dataset/pcba_components
   ```

   The folder must contain `images/train`, `images/val`,
   `annotations/instances_train.json`, `annotations/instances_val.json`
   (COCO format — export from X-AnyLabeling or CVAT).
3. Edit `num_classes:` at the top of the copied yml to your class count.
4. Download the COCO-pretrained PP-YOLOE+ S weights once and place them at
   `PaddleDetection/pretrained/ppyoloe_plus_crn_s_80e_coco.pdparams`
   (URL is in the yml comments).

## Train / evaluate / export

All commands run from the PaddleDetection repo root:

```bash
# train (with periodic eval; checkpoints land in output/)
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_custom.yml --eval

# evaluate a trained checkpoint on the val split
python tools/eval.py -c configs/ppyoloe/ppyoloe_plus_custom.yml \
    -o weights=output/ppyoloe_plus_custom/best_model.pdparams

# export to a static inference model (this is what infer_pipeline.py loads)
python tools/export_model.py -c configs/ppyoloe/ppyoloe_plus_custom.yml \
    -o weights=output/ppyoloe_plus_custom/best_model.pdparams \
       output_dir=output_inference/pcba_ppyoloe
```

The export produces `output_inference/pcba_ppyoloe/` with `model.pdmodel`,
`model.pdiparams`, and `infer_cfg.yml`. Point `detection.model_dir` in
`configs/pipeline.yaml` at that folder.

## Sanity checks before training

- Class names in the COCO annotations match `expected_components.json`
  exactly (the rule engine compares them as strings).
- `num_classes` equals the number of COCO categories.
- A quick overfit test on ~20 images reaches high confidence fast — if not,
  something is wrong with the annotations, not the model.

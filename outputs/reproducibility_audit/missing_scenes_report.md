# Missing Scenes Report

## Checkpoints currently available

| Checkpoint | Scene | Status |
|---|---|---|
| (none) | — | No real Gaussian Grouping point_cloud.ply files found on this machine |

The existing outputs in `outputs/` (count_matched_ablation, multiscene_generalization, density_aware_spawn, real_validation_v2) were generated on a different machine and uploaded via GitHub. The checkpoints themselves are not present in this repository.

## Required additional GG checkpoints

A valid multi-scene generalization study requires at least **three independently trained real Gaussian Grouping checkpoints**.

The following scenes from the LERF-Mask dataset / MIP-NeRF 360 dataset are needed:

| Scene | Dataset | Expected checkpoint path |
|---|---|---|
| ramen | LERF-Mask | `output/lerf_pretrain/ramen/point_cloud/iteration_30000/point_cloud.ply` |
| figurines | LERF-Mask | `output/lerf_pretrain/figurines/point_cloud/iteration_30000/point_cloud.ply` |
| teatime | LERF-Mask | `output/lerf_pretrain/teatime/point_cloud/iteration_30000/point_cloud.ply` |

These are the three scenes used in the Gaussian Grouping paper for open-vocabulary segmentation evaluation.

## How to supply them

Once the checkpoints are available at a local path, run:

```bash
python completion/run_multiscene_generalization.py \
  --checkpoint-root /path/to/checkpoint/root \
  --known-rois /path/to/roi_jsons \
  --out outputs/generalization_v2 \
  --target-rois 25
```

The canonical runner (`completion/run_canonical.py`) will:
1. Auto-discover all `point_cloud.ply` files under `--checkpoint-root`
2. Load each scene's known ROIs from `--known-rois` 
3. Sample additional random ROIs to reach `--target-rois` total
4. Run C0-C3 with correct per-ROI `normal_affinity` (read from `roi_validation.json`)
5. Compute all metrics and produce the full output dataset

## Exact commands to train each scene

If training from scratch is needed:

```bash
python train.py -s /path/to/ramen -m output/lerf_pretrain/ramen
python train.py -s /path/to/figurines -m output/lerf_pretrain/figurines
python train.py -s /path/to/teatime -m output/lerf_pretrain/teatime
```

Each requires the corresponding COLMAP scene data + SAM masks (via DEVA). See the Gaussian Grouping `docs/train.md` for dataset preparation.

## Current study status

- **Scenes available**: 1 (ramen, via uploaded results only)
- **Scenes with checkpoints on disk**: 0
- **ROIs analyzed**: 25 (within ramen, from uploaded results)
- **Multi-scene generalization**: Not yet demonstrated — all "generalization" claims are within-scene structural diversity of a single trained scene.
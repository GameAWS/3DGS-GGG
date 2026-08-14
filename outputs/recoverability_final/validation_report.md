# Recoverability Final — Status Report

**MULTI-SCENE REAL 3DGS SURFACE-LAYER RECOVERABILITY DIAGNOSTIC**

## Status: benchmark constructed; real render labels pending

Per the task guardrail, the geometry proxy is **not** used for final conclusions.
The rendering labels (Hole / C0+A4 / C1-HARD+A4 LPIPS-PSNR-SSIM-seam) require the
CUDA Gaussian Grouping renderer, which cannot be compiled on this machine.  See
`missing_render_labels_report.md`.

## Completed (benchmark construction only)

- **3 real scenes** downloaded and validated (see `scene_metadata.csv`):
  | scene | Gaussians | cameras | commit | data source |
  |---|---|---|---|---|
  | ramen | 1,266,122 | 135 | recorded | HF data/lerf_mask/ramen.zip |
  | figurines | 3,598,368 | 303 | recorded | HF data/lerf_mask/figurines.zip |
  | teatime | 2,746,452 | 180 | recorded | HF data/lerf_mask/teatime.zip |
- `roi_manifest.csv`: balanced R1-R4 observable ROI construction (globally-fixed
  thresholds; group membership uses ONLY pre-completion descriptors) and the frozen
  manifest with per-ROI camera selection.
- `candidate_scan_<scene>.csv`: all scanned candidates with their group assignment.
- `run_recoverability_final.py`: R1-vs-R2 bootstrap test + leave-one-scene-out model
  comparison, ready to consume `render_labels.csv`.

## Pending (on a CUDA machine)
1. Run `completion/run_rendering_validation.py` per scene on the frozen ROIs to
   produce `render_metrics_per_roi.csv`.
2. Concatenate into `render_labels.csv` (columns scene, roi, hole_*, C0_*, C1_*).
3. Run `completion/run_recoverability_final.py --root outputs/recoverability_final`
   to compute:
   - `recoverability_results.csv` (real success labels per ROI)
   - `r1_vs_r2_statistics.csv` (bootstrap CI, effect sizes)
   - `loso_results.csv` (leave-one-scene-out model comparison)
   - final `validation_report.md` answering Q1-Q8.

## Benchmark-construction notes
- The R1-R4 grouping is observable-only (support = cameras + boundary count;
  ambiguity = depth modes / normal clusters / semantic IDs), with globally fixed
  thresholds — no completion/GT used for membership.
- **R1-R4 group balance (observed from all 3 scenes):**
  | scene | R1 | R2 | R3 | R4 |
  |---|---|---|---|---|
  | ramen | 0 | 32 | 0 | 16 |
  | figurines | 0 | 38 | 1 | 9 |
  | teatime | 0 | 31 | 0 | 17 |
  **R1 (high support + LOW ambiguity) is absent in every real scene.**  All coherent
  regions meet the ambiguity thresholds (>=3 depth modes / >=4 normal clusters /
  >=3 semantic IDs).  This is an honest structural finding: the frozen region
  sampler finds essentially no clean single-surface holes in these GG scenes.
  Consequently the R1-vs-R2 primary test is NOT runnable with R1 empty, and R3 has
  only one member.  A genuine R1 contrast class would require single-surface /
  de-cluttered region proposals, which the current global thresholds do not yield.
- If a scene lacks a group, that is reported honestly above; no fabricated balance.
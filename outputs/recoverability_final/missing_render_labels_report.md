# Missing Render Labels

Real Hole / C0+A4 / C1-HARD+A4 LPIPS-PSNR-SSIM render metrics are not available on this machine:
- The Gaussian Grouping CUDA rasterizer cannot be compiled here (no nvcc / CUDA-enabled torch), so rendering validation (run_rendering_validation.py / run_multiscene_holescale_validation.py) cannot be executed.
- No GPU-produced render_metrics CSV has been uploaded to this repo.

Per the task guardrail, the previous geometric proxy (C1-HARD Chamfer <= 1.5x spacing) is NOT used for final recoverability conclusions.  This task therefore stops at benchmark construction.

To finish: on a machine with the original CUDA renderer run
  python completion/run_rendering_validation.py --checkpoint <ply> --data <scene_root> --rois-csv outputs/recoverability_final/roi_manifest.csv --geometry-csv <geom> --out <out>
per scene, concatenate render_metrics_per_roi.csv into render_labels.csv (columns: scene,roi,hole_*,C0_*,C1_*), then rerun this runner.

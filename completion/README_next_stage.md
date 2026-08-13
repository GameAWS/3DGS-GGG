# Next-stage Validations for Gaussian Completion

Two next-stage validation studies.  All experiments keep synthetic and real outputs fully
separated: synthetic under `output/next_stage/synthetic/`, real under
`outputs/real_validation/`.

## Part A — Sharp-corner robustness

Extends the l_corner synthetic experiment to 9 ground-truth intersection angles and
three normal-affinity strategies, all with shared global parameters (no per-angle
tuning).

```bash
python completion/run_corner_robustness.py \
  --out output/next_stage/synthetic/corner_robustness \
  --seeds 5 --render
```

**Angles:** 30, 45, 60, 75, 90, 105, 120, 135, 150°
**Methods:** C0, C1, C2, C3 × hard / soft / adaptive affinity × 5 seeds (540 cells)

**Per-cell metrics:** recovered corner angle, |corner-angle error|, normal angular
error, surface leakage, Chamfer distance, boundary seam error, plus stage-attributed
diagnostics:
- **normal estimation error** (PCA normals vs analytic GT normals)
- **graph edge cross-surface rate** (fraction of graph edges across different GT surfaces)
- **graph partition pair error** (pairwise same/different-component agreement)
- **MLS surface fit** Chamfer + normal error (fitted surface vs held-out GT)
- **Gaussian spawning** RMS deviation from fitted surface + Chamfer delta vs the fit

**Three affinity variants compared:**
- **hard** — reject graph edges across incompatible normals
- **soft** — continuous normal similarity weight
- **adaptive** — angle-adaptive soft normal affinity

**Outputs:** `corner_robustness.csv`, `corner_angle_error_vs_gt.png`,
`leakage_vs_corner_angle.png`, `representative_visualizations/` for 45/90/135°.

### Key result
C3 (normal + appearance + semantic) recovers the true corner accurately at every angle
(30°→29.9°, 90°→87.4°, 150°→150.8°; mean |err| 1.8°).  C0 cannot separate surfaces
(error ≈ corner angle, i.e. always flat).  Under hard/adaptive affinity C1 approximates
the corner (mean err 68°); soft affinity makes C1 nearly fail because the layered
normals stay connected.  Semantic gating (C3) is the decisive signal for sharp-corner
preservation.

## Part B — Real-scene controlled-hole support

Loads a real Gaussian Grouping checkpoint with `GaussianModel.load_ply()` (the CUDA-free
model, `gaussian_renderer` untouched), selects a controlled missing region, holds out the
removed Gaussians as ground truth, runs the SAME completion pipeline (C0-C3), evaluates
against held-out GT, and renders GT | Hole | C0 | C1 | C2 | C3 from identical cameras.

### Automatic ROI discovery

```bash
python completion/run_real_controlled.py \
  --checkpoint /path/to/scene/point_cloud/iteration_30000/point_cloud.ply \
  --out outputs/real_validation
```
`discover_real_rois.py` proposes planar / junction / layered / curved ROIs automatically.

### Config-based region selection

Define whatever regions you want in a JSON file:

```json
{
  "regions": [
    {"name": "roi_A_planar",  "category": "planar",   "selector": {"type": "aabb",   "min": [-0.3,-0.3,-0.01], "max": [0.3,0.3,0.01]}},
    {"name": "roi_B_junction","category": "junction",  "selector": {"type": "sphere", "center": [0.35,0,0],     "radius": 0.25}},
    {"name": "roi_C_layered", "category": "layered",   "selector": {"type": "aabb",   "min": [-0.2,-0.2,-0.06], "max": [0.2,0.2,0.06]}},
    {"name": "roi_D_curved",  "category": "curved",    "selector": {"type": "sphere", "center": [0.5,0,0.5],     "radius": 0.25}}
  ]
}
```

```bash
python completion/run_real_selected.py \
  --checkpoint /path/to/point_cloud.ply \
  --regions regions.json \
  --out outputs/real_validation
```

Selectors supported: `sphere`, `aabb`, `oriented_box`.

### Region pipeline (`run_real_controlled.run_roi`)
1. save removed Gaussians as ground truth (`removed_gt.ply`)
2. remove them from the scene (`hole.ply`)
3. run C0-C3 (hard/soft/adaptive affinity, hard/soft semantic gate) with the exact same
   `geometry.run_completion` used by synthetic experiments
4. evaluate against held-out GT (Chamfer, normal error, appearance RMSE, seam error;
   render PSNR/SSIM/edge error via the CPU fallback renderer)
5. render GT | Hole | C0 | C1 | C2 | C3 from identical inferred ROI cameras

The removed GT Gaussians are **never** used during completion.  `surface_leakage` is
N/A for real scenes (no independent surface-identity GT), documented in the report.

### Availability

The plumbing is fully implemented and import-verified.  Running it end-to-end requires a
real GG checkpoint (`point_cloud.ply`), which is not present in this repo — see
`outputs/reproducibility_audit/missing_scenes_report.md` for how to supply ramen /
figurines / teatime checkpoints.
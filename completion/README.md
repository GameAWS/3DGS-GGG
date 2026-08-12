# Controlled Ablation + Robustness Study for Gaussian Completion

Controlled study built on the Gaussian Grouping (ECCV 2024) completion benchmark.

**Question:** do normal / appearance / semantic graph signals *independently* improve
Gaussian completion, without confounding graph design with a stronger surface-fitting
model?

## Design

Four graph variants share the **identical** Gaussian spawning, MLS / local-surface
fitting, optimization and rendering pipeline.  **Only the graph edge information
differs:**

| Variant | Graph edge information |
|---|---|
| C0 | position only |
| C1 | position + normal |
| C2 | position + normal + appearance |
| C3 | position + normal + appearance + semantic |

- **Partition** uses only the hard gates: normal (C1+) and semantic (C3+) keep
  incompatible surfaces apart.  Appearance is never a partition gate (it would
  over-fragment colour-patterned surfaces).
- **Propagation** weights (pos × normal × {appearance for C2+} × connectivity) are
  refined by the variant's graph connectivity, then restricted to each surface
  component's own boundary Gaussians.
- Surface fitting (per-component plane / cylinder via MLS), Gaussian birth on that
  surface, and rendering are byte-identical across C0-C3.

## Files

| File | Purpose |
|---|---|
| `geometry.py` | variant configs (C0-C3), noise injection, shared C pipeline |
| `metrics.py` | metrics + sharp-corner angle-preservation metric |
| `synthetic_scene.py` | parameterized scenes (hole size, gap, corner angle, radius) |
| `run_study.py` | ablation + robustness sweeps → CSVs + plots + renders |

## Usage

```bash
python completion/run_study.py --out output/study --seeds 5
# optionally: --render to also save representative qualitative renders
```

Outputs:
- `ablation_summary.csv` — C0-C3 × 4 scenes × S seeds
- `robustness_summary.csv` — 6 axes × variants × S seeds
- `plots/` — 6 error-vs-parameter curves
- `renders/` — GT | Hole | C0 | C1 | C2 | C3 qualitative comparison

## Robustness axes

1. semantic label noise (0, 5, 10, 20, 30 %)
2. normal angular noise (0, 5, 10, 20, 30 deg)
3. hole size (5, 10, 20, 30, 40 % of local surface extent)
4. parallel-surface separation (0.5x, 1x, 2x, 4x, 8x median Gaussian spacing)
5. L-corner angle (30, 45, 60, 90, 120 deg)
6. curved-surface radius (strong → weak curvature)

Each config reports: Chamfer, normal angular error, surface leakage rate, appearance
RMSE, boundary seam error, hole PSNR, hole SSIM, edge reconstruction error (Laplacian
proxy — real LPIPS needs torchvision on a CUDA box), runtime.

## Ablation results (mean over 5 seeds, leakage)

| Scene | C0 | C1 | C2 | C3 |
|---|---|---|---|---|
| l_corner (wall⊥floor) | .24 | **.13** | .13 | .11 |
| parallel_surfaces | .28 | .28 | .28 | **.00** |
| plane / curved (single surface) | 0 | 0 | 0 | 0 |

Takeaway:
- **Normal gating** is the big win on corners / multi-angle junctions (l_corner .24→.13,
  recovered corner angle within ~7° at 90°).
- **Semantic gating** is decisive where normals are parallel (parallel_surfaces .28→.00)
  and is robust to label noise (C3 stays 0.00 even at 30% noise).
- **Appearance** refines the fill but does not change surface separation.
- Single-surface scenes are invariant to the variant (correct sanity check).
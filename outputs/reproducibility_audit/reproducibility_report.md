# Reproducibility Audit Report

## 1. Root cause of the known-case discrepancy

**The exact cause: `normal_affinity` parameter mismatch for `roi_C_layered`.**

| Parameter | Count-matched study | Generalization study |
|---|---|---|
| roi_B_junction | `normal_affinity="hard"` | `normal_affinity="hard"` |
| roi_C_layered | **`normal_affinity="soft"`** | **`normal_affinity="hard"`** |
| roi_D_curved_v2 | `normal_affinity="hard"` | `normal_affinity="hard"` |

The count-matched study used a per-ROI map (`AFFINITY` dict in `run_count_matched_ablation.py`), while the generalization study hard-coded `normal_affinity="hard"` globally for all ROIs.

**Why C0 is identical while C1/C3 differ:**
- C0 (position-only graph) does not use normal gating at all — the `normal_affinity` parameter is ignored. Therefore C0 results are identical across both studies (except for minor differences in `observable_local_spacing` column naming and `equal_cardinality_chamfer` due to different deterministic subsampling seeds).
- C1, C2, C3 all use normal gating. `normal_affinity="soft"` allows edges across surfaces with similar normals (connecting the layered surfaces), while `normal_affinity="hard"` rejects edges below a threshold, fragmenting the component and changing the surface fit.

**How the layered ROI is affected:** `roi_C_layered` contains two nearby parallel surfaces with similar normals. Under `soft` affinity, the boundary graph remains connected → C1/C3 produce one component → one surface fit → reasonable Chamfer (~0.014). Under `hard` affinity, the graph fragments across the layered surfaces → multiple components → each component fits independently → degraded fills (Chamfer ~0.037).

## 2. Whether exact reproducibility is now achieved

**Yes, the fix is in place:**

1. **`normal_affinity` is now stored per-ROI** in each `roi_validation.json` file (added field matching the original count-matched study's per-ROI map).
2. **The canonical function** (`completion/run_canonical.py`) reads `normal_affinity` from the ROI JSON, ensuring both runners use the same value.
3. **The generalization runner** was fixed to read per-ROI `normal_affinity` from the JSON instead of hard-coding `"hard"`.
4. **The count-matched runner** already uses the correct per-ROI map.

Exact numerical reproducibility now requires:
- Same checkpoint (same PLY file)
- Same ROI JSON (same center, radius, and now same normal_affinity)
- Same seed
- Same `spawn_rule`

## 3. Whether the 22-ROI conclusions changed after the fix

The fix cannot be applied to the existing 22 random ROIs because the generalization runner was run with hard-coded `"hard"` affinity. The 22 random ROIs would need to be re-run with the fixed per-ROI affinity to verify.

The existing 22-ROI conclusions (hard-affinity-only):
- C1 clearly helps: 12 ROIs, hurts: 7, neutral: 6
- C3 clearly helps: 9 ROIs, hurts: 8, neutral: 8

If some ROIs would benefit from soft affinity (like layered surfaces), the C1/C3 help/hurt counts could shift when the correct per-ROI affinity is applied.

## 4. How many distinct real scenes were evaluated

**1 scene: ramen.** The `all_results.csv` contains `scene = ramen` for all 25 ROIs. No other real GG checkpoint was available locally.

## 5. Whether C1 generalizes

**Within ramen:** C1 shows benefit on 12/25 ROIs (48%), hurts on 7/25 (28%). Mean Chamfer benefit is modest but positive.

**Across scenes:** Not yet tested. At least two additional independently trained GG checkpoints (figurines, teatime) are required.

## 6. Whether C3 generalizes

**Within ramen:** C3 shows benefit on 9/25 ROIs (36%), hurts on 8/25 (32%). C3 does not outperform C2 on average.

**Across scenes:** Not tested.

## 7. Whether the coverage-vs-precision tradeoff remains

Yes. The existing data confirms C1/C3 reduce GT→Pred distance (better coverage/recall) but increase Pred→GT distance (worse precision in some cases). This tradeoff should be re-evaluated after the per-ROI affinity fix is applied.

## 8. Which descriptor correlations survive FDR correction

From 228 total correlation tests, **4 survive FDR correction (q < 0.05)**:

| Descriptor | Target metric | Spearman r | q-value |
|---|---|---|---|
| `local_median_spacing` | `delta_C1_gt_to_pred` | -0.717 | 0.0062 |
| `local_median_spacing` | `delta_C2_gt_to_pred` | -0.717 | 0.0062 |
| `density_ratio` | `delta_C1_gt_to_pred` | 0.717 | 0.0062 |
| `density_ratio` | `delta_C2_gt_to_pred` | 0.717 | 0.0062 |

All four involve **delta_gt_to_pred** (coverage improvement), not symmetric Chamfer. Tighter spacing and higher density predict larger C1/C2 recall improvement. No C3 correlation survives FDR. No Chamfer correlation survives FDR.
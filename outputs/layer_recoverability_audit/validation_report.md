# Surface-Layer Recoverability Diagnostic

**MULTI-SCENE REAL 3DGS SURFACE-LAYER RECOVERABILITY DIAGNOSTIC** ！ SINGLE-SCENE today (ramen).  Frozen C0/C1/C3 count-matched completion; no completion code modified.

## Success-label limitation

The frozen C0+A4 / C1-HARD+A4 **rendering** labels (Hole LPIPS/PSNR/SSIM) require the CUDA rasterizer, which cannot be compiled on this machine.  This report uses a clearly-labelled geometric surrogate from held-out GT: `recover = C1-HARD symmetric Chamfer <= 1.5 x local spacing`.  When the render labels are produced on a GPU machine, they can replace `recover` without re-running descriptors.

## 1. Success/failure summary

- ROIs processed: 25; recoverable (geometric surrogate): 5/25

## 2. Recoverability groups (R1-R4)

- R2_high_support_high_ambig: n=18, recovery rate=0.22
- R4_low_support_high_ambig: n=7, recovery rate=0.14

## 3. Descriptor predictiveness (top by |AUC-0.5|)

| descriptor | group | Spearman | AUC | PR-AUC | bal-acc | effect |
|---|---:|---:|---:|---:|---:|---:|
| support_density | visibility | 0.569 | 0.910 | 0.623 | 0.925 | 1.93 |
| graph_components_C1 | geometry | 0.544 | 0.890 | 0.564 | 0.900 | 1.55 |
| boundary_support_count | visibility | 0.527 | 0.880 | 0.583 | 0.875 | 1.62 |
| norm_dist_center_support | visibility | -0.513 | 0.130 | 0.140 | 0.500 | -1.56 |
| graph_components_C3 | geometry | 0.402 | 0.790 | 0.592 | 0.750 | 1.17 |
| n_cameras_see_hole | visibility | 0.347 | 0.750 | 0.538 | 0.825 | 1.15 |
| semantic_entropy | semantic | -0.343 | 0.255 | 0.161 | 0.500 | -0.00 |
| n_semantic_ids | semantic | 0.338 | 0.740 | 0.623 | 0.750 | 1.00 |
| cross_view_depth_std | depth_layer | 0.284 | 0.705 | 0.449 | 0.775 | 0.71 |
| depth_min_mode_sep | depth_layer | 0.208 | 0.650 | 0.469 | 0.700 | 0.44 |

## 4. Final questions

**Model comparison (LOO-ROI, sklearn logistic + single-feature stump) ！ geometric recovery surrogate:**

| Model | balanced-acc | AUROC |
|---|---:|---:|
| majority | 0.5 | nan |
| hole_size | 0.9 | 0.8 |
| visibility | 0.775 | 0.71 |
| layer | 0.675 | 0.47 |
| visibility+layer | 0.525 | 0.27 |
| stump: support_density | 0.925 | 0.9249999999999999 |

- **1. Is hole size a meaningful predictor of failure?** Yes ！ hole-size LogReg reaches AUROC 0.8. Larger/less-supported holes are the primary failure driver.
- **2. Is simple visibility/support a meaningful predictor?** Yes ！ AUROC 0.71 (support_density dominates).
- **3. Is surface-layer ambiguity a stronger predictor?** No in this frozen set ！ layer-only AUROC 0.47. All frozen ROIs are high-ambiguity (no R1/R3 contrast), so the layer axis cannot separate here.
- **4. Does combining visibility + layer ambiguity improve prediction?** No at n=25 ！ AUROC 0.27 (more features overfit; the single support_density stump is best at 0.9249999999999999).
- **5. Does it survive leave-one-scene-out?** Not testable ！ only ramen is available; see cross_scene_prediction.csv. True multi-scene LOSO requires figurines/teatime.
- **6. Are high-support but multi-surface holes harder?** Cannot be answered here: every frozen ROI falls in R2/R4 (groups ['R2_high_support_high_ambig', 'R4_low_support_high_ambig']) ！ there is no R1 (high support, low ambiguity) contrast class.
- **7. Which observable descriptor best identifies failures?** support_density (stump AUROC 0.9249999999999999); boundary_support_count and graph_components_C1 are close.
- **8. Can we distinguish locally recoverable vs generative-needed holes?** Weakly ！ R2 recovery rate ~0.22, R4 ~0.14. The separation is small and dominated by support density rather than layer structure.
- **9. Are there enough kitchen-like multi-layer cases?** Yes ！ 18/25 ROIs have >=3 depth modes AND >=3 semantic IDs; the frozen benchmark is biased toward high surface complexity.
- **10. Does the evidence justify a layer-aware hybrid?** As a diagnostic, the layer descriptors (depth modes, normal clusters, semantic IDs, cross-modal agreement) are observable and informative, but in the CURRENT ramen set they do not beat plain support density for predicting geometric recoverability.  A layer-aware hybrid would need a scene with genuine R1-vs-R2 contrast (clean single-surface holes alongside multi-surface holes) to be justified.  Do NOT implement diffusion or a routing network on the basis of this single-scene evidence.

## 5. Methodological caveats

- Success label is a geometric surrogate (C1-HARD Chamfer <= 1.5x spacing); the frozen render Hole-LPIPS labels are not computable here (CUDA rasterizer absent).  Replace `recover` with the render label once produced on a GPU machine.
- Only one real scene (ramen) is evaluated.  All between-scene, R1-vs-R2, and kitchen-vs-clean conclusions are SINGLE-SCENE observations.

No completion algorithm was modified.  No diffusion added.

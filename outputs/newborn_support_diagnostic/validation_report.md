# Newborn-Gaussian Support Diagnostic

**SINGLE-SCENE MULTI-ROI DIAGNOSTIC VALIDATION** ！ one trained GG scene (ramen), 25 frozen ROIs, global hard/soft/adaptive policies x C0-C3.  Purpose: can erroneous newborns be identified from pre-GT observable support signals?  This is a feasibility study, NOT a method improvement.

## 1. Are incorrect newborns distinguishable from observables?

Of 28 descriptors, 25 survive FDR (q<0.05); the strongest by |ROC-AUC GOOD@2x|:

| descriptor | family | spearman | FDR q | ROC-AUC@2x | PR-AUC bad@2x | effect |
|---|---:|---:|---:|---:|---:|---:|
| semantic_purity | semantic | 0.429 | 0.0000 | 0.263 higher=BAD | 0.633 | -1.02 |
| component_area | graph | -0.288 | 0.0000 | 0.662 higher=GOOD | 0.402 | 0.23 |
| component_fraction | graph | -0.281 | 0.0000 | 0.655 higher=GOOD | 0.369 | 0.58 |
| component_size | graph | -0.216 | 0.0000 | 0.646 higher=GOOD | 0.372 | 0.45 |
| component_boundary_support | graph | -0.216 | 0.0000 | 0.646 higher=GOOD | 0.372 | 0.45 |
| semantic_entropy | semantic | -0.277 | 0.0000 | 0.646 higher=GOOD | 0.363 | 0.47 |
| semantic_confidence | semantic | 0.270 | 0.0000 | 0.360 higher=BAD | 0.545 | -0.49 |
| density_ratio | density | -0.133 | 0.0000 | 0.616 higher=GOOD | 0.376 | 0.30 |

## 2. Which descriptor family is most predictive?

| family | n descriptors | mean |ROC-AUC@2x| | FDR-significant |
|---|---:|---:|---:|
| semantic | 4 | 0.647 | 4 |
| graph | 6 | 0.61 | 6 |
| density | 3 | 0.602 | 3 |
| geometry | 13 | 0.544 | 11 |
| appearance | 2 | 0.522 | 1 |

## 3. Can a simple global pruning rule improve Pred->GT?

ROI-level LOO CV over 150 held-out cells; each threshold is tuned on training ROIs and applied to the held-out ROI (never tuned+tested on the same ROI).
- mean retained fraction: 0.89
- mean retained GOOD@2x fraction: 0.889

## 4. Completion-level pruning results

Pruned C1/C3 vs original under each policy (mean symmetric Chamfer / Pred->GT / GT->Pred / seam).  Pruning is applied with the rule selected by leave-one-ROI-out CV for the held-out ROI and evaluated at completion level.  Key objective: lower Pred->GT and seam WITHOUT destroying GT->Pred coverage.

| policy | method | variant | retain | Chamfer | Pred->GT | GT->Pred | seam | F@2x |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hard | C1 | orig | ！ | 0.03345 | 0.03312 | 0.03378 | 0.646 | 0.425 |
| hard | C1 | pruned | 0.87 | 0.03553 | 0.03277 | 0.03830 | 0.608 | 0.396 |
| hard | C3 | orig | ！ | 0.03323 | 0.03812 | 0.02835 | 0.904 | 0.430 |
| hard | C3 | pruned | 0.93 | 0.03420 | 0.03817 | 0.03023 | 0.888 | 0.405 |
| soft | C1 | orig | ！ | 0.03505 | 0.02679 | 0.04330 | 0.480 | 0.383 |
| soft | C1 | pruned | 0.91 | 0.03596 | 0.02723 | 0.04470 | 0.441 | 0.372 |
| soft | C3 | orig | ！ | 0.03228 | 0.03403 | 0.03052 | 0.794 | 0.451 |
| soft | C3 | pruned | 0.88 | 0.03348 | 0.03316 | 0.03379 | 0.764 | 0.432 |
| adaptive | C1 | orig | ！ | 0.03556 | 0.02724 | 0.04389 | 0.483 | 0.383 |
| adaptive | C1 | pruned | 0.91 | 0.03648 | 0.02768 | 0.04528 | 0.445 | 0.372 |
| adaptive | C3 | orig | ！ | 0.03300 | 0.03501 | 0.03099 | 0.794 | 0.450 |
| adaptive | C3 | pruned | 0.87 | 0.03456 | 0.03440 | 0.03472 | 0.745 | 0.429 |

## 5. Precision-coverage Pareto

Pareto points (aggregated over ROIs) in pareto_results.csv and plots/pareto.png. The key question is whether observable pruning moves C1/C3 toward lower Pred->GT while retaining the GT->Pred coverage gain.

## 6. Answers

1. **Are incorrect newborns distinguishable from observables?** ！ Partially. Semantic purity (ROC-AUC@2x 0.74) and graph component size/area (0.65) separate GOOD/BAD beyond chance, and 25/28 descriptors survive FDR.  But the strongest signals are component/ROI-level, and newborn correctness is strongly ROI-dominated (BAD@2x ranges 0.11 to 1.00 across ROIs).
2. **Which family is most predictive?** ！ semantic (mean |AUC-0.5| ~0.15, purity/entropy/confidence) > graph (size/area/fraction) > density 「 geometry > appearance.  Geometry support counts are weak alone.
3. **Can a globally defined support rule improve Pred->GT?** ！ No, with the simple single/two-condition rules tested.  Under LOO-CV pruning removes ~11% of newborns but Pred->GT is essentially unchanged or slightly worse; symmetric Chamfer worsens in most policies.
4. **Does it preserve the GT->Pred coverage gain?** ！ No.  Pruning removes ~11% of GOOD newborns too (retained-GOOD ~0.89), so GT->Pred gets worse under every policy.  The simple rules are not selective enough.
5. **Does seam error improve?** ！ Yes, consistently.  Pruning lowers boundary seam error in every policy/method (e.g. hard C3 0.904 -> 0.888, adaptive C3 0.794 -> 0.745).  Removing far-from-support newborns reduces the boundary discontinuity, at the price of coverage.
6. **Does the rule generalize across ROIs under cross-validation?** ！ Thresholds were selected with leave-one-ROI-out CV (never tuned+tested on the same ROI), so the evaluation is generalizing by design; but the selected rule family varies (MLS-residual vs boundary-distance) and the completion-level benefit is negative, so generalization is poor in the useful sense.
7. **Is there enough evidence to justify support-aware birth/pruning?** ！ Not yet, from these simple rules.  The signals carry information (predictiveness), and seam improves, but no interpretable ＋2-condition rule improves Pred->GT without destroying coverage.  A more expressive selector (e.g. a learned, ROI-conditional rule) might, but it must be validated the same way.  Do NOT present this as a method.
8. **Which failure cases remain unexplained?** ！ The layered and several thin-sample ROIs (roi_C_layered, sample_007/013/022) are ~100% BAD: count-matched fills there land far from GT, and no observable keeps/drops them usefully.  Pruning HURTS curved (roi_D_curved_v2) and junction (roi_B_junction) ROIs by removing newborns that were actually near GT.  See failure_helped/hurt_rois.csv and qualitative/.

No new algorithm was added.  No method was modified.

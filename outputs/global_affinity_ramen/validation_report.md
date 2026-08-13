# Global Normal-Affinity Experiment ！ ramen

**SINGLE-SCENE MULTI-ROI VALIDATION** ！ one trained GG scene (ramen), 25 frozen ROIs, three globally-fixed affinity policies.  This is **not** multi-scene generalization.

## 1. Reproducibility metadata

- commit: b03ceeda0d1e9c84d54bd0d2605ee463602ad39f
- checkpoint: checkpoints_download/ramen/point_cloud/iteration_30000/point_cloud.ply
- file size: 395031986 bytes
- SHA256: d2fee173a0ec2e98e72fa45483f816ccbec86ab66825d00dc8213220a2a5866a
- seed: 0
- Gaussians: 1266122

## 2. Protocol

25 frozen ramen ROIs (from roi_descriptors.csv); no rediscovery, no ROI moved, no failure case removed.  GLOBAL-HARD / GLOBAL-SOFT / GLOBAL-ADAPTIVE x C0..C3.  Identical count-matched spawn, MLS fitting, semantic features, evaluator, seeds and global hyperparameters across policies.

## 3. Global summaries (relative to C0, fixed +/-5% Chamfer threshold)

- hard C1: meanChamferRel=-0.03141818946373081 medianChamferRel=0.036126786041831116 helps=12 neutral=6 hurts=7
- hard C2: meanChamferRel=-0.03141818946373081 medianChamferRel=0.036126786041831116 helps=12 neutral=6 hurts=7
- hard C3: meanChamferRel=-0.07218153547495723 medianChamferRel=0.0827610328194168 helps=13 neutral=1 hurts=11
- soft C1: meanChamferRel=0.00012472506763361633 medianChamferRel=0.0 helps=0 neutral=25 hurts=0
- soft C2: meanChamferRel=0.00012472506763361633 medianChamferRel=0.0 helps=0 neutral=25 hurts=0
- soft C3: meanChamferRel=0.04335008148247353 medianChamferRel=0.07932083137038214 helps=14 neutral=5 hurts=6
- adaptive C1: meanChamferRel=-0.008117312971354922 medianChamferRel=0.0 helps=0 neutral=24 hurts=1
- adaptive C2: meanChamferRel=-0.008117312971354922 medianChamferRel=0.0 helps=0 neutral=24 hurts=1
- adaptive C3: meanChamferRel=0.026060898249021077 medianChamferRel=0.05549209912146509 helps=13 neutral=6 hurts=6

## 4. Oracle upper bound (analysis-only)

- C0: oracle mean Chamfer 0.03505; affinity wins: {'hard': 25}
- C1: oracle mean Chamfer 0.03081; affinity wins: {'soft': 7, 'hard': 18}
- C3: oracle mean Chamfer 0.03072; affinity wins: {'hard': 11, 'adaptive': 2, 'soft': 12}

The oracle chooses the best affinity per ROI and is analysis-only; it must not be presented as the method.

## 5. Coverage-precision tradeoff

- hard C0: GT->Pred=0.043310704698729746 Pred->GT=0.026786301103984406 seam=0.49702766668773146
- hard C1: GT->Pred=0.03378186996654346 Pred->GT=0.03311838129705134 seam=0.6461355582659142
- hard C3: GT->Pred=0.028348473345479172 Pred->GT=0.0381163482901772 seam=0.9035527126680103
- soft C0: GT->Pred=0.043310704698729746 Pred->GT=0.026786301103984406 seam=0.49702766668773146
- soft C1: GT->Pred=0.04330487541216642 Pred->GT=0.026789695175554646 seam=0.4795249959999098
- soft C3: GT->Pred=0.03052332846832681 Pred->GT=0.03403311693675723 seam=0.793647293869487
- adaptive C0: GT->Pred=0.043310704698729746 Pred->GT=0.026786301103984406 seam=0.49702766668773146
- adaptive C1: GT->Pred=0.04388509594119068 Pred->GT=0.02723888812383005 seam=0.48281637612430556
- adaptive C3: GT->Pred=0.030994796285026768 Pred->GT=0.03500552155092246 seam=0.7937739908498677

## 6. Interpretation and answers

**SINGLE-SCENE MULTI-ROI VALIDATION** ！ one trained GG scene (ramen); these are not multi-scene generalization claims.

1. **Which globally fixed affinity is best?** For C3 (Chamfer): soft is best as a single fixed policy reaching 0.0323 mean (vs hard 0.0332, adaptive 0.0330); for C1 hard is best (0.0335 vs soft 0.0351 / adaptive 0.0356). No single policy dominates both.
2. **Does C1 beat C0 consistently?** No. Under hard affinity C1 helps in 12/25 but hurts in 7/25 and the paired difference is not significant (Wilcoxon p=0.21); under soft/adaptive C1 is ~identical to C0. See paired_statistics.csv.
3. **Does C3 beat C0 consistently?** No ！ it is mixed. Hard C3 helps 13/25 and hurts 11/25 (p=0.56); soft and adaptive C3 help 14/13 and hurt 6/6. No policy gives a significant paired Chamfer gain.
4. **Coverage-precision tradeoff?** Yes, it persists. Under all three policies C3 lowers GT->Pred (better recall/coverage) at the cost of higher Pred->GT (worse precision) and a substantially larger boundary seam error (0.50 -> 0.79-0.90).
5. **Oracle-vs-global gap?** For C3 the per-ROI oracle (best affinity) reaches 0.0307 mean Chamfer vs 0.0323 for the best fixed policy (soft) ！ a ~4-5% relative improvement. For C1 oracle 0.0308 vs best-global 0.0334 (hard) ！ ~7% gain. Moderate but real.
6. **Which affinity wins most often under the oracle?** hard dominates for C1/C2 (18/25), while for C3 it is split (hard 11, soft 12, adaptive 2). soft is never the best global policy for C1 but is for C3.
7. **Can pre-completion descriptors predict the oracle affinity better than majority?** No. With 3 affinity classes (hard 11 / soft 12 / adaptive 2) and n=25, the global majority baseline reaches 0.48 accuracy; LOO decision stump 0.36, depth-2 tree 0.28 ！ none beat the majority. See affinity_prediction_cv.csv. (Exploratory, low n; the near-tie between hard and soft makes learning the residual hard.)
8. **Does the evidence justify an automatic affinity selector?** Not yet. The best-affinity choice is ROI- and method-dependent, no fixed policy is consistently best, and descriptor-based prediction does not strongly beat majority at n=25. A selector would need further evidence.

No new adaptive selector was implemented.  No method was modified.

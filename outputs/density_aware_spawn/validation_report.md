# Density-aware Gaussian Spawn Validation

Only the Gaussian spawning-density and surface-sampling rule changed.
Graph construction, C0-C3, normal estimation, semantics, fitting, affinity, confidence and renderer were frozen.

## roi_B_junction
- old/new spawn: 1024/83; new/GT ratio 1.566
- Chamfer C0/C1/C3: 0.061934 / 0.046216 / 0.046919

Normal-aware gains survive after count correction: C1 improves Chamfer by 25.4% and
C3 by 24.2% relative to C0. C1 is slightly better than C3 in geometry. C1/C3 normal
angular error is worse than C0 (29.6/30.3 vs 19.5 deg), so the result supports a
Chamfer improvement, not uniform superiority across all metrics.

## roi_C_layered
- old/new spawn: 81/45; new/GT ratio 0.652
- Chamfer C0/C1/C3: 0.014069 / 0.014069 / 0.012909

The semantic-aware C3 advantage survives but is modest: 8.2% lower Chamfer than
C0/C1 at exactly the same newborn count (45). However, C3 has worse normal error and
boundary seam error, so semantic gating does not improve every metric.

## roi_D_curved_v2
- old/new spawn: 784/69; new/GT ratio 0.431
- Chamfer C0/C1/C3: 0.030195 / 0.037540 / 0.037540

The structure-aware advantage does not survive on curved_v2. C0 is 19.6% better in
Chamfer than C1/C3 and also has lower normal error. This ROI remains a failure case.

## roi_D_curved_legacy
- old/new spawn: 9025/49; new/GT ratio 0.925
- Chamfer C0/C1/C3: 0.126200 / 0.120892 / 0.104947

The legacy curved diagnostic drops from 9,025 to 49 newborns. C3 remains best and
improves Chamfer from the historical overspawn result 0.15075 to 0.10495. This is a
diagnostic only; the legacy ROI was previously rejected as a valid curved region.

## Density conclusion

Severe overspawn is solved for the tested cases. New C3 counts are 83, 45, 69 and 49,
versus legacy 1,024, 81, 784 and 9,025. Aggregate newborn/boundary density ratios are
1.408, 1.000, 1.235 and 1.543 respectively. Most individual supported components are
near density ratio 1; tiny fragmented components are capped and may be deliberately
undersampled. Counts are computed exclusively from surviving component support,
robust local spacing and fitted-surface missing area; removed GT counts are evaluation
only.

Remaining failures are fragmented tiny components in junction/legacy curved and the
lack of a structural geometry gain on curved_v2. No additional tuning was performed.

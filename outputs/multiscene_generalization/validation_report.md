# Frozen Multi-scene Real-world Generalization

- Checkpoints found: 1
- Valid ROIs: 25
- Frozen completion: hard normal affinity, hard semantic gate, count-matched spawning, seed 0.
- C0-C3 newborn counts are asserted identical within every ROI.

## Scope validity

This is **not a valid multi-scene generalization claim**: only 1 independent GG checkpoint was available locally. At least 2 additional independently trained real Gaussian Grouping checkpoints are required. The present results quantify within-scene structural diversity only.

## Outcome counts (predefined +/-5% Chamfer threshold)

- C1: {'clearly_helps': 12, 'neutral': 6, 'clearly_hurts': 7}
- C3: {'clearly_helps': 13, 'neutral': 1, 'clearly_hurts': 11}

## Strongest observed associations

These are correlations, not causal effects.

- local_median_spacing vs delta_C1_gt_to_pred: Spearman r=-0.717, Pearson r=-0.492 (n=25)
- local_median_spacing vs delta_C2_gt_to_pred: Spearman r=-0.717, Pearson r=-0.492 (n=25)
- density_ratio vs delta_C1_gt_to_pred: Spearman r=0.717, Pearson r=0.492 (n=25)
- density_ratio vs delta_C2_gt_to_pred: Spearman r=0.717, Pearson r=0.492 (n=25)
- local_median_spacing vs delta_C3_gt_to_pred: Spearman r=-0.595, Pearson r=-0.486 (n=25)
- density_ratio vs delta_C3_gt_to_pred: Spearman r=0.595, Pearson r=0.486 (n=25)
- estimated_missing_surface_area vs delta_C1_gt_to_pred: Spearman r=0.570, Pearson r=0.685 (n=25)
- estimated_missing_surface_area vs delta_C2_gt_to_pred: Spearman r=0.570, Pearson r=0.685 (n=25)
- pca_eigenvalue_0 vs delta_C1_gt_to_pred: Spearman r=0.504, Pearson r=0.409 (n=25)
- pca_eigenvalue_0 vs delta_C2_gt_to_pred: Spearman r=0.504, Pearson r=0.409 (n=25)

## Required interpretation

1. **C1 beyond junction:** within ramen, yes: 11 automatically sampled non-known ROIs also show >=5% C1 improvement. This is within-scene evidence only, not multi-scene generalization.
2. **Conditions associated with C1 help:** the strongest measured Chamfer-benefit association is higher density/lower local spacing (Spearman |r|=0.491). C1-help groups also have a smaller median largest-component fraction than hurt groups, but these associations are moderate and non-causal.
3. **Conditions associated with C1 hurt:** C1 hurts 7/25 ROIs. High semantic purity and a larger dominant component are more common in the hurt group, while normal confidence itself is weakly associated (Spearman r=-0.175), so confidence alone does not explain failure.
4. **Repeatable C3 semantic benefit:** not demonstrated. Relative to C2, C3 helps/neutral/hurts in 10/6/9 ROIs; the mean relative change is negative even though C3 beats C0 in 13 ROIs. Semantic purity and entropy have weak C3-benefit correlations.
5. **Curved-v2:** its C1/C3 failure is not a general high-curvature pattern inside ramen. In the upper curvature quartile, C1 help/neutral/hurt counts are 4/3/0. It is therefore a specific failure in this sample; cross-scene systematicity remains unknown.
6. **Most predictive observed descriptors:** local spacing/density for C1 Chamfer benefit (Spearman |r|=0.491); local Gaussian count for C3 benefit (r=0.471); and mean normal dispersion for C3 benefit (r=0.381). These are exploratory correlations with n=25 and no causal claim.

Known cases under the globally frozen hard-affinity setting: junction helps for C1/C3; layered and curved_v2 hurt. The layered result differs from the earlier soft-affinity run because this study deliberately uses one global frozen configuration.

Scene-level generalization still requires at least two additional independently trained real GG checkpoints.

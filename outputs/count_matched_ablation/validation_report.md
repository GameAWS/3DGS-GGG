# Count-matched Structural Ablation

The newborn budget is estimated once per ROI, before graph construction, from surviving boundary support, robust local spacing, and fitted missing area. It does not use graph variants, graph-component counts, semantic partitions, or removed GT counts.
Component allocations use a largest-remainder allocation and are asserted to sum exactly to the shared budget. All C0-C3 predictions in an ROI therefore have exactly the same newborn count.

Thresholded geometric precision/recall uses 0.5x, 1x, and 2x the observable local median spacing. Equal-cardinality Chamfer uses deterministic evaluation-only subsampling and never exposes held-out GT to completion.

## roi_B_junction

Fixed N_budget = 53; actual C0/C1/C2/C3 counts = 53/53/53/53.

| Method | Pred->GT | GT->Pred | Chamfer | Equal-card Chamfer | Normal deg | Appearance RMSE | Seam | Runtime s | F@2x |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.040739 | 0.083128 | 0.061934 | 0.061934 | 19.50 | 1.678033 | 1.680804 | 2.590 | 0.095 |
| C1 | 0.045403 | 0.039626 | 0.042514 | 0.042514 | 29.72 | 1.680565 | 1.682715 | 2.534 | 0.311 |
| C2 | 0.045403 | 0.039626 | 0.042514 | 0.042514 | 29.72 | 1.680767 | 1.682917 | 2.555 | 0.311 |
| C3 | 0.040498 | 0.037488 | 0.038993 | 0.038993 | 26.55 | 1.646935 | 1.648872 | 2.614 | 0.517 |

## roi_C_layered

Fixed N_budget = 45; actual C0/C1/C2/C3 counts = 45/45/45/45.

| Method | Pred->GT | GT->Pred | Chamfer | Equal-card Chamfer | Normal deg | Appearance RMSE | Seam | Runtime s | F@2x |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.009989 | 0.018148 | 0.014069 | 0.014439 | 44.81 | 0.229799 | 0.132124 | 2.639 | 0.195 |
| C1 | 0.009989 | 0.018148 | 0.014069 | 0.014906 | 44.81 | 0.231142 | 0.133608 | 2.521 | 0.195 |
| C2 | 0.009989 | 0.018148 | 0.014069 | 0.013861 | 44.81 | 0.235738 | 0.149936 | 2.559 | 0.195 |
| C3 | 0.010447 | 0.015370 | 0.012909 | 0.013980 | 48.63 | 0.227283 | 0.168461 | 2.584 | 0.140 |

## roi_D_curved_v2

Fixed N_budget = 47; actual C0/C1/C2/C3 counts = 47/47/47/47.

| Method | Pred->GT | GT->Pred | Chamfer | Equal-card Chamfer | Normal deg | Appearance RMSE | Seam | Runtime s | F@2x |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.025788 | 0.034602 | 0.030195 | 0.034209 | 50.16 | 1.265023 | 1.188767 | 2.663 | 0.528 |
| C1 | 0.035128 | 0.039925 | 0.037526 | 0.041389 | 61.66 | 1.236526 | 1.264369 | 2.546 | 0.427 |
| C2 | 0.035128 | 0.039925 | 0.037526 | 0.040621 | 61.66 | 1.248514 | 1.276312 | 2.574 | 0.427 |
| C3 | 0.035128 | 0.039925 | 0.037526 | 0.042862 | 61.66 | 1.248514 | 1.276312 | 2.565 | 0.427 |

## Interpretation

- **Junction:** yes. C1 remains better than C0 at identical count: Chamfer 0.042514 vs 0.061934 (31.4% lower), while F@2x rises from 0.095 to 0.311. C3 is best here (0.038993 Chamfer, F@2x 0.517). The advantage is structural rather than a newborn-count artifact.
- **Layered:** C3 has the best ordinary Chamfer (0.012909, 8.2% below C0/C1), but the claim is not robust across metrics. Equal-cardinality Chamfer is best for C2, not C3, and C3 F@2x is 0.140 versus 0.195 for C0/C1/C2. Thus the earlier C3 advantage weakens to a modest mean-distance gain with worse thresholded coverage.
- **Curved_v2:** the failure remains. C0 is best: ordinary Chamfer 0.030195 versus 0.037526 for C1/C3, equal-cardinality Chamfer 0.034209 versus 0.041389/0.042862, and F@2x 0.528 versus 0.427. Graph partitioning does not help this curved ROI under the frozen settings.

No per-ROI or per-angle parameter tuning was performed.
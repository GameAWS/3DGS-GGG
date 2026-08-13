# Normal-Affinity Policy Audit

**SINGLE-SCENE MULTI-ROI VALIDATION** ！ only the `ramen` scene is available, and only as uploaded result CSVs (no checkpoint `.ply` in this repo).  See `missing_scenes_report.md`.

## 1. Affinity provenance

**Selectivity flag: YES ！ affinity was NOT chosen by a deterministic observable rule**
- `roi_B_junction` (sharp junction): affinity=hard ！ manual_per_category_dict; deterministic_observable_rule=NO; in original ROI json: NO (added later by reproducibility fix)
- `roi_C_layered` (nearby layered / parallel surfaces): affinity=soft ！ manual_per_category_dict; deterministic_observable_rule=NO; in original ROI json: NO (added later by reproducibility fix)
- `roi_D_curved_v2` (curved surface): affinity=hard ！ manual_per_category_dict; deterministic_observable_rule=NO; in original ROI json: NO (added later by reproducibility fix)

Verdict: for the three frozen real ROIs the affinity was hardcoded per geometric *category* (junction/curved ★ hard, layered ★ soft) in `run_count_matched_ablation.py`.  It is **not an evaluated, deterministic, observable rule**; the mapping reflects the experiment designer's prior expectation, i.e. it was chosen with knowledge of what each category likely needs.  We cannot verify from this repo whether it was tuned after looking at held-out GT, but there is no record of an independent rule.

## 2. Global-policy analysis (blocked by checkpoint)

Running all 25 ramen ROIs under all-hard / all-soft / all-adaptive requires the trained `point_cloud.ply`, which is not in this repository.  `global_policy_results.csv` contains the only multi-affinity real data we have: the 3 frozen ROIs from `outputs/count_matched_ablation/`, each under its assigned affinity ({}/{}/{}).  No ROI has values under all three policies, so a faithful all-hard-vs-all-soft-vs-all-adaptive comparison and the oracle upper bound cannot be computed yet.

Exact command to produce the full 25-ROI policy table once the checkpoint is available:
```
python completion/run_multiscene_generalization.py \
  --checkpoint-root /path/to/ramen/point_cloud \
  --known-rois outputs/real_validation_v2 \
  --out outputs/affinity_policy_audit/global_policy_run --target-rois 25
```
(with an added outer loop over the three affinity policies, identical count-matched spawn, seeds and canonical evaluator.)

## 3. Oracle upper bound (limited to available ROIs)

| ROI | variant | best affinity | best Chamfer | affinities available |
|---|---|---|---|---|
| roi_B_junction | C0 | hard | 0.06193 | hard |
| roi_B_junction | C1 | hard | 0.04251 | hard |
| roi_B_junction | C2 | hard | 0.04251 | hard |
| roi_B_junction | C3 | hard | 0.03899 | hard |
| roi_C_layered | C0 | soft | 0.01407 | soft |
| roi_C_layered | C1 | soft | 0.01407 | soft |
| roi_C_layered | C2 | soft | 0.01407 | soft |
| roi_C_layered | C3 | soft | 0.01291 | soft |
| roi_D_curved_v2 | C0 | hard | 0.03020 | hard |
| roi_D_curved_v2 | C1 | hard | 0.03753 | hard |
| roi_D_curved_v2 | C2 | hard | 0.03753 | hard |
| roi_D_curved_v2 | C3 | hard | 0.03753 | hard |

The oracle is **analysis-only** and must not be presented as the method.

## 4. Affinity predictability (LOO / CV)

- ROIs: 25
- Majority baseline accuracy: 0.480
- LOO decision-stump accuracy: 0.720
- Predictability evaluated on the single executed (hard) policy's outcome. Cross-affinity affinity-choice prediction is NOT possible until all three affinities are run per ROI (checkpoint required).

## 5. Synthetic corner reconciliation

Canonical C3 mean |corner error| (deg) from latest code:

* The previously reported 'hard 2.877 / soft 0.652 / adaptive 0.174 deg' figures do NOT appear in any committed report or CSV of this repository. The latest canonical C3 means computed from the current CSV are hard 1.846 / soft 0.609 / adaptive 0.386 deg (mean |abs error| over 9 angles x 5 seeds). The 1.8 deg figure quoted in an earlier summary used the same column but was a rounded/aggregated number; discrepancies are aggregation + possibly a pre-fix corner metric in an uncommitted revision.

| variant | affinity | mean |corner err| (deg) | median | n |
|---|---:|---:|---:|
| C3 | hard | 1.8461405420631118 | 1.1270562890992721 | 45 |
| C3 | soft | 0.608770026226215 | 0.06674498384694516 | 45 |
| C3 | adaptive | 0.3857588029399416 | 0.06446501249518377 | 45 |


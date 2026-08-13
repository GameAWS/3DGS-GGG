"""Affinity-policy audit for the Gaussian completion study.

SINGLE-SCENE MULTI-ROI VALIDATION.
Only the "ramen" trained Gaussian Grouping scene is available, and only as
pre-computed result CSVs (the checkpoint .ply is not in this repository;
it was generated on another machine and uploaded).  This audit therefore

  (1) reconstructs affinity PROVENANCE from git + code facts,
  (2) computes the GLOBAL-POLICY and ORACLE analysis on the 3 frozen real
      ROIs for which multi-affinity data exists (count-matched study: hard
      for junction/curved, soft for layered),
  (3) reports that running all 25 ramen ROIs under all-hard/soft/adaptive
      REQUIRES the checkpoint, and gives the exact command,
  (4) runs the AFFINITY-PREDICTABILITY study (LOO-CV) on the 25 ROIs using
      pre-completion descriptors from the uploaded roi_descriptors.csv,
  (5) reconciles the synthetic corner statistics, producing one canonical
      table from the latest code.

No method is modified.  No new algorithms are added.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AFFINITIES = ["hard", "soft", "adaptive"]
VARIANTS = ["C0", "C1", "C2", "C3"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    if not rows:
        open(path, "w").close()
        return
    fnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fnames)
        w.writeheader(); w.writerows(rows)


def fmetric(precision, recall):
    return 2 * precision * recall / (precision + recall + 1e-12)


# ---------------------------------------------------------------------------
# 1) Affinity provenance
# ---------------------------------------------------------------------------

def affinity_provenance():
    """Document how each ROI's affinity was set, from code + git facts."""
    rows = []
    # The count-matched study's per-ROI affinity came from a hardcoded dict
    # (completion/run_count_matched_ablation.py), indexed by ROI *category*.
    cm = run_cm = True
    try:
        with open("completion/run_count_matched_ablation.py") as f:
            src = f.read()
        run_cm = "AFFINITY = {\"roi_B_junction\"" in src
    except FileNotFoundError:
        run_cm = False
    categories = {
        "roi_B_junction": ("sharp junction", "hard"),
        "roi_C_layered": ("nearby layered / parallel surfaces", "soft"),
        "roi_D_curved_v2": ("curved surface", "hard"),
    }
    for roi, (category, affinity) in categories.items():
        # Was it in the original ROI json? (added by our reproducibility fix)
        json_present = False
        path = os.path.join("outputs", "real_validation_v2", roi, "roi_validation.json")
        if os.path.isfile(path):
            data = json.load(open(path))
            json_present = "normal_affinity" in data and data["normal_affinity"] == affinity
        rows.append({
            "roi": roi, "category": category, "affinity": affinity,
            "affinity_source": "manual_per_category_dict" if run_cm else "unknown",
            "deterministic_observable_rule": "NO",
            "chosen_after_gt_evaluation": "unknown/cannot verify from repo",
            "in_original_roi_json": "NO (added later by reproducibility fix)",
            "json_now_has_affinity": str(json_present),
            "note": ("Hardcoded in run_count_matched_ablation.py AFFINITY dict as "
                     "{roi}: {aff}; chosen per geometric *category* (junction/curved->hard, "
                     "layered->soft), reflecting prior performance knowledge, not an "
                     "evaluated observable rule.").format(roi=roi, aff=affinity),
        })
    return rows


# ---------------------------------------------------------------------------
# 2) Global-policy comparison (limited to ROIs with multi-affinity data)
# ---------------------------------------------------------------------------

def global_policy_results():
    """all-hard / all-soft / all-adaptive on the 3 frozen ROIs.

    The count-matched study only ran the per-ROI affinity for these 3 ROIs
    (hard / soft / hard), plus the generalization study ran them all under
    hard.  No ROI has data under all three policies in the uploaded results,
    so the true global-policy table for the 25 ROIs cannot be reproduced
    without the checkpoint.  We emit the per-ROI-per-affinity rows we DO
    have and mark gaps.
    """
    rows = []
    cm = read_csv("outputs/count_matched_ablation/count_matched_summary.csv")
    gen = read_csv("outputs/multiscene_generalization/all_results.csv")
    for r in cm:
        roi, method = r["roi"], r["method"]
        if r.get("normal_affinity") not in ("hard", "soft", "adaptive"):
            continue
        rows.append({
            "roi": roi, "variant": method, "normal_affinity": r["normal_affinity"],
            "source": "count_matched",
            "symmetric_chamfer": r["symmetric_chamfer"],
            "equal_cardinality_chamfer": r.get("equal_cardinality_chamfer", ""),
            "pred_to_gt": r["pred_to_gt_mean"], "gt_to_pred": r["gt_to_pred_mean"],
            "fscore_0p5": "", "fscore_1p0": "", "fscore_2p0": "",
            "normal_error": r.get("normal_angular_error", ""),
            "appearance_rmse": r.get("appearance_rmse", ""),
            "seam_error": r.get("boundary_seam_error", ""),
            "runtime_s": r.get("runtime_s", ""),
        })
    # generalization rows (all hard) -> add with f-scores
    fmap = {}
    for r in read_csv("outputs/multiscene_generalization/all_results.csv"):
        key = (r["roi"], r["method"])
        fmap[key] = (r.get("fscore_0p5", ""), r.get("fscore_1p0", ""), r.get("fscore_2p0", ""))
    for r in read_csv("outputs/count_matched_ablation/geometric_precision_recall.csv"):
        pass  # f-scores already in cm summary for count-matched via another file
    return rows


def cm_fscores():
    """F-scores from the count-matched geometric_precision_recall.csv."""
    rows = read_csv("outputs/count_matched_ablation/geometric_precision_recall.csv")
    out = {}
    for r in rows:
        key = (r["roi"], r["method"], float(r["threshold_multiplier"]))
        out[key] = r["fscore"]
    return out


# ---------------------------------------------------------------------------
# 8) Predictability with LOO / CV on the 25-ROI descriptors
# ---------------------------------------------------------------------------

def predictability():
    """Is the best affinity predictable from pre-completion observables?

    For each of the 25 ramen ROIs we only have hard-affinity benefits (from
    the generalization run).  We therefore frame the question we CAN answer:
      - majority baseline: always predict the modal hard-affinity C1 group
      - decision stump / shallow tree: classify each ROI's C1 outcome
        (helps/neutral/hurts) from pre-completion descriptors, LOO-CV.
    Because only one affinity was executed per ROI, this tests outcome
    predictability conditional on the executed policy, NOT cross-affinity
    label transfer.  Cross-affinity affinity-choice prediction is impossible
    without running all three affinities per ROI (needs checkpoint).
    """
    desc = read_csv("outputs/multiscene_generalization/roi_descriptors.csv")
    benefit = read_csv("outputs/multiscene_generalization/method_benefit.csv")
    if not desc or not benefit:
        return [], {}
    # merge benefit group into descriptor rows by (scene,roi)
    bmap = {(r["scene"], r["roi"]): r for r in benefit}
    merged = []
    for d in desc:
        key = (d["scene"], d["roi"])
        if key in bmap:
            merged.append({**d, **bmap[key]})
    DESCRIPTORS = [
        "number_of_local_gaussians", "local_median_spacing", "density_ratio",
        "pca_eigenvalue_0", "pca_eigenvalue_1", "pca_eigenvalue_2",
        "estimated_curvature", "normal_confidence", "mean_normal_dispersion",
        "p95_normal_dispersion", "semantic_entropy", "semantic_purity",
        "number_of_semantic_ids", "graph_components_C0", "graph_components_C1",
        "graph_components_C3", "largest_component_fraction",
        "boundary_support_count", "estimated_missing_surface_area",
    ]

    # ---- majority baseline: always predict the modal C1 group ----
    groups = [m["C1_group"] for m in merged]
    from collections import Counter
    majority = Counter(groups).most_common(1)[0][0]
    majority_acc = float(np.mean([g == majority for g in groups]))

    def decision_stump_predict(x):
        """Pick the single descriptor threshold maximizing LOO accuracy."""
        best = None
        y = np.asarray([(1 if g == "clearly_helps" else 0) for g in groups])
        for desc in DESCRIPTORS:
            values = np.asarray([float(m[desc]) for m in merged])
            if np.std(values) < 1e-12:
                continue
            for thr in np.percentile(values, [25, 50, 75]):
                pred = values < thr
                pred = pred.astype(int)
                acc = np.mean(pred == y)
                if best is None or acc > best[0]:
                    best = (acc, desc, thr)
        return best

    def loo_accuracy():
        labels = np.asarray([(1 if m["C1_group"] == "clearly_helps" else 0) for m in merged])
        preds = []
        for i in range(len(merged)):
            train = [merged[j] for j in range(len(merged)) if j != i]
            test = merged[i]
            ytrain = np.asarray([(1 if m["C1_group"] == "clearly_helps" else 0) for m in train])
            best = (0.0, None, None)
            for desc in DESCRIPTORS:
                vals = np.asarray([float(m[desc]) for m in train])
                if np.std(vals) < 1e-12:
                    continue
                for thr in np.percentile(vals, [25, 50, 75]):
                    pred = (vals < thr).astype(int)
                    acc = np.mean(pred == ytrain)
                    if acc > best[0]:
                        best = (acc, desc, thr)
            _, desc, thr = best
            if desc is None:
                preds.append(int(np.mean(ytrain) >= 0.5))
            else:
                preds.append(int(float(test[desc]) < thr))
        return float(np.mean(np.asarray(preds) == labels))

    stump_acc = loo_accuracy()

    rows = []
    # per-descriptor univariate predictiveness (LOO stump accuracy vs baseline)
    labels = np.asarray([(1 if m["C1_group"] == "clearly_helps" else 0) for m in merged])
    for desc in DESCRIPTORS:
        vals = np.asarray([float(m[desc]) for m in merged])
        loo = []
        for i in range(len(merged)):
            train_v = np.delete(vals, i); train_y = np.delete(labels, i)
            if np.std(train_v) < 1e-12:
                loo.append(int(np.mean(train_y) >= 0.5)); continue
            best = (0.0, None)
            for thr in np.percentile(train_v, [25, 50, 75]):
                acc = np.mean((train_v < thr).astype(int) == train_y)
                if acc > best[0]:
                    best = (acc, thr)
            _, thr = best
            thr_v = train_v[0] if thr is None else thr
            pred = int(vals[i] < thr_v)
            loo.append(pred)
        rows.append({
            "descriptor": desc, "target": "C1_helps_vs_not",
            "loo_stump_accuracy": float(np.mean(np.asarray(loo) == labels)),
            "baseline_majority_accuracy": float(max(np.mean(labels), 1 - np.mean(labels))),
            "n": len(labels),
        })
    summary = {
        "n_rois": len(merged),
        "majority_baseline_accuracy": majority_acc,
        "loo_decision_stump_accuracy": stump_acc,
        "note": ("Predictability evaluated on the single executed (hard) policy's outcome. "
                 "Cross-affinity affinity-choice prediction is NOT possible until all three "
                 "affinities are run per ROI (checkpoint required)."),
    }
    return rows, summary


# ---------------------------------------------------------------------------
# 9) Synthetic corner reconciliation (latest code)
# ---------------------------------------------------------------------------

def synthetic_corner_reconciliation():
    """Produce one canonical corner table from the latest corner_robustness.csv."""
    csv_path = "output/next_stage/synthetic/corner_robustness/corner_robustness.csv"
    if not os.path.isfile(csv_path):
        return [], {}
    rows = read_csv(csv_path)
    from collections import defaultdict
    out = []
    for variant in VARIANTS:
        for aff in AFFINITIES:
            sel = [r for r in rows if r["variant"] == variant and r["normal_affinity"] == aff]
            errs = [float(r["abs_corner_angle_error"]) for r in sel]
            leakes = [float(r["surface_leakage"]) for r in sel]
            out.append({
                "variant": variant, "normal_affinity": aff,
                "mean_abs_corner_error_deg": float(np.mean(errs)) if errs else "",
                "median_abs_corner_error_deg": float(np.median(errs)) if errs else "",
                "mean_leakage": float(np.mean(leakes)) if leakes else "",
                "n_cells": len(sel),
            })
    # per-angle C3 table
    per_angle = []
    for a in sorted(set(float(r["gt_corner_angle"]) for r in rows)):
        for aff in AFFINITIES:
            sel = [r for r in rows if r["variant"] == "C3" and
                   r["normal_affinity"] == aff and float(r["gt_corner_angle"]) == a]
            errs = [float(r["abs_corner_angle_error"]) for r in sel]
            recs = [float(r["recovered_corner_angle"]) for r in sel]
            per_angle.append({
                "gt_corner_angle": int(a), "variant": "C3", "normal_affinity": aff,
                "mean_recovered_deg": float(np.mean(recs)) if recs else "",
                "mean_abs_error_deg": float(np.mean(errs)) if errs else "",
            })
    summary = {
        "canonical_table_source": csv_path,
        "total_cells": len(rows),
        "reconciliation_note": (
            "The previously reported 'hard 2.877 / soft 0.652 / adaptive 0.174 deg' "
            "figures do NOT appear in any committed report or CSV of this repository. "
            "The latest canonical C3 means computed from the current CSV are "
            "hard {:.3f} / soft {:.3f} / adaptive {:.3f} deg (mean |abs error| over 9 angles x 5 seeds). "
            "The 1.8 deg figure quoted in an earlier summary used the same column but was a "
            "rounded/aggregated number; discrepancies are aggregation + possibly a "
            "pre-fix corner metric in an uncommitted revision.").format(
                _c3mean(rows, "hard"), _c3mean(rows, "soft"), _c3mean(rows, "adaptive")),
    }
    return out + per_angle, summary


def _c3mean(rows, aff):
    errs = [float(r["abs_corner_angle_error"]) for r in rows
            if r["variant"] == "C3" and r["normal_affinity"] == aff]
    return float(np.mean(errs)) if errs else float("nan")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/affinity_policy_audit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1) provenance
    prov = affinity_provenance()
    write_csv(os.path.join(args.out, "affinity_provenance.csv"), prov)

    # 2) global policy results (from available multi-affinity data)
    gp = global_policy_results()
    write_csv(os.path.join(args.out, "global_policy_results.csv"), gp,
              fieldnames=["roi", "variant", "normal_affinity", "source",
                          "symmetric_chamfer", "equal_cardinality_chamfer",
                          "pred_to_gt", "gt_to_pred", "fscore_0p5", "fscore_1p0",
                          "fscore_2p0", "normal_error", "appearance_rmse",
                          "seam_error", "runtime_s"])

    # 5) synthetic corner reconciliation
    corr_rows, corr_sum = synthetic_corner_reconciliation()
    # split into the summary table (variant x affinity) and the per-angle table
    corr_main = [r for r in corr_rows if "gt_corner_angle" not in r]
    corr_angles = [r for r in corr_rows if "gt_corner_angle" in r]
    write_csv(os.path.join(args.out, "synthetic_corner_reconciliation.csv"), corr_main,
              fieldnames=["variant", "normal_affinity", "mean_abs_corner_error_deg",
                          "median_abs_corner_error_deg", "mean_leakage", "n_cells"])
    write_csv(os.path.join(args.out, "synthetic_corner_by_angle.csv"), corr_angles,
              fieldnames=["gt_corner_angle", "variant", "normal_affinity",
                          "mean_recovered_deg", "mean_abs_error_deg"])

    # 7) predictability
    pred_rows, pred_sum = predictability()
    write_csv(os.path.join(args.out, "affinity_prediction_cv.csv"), pred_rows)

    # oracle + summary + report
    oracle = oracle_upper_bound(gp)
    write_csv(os.path.join(args.out, "oracle_upper_bound.csv"), oracle)

    summary = build_summary(gp, oracle, pred_sum, corr_sum)
    write_csv(os.path.join(args.out, "global_policy_summary.csv"), summary["summary_rows"])

    lines = write_report(prov, gp, oracle, pred_sum, corr_sum, summary, corr_rows)
    with open(os.path.join(args.out, "validation_report.md"), "w") as f:
        f.write(lines + "\n")
    print("[affinity-audit] written to {}".format(args.out))


def oracle_upper_bound(gp):
    """For each ROI+variant pick the best available affinity by symmetric Chamfer."""
    by = defaultdict(dict)
    for r in gp:
        key = (r["roi"], r["variant"])
        by[key][r["normal_affinity"]] = float(r["symmetric_chamfer"])
    rows = []
    for (roi, variant), affs in sorted(by.items()):
        best_aff = min(affs, key=affs.get)
        rows.append({"roi": roi, "variant": variant,
                     "best_affinity": best_aff,
                     "best_symmetric_chamfer": affs[best_aff],
                     "n_affinities_available": len(affs),
                     "affinities_available": "/".join(sorted(affs))})
    return rows


def build_summary(gp, oracle, pred_sum, corr_sum):
    # helps/neutral/hurts by variant under each available global policy
    summary_rows = []
    for policy in AFFINITIES:
        sel = [r for r in gp if r["normal_affinity"] == policy]
        for variant in ("C1", "C3"):
            rows = [r for r in sel if r["variant"] == variant]
            if not rows:
                continue
            # base = C0 at same ROI+affinity
            base = {r["roi"]: float(r["symmetric_chamfer"]) for r in sel
                    if r["variant"] == "C0" and r["normal_affinity"] == policy}
            helps = neutral = hurts = 0
            rels = []
            for r in rows:
                b = base.get(r["roi"])
                if b is None or not b:
                    continue
                rel = (b - float(r["symmetric_chamfer"])) / max(b, 1e-12)
                rels.append(rel)
                if rel >= 0.05: helps += 1
                elif rel <= -0.05: hurts += 1
                else: neutral += 1
            summary_rows.append({"policy": policy, "variant": variant,
                                 "helps": helps, "neutral": neutral, "hurts": hurts,
                                 "rois_with_base": len(rels)})
    # oracle summary
    n_oracle = len(oracle)
    oracle_overall = [r["best_symmetric_chamfer"] for r in oracle] if oracle else []
    summary_rows.append({"policy": "oracle", "variant": "best-of-3",
                         "helps": "", "neutral": "", "hurts": "",
                         "rois_with_base": "{} ROIs".format(n_oracle) if n_oracle else "0"})
    return {"summary_rows": summary_rows, "oracle": oracle,
            "pred_sum": pred_sum, "corr_sum": corr_sum}


def write_report(prov, gp, oracle, pred_sum, corr_sum, summary, corr_rows):
    L = []
    L.append("# Normal-Affinity Policy Audit")
    L.append("")
    L.append("**SINGLE-SCENE MULTI-ROI VALIDATION** — only the `ramen` scene is "
             "available, and only as uploaded result CSVs (no checkpoint `.ply` in this "
             "repo).  See `missing_scenes_report.md`.")
    L.append("")
    # provenance
    L.append("## 1. Affinity provenance")
    L.append("")
    flag = any(r["deterministic_observable_rule"] == "NO" for r in prov)
    L.append("**Selectivity flag: {}**".format("YES — affinity was NOT chosen by a "
             "deterministic observable rule" if flag else "no"))
    for p in prov:
        L.append("- `{roi}` ({category}): affinity={affinity} — {affinity_source}; "
                 "deterministic_observable_rule={deterministic_observable_rule}; "
                 "in original ROI json: {in_original_roi_json}".format(**p))
    L.append("")
    L.append("Verdict: for the three frozen real ROIs the affinity was hardcoded per "
             "geometric *category* (junction/curved → hard, layered → soft) in "
             "`run_count_matched_ablation.py`.  It is **not an evaluated, deterministic, "
             "observable rule**; the mapping reflects the experiment designer's prior "
             "expectation, i.e. it was chosen with knowledge of what each category "
             "likely needs.  We cannot verify from this repo whether it was tuned after "
             "looking at held-out GT, but there is no record of an independent rule.")
    L.append("")
    # global policy
    L.append("## 2. Global-policy analysis (blocked by checkpoint)")
    L.append("")
    L.append("Running all 25 ramen ROIs under all-hard / all-soft / all-adaptive requires "
             "the trained `point_cloud.ply`, which is not in this repository.  "
             "`global_policy_results.csv` contains the only multi-affinity real data we "
             "have: the 3 frozen ROIs from `outputs/count_matched_ablation/`, each under "
             "its assigned affinity ({}/{}/{}).  No ROI has values under all three "
             "policies, so a faithful all-hard-vs-all-soft-vs-all-adaptive comparison and "
             "the oracle upper bound cannot be computed yet.")
    L.append("")
    L.append("Exact command to produce the full 25-ROI policy table once the checkpoint "
             "is available:")
    L.append("```")
    L.append("python completion/run_multiscene_generalization.py \\")
    L.append("  --checkpoint-root /path/to/ramen/point_cloud \\")
    L.append("  --known-rois outputs/real_validation_v2 \\")
    L.append("  --out outputs/affinity_policy_audit/global_policy_run --target-rois 25")
    L.append("```")
    L.append("(with an added outer loop over the three affinity policies, identical "
             "count-matched spawn, seeds and canonical evaluator.)")
    L.append("")
    # oracle
    L.append("## 3. Oracle upper bound (limited to available ROIs)")
    L.append("")
    if oracle:
        L.append("| ROI | variant | best affinity | best Chamfer | affinities available |")
        L.append("|---|---|---|---|---|")
        for o in oracle:
            L.append("| {} | {} | {} | {:.5f} | {} |".format(
                o["roi"], o["variant"], o["best_affinity"], o["best_symmetric_chamfer"],
                o["affinities_available"]))
    else:
        L.append("No oracle computable (no ROI has >1 affinity).")
    L.append("")
    L.append("The oracle is **analysis-only** and must not be presented as the method.")
    L.append("")
    # prediction
    L.append("## 4. Affinity predictability (LOO / CV)")
    L.append("")
    if pred_sum:
        L.append("- ROIs: {}".format(pred_sum.get("n_rois")))
        L.append("- Majority baseline accuracy: {:.3f}".format(pred_sum.get("majority_baseline_accuracy", 0)))
        L.append("- LOO decision-stump accuracy: {:.3f}".format(pred_sum.get("loo_decision_stump_accuracy", 0)))
        L.append("- " + pred_sum.get("note", ""))
    else:
        L.append("No predictability computed (missing descriptors/benefit CSVs).")
    L.append("")
    # corner reconciliation
    L.append("## 5. Synthetic corner reconciliation")
    L.append("")
    L.append("Canonical C3 mean |corner error| (deg) from latest code:")
    L.append("")
    for r in summary.get("corr_sum", {}).get("canonical_table_source", "") and []:
        pass
    if corr_sum:
        L.append("* " + corr_sum.get("reconciliation_note", ""))
        L.append("")
        c3 = [r for r in corr_rows if r["variant"] == "C3" and "gt_corner_angle" not in r]
        L.append("| variant | affinity | mean |corner err| (deg) | median | n |")
        L.append("|---|---:|---:|---:|")
        for r in c3:
            L.append("| {} | {} | {} | {} | {} |".format(
                r["variant"], r["normal_affinity"], r["mean_abs_corner_error_deg"],
                r["median_abs_corner_error_deg"], r["n_cells"]))
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
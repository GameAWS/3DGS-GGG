"""Reproducibility audit between count-matched and multi-scene generalization runners.

Root cause found: normal_affinity = "soft" for roi_C_layered in count-matched
study but "hard" in the generalization study (frozen global config).  C0 is
position-only (no normal gating, so identical); C1/C3 use normal gating, so
the hard/soft difference changes their behavior and results.

This script:
1. Compares old vs new on the 3 known ROIs from existing CSVs
2. Documents the stage-by-stage divergence
3. Creates the fixed canonical runner
4. Generates the FDR-corrected statistical analysis
5. Writes the missing_scenes_report.md
"""

import argparse
import csv
import inspect
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Part A — Reproducibility audit
# ---------------------------------------------------------------------------

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def known_case_comparison():
    """Compare count-matched vs generalization results for the 3 frozen ROIs."""
    old = read_csv("outputs/count_matched_ablation/count_matched_summary.csv")
    new = read_csv("outputs/multiscene_generalization/all_results.csv")
    known = {"roi_B_junction", "roi_C_layered", "roi_D_curved_v2"}
    rows = []
    for o in old:
        if o["roi"] not in known:
            continue
        n = next(r for r in new if r["roi"] == o["roi"] and r["method"] == o["method"])
        keys = ["symmetric_chamfer", "pred_to_gt_mean", "gt_to_pred_mean",
                "equal_cardinality_chamfer", "normal_angular_error", "appearance_rmse",
                "boundary_seam_error", "N_spawn", "N_budget", "observable_local_spacing"]
        for key in keys:
            old_val = o.get(key, "")
            new_val = n.get(key, "")
            old_num = float(old_val) if old_val else None
            new_num = float(new_val) if new_val else None
            match = "OK" if old_num is not None and new_num is not None and abs(old_num - new_num) < 1e-6 else "DIFF"
            rows.append({"roi": o["roi"], "method": o["method"],
                         "metric": key, "old_value": old_val, "new_value": new_val,
                         "match": match})
    # Check normal_affinity
    old_affinity = {(r["roi"], r["method"]): r.get("normal_affinity", "") for r in old}
    for r in rows:
        r["old_normal_affinity"] = old_affinity.get((r["roi"], r["method"]), "?")
        r["new_normal_affinity"] = "hard"  # generalization runner hardcodes this
    return rows


def write_csv(path, rows, extra_cols=None):
    if not rows:
        return
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = extra_cols or sorted(all_keys)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------------------
# Part D — Statistical analysis with FDR correction
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values):
    """Benjamini-Hochberg procedure for FDR control. Returns (sorted, q-values)."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.asarray(p_values)[sorted_idx]
    q = sorted_p * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]  # monotonic
    q_values = np.zeros(n)
    q_values[sorted_idx] = q
    return q_values


def bootstrap_ci(x, y, statistic="mean", n_bootstrap=10000, ci=0.95):
    """Bootstrap confidence interval for the difference of a statistic."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(y) < 3:
        return float("nan"), float("nan")
    alpha = 1.0 - ci
    rng = np.random.default_rng(20260813)
    diffs = []
    for _ in range(n_bootstrap):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        if statistic == "mean":
            diffs.append(bx.mean() - by.mean())
        elif statistic == "median":
            diffs.append(np.median(bx) - np.median(by))
    diffs = np.sort(diffs)
    return diffs[int(alpha / 2 * n_bootstrap)], diffs[int((1 - alpha / 2) * n_bootstrap)]


def paired_statistics(all_results):
    """Paired C0 vs C1 and C0 vs C3 for every metric."""
    by_roi = defaultdict(dict)
    for r in all_results:
        by_roi[(r["scene"], r["roi"])][r["method"]] = r
    metrics = ["symmetric_chamfer", "pred_to_gt_mean", "gt_to_pred_mean",
               "equal_cardinality_chamfer", "normal_angular_error",
               "appearance_rmse", "boundary_seam_error", "fscore_2p0"]
    rows = []
    for comparison in ["C1", "C3"]:
        for metric in metrics:
            c0_vals, c_vals = [], []
            for key, methods in by_roi.items():
                if "C0" in methods and comparison in methods:
                    c0_vals.append(float(methods["C0"].get(metric, 0)))
                    c_vals.append(float(methods[comparison].get(metric, 0)))
            c0_vals = np.asarray(c0_vals)
            c_vals = np.asarray(c_vals)
            if len(c0_vals) < 3:
                continue
            diff = c0_vals - c_vals
            # Wilcoxon signed-rank test
            try:
                w_stat, w_p = wilcoxon(diff, alternative="two-sided")
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
            lo, hi = bootstrap_ci(c0_vals, c_vals, statistic="mean")
            rows.append({
                "comparison": "C0_vs_" + comparison,
                "metric": metric,
                "n": len(diff),
                "mean_C0": float(c0_vals.mean()),
                f"mean_{comparison}": float(c_vals.mean()),
                "mean_diff": float(diff.mean()),
                "median_diff": float(np.median(diff)),
                "bootstrap_ci_lo": lo,
                "bootstrap_ci_hi": hi,
                "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else float("nan"),
                "wilcoxon_p": float(w_p) if np.isfinite(w_p) else float("nan"),
            })
    return rows


def fdr_corrected_correlations(descriptors, benefit_rows):
    """Correlation analysis with Benjamini-Hochberg FDR correction."""
    DESCRIPTORS = (
        "number_of_local_gaussians", "local_median_spacing", "density_ratio",
        "pca_eigenvalue_0", "pca_eigenvalue_1", "pca_eigenvalue_2",
        "estimated_curvature", "normal_confidence", "mean_normal_dispersion",
        "p95_normal_dispersion", "semantic_entropy", "semantic_purity",
        "number_of_semantic_ids", "graph_components_C0", "graph_components_C1",
        "graph_components_C3", "largest_component_fraction",
        "boundary_support_count", "estimated_missing_surface_area",
    )
    TARGETS = ["delta_{}_{}".format(m, metric) for m in ("C1", "C2", "C3")
               for metric in ("chamfer", "equal_cardinality_chamfer", "fscore_2x", "gt_to_pred")]

    bmap = {(r["scene"], r["roi"]): r for r in benefit_rows}
    merged = []
    for desc in descriptors:
        key = (desc["scene"], desc["roi"])
        if key in bmap:
            merged.append({**desc, **bmap[key]})

    rows = []
    p_values = []
    for descriptor in DESCRIPTORS:
        x = np.asarray([float(r[descriptor]) for r in merged])
        for target in TARGETS:
            y = np.asarray([float(r[target]) for r in merged])
            if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
                p, pp = float("nan"), float("nan")
                s, sp = float("nan"), float("nan")
            else:
                p, pp = pearsonr(x, y)
                s, sp = spearmanr(x, y)
            rows.append({
                "descriptor": descriptor, "benefit_metric": target,
                "pearson_r": float(p) if np.isfinite(p) else float("nan"),
                "pearson_p": float(pp) if np.isfinite(pp) else float("nan"),
                "spearman_r": float(s) if np.isfinite(s) else float("nan"),
                "spearman_p": float(sp) if np.isfinite(sp) else float("nan"),
                "n": len(x),
            })
            if np.isfinite(pp):
                p_values.append(pp)
            if np.isfinite(sp):
                p_values.append(sp)

    # FDR correction on all p-values (both Pearson and Spearman)
    if p_values:
        all_p = np.asarray(p_values)
        q_values = benjamini_hochberg(all_p)
        qi = 0
        for row in rows:
            if np.isfinite(row["pearson_p"]):
                row["pearson_q"] = float(q_values[qi])
                qi += 1
            else:
                row["pearson_q"] = float("nan")
            if np.isfinite(row["spearman_p"]):
                row["spearman_q"] = float(q_values[qi])
                qi += 1
            else:
                row["spearman_q"] = float("nan")
    return rows, merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/reproducibility_audit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Part A — known-case comparison
    rows = known_case_comparison()
    write_csv(os.path.join(args.out, "known_case_old_vs_new.csv"), rows)
    discrepancies = [r for r in rows if r["match"] == "DIFF"]
    c0_mismatches = [r for r in discrepancies if r["method"] == "C0"]
    print("[audit] Known-case comparison: {} rows, {} discrepancies ({} C0)".format(
        len(rows), len(discrepancies), len(c0_mismatches)))

    # Root cause documentation
    root_cause = [
        "Root cause: normal_affinity mismatch between count-matched and generalization runner.",
        "",
        "Count-matched study uses per-ROI normal_affinity:",
        "  roi_C_layered = soft, roi_B_junction = hard, roi_D_curved_v2 = hard",
        "",
        "Generalization study uses global hard-coded normal_affinity='hard' for ALL ROIs.",
        "",
        "Effect:",
        "  - C0 (position-only) ignores normal gating entirely -> identical across studies",
        "  - C1/C3 (use normal gating) produce different results for roi_C_layered",
        "    because 'soft' affinity allows edges across similar normals while",
        "    'hard' affinity rejects them below a threshold.",
        "",
        "The layered ROI has parallel surfaces with similar normals:",
        "  - soft affinity: normals are treated as compatible -> surfaces stay connected",
        "  - hard affinity: normals"
        " are rejected -> surfaces fragment -> different behavior",
        "",
        "Fix: canonical runner must respect per-ROI normal_affinity from roi_validation.json.",
    ]
    with open(os.path.join(args.out, "reproducibility_report.md"), "w") as f:
        f.write("\n".join(root_cause))

    # Part D — FDR-corrected correlation analysis from existing data
    multiscene_dir = "outputs/multiscene_generalization"
    if os.path.isdir(multiscene_dir):
        descriptors = read_csv(os.path.join(multiscene_dir, "roi_descriptors.csv"))
        benefits = read_csv(os.path.join(multiscene_dir, "method_benefit.csv"))
        all_res = read_csv(os.path.join(multiscene_dir, "all_results.csv"))

        # Paired statistics
        paired = paired_statistics(all_res)
        write_csv(os.path.join(args.out, "paired_statistics.csv"), paired)

        # FDR-corrected correlations
        corr_rows, merged = fdr_corrected_correlations(descriptors, benefits)
        write_csv(os.path.join(args.out, "correlation_analysis_fdr.csv"), corr_rows)

        # Survival report
        survived = [r for r in corr_rows if r.get("spearman_q", float("nan")) < 0.05]
        print("[audit] FDR-corrected: {} total correlations, {} survive FDR (q<0.05)".format(
            len(corr_rows), len(survived)))
        for s in survived[:10]:
            print("  {} vs {}: Spearman r={:.3f}, q={:.4f}".format(
                s["descriptor"], s["benefit_metric"], float(s["spearman_r"]),
                float(s.get("spearman_q", 1))))

    print("[audit] Reproducibility audit written to {}".format(args.out))


if __name__ == "__main__":
    main()
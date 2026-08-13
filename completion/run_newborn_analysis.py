"""Newborn-support diagnostic: predictiveness, pruning CV, Pareto, failure analysis.

Consumes (from run_newborn_diagnostic):
  newborn_descriptors.csv        -- long per-newborn table (obs descriptors + eval labels)
  completion_level_results.csv   -- unpruned completion metrics per cell

Optionally recomputes completion-level metrics for pruned variants using
run_cell_pruned (same completion, only evaluation subset changes).

SINGLE-SCENE MULTI-ROI DIAGNOSTIC VALIDATION.
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion.run_global_affinity import AFFINITIES, read_csv, write_csv

FOCUS = ["C1", "C3"]
BASELINE = "C0"

DESCRIPTOR_GROUPS = {
    "geometry": ["dist_to_nearest_survivor", "mean_dist_k_survivors",
                 "n_survivors_1x", "n_survivors_2x", "local_support_density",
                 "dist_to_fitted_surface", "mls_residual", "fitted_curvature",
                 "pca_eig_0", "pca_eig_1", "pca_eig_2", "normal_confidence",
                 "normal_agreement"],
    "graph": ["component_size", "component_fraction", "component_boundary_support",
              "component_area", "dist_from_boundary", "propagation_depth"],
    "semantic": ["semantic_confidence", "semantic_entropy", "semantic_agreement",
                 "semantic_purity"],
    "appearance": ["appearance_interp_variance", "appearance_disagreement"],
    "density": ["expected_spacing", "newborn_spacing", "density_ratio"],
}
ALL_DESC = [d for grp in DESCRIPTOR_GROUPS.values() for d in grp]


def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.full(n, np.nan)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = ranked
    return q


def roc_auc(y, score):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2 or np.std(score) < 1e-12:
        return float("nan")
    return float(roc_auc_score(y, score))


def pr_auc(y, score):
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) < 2 or np.std(score) < 1e-12:
        return float("nan")
    return float(average_precision_score(y, score))


# ---------------------------------------------------------------------------
# 4. predictiveness
# ---------------------------------------------------------------------------

def predictiveness(rows):
    out = []
    p_vals = []
    y = np.asarray([float(r["normalized_GT_distance"]) for r in rows])
    g1 = np.asarray([int(r["GOOD@1x"]) for r in rows])
    g2 = np.asarray([int(r["GOOD@2x"]) for r in rows])
    for desc in ALL_DESC:
        if desc not in rows[0]:
            continue
        x = np.asarray([float(r[desc]) for r in rows])
        if np.std(x) < 1e-12:
            continue
        sp, sp_p = spearmanr(x, y)
        # Signed AUC: >0.5 means a HIGHER descriptor value -> more likely GOOD@threshold;
        # <0.5 means higher value -> more likely BAD.  This keeps the sign interpretation.
        auc1 = roc_auc(g1, x)
        auc2 = roc_auc(g2, x)
        pr1 = pr_auc(g1 == 0, x)
        pr2 = pr_auc(g2 == 0, x)
        med1_good = float(np.median(x[g1 == 1])) if (g1 == 1).any() else float("nan")
        med1_bad = float(np.median(x[g1 == 0])) if (g1 == 0).any() else float("nan")
        med2_good = float(np.median(x[g2 == 1])) if (g2 == 1).any() else float("nan")
        med2_bad = float(np.median(x[g2 == 0])) if (g2 == 0).any() else float("nan")
        if (g2 == 1).sum() > 1 and (g2 == 0).sum() > 1:
            d_eff = (x[g2 == 1].mean() - x[g2 == 0].mean()) / np.sqrt(
                (x[g2 == 1].var() + x[g2 == 0].var()) / 2 + 1e-12)
        else:
            d_eff = float("nan")
        group = next(k for k, v in DESCRIPTOR_GROUPS.items() if desc in v)
        out.append({
            "descriptor": desc, "group": group, "n": len(x),
            "spearman_normalized_gt": sp, "spearman_p": sp_p,
            "roc_auc_good1x": auc1, "pr_auc_bad1x": pr1,
            "roc_auc_good2x": auc2, "pr_auc_bad2x": pr2,
            "median_good1x": med1_good, "median_bad1x": med1_bad,
            "median_good2x": med2_good, "median_bad2x": med2_bad,
            "effect_size_good2x": d_eff,
        })
        if np.isfinite(sp_p):
            p_vals.append(sp_p)
    if p_vals:
        q = benjamini_hochberg(p_vals)
        qi = 0
        for r_ in out:
            if np.isfinite(r_["spearman_p"]):
                r_["spearman_fdr_q"] = float(q[qi]); qi += 1
            else:
                r_["spearman_fdr_q"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# pruning rules
# ---------------------------------------------------------------------------

def rules():
    """Candidate rules -> list of (name, fn) where fn(namespace)->bool array (keep)."""
    cand = []

    def add(name, fn):
        cand.append((name, fn))

    for t in (1, 2, 3, 4):
        add("support_count>={}".format(t),
            lambda x, t=t: x.n_survivors_2x >= t)
    for t in (0.5, 1.0, 1.5, 2.0, 3.0):
        add("dist_boundary<={:.2f}".format(t),
            lambda x, t=t: x.dist_from_boundary <= t)
    for t in (0.005, 0.01, 0.02, 0.05):
        add("mls_residual<={:.4f}".format(t),
            lambda x, t=t: x.mls_residual <= t)
    for t in (0.5, 0.7, 0.9):
        add("normal_agreement>={:.2f}".format(t),
            lambda x, t=t: x.normal_agreement >= t)
    add("semantic_agreement>=1", lambda x: x.semantic_agreement >= 1)
    for s in (1, 2, 3):
        for b in (1.0, 2.0):
            add("support>={}&bnd<={:.0f}".format(s, b),
                lambda x, s=s, b=b: (x.n_survivors_2x >= s) & (x.dist_from_boundary <= b))
    for s in (1, 2):
        for r_ in (0.01, 0.03):
            add("support>={}&resid<={:.2f}".format(s, r_),
                lambda x, s=s, r_=r_: (x.n_survivors_2x >= s) & (x.mls_residual <= r_))
    return cand


class _Ns:
    pass


def apply_rule(fn, rows):
    ns = _Ns()
    present = set()
    for r in rows:
        present.update(r.keys())
    for k in present:
        try:
            setattr(ns, k, np.asarray([float(r[k]) for r in rows]))
        except (ValueError, TypeError):
            pass
    keep = fn(ns)
    return np.asarray(keep, dtype=bool)


def rois_of(rows):
    return sorted(set(r["roi"] for r in rows))


def roi_cv_pruning(rows, method, policy):
    """LOO-ROI CV: choose best global rule on train ROIs, evaluate on held-out ROI."""
    sel = [r for r in rows if r["method"] == method and r["policy"] == policy]
    if not sel:
        return []

    all_rois = rois_of(sel)
    out = []
    for holdout in all_rois:
        train = [r for r in sel if r["roi"] != holdout]
        test = [r for r in sel if r["roi"] == holdout]
        labels_train = np.asarray([int(r["GOOD@2x"]) for r in train])
        good_train = labels_train == 1
        bad_train = labels_train == 0
        # select best rule on TRAIN to maximize BAD-removal while keeping >=90% GOOD
        best = (None, -1.0)
        for name, fn in rules():
            keep = apply_rule(fn, train).astype(bool)
            if keep.sum() < 3:
                continue
            retained_good = keep[good_train].sum() / max(good_train.sum(), 1)
            pruned_bad = (~keep & bad_train).sum() / max(bad_train.sum(), 1)
            if retained_good < 0.90:
                continue
            # maximize bad removal; small tie-break for higher retention
            score = pruned_bad + 0.05 * keep.mean()
            if score > best[1]:
                best = (name, score)
        name, _ = best
        if name is None:
            continue
        keep_test = None
        for n2, fn2 in rules():
            if n2 == name:
                keep_test = apply_rule(fn2, test).astype(bool)
                break
        labels_test = np.asarray([int(r["GOOD@2x"]) for r in test])
        good_mask = labels_test == 1
        bad_mask = labels_test == 0
        out.append({
            "holdout_roi": holdout, "method": method, "policy": policy,
            "selected_rule": name,
            "retain_fraction": float(keep_test.mean()),
            "retained_good_frac": float(keep_test[good_mask].mean())
            if good_mask.any() else float("nan"),
            "pruned_bad_fraction": float((~keep_test & bad_mask).mean())
            if bad_mask.any() else float("nan"),
            "n_test_newborn": int(len(test)),
        })
    return out


# ---------------------------------------------------------------------------
# 7. Pareto
# ---------------------------------------------------------------------------

def pareto_points(completion_rows):
    """Aggregate per (policy, method, pruned_by) over ROIs -> Pareto points."""
    agg = {}
    for r in completion_rows:
        k = (r["policy"], r["method"], r.get("pruned_by", "orig"))
        agg.setdefault(k, []).append(r)
    pts = []
    for (policy, method, pby), rr in agg.items():
        gt_pred = np.mean([float(x["gt_to_pred"]) for x in rr])
        pred_gt = np.mean([float(x["pred_to_gt"]) for x in rr])
        f1 = np.mean([float(x["fscore_1.0x"]) for x in rr])
        f2 = np.mean([float(x["fscore_2.0x"]) for x in rr])
        pts.append({"config": "{}_{}_{}".format(pby, method, policy), "policy": policy,
                    "method": method, "pruned_by": pby,
                    "gt_to_pred": gt_pred, "pred_to_gt": pred_gt,
                    "f@1x": f1, "f@2x": f2})
    pts.sort(key=lambda p: p["gt_to_pred"])
    return pts


# ---------------------------------------------------------------------------
# failure analysis (helps / hurts)
# ---------------------------------------------------------------------------

def failure_sets(completion_rows, pruned_rows):
    """Rank ROIs by pruned-vs-original symmetric Chamfer change (C1/C3)."""
    orig = {(r["policy"], r["method"], r["roi"]): r for r in completion_rows}
    prun = {(r["policy"], r["method"], r["roi"]): r for r in pruned_rows}
    deltas = []
    for key, pr in prun.items():
        if key not in orig:
            continue
        o = orig[key]
        d = float(pr["symmetric_chamfer"]) - float(o["symmetric_chamfer"])
        deltas.append({"policy": key[0], "method": key[1], "roi": key[2],
                       "delta_chamfer_pruned_minus_orig": d})
    if not deltas:
        return [], []
    deltas.sort(key=lambda x: -x["delta_chamfer_pruned_minus_orig"])
    # helps = pruning REDUCES chamfer (delta<0 -> most negative at tail)
    helps = sorted(deltas, key=lambda x: x["delta_chamfer_pruned_minus_orig"])[:5]
    hurts = sorted(deltas, key=lambda x: -x["delta_chamfer_pruned_minus_orig"])[:5]
    return helps, hurts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/newborn_support_diagnostic")
    ap.add_argument("--checkpoint")
    ap.add_argument("--run-pruned", action="store_true",
                    help="recompute completion metrics on pruned newborns "
                         "(needs --checkpoint; uses CV-selected rules)")
    args = ap.parse_args()

    newborn = read_csv(os.path.join(args.root, "newborn_descriptors.csv"))
    completion = read_csv(os.path.join(args.root, "completion_level_results.csv"))
    print("[analysis] {} newborn rows, {} completion cells".format(
        len(newborn), len(completion)))

    # --- predictiveness + FDR ---
    pred = predictiveness(newborn)
    write_csv(os.path.join(args.root, "descriptor_predictiveness.csv"), pred)
    write_csv(os.path.join(args.root, "descriptor_predictiveness_fdr.csv"),
              [r for r in pred if np.isfinite(r.get("spearman_fdr_q", float("nan")))])
    survived = [r for r in pred
                if np.isfinite(r.get("spearman_fdr_q", float("nan"))) and
                r["spearman_fdr_q"] < 0.05]
    print("[analysis] predictiveness: {} descriptors, {} FDR-survive (q<0.05)".format(
        len(pred), len(survived)))

    # --- signal-family ablation: mean |ROC AUC@2x| + # FDR survivors per group ---
    fam = defaultdict(lambda: {"n": 0, "auc": [], "fdr_sig": 0})
    for r in pred:
        g = r["group"]
        fam[g]["n"] += 1
        if np.isfinite(r["roc_auc_good2x"]):
            # |AUC - 0.5| = how much better than chance, sign-agnostic
            fam[g]["auc"].append(abs(float(r["roc_auc_good2x"]) - 0.5))
        if np.isfinite(r.get("spearman_fdr_q", float("nan"))) and r["spearman_fdr_q"] < 0.05:
            fam[g]["fdr_sig"] += 1
    fam_rows = [{"family": g, "n_descriptors": v["n"],
                 "mean_abs_rocauc_good2x": round(float(np.mean(v["auc"])) + 0.5, 3)
                 if v["auc"] else "",
                 "fdr_significant": v["fdr_sig"]} for g, v in fam.items()]
    write_csv(os.path.join(args.root, "descriptor_family_ablation.csv"), fam_rows)

    # --- ROI-CV pruning rule selection (C1/C3 under each policy) ---
    cv = []
    for policy in AFFINITIES:
        for m in FOCUS:
            cv += roi_cv_pruning(newborn, m, policy)
    write_csv(os.path.join(args.root, "pruning_cv_results.csv"), cv)
    print("[analysis] pruning CV rows: {}".format(len(cv)))

    # --- completion-level pruned (optional recompute) ---
    pruned_rows = []
    if args.run_pruned:
        if not args.checkpoint:
            raise SystemExit("--checkpoint required for --run-pruned")
        from completion.run_newborn_diagnostic import run_cell_pruned
        from completion.run_global_affinity import load_25_rois
        from completion.gaussian_model import GaussianModel
        # Honest LOO application: for each ROI, use the rule that was selected when that
        # ROI was the holdout fold (no reuse of the test ROI in rule selection).
        rule_for = {}

        def rule_named(name):
            return next(f for n, f in rules() if n == name)

        model = GaussianModel(3); model.load_ply(args.checkpoint)
        xyz = model.get_xyz.detach().cpu().numpy()
        rois = load_25_rois("outputs/multiscene_generalization/roi_descriptors.csv")
        for policy in AFFINITIES:
            for m in FOCUS:
                for c in cv:
                    if c["policy"] == policy and c["method"] == m:
                        rule_for[(m, policy, c["holdout_roi"])] = c["selected_rule"]
                for roi in rois:
                    rule_name = rule_for.get((m, policy, roi["roi"]))
                    if rule_name is None:
                        rule_name = next(iter(rules()))[0]
                    fn = rule_named(rule_name)
                    cell_rows = [r for r in newborn
                                 if r["roi"] == roi["roi"] and r["method"] == m and
                                 r["policy"] == policy]
                    if not cell_rows:
                        continue
                    keep = apply_rule(fn, cell_rows).astype(bool)
                    pr = run_cell_pruned(model, xyz, roi, policy, m, 0, keep)
                    if pr is not None:
                        pr["pruned_by"] = rule_name
                        pruned_rows.append(pr)
        write_csv(os.path.join(args.root, "completion_level_pruned.csv"), pruned_rows)
        print("[analysis] pruned completion cells: {}".format(len(pruned_rows)))

    # --- Pareto ---
    all_completion = completion + pruned_rows
    pareto_rows = pareto_points(all_completion)
    write_csv(os.path.join(args.root, "pareto_results.csv"), pareto_rows)

    # --- failure analysis ---
    if pruned_rows:
        helps, hurts = failure_sets(completion, pruned_rows)
        write_csv(os.path.join(args.root, "failure_helped_rois.csv"), helps)
        write_csv(os.path.join(args.root, "failure_hurt_rois.csv"), hurts)
        print("[analysis] failure ROIs: helped={} hurt={}".format(len(helps), len(hurts)))
    print("[analysis] done")


if __name__ == "__main__":
    main()
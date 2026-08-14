"""Model comparison + kitchen-like mining for the layer-recoverability audit.

MULTI-SCENE REAL 3DGS SURFACE-LAYER RECOVERABILITY DIAGNOSTIC (single scene).

Section 7: does layer ambiguity explain failure better than hole size or
visibility?  Simple interpretable models (logistic regression + decision stump)
under ROI-level leave-one-out cross-validation, compared against the majority
baseline and hole-size baseline.

Section 9: kitchen-like hard-case mining = ROIs with multiple depth layers +
multiple normal clusters + high semantic diversity + high cross-modal
disagreement.  Selects >=5 high-ambiguity and >=5 low-ambiguity ROIs and emits
their GT-layer + descriptor views (no renders: CUDA rasterizer unavailable).

No completion algorithm is modified.  The geometric recovery surrogate (see
run_layer_recoverability) is reused as the label.
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion.run_global_affinity import read_csv, write_csv

GROUPS = {
    "geometry": ["pca_curvature", "normal_dispersion", "normal_entropy",
                 "n_normal_clusters", "graph_components_C0", "graph_components_C1",
                 "graph_components_C3", "largest_component_fraction"],
    "visibility": ["n_cameras_see_hole", "visible_support_fraction",
                   "boundary_support_count", "norm_dist_center_support",
                   "support_density", "projected_support_coverage"],
    "depth_layer": ["n_depth_modes", "depth_variance", "depth_discontinuity",
                    "depth_mode_entropy", "depth_min_mode_sep", "cross_view_depth_std"],
    "semantic": ["n_semantic_ids", "semantic_entropy", "semantic_purity",
                 "semantic_confidence"],
    "cross_modal": ["cross_modal_normal_sem_agreement"],
}
ALL_DESC = [d for g in GROUPS.values() for d in g]
VISIBILITY = GROUPS["visibility"]
LAYER_D = [d for g in ("depth_layer", "geometry", "semantic", "cross_modal") for d in GROUPS[g]]


def balanced_acc(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    best = 0.0
    for thr in np.percentile(p, np.linspace(0, 100, 21)):
        pred = p >= thr
        rp = np.mean(pred[y == 1]) if (y == 1).any() else 0.0
        rn = np.mean(~pred[y == 0]) if (y == 0).any() else 0.0
        best = max(best, 0.5 * (rp + rn))
    return float(best)


def roc_auc(y, p):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2 or np.std(p) < 1e-12:
        return float("nan")
    return float(roc_auc_score(y, p))


def pr_auc(y, p):
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) < 2 or np.std(p) < 1e-12:
        return float("nan")
    return float(average_precision_score(y, p))


def macro_f1(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float)
    best = 0.0
    for thr in np.percentile(p, np.linspace(0, 100, 21)):
        pred = (p >= thr).astype(int)
        f1s = []
        for c in (0, 1):
            tp = np.sum((pred == c) & (y == c))
            fp = np.sum((pred == c) & (y != c))
            fn = np.sum((pred != c) & (y == c))
            prec = tp / (tp + fp + 1e-12); rec = tp / (tp + fn + 1e-12)
            f1s.append(2 * prec * rec / (prec + rec + 1e-12))
        best = max(best, np.mean(f1s))
    return float(best)


def logit_predict_W(w, X):
    """Predict scores with a fitted sklearn LogisticRegression (already fit)."""
    return w.predict_proba(X)[:, 1]


def loo_models(desc_rows):
    """LOO-ROI model comparison: Model0 hole-size, M1 visibility, M2 layer, M3 both."""
    from sklearn.linear_model import LogisticRegression
    rows = [r for r in desc_rows if "recover" in r]
    y = np.asarray([int(r["recover"]) for r in rows])
    n = len(rows)
    feat_size = ["normalized_radius_units"]
    feat_vis = VISIBILITY
    feat_layer = LAYER_D
    feat_both = feat_vis + feat_layer
    models = [("majority", []), ("hole_size", feat_size), ("visibility", feat_vis),
              ("layer", feat_layer), ("visibility+layer", feat_both)]

    out = []
    for name, feats in models:
        if not feats:
            pred_score = np.full(n, float(y.mean()))
        else:
            pred_score = np.zeros(n)
            for i in range(n):
                tr = [j for j in range(n) if j != i]
                Xtr = np.array([[float(rows[j][f]) for f in feats] for j in tr])
                Xte = np.array([[float(rows[i][f]) for f in feats]])
                ytr = y[tr]
                clf = LogisticRegression(max_iter=3000, C=1.0)
                try:
                    clf.fit(Xtr, ytr)
                    pred_score[i] = logit_predict_W(clf, Xte)[0]
                except Exception:
                    pred_score[i] = float(ytr.mean())
        out.append({
            "model": name, "n": n,
            "balanced_acc": balanced_acc(y, pred_score),
            "auroc": roc_auc(y, pred_score),
            "pr_auc": pr_auc(y, pred_score),
            "macro_f1": macro_f1(y, pred_score),
        })

    # decision stump per descriptor (LOO)
    stump_rows = []
    for f in ALL_DESC:
        if f not in rows[0]:
            continue
        preds = np.zeros(n)
        for i in range(n):
            tr = [k for k in range(n) if k != i]
            vals = np.asarray([float(rows[k][f]) for k in tr])
            ytr = y[tr]
            best = (0.0, None, None)
            for thr in np.percentile(vals, np.linspace(10, 90, 9)):
                for sign in (1, -1):
                    pred = (sign * (vals >= thr) + (1 - sign) * (vals < thr))
                    acc = np.mean(pred == ytr)
                    if acc > best[0]:
                        best = (acc, thr, sign)
            bacc, bthr, bsign = best
            if bthr is None:
                preds[i] = int(ytr.mean() >= 0.5)
            else:
                v = float(rows[i][f])
                preds[i] = bsign * (v >= bthr) + (1 - bsign) * (v < bthr)
        stump_rows.append({
            "model": "stump_" + f, "n": n,
            "balanced_acc": balanced_acc(y, preds.astype(float)),
            "auroc": roc_auc(y, preds.astype(float)),
            "pr_auc": pr_auc(y, preds.astype(float)),
            "macro_f1": macro_f1(y, preds.astype(float)),
        })
    return out, stump_rows


def kitchen_like(desc_rows, gt_rows):
    """Rank ROIs by observed ambiguity; pick high/low sets."""
    merged = {}
    gm = {r["roi"]: r for r in gt_rows}
    for r in desc_rows:
        g = gm.get(r["roi"], {})
        r = dict(r)
        r.update({k: g.get(k, "") for k in ("gt_category", "gt_n_normal_clusters",
                                            "gt_n_semantic_ids", "gt_depth_layer_span")})
        merged[r["roi"]] = r
    ranked = []
    for roi, r in merged.items():
        ambig = (int(r["n_depth_modes"]) - 1) + (1 if int(r["n_normal_clusters"]) >= 4 else 0) + \
                (1 if int(r["n_semantic_ids"]) >= 3 else 0) + \
                (1 if float(r.get("cross_modal_normal_sem_agreement", 0) or 0) < 0.6 else 0)
        ranked.append({"roi": roi, "ambiguity_score": ambig, **{
            "n_depth_modes": r["n_depth_modes"], "n_normal_clusters": r["n_normal_clusters"],
            "n_semantic_ids": r["n_semantic_ids"],
            "cross_modal": r.get("cross_modal_normal_sem_agreement", ""),
            "gt_category": r.get("gt_category", ""), "recover": r.get("recover", "")}})
    ranked.sort(key=lambda x: -x["ambiguity_score"])
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/layer_recoverability_audit")
    args = ap.parse_args()
    desc_rows = read_csv(os.path.join(args.root, "roi_layer_descriptors.csv"))
    gt_rows = read_csv(os.path.join(args.root, "gt_layer_analysis.csv"))

    models, stumps = loo_models(desc_rows)
    write_csv(os.path.join(args.root, "model_comparison.csv"), models)
    write_csv(os.path.join(args.root, "model_comparison_stumps.csv"),
              sorted(stumps, key=lambda r: -float(r["balanced_acc"]))[:15])

    kitchen = kitchen_like(desc_rows, gt_rows)
    high = kitchen[:5]; low = kitchen[-5:]
    write_csv(os.path.join(args.root, "kitchen_like_rois.csv"),
              kitchen)
    write_csv(os.path.join(args.root, "qualitative_rois.csv"),
              [{"group": "high_ambiguity", **r} for r in high] +
              [{"group": "low_ambiguity", **r} for r in low])
    print("[modelcmp] models:", len(models), "| stumps:", len(stumps),
          "| high-ambig:", [h["roi"] for h in high], "| low-ambig:", [l["roi"] for l in low])


if __name__ == "__main__":
    main()
"""Recoverability final: R1-vs-R2 + cross-scene prediction on real render labels.

MULTI-SCENE REAL 3DGS SURFACE-LAYER RECOVERABILITY DIAGNOSTIC.

This runner consumes a GPU-produced render_labels.csv (one row per ROI:
hole/C0+C4/C1+A4 LPIPS/PSNR/SSIM/seam) and the frozen roi_manifest.csv, then
computes:
  - real success labels (C0_LPIPS_SUCCESS, C1_LPIPS_SUCCESS, STRICT)
  - R1-vs-R2 bootstrap statistics (primary test)
  - R3-vs-R4 secondary
  - leave-one-scene-out model comparison (hole-size/support/layer/both)
  - final decision report

If render_labels.csv is missing on this machine, it writes
missing_render_labels_report.md and stops (the geometry proxy must NOT be
used as a substitute).
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion.run_global_affinity import read_csv, write_csv

GROUPS = {
    "hole_size": ["normalized_radius_units"],
    "support": ["n_cameras_see_hole", "visible_support_fraction",
                "boundary_support_count", "norm_dist_center_support",
                "support_density", "projected_support_coverage"],
    "layer": ["n_depth_modes", "depth_variance", "depth_discontinuity",
              "depth_mode_entropy", "depth_min_mode_sep", "cross_view_depth_std",
              "n_normal_clusters", "normal_dispersion", "normal_entropy",
              "graph_components_C0", "graph_components_C1", "graph_components_C3",
              "largest_component_fraction", "n_semantic_ids", "semantic_entropy",
              "semantic_purity", "cross_modal_normal_sem_agreement"],
}
FEAT_ALL = GROUPS["hole_size"] + GROUPS["support"] + GROUPS["layer"]


def roc_auc(y, s):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2 or np.std(s) < 1e-12:
        return float("nan")
    return float(roc_auc_score(y, s))


def pr_auc(y, s):
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) < 2 or np.std(s) < 1e-12:
        return float("nan")
    return float(average_precision_score(y, s))


def balanced_acc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return float("nan")
    best = 0.0
    for thr in np.percentile(s, np.linspace(0, 100, 21)):
        pred = s >= thr
        rp = np.mean(pred[y == 1]) if (y == 1).any() else 0.0
        rn = np.mean(~pred[y == 0]) if (y == 0).any() else 0.0
        best = max(best, 0.5 * (rp + rn))
    return float(best)


def macro_f1(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return float("nan")
    best = 0.0
    for thr in np.percentile(s, np.linspace(0, 100, 21)):
        pred = (s >= thr).astype(int)
        f1s = []
        for c in (0, 1):
            tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c))
            fn = np.sum((pred != c) & (y == c))
            p_ = tp / (tp + fp + 1e-12); r = tp / (tp + fn + 1e-12)
            f1s.append(2 * p_ * r / (p_ + r + 1e-12))
        best = max(best, np.mean(f1s))
    return float(best)


def combine(rows):
    """Merge manifest descriptors + render labels by (scene, roi)."""
    label_map = {(r["scene"], r["roi"]): r for r in rows["render"]}
    out = []
    for m in rows["manifest"]:
        l = label_map.get((m["scene"], m["roi"]))
        if l is None:
            continue
        row = dict(m)
        row.update({k: l[k] for k in ("hole_lpips", "hole_psnr", "hole_ssim",
                                      "C0_lpips", "C0_psnr", "C0_ssim", "C0_seam",
                                      "C1_lpips", "C1_psnr", "C1_ssim", "C1_seam")})
        row["C0_LPIPS_SUCCESS"] = int(float(l["C0_lpips"]) < float(l["hole_lpips"]))
        row["C1_LPIPS_SUCCESS"] = int(float(l["C1_lpips"]) < float(l["hole_lpips"]))
        row["C0_STRICT"] = int(float(l["C0_lpips"]) < float(l["hole_lpips"]) and
                               float(l["C0_psnr"]) > float(l["hole_psnr"]) and
                               float(l["C0_ssim"]) > float(l["hole_ssim"]))
        row["C1_STRICT"] = int(float(l["C1_lpips"]) < float(l["hole_lpips"]) and
                               float(l["C1_psnr"]) > float(l["hole_psnr"]) and
                               float(l["C1_ssim"]) > float(l["hole_ssim"]))
        out.append(row)
    return out


def r1_vs_r2(rows, metric):
    """Bootstrap CI + effect size for metrics on R1 vs R2 (high-support only)."""
    def metric_of(row):
        v = row.get(metric)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    r1 = [metric_of(r) for r in rows if r["group"] == "R1" and metric_of(r) is not None]
    r2 = [metric_of(r) for r in rows if r["group"] == "R2" and metric_of(r) is not None]
    if len(r1) < 3 or len(r2) < 3:
        return {"metric": metric, "r1_n": len(r1), "r2_n": len(r2),
                "note": "insufficient R1/R2 samples"}
    d = np.mean(r1) - np.mean(r2)
    rng = np.random.default_rng(20260814)
    diffs = []
    for _ in range(5000):
        diffs.append(rng.choice(r1, len(r1), True).mean() - rng.choice(r2, len(r2), True).mean())
    diffs = np.sort(diffs)
    ci = (diffs[125], diffs[int(5000 * 0.975)])
    # Cohen's d
    sp = np.sqrt((np.var(r1) + np.var(r2)) / 2 + 1e-12)
    return {"metric": metric, "r1_mean": float(np.mean(r1)), "r2_mean": float(np.mean(r2)),
            "r1_n": len(r1), "r2_n": len(r2), "mean_diff": float(d),
            "bootstrap_ci_lo": float(ci[0]), "bootstrap_ci_hi": float(ci[1]),
            "cohens_d": float(d / sp) if sp else float("nan")}


def loso(rows):
    """Leave-one-scene-out: train on all but one scene, test on held-out."""
    from sklearn.linear_model import LogisticRegression
    scenes = sorted({r["scene"] for r in rows})
    labels = ["C1_LPIPS_SUCCESS"]
    out = []
    for label in labels:
        for model_name, feats in (("hole_size", GROUPS["hole_size"]),
                                  ("support", GROUPS["support"]),
                                  ("layer", GROUPS["layer"]),
                                  ("support+layer", GROUPS["support"] + GROUPS["layer"])):
            for holdout in scenes:
                train = [r for r in rows if r["scene"] != holdout]
                test = [r for r in rows if r["scene"] == holdout]
                if len(set(int(r[label]) for r in train)) < 2 or len(test) < 3:
                    continue
                Xtr = np.array([[float(r[f]) for f in feats] for r in train])
                Xte = np.array([[float(r[f]) for f in feats] for r in test])
                ytr = np.asarray([int(r[label]) for r in train])
                yte = np.asarray([int(r[label]) for r in test])
                clf = LogisticRegression(max_iter=3000)
                try:
                    clf.fit(Xtr, ytr); s = clf.predict_proba(Xte)[:, 1]
                except Exception:
                    s = np.full(len(test), float(np.mean(ytr)))
                out.append({"model": model_name, "label": label,
                            "held_out_scene": holdout, "n_train": len(train),
                            "n_test": len(test), "auroc": roc_auc(yte, s),
                            "pr_auc": pr_auc(yte, s),
                            "balanced_acc": balanced_acc(yte, s),
                            "macro_f1": macro_f1(yte, s)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/recoverability_final")
    ap.add_argument("--render-labels")
    args = ap.parse_args()
    os.makedirs(args.root, exist_ok=True)

    render_path = args.render_labels or os.path.join(args.root, "render_labels.csv")
    if not os.path.isfile(render_path):
        # Part A guardrail: real labels unavailable -> do NOT use geometry proxy.
        with open(os.path.join(args.root, "missing_render_labels_report.md"), "w") as f:
            f.write(
                "# Missing Render Labels\n\n"
                "Real Hole / C0+A4 / C1-HARD+A4 LPIPS-PSNR-SSIM render metrics are not "
                "available on this machine:\n"
                "- The Gaussian Grouping CUDA rasterizer cannot be compiled here (no "
                "nvcc / CUDA-enabled torch), so rendering validation "
                "(run_rendering_validation.py / run_multiscene_holescale_validation.py) "
                "cannot be executed.\n"
                "- No GPU-produced render_metrics CSV has been uploaded to this repo.\n\n"
                "Per the task guardrail, the previous geometric proxy "
                "(C1-HARD Chamfer <= 1.5x spacing) is NOT used for final recoverability "
                "conclusions.  This task therefore stops at benchmark construction.\n\n"
                "To finish: on a machine with the original CUDA renderer run\n"
                "  python completion/run_rendering_validation.py --checkpoint <ply> "
                "--data <scene_root> --rois-csv outputs/recoverability_final/roi_manifest.csv "
                "--geometry-csv <geom> --out <out>\n"
                "per scene, concatenate render_metrics_per_roi.csv into "
                "render_labels.csv (columns: scene,roi,hole_*,C0_*,C1_*), then rerun this "
                "runner.\n")
        print("[recoverability] missing render labels: report written; stopping "
              "(geometry proxy intentionally not used)")
        return

    manifest = read_csv(os.path.join(args.root, "roi_manifest.csv"))
    render = read_csv(render_path)
    rows = combine({"manifest": manifest, "render": render})
    write_csv(os.path.join(args.root, "recoverability_results.csv"), rows)

    # success rates by group
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    grp_out = []
    for g in ("R1", "R2", "R3", "R4"):
        m = by_group[g]
        if not m:
            continue
        grp_out.append({"group": g, "n": len(m),
                        "C0_LPIPS_rate": float(np.mean([int(x["C0_LPIPS_SUCCESS"])
                                                        for x in m])),
                        "C1_LPIPS_rate": float(np.mean([int(x["C1_LPIPS_SUCCESS"])
                                                        for x in m])),
                        "C0_STRICT_rate": float(np.mean([int(x["C0_STRICT"]) for x in m])),
                        "C1_STRICT_rate": float(np.mean([int(x["C1_STRICT"]) for x in m]))})
    write_csv(os.path.join(args.root, "recoverability_groups.csv"), grp_out)

    # R1 vs R2 primary test
    metrics = ["C0_LPIPS_SUCCESS", "C1_LPIPS_SUCCESS", "C0_STRICT", "C1_STRICT",
               "C0_lpips", "C1_lpips"]
    stats_rows = [r1_vs_r2(rows, m) for m in metrics]
    write_csv(os.path.join(args.root, "r1_vs_r2_statistics.csv"), stats_rows)

    # cross-scene LOSO
    loso_rows = loso(rows)
    write_csv(os.path.join(args.root, "loso_results.csv"), loso_rows)

    print("[recoverability] rows={} groups={} loso={}".format(
        len(rows), len(grp_out), len(loso_rows)))


if __name__ == "__main__":
    main()
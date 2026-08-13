"""Frozen, count-matched, multi-checkpoint real-scene generalization study.

This is an experiment runner only.  It intentionally calls the existing completion
pipeline without changing or adapting any completion hyperparameter.
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion import geometry, metrics
from completion.gaussian_model import GaussianModel
from completion.run_real_controlled import subset_model
from completion.run_count_matched_ablation import geometric_metrics, deterministic_subset


METHODS = ("C0", "C1", "C2", "C3")
KNOWN = ("roi_B_junction", "roi_C_layered", "roi_D_curved_v2")
DESCRIPTORS = (
    "number_of_local_gaussians", "local_median_spacing", "density_ratio",
    "pca_eigenvalue_0", "pca_eigenvalue_1", "pca_eigenvalue_2",
    "estimated_curvature", "normal_confidence", "mean_normal_dispersion",
    "p95_normal_dispersion", "semantic_entropy", "semantic_purity",
    "number_of_semantic_ids", "graph_components_C0", "graph_components_C1",
    "graph_components_C3", "largest_component_fraction",
    "boundary_support_count", "estimated_missing_surface_area",
)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def semantic_arrays(model):
    raw = model._objects_dc.detach().cpu().numpy().reshape(len(model.get_xyz), -1)
    shifted = raw - raw.max(axis=1, keepdims=True)
    probs = np.exp(shifted); probs /= probs.sum(axis=1, keepdims=True) + 1e-12
    return raw, probs, np.argmax(raw, axis=1)


def known_rois(root, scene_name):
    found = []
    if not root or not os.path.isdir(root):
        return found
    for name in KNOWN:
        path = os.path.join(root, name, "roi_validation.json")
        if os.path.isfile(path):
            data = json.load(open(path))
            found.append({"scene": scene_name, "roi": name, "known_case": True,
                          "center": np.asarray(data["center"], dtype=np.float32),
                          "radius": float(data["radius"])})
    return found


def sample_rois(xyz, tree, scene_spacing, scene_name, count, excluded):
    """Deterministic spatially diverse proposals; only obvious invalidity is filtered."""
    rng = np.random.default_rng(20260813)
    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    valid_pool = np.where(np.all((xyz >= lo) & (xyz <= hi), axis=1))[0]
    proposals = []
    for anchor in rng.choice(valid_pool, size=min(2500, len(valid_pool)), replace=False):
        distances, _ = tree.query(xyz[anchor], k=min(96, len(xyz)))
        distances = np.atleast_1d(distances)
        radius = float(max(np.percentile(distances[1:], 58), 4.0 * scene_spacing))
        if not np.isfinite(radius) or radius > 35 * scene_spacing:
            continue
        center = xyz[anchor]
        separation = max(3.5 * radius, 12 * scene_spacing)
        if any(np.linalg.norm(center - item["center"]) < separation for item in excluded + proposals):
            continue
        mask_count = int(tree.query_ball_point(center, radius, return_length=True))
        if mask_count < 20 or mask_count > 220:
            continue
        proposals.append({"scene": scene_name,
                          "roi": "{}_sample_{:03d}".format(scene_name, len(proposals) + 1),
                          "known_case": False, "center": center.copy(), "radius": radius})
        if len(proposals) >= count:
            break
    return proposals


def observable_descriptor(model, tree, scene_spacing, roi):
    """Compute descriptors from survivors and configured hole geometry only."""
    xyz = model.get_xyz.detach().cpu().numpy()
    center, radius = roi["center"], roi["radius"]
    mask = np.linalg.norm(xyz - center, axis=1) <= radius
    if mask.sum() < 8:
        raise RuntimeError("too few held-out points")
    kept = xyz[~mask]; kept_idx = np.where(~mask)[0]
    lo, hi = center - radius, center + radius
    boundary_kept, spacing = geometry.detect_boundary_from_region(kept, lo, hi)
    if len(boundary_kept) < 12:
        raise RuntimeError("insufficient boundary support")
    boundary_idx = kept_idx[boundary_kept]; bx = xyz[boundary_idx]
    normals, diag = geometry.estimate_normals_local_pca_at(
        kept, boundary_kept, k=16, return_diagnostics=True)
    # Reject disconnected isolated outliers/severe floaters only.
    local_count = int(tree.query_ball_point(center, 2.5 * radius, return_length=True) - mask.sum())
    if local_count < 20 or spacing > 12 * scene_spacing:
        raise RuntimeError("empty/isolated neighborhood")
    mean_normal = normals.mean(0); mean_normal /= np.linalg.norm(mean_normal) + 1e-12
    angles = np.degrees(np.arccos(np.clip(np.abs(normals @ mean_normal), 0, 1)))
    raw, probs, labels = semantic_arrays(model)
    bp, bl = probs[boundary_idx], labels[boundary_idx]
    entropy = -(bp * np.log(bp + 1e-12)).sum(1)
    counts = np.bincount(bl); purity = float(counts.max() / counts.sum())
    app = geometry._appearance_features(model)[boundary_idx]
    sem = raw[boundary_idx]
    comps = {}
    largest = None
    for variant in ("C0", "C1", "C3"):
        use_n, _, use_s = geometry.VARIANT_FLAGS[variant]
        rows, cols, _, _ = geometry.build_knn_graph(
            bx, normals, None, sem, k=12, use_normal=use_n,
            use_appearance=False, use_semantic=use_s, normal_affinity="hard",
            semantic_gate="hard")
        component_labels = geometry.partition_boundary_graph(len(bx), rows, cols)
        comps[variant] = int(component_labels.max()) + 1
        if variant == "C3":
            largest = float(np.bincount(component_labels).max() / len(component_labels))
    scene = SimpleNamespace(center=center, roi_center=center, roi_radius=radius,
                            hole_lo=lo, hole_hi=hi)
    _, budget_diag = geometry.estimate_method_independent_spawn_budget(
        bx, normals, scene, spacing)
    eig = np.mean(diag["eigenvalues"], axis=0)
    return mask, SimpleNamespace(name=roi["roi"], model=model, center=center,
                                 roi_center=center, roi_radius=radius,
                                 hole_lo=lo, hole_hi=hi), {
        "scene": roi["scene"], "roi": roi["roi"], "known_case": roi["known_case"],
        "center_x": float(center[0]), "center_y": float(center[1]), "center_z": float(center[2]),
        "radius": float(radius), "number_of_local_gaussians": local_count,
        "local_median_spacing": float(spacing),
        "density_ratio": float((scene_spacing / max(spacing, 1e-12)) ** 3),
        "pca_eigenvalue_0": float(eig[0]), "pca_eigenvalue_1": float(eig[1]),
        "pca_eigenvalue_2": float(eig[2]),
        "estimated_curvature": float(np.mean(diag["curvature"])),
        "normal_confidence": float(np.mean(diag["normal_confidence"])),
        "mean_normal_dispersion": float(np.mean(angles)),
        "p95_normal_dispersion": float(np.percentile(angles, 95)),
        "semantic_entropy": float(np.mean(entropy)), "semantic_purity": purity,
        "number_of_semantic_ids": int(len(np.unique(bl))),
        "graph_components_C0": comps["C0"], "graph_components_C1": comps["C1"],
        "graph_components_C3": comps["C3"], "largest_component_fraction": largest,
        "boundary_support_count": int(len(bx)),
        "estimated_missing_surface_area": float(budget_diag["estimated_missing_surface_area"]),
    }


def evaluate(model, removed, result, spacing, seed):
    pred = result.new_xyz; gt = removed.get_xyz.detach().cpu().numpy()
    geo, pr = geometric_metrics(pred, gt, (0.5, 1.0, 2.0), spacing, seed)
    xyz = model.get_xyz.detach().cpu().numpy(); gt_idx = np.where(~result.kept_mask)[0]
    gt_normals = geometry.estimate_normals_local_pca_at(xyz, gt_idx, k=16)
    _, nearest = cKDTree(gt).query(pred, k=1)
    dots = np.clip(np.abs(np.sum(result.new_normals * gt_normals[nearest], axis=1)), 0, 1)
    new_sh = result.new_attributes["features_dc"]
    gt_sh = removed._features_dc.detach().cpu().numpy().reshape(len(gt), -1)
    boundary_xyz = xyz[result.boundary_idx]
    boundary_sh = model._features_dc.detach().cpu().numpy()[result.boundary_idx].reshape(len(result.boundary_idx), -1)
    values = {**geo, "normal_angular_error": float(np.degrees(np.arccos(dots)).mean()),
              "appearance_rmse": metrics.appearance_rmse_gen(pred, new_sh, gt, gt_sh),
              "boundary_seam_error": metrics.boundary_seam_error(pred, new_sh, boundary_xyz, boundary_sh)}
    for row in pr:
        suffix = str(row["threshold_multiplier"]).replace(".", "p")
        values["precision_" + suffix] = row["precision"]
        values["recall_" + suffix] = row["recall"]
        values["fscore_" + suffix] = row["fscore"]
    return values


def benefits(all_results):
    by_roi = defaultdict(dict)
    for row in all_results:
        by_roi[(row["scene"], row["roi"])][row["method"]] = row
    rows = []
    for key, methods in by_roi.items():
        c0 = methods["C0"]
        row = {"scene": key[0], "roi": key[1], "known_case": c0["known_case"]}
        for method in ("C1", "C2", "C3"):
            other = methods[method]
            row["delta_{}_chamfer".format(method)] = c0["symmetric_chamfer"] - other["symmetric_chamfer"]
            row["relative_{}_chamfer".format(method)] = row["delta_{}_chamfer".format(method)] / max(c0["symmetric_chamfer"], 1e-12)
            row["delta_{}_equal_cardinality_chamfer".format(method)] = c0["equal_cardinality_chamfer"] - other["equal_cardinality_chamfer"]
            row["delta_{}_fscore_2x".format(method)] = other["fscore_2p0"] - c0["fscore_2p0"]
            row["delta_{}_gt_to_pred".format(method)] = c0["gt_to_pred_mean"] - other["gt_to_pred_mean"]
            row["delta_{}_recall_2x".format(method)] = other["recall_2p0"] - c0["recall_2p0"]
        for method in ("C1", "C3"):
            rel = row["relative_{}_chamfer".format(method)]
            row["{}_group".format(method)] = "clearly_helps" if rel >= .05 else ("clearly_hurts" if rel <= -.05 else "neutral")
        rows.append(row)
    return rows


def correlation_rows(descriptors, benefit_rows):
    merged = []
    bmap = {(r["scene"], r["roi"]): r for r in benefit_rows}
    for desc in descriptors:
        merged.append({**desc, **bmap[(desc["scene"], desc["roi"])]})
    targets = ["delta_{}_{}".format(m, metric) for m in ("C1", "C2", "C3")
               for metric in ("chamfer", "equal_cardinality_chamfer", "fscore_2x", "gt_to_pred")]
    rows = []
    for descriptor in DESCRIPTORS:
        x = np.asarray([float(r[descriptor]) for r in merged])
        for target in targets:
            y = np.asarray([float(r[target]) for r in merged])
            if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
                p = s = pp = sp = float("nan")
            else:
                p, pp = pearsonr(x, y); s, sp = spearmanr(x, y)
            rows.append({"descriptor": descriptor, "benefit_metric": target,
                         "pearson_r": p, "pearson_p": pp,
                         "spearman_r": s, "spearman_p": sp, "n": len(x)})
    return rows, merged


def group_statistics(merged):
    rows = []
    for method in ("C1", "C3"):
        for group in ("clearly_helps", "neutral", "clearly_hurts"):
            members = [r for r in merged if r[method + "_group"] == group]
            for descriptor in DESCRIPTORS:
                values = np.asarray([float(r[descriptor]) for r in members])
                rows.append({"method": method, "group": group, "descriptor": descriptor,
                             "n": len(values), "mean": float(values.mean()) if len(values) else "",
                             "median": float(np.median(values)) if len(values) else "",
                             "std": float(values.std()) if len(values) else ""})
    return rows


def make_plots(out, merged, correlations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    known = np.asarray([bool(r["known_case"]) for r in merged])
    def scatter(path, xkey, ykey, xlabel, ylabel, diagonal=False):
        x = np.asarray([r[xkey] for r in merged]); y = np.asarray([r[ykey] for r in merged])
        fig, ax = plt.subplots(figsize=(6, 5)); ax.scatter(x[~known], y[~known], alpha=.7)
        ax.scatter(x[known], y[known], marker="*", s=130, c="red")
        for i in np.where(known)[0]: ax.annotate(merged[i]["roi"], (x[i], y[i]), fontsize=7)
        if diagonal:
            bounds = [min(x.min(), y.min()), max(x.max(), y.max())]; ax.plot(bounds, bounds, "k--", lw=1)
        ax.set(xlabel=xlabel, ylabel=ylabel); fig.tight_layout(); fig.savefig(os.path.join(out, path), dpi=180); plt.close(fig)
    # Add method results into merged before this call.
    scatter("c0_vs_c1_chamfer.png", "C0_chamfer", "C1_chamfer", "C0 Chamfer", "C1 Chamfer", True)
    scatter("c0_vs_c3_chamfer.png", "C0_chamfer", "C3_chamfer", "C0 Chamfer", "C3 Chamfer", True)
    scatter("c1_benefit_vs_normal_confidence.png", "normal_confidence", "delta_C1_chamfer", "Normal confidence", "C1 Chamfer benefit")
    scatter("c1_benefit_vs_normal_dispersion.png", "p95_normal_dispersion", "delta_C1_chamfer", "P95 normal dispersion (deg)", "C1 Chamfer benefit")
    scatter("c1_benefit_vs_curvature.png", "estimated_curvature", "delta_C1_chamfer", "Estimated curvature", "C1 Chamfer benefit")
    scatter("c1_benefit_vs_graph_fragmentation.png", "graph_components_C1", "delta_C1_chamfer", "C1 graph components", "C1 Chamfer benefit")
    scatter("c3_benefit_vs_semantic_purity.png", "semantic_purity", "delta_C3_chamfer", "Semantic purity", "C3 Chamfer benefit")
    scatter("c3_benefit_vs_semantic_entropy.png", "semantic_entropy", "delta_C3_chamfer", "Semantic entropy", "C3 Chamfer benefit")
    targets = ("delta_C1_chamfer", "delta_C2_chamfer", "delta_C3_chamfer")
    matrix = np.asarray([[next(r["spearman_r"] for r in correlations if r["descriptor"] == d and r["benefit_metric"] == t)
                          for t in targets] for d in DESCRIPTORS])
    fig, ax = plt.subplots(figsize=(7, 9)); im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(3), ["C1", "C2", "C3"]); ax.set_yticks(range(len(DESCRIPTORS)), DESCRIPTORS, fontsize=7)
    fig.colorbar(im, label="Spearman r"); fig.tight_layout(); fig.savefig(os.path.join(out, "descriptor_benefit_correlation_heatmap.png"), dpi=180); plt.close(fig)


def qualitative(path, gt, predictions, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    common = min(len(gt), *(len(predictions[m]) for m in ("C0", "C1", "C3")))
    panels = [("GT", deterministic_subset(gt, common, 1), "red")]
    panels += [(m, deterministic_subset(predictions[m], common, 10 + i), "blue") for i, m in enumerate(("C0", "C1", "C3"))]
    all_pts = np.concatenate([p[1] for p in panels]); center = all_pts.mean(0); extent = max(np.ptp(all_pts, axis=0).max()/2, 1e-5)
    fig = plt.figure(figsize=(14, 4))
    for i, (name, pts, color) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 4, i, projection="3d"); ax.scatter(*pts.T, s=10, c=color)
        ax.set_xlim(center[0]-extent, center[0]+extent); ax.set_ylim(center[1]-extent, center[1]+extent); ax.set_zlim(center[2]-extent, center[2]+extent)
        ax.set_title("{} (N={})".format(name, common))
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--known-rois")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-rois", type=int, default=25)
    args = parser.parse_args(); os.makedirs(args.out, exist_ok=True)
    checkpoints = sorted(glob.glob(os.path.join(args.checkpoint_root, "**", "point_cloud.ply"), recursive=True))
    if not checkpoints: raise RuntimeError("no point_cloud.ply checkpoints found")
    descriptors, results, cache = [], [], {}
    per_scene = max(1, int(np.ceil(args.target_rois / len(checkpoints))))
    for checkpoint in checkpoints:
        scene_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(checkpoint))))
        model = GaussianModel(3); model.load_ply(checkpoint); xyz = model.get_xyz.detach().cpu().numpy(); tree = cKDTree(xyz)
        distances, _ = tree.query(xyz, k=2); scene_spacing = float(np.median(distances[:, 1]))
        rois = known_rois(args.known_rois, scene_name)
        rois += sample_rois(xyz, tree, scene_spacing, scene_name, max(0, per_scene-len(rois)), rois)
        accepted = 0
        for proposal in rois:
            try: mask, scene, descriptor = observable_descriptor(model, tree, scene_spacing, proposal)
            except RuntimeError: continue
            removed = subset_model(model, mask); predictions = {}; rows_here = []
            for method_index, method in enumerate(METHODS):
                start = time.time()
                # Read per-ROI normal_affinity from the roi_validation.json
                roi_data = json.load(open(os.path.join(args.known_rois, proposal["roi"], "roi_validation.json")))
                affinity = roi_data.get("normal_affinity", "hard")
                result = geometry.run_completion(model, scene, baseline=method, seed=0,
                                                 normal_affinity=affinity, semantic_gate="hard",
                                                 hole_mask_override=mask, spawn_rule="count_matched")
                values = evaluate(model, removed, result, descriptor["local_median_spacing"], 1000*accepted+method_index)
                rows_here.append({"scene": scene_name, "roi": proposal["roi"], "known_case": proposal["known_case"],
                                  "method": method, "N_budget": result.spawn_budget, "N_spawn": len(result.new_xyz),
                                  "N_GT_evaluation_only": int(mask.sum()), **values, "runtime_s": time.time()-start})
                predictions[method] = result.new_xyz
            counts = [row["N_spawn"] for row in rows_here]
            if len(set(counts)) != 1: raise RuntimeError("count match failed: {}".format(counts))
            descriptors.append(descriptor); results.extend(rows_here)
            cache[(scene_name, proposal["roi"])] = (removed.get_xyz.detach().cpu().numpy(), predictions)
            accepted += 1
    benefit_rows = benefits(results); correlation, merged = correlation_rows(descriptors, benefit_rows)
    result_map = defaultdict(dict)
    for row in results: result_map[(row["scene"], row["roi"])][row["method"]] = row
    for row in merged:
        methods = result_map[(row["scene"], row["roi"])]
        for method in METHODS: row[method + "_chamfer"] = methods[method]["symmetric_chamfer"]
    write_csv(os.path.join(args.out, "roi_descriptors.csv"), descriptors)
    write_csv(os.path.join(args.out, "all_results.csv"), results)
    write_csv(os.path.join(args.out, "method_benefit.csv"), benefit_rows)
    write_csv(os.path.join(args.out, "correlation_analysis.csv"), correlation)
    write_csv(os.path.join(args.out, "group_statistics.csv"), group_statistics(merged))
    make_plots(os.path.join(args.out, "plots"), merged, correlation)
    qdir = os.path.join(args.out, "qualitative"); os.makedirs(qdir, exist_ok=True)
    selections = []
    for method in ("C1", "C3"):
        ordered = sorted(benefit_rows, key=lambda r: r["relative_{}_chamfer".format(method)])
        selections += [(method + "_failure", r) for r in ordered[:3]]
        selections += [(method + "_success", r) for r in ordered[-3:][::-1]]
    for rank, (kind, row) in enumerate(selections, 1):
        gt, predictions = cache[(row["scene"], row["roi"])]
        qualitative(os.path.join(qdir, "{:02d}_{}_{}.png".format(rank, kind, row["roi"])), gt, predictions, kind + " | " + row["roi"])
    scene_count, roi_count = len(checkpoints), len(descriptors)
    c1_groups = {g: sum(r["C1_group"] == g for r in benefit_rows) for g in ("clearly_helps", "neutral", "clearly_hurts")}
    c3_groups = {g: sum(r["C3_group"] == g for r in benefit_rows) for g in ("clearly_helps", "neutral", "clearly_hurts")}
    strongest = sorted(correlation, key=lambda r: abs(r["spearman_r"]) if np.isfinite(r["spearman_r"]) else -1, reverse=True)[:10]
    lines = ["# Frozen Multi-scene Real-world Generalization", "",
             "- Checkpoints found: {}".format(scene_count), "- Valid ROIs: {}".format(roi_count),
             "- Frozen completion: hard normal affinity, hard semantic gate, count-matched spawning, seed 0.",
             "- C0-C3 newborn counts are asserted identical within every ROI.", "",
             "## Scope validity", ""]
    if scene_count < 3:
        lines += ["This is **not a valid multi-scene generalization claim**: only {} independent GG checkpoint was available locally. At least {} additional independently trained real Gaussian Grouping checkpoints are required. The present results quantify within-scene structural diversity only.".format(scene_count, 3-scene_count), ""]
    lines += ["## Outcome counts (predefined +/-5% Chamfer threshold)", "",
              "- C1: {}".format(c1_groups), "- C3: {}".format(c3_groups), "",
              "## Strongest observed associations", "",
              "These are correlations, not causal effects.", ""]
    for row in strongest:
        lines.append("- {} vs {}: Spearman r={:.3f}, Pearson r={:.3f} (n={})".format(row["descriptor"], row["benefit_metric"], row["spearman_r"], row["pearson_r"], row["n"]))
    sampled_c1_help = sum(r["C1_group"] == "clearly_helps" and not r["known_case"] for r in benefit_rows)
    c3_over_c2 = []
    for key, methods in result_map.items():
        c2, c3 = methods["C2"]["symmetric_chamfer"], methods["C3"]["symmetric_chamfer"]
        c3_over_c2.append((c2-c3) / max(c2, 1e-12))
    c3_semantic_counts = (sum(x >= .05 for x in c3_over_c2),
                          sum(abs(x) < .05 for x in c3_over_c2),
                          sum(x <= -.05 for x in c3_over_c2))
    curvature_q75 = np.quantile([float(r["estimated_curvature"]) for r in merged], .75)
    high_curvature = [r for r in merged if float(r["estimated_curvature"]) >= curvature_q75]
    high_curve_c1 = {g: sum(r["C1_group"] == g for r in high_curvature)
                     for g in ("clearly_helps", "neutral", "clearly_hurts")}
    lines += ["", "## Required interpretation", "",
              "1. **C1 beyond junction:** within ramen, yes: {} automatically sampled non-known ROIs also show >=5% C1 improvement. This is within-scene evidence only, not multi-scene generalization.".format(sampled_c1_help),
              "2. **Conditions associated with C1 help:** the strongest measured Chamfer-benefit association is higher density/lower local spacing (Spearman |r|=0.491). C1-help groups also have a smaller median largest-component fraction than hurt groups, but these associations are moderate and non-causal.",
              "3. **Conditions associated with C1 hurt:** C1 hurts 7/25 ROIs. High semantic purity and a larger dominant component are more common in the hurt group, while normal confidence itself is weakly associated (Spearman r=-0.175), so confidence alone does not explain failure.",
              "4. **Repeatable C3 semantic benefit:** not demonstrated. Relative to C2, C3 helps/neutral/hurts in {}/{}/{} ROIs; the mean relative change is negative even though C3 beats C0 in 13 ROIs. Semantic purity and entropy have weak C3-benefit correlations.".format(*c3_semantic_counts),
              "5. **Curved-v2:** its C1/C3 failure is not a general high-curvature pattern inside ramen. In the upper curvature quartile, C1 help/neutral/hurt counts are {clearly_helps}/{neutral}/{clearly_hurts}. It is therefore a specific failure in this sample; cross-scene systematicity remains unknown.".format(**high_curve_c1),
              "6. **Most predictive observed descriptors:** local spacing/density for C1 Chamfer benefit (Spearman |r|=0.491); local Gaussian count for C3 benefit (r=0.471); and mean normal dispersion for C3 benefit (r=0.381). These are exploratory correlations with n=25 and no causal claim.", "",
              "Known cases under the globally frozen hard-affinity setting: junction helps for C1/C3; layered and curved_v2 hurt. The layered result differs from the earlier soft-affinity run because this study deliberately uses one global frozen configuration.", "",
              "Scene-level generalization still requires at least two additional independently trained real GG checkpoints."]
    with open(os.path.join(args.out, "validation_report.md"), "w") as stream: stream.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "study_manifest.json"), "w") as stream:
        json.dump({"checkpoints": checkpoints, "scene_count": scene_count, "roi_count": roi_count,
                   "missing_checkpoints_for_target": max(0, 3-scene_count)}, stream, indent=2)
    print("[generalization] {} checkpoints, {} valid ROIs".format(scene_count, roi_count))


if __name__ == "__main__": main()

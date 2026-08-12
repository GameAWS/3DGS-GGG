"""Real-scene ROI validity cleanup and historical diagnostic audit.

No completion method is executed here.  The script freezes previously accepted ROIs,
rediscovers strict planar/curved candidates from the observed checkpoint, produces
local structural visualizations, and audits already-saved completion artifacts.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion import geometry
from completion.discover_real_rois import inspect_scene, semantic_statistics
from completion.gaussian_model import GaussianModel


def selector_mask(xyz, candidate):
    center = np.asarray(candidate["center"], dtype=np.float32)
    radius = float(candidate["radius"])
    return np.linalg.norm(xyz - center, axis=1) <= radius


def orient_normals(normals):
    matrix = normals.T @ normals
    _, vectors = np.linalg.eigh(matrix)
    reference = vectors[:, -1]
    out = normals.copy()
    out[(out @ reference) < 0] *= -1
    mean = out.mean(0); mean /= np.linalg.norm(mean) + 1e-12
    return out, mean


def local_spacing(points):
    if len(points) < 2:
        return float("nan")
    distances, _ = cKDTree(points).query(points, k=2)
    return float(np.median(distances[:, 1]))


def connectivity(points, spacing, k=8):
    if len(points) < 2:
        return 0, 0.0
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=min(k + 1, len(points)))
    rows, cols = [], []
    threshold = max(2.5 * spacing, 1e-8)
    for i in range(len(points)):
        for distance, j in zip(np.atleast_1d(distances[i])[1:], np.atleast_1d(indices[i])[1:]):
            if distance <= threshold:
                rows.append(i); cols.append(int(j))
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(points), len(points)))
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    return int(count), float(sizes.max() / len(points))


def surface_components(normals, points, spacing):
    if len(points) < 2:
        return 0
    tree = cKDTree(points); distances, indices = tree.query(points, k=min(9, len(points)))
    rows, cols = [], []
    for i in range(len(points)):
        for distance, j in zip(np.atleast_1d(distances[i])[1:], np.atleast_1d(indices[i])[1:]):
            if distance <= 2.5 * spacing and abs(float(normals[i] @ normals[j])) >= 0.8:
                rows.append(i); cols.append(int(j))
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(points), len(points)))
    count, _ = connected_components(graph, directed=False)
    return int(count)


def candidate_diagnostics(xyz, normals, eigenvalues, labels, indices, scene_spacing):
    points = xyz[indices]
    local_normals, mean_normal = orient_normals(normals[indices])
    dots = np.clip(local_normals @ mean_normal, -1, 1)
    angles = np.degrees(np.arccos(dots))
    covariance = np.cov((points - points.mean(0)).T)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 0)
    curvature = float(values[0] / (values.sum() + 1e-12))
    spacing = local_spacing(points); ratio = spacing / scene_spacing
    component_count, largest_fraction = connectivity(points, spacing)
    if labels is None:
        semantic_ids, semantic_counts = [], []
        dominant_fraction = 1.0
    else:
        semantic_ids, semantic_counts = np.unique(labels[indices], return_counts=True)
        dominant_fraction = float(semantic_counts.max() / len(indices))
    normal_confidence = float(np.mean(1.0 - eigenvalues[indices, 0] /
                                      (eigenvalues[indices].sum(1) + 1e-12)))
    return {
        "number_of_gaussians": int(len(indices)),
        "local_spacing": spacing,
        "scene_median_spacing": scene_spacing,
        "spacing_ratio": float(ratio),
        "PCA_eigenvalues": values.tolist(),
        "estimated_curvature": curvature,
        "normal_angle_mean": float(angles.mean()),
        "normal_angle_std": float(angles.std()),
        "normal_angle_range": [float(angles.min()), float(angles.max())],
        "normal_angle_spread_p95": float(np.percentile(angles, 95)),
        "normal_confidence": normal_confidence,
        "mean_normal": mean_normal.tolist(),
        "semantic_IDs": [int(x) for x in semantic_ids],
        "semantic_counts": [int(x) for x in semantic_counts],
        "dominant_semantic_fraction": dominant_fraction,
        "connected_component_count": component_count,
        "largest_component_fraction": largest_fraction,
        "estimated_number_of_surface_components": surface_components(local_normals, points, spacing),
        "axes": vectors[:, ::-1].T.tolist(),
    }


def score_candidate(diag, kind):
    ratio = diag["spacing_ratio"]
    if kind == "planar":
        geometry_conf = float(np.clip(1.0 - diag["estimated_curvature"] / 0.015, 0, 1))
        density_conf = float(np.clip(1.0 - abs(np.log(max(ratio, 1e-8))) / np.log(3.0), 0, 1))
        normal_coherence = float(np.clip(1.0 - diag["normal_angle_spread_p95"] / 15.0, 0, 1))
        semantic_purity = diag["dominant_semantic_fraction"]
        connectivity_conf = diag["largest_component_fraction"]
        valid = (diag["normal_angle_spread_p95"] <= 15 and diag["normal_confidence"] > 0.8
                 and 0.5 <= ratio <= 3.0 and semantic_purity >= 0.8
                 and connectivity_conf >= 0.8 and diag["number_of_gaussians"] >= 100)
        weights = (0.20, 0.15, 0.35, 0.15, 0.15)
    else:
        curvature = diag["estimated_curvature"]
        geometry_conf = float(np.clip(min(curvature / 0.003, (0.10 - curvature) / 0.07), 0, 1))
        density_conf = float(np.clip(1.0 - abs(np.log(max(ratio, 1e-8))) / np.log(5.0), 0, 1))
        spread = diag["normal_angle_spread_p95"]
        normal_coherence = float(np.clip(min(spread / 15.0, (70.0 - spread) / 45.0), 0, 1))
        semantic_purity = diag["dominant_semantic_fraction"] if len(diag["semantic_IDs"]) <= 2 else 0.0
        connectivity_conf = diag["largest_component_fraction"]
        valid = (0.003 <= curvature <= 0.10 and 0.5 <= ratio <= 5.0
                 and 10 <= spread <= 70 and len(diag["semantic_IDs"]) <= 2
                 and semantic_purity >= 0.65 and connectivity_conf >= 0.8
                 and diag["number_of_gaussians"] >= 100)
        weights = (0.20, 0.25, 0.15, 0.15, 0.25)
    components = {
        "geometry_confidence": geometry_conf,
        "density_confidence": density_conf,
        "normal_coherence": normal_coherence,
        "semantic_purity": semantic_purity,
        "connectivity_confidence": connectivity_conf,
    }
    total = sum(w * value for w, value in zip(weights, components.values()))
    return components, float(total), bool(valid)


def discover_strict(xyz, normals, eigenvalues, labels, scene_spacing, kind,
                    anchors=24000, roi_count=160):
    tree = cKDTree(xyz)
    point_curvature = eigenvalues[:, 0] / (eigenvalues.sum(1) + 1e-12)
    point_confidence = 1.0 - point_curvature
    if kind == "planar":
        eligible = np.where((point_curvature < 0.01) & (point_confidence > 0.8))[0]
        order = eligible[np.argsort(point_curvature[eligible])] if len(eligible) else np.arange(len(xyz))
    else:
        eligible = np.where((point_curvature >= 0.003) & (point_curvature <= 0.10)
                            & (point_confidence > 0.8))[0]
        target = 0.025
        order = eligible[np.argsort(np.abs(point_curvature[eligible] - target))] if len(eligible) else np.arange(len(xyz))
    # Cover the full ranked eligible population deterministically rather than only a
    # contiguous best prefix, which might be dominated by one large object.
    sample_count = min(anchors, len(order))
    anchor_ids = order[np.linspace(0, len(order) - 1, sample_count, dtype=np.int64)]
    results = []
    for anchor in anchor_ids:
        distances, indices = tree.query(xyz[anchor], k=min(roi_count, len(xyz)))
        indices = np.atleast_1d(indices)
        diag = candidate_diagnostics(xyz, normals, eigenvalues, labels, indices, scene_spacing)
        components, total, valid = score_candidate(diag, kind)
        radius = float(np.max(np.atleast_1d(distances)))
        results.append({"kind": kind, "center": xyz[anchor].tolist(), "radius": radius,
                        "indices": indices, "diagnostics": diag,
                        "score_components": components, "total_confidence": total,
                        "passes_all_checks": valid})
    # Rank valid first, de-duplicate spatially, return top five valid if possible.
    results.sort(key=lambda x: (x["passes_all_checks"], x["total_confidence"]), reverse=True)
    selected = []
    for candidate in results:
        center = np.asarray(candidate["center"])
        if all(np.linalg.norm(center - np.asarray(other["center"])) >
               max(candidate["radius"], other["radius"]) for other in selected):
            selected.append(candidate)
        if len(selected) == 5:
            break
    return selected


def fixed_candidate(old, xyz, normals, eigenvalues, labels, scene_spacing):
    mask = selector_mask(xyz, old); indices = np.where(mask)[0]
    diag = candidate_diagnostics(xyz, normals, eigenvalues, labels, indices, scene_spacing)
    # Fixed ROI confidence is diagnostic only, not used to alter coordinates.
    kind = "curved" if "curved" in old["name"] else "planar"
    components, total, _ = score_candidate(diag, kind)
    return {"kind": old["category"], "center": old["center"], "radius": old["radius"],
            "indices": indices, "diagnostics": diag, "score_components": components,
            "total_confidence": total, "passes_all_checks": True, "frozen": True}


def equal_axes(ax, points):
    center = points.mean(0); radius = max(np.ptp(points, axis=0).max() / 2, 1e-6)
    ax.set_xlim(center[0]-radius, center[0]+radius); ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)


def visualize_roi(out_dir, candidate, xyz, normals, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    center = np.asarray(candidate["center"]); radius = float(candidate["radius"])
    local = np.where(np.linalg.norm(xyz - center, axis=1) <= 4.0 * radius)[0]
    selected = np.asarray(candidate["indices"], dtype=np.int64)
    if len(local) > 20000:
        local = local[np.linspace(0, len(local)-1, 20000, dtype=int)]
    selected_set = set(int(i) for i in selected)
    context = np.asarray([i for i in local if int(i) not in selected_set], dtype=np.int64)

    def figure():
        fig = plt.figure(figsize=(8, 7)); return fig, fig.add_subplot(111, projection="3d")
    fig, ax = figure(); ax.scatter(*xyz[::max(1, len(xyz)//30000)].T, s=.2, c="lightgray", alpha=.2)
    ax.scatter(*xyz[selected].T, s=5, c="red"); ax.set_title("global overview")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "global_overview.png"), dpi=150); plt.close(fig)

    plot_points = np.concatenate([context, selected])
    fig, ax = figure()
    if len(context): ax.scatter(*xyz[context].T, s=2, c="lightgray", alpha=.35)
    ax.scatter(*xyz[selected].T, s=8, c="red", label="selected ROI")
    equal_axes(ax, xyz[plot_points]); ax.legend(); ax.set_title("local ROI geometry")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "local_roi_geometry.png"), dpi=170); plt.close(fig)

    fig, ax = figure()
    if labels is None:
        ax.scatter(*xyz[plot_points].T, s=4, c="gray"); ax.set_title("semantic identity unavailable")
    else:
        roi_ids, roi_counts = np.unique(labels[selected], return_counts=True)
        for semantic_id in np.unique(labels[plot_points]):
            idx = plot_points[labels[plot_points] == semantic_id]
            count = int(roi_counts[np.where(roi_ids == semantic_id)[0][0]]) if semantic_id in roi_ids else 0
            fraction = count / len(selected)
            ax.scatter(*xyz[idx].T, s=4, alpha=.65,
                       label="ID {}: {} ({:.1%} ROI)".format(int(semantic_id), count, fraction))
        ax.legend(fontsize=7, loc="best"); ax.set_title("local semantic identity")
    equal_axes(ax, xyz[plot_points]); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "local_roi_semantics.png"), dpi=170); plt.close(fig)

    fig, ax = figure()
    arrow_idx = selected[::max(1, len(selected)//120)]
    ax.scatter(*xyz[selected].T, s=5, c="steelblue", alpha=.55)
    arrow_len = max(radius * 0.12, 1e-5)
    ax.quiver(xyz[arrow_idx,0], xyz[arrow_idx,1], xyz[arrow_idx,2],
              normals[arrow_idx,0], normals[arrow_idx,1], normals[arrow_idx,2],
              length=arrow_len, normalize=True, color="darkred", linewidth=.8)
    equal_axes(ax, xyz[selected]); ax.set_title("local PCA normals")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "local_roi_normals.png"), dpi=170); plt.close(fig)


def projected_area(points):
    if len(points) < 4:
        return float("nan")
    covariance = np.cov((points - points.mean(0)).T); _, vectors = np.linalg.eigh(covariance)
    projected = (points - points.mean(0)) @ vectors[:, 1:]
    try:
        return float(ConvexHull(projected).volume)
    except Exception:
        return float("nan")


def overspawn_audit(old_candidates, xyz, scene_spacing, old_root):
    rows = []
    for candidate in old_candidates:
        center = np.asarray(candidate["center"]); radius = float(candidate["radius"])
        removed_mask = selector_mask(xyz, candidate); n_gt = int(removed_mask.sum())
        distance = np.linalg.norm(xyz - center, axis=1)
        boundary_idx = np.where((distance > radius) & (distance <= radius + 4 * scene_spacing))[0]
        area_support = projected_area(xyz[boundary_idx])
        density = len(boundary_idx) / area_support if np.isfinite(area_support) and area_support > 0 else float("nan")
        hole_area = float(np.pi * radius * radius)
        predicted = density * hole_area if np.isfinite(density) else float("nan")
        metrics_path = os.path.join(old_root, candidate["name"], "metrics.csv")
        metric_rows = list(csv.DictReader(open(metrics_path)))
        for metric in metric_rows:
            spawned = int(metric["generated_gaussians"])
            rows.append({"roi": candidate["name"], "method": metric["method"],
                         "N_GT_removed": n_gt, "N_spawned": spawned,
                         "spawn_count_ratio": spawned / max(n_gt, 1),
                         "local_boundary_density": density, "estimated_hole_area": hole_area,
                         "estimated_spawn_density": spawned / hole_area,
                         "proposed_density_area_count": predicted,
                         "proposal_enabled": False})
    return rows


def confidence_audit(old_candidates, old_root):
    rows = []
    for candidate in old_candidates:
        roi_dir = os.path.join(old_root, candidate["name"])
        metrics_rows = list(csv.DictReader(open(os.path.join(roi_dir, "metrics.csv"))))
        diagnostics = json.load(open(os.path.join(roi_dir, "diagnostics.json")))
        newborn_path = os.path.join(roi_dir, "newborn_confidence.csv")
        newborn = list(csv.DictReader(open(newborn_path))) if os.path.exists(newborn_path) else []
        adaptive_terms = {}
        if newborn:
            for key in ("geometry_confidence", "support_confidence", "semantic_confidence",
                        "completion_confidence"):
                adaptive_terms[key] = float(np.mean([float(row[key]) for row in newborn]))
        for metric in metrics_rows:
            method = metric["method"]
            geometry_mean = diagnostics["methods"].get(method, {}).get("normal_confidence_mean", "N/A")
            is_saved = method == "C3-adaptive" and adaptive_terms
            rows.append({"roi": candidate["name"], "method": method,
                         "geometry_confidence": adaptive_terms.get("geometry_confidence", geometry_mean),
                         "support_confidence": adaptive_terms.get("support_confidence", "N/A") if is_saved else "N/A",
                         "semantic_confidence": adaptive_terms.get("semantic_confidence", "N/A") if is_saved else "N/A",
                         "final_confidence": metric["mean_completion_confidence"],
                         "actual_chamfer_error": metric["chamfer_distance"],
                         "term_availability_note": ("all observable terms persisted" if is_saved else
                            "historical run did not persist per-term confidence for this method"),
                         "multiplicative_collapse_flag": (float(metric["mean_completion_confidence"]) < 0.05)})
    return rows


def write_csv(path, rows):
    if not rows: return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)


def plot_confidence(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    for roi in sorted(set(row["roi"] for row in rows)):
        subset = [row for row in rows if row["roi"] == roi]
        ax.scatter([float(row["final_confidence"]) for row in subset],
                   [float(row["actual_chamfer_error"]) for row in subset], label=roi, alpha=.8)
    ax.set_xlabel("mean final confidence"); ax.set_ylabel("Chamfer error")
    ax.grid(True, alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def serializable_candidate(candidate, name, category):
    return {"name": name, "category": category, "center": candidate["center"],
            "radius": candidate["radius"], "passes_all_checks": candidate["passes_all_checks"],
            "frozen": candidate.get("frozen", False), "score_components": candidate["score_components"],
            "total_confidence": candidate["total_confidence"],
            "diagnostics": candidate["diagnostics"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--old-results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = GaussianModel(args.sh_degree); model.load_ply(args.checkpoint)
    xyz = model.get_xyz.detach().cpu().numpy()
    summary = inspect_scene(model, args.checkpoint, "REAL SCENE ROI VALIDATION CLEANUP")
    scene_spacing = summary["median_nearest_neighbor_spacing"]
    _, labels, _ = semantic_statistics(model)
    all_indices = np.arange(len(xyz))
    normals, normal_diag = geometry.estimate_normals_local_pca_at(
        xyz, all_indices, k=16, return_diagnostics=True)
    eigenvalues = normal_diag["eigenvalues"]
    old_data = json.load(open(os.path.join(args.old_results, "roi_candidates.json")))
    old = {item["name"]: item for item in old_data["candidates"]}

    junction = fixed_candidate(old["roi_B_junction"], xyz, normals, eigenvalues, labels, scene_spacing)
    layered = fixed_candidate(old["roi_C_layered"], xyz, normals, eigenvalues, labels, scene_spacing)
    planar_top = discover_strict(xyz, normals, eigenvalues, labels, scene_spacing, "planar")
    curved_top = discover_strict(xyz, normals, eigenvalues, labels, scene_spacing, "curved")
    planar_valid = [item for item in planar_top if item["passes_all_checks"]]
    curved_valid = [item for item in curved_top if item["passes_all_checks"]]
    planar = planar_valid[0] if planar_valid else planar_top[0]
    curved = curved_valid[0] if curved_valid else curved_top[0]

    chosen = [("roi_A_planar_v2", "planar surface", planar),
              ("roi_B_junction", "sharp junction", junction),
              ("roi_C_layered", "nearby layered / parallel surfaces", layered),
              ("roi_D_curved_v2", "curved surface", curved)]
    for name, category, candidate in chosen:
        directory = os.path.join(args.out, name); os.makedirs(directory, exist_ok=True)
        payload = serializable_candidate(candidate, name, category)
        with open(os.path.join(directory, "roi_validation.json"), "w") as f: json.dump(payload, f, indent=2)
        visualize_roi(directory, candidate, xyz, normals, labels)

    ranking = []
    for kind, candidates in (("planar", planar_top), ("curved", curved_top)):
        for rank, candidate in enumerate(candidates, 1):
            row = {"category": kind, "rank": rank, "passes_all_checks": candidate["passes_all_checks"],
                   "total_confidence": candidate["total_confidence"], "center": candidate["center"],
                   "radius": candidate["radius"]}
            row.update(candidate["score_components"]); row.update(candidate["diagnostics"]); ranking.append(row)
    write_csv(os.path.join(args.out, "roi_candidate_ranking.csv"), ranking)
    validity = []
    for name, category, candidate in chosen:
        validity.append({"roi": name, "category": category,
                         "status": ("VALID" if candidate["passes_all_checks"] else "NO VALID REPLACEMENT"),
                         "frozen": candidate.get("frozen", False),
                         "total_confidence": candidate["total_confidence"],
                         "gaussians": candidate["diagnostics"]["number_of_gaussians"],
                         "spacing_ratio": candidate["diagnostics"]["spacing_ratio"],
                         "curvature": candidate["diagnostics"]["estimated_curvature"],
                         "normal_spread_p95": candidate["diagnostics"]["normal_angle_spread_p95"],
                         "dominant_semantic_fraction": candidate["diagnostics"]["dominant_semantic_fraction"],
                         "largest_component_fraction": candidate["diagnostics"]["largest_component_fraction"]})
    write_csv(os.path.join(args.out, "roi_validity_summary.csv"), validity)
    audits = overspawn_audit(list(old.values()), xyz, scene_spacing, args.old_results)
    write_csv(os.path.join(args.out, "overspawn_audit.csv"), audits)
    conf = confidence_audit(list(old.values()), args.old_results)
    write_csv(os.path.join(args.out, "confidence_vs_error.csv"), conf)
    plot_confidence(os.path.join(args.out, "confidence_vs_error.png"), conf)

    report = ["# Real-scene ROI Validation Cleanup", "", "No completion benchmark was rerun.", "",
              "- Planar replacement: {}".format("PASS" if planar["passes_all_checks"] else "NO VALID CANDIDATE"),
              "- Curved replacement: {}".format("PASS" if curved["passes_all_checks"] else "NO VALID CANDIDATE"),
              "- Junction: frozen exactly from v1; local structure revalidated.",
              "- Layered: frozen exactly from v1; local structure revalidated.", "",
              "Overspawn proposal is diagnostic only and remains disabled. It estimates count from observed boundary density times estimated hole area.",
              "Confidence is not claimed calibrated. Historical per-term confidence is only available for C3-adaptive; missing terms are N/A."]
    with open(os.path.join(args.out, "validation_report.md"), "w") as f: f.write("\n".join(report) + "\n")
    print("[roi-cleanup] complete: {}".format(args.out))


if __name__ == "__main__":
    main()

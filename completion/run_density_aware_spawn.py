"""Isolated real-scene evaluation of the density-aware spawning replacement."""

import argparse
import csv
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion import geometry, metrics
from completion.gaussian_model import GaussianModel
from completion.run_real_controlled import subset_model


VARIANTS = ["C0", "C1", "C2", "C3"]
AFFINITY = {"roi_B_junction": "hard", "roi_C_layered": "soft",
            "roi_D_curved_v2": "hard", "roi_D_curved_legacy": "hard"}


def load_rois(v2_root, old_root):
    rois = []
    for name in ("roi_B_junction", "roi_C_layered", "roi_D_curved_v2"):
        item = json.load(open(os.path.join(v2_root, name, "roi_validation.json")))
        rois.append({"name": name, "center": item["center"], "radius": item["radius"],
                     "primary": True, "source": "v2"})
    old = json.load(open(os.path.join(old_root, "roi_candidates.json")))["candidates"]
    curved = next(item for item in old if item["name"] == "roi_D_curved")
    rois.append({"name": "roi_D_curved_legacy", "center": curved["center"],
                 "radius": curved["radius"], "primary": False, "source": "v1 legacy"})
    return rois


def aggregate_spawn(result):
    diagnostics = result.spawn_diagnostics or []
    support = sum(item["support_gaussians"] for item in diagnostics)
    support_area = sum(item["observable_support_area"] for item in diagnostics)
    missing_area = sum(item["estimated_missing_surface_area"] for item in diagnostics)
    spawned = sum(item["spawned_count"] for item in diagnostics)
    boundary_density = support / max(support_area, 1e-12)
    newborn_density = spawned / max(missing_area, 1e-12)
    return {"reliable_boundary_gaussians": support,
            "observable_support_area": support_area,
            "boundary_density": boundary_density,
            "estimated_missing_surface_area": missing_area,
            "estimated_spawn_density": boundary_density,
            "resulting_newborn_density": newborn_density,
            "density_ratio_newborn_boundary": newborn_density / max(boundary_density, 1e-12)}


def evaluate(original, removed, result):
    generated = result.new_xyz
    gt = removed.get_xyz.detach().cpu().numpy()
    gt_indices = np.where(~result.kept_mask)[0]
    xyz = original.get_xyz.detach().cpu().numpy()
    gt_normals = geometry.estimate_normals_local_pca_at(xyz, gt_indices, k=16)
    _, nearest = cKDTree(gt).query(generated, k=1)
    dots = np.clip(np.abs(np.sum(result.new_normals * gt_normals[nearest], axis=1)), 0, 1)
    new_sh = result.new_attributes["features_dc"]
    gt_sh = removed._features_dc.detach().cpu().numpy().reshape(len(gt), -1)
    bidx = result.boundary_idx
    battrs = original._features_dc.detach().cpu().numpy()[bidx].reshape(len(bidx), -1)
    return {"chamfer_distance": metrics.chamfer_distance(generated, gt),
            "normal_angular_error": float(np.degrees(np.arccos(dots)).mean()),
            "appearance_rmse": metrics.appearance_rmse_gen(generated, new_sh, gt, gt_sh),
            "boundary_seam_error": metrics.boundary_seam_error(
                generated, new_sh, xyz[bidx], battrs)}


def equal_axes(ax, points):
    center = points.mean(0); radius = max(np.ptp(points, axis=0).max() / 2, 1e-5)
    ax.set_xlim(center[0]-radius, center[0]+radius); ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)


def local_context(xyz, center, radius, mask, limit=6000):
    local = np.where(np.linalg.norm(xyz - center, axis=1) <= 2.0 * radius)[0]
    context = local[~mask[local]]
    if len(context) > limit:
        context = context[np.linspace(0, len(context)-1, limit, dtype=int)]
    return context


def plot_panel(ax, title, context, selected=None, generated=None, boundary=None,
               fitted=None, old=None):
    if len(context): ax.scatter(*context.T, s=1, c="lightgray", alpha=.25)
    if selected is not None and len(selected): ax.scatter(*selected.T, s=8, c="red", label="removed GT")
    if boundary is not None and len(boundary): ax.scatter(*boundary.T, s=6, c="black", label="boundary")
    if fitted is not None and len(fitted): ax.scatter(*fitted.T, s=4, c="deepskyblue", alpha=.4, label="fitted surface")
    if old is not None and len(old): ax.scatter(*old.T, s=4, c="orange", alpha=.55, label="old spawn")
    if generated is not None and len(generated): ax.scatter(*generated.T, s=7, c="blue", alpha=.8, label="newborn")
    all_points = [array for array in (context, selected, generated, boundary, fitted, old)
                  if array is not None and len(array)]
    equal_axes(ax, np.concatenate(all_points)); ax.set_title(title, fontsize=9)


def visualizations(directory, xyz, mask, results, legacy, radius, center):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    context_idx = local_context(xyz, center, radius, mask)
    context = xyz[context_idx]; gt = xyz[mask]

    fig = plt.figure(figsize=(20, 4))
    panels = [("Removed GT", gt, None), ("Hole", None, None)] + [
        (variant + "-new", None, results[variant].new_xyz) for variant in VARIANTS]
    for i, (title, selected, generated) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 6, i, projection="3d")
        plot_panel(ax, title, context, selected=selected, generated=generated)
    fig.tight_layout(); fig.savefig(os.path.join(directory, "methods_local_comparison.png"), dpi=170); plt.close(fig)

    c3 = results["C3"]
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    plot_panel(ax, "Removed GT vs C3 newborn", np.empty((0, 3)), selected=gt, generated=c3.new_xyz)
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(directory, "gt_vs_c3_newborn.png"), dpi=170); plt.close(fig)

    boundary = xyz[c3.boundary_idx]
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    plot_panel(ax, "Boundary + fitted surface + newborn", context, boundary=boundary,
               fitted=c3.fitted_xyz, generated=c3.new_xyz)
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(directory, "boundary_surface_newborn.png"), dpi=170); plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection="3d"); plot_panel(ax, "Legacy AABB grid", context, old=legacy.new_xyz)
    ax = fig.add_subplot(122, projection="3d"); plot_panel(ax, "Density-aware surface spawn", context, generated=c3.new_xyz)
    fig.tight_layout(); fig.savefig(os.path.join(directory, "old_vs_new_spawning.png"), dpi=170); plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--v2-rois", required=True)
    parser.add_argument("--old-results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = GaussianModel(3); model.load_ply(args.checkpoint)
    xyz = model.get_xyz.detach().cpu().numpy()
    rois = load_rois(args.v2_rois, args.old_results)
    summary_rows, diagnostic_rows, old_new_rows = [], [], []

    for roi in rois:
        name = roi["name"]; center = np.asarray(roi["center"], dtype=np.float32); radius = float(roi["radius"])
        mask = np.linalg.norm(xyz - center, axis=1) <= radius
        removed = subset_model(model, mask)
        scene = SimpleNamespace(name=name, model=model, hole_lo=center-radius, hole_hi=center+radius,
                                center=center, roi_center=center, roi_radius=radius)
        affinity = AFFINITY[name]
        legacy = geometry.run_completion(model, scene, baseline="C3", seed=0,
                                         normal_affinity=affinity, hole_mask_override=mask,
                                         spawn_rule="legacy")
        results = {}
        for variant in VARIANTS:
            start = time.time()
            result = geometry.run_completion(model, scene, baseline=variant, seed=0,
                                             normal_affinity=affinity, hole_mask_override=mask,
                                             spawn_rule="density_aware")
            elapsed = time.time() - start; results[variant] = result
            density = aggregate_spawn(result); quality = evaluate(model, removed, result)
            row = {"roi": name, "primary": roi["primary"], "method": variant,
                   "normal_affinity": affinity, "N_GT_removed": int(mask.sum()),
                   "N_spawn_old": len(legacy.new_xyz), "N_spawn_new": len(result.new_xyz),
                   "spawn_count_ratio": len(result.new_xyz) / max(int(mask.sum()), 1),
                   **density, **quality, "runtime_s": elapsed}
            summary_rows.append(row)
            for component in result.spawn_diagnostics:
                diagnostic_rows.append({"roi": name, "method": variant, **component})
        directory = os.path.join(args.out, name); os.makedirs(directory, exist_ok=True)
        visualizations(directory, xyz, mask, results, legacy, radius, center)
        c3 = next(row for row in summary_rows if row["roi"] == name and row["method"] == "C3")
        old_metrics_name = "roi_D_curved" if name == "roi_D_curved_legacy" else name
        old_metrics_path = os.path.join(args.old_results, old_metrics_name, "metrics.csv")
        old_metric = None
        if os.path.exists(old_metrics_path):
            old_rows = list(csv.DictReader(open(old_metrics_path)))
            old_metric = next((row for row in old_rows if row["method"] ==
                               ("C3-soft" if affinity == "soft" else "C3-hard")), None)
        old_new_rows.append({"roi": name, "N_GT_removed": int(mask.sum()),
                             "N_spawn_old": len(legacy.new_xyz), "N_spawn_new": c3["N_spawn_new"],
                             "old_spawn_ratio": len(legacy.new_xyz)/max(int(mask.sum()),1),
                             "new_spawn_ratio": c3["spawn_count_ratio"],
                             "old_chamfer": old_metric["chamfer_distance"] if old_metric else "N/A",
                             "new_chamfer": c3["chamfer_distance"],
                             "old_normal_error": old_metric["normal_angular_error"] if old_metric else "N/A",
                             "new_normal_error": c3["normal_angular_error"]})

    write_csv(os.path.join(args.out, "spawn_density_summary.csv"), summary_rows)
    write_csv(os.path.join(args.out, "density_diagnostics.csv"), diagnostic_rows)
    write_csv(os.path.join(args.out, "old_vs_new_metrics.csv"), old_new_rows)

    lines = ["# Density-aware Gaussian Spawn Validation", "",
             "Only the Gaussian spawning-density and surface-sampling rule changed.",
             "Graph construction, C0-C3, normal estimation, semantics, fitting, affinity, confidence and renderer were frozen.", ""]
    for roi in rois:
        rr = [row for row in summary_rows if row["roi"] == roi["name"]]
        c0 = next(row for row in rr if row["method"] == "C0")
        c1 = next(row for row in rr if row["method"] == "C1")
        c3 = next(row for row in rr if row["method"] == "C3")
        lines += ["## {}".format(roi["name"]),
                  "- old/new spawn: {}/{}; new/GT ratio {:.3f}".format(
                      c3["N_spawn_old"], c3["N_spawn_new"], c3["spawn_count_ratio"]),
                  "- Chamfer C0/C1/C3: {:.6f} / {:.6f} / {:.6f}".format(
                      c0["chamfer_distance"], c1["chamfer_distance"], c3["chamfer_distance"]), ""]
    with open(os.path.join(args.out, "validation_report.md"), "w") as f: f.write("\n".join(lines))
    print("[density-aware] completed {} cells".format(len(summary_rows)))


if __name__ == "__main__":
    main()

"""One-command real GG checkpoint -> automatic ROI -> controlled-hole validation."""

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
from completion.discover_real_rois import run as discover_rois, semantic_statistics
from completion.gaussian_model import GaussianModel, make_cameras_from_poses, make_orbit_poses


CANONICAL = ["C0", "C1", "C2", "C3"]
METHODS = [
    ("C0", "C0", "hard", "hard"),
    ("C1-hard", "C1", "hard", "hard"),
    ("C1-soft", "C1", "soft", "hard"),
    ("C1-adaptive", "C1", "adaptive", "hard"),
    ("C2-hard", "C2", "hard", "hard"),
    ("C2-soft", "C2", "soft", "hard"),
    ("C2-adaptive", "C2", "adaptive", "hard"),
    ("C3-hard", "C3", "hard", "hard"),
    ("C3-soft", "C3", "soft", "hard"),
    ("C3-adaptive", "C3", "adaptive", "hard"),
    ("C3-no-semantic-hard-gate", "C3", "adaptive", "soft"),
]

FIELDS = ["result_type", "roi", "category", "method", "variant", "normal_affinity",
          "semantic_gate", "chamfer_distance", "normal_angular_error",
          "surface_leakage", "surface_leakage_note", "appearance_rmse",
          "boundary_seam_error", "hole_psnr", "hole_ssim", "edge_reconstruction_error",
          "render_metric_note", "generated_gaussians", "runtime_s",
          "mean_completion_confidence", "low_confidence_fraction",
          "confidence_error_correlation", "graph_component_count"]


def subset_model(model, mask):
    import torch.nn as nn
    out = GaussianModel(model.max_sh_degree)
    out.active_sh_degree = model.active_sh_degree; out.num_objects = model.num_objects
    out.spatial_lr_scale = model.spatial_lr_scale
    for name in ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
                 "_rotation", "_objects_dc"]:
        setattr(out, name, nn.Parameter(getattr(model, name).detach().cpu()[mask]))
    return out


def selector_mask(xyz, selector):
    kind = selector.get("type", "sphere")
    if kind == "sphere":
        center = np.asarray(selector["center"], dtype=np.float32); radius = float(selector["radius"])
        mask = np.linalg.norm(xyz - center, axis=1) <= radius
        return mask, center - radius, center + radius
    if kind == "aabb":
        lo = np.asarray(selector["min"], dtype=np.float32); hi = np.asarray(selector["max"], dtype=np.float32)
        return np.all((xyz >= lo) & (xyz <= hi), axis=1), lo, hi
    if kind == "oriented_box":
        center = np.asarray(selector["center"], dtype=np.float32)
        axes = np.asarray(selector["axes"], dtype=np.float32)
        half = np.asarray(selector["half_extent"], dtype=np.float32)
        local = (xyz - center) @ axes.T; mask = np.all(np.abs(local) <= half, axis=1)
        corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * half
        world = corners @ axes + center
        return mask, world.min(0), world.max(0)
    raise ValueError("unsupported selector {}".format(kind))


def cameras_for_roi(center, extent, resolution, count=3):
    poses = make_orbit_poses(n=count, elevation_deg=30.0, radius=max(4.0 * extent, 0.1),
                             center=tuple(center), surface_axis=2)
    return make_cameras_from_poses(poses, height=resolution, width=int(resolution * 4 / 3),
                                   fov_deg=50.0)


def render_metrics(original, hole, completed, views):
    from completion.render import render_original_hole_completed
    rendered = render_original_hole_completed(original, hole, completed, views)
    orig, empty, filled = rendered["original"][0], rendered["hole"][0], rendered["completed"][0]
    masks = (orig - empty).abs().mean(1) > 0.02
    p, s, e = [], [], []
    for index in range(len(views)):
        p.append(metrics.psnr(filled[index], orig[index], masks[index]))
        s.append(metrics.ssim(filled[index], orig[index], masks[index]))
        e.append(metrics.edge_reconstruction_error(filled[index], orig[index], masks[index]))
    def safe_mean(values):
        finite = [float(value) for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else "N/A"
    return safe_mean(p), safe_mean(s), safe_mean(e)


def evaluate(original, hole, removed, result, completed, gt_normals, views, render_enabled):
    gen = result.new_xyz; gt = removed.get_xyz.detach().cpu().numpy()
    nearest_distance, nearest = cKDTree(gt).query(gen, k=1)
    dots = np.clip(np.abs(np.sum(result.new_normals * gt_normals[nearest], axis=1)), 0, 1)
    normal_error = float(np.degrees(np.arccos(dots)).mean())
    new_sh = result.new_attributes["features_dc"]
    gt_sh = removed._features_dc.detach().cpu().numpy().reshape(len(gt), -1)
    bidx = result.boundary_idx; xyz = original.get_xyz.detach().cpu().numpy()
    battrs = original._features_dc.detach().cpu().numpy()[bidx].reshape(len(bidx), -1)
    confidence = result.completion_confidence
    correlation = "N/A"
    if len(confidence) > 2 and np.std(confidence) > 1e-8 and np.std(nearest_distance) > 1e-8:
        correlation = float(np.corrcoef(confidence, nearest_distance)[0, 1])
    if render_enabled:
        psnr, ssim, edge = render_metrics(original, hole, completed, views)
        if any(value == "N/A" for value in (psnr, ssim, edge)):
            render_note = ("N/A where the inferred low-resolution camera produced no "
                           "non-empty hole-region pixels; CPU fallback renderer; no "
                           "checkpoint camera file supplied")
        else:
            render_note = "CPU fallback renderer; identical inferred ROI cameras; no checkpoint camera file supplied"
    else:
        psnr = ssim = edge = "N/A"; render_note = "rendering disabled"
    return {
        "chamfer_distance": metrics.chamfer_distance(gen, gt),
        "normal_angular_error": normal_error,
        "surface_leakage": "N/A",
        "surface_leakage_note": "No independent real surface-identity ground truth is available",
        "appearance_rmse": metrics.appearance_rmse_gen(gen, new_sh, gt, gt_sh),
        "boundary_seam_error": metrics.boundary_seam_error(gen, new_sh, xyz[bidx], battrs),
        "hole_psnr": psnr, "hole_ssim": ssim, "edge_reconstruction_error": edge,
        "render_metric_note": render_note, "generated_gaussians": len(gen),
        "mean_completion_confidence": float(np.mean(confidence)),
        "low_confidence_fraction": float(np.mean(confidence < 0.5)),
        "confidence_error_correlation": correlation,
        "graph_component_count": int(result.n_components),
    }


def save_confidence(region_dir, result):
    path = os.path.join(region_dir, "newborn_confidence.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["x", "y", "z", "geometry_confidence",
                                                 "support_confidence", "semantic_confidence",
                                                 "completion_confidence"])
        terms = result.confidence_terms
        for i, point in enumerate(result.new_xyz):
            writer.writerow([*point, terms["geometry"][i], terms["support"][i],
                             terms["semantic"][i], result.completion_confidence[i]])


def confidence_visualization(path, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    points = result.new_xyz; conf = result.completion_confidence
    fig = plt.figure(figsize=(7, 5)); ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(*points.T, c=conf, cmap="viridis", vmin=0, vmax=1, s=7)
    fig.colorbar(scatter, label="completion confidence"); fig.tight_layout(); fig.savefig(path, dpi=150)
    plt.close(fig)


def strip_image(path, panels, views):
    from PIL import Image, ImageDraw
    from completion import render
    images = []
    for label, model in panels:
        tensor = render.render_set(views[:1], model)[0][0]
        arr = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
        im = Image.fromarray(arr); panel = Image.new("RGB", (im.width, im.height + 22), "white")
        panel.paste(im, (0, 22)); ImageDraw.Draw(panel).text((5, 4), label, fill="black"); images.append(panel)
    strip = Image.new("RGB", (sum(x.width for x in images), images[0].height), "white")
    offset = 0
    for image in images: strip.paste(image, (offset, 0)); offset += image.width
    strip.save(path)


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)


def run_roi(model, candidate, scene_dir, result_type, resolution, no_render, skip_ply=False):
    xyz = model.get_xyz.detach().cpu().numpy()
    mask, lo, hi = selector_mask(xyz, candidate["selector"])
    if mask.sum() < 8: raise RuntimeError("{} selected fewer than 8 Gaussians".format(candidate["name"]))
    region_dir = os.path.join(scene_dir, candidate["name"]); os.makedirs(region_dir, exist_ok=True)
    config = {"result_type": result_type, **candidate, "fixed_hyperparameters": {
        "boundary_k": 16, "knn_k": 12, "normal_sigma_deg": 30.0,
        "normal_edge_min": 0.10, "adaptive_min_sigma_deg": 5.0}}
    with open(os.path.join(region_dir, "config.json"), "w") as f: json.dump(config, f, indent=2)
    hole = subset_model(model, ~mask); removed = subset_model(model, mask)
    if not skip_ply:
        model.save_ply(os.path.join(region_dir, "original.ply")); hole.save_ply(os.path.join(region_dir, "hole.ply"))
        removed.save_ply(os.path.join(region_dir, "removed_gt.ply"))
    gt_normals = geometry.estimate_normals_local_pca_at(xyz, np.where(mask)[0], k=16)
    scene = SimpleNamespace(name="real_" + candidate["name"], model=model, hole_lo=lo,
                            hole_hi=hi, center=0.5 * (lo + hi))
    views = cameras_for_roi(scene.center, max(float(np.max(hi - lo)), 1e-3), resolution)
    rows, models, results = [], {}, {}
    for method, variant, affinity, semantic_gate in METHODS:
        start = time.time()
        result = geometry.run_completion(model, scene, baseline=variant, seed=0,
                                         normal_affinity=affinity, semantic_gate=semantic_gate,
                                         hole_mask_override=mask)
        completed = geometry.append_gaussians(hole, result)
        row = {"result_type": result_type, "roi": candidate["name"],
               "category": candidate["category"], "method": method, "variant": variant,
               "normal_affinity": affinity, "semantic_gate": semantic_gate}
        row.update(evaluate(model, hole, removed, result, completed, gt_normals, views,
                            not no_render)); row["runtime_s"] = time.time() - start
        rows.append(row); models[method] = completed; results[method] = result
    # Canonical files use hard affinity for C1-C3, exactly as named in the ablation.
    canonical_methods = {"C0": "C0", "C1": "C1-hard", "C2": "C2-hard", "C3": "C3-hard"}
    if not skip_ply:
        for name, method in canonical_methods.items():
            models[method].save_ply(os.path.join(region_dir, name + ".ply"))
    save_confidence(region_dir, results["C3-adaptive"])
    confidence_visualization(os.path.join(region_dir, "confidence_visualization.png"),
                             results["C3-adaptive"])
    raw_sem, labels, sem_stats = semantic_statistics(model)
    diagnostics = {"result_type": result_type, "candidate": candidate,
                   "semantic_statistics": sem_stats, "methods": {}}
    for method, result in results.items():
        diagnostics["methods"][method] = {
            "pca_eigenvalue_mean": np.mean(result.pca_eigenvalues, axis=0).tolist(),
            "normal_confidence_mean": float(np.mean(result.normal_confidence)),
            "neighbor_count": int(np.median(result.neighbor_count)),
            "local_curvature_mean": float(np.mean(result.local_curvature)),
            "graph_component_count": int(result.n_components)}
    with open(os.path.join(region_dir, "diagnostics.json"), "w") as f: json.dump(diagnostics, f, indent=2)
    write_rows(os.path.join(region_dir, "metrics.csv"), rows)
    if not no_render:
        strip_image(os.path.join(region_dir, "comparison.png"),
                    [("GT", model), ("Hole", hole), ("C0", models["C0"]),
                     ("C1", models["C1-hard"]), ("C2", models["C2-hard"]),
                     ("C3", models["C3-hard"])], views)
        strip_image(os.path.join(region_dir, "normal_affinity_comparison.png"),
                    [("C3-hard", models["C3-hard"]), ("C3-soft", models["C3-soft"]),
                     ("C3-adaptive", models["C3-adaptive"])], views)
        render_dir = os.path.join(region_dir, "novel_views"); os.makedirs(render_dir, exist_ok=True)
        for index in (1, 2):
            strip_image(os.path.join(render_dir, "view_{}.png".format(index)),
                        [("GT", model), ("Hole", hole), ("C3", models["C3-hard"])], views[index:index+1])
    return rows


def report(scene_dir, result_type, summary, candidates, rows):
    ablation = [r for r in rows if r["method"] in ("C0", "C1-hard", "C2-hard", "C3-hard")]
    affinity = [r for r in rows if r["variant"] in ("C1", "C2", "C3")]
    write_rows(os.path.join(scene_dir, "real_ablation_summary.csv"), ablation)
    write_rows(os.path.join(scene_dir, "real_affinity_summary.csv"), affinity)
    lines = ["# Real Gaussian Completion Validation", "", "**{}**".format(result_type), "",
             "- Checkpoint: `{}`".format(summary["checkpoint"]),
             "- Gaussians: {}".format(summary["number_of_gaussians"]),
             "- Automatically selected ROIs: {}".format(len(candidates)), "",
             "Surface leakage is N/A because no independent real surface-identity ground truth is available.",
             "Render metrics use identical inferred ROI cameras and the CPU fallback renderer when checkpoint cameras are absent."]
    with open(os.path.join(scene_dir, "real_validation_report.md"), "w") as f: f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/real_validation")
    parser.add_argument("--scene-name")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--skip-ply", action="store_true",
                        help="low-disk mode: run all methods/metrics but omit repeated PLY snapshots")
    args = parser.parse_args()
    result_type = "SMOKE TEST" if args.smoke_test else "REAL SCENE RESULT"
    scene_name = args.scene_name or os.path.basename(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(args.checkpoint))))) or "scene"
    scene_dir = os.path.join(args.out, scene_name); os.makedirs(scene_dir, exist_ok=True)
    summary, candidates = discover_rois(args.checkpoint, scene_dir, args.sh_degree, result_type)
    model = GaussianModel(args.sh_degree); model.load_ply(args.checkpoint)
    rows = []
    for candidate in candidates:
        rows.extend(run_roi(model, candidate, scene_dir, result_type, args.resolution,
                            args.no_render, args.skip_ply))
    report(scene_dir, result_type, summary, candidates, rows)
    print("[real-validation] {} completed: {}".format(result_type, scene_dir))


if __name__ == "__main__":
    main()

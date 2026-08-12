"""Sharp-corner robustness validation for Gaussian completion.

Runs every combination of 9 ground-truth angles, C0-C3, three normal-affinity
strategies, and deterministic seeds.  All affinity parameters are global constants;
nothing is tuned per angle.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion import geometry, metrics
from completion.gaussian_model import GaussianModel, make_cameras_from_poses, make_orbit_poses
from completion.synthetic_scene import get_scene


ANGLES = [30, 45, 60, 75, 90, 105, 120, 135, 150]
VARIANTS = ["C0", "C1", "C2", "C3"]
AFFINITIES = ["hard", "soft", "adaptive"]

FIELDS = [
    "gt_corner_angle", "normal_affinity", "variant", "seed",
    "recovered_corner_angle", "abs_corner_angle_error", "normal_angular_error",
    "surface_leakage", "chamfer_distance", "boundary_seam_error",
    "normal_estimation_error", "graph_edge_cross_surface_rate",
    "graph_partition_pair_error", "mls_fit_chamfer", "mls_fit_normal_error",
    "gaussian_spawn_rms", "gaussian_spawn_chamfer_delta", "generated", "runtime_s",
]


def subset_model(model, mask):
    import torch.nn as nn
    out = GaussianModel(model.max_sh_degree)
    out.active_sh_degree = model.active_sh_degree
    out.num_objects = model.num_objects
    for name in ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
                 "_rotation", "_objects_dc"]:
        setattr(out, name, nn.Parameter(getattr(model, name).detach().cpu()[mask]))
    return out


def angular_error(normals, reference):
    if len(normals) == 0:
        return float("nan")
    dots = np.clip(np.abs(np.sum(normals * reference, axis=1)), 0.0, 1.0)
    return float(np.degrees(np.arccos(dots)).mean())


def pair_partition_error(pred, truth):
    """Fraction of node pairs for which same/different-component decisions disagree."""
    if len(pred) < 2:
        return float("nan")
    iu = np.triu_indices(len(pred), k=1)
    return float(np.mean((pred[iu[0]] == pred[iu[1]]) !=
                         (truth[iu[0]] == truth[iu[1]])))


def stage_diagnostics(scene, result, removed_gt):
    from scipy.spatial import cKDTree

    bidx = result.boundary_idx
    normal_est = angular_error(result.normals, scene.gt_normal[bidx])
    edge_cross = float("nan")
    partition_error = float("nan")
    if result.graph_rows is not None:
        labels = scene.gt_surface[bidx]
        if len(result.graph_rows):
            edge_cross = float(np.mean(labels[result.graph_rows] != labels[result.graph_cols]))
        partition_error = pair_partition_error(result.component_labels, labels)

    gt_xyz = removed_gt.get_xyz.detach().cpu().numpy()
    hole_idx = np.where(~result.kept_mask)[0]
    gt_normals = scene.gt_normal[hole_idx]
    fit_chamfer = metrics.chamfer_distance(result.fitted_xyz, gt_xyz)
    _, nearest = cKDTree(gt_xyz).query(result.fitted_xyz, k=1)
    fit_normal = angular_error(result.fitted_normals, gt_normals[nearest])
    spawn_rms = float(np.sqrt(np.mean(np.sum(
        (result.new_xyz - result.fitted_xyz) ** 2, axis=1))))
    spawn_delta = metrics.chamfer_distance(result.new_xyz, gt_xyz) - fit_chamfer
    return {
        "normal_estimation_error": normal_est,
        "graph_edge_cross_surface_rate": edge_cross,
        "graph_partition_pair_error": partition_error,
        "mls_fit_chamfer": fit_chamfer,
        "mls_fit_normal_error": fit_normal,
        "gaussian_spawn_rms": spawn_rms,
        "gaussian_spawn_chamfer_delta": spawn_delta,
    }


def run_cell(angle, affinity, variant, seed):
    scene = get_scene("l_corner", seed=seed, corner_angle=angle)
    model = scene.model
    start = time.time()
    result = geometry.run_completion(model, scene, baseline=variant, seed=seed,
                                     normal_affinity=affinity)
    kept = result.kept_mask
    hole_model = subset_model(model, kept)
    removed = subset_model(model, ~kept)
    completed = geometry.append_gaussians(hole_model, result)
    report = metrics.report_metrics(model, hole_model, completed, removed, scene, result,
                                    views=None)
    row = {
        "gt_corner_angle": angle,
        "normal_affinity": affinity,
        "variant": variant,
        "seed": seed,
        "recovered_corner_angle": report["recovered_angle"],
        "abs_corner_angle_error": report["corner_angle_err"],
        "normal_angular_error": report["normal_error_deg"],
        "surface_leakage": report["leakage"],
        "chamfer_distance": report["chamfer"],
        "boundary_seam_error": report["seam_error"],
        "generated": report["generated"],
    }
    row.update(stage_diagnostics(scene, result, removed))
    row["runtime_s"] = time.time() - start
    return row, scene, hole_model, completed


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(out_dir, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [
        ("abs_corner_angle_error", "absolute corner-angle error (deg)",
         "corner_angle_error_vs_gt.png"),
        ("surface_leakage", "surface leakage", "leakage_vs_corner_angle.png"),
    ]
    for metric, ylabel, filename in specs:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for ax, affinity in zip(axes, AFFINITIES):
            for variant in VARIANTS:
                ys = []
                for angle in ANGLES:
                    vals = [float(r[metric]) for r in rows
                            if r["normal_affinity"] == affinity and
                            r["variant"] == variant and r["gt_corner_angle"] == angle]
                    ys.append(float(np.nanmean(vals)))
                ax.plot(ANGLES, ys, marker="o", label=variant)
            ax.set_title(affinity)
            ax.set_xlabel("ground-truth corner angle (deg)")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, filename), dpi=160)
        plt.close(fig)


def render_representatives(out_dir, resolution):
    from PIL import Image, ImageDraw
    from completion import render

    render_dir = os.path.join(out_dir, "representative_visualizations")
    os.makedirs(render_dir, exist_ok=True)
    for angle in (45, 90, 135):
        for affinity in AFFINITIES:
            models = {}
            scene = hole_model = None
            for variant in VARIANTS:
                _, scene, hole_model, models[variant] = run_cell(angle, affinity, variant, 0)
            poses = make_orbit_poses(n=1, elevation_deg=35.0, radius=1.2,
                                     center=tuple(scene.center), surface_axis=2)
            views = make_cameras_from_poses(poses, height=resolution,
                                            width=int(resolution * 4 / 3), fov_deg=50.0)
            panels = [("GT", scene.model), ("Hole", hole_model)] + list(models.items())
            images = []
            for label, model in panels:
                tensor = render.render_set(views, model)[0][0]
                arr = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
                im = Image.fromarray(arr)
                canvas = Image.new("RGB", (im.width, im.height + 22), "white")
                canvas.paste(im, (0, 22))
                ImageDraw.Draw(canvas).text((5, 4), label, fill="black")
                images.append(canvas)
            strip = Image.new("RGB", (sum(i.width for i in images), images[0].height), "white")
            x = 0
            for image in images:
                strip.paste(image, (x, 0)); x += image.width
            strip.save(os.path.join(render_dir, "corner_{}_{}.png".format(angle, affinity)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/next_stage/synthetic/corner_robustness")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resolution", type=int, default=180)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    total = len(ANGLES) * len(AFFINITIES) * len(VARIANTS) * args.seeds
    for angle in ANGLES:
        for affinity in AFFINITIES:
            for variant in VARIANTS:
                for seed in range(args.seeds):
                    row, _, _, _ = run_cell(angle, affinity, variant, seed)
                    rows.append(row)
                    print("[corner] {}/{} angle={} affinity={} variant={} seed={}".format(
                        len(rows), total, angle, affinity, variant, seed), flush=True)
    write_csv(os.path.join(args.out, "corner_robustness.csv"), rows)
    make_plots(args.out, rows)
    if args.render:
        render_representatives(args.out, args.resolution)


if __name__ == "__main__":
    main()

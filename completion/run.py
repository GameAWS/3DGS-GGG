"""CLI driver for the Gaussian completion benchmark.

Usage:
    python completion/run.py --baseline C --out output/completion
    python completion/run.py --baseline A
    python completion/run.py --baseline B --hole-size 0.3

Workflow (steps mapped to geometry.py):
  1. load a "trained" GG model  (synthetic stand-in here)
  2. select axis-aligned cuboid on smooth surface & save GT Gaussians
  3. remove them -> hole
  4. detect boundary via KNN
  5. local-PCA normals
  6. KNN graph (pos / normal / SH-DC)
  7. weighted local plane fit
  8. sample centers at median spacing
  9. propagate SH/opacity/scale/rot/semantic from boundary
 10. append to GaussianModel
 11. save original.ply / hole.ply / completed.ply / removed_ground_truth.ply
 12. render original, hole, completed from identical cameras
 13. report hole-region PSNR/SSIM/LPIPS, Chamfer, counts, runtime
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# Make the repo root importable when run as `python completion/run.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion.gaussian_model import GaussianModel, make_cameras_from_poses, make_orbit_poses


def build_hole_model(model, kept_mask):
    """Return a copy of `model` with the hole Gaussians removed (kept only)."""
    new = GaussianModel(model.max_sh_degree)
    new.max_sh_degree = model.max_sh_degree
    new.active_sh_degree = model.active_sh_degree
    new.num_objects = model.num_objects
    new.spatial_lr_scale = model.spatial_lr_scale
    for name in ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
                 "_rotation", "_objects_dc"]:
        t = getattr(model, name).detach().cpu()
        setattr(new, name, torch.nn.Parameter(t[kept_mask]))
    return new


def build_removed_gt_model(model, hole_mask):
    """Return a copy of `model` containing ONLY the removed ground-truth Gaussians."""
    new = GaussianModel(model.max_sh_degree)
    new.max_sh_degree = model.max_sh_degree
    new.active_sh_degree = model.active_sh_degree
    new.num_objects = model.num_objects
    new.spatial_lr_scale = model.spatial_lr_scale
    for name in ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
                 "_rotation", "_objects_dc"]:
        t = getattr(model, name).detach().cpu()
        setattr(new, name, torch.nn.Parameter(t[hole_mask]))
    return new


def main():
    parser = argparse.ArgumentParser(description="Gaussian completion benchmark")
    parser.add_argument("--baseline", choices=["A", "B", "C"], default="C",
                        help="completion baseline: A=NN clone, B=pos-only graph, "
                             "C=pos+normal+appearance graph")
    parser.add_argument("--out", default="output/completion",
                        help="output directory for PLYs and metrics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hole-size", type=float, default=None,
                        help="cuboid in-plane half-size (auto if omitted)")
    parser.add_argument("--resolution", type=int, default=360,
                        help="render height; width = int(height*4/3)")
    parser.add_argument("--n-views", type=int, default=3)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--spacing", type=float, default=0.02,
                        help="Gaussian spacing in the synthetic scene")
    args = parser.parse_args()

    from completion.synthetic_scene import build_synthetic_gaussians
    from completion import geometry
    from completion import metrics

    t_start = time.time()

    # Step 1: load the "trained" model (synthetic stand-in).
    model = build_synthetic_gaussians(seed=args.seed, sh_degree=args.sh_degree,
                                      spacing=args.spacing)
    xyz = model._xyz.detach().cpu().numpy()

    # Step 2-3: cuboid selection + hole carving.
    lo, hi = geometry.select_ground_truth_cuboid(xyz, surface_axis=2,
                                                 size=args.hole_size)
    kept_mask, hole_mask = geometry.carve_hole(xyz, lo, hi)
    print("[run] cuboid lo={} hi={}  kept={} hole={}".format(
        np.round(lo, 3), np.round(hi, 3), kept_mask.sum(), hole_mask.sum()))

    removed_gt = build_removed_gt_model(model, hole_mask)   # evaluation only
    hole_model = build_hole_model(model, kept_mask)          # artificial hole

    # Steps 4-9: run the completion on the ORIGINAL model (it carves the hole itself).
    result = geometry.run_completion(
        model, lo, hi, baseline=args.baseline, seed=args.seed,
        sh_degree=args.sh_degree)

    # Step 10: append new Gaussians.
    completed = geometry.append_gaussians(hole_model, result)

    # Step 11: save PLYs.
    os.makedirs(args.out, exist_ok=True)
    model.save_ply(os.path.join(args.out, "original.ply"))
    hole_model.save_ply(os.path.join(args.out, "hole.ply"))
    completed.save_ply(os.path.join(args.out, "completed.ply"))
    removed_gt.save_ply(os.path.join(args.out, "removed_ground_truth.ply"))
    print("[run] saved 4 PLYs to {}".format(args.out))

    # Step 12: render from identical cameras.
    H = args.resolution
    W = int(H * 4 / 3)
    poses = make_orbit_poses(n=args.n_views, elevation_deg=50.0, radius=1.2)
    views = make_cameras_from_poses(poses, height=H, width=W, fov_deg=50.0)

    # Step 12b: save the rendered views as PNGs (original / hole / completed side by side).
    try:
        from completion.render import render_original_hole_completed
        from PIL import Image
        rendered = render_original_hole_completed(model, hole_model, completed, views)
        for name in ("original", "hole", "completed"):
            imgs = rendered[name][0]
            for v in range(len(views)):
                img = (imgs[v].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
                Image.fromarray(img).save(
                    os.path.join(args.out, "{}_{}.png".format(name, v)))
        print("[run] saved rendered views to {}".format(args.out))
    except Exception as e:
        print("[run] warning: could not save preview PNGs ({})".format(e))

    # Step 13: report.
    out = metrics.report_metrics(model, hole_model, completed, removed_gt, views,
                                 generated_xyz=result.new_xyz)
    runtime_s = time.time() - t_start
    print(metrics.format_report(out, args.baseline, runtime_s))

    # Save a machine-readable copy.
    import json
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"baseline": args.baseline, **out, "runtime_s": runtime_s}, f, indent=2)
    print("[run] metrics written to {}/metrics.json".format(args.out))


if __name__ == "__main__":
    main()
"""4 scenes x 3 baselines x 5 seeds sweep -> one CSV + side-by-side rendered images.

Each cell runs the completion and reports geometric + render metrics.  Render metrics
use a low resolution by default to keep the sweep fast; the geometric metrics (leakage,
normal error, Chamfer, appearance RMSE, seam error) are exact.

Outputs:
  output/sweep/sweep.csv            -- one row per (scene, baseline, seed)
  output/sweep/<scene>/<seed>/<A|B|C>_v{v}.png   -- completed renders
  output/sweep/side_by_side/<scene>.png          -- GT | Hole | A | B | C (seed 0)
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion.synthetic_scene import get_scene, SCENES
from completion import geometry, metrics
from completion.gaussian_model import GaussianModel, make_orbit_poses, make_cameras_from_poses


def _subset(m, mask):
    import torch.nn as nn
    n = GaussianModel(m.max_sh_degree)
    for name in ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
                 "_rotation", "_objects_dc"]:
        setattr(n, name, nn.Parameter(getattr(m, name).detach().cpu()[mask]))
    return n


def render_views(model, scene, resolution=140, n_views=3):
    """Render a model from cameras looking at the scene centre."""
    H = resolution
    W = int(H * 4 / 3)
    poses = make_orbit_poses(n=n_views, elevation_deg=50.0, radius=1.2,
                             center=tuple(scene.center), surface_axis=2)
    return make_cameras_from_poses(poses, height=H, width=W, fov_deg=50.0)


def run_cell(scene_name, baseline, seed, sh_degree=3, render=True, resolution=140):
    sc = get_scene(scene_name, seed=seed, sh_degree=sh_degree)
    model = sc.model
    t0 = time.time()
    r = geometry.run_completion(model, sc, baseline=baseline, seed=seed)
    kept, hole = r.kept_mask, ~r.kept_mask
    hole_m = _subset(model, kept)
    removed = _subset(model, hole)
    completed = geometry.append_gaussians(hole_m, r)

    views = render_views(model, sc, resolution=resolution) if render else None
    out = metrics.report_metrics(model, hole_m, completed, removed, sc, r, views=views,
                                 resolution=resolution)
    out["runtime_s"] = time.time() - t0
    out["scene"] = scene_name
    out["baseline"] = baseline
    out["seed"] = seed
    return completed, views, out


COLUMNS = ["scene", "baseline", "seed", "psnr", "ssim", "edge_err", "chamfer",
           "normal_error_deg", "leakage", "appearance_rmse", "seam_error",
           "generated", "gaussians", "runtime_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/sweep")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--render", action="store_true", default=True)
    ap.add_argument("--no-render", dest="render", action="store_false")
    ap.add_argument("--resolution", type=int, default=140)
    ap.add_argument("--scene", default=None, help="restrict to one scene for debug")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scene_names = [args.scene] if args.scene else list(SCENES)
    rows = []

    for scene_name in scene_names:
        sc0 = get_scene(scene_name, seed=0)
        views0 = render_views(sc0.model, sc0, resolution=args.resolution) if args.render else None
        if args.render:
            # save GT + hole renders (seed 0) for the side-by-side
            kept, hole = geometry.carve_hole(
                sc0.model._xyz.detach().cpu().numpy(), sc0.hole_lo, sc0.hole_hi)
            hole_m = _subset(sc0.model, kept)
        for baseline in ["A", "B", "C"]:
            for seed in range(args.seeds):
                completed, views, out = run_cell(scene_name, baseline, seed,
                                                 render=args.render,
                                                 resolution=args.resolution)
                rows.append(out)
                # save completed renders
                if views is not None:
                    from PIL import Image
                    from completion import render as re
                    d = os.path.join(args.out, scene_name)
                    os.makedirs(d, exist_ok=True)
                    imgs = re.render_set(views, completed)[0]
                    for v in range(len(views)):
                        im = (imgs[v].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
                        Image.fromarray(im).save(
                            os.path.join(d, "{}_s{}_v{}.png".format(baseline, seed, v)))
                print("[sweep] {}/{} seed={}  PSNR={:.1f} leak={:.2f} norm={:.1f}".format(
                    scene_name, baseline, seed, out["psnr"], out["leakage"],
                    out["normal_error_deg"]))

        # side-by-side image: GT | Hole | A | B | C (seed 0)
        if args.render:
            from PIL import Image
            from completion import render as re
            ss_dir = os.path.join(args.out, "side_by_side")
            os.makedirs(ss_dir, exist_ok=True)
            gt_imgs = re.render_set(views0, sc0.model)[0]
            hole_imgs = re.render_set(views0, hole_m)[0]
            c_imgs = {}
            for b in ["A", "B", "C"]:
                cb, _, _ = run_cell(scene_name, b, 0, render=True, resolution=args.resolution)
                c_imgs[b] = re.render_set(views0, cb)[0]
            v = 0
            panels = []
            for name, arr in [("GT", gt_imgs), ("Hole", hole_imgs)] + \
                             [("A", c_imgs["A"]), ("B", c_imgs["B"]), ("C", c_imgs["C"])]:
                im = (arr[v].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
                panels.append(Image.fromarray(im))
            w = sum(p.width for p in panels)
            h = panels[0].height
            canvas = Image.new("RGB", (w, h), (255, 255, 255))
            x = 0
            for p in panels:
                canvas.paste(p, (x, 0)); x += p.width
            canvas.save(os.path.join(ss_dir, "{}.png".format(scene_name)))
            print("[sweep] side-by-side saved to {}".format(os.path.join(ss_dir, scene_name + ".png")))

    with open(os.path.join(args.out, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in COLUMNS})
    print("[sweep] CSV written to {}/sweep.csv".format(args.out))


if __name__ == "__main__":
    main()
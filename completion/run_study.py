"""Controlled ablation + robustness study for the graph-variant completion pipeline.

Four graph variants C0..C3 share the IDENTICAL spawning, MLS/local-surface fitting,
optimization and rendering pipeline; only the graph edge information differs
(position / normal / appearance / semantic).  This isolates whether each graph signal
independently improves Gaussian completion.

Outputs (under --out):
  ablation_summary.csv     -- C0-C3 x 4 scenes x S seeds
  robustness_summary.csv   -- 6 robustness axes x variants x S seeds
  plots/*.png              -- per-axis error curves
  renders/*.png            -- representative qualitative renders (ablated variants)
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion.synthetic_scene import get_scene
from completion import geometry
from completion.gaussian_model import GaussianModel

VARIANTS = ["C0", "C1", "C2", "C3"]

METRICS = ["chamfer", "normal_error_deg", "leakage", "appearance_rmse", "seam_error",
           "psnr", "ssim", "edge_err", "corner_angle_err", "generated", "gaussians",
           "runtime_s"]


def _subset(m, mask):
    import torch.nn as nn
    n = GaussianModel(m.max_sh_degree)
    for name in ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
                 "_rotation", "_objects_dc"]:
        setattr(n, name, nn.Parameter(getattr(m, name).detach().cpu()[mask]))
    return n


def run_cell(scene_name, variant, seed, render=False, resolution=120, **kw):
    """Run one (scene, variant, seed, kw) cell; return (row dict, completed, views).

    kw may contain scene params (hole_scale/gap_mult/corner_angle/radius/hole_frac) and
    completion params (semantic_noise, normal_noise) which are split appropriately.
    """
    from completion import metrics
    from completion.gaussian_model import make_orbit_poses, make_cameras_from_poses
    scene_kw = dict(kw)
    comp_kw = {}
    for k in ("semantic_noise", "normal_noise"):
        if k in scene_kw:
            comp_kw[k] = scene_kw.pop(k)
    # hole_frac (fraction of local extent) -> hole_scale
    if "hole_frac" in scene_kw:
        # base plane hole is ~0.48 wide of a ~1.6/1.4 extent; express as multiple of the
        # default hole (hole_scale x default).  frac 0.20 of in-plane extent ~ default.
        base_inplane = 0.24  # default hole half-width x
        frac = scene_kw.pop("hole_frac")
        scene_kw["hole_scale"] = frac / 0.30 if frac > 0 else 0.3

    sc = get_scene(scene_name, seed=seed, **scene_kw)
    model = sc.model
    t0 = time.time()
    r = geometry.run_completion(model, sc, baseline=variant, seed=seed, **comp_kw)
    kept, hole = r.kept_mask, ~r.kept_mask
    hole_m = _subset(model, kept)
    removed = _subset(model, hole)
    completed = geometry.append_gaussians(hole_m, r)

    views = None
    if render:
        H = resolution
        W = int(H * 4 / 3)
        poses = make_orbit_poses(n=3, elevation_deg=50.0, radius=1.2,
                                 center=tuple(sc.center), surface_axis=2)
        views = make_cameras_from_poses(poses, height=H, width=W, fov_deg=50.0)

    out = metrics.report_metrics(model, hole_m, completed, removed, sc, r, views=views,
                                 resolution=resolution)
    row = {"scene": scene_name, "variant": variant, "seed": seed, **scene_kw}
    row.update({k: (float(out[k]) if out.get(k) is not None else "")
                for k in METRICS})
    row["runtime_s"] = time.time() - t0
    return row, completed, views


def write_csv(path, rows, extra_cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=extra_cols + METRICS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in
                        (extra_cols + METRICS)})
    print("[study] wrote {}".format(path))


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def run_ablation(out_dir, seeds):
    rows = []
    for sn in ["plane_checker", "l_corner", "parallel_surfaces", "curved_surface"]:
        for v in VARIANTS:
            for s in range(seeds):
                row, _, _ = run_cell(sn, v, s)
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def run_robustness(out_dir, seeds):
    rows = []

    # 1. semantic label noise (l_corner + parallel, C3)
    for frac in [0.0, 0.05, 0.10, 0.20, 0.30]:
        for v in ["C3", "C2", "C1"]:
            for s in range(seeds):
                row, _, _ = run_cell("parallel_surfaces", v, s, semantic_noise=frac)
                row["axis"] = "semantic_noise"; row["param"] = frac
                rows.append(row)

    # 2. normal angular noise
    for deg in [0, 5, 10, 20, 30]:
        for v in ["C3", "C2", "C1"]:
            for s in range(seeds):
                row, _, _ = run_cell("l_corner", v, s, normal_noise=deg)
                row["axis"] = "normal_noise"; row["param"] = deg
                rows.append(row)

    # 3. hole size (% of local surface extent) on plane_checker
    for frac in [0.05, 0.10, 0.20, 0.30, 0.40]:
        for v in VARIANTS:
            for s in range(seeds):
                row, _, _ = run_cell("plane_checker", v, s, hole_frac=frac)
                row["axis"] = "hole_size"; row["param"] = frac
                rows.append(row)

    # 4. parallel-surface separation (x median Gaussian spacing)
    for g in [0.5, 1.0, 2.0, 4.0, 8.0]:
        for v in ["C0", "C3"]:
            for s in range(seeds):
                row, _, _ = run_cell("parallel_surfaces", v, s, gap_mult=g)
                row["axis"] = "separation"; row["param"] = g
                rows.append(row)

    # 5. L-corner angle
    for a in [30, 45, 60, 90, 120]:
        for v in ["C0", "C1", "C3"]:
            for s in range(seeds):
                row, _, _ = run_cell("l_corner", v, s, corner_angle=a)
                row["axis"] = "corner_angle"; row["param"] = a
                rows.append(row)

    # 6. curved-surface radius (strong -> weak curvature)
    for r in [0.3, 0.5, 0.8, 1.2, 1.8]:
        for v in ["C0", "C3"]:
            for s in range(seeds):
                row, _, _ = run_cell("curved_surface", v, s, radius=r)
                row["axis"] = "radius"; row["param"] = r
                rows.append(row)

    return rows


def make_plots(out_dir, robustness_rows, ablation_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not robustness_rows:
        return
    pdir = os.path.join(out_dir, "plots")
    os.makedirs(pdir, exist_ok=True)

    from collections import defaultdict
    agg = defaultdict(list)
    for r in robustness_rows:
        key = (r["axis"], r["variant"], float(r["param"]))
        agg[key].append(r)

    def mean(key, list_of_rows):
        vals = [float(x.get(key)) for x in list_of_rows
                if x.get(key) not in (None, "", "nan")]
        return float(np.mean(vals)) if vals else float("nan")

    spec = [
        # (axis, xlabel, y-key, ylabel, filename)
        ("semantic_noise", "semantic label noise", "leakage", "surface leakage",
         "error_vs_semantic_noise.png"),
        ("normal_noise", "normal angular noise (deg)", "leakage", "surface leakage",
         "error_vs_normal_noise.png"),
        ("hole_size", "hole size (frac of extent)", "chamfer", "Chamfer distance",
         "error_vs_hole_size.png"),
        ("separation", "surface separation (x spacing)", "leakage", "surface leakage",
         "leakage_vs_separation.png"),
        ("corner_angle", "corner angle (deg)", "corner_angle_err", "corner angle err (deg)",
         "corner_error_vs_angle.png"),
        ("radius", "cylinder radius", "normal_error_deg", "normal error (deg)",
         "normal_error_vs_curvature.png"),
    ]
    for axis, xlab, ykey, ylab, fname in spec:
        plt.figure()
        plotted = False
        for v in VARIANTS:
            xs = sorted([p for (a, vv, p) in agg if a == axis and vv == v])
            ys = [mean(ykey, agg[(axis, v, p)]) for p in xs]
            if ys and len(xs) > 0:
                plt.plot(xs, ys, marker="o", label=v)
                plotted = True
        if not plotted:
            plt.close(); continue
        plt.xlabel(xlab); plt.ylabel(ylab); plt.title("{} vs {}".format(ylab, xlab))
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(pdir, fname), dpi=120)
        plt.close()
    print("[study] plots written to {}".format(pdir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/study")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ablation", action="store_true", default=True)
    ap.add_argument("--no-ablation", dest="ablation", action="store_false")
    ap.add_argument("--robustness", action="store_true", default=True)
    ap.add_argument("--render", action="store_true", default=False)
    ap.add_argument("--resolution", type=int, default=120)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base_cols = ["scene", "variant", "seed"]

    if args.ablation:
        rows = run_ablation(args.out, args.seeds)
        write_csv(os.path.join(args.out, "ablation_summary.csv"), rows, base_cols)

    if args.robustness:
        rows = run_robustness(args.out, args.seeds)
        extra = ["axis", "scene", "variant", "param", "seed"]
        write_csv(os.path.join(args.out, "robustness_summary.csv"), rows, extra)
        # plots need robustness rows (re-load to avoid memory of this run)
        make_plots_loader(args.out)

    # representative qualitative renders
    if args.render:
        render_representative(args.out, args.resolution)


def make_plots_loader(out_dir):
    import csv
    rows = list(csv.DictReader(open(os.path.join(out_dir, "robustness_summary.csv"))))
    make_plots(out_dir, rows, [])


def render_representative(out_dir, resolution):
    from PIL import Image
    from completion import render as re
    from completion.gaussian_model import make_orbit_poses, make_cameras_from_poses
    rdir = os.path.join(out_dir, "renders")
    os.makedirs(rdir, exist_ok=True)
    for sn in ["l_corner", "parallel_surfaces", "curved_surface"]:
        _, comp_completed, views = None, {}, None
        # build completed for each variant from seed 0
        models = {}
        views = None
        for v in VARIANTS:
            row, completed, views = run_cell(sn, v, 0, render=True, resolution=resolution)
            models[v] = completed
        # hole model = original minus hole
        sc = get_scene(sn, seed=0)
        kept = geometry.carve_hole(sc.model._xyz.detach().cpu().numpy(),
                                   sc.hole_lo, sc.hole_hi)[0]
        hole_m = _subset(sc.model, kept)
        panels = ("GT", sc.model), ("Hole", hole_m)
        for v, m in models.items():
            panels = panels + ((v, m),)
        imgs = []
        for name, m in panels:
            img = re.render_set(views, m)[0][0]
            imgs.append((name, (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")))
        w = sum(Image.fromarray(i[1]).width for i in imgs)
        h = imgs[0][1].shape[0]
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        x = 0
        for _, arr in imgs:
            im = Image.fromarray(arr)
            canvas.paste(im, (x, 0)); x += im.width
        canvas.save(os.path.join(rdir, "{}.png".format(sn)))
    print("[study] representative renders in {}".format(rdir))


if __name__ == "__main__":
    main()
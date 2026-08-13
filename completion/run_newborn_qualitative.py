"""3D qualitative views for the newborn pruning failure analysis.

For the helped/hurt ROIs (from failure_helped/hurt_rois.csv) renders:
  GT (removed) | Hole survivors | original newborns | retained | rejected
colored by the observable descriptor responsible for pruning.

SINGLE-SCENE MULTI-ROI DIAGNOSTIC VALIDATION.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion import geometry
from completion.gaussian_model import GaussianModel
from completion.run_global_affinity import load_25_rois, AFFINITIES, read_csv
from completion.run_real_controlled import subset_model
from completion.run_newborn_analysis import FOCUS, apply_rule, rules


def keep_mask_for_cell(newborn_rows, rule_name):
    fn = next(f for n, f in rules() if n == rule_name)
    keep = apply_rule(fn, newborn_rows).astype(bool)
    return keep


def render_roi(root, model, xyz, rois, newborn, pruned_rows, roi_name, method,
               policy, rule_name, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from types import SimpleNamespace

    qdir = os.path.join(root, "qualitative3d")
    os.makedirs(qdir, exist_ok=True)
    roi = next(r for r in rois if r["roi"] == roi_name)
    center, radius = roi["center"], roi["radius"]
    mask = np.linalg.norm(xyz - center, axis=1) <= radius
    scene = SimpleNamespace(name=roi_name, model=model, hole_lo=center - radius,
                            hole_hi=center + radius, center=center,
                            roi_center=center, roi_radius=radius)
    result = geometry.run_completion(model, scene, baseline=method, seed=0,
                                     normal_affinity=policy, semantic_gate="hard",
                                     hole_mask_override=mask, spawn_rule="count_matched")
    removed = subset_model(model, mask)
    gt = removed.get_xyz.detach().cpu().numpy()
    kept = xyz[~mask]
    newborn_xyz = result.new_xyz
    cell_rows = [r for r in newborn if r["roi"] == roi_name and r["method"] == method
                 and r["policy"] == policy]
    keep = keep_mask_for_cell(cell_rows, rule_name)
    retained = newborn_xyz[keep]
    rejected = newborn_xyz[~keep]

    # descriptor used for coloring rejected (primary for the rule's family)
    for d in ("mls_residual", "dist_from_boundary", "n_survivors_2x",
              "normal_agreement", "semantic_agreement"):
        if all(d in r for r in cell_rows):
            color_desc = d
            break
    reject_colors = [float(r[color_desc]) for r, k in zip(cell_rows, keep) if not k]

    fig = plt.figure(figsize=(18, 4.2))
    def panel(ax, pts, color, title, cmap=None, vals=None):
        if len(pts) == 0:
            ax.set_title(title + " (empty)"); return
        if cmap is not None and vals is not None:
            sc = ax.scatter(*pts.T, c=vals, cmap=cmap, s=8)
            ax.figure.colorbar(sc, ax=ax, shrink=0.7)
        else:
            ax.scatter(*pts.T, c=color, s=8, alpha=.7)
        ax.set_title("{} (n={})".format(title, len(pts)), fontsize=9)
        cen = pts.mean(0); ext = max(np.ptp(pts, axis=0).max() / 2, 1e-4)
        ax.set_xlim(cen[0]-ext, cen[0]+ext); ax.set_ylim(cen[1]-ext, cen[1]+ext)
        ax.set_zlim(cen[2]-ext, cen[2]+ext)

    panel(fig.add_subplot(1, 5, 1, projection="3d"), gt, "red", "GT removed")
    panel(fig.add_subplot(1, 5, 2, projection="3d"), kept, "gray", "Hole survivors")
    panel(fig.add_subplot(1, 5, 3, projection="3d"), newborn_xyz, "blue", "orig newborns")
    panel(fig.add_subplot(1, 5, 4, projection="3d"), retained, "green", "retained")
    panel(fig.add_subplot(1, 5, 5, projection="3d"), rejected, "orange", "rejected",
          cmap="viridis", vals=reject_colors)
    fig.suptitle("{} {} {} — pruned by {} (descriptor {})".format(
        tag, method, policy, rule_name, color_desc))
    fig.tight_layout()
    fig.savefig(os.path.join(qdir, "{}_{}_{}_{}.png".format(tag, method, policy, roi_name)),
                dpi=150)
    plt.close(fig)
    print("[qual3d] {}_{}_{}_{} -> retained {} / rejected {}".format(
        tag, method, policy, roi_name, len(retained), len(rejected)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", default="outputs/newborn_support_diagnostic")
    args = ap.parse_args()
    model = GaussianModel(3); model.load_ply(args.checkpoint)
    xyz = model.get_xyz.detach().cpu().numpy()
    rois = load_25_rois("outputs/multiscene_generalization/roi_descriptors.csv")
    newborn = read_csv(os.path.join(args.root, "newborn_descriptors.csv"))
    pruned = read_csv(os.path.join(args.root, "completion_level_pruned.csv"))
    # pruned rows carry roi/method/policy/pruned_by
    targets = []
    for f in ("failure_helped_rois.csv", "failure_hurt_rois.csv"):
        for r in read_csv(os.path.join(args.root, f)):
            targets.append((r["roi"], r["method"], r["policy"],
                            "helped" if "helped" in f else "hurt"))
    # fetch the exact pruned_by rule for each cell
    for t in targets:
        m = [r for r in pruned if r["roi"] == t[0] and r["method"] == t[1] and
             r["policy"] == t[2]]
        rule = m[0].get("pruned_by", "mls_residual<=0.0500") if m else "mls_residual<=0.0500"
        render_roi(args.root, model, xyz, rois, newborn, pruned, t[0], t[1], t[2],
                   rule, t[3])
    print("[qual3d] done")


if __name__ == "__main__":
    main()
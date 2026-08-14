"""Kitchen-like qualitative 3D views for the layer-recoverability audit.

For the high/low ambiguity ROIs (qualitative_rois.csv), renders
  GT(removed) | survivors | C0 newborns | C1-HARD newborns
colored by surface layer (depth bracket or normal cluster).

No CUDA: point-cloud scatter only.  Completion is the frozen C0 / C1-HARD
count-matched path; labeled geometric.

MULTI-SCENE REAL 3DGS SURFACE-LAYER RECOVERABILITY DIAGNOSTIC.
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion import geometry
from completion.gaussian_model import GaussianModel
from completion.run_global_affinity import load_25_rois, read_csv
from completion.run_real_controlled import subset_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(ROOT, "checkpoints_download/ramen/point_cloud/"
                                "iteration_30000/point_cloud.ply")


def run_cell(model, xyz, roi, method, affinity="hard"):
    center, radius = roi["center"], roi["radius"]
    mask = np.linalg.norm(xyz - center, axis=1) <= radius
    scene = SimpleNamespace(name=roi["roi"], model=model, hole_lo=center - radius,
                            hole_hi=center + radius, center=center,
                            roi_center=center, roi_radius=radius)
    result = geometry.run_completion(model, scene, baseline=method, seed=0,
                                     normal_affinity=affinity, semantic_gate="hard",
                                     hole_mask_override=mask, spawn_rule="count_matched")
    removed = subset_model(model, mask)
    return result.new_xyz, removed.get_xyz.detach().cpu().numpy(), xyz[~mask]


def color_by_depth(pts, n=3):
    """Color points by depth bracket along their dominant axis (visual only)."""
    from scipy.spatial import cKDTree
    if len(pts) < 4:
        return np.full(len(pts), 0)
    # depth axis = axis of smallest spatial extent of the local cloud
    axis = int(np.argmin(np.ptp(pts, axis=0)))
    z = pts[:, axis]
    edges = np.percentile(z, np.linspace(0, 100, n + 1))
    idx = np.digitize(z, edges[1:-1])
    return idx


def render_roi(outdir, model, xyz, roi, gt0, low, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    center, radius = roi["center"], roi["radius"]
    mask = np.linalg.norm(xyz - center, axis=1) <= radius
    gt = xyz[mask]
    survivors = xyz[~mask]

    c0_new, _, _ = run_cell(model, xyz, roi, "C0", "hard")
    c1_new, _, _ = run_cell(model, xyz, roi, "C1", "hard")

    fig = plt.figure(figsize=(20, 4.6))
    def panel(ax, pts, title, color=None, cmap_vals=None, cmap="tab20"):
        if len(pts) == 0:
            ax.set_title(title + " (empty)"); return
        if cmap_vals is None:
            ax.scatter(*pts.T, color=color, s=7, alpha=.7)
        else:
            ax.scatter(*pts.T, c=cmap_vals, cmap=cmap, s=7)
        ax.set_title("{} (n={})".format(title, len(pts)), fontsize=9)
        cen = pts.mean(0); ext = max(np.ptp(pts, axis=0).max() / 2, 1e-3)
        ax.set_xlim(cen[0]-ext, cen[0]+ext); ax.set_ylim(cen[1]-ext, cen[1]+ext)
        ax.set_zlim(cen[2]-ext, cen[2]+ext)
    panel(fig.add_subplot(1, 5, 1, projection="3d"), gt, "GT removed", color="red")
    panel(fig.add_subplot(1, 5, 2, projection="3d"), survivors, "survivors", color="gray")
    panel(fig.add_subplot(1, 5, 3, projection="3d"), c0_new, "C0 newborns",
          cmap_vals=color_by_depth(c0_new, max(2, int(np.ceil(len(c0_new)/20)))))
    panel(fig.add_subplot(1, 5, 4, projection="3d"), c1_new, "C1-HARD newborns",
          cmap_vals=color_by_depth(c1_new, max(2, int(np.ceil(len(c1_new)/20)))))
    # combined with GT overlay in red
    ax5 = fig.add_subplot(1, 5, 5, projection="3d")
    ax5.scatter(*c1_new.T, c="blue", s=7, alpha=.6, label="C1")
    ax5.scatter(*gt.T, c="red", s=7, alpha=.6, label="GT")
    ax5.set_title("C1 (blue) vs GT (red)"); ax5.legend(fontsize=8)
    cen = np.concatenate([gt, c1_new]).mean(0)
    ext = max(np.ptp(np.concatenate([gt, c1_new]), axis=0).max() / 2, 1e-3)
    ax5.set_xlim(cen[0]-ext, cen[0]+ext); ax5.set_ylim(cen[1]-ext, cen[1]+ext)
    ax5.set_zlim(cen[2]-ext, cen[2]+ext)
    fig.suptitle("{} | {} | depth-colored newborns".format(tag, roi["roi"]))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "{}_{}.png".format(tag, roi["roi"])), dpi=150)
    plt.close(fig)
    print("[qual] {} {} rendered (C0 {}, C1 {})".format(tag, roi["roi"], len(c0_new), len(c1_new)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--root", default="outputs/layer_recoverability_audit")
    args = ap.parse_args()
    model = GaussianModel(3); model.load_ply(args.checkpoint)
    xyz = model.get_xyz.detach().cpu().numpy()
    rois = {r["roi"]: r for r in load_25_rois(
        os.path.join(ROOT, "outputs/multiscene_generalization/roi_descriptors.csv"))}
    q = read_csv(os.path.join(args.root, "qualitative_rois.csv"))
    qdir = os.path.join(args.root, "qualitative")
    os.makedirs(qdir, exist_ok=True)
    for r in q:
        if r["roi"] in rois:
            render_roi(qdir, model, xyz, rois[r["roi"]], None, None, r["group"])
    print("[qual] done -> {}".format(qdir))


if __name__ == "__main__":
    main()
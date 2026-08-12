"""Automatic ROI proposals for controlled-hole validation on a real GG checkpoint.

This is a deterministic scene-inspection utility, not a completion algorithm.  It uses
only the loaded scene itself (no held-out information) and emits one best candidate per
requested category plus overview point-cloud previews.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion import geometry
from completion.gaussian_model import GaussianModel


CATEGORY_MAP = {
    "A": ("planar surface", "roi_A_planar"),
    "B": ("sharp junction", "roi_B_junction"),
    "C": ("nearby layered / parallel surfaces", "roi_C_layered"),
    "D": ("curved / irregular surface", "roi_D_curved"),
}


def semantic_statistics(model):
    raw = model._objects_dc.detach().cpu().numpy().reshape(len(model.get_xyz), -1)
    exists = raw.shape[1] > 0 and np.isfinite(raw).all() and float(np.std(raw)) > 1e-8
    if not exists:
        return raw, None, {"exists": False, "feature_dimensions": int(raw.shape[1]),
                           "number_of_groups": None, "mean_confidence": None,
                           "mean_entropy": None}
    shifted = raw - raw.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / (np.exp(shifted).sum(axis=1, keepdims=True) + 1e-12)
    labels = np.argmax(raw, axis=1)
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    return raw, labels, {"exists": True, "feature_dimensions": int(raw.shape[1]),
                         "number_of_groups": int(len(np.unique(labels))),
                         "mean_confidence": float(probs.max(axis=1).mean()),
                         "mean_entropy": float(entropy.mean())}


def inspect_scene(model, checkpoint, run_label):
    xyz = model.get_xyz.detach().cpu().numpy()
    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, k=min(2, len(xyz)))
    spacing = float(np.median(distances[:, -1]))
    opacity = model.get_opacity.detach().cpu().numpy()
    scale = model.get_scaling.detach().cpu().numpy()
    _, _, semantic = semantic_statistics(model)
    return {
        "result_type": run_label,
        "checkpoint": os.path.abspath(checkpoint),
        "number_of_gaussians": int(len(xyz)),
        "xyz_bounding_box": {"min": xyz.min(0).tolist(), "max": xyz.max(0).tolist()},
        "median_nearest_neighbor_spacing": spacing,
        "sh_feature_dimensions": {
            "dc": list(model._features_dc.shape[1:]),
            "rest": list(model._features_rest.shape[1:]),
            "active_degree": int(model.active_sh_degree)},
        "opacity_statistics": {"min": float(opacity.min()), "median": float(np.median(opacity)),
                               "mean": float(opacity.mean()), "max": float(opacity.max())},
        "scale_statistics": {"min": float(scale.min()), "median": float(np.median(scale)),
                             "mean": float(scale.mean()), "max": float(scale.max())},
        "semantic_identity": semantic,
    }


def _candidate_features(xyz, normals, eigenvalues, labels, neighborhood):
    pts = xyz[neighborhood]
    ns = normals[neighborhood]
    mean_n = ns.mean(0); mean_n /= np.linalg.norm(mean_n) + 1e-8
    normal_angles = np.degrees(np.arccos(np.clip(np.abs(ns @ mean_n), 0, 1)))
    curvature = eigenvalues[neighborhood, 0] / (eigenvalues[neighborhood].sum(1) + 1e-12)
    semantic_ids = [] if labels is None else np.unique(labels[neighborhood]).astype(int).tolist()
    return pts, mean_n, normal_angles, curvature, semantic_ids


def discover_candidates(model, summary, max_anchors=512, neighborhood_k=96):
    xyz = model.get_xyz.detach().cpu().numpy()
    n = len(xyz)
    tree = cKDTree(xyz)
    k = min(neighborhood_k, n)
    # Estimate normals/eigenvalues from the full observed scene only.
    all_idx = np.arange(n)
    normals, diag = geometry.estimate_normals_local_pca_at(
        xyz, all_idx, k=min(16, n), return_diagnostics=True)
    _, labels, _ = semantic_statistics(model)
    anchors = np.linspace(0, n - 1, min(max_anchors, n), dtype=np.int64)
    scored = {key: [] for key in CATEGORY_MAP}
    for anchor in anchors:
        distances, neighborhood = tree.query(xyz[anchor], k=k)
        neighborhood = np.atleast_1d(neighborhood)
        distances = np.atleast_1d(distances)
        pts, mean_n, angles, curvature, semantic_ids = _candidate_features(
            xyz, normals, diag["eigenvalues"], labels, neighborhood)
        radius = max(float(np.percentile(distances, 55)), summary["median_nearest_neighbor_spacing"] * 4)
        inner = neighborhood[distances <= radius]
        if len(inner) < 12:
            continue
        sem_boundary = max(0, len(semantic_ids) - 1)
        planar = np.clip(1.0 - float(np.median(curvature)) / 0.02, 0, 1)
        consistency = np.clip(1.0 - float(np.percentile(angles, 80)) / 30.0, 0, 1)
        density = np.clip(len(inner) / 48.0, 0, 1)
        # Junction evidence: two normal populations with a useful separation.
        dots = np.clip(np.abs(normals[neighborhood] @ mean_n), 0, 1)
        spread = float(np.percentile(np.degrees(np.arccos(dots)), 90))
        junction = np.clip(1.0 - abs(spread - 45.0) / 45.0, 0, 1)
        # Layer evidence: nearby semantic diversity plus parallel local normals.
        layered = consistency * (1.0 if sem_boundary > 0 else 0.25) * density
        curved = np.clip(float(np.median(curvature)) / 0.03, 0, 1) * consistency * density
        scores = {
            "A": planar * consistency * density / (1.0 + sem_boundary),
            "B": junction * density,
            "C": layered,
            "D": curved,
        }
        covariance = np.cov((pts - pts.mean(0)).T)
        _, axes = np.linalg.eigh(covariance)
        axes = axes[:, ::-1].T
        local = (pts - xyz[anchor]) @ axes.T
        half_extent = np.maximum(np.percentile(np.abs(local), 55, axis=0),
                                 summary["median_nearest_neighbor_spacing"] * 2)
        base = {
            "center": xyz[anchor].tolist(), "radius": radius,
            "oriented_box_axes": axes.tolist(), "half_extent": half_extent.tolist(),
            "number_of_contained_gaussians": int(len(inner)),
            "estimated_curvature": float(np.median(curvature)),
            "mean_normal": mean_n.tolist(), "semantic_ids_present": semantic_ids,
            "local_gaussian_spacing": float(np.median(distances[1:min(len(distances), 16)])),
            "normal_angle_spread_deg": spread,
        }
        for key, score in scores.items():
            scored[key].append((float(score), base))

    candidates = []
    used_centers = []
    scene_diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0))) + 1e-8
    for key in ("A", "B", "C", "D"):
        choices = sorted(scored[key], key=lambda item: item[0], reverse=True)
        chosen = None
        for score, base in choices:
            center = np.asarray(base["center"])
            if all(np.linalg.norm(center - prev) > 0.02 * scene_diag for prev in used_centers):
                chosen = (score, base); break
        if chosen is None and choices:
            chosen = choices[0]
        if chosen is None:
            raise RuntimeError("scene is too small/sparse to discover ROI {}".format(key))
        score, base = chosen
        used_centers.append(np.asarray(base["center"]))
        category, name = CATEGORY_MAP[key]
        candidate = dict(base)
        candidate.update({"name": name, "category": category,
                          "confidence_score": score,
                          "selector": {"type": "sphere", "center": base["center"],
                                       "radius": base["radius"]}})
        candidates.append(candidate)
    return candidates


def save_previews(xyz, candidates, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    preview_dir = os.path.join(out_dir, "roi_previews")
    os.makedirs(preview_dir, exist_ok=True)
    stride = max(1, len(xyz) // 20000)
    for candidate in candidates:
        center = np.asarray(candidate["center"]); radius = candidate["radius"]
        selected = np.linalg.norm(xyz - center, axis=1) <= radius
        fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
        ax.scatter(*xyz[::stride].T, s=0.3, c="lightgray", alpha=0.35)
        ax.scatter(*xyz[selected].T, s=3, c="red", alpha=0.8)
        ax.set_title("{} | confidence {:.3f}".format(candidate["category"],
                                                     candidate["confidence_score"]))
        fig.tight_layout(); fig.savefig(os.path.join(preview_dir, candidate["name"] + ".png"), dpi=140)
        plt.close(fig)


def run(checkpoint, out_dir, sh_degree=3, run_label="REAL SCENE RESULT"):
    os.makedirs(out_dir, exist_ok=True)
    model = GaussianModel(sh_degree); model.load_ply(checkpoint)
    summary = inspect_scene(model, checkpoint, run_label)
    with open(os.path.join(out_dir, "real_scene_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    candidates = discover_candidates(model, summary)
    with open(os.path.join(out_dir, "roi_candidates.json"), "w") as f:
        json.dump({"result_type": run_label, "candidates": candidates}, f, indent=2)
    save_previews(model.get_xyz.detach().cpu().numpy(), candidates, out_dir)
    return summary, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run(args.checkpoint, args.out, args.sh_degree,
        "SMOKE TEST" if args.smoke_test else "REAL SCENE RESULT")


if __name__ == "__main__":
    main()

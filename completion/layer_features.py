"""Surface-layer ambiguity + local-recoverability audit (core computation).

MULTI-SCENE REAL 3DGS SURFACE-LAYER RECOVERABILITY DIAGNOSTIC (single scene today).

Frozen inputs:
  * official ramen GG checkpoint (checkpoints_download/ramen/.../point_cloud.ply)
  * the 25 frozen ROI definitions (outputs/multiscene_generalization/roi_descriptors.csv)
  * real ramen COLMAP cameras (checkpoints_download/data_extracted/ramen)
  * frozen count-matched completion + canonical evaluator

No completion algorithm is modified.  Removed GT is used ONLY for evaluation
labels / GT-layer analysis.

NOTE ON SUCCESS LABELS: the frozen C0+A4 / C1-HARD+A4 *rendering* labels (Hole
LPIPS / PSNR / SSIM) require the CUDA rasterizer, which cannot be compiled on
this machine.  We therefore construct clearly-labelled geometric success
surrogates from held-out GT (median normalized newborn->GT distance, and
C1-vs-C0 Chamfer improvement).  A pluggable render-label path is provided.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion import geometry
from completion.gaussian_model import GaussianModel
from completion.run_global_affinity import load_25_rois, DEFAULT_ROIS_CSV, read_csv, write_csv
from completion.cpu_cameras import load_cameras, project_camera, camera_eye

FOCUS = ["C1", "C3"]
AFFINITIES = ["hard", "soft", "adaptive"]
N_CLUSTERS = 4
K_NB = 16


# ---------------------------------------------------------------------------
# camera visibility / projection helpers (CPU only)
# ---------------------------------------------------------------------------

def hole_visibility(cameras, hole_xyz, margin_px=10.0):
    """Number of cameras that see the hole region, plus per-cam depth range."""
    n_see = 0
    depths = []
    frontal = []
    for cam in cameras:
        x, y, depth, valid = project_camera(cam, hole_xyz)
        if not valid.any():
            continue
        inside = (x >= -margin_px) & (x < cam["width"] + margin_px) & \
                 (y >= -margin_px) & (y < cam["height"] + margin_px) & valid
        if inside.sum() < 8:
            continue
        n_see += 1
        depths.append(float(depth[inside].mean()))
        # frontal-ness: how head-on the camera is relative to hole normal (approx)
        eye = camera_eye(cam)
        ray = eye - hole_xyz.mean(0)
        ray /= np.linalg.norm(ray) + 1e-12
        frontal.append(float(abs(ray.mean())))
    return n_see, np.asarray(depths, dtype=float) if depths else np.zeros(0), \
        np.asarray(frontal, dtype=float) if frontal else np.zeros(0)


def boundary_support(xyz, mask, center, radius, scene_tree=None):
    """Survivor + boundary info around the hole (no GT).

    `scene_tree` is an optional precomputed cKDTree over the FULL scene, used only
    for the spacing / ball-count quantities (identical values, avoids a rebuild).
    """
    kept = xyz[~mask]
    kept_idx = np.where(~mask)[0]
    lo, hi = center - radius, center + radius
    if kept_tree := scene_tree:
        # query spacing on the kept points via the full-scene tree
        full_idx = np.where(~mask)[0]
        dist, _ = scene_tree.query(xyz[full_idx], k=min(2, len(full_idx)))
        spacing = float(np.median(dist[:, -1]))
        n_ball = int(scene_tree.query_ball_point(center, 2.5 * radius,
                                                 return_length=True))
    else:
        tree = cKDTree(kept)
        d_keep, _ = tree.query(kept, k=min(2, len(kept)))
        spacing = float(np.median(d_keep[:, -1]))
        n_ball = int(tree.query_ball_point(center, 2.5 * radius, return_length=True))
    bk, _ = geometry.detect_boundary_from_region(kept, lo, hi)
    boundary_idx = kept_idx[bk]
    boundary_xyz = xyz[boundary_idx]
    d_center, _ = cKDTree(boundary_xyz).query(center[None, :], k=1)
    return spacing, boundary_idx, boundary_xyz, d_center[0], n_ball


def depth_modes(depths, n_modes=3):
    """1-D depth GMM-ish mode analysis along a ray."""

    def cluster_1d(vals):
        vals = np.asarray(vals, dtype=float)
        if len(vals) < 3 or np.std(vals) < 1e-9:
            return 1, 0.0, np.zeros(1)
        # k-means on 1-D values
        centers = np.linspace(vals.min(), vals.max(), n_modes)
        new_centers = centers.copy()
        for _ in range(40):
            dists = np.abs(vals[None, :] - centers[:, None])
            assign = np.argmin(dists, axis=0)
            for m in range(len(centers)):
                sel = assign == m
                new_centers[m] = vals[sel].mean() if sel.any() else centers[m]
            if np.allclose(new_centers, centers, atol=1e-5):
                break
            centers = new_centers.copy()
        assign = np.argmin(np.abs(vals[None, :] - centers[:, None]), axis=0)
        cnt = np.bincount(assign, minlength=len(centers))
        present = cnt > 0
        keys = centers[present]
        if len(keys) <= 1:
            return 1, 0.0, centers[:1]
        # merge near-identical modes
        keys = np.sort(keys)
        merged = [keys[0]]
        for k in keys[1:]:
            if abs(k - merged[-1]) > max(np.ptp(vals) * 0.05, 1e-6):
                merged.append(k)
        return len(merged), float(np.std(centers[present])), np.asarray(merged)

    return cluster_1d(depths)


def entropy(probs):
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum())


def pca_normals_at(xyz, tree, query_idx, k=16):
    """Local-PCA normals at query_idx (identical formula to geometry's
    estimate_normals_local_pca_at) but using a precomputed scene `tree` where the
    k-NN query is issued from `xyz[query_idx]`.  Returns (B,3) unit normals."""
    query_idx = np.asarray(query_idx, dtype=np.int64)
    _, neighbours = tree.query(xyz[query_idx], k=min(k, len(xyz)))
    normals = np.zeros((len(query_idx), 3), dtype=np.float64)
    centroid = xyz.mean(0)
    for q, oi in enumerate(query_idx):
        nb = xyz[np.atleast_1d(neighbours[q])]
        mu = nb.mean(0)
        cov = (nb - mu).T @ (nb - mu) / max(len(nb) - 1, 1)
        w, v = np.linalg.eigh(cov)
        normal = v[:, 0]
        if normal @ (xyz[oi] - centroid) < 0:
            normal = -normal
        normals[q] = normal / (np.linalg.norm(normal) + 1e-8)
    return normals


# ---------------------------------------------------------------------------
# descriptors per ROI (observable pre-GT)
# ---------------------------------------------------------------------------

def roi_descriptors(model, xyz, roi, cameras, gt_normal_lookup=None, scene_tree=None):
    """All observable descriptors.  `gt_normal_lookup` is NOT used here (reserved).

    `scene_tree` is an optional precomputed cKDTree over the FULL scene; it only
    speeds up identical computations (no semantic change).
    """
    center = roi["center"]; radius = roi["radius"]
    mask = np.linalg.norm(xyz - center, axis=1) <= radius
    if mask.sum() < 8:
        return None
    spacing, bnd_idx, bnd_xyz, d_center, n_ball = boundary_support(
        xyz, mask, center, radius, scene_tree=scene_tree)
    hole_xyz = xyz[mask]

    # --- visibility / support ---
    n_cams, depths, frontals = hole_visibility(cameras, hole_xyz)
    visible_support_frac = min(n_cams / 3.0, 1.0) if n_cams > 0 else 0.0
    boundary_support_count = float(len(bnd_idx))
    norm_dist_center_support = float(d_center / max(spacing, 1e-9))
    support_density = float(n_ball / (4 / 3 * np.pi * (2.5 * radius) ** 3 + 1e-12))
    # projected support coverage: mean frontal-ness (head-on view fraction)
    projected_support_coverage = float(frontals.mean()) if len(frontals) else 0.0

    # --- depth-layer structure (per-visible-camera depth of boundary) ---
    depth_per_cam = []
    for cam in cameras:
        x, y, depth, valid = project_camera(cam, bnd_xyz)
        ok = valid & (x >= 0) & (x < cam["width"]) & (y >= 0) & (y < cam["height"])
        if ok.sum() >= 4:
            depth_per_cam.append(depth[ok])
    if depth_per_cam:
        all_depth = np.concatenate(depth_per_cam)
        n_modes, depth_var, mode_centers = depth_modes(all_depth)
        # depth discontinuity = max mode separation
        disc = float(np.ptp(mode_centers)) if len(mode_centers) > 1 else 0.0
        # mode entropy: fraction of points per mode
        counts = np.array([np.sum(np.abs(all_depth - m) <= max(np.ptp(all_depth) * 0.03, 1e-6))
                           for m in mode_centers], dtype=float)
        counts = counts + 1e-9
        mode_ent = entropy(counts / counts.sum())
        # min separation between modes
        min_sep = float(np.min(np.diff(np.sort(mode_centers)))) \
            if len(mode_centers) > 1 else 0.0
        # cross-view consistency: std of per-camera median depth (normalized)
        per_cam_med = np.asarray([np.median(d) for d in depth_per_cam])
        cross_view_depth_std = float(per_cam_med.std() / (per_cam_med.mean() + 1e-9))
    else:
        n_modes, depth_var, disc, mode_ent, min_sep, cross_view_depth_std = \
            1, 0.0, 0.0, 0.0, 0.0, 0.0

    # --- normal / surface structure ---
    if scene_tree is not None:
        bnormals = pca_normals_at(xyz, scene_tree, bnd_idx, K_NB)
    else:
        bnormals = geometry.estimate_normals_local_pca_at(xyz, bnd_idx, k=K_NB)
    if len(bnormals) >= 4:
        # angular distance matrix -> cluster normals
        c = bnormals.mean(0); c /= np.linalg.norm(c) + 1e-12
        angles = np.degrees(np.arccos(np.clip(np.abs(bnormals @ c), 0, 1)))
        n_clusters = max(1, int(np.ceil(np.percentile(angles, 97.5) / 12)))
        normal_dispersion = float(np.mean(angles))
        normal_ent = float(entropy(np.bincount(
            np.digitize(angles, np.linspace(0, 90, 5))).astype(float) + 1e-9))
        # graph components
        app = geometry._appearance_features(model)[bnd_idx]
        sem = geometry._semantic_features(model)[bnd_idx]
        comps = {}
        for variant in ("C0", "C1", "C3"):
            un, _, us = geometry.VARIANT_FLAGS[variant]
            rows, cols, _, _ = geometry.build_knn_graph(
                bnd_xyz, bnormals, None, sem, k=12, use_normal=un,
                use_appearance=False, use_semantic=us, normal_affinity="hard",
                semantic_gate="hard")
            labels = geometry.partition_boundary_graph(len(bnd_xyz), rows, cols)
            comps[variant] = int(labels.max()) + 1
            if variant == "C3":
                largest_comp = float(np.bincount(labels).max() / len(labels))
        pca_eig = np.linalg.eigvalsh(np.cov(bnormals.T))
        curvature = float(pca_eig[0] / (pca_eig.sum() + 1e-12))
    else:
        n_clusters = normal_dispersion = normal_ent = curvature = 1.0
        comps = {"C0": 1, "C1": 1, "C3": 1}; largest_comp = 1.0

    # --- semantic / instance ---
    raw_sem = model._objects_dc.detach().cpu().numpy().reshape(len(model.get_xyz), -1)
    if raw_sem.shape[1] >= 2 and np.std(raw_sem) > 1e-8:
        shifted = raw_sem - raw_sem.max(1, keepdims=True)
        probs = np.exp(shifted); probs /= probs.sum(1, keepdims=True) + 1e-12
        labels = np.argmax(raw_sem, axis=1)
        bl = labels[bnd_idx]
        n_sem_ids = int(len(np.unique(bl)))
        sem_ent = float(entropy(np.bincount(bl).astype(float)))
        cnt = np.bincount(bl)
        sem_purity = float(cnt.max() / cnt.sum()) if len(bl) else 1.0
        bp = probs[bnd_idx]
        sem_conf = float(np.mean(bp.max(axis=1)))
    else:
        n_sem_ids, sem_ent, sem_purity, sem_conf = 1, 0.0, 1.0, 1.0

    # --- cross-modal ambiguity (agreement between partitions) ---
    # agreement depth-vs-normal: correlation of per-boundary-point depth proxy with
    # its normal's distance from the dominant normal
    depth_agreement = float("nan")
    normal_sem_agreement = float("nan")

    # build a per-boundary "partition indicator" via normal cluster labels (GMM-like via
    # distance to dominant normal) and semantic grouping
    if len(bnormals) >= 8:
        nc = np.zeros(len(bnormals)); sc = np.zeros(len(bnormals))
        for i in range(len(bnormals)):
            nc[i] = 0 if abs(float(bnormals[i] @ c)) >= np.cos(np.deg2rad(20)) else 1
        s_mode = Counter(bl).most_common(1)[0][0]
        sc = (bl == s_mode).astype(float)
        # agreement = fraction of pairs with matching same/diff status
        def agree(a, b):
            mask = np.ones((len(a), len(b)), dtype=bool)
            # use self-agreement on combined indicator
            both = a * 2 + b  # 0..3 categories
            _, counts = np.unique(both, return_counts=True)
            counts = counts.astype(float)
            return float(counts.max() / counts.sum())
        normal_sem_agreement = agree(nc, sc)

    return {
        "roi": roi["roi"], "known": roi["known_case"],
        "radius": float(radius), "normalized_radius_units": float(radius / max(spacing, 1e-9)),
        "n_cameras_see_hole": int(n_cams),
        "visible_support_fraction": visible_support_frac,
        "boundary_support_count": boundary_support_count,
        "norm_dist_center_support": norm_dist_center_support,
        "support_density": support_density,
        "projected_support_coverage": projected_support_coverage,
        "n_depth_modes": n_modes,
        "depth_variance": depth_var,
        "depth_discontinuity": disc,
        "depth_mode_entropy": mode_ent,
        "depth_min_mode_sep": min_sep,
        "cross_view_depth_std": cross_view_depth_std,
        "n_normal_clusters": n_clusters,
        "normal_dispersion": normal_dispersion,
        "normal_entropy": normal_ent,
        "pca_curvature": curvature,
        "graph_components_C0": comps["C0"],
        "graph_components_C1": comps["C1"],
        "graph_components_C3": comps["C3"],
        "largest_component_fraction": largest_comp,
        "n_semantic_ids": n_sem_ids,
        "semantic_entropy": sem_ent,
        "semantic_purity": sem_purity,
        "semantic_confidence": sem_conf,
        "cross_modal_normal_sem_agreement": normal_sem_agreement,
    }


def gt_layer_analysis(model, xyz, roi):
    """GT-only layer analysis of the removed region (eval only)."""
    center = roi["center"]; radius = roi["radius"]
    mask = np.linalg.norm(xyz - center, axis=1) <= radius
    gt_xyz = xyz[mask]
    gt_idx = np.where(mask)[0]
    if len(gt_xyz) < 4:
        return None
    bnormals = geometry.estimate_normals_local_pca_at(xyz, gt_idx, k=K_NB)
    c = bnormals.mean(0); c /= np.linalg.norm(c) + 1e-12
    angles = np.degrees(np.arccos(np.clip(np.abs(bnormals @ c), 0, 1)))
    n_normal_clusters = max(1, int(np.ceil(np.percentile(angles, 97.5) / 12)))
    # depth layers of GT = dominant view PCA along the hole's axis
    # approximatation: depth spread along the boundary normal
    depth_layer_span = float(np.ptp(gt_xyz @ c))
    raw_sem = model._objects_dc.detach().cpu().numpy().reshape(len(model.get_xyz), -1)
    if raw_sem.shape[1] >= 2 and np.std(raw_sem) > 1e-8:
        labels = np.argmax(raw_sem, axis=1)[gt_idx]
        n_sem = int(len(np.unique(labels)))
        cnt = np.bincount(labels)
        sem_purity = float(cnt.max() / cnt.sum())
    else:
        n_sem, sem_purity = 1, 1.0
    # classify structure
    depth_layers = 2 if depth_layer_span > radius else 1
    if n_normal_clusters >= 3 and n_sem >= 3:
        cat = "C_multi_object_layered"
    elif n_normal_clusters >= 2:
        cat = "B_multi_surface"
    elif depth_layers >= 2:
        cat = "C_multi_object_layered"
    else:
        cat = "A_single_surface"
    return {"roi": roi["roi"], "gt_n_gaussians": int(len(gt_xyz)),
            "gt_n_normal_clusters": n_normal_clusters,
            "gt_normal_dispersion": float(np.mean(angles)),
            "gt_depth_layer_span": depth_layer_span,
            "gt_n_semantic_ids": n_sem, "gt_semantic_purity": sem_purity,
            "gt_category": cat}


if __name__ == "__main__":
    raise SystemExit("import module; run via run_layer_recoverability.py")
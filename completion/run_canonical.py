"""Canonical experiment runner for the reproducibility audit.

One function that both count-matched and generalization studies call.
Given checkpoint + ROI + method + config + seed, returns exactly one
deterministic result.

The canonical function reads per-ROI normal_affinity from the ROI's
validation JSON (or falls back to a configurable default), so the
reproducibility bug (roi_C_layered: soft in count-matched, hard in
generalization) is fixed by design.
"""

import json
import os
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from completion import geometry, metrics
from completion.gaussian_model import GaussianModel
from completion.run_real_controlled import subset_model
from completion.run_count_matched_ablation import geometric_metrics, deterministic_subset


# Affinity key: per-ROI normal_affinity from roi_validation.json
# If the JSON doesn't have a "normal_affinity" field, the default is used.
DEFAULT_AFFINITY = "hard"


def canonical_completion(checkpoint_path, roi_path, method, seed=0, **kwargs):
    """Run one deterministic completion cell.

    Parameters
    ----------
    checkpoint_path : str
        Path to a GG-format point_cloud.ply file.
    roi_path : str
        Path to a directory containing roi_validation.json.
    method : str
        One of C0, C1, C2, C3.
    seed : int
        Random seed (default 0).
    **kwargs : optional overrides for normal_affinity, semantic_gate, spawn_rule, etc.

    Returns
    -------
    dict
        A single row of results (metrics + metadata) ready for CSV output.
    """
    # Load the model
    model = GaussianModel(3)
    model.load_ply(checkpoint_path)
    xyz = model.get_xyz.detach().cpu().numpy()

    # Load the ROI definition
    roi_json = json.load(open(os.path.join(roi_path, "roi_validation.json")))
    center = np.asarray(roi_json["center"], dtype=np.float32)
    radius = float(roi_json["radius"])
    mask = np.linalg.norm(xyz - center, axis=1) <= radius
    removed = subset_model(model, mask)
    gt = removed.get_xyz.detach().cpu().numpy()

    # Determine normal_affinity: per-ROI from JSON, or fallback to default
    normal_affinity = kwargs.get("normal_affinity",
                                  roi_json.get("normal_affinity", DEFAULT_AFFINITY))

    # Build the scene(SimpleNamespace) required by run_completion
    lo = center - radius
    hi = center + radius
    scene = SimpleNamespace(
        name=roi_json.get("name", os.path.basename(roi_path)),
        model=model, hole_lo=lo, hole_hi=hi,
        center=center, roi_center=center, roi_radius=radius,
    )

    # Run completion with the canonical config
    start = time.time()
    result = geometry.run_completion(
        model, scene, baseline=method, seed=seed,
        normal_affinity=normal_affinity,
        hole_mask_override=mask, spawn_rule="count_matched",
        **{k: v for k, v in kwargs.items()
           if k in ("semantic_gate", "boundary_k", "knn_k", "sh_degree")}
    )
    runtime = time.time() - start

    # Evaluate
    spacing = float(result.spawn_budget_diagnostics["robust_spacing"])
    geo, pr = geometric_metrics(result.new_xyz, gt, (0.5, 1.0, 2.0), spacing, seed)

    # Quality metrics
    pred = result.new_xyz
    gt_idx = np.where(~result.kept_mask)[0]
    gt_normals = geometry.estimate_normals_local_pca_at(xyz, gt_idx, k=16)
    _, nearest = cKDTree(gt).query(pred, k=1)
    dots = np.clip(np.abs(np.sum(result.new_normals * gt_normals[nearest], axis=1)), 0, 1)
    new_sh = result.new_attributes["features_dc"]
    gt_sh = removed._features_dc.detach().cpu().numpy().reshape(len(gt), -1)
    bnd_idx = result.boundary_idx
    bnd_sh = model._features_dc.detach().cpu().numpy()[bnd_idx].reshape(len(bnd_idx), -1)

    row = {
        "method": method, "normal_affinity": normal_affinity,
        "N_budget": result.spawn_budget, "N_spawn": len(result.new_xyz),
        "N_GT_evaluation_only": len(gt),
        "observable_local_spacing": spacing,
        **geo,
        "normal_angular_error": float(np.degrees(np.arccos(dots)).mean()),
        "appearance_rmse": metrics.appearance_rmse_gen(pred, new_sh, gt, gt_sh),
        "boundary_seam_error": metrics.boundary_seam_error(
            pred, new_sh, xyz[bnd_idx], bnd_sh),
        "runtime_s": runtime,
    }
    # Add precision/recall
    for prec_row in pr:
        suffix = str(prec_row["threshold_multiplier"]).replace(".", "p")
        row["precision_" + suffix] = prec_row["precision"]
        row["recall_" + suffix] = prec_row["recall"]
        row["fscore_" + suffix] = prec_row["fscore"]
    return row
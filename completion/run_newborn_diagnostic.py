"""Newborn-support diagnostic + pruning-feasibility study (real ramen).

SINGLE-SCENE MULTI-ROI DIAGNOSTIC VALIDATION.

Freezes the definitive global-affinity experiment (commit ec04143, official
ramen checkpoint, 25 frozen ROIs, count-matched spawn, seeds, C0-C3, global
hard/soft/adaptive, canonical evaluator) and adds a per-newborn observable
diagnostic layer.

The completion algorithm is NOT changed: run_completion is called exactly as
in run_global_affinity.  This script only (a) records pre-GT observable
descriptors for every newborn and (b) uses held-out removed GT purely for
evaluation labels (GOOD@1x / GOOD@2x / normalized_GT_distance).  No
descriptor or label feeds back into completion.

Answering: can erroneous newborn Gaussians be identified using ONLY
pre-GT observable support signals?
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, ranksums

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion import geometry, metrics
from completion.gaussian_model import GaussianModel
from completion.run_real_controlled import subset_model
from completion.run_count_matched_ablation import geometric_metrics, deterministic_subset
from completion.run_global_affinity import (GLOBAL_POLICIES, VARIANTS, AFFINITIES,
                                            load_25_rois, collect_metadata,
                                            read_csv, write_csv, DEFAULT_ROIS_CSV)

FOCUS = ["C1", "C3"]
BASELINE = "C0"
DESCRIPTOR_GROUPS = {
    "geometry": ["dist_to_nearest_survivor", "mean_dist_k_survivors",
                 "n_survivors_1x", "n_survivors_2x", "local_support_density",
                 "dist_to_fitted_surface", "mls_residual", "fitted_curvature",
                 "pca_eig_0", "pca_eig_1", "pca_eig_2", "normal_confidence",
                 "normal_agreement"],
    "graph": ["component_size", "component_fraction", "component_boundary_support",
              "component_area", "dist_from_boundary", "propagation_depth"],
    "semantic": ["semantic_confidence", "semantic_entropy", "semantic_agreement",
                 "semantic_purity"],
    "appearance": ["appearance_interp_variance", "appearance_disagreement"],
    "density": ["expected_spacing", "newborn_spacing", "density_ratio"],
}


# ---------------------------------------------------------------------------
# per-newborn observable descriptors (pre-GT)
# ---------------------------------------------------------------------------

def newborn_descriptors(model, xyz, roi, result, method):
    """Compute observable descriptors per newborn.  NO removed-GT used here.

    `result` carries boundary_idx, new_xyz, new_normals, fitted_xyz,
    normal_confidence, pca_eigenvalues, component_labels, surface_label,
    propagation weights & confidence_terms, graph_rows/cols.
    """
    new = result.new_xyz
    P = len(new)
    if P == 0:
        return None
    kept = result.kept_mask
    kept_xyz = xyz[kept]
    kept_idx = np.where(kept)[0]
    bnd_idx = result.boundary_idx
    bnd_xyz = xyz[bnd_idx]
    spacing = float(result.boundary_spacing) if result.boundary_spacing else \
        float(result.spawn_budget_diagnostics["robust_spacing"])

    # --- geometry / support ---
    tree_kept = cKDTree(kept_xyz)
    d1, i1 = tree_kept.query(new, k=1)
    kk = min(8, len(kept_xyz))
    dk, _ = tree_kept.query(new, k=kk)
    mean_dk = dk.mean(axis=1)
    n1 = np.sum(dk <= (1.0 * spacing), axis=1)
    n2 = np.sum(dk <= (2.0 * spacing), axis=1)
    local_density = n2 / (np.pi * (2.0 * spacing) ** 2 + 1e-12)
    fit = result.fitted_xyz if result.fitted_xyz is not None else new
    dist_to_surf = np.linalg.norm(new - fit, axis=1)
    # MLS residual: how well the birth grid fits the local tangent plane of survivors
    mls_residual = np.zeros(P, dtype=np.float32)
    tree_b = cKDTree(bnd_xyz)
    _, nb = tree_b.query(new, k=min(8, len(bnd_xyz)))
    for p in range(P):
        pts = bnd_xyz[nb[p]]
        mu = pts.mean(0)
        cov = (pts - mu).T @ (pts - mu) / max(len(pts) - 1, 1)
        eig, v = np.linalg.eigh(cov)
        normal = v[:, 0]
        mls_residual[p] = float(abs((new[p] - mu) @ normal))
    curv = result.local_curvature if result.local_curvature is not None else \
        np.full(P, 0.0)
    eig = result.pca_eigenvalues if result.pca_eigenvalues is not None else \
        np.zeros((P, 3))
    nconf = result.normal_confidence if result.normal_confidence is not None else \
        np.full(P, np.nan)

    # per-boundary instrument arrays -> broadcast to each newborn via nearest boundary
    tree_b0 = cKDTree(bnd_xyz)
    _, iB = tree_b0.query(new, k=1)
    if np.asarray(eig).ndim == 2 and np.asarray(eig).shape[0] == len(bnd_xyz):
        eig_new = np.asarray(eig)[iB]
    else:
        eig_new = np.zeros((P, 3), dtype=float)
    if np.asarray(curv).ndim == 1 and len(np.asarray(curv)) == len(bnd_xyz):
        curv_new = np.asarray(curv)[iB]
    else:
        curv_new = np.zeros(P)
    if np.asarray(nconf).ndim == 1 and len(np.asarray(nconf)) == len(bnd_xyz):
        nconf_new = np.asarray(nconf)[iB]
    else:
        nconf_new = np.full(P, np.nan)

    # normal agreement with nearest supporting survivor (abs dot, no GT)
    _, si = tree_kept.query(new, k=1)              # nearest surviving gaussian
    surv_normals = result.normals  # boundary normals... need kept normals
    # fallback: recompute PCA normals at nearest survivors
    kept_normals = geometry.estimate_normals_local_pca_at(
        xyz, kept_idx[np.unique(si)], k=16)
    # map back
    reverse = {}
    for q, idx in enumerate(np.unique(si)):
        reverse[idx] = q
    surv_n_aggr = np.array([kept_normals[reverse[int(s)]] for s in si])
    nd = np.abs(np.sum(result.new_normals * surv_n_aggr, axis=1))
    normal_agreement = np.clip(nd, 0, 1)

    # --- graph / component ---
    comp_labels = result.component_labels  # labels of boundary gaussians
    comp_of_new = result.surface_label     # component id per newborn
    ncomp = int(comp_labels.max()) + 1 if comp_labels is not None else 1
    comp_sizes = np.bincount(comp_labels, minlength=ncomp) if comp_labels is not None else \
        np.array([len(bnd_xyz)])
    comp_size = np.array([comp_sizes[c] for c in comp_of_new], dtype=float)
    comp_fraction = comp_size / max(len(bnd_xyz), 1)
    comp_boundary_support = comp_size.astype(float)
    # component area: from PCA eigenvalues of the component boundary spread
    comp_area = np.zeros(P, dtype=float)
    if comp_labels is not None:
        for c in range(ncomp):
            members = bnd_xyz[comp_labels == c]
            if len(members) < 4:
                for p in np.where(comp_of_new == c)[0]:
                    comp_area[p] = float(spacing ** 2)
                continue
            mu = members.mean(0)
            eig, _ = np.linalg.eigh((members - mu).T @ (members - mu) / len(members))
            area = float(np.sqrt(max(eig[-1], 0) * max(eig[-2], 0)))  # in-plane spread
            for p in np.where(comp_of_new == c)[0]:
                comp_area[p] = area
    else:
        comp_area[:] = 1.0

    tree_b = cKDTree(bnd_xyz)
    d_boundary, i_boundary = tree_b.query(new, k=1)
    dist_from_boundary = d_boundary
    # propagation depth: distance from boundary / local spacing
    prop_depth = d_boundary / max(spacing, 1e-8)

    # --- semantic ---
    raw_sem = model._objects_dc.detach().cpu().numpy().reshape(len(model.get_xyz), -1)
    if raw_sem.shape[1] == 0:
        sem_conf = np.full(P, 1.0); sem_ent = np.zeros(P)
        sem_agr = np.ones(P); sem_pur = np.ones(P)
    else:
        shifted = raw_sem - raw_sem.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True) + 1e-12
        sem_ent = -(probs * np.log(probs + 1e-12)).sum(axis=1)
        sem_conf = probs.max(axis=1)
        # component's dominant semantic label (from boundary members only)
        comp_sem_mode = {}
        for c in np.unique(comp_of_new):
            members = bnd_idx[comp_labels == c]
            lbl = np.argmax(raw_sem[members], axis=1)
            comp_sem_mode[c] = Counter(lbl).most_common(1)[0][0]
        # semantic agreement = nearest surviving gaussian label == component mode
        surv_labels = np.argmax(raw_sem[kept_idx[i1]], axis=1)
        comp_mode = np.array([comp_sem_mode[c] for c in comp_of_new])
        sem_agr = (surv_labels == comp_mode).astype(float)
        # component purity = max label fraction over boundary members
        sem_pur = np.zeros(P)
        for c in np.unique(comp_of_new):
            members = bnd_idx[comp_labels == c]
            lbl = np.argmax(raw_sem[members], axis=1)
            cnt = Counter(lbl)
            pur = cnt.most_common(1)[0][1] / len(members)
            sem_pur[comp_of_new == c] = pur
        # confidence + entropy of the nearest surviving gaussian
        sem_conf = sem_conf[kept_idx[i1]]
        sem_ent = sem_ent[kept_idx[i1]]

    # --- appearance ---
    app = model._features_dc.detach().cpu().numpy().reshape(len(model.get_xyz), -1)
    surv_app = app[kept_idx[i1]]
    new_app = result.new_attributes["features_dc"]
    appearance_disagreement = np.sqrt(((new_app - surv_app) ** 2).sum(axis=1))
    # interpolation variance: variance of nearest survivors' appearances around mean
    app_var = np.zeros(P)
    for p in range(P):
        nbr_app = app[kept_idx[nb[p]]]
        app_var[p] = nbr_app.var(axis=0).sum()

    # --- density ---
    tree_new = cKDTree(new)
    dnn = tree_new.query(new, k=2)[0][:, 1] if P > 1 else np.full(P, spacing)
    expected_spacing = np.full(P, spacing)
    newborn_spacing = dnn
    density_ratio = (expected_spacing / (newborn_spacing + 1e-12))

    out = {
        "roi": roi, "policy": result.normal_affinity, "method": method,
        "n_newborn": P,
        "dist_to_nearest_survivor": d1.astype(float),
        "mean_dist_k_survivors": mean_dk.astype(float),
        "n_survivors_1x": n1.astype(int),
        "n_survivors_2x": n2.astype(int),
        "local_support_density": local_density.astype(float),
        "dist_to_fitted_surface": dist_to_surf.astype(float),
        "mls_residual": mls_residual.astype(float),
        "fitted_curvature": curv_new.astype(float),
        "pca_eig_0": eig_new[:, 0].astype(float),
        "pca_eig_1": eig_new[:, 1].astype(float),
        "pca_eig_2": eig_new[:, 2].astype(float),
        "normal_confidence": nconf_new.astype(float),
        "normal_agreement": normal_agreement.astype(float),
        "component_size": comp_size,
        "component_fraction": comp_fraction,
        "component_boundary_support": comp_boundary_support,
        "component_area": comp_area,
        "dist_from_boundary": dist_from_boundary.astype(float),
        "propagation_depth": prop_depth.astype(float),
        "semantic_confidence": sem_conf.astype(float),
        "semantic_entropy": sem_ent.astype(float),
        "semantic_agreement": sem_agr.astype(float),
        "semantic_purity": sem_pur.astype(float),
        "appearance_interp_variance": app_var.astype(float),
        "appearance_disagreement": appearance_disagreement.astype(float),
        "expected_spacing": expected_spacing,
        "newborn_spacing": newborn_spacing.astype(float),
        "density_ratio": density_ratio.astype(float),
    }
    return out


def run_cell_pruned(model, xyz, roi, affinity, method, seed, keep_mask):
    """Recompute completion metrics on a retained subset of newborns.

    Same cell as run_cell_observed (same completion call, no change), but metrics are
    evaluated on `keep_mask`ed newborns only.  Used for the pruning feasibility study;
    the pruning rule was selected by held-out ROI CV and is applied ONLY at evaluation.
    """
    center = roi["center"]; radius = roi["radius"]
    mask = np.linalg.norm(xyz - center, axis=1) <= radius
    if mask.sum() < 8:
        return None
    scene = SimpleNamespace(name=roi["roi"], model=model, hole_lo=center - radius,
                            hole_hi=center + radius, center=center,
                            roi_center=center, roi_radius=radius)
    result = geometry.run_completion(model, scene, baseline=method, seed=seed,
                                     normal_affinity=affinity, semantic_gate="hard",
                                     hole_mask_override=mask, spawn_rule="count_matched")
    removed = subset_model(model, mask)
    gt = removed.get_xyz.detach().cpu().numpy()
    spacing = float(result.spawn_budget_diagnostics["robust_spacing"])
    if keep_mask is None or len(keep_mask) != len(result.new_xyz):
        keep = np.ones(len(result.new_xyz), dtype=bool)
    else:
        keep = np.asarray(keep_mask, dtype=bool)

    if keep.sum() == 0:
        return None
    kept_new = result.new_xyz[keep]
    kept_normals = result.new_normals[keep]
    kept_sh = result.new_attributes["features_dc"][keep]

    geo, pr = geometric_metrics(kept_new, gt, (0.5, 1.0, 2.0), spacing, seed)
    _, nearest = cKDTree(gt).query(kept_new, k=1)
    gt_idx = np.where(~result.kept_mask)[0]
    gt_normals = geometry.estimate_normals_local_pca_at(xyz, gt_idx, k=16)
    dots = np.clip(np.abs(np.sum(kept_normals * gt_normals[nearest], axis=1)), 0, 1)
    gt_sh = removed._features_dc.detach().cpu().numpy().reshape(len(gt), -1)
    bnd_idx = result.boundary_idx
    bnd_sh = model._features_dc.detach().cpu().numpy()[bnd_idx].reshape(len(bnd_idx), -1)

    row = {
        "roi": roi["roi"], "policy": affinity, "method": method, "seed": seed,
        "N_budget": result.spawn_budget, "N_spawn": len(result.new_xyz),
        "N_retained": int(keep.sum()), "retain_fraction": float(keep.mean()),
        "N_GT_evaluation_only": len(gt),
        "pred_to_gt": geo["pred_to_gt_mean"], "gt_to_pred": geo["gt_to_pred_mean"],
        "symmetric_chamfer": geo["symmetric_chamfer"],
        "equal_cardinality_chamfer": geo["equal_cardinality_chamfer"],
        "normal_error": float(np.degrees(np.arccos(dots)).mean()),
        "appearance_rmse": metrics.appearance_rmse_gen(kept_new, kept_sh, gt, gt_sh),
        "seam_error": metrics.boundary_seam_error(
            kept_new, kept_sh, xyz[bnd_idx], bnd_sh),
    }
    for p in pr:
        t = p["threshold_multiplier"]
        row["fscore_{}x".format(t)] = p["fscore"]
        row["precision_{}x".format(t)] = p["precision"]
        row["recall_{}x".format(t)] = p["recall"]
    return row


def run_cell_observed(model, xyz, roi, affinity, method, seed):
    """Run one cell and return (metric_row, descriptor dict)."""
    center = roi["center"]; radius = roi["radius"]
    mask = np.linalg.norm(xyz - center, axis=1) <= radius
    if mask.sum() < 8:
        return None, None
    scene = SimpleNamespace(name=roi["roi"], model=model, hole_lo=center - radius,
                            hole_hi=center + radius, center=center,
                            roi_center=center, roi_radius=radius)
    start = time.time()
    result = geometry.run_completion(model, scene, baseline=method, seed=seed,
                                     normal_affinity=affinity, semantic_gate="hard",
                                     hole_mask_override=mask, spawn_rule="count_matched")
    runtime = time.time() - start
    removed = subset_model(model, mask)
    gt = removed.get_xyz.detach().cpu().numpy()
    spacing = float(result.spawn_budget_diagnostics["robust_spacing"])
    geo, pr = geometric_metrics(result.new_xyz, gt, (0.5, 1.0, 2.0), spacing, seed)

    gt_idx = np.where(~result.kept_mask)[0]
    gt_normals = geometry.estimate_normals_local_pca_at(xyz, gt_idx, k=16)
    _, nearest = cKDTree(gt).query(result.new_xyz, k=1)
    dots = np.clip(np.abs(np.sum(result.new_normals * gt_normals[nearest], axis=1)), 0, 1)
    new_sh = result.new_attributes["features_dc"]
    gt_sh = removed._features_dc.detach().cpu().numpy().reshape(len(gt), -1)
    bnd_idx = result.boundary_idx
    bnd_sh = model._features_dc.detach().cpu().numpy()[bnd_idx].reshape(len(bnd_idx), -1)

    row = {
        "roi": roi["roi"], "policy": affinity, "method": method, "seed": seed,
        "N_budget": result.spawn_budget, "N_spawn": len(result.new_xyz),
        "N_GT_evaluation_only": len(gt),
        "pred_to_gt": geo["pred_to_gt_mean"], "gt_to_pred": geo["gt_to_pred_mean"],
        "symmetric_chamfer": geo["symmetric_chamfer"],
        "equal_cardinality_chamfer": geo["equal_cardinality_chamfer"],
        "normal_error": float(np.degrees(np.arccos(dots)).mean()),
        "appearance_rmse": metrics.appearance_rmse_gen(result.new_xyz, new_sh, gt, gt_sh),
        "seam_error": metrics.boundary_seam_error(result.new_xyz, new_sh, xyz[bnd_idx], bnd_sh),
        "runtime_s": runtime,
    }
    for p in pr:
        t = p["threshold_multiplier"]
        row["fscore_{}x".format(t)] = p["fscore"]
        row["precision_{}x".format(t)] = p["precision"]
        row["recall_{}x".format(t)] = p["recall"]

    # per-newborn descriptors + gt labels
    desc = newborn_descriptors(model, xyz, roi["roi"], result, method)
    if desc is not None:
        # evaluation-only labels from held-out GT
        d_gt, _ = cKDTree(gt).query(result.new_xyz, k=1)
        norm_gt = d_gt / max(spacing, 1e-8)
        desc["normalized_GT_distance"] = norm_gt.astype(float)
        desc["GOOD@1x"] = (norm_gt <= 1.0).astype(int)
        desc["GOOD@2x"] = (norm_gt <= 2.0).astype(int)
        desc["distance_to_GT"] = d_gt.astype(float)
    return row, desc


def long_format(descs):
    """Stack per-newborn descriptor dicts into a long-format table.

    Scalar ids (roi/method/policy/seed) are repeated; 1-D per-newborn arrays are
    expanded.  Every dict must carry the full field set.
    """
    rows = []
    for d in descs:
        P = d["n_newborn"]
        # scalar fields (strings or 0-dim) -> repeated per newborn
        base = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                continue
            if isinstance(v, (list, tuple)):
                continue
            base[k] = v
        for p in range(P):
            rec = dict(base)
            for k, v in d.items():
                if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == P:
                    rec[k] = v[p]
            rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--rois-csv", default=DEFAULT_ROIS_CSV)
    ap.add_argument("--out", default="outputs/newborn_support_diagnostic")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta = collect_metadata(args.checkpoint, args.seed)
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    rois = load_25_rois(args.rois_csv)
    model = GaussianModel(3)
    model.load_ply(args.checkpoint)
    xyz = model.get_xyz.detach().cpu().numpy()
    meta["number_of_gaussians"] = int(len(xyz))

    descs, rows = [], []
    total = 25 * 3 * 4
    for aff in AFFINITIES:
        for roi in rois:
            for method in VARIANTS:
                row, desc = run_cell_observed(model, xyz, roi, aff, method, args.seed)
                if row is not None:
                    rows.append(row)
                    if desc is not None:
                        descs.append(desc)
                n = len(rows)
                if n % 25 == 0:
                    print("[newborn] {}/{}".format(n, total), flush=True)

    write_csv(os.path.join(args.out, "completion_level_results.csv"), rows)
    if descs:
        long_rows = long_format(descs)
        write_csv(os.path.join(args.out, "newborn_descriptors.csv"), long_rows)
    print("[newborn] {} cells, {} newborn records".format(len(rows), len(long_rows)))


if __name__ == "__main__":
    main()
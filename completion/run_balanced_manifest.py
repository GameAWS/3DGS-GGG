"""Balanced R1-R4 observable ROI constructor across real scenes.

MULTI-SCENE REAL 3DGS SURFACE-LAYER RECOVERABILITY DIAGNOSTIC — benchmark
construction stage.

For every scene we scan many candidate ROIs, compute ONLY pre-completion
observable descriptors (layer_features.roi_descriptors), and assign each
candidate to an R1-R4 support/ambiguity group with GLOBALLY FIXED
thresholds.  Completion performance / removed-GT reconstruction quality are
NEVER used to select group membership.

Outputs per scene (and a combined manifest):
  roi_manifest.csv           -- frozen candidate list with groups + descriptors
  candidate_scan.csv         -- all scanned candidates with their groups
  manifest.json              -- thresholds + scene metadata + group counts

Targets >=5 ROIs per group per scene where the data permits; a scene that
naturally lacks a group reports that honestly (no fabricated balance).
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion.gaussian_model import GaussianModel
from completion.run_global_affinity import read_csv, write_csv
from completion.cpu_cameras import load_cameras
from completion.run_multiscene_generalization import sample_rois
from completion.layer_features import roi_descriptors

# ---- globally fixed thresholds (pre-defined, never tuned per scene) ----
SUPPORT_CAMERAS = 5
SUPPORT_BOUNDARY = 60
AMBIG_DEPTH_MODES = 3
AMBIG_NORMAL_CLUSTERS = 4
AMBIG_SEMANTIC_IDS = 3

SCENES = {
    "ramen": {
        "checkpoint": "checkpoints_download/ramen/point_cloud/iteration_30000/point_cloud.ply",
        "data": "checkpoints_download/data_extracted/ramen",
    },
    "figurines": {
        "checkpoint": "checkpoints_download/assets/figurines/point_cloud/iteration_30000/point_cloud.ply",
        "data": "checkpoints_download/assets/data_extracted/figurines",
    },
    "teatime": {
        "checkpoint": "checkpoints_download/assets/teatime/point_cloud/iteration_30000/point_cloud.ply",
        "data": "checkpoints_download/assets/data_extracted/teatime",
    },
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def group_of(d):
    """Observable-only R1-R4 assignment (globally fixed thresholds)."""
    support = (int(d["n_cameras_see_hole"]) >= SUPPORT_CAMERAS) and \
              (float(d["boundary_support_count"]) >= SUPPORT_BOUNDARY)
    high_ambig = (int(d["n_depth_modes"]) >= AMBIG_DEPTH_MODES) or \
                 (int(d["n_normal_clusters"]) >= AMBIG_NORMAL_CLUSTERS) or \
                 (int(d["n_semantic_ids"]) >= AMBIG_SEMANTIC_IDS)
    if support and not high_ambig:
        return "R1"
    if support and high_ambig:
        return "R2"
    if not support and not high_ambig:
        return "R3"
    return "R4"


def select_cameras(cams, center, radius, n_groups=4):
    """Deterministic camera selection: up to 2 per quadrant by depth."""
    if not cams:
        return []
    eyes = {c["uid"]: __import__("completion.cpu_cameras", fromlist=["camera_eye"]).camera_eye(c)
            for c in cams}
    scored = []
    for c in cams:
        eye = eyes[c["uid"]]
        ray = eye - center
        d = float(np.linalg.norm(ray))
        scored.append((d, c["uid"]))
    scored.sort()
    return [uid for _, uid in scored[:n_groups]]


def scan_scene(scene_name, scene, out_dir, target_per_group, max_candidates):
    model = GaussianModel(3) if os.path.isfile(scene["checkpoint"]) else None
    if model is None:
        print("[scan] {}: checkpoint MISSING, skipping".format(scene_name))
        return [], {}
    model.load_ply(scene["checkpoint"])
    xyz = model.get_xyz.detach().cpu().numpy()
    cams = load_cameras(scene["data"], 800) if os.path.isdir(scene["data"]) else []
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=2)
    spacing = float(np.median(d[:, 1]))

    candidates = sample_rois(xyz, tree, spacing, scene_name, max_candidates, [])
    rows = []
    group_counts = Counter()
    for cand in candidates:
        d = roi_descriptors(model, xyz, cand, cams, scene_tree=tree)
        if d is None:
            continue
        g = group_of(d)
        d["group"] = g
        d["scene"] = scene_name
        d["center_x"], d["center_y"], d["center_z"] = (float(x) for x in cand["center"])
        d["camera_selected_ids"] = ";".join(str(i) for i in
                                            select_cameras(cams, cand["center"], cand["radius"]))
        rows.append(d)
        group_counts[g] += 1
    print("[scan] {}: {} candidates -> groups {}".format(
        scene_name, len(rows), dict(group_counts)))
    write_csv(os.path.join(out_dir, "candidate_scan_" + scene_name + ".csv"), rows)
    return rows, {"candidates": len(rows), "groups": dict(group_counts)}


def build_manifest(scenes_rows, out_dir, target_per_group):
    """Select up to target_per_group per group per scene, freeze the manifest."""
    manifest = []
    for scene_name, rows in scenes_rows.items():
        for g in ("R1", "R2", "R3", "R4"):
            members = [r for r in rows if r["group"] == g]
            for m in members[:target_per_group]:
                manifest.append({k: m.get(k, "") for k in (
                    "scene", "roi", "group", "center_x", "center_y", "center_z", "radius",
                    "n_cameras_see_hole", "visible_support_fraction",
                    "boundary_support_count", "norm_dist_center_support",
                    "support_density", "n_depth_modes", "depth_discontinuity",
                    "n_normal_clusters", "normal_dispersion", "n_semantic_ids",
                    "semantic_entropy", "cross_modal_normal_sem_agreement",
                    "camera_selected_ids")})
    write_csv(os.path.join(out_dir, "roi_manifest.csv"), manifest)
    return manifest


def make_group_balance_plot(out_dir, scenes_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir = os.path.join(out_dir, "plots")
    os.makedirs(pdir, exist_ok=True)
    groups = ("R1", "R2", "R3", "R4")
    scenes = list(scenes_rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.2
    for i, g in enumerate(groups):
        vals = [len([r for r in scenes_rows[s] if r["group"] == g]) for s in scenes]
        ax.bar(np.arange(len(scenes)) + i * width, vals, width, label=g)
    ax.set_xticks(np.arange(len(scenes)) + 1.5 * width, scenes)
    ax.set_ylabel("candidate count"); ax.set_title("R1-R4 observable group balance (candidates)")
    ax.legend(); ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(pdir, "group_balance.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/recoverability_final")
    ap.add_argument("--target-per-group", type=int, default=8)
    ap.add_argument("--max-candidates", type=int, default=48)
    ap.add_argument("--from-existing", action="store_true",
                    help="reuse candidate_scan_<scene>.csv instead of re-scanning")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scenes_rows = {}
    for name, scene in SCENES.items():
        if args.from_existing:
            p = os.path.join(args.out, "candidate_scan_" + name + ".csv")
            if os.path.isfile(p):
                scenes_rows[name] = read_csv(p)
                print("[manifest] reused {} candidates for {}".format(
                    len(scenes_rows[name]), name))
                continue
        rows, _ = scan_scene(name, scene, args.out, args.target_per_group,
                             args.max_candidates)
        scenes_rows[name] = rows

    make_group_balance_plot(args.out, scenes_rows)

    manifest = build_manifest(scenes_rows, args.out, args.target_per_group)
    manifest_summary = defaultdict(Counter)
    for m in manifest:
        manifest_summary[m["scene"]][m["group"]] += 1

    metadata = []
    data_sources = {
        "ramen": "HF mqye/Gaussian-Grouping data/lerf_mask/ramen.zip",
        "figurines": "HF mqye/Gaussian-Grouping data/lerf_mask/figurines.zip",
        "teatime": "HF mqye/Gaussian-Grouping data/lerf_mask/teatime.zip",
    }
    for name, scene in SCENES.items():
        meta = {"scene": name, "git_commit": git_commit(),
                "checkpoint": scene["checkpoint"],
                "data_root": scene["data"],
                "image_data_source": data_sources.get(name, "")}
        if os.path.isfile(scene["checkpoint"]):
            meta["checkpoint_sha256"] = sha256_file(scene["checkpoint"])
            m = GaussianModel(3); m.load_ply(scene["checkpoint"])
            meta["gaussians"] = int(m.get_xyz.shape[0])
            meta["sh_degree"] = m.max_sh_degree
            meta["num_objects"] = m.num_objects
        else:
            meta["checkpoint_missing"] = True
        if os.path.isdir(scene["data"]):
            try:
                cams = load_cameras(scene["data"], 800)
                meta["cameras"] = len(cams)
            except Exception as e:
                meta["cameras"] = -1; meta["camera_error"] = str(e)[:60]
        meta["manifest_groups"] = dict(manifest_summary[name])
        metadata.append(meta)
    write_csv(os.path.join(args.out, "scene_metadata.csv"), metadata)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({
            "thresholds": {"support_cameras": SUPPORT_CAMERAS,
                           "support_boundary": SUPPORT_BOUNDARY,
                           "ambig_depth_modes": AMBIG_DEPTH_MODES,
                           "ambig_normal_clusters": AMBIG_NORMAL_CLUSTERS,
                           "ambig_semantic_ids": AMBIG_SEMANTIC_IDS},
            "scene_metadata": metadata,
        }, f, indent=2)
    print("[manifest] {} ROIs frozen: {}".format(
        len(manifest), {k: dict(v) for k, v in manifest_summary.items()}))


if __name__ == "__main__":
    main()
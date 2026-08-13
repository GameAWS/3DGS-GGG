"""Config-based region selector for real-scene controlled-hole validation.

Lets a user define controlled missing regions in a real Gaussian Grouping
checkpoint without modifying gaussian_renderer.  The completion pipeline
(geometry.run_completion) is called with the exact same code as the
synthetic study; removed Gaussians are held out as ground truth and never
used during completion.

Region categories supported:
  - wall-floor junction      ("junction")
  - nearby parallel surfaces ("layered")
  - planar textured region   ("planar")
  - curved surface           ("curved")

Region JSON schema (one per category):
{
  "name": "roi_B_junction",
  "category": "junction",
  "selector": {"type": "sphere", "center": [x, y, z], "radius": r}   # or aabb / oriented_box
}

Usage:
  python completion/run_real_selected.py \
    --checkpoint /path/to/point_cloud.ply \
    --regions regions.json \
    --out outputs/real_validation \
    --render

Outputs per region (under --out/<name>/):
  original.ply hole.ply removed_gt.ply   GT / hole / held-out ground truth
  C0.ply C1.ply C2.ply C3.ply            completed models
  metrics.csv                            all metrics (ablation + affinity configs)
  comparison.png                         GT | Hole | C0 | C1 | C2 | C3 renders
  config.json diagnostics.json           fixed hyperparameters + diagnostics
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from completion import geometry
from completion.gaussian_model import GaussianModel
from completion.run_real_controlled import (
    subset_model, selector_mask, cameras_for_roi, run_roi, write_rows, FIELDS)


def load_regions(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("regions", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="path to a real GG checkpoint point_cloud.ply")
    parser.add_argument("--regions", required=True,
                        help="JSON file listing the controlled regions to remove")
    parser.add_argument("--out", default="outputs/real_validation")
    parser.add_argument("--scene-name")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--skip-ply", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise SystemExit("checkpoint not found: {}".format(args.checkpoint))
    regions = load_regions(args.regions)
    if not regions:
        raise SystemExit("no regions found in {}".format(args.regions))

    model = GaussianModel(args.sh_degree)
    model.load_ply(args.checkpoint)
    scene_name = args.scene_name or os.path.basename(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))))
    scene_dir = os.path.join(args.out, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    all_rows = []
    result_types = []
    for candidate in regions:
        candidate["name"] = candidate.get("name", "roi_" + candidate["category"])
        try:
            rows = run_roi(model, candidate, scene_dir, "REAL SELECTED REGION",
                           args.resolution, args.no_render, args.skip_ply)
            all_rows.extend(rows)
            result_types.append(candidate["name"])
            print("[real-selected] {}: {} rows".format(candidate["name"], len(rows)))
        except RuntimeError as e:
            print("[real-selected] SKIP {}: {}".format(candidate["name"], e))

    if all_rows:
        write_rows(os.path.join(scene_dir, "selected_region_metrics.csv"), all_rows)
    summary = {
        "scene": scene_name,
        "regions_requested": [c["name"] for c in regions],
        "regions_completed": result_types,
        "note": "Removed Gaussians are held out as ground truth and never used during "
                "completion. Render metrics use identical inferred ROI cameras and the "
                "CPU fallback renderer.",
    }
    with open(os.path.join(scene_dir, "selected_region_manifest.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[real-selected] completed {} regions -> {}".format(len(result_types), scene_dir))


if __name__ == "__main__":
    main()
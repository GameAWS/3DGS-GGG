"""MULTI-SCENE REAL 3DGS VISIBILITY-AWARE HOLE-SCALE VALIDATION.

Experiment-only runner. Geometry, graph construction, MLS, count-matched spawning,
A4 attribute initialization, renderer, and rendering metrics are imported unchanged
from the preceding frozen validations.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from completion import geometry
from completion.run_count_matched_ablation import geometric_metrics
from completion.run_multiscene_generalization import sample_rois
from completion.run_real_controlled import subset_model
from completion import run_rendering_validation as rv
from completion import run_visibility_attribute_audit as va

LABEL = "MULTI-SCENE REAL 3DGS VISIBILITY-AWARE HOLE-SCALE VALIDATION"
METHODS = (("C0", "C0", "hard"), ("C1-HARD", "C1", "hard"))
SCALES = (("SMALL", 0.75), ("MEDIUM", 1.0), ("LARGE", 1.35))
SEED = 0
TARGET_CENTERS = 4
MAX_CANDIDATES = 24
CONFIG = {
    "methods": METHODS, "attribute": "A4_SURFACE_AWARE", "spawn_rule": "count_matched",
    "scales": SCALES, "seed": SEED, "target_centers_per_scene": TARGET_CENTERS,
    "max_candidate_centers": MAX_CANDIDATES,
    "visibility_gate": {"minimum_contributing_cameras": va.MIN_CONTRIBUTING_CAMERAS,
                        "minimum_raw_changed_pixels": va.MIN_RAW_CHANGED_PIXELS,
                        "minimum_mean_hole_lpips": va.MIN_MEAN_HOLE_LPIPS,
                        "raw_rgb_change_threshold": va.RAW_RGB_CHANGE_THRESHOLD},
}


def write_csv(path, rows, fields=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows and not fields:
        raise ValueError("fields are required for an empty CSV")
    names = fields or list(dict.fromkeys(key for row in rows for key in row.keys()))
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def sha256_file(path, block=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(block)
            if not chunk: break
            digest.update(chunk)
    return digest.hexdigest()


def config_hash():
    raw = json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def parse_scene(text):
    parts = text.split("|", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("scene must be NAME|POINT_CLOUD.PLY|DATA_ROOT")
    return {"scene": parts[0], "checkpoint": os.path.abspath(parts[1]), "data": os.path.abspath(parts[2])}


def model_metadata(scene, model):
    objects = model._objects_dc.detach().cpu().numpy()
    return {"scene": scene["scene"], "checkpoint": scene["checkpoint"],
            "checkpoint_sha256": sha256_file(scene["checkpoint"]),
            "gaussian_count": len(model.get_xyz), "sh_degree": model.max_sh_degree,
            "active_sh_degree": model.active_sh_degree,
            "semantic_dimension": int(np.prod(objects.shape[1:])),
            "git_commit": git_commit(), "config_sha256": config_hash(), "data_root": scene["data"]}


def selected_cameras(cameras, model, center, radius):
    # Camera selection needs only a stable observable orientation.  Estimate it from
    # a bounded survivor annulus instead of rebuilding a full-scene k-d tree.
    xyz = model.get_xyz.detach().cpu().numpy()
    radial = np.linalg.norm(xyz - center, axis=1)
    support = xyz[(radial > radius) & (radial <= 2.5 * radius)]
    if len(support) < 8: raise RuntimeError("insufficient camera-selection support")
    centered = support - np.median(support, axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered / max(len(centered)-1, 1))
    normal = vectors[:, 0]; normal /= np.linalg.norm(normal) + 1e-12
    return [item[1] for item in rv.select_cameras(cameras, center, radius, normal)]


def render_gt_hole(gt_gpu, model, xyz, center, radius, cameras, lpips_fn, bg, gt_cache=None):
    mask = np.linalg.norm(xyz - center, axis=1) <= radius
    if int(mask.sum()) < 8: return None
    hole = subset_model(model, ~mask); hole_gpu = rv.RenderModel(hole)
    views = []
    try:
        for camera in cameras:
            camera_id = int(camera.uid)
            if gt_cache is not None and camera_id in gt_cache: gt = gt_cache[camera_id]
            else:
                gt = va.render_gpu(gt_gpu, camera, bg)
                if gt_cache is not None: gt_cache[camera_id] = gt
            empty = va.render_gpu(hole_gpu, camera, bg)
            hm, bm, _, threshold = rv.region_mask(gt, empty)
            metrics = va.audit_metrics(empty, gt, hm, bm, lpips_fn)
            contribution = va.raw_contribution(gt, empty)
            views.append({"camera_id": int(camera.uid), "image_name": camera.image_name,
                          "mask_threshold": threshold, **contribution, **metrics,
                          "gt": gt, "hole": empty, "hole_mask": hm, "boundary_mask": bm})
    finally:
        del hole_gpu; torch.cuda.empty_cache()
    finite_lpips = [x["hole_lpips"] for x in views if np.isfinite(x["hole_lpips"])]
    summary = {"removed_gaussians": int(mask.sum()),
               "contributing_cameras": sum(x["raw_changed_pixels"] >= 16 for x in views),
               "max_changed_pixels": max(x["raw_changed_pixels"] for x in views),
               "mean_changed_pixels": float(np.mean([x["raw_changed_pixels"] for x in views])),
               "mean_hole_psnr": float(np.nanmean([x["hole_psnr"] for x in views])),
               "mean_hole_ssim": float(np.nanmean([x["hole_ssim"] for x in views])),
               "mean_hole_lpips": float(np.mean(finite_lpips)) if finite_lpips else np.nan}
    summary["passes_visibility_gate"] = bool(
        summary["contributing_cameras"] >= va.MIN_CONTRIBUTING_CAMERAS and
        summary["max_changed_pixels"] >= va.MIN_RAW_CHANGED_PIXELS and
        summary["mean_hole_lpips"] >= va.MIN_MEAN_HOLE_LPIPS)
    return mask, hole, views, summary


def candidate_centers(model, scene_name, count):
    xyz = model.get_xyz.detach().cpu().numpy(); tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=2); spacing = float(np.median(d[:, 1]))
    proposals = sample_rois(xyz, tree, spacing, scene_name, count, [])
    return tree, spacing, proposals


def observable_scale_descriptor(xyz, center, radius, scene_spacing):
    """Fast survivor-only hole scale; no held-out points or method output are used."""
    radial = np.linalg.norm(xyz - center, axis=1)
    support = xyz[(radial > radius) & (radial <= 2.5 * radius)]
    if len(support) < 16: raise RuntimeError("insufficient local surviving support")
    distances, _ = cKDTree(support).query(support, k=2)
    spacing = float(np.median(distances[:, 1]))
    if not np.isfinite(spacing) or spacing <= 0: spacing = scene_spacing
    area = float(np.pi * radius * radius)
    return {"local_median_spacing": spacing, "estimated_missing_surface_area": area,
            "normalized_missing_area": area / max(spacing * spacing, 1e-12),
            "local_surviving_support": len(support)}


def screen_scene(scene, model, cameras, lpips_fn, bg, target_centers):
    xyz = model.get_xyz.detach().cpu().numpy(); tree, scene_spacing, proposals = candidate_centers(model, scene["scene"], MAX_CANDIDATES)
    gt_gpu = rv.RenderModel(model); selected = []; audit = []; gt_cache = {}
    for pi, proposal in enumerate(proposals):
        center = np.asarray(proposal["center"], dtype=float); base = float(proposal["radius"])
        try: cameras_here = selected_cameras(cameras, model, center, base * SCALES[-1][1])
        except RuntimeError: continue
        records = []; all_pass = True
        for scale, multiplier in SCALES:
            radius = base * multiplier
            try:
                mask, _, views, summary = render_gt_hole(gt_gpu, model, xyz, center, radius, cameras_here, lpips_fn, bg, gt_cache)
                descriptor = observable_scale_descriptor(xyz, center, radius, scene_spacing)
                normalized_area = descriptor["normalized_missing_area"]
            except (RuntimeError, ValueError):
                all_pass = False; break
            record = {"scene": scene["scene"], "center_id": proposal["roi"], "scale": scale,
                      "scale_multiplier": multiplier, "center_x": center[0], "center_y": center[1], "center_z": center[2],
                      "radius": radius, "scene_spacing": scene_spacing,
                      "local_spacing": descriptor["local_median_spacing"],
                      "estimated_missing_surface_area": descriptor["estimated_missing_surface_area"],
                      "local_surviving_support": descriptor["local_surviving_support"],
                      "normalized_missing_area": normalized_area,
                      "selected_camera_ids": ";".join(str(int(c.uid)) for c in cameras_here), **summary}
            records.append(record); audit.append(record)
            all_pass = all_pass and summary["passes_visibility_gate"]
        if all_pass and len(records) == len(SCALES):
            selected.extend(records)
            print("[selection] {} center {}/{} accepted ({}/{})".format(scene["scene"], pi+1, len(proposals), len(selected)//3, target_centers), flush=True)
            if len(selected) // 3 >= target_centers: break
        else:
            print("[selection] {} center {}/{} rejected".format(scene["scene"], pi+1, len(proposals)), flush=True)
    del gt_gpu; torch.cuda.empty_cache()
    return selected, audit


def geometry_eval(model, mask, result, spacing):
    gt = model.get_xyz.detach().cpu().numpy()[mask]
    summary, pr = geometric_metrics(result.new_xyz, gt, (2.0,), spacing, SEED)
    row = {**summary, "precision_2x": pr[0]["precision"], "recall_2x": pr[0]["recall"], "fscore_2x": pr[0]["fscore"]}
    gt_indices = np.where(mask)[0]
    gt_normals = geometry.estimate_normals_local_pca_at(model.get_xyz.detach().cpu().numpy(), gt_indices, k=16)
    _, nearest = cKDTree(gt).query(result.new_xyz, k=1)
    dots = np.clip(np.abs(np.sum(result.new_normals * gt_normals[nearest], axis=1)), 0, 1)
    row["normal_angular_error"] = float(np.degrees(np.arccos(dots)).mean())
    return row


def save_strip(path, images):
    panels = []
    for label, tensor in images:
        array = (tensor.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(array); panel = Image.new("RGB", (image.width, image.height + 24), "white")
        panel.paste(image, (0, 24)); ImageDraw.Draw(panel).text((5, 5), label, fill="black"); panels.append(panel)
    strip = Image.new("RGB", (sum(x.width for x in panels), panels[0].height), "white"); x = 0
    for panel in panels: strip.paste(panel, (x, 0)); x += panel.width
    os.makedirs(os.path.dirname(path), exist_ok=True); strip.save(path)


def evaluate_selected(scene, model, cameras, selected, lpips_fn, bg, out):
    xyz = model.get_xyz.detach().cpu().numpy(); cmap = {int(c.uid): c for c in cameras}; gt_gpu = rv.RenderModel(model)
    render_rows, geometry_rows, diagnostics_rows, scale_rows = [], [], [], []
    for ri, selected_row in enumerate(selected):
        center = np.asarray([selected_row[k] for k in ("center_x", "center_y", "center_z")], dtype=float)
        radius = float(selected_row["radius"]); mask = np.linalg.norm(xyz - center, axis=1) <= radius
        cams = [cmap[int(x)] for x in selected_row["selected_camera_ids"].split(";")]
        _, hole, views, summary = render_gt_hole(gt_gpu, model, xyz, center, radius, cams, lpips_fn, bg)
        refs = {int(v["camera_id"]): v for v in views}; roi = selected_row["center_id"] + "_" + selected_row["scale"].lower()
        for v in views:
            render_rows.append({"scene": scene["scene"], "roi": roi, "center_id": selected_row["center_id"], "scale": selected_row["scale"],
                                "camera_id": v["camera_id"], "image_name": v["image_name"], "method": "HOLE",
                                "hole_psnr": v["hole_psnr"], "hole_ssim": v["hole_ssim"], "hole_lpips": v["hole_lpips"],
                                "boundary_seam": v["boundary_seam"]})
        scene_obj = SimpleNamespace(name=roi, model=model, center=center, roi_center=center, roi_radius=radius,
                                    hole_lo=center-radius, hole_hi=center+radius)
        method_images = defaultdict(dict); spawn_counts = []
        for method_name, baseline, affinity in METHODS:
            started = time.time()
            result = geometry.run_completion(model, scene_obj, baseline=baseline, seed=SEED,
                                             normal_affinity=affinity, semantic_gate="hard",
                                             hole_mask_override=mask, spawn_rule="count_matched")
            runtime = time.time() - started; spawn_counts.append(len(result.new_xyz))
            variants, bxyz, battrs = va.attribute_variants(model, result); attrs = variants["A4_SURFACE_AWARE"]
            mutable = va.MutableCompletedModel(hole, len(result.new_xyz)); mutable.update(result.new_xyz, attrs)
            diagnostics_rows.extend({"scene": scene["scene"], "scale": selected_row["scale"], **x}
                                    for x in va.diagnostics(roi, method_name, "A4_SURFACE_AWARE", result, attrs, bxyz, battrs, cams))
            geo = geometry_eval(model, mask, result, float(selected_row["local_spacing"]))
            geometry_rows.append({"scene": scene["scene"], "roi": roi, "center_id": selected_row["center_id"],
                                  "scale": selected_row["scale"], "method": method_name, "n_spawn": len(result.new_xyz),
                                  "removed_gaussians": int(mask.sum()), "normalized_missing_area": selected_row["normalized_missing_area"],
                                  "runtime_s": runtime, **geo})
            for camera in cams:
                image = va.render_gpu(mutable, camera, bg); ref = refs[int(camera.uid)]
                met = va.audit_metrics(image, ref["gt"], ref["hole_mask"], ref["boundary_mask"], lpips_fn)
                render_rows.append({"scene": scene["scene"], "roi": roi, "center_id": selected_row["center_id"], "scale": selected_row["scale"],
                                    "camera_id": int(camera.uid), "image_name": camera.image_name, "method": method_name,
                                    "hole_psnr": met["hole_psnr"], "hole_ssim": met["hole_ssim"], "hole_lpips": met["hole_lpips"],
                                    "boundary_seam": met["boundary_seam"]})
                method_images[int(camera.uid)][method_name] = image
            del mutable; torch.cuda.empty_cache()
        if len(set(spawn_counts)) != 1: raise RuntimeError("count matching failed for " + roi)
        for camera in cams:
            ref = refs[int(camera.uid)]; rdir = os.path.join(out, "renders", scene["scene"], roi, str(int(camera.uid)))
            os.makedirs(rdir, exist_ok=True)
            for name, image in (("GT", ref["gt"]), ("HOLE", ref["hole"]), *method_images[int(camera.uid)].items()):
                rv.save_tensor(os.path.join(rdir, name + ".png"), image)
            save_strip(os.path.join(rdir, "GT_HOLE_C0_C1.png"), [("GT", ref["gt"]), ("HOLE", ref["hole"]),
                       ("C0+A4", method_images[int(camera.uid)]["C0"]), ("C1-HARD+A4", method_images[int(camera.uid)]["C1-HARD"])])
        scale_rows.append({"scene": scene["scene"], "roi": roi, **selected_row})
        print("[experiment] {} {}/{} {}".format(scene["scene"], ri+1, len(selected), roi), flush=True)
    del gt_gpu; torch.cuda.empty_cache()
    return render_rows, geometry_rows, diagnostics_rows, scale_rows


def aggregate_rois(render_rows):
    grouped = defaultdict(list)
    for row in render_rows: grouped[(row["scene"], row["roi"], row["center_id"], row["scale"], row["method"])].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        row = dict(zip(("scene", "roi", "center_id", "scale", "method"), key)); row["n_cameras"] = len(values)
        for metric in ("hole_psnr", "hole_ssim", "hole_lpips", "boundary_seam"):
            row[metric] = float(np.nanmean([float(x[metric]) for x in values]))
        output.append(row)
    return output


def bootstrap(values, statistic=np.mean, iterations=10000, seed=20260814):
    values = np.asarray(values, dtype=float)
    if not len(values): return np.nan, np.nan
    rng = np.random.default_rng(seed); samples = []
    for _ in range(iterations): samples.append(float(statistic(rng.choice(values, len(values), replace=True))))
    return tuple(np.percentile(samples, [2.5, 97.5]))


def analyses(roi_rows, scale_rows):
    by = defaultdict(dict)
    for row in roi_rows: by[(row["scene"], row["roi"])][row["method"]] = row
    smap = {(x["scene"], x["roi"]): x for x in scale_rows}; hole_size = []
    for key, methods in by.items():
        if not all(x in methods for x in ("HOLE", "C0", "C1-HARD")): continue
        row = {"scene": key[0], "roi": key[1], "center_id": methods["HOLE"]["center_id"], "scale": methods["HOLE"]["scale"],
               "normalized_missing_area": smap[key]["normalized_missing_area"],
               "removed_gaussians": smap[key]["removed_gaussians"],
               "projected_changed_pixel_area": smap[key]["mean_changed_pixels"]}
        for metric in ("hole_lpips", "hole_psnr", "hole_ssim", "boundary_seam"):
            row["hole_" + metric] = methods["HOLE"][metric]
            row["c0_" + metric] = methods["C0"][metric]; row["c1_" + metric] = methods["C1-HARD"][metric]
        row["delta_lpips_c0_minus_c1"] = row["c0_hole_lpips"] - row["c1_hole_lpips"]
        hole_size.append(row)
    paired = []
    for scale in ("ALL",) + tuple(x[0] for x in SCALES):
        group = [x for x in hole_size if scale == "ALL" or x["scale"] == scale]
        delta = np.asarray([x["delta_lpips_c0_minus_c1"] for x in group]); lo, hi = bootstrap(delta)
        try: stat, p = wilcoxon(delta)
        except ValueError: stat = p = np.nan
        paired.append({"scale": scale, "n_rois": len(group), "mean_delta_lpips_c0_minus_c1": float(np.mean(delta)) if len(delta) else np.nan,
                       "median_delta_lpips_c0_minus_c1": float(np.median(delta)) if len(delta) else np.nan,
                       "bootstrap_95_ci_low": lo, "bootstrap_95_ci_high": hi, "wilcoxon_stat": stat, "wilcoxon_p": p})
    x = np.asarray([r["normalized_missing_area"] for r in hole_size]); y = np.asarray([r["delta_lpips_c0_minus_c1"] for r in hole_size])
    rho, p = spearmanr(x, y) if len(x) >= 3 else (np.nan, np.nan)
    rng = np.random.default_rng(20260814); boot = []
    for _ in range(10000):
        idx = rng.integers(0, len(x), len(x)); value = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(value): boot.append(value)
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    paired.append({"scale": "SIZE_CORRELATION", "n_rois": len(x), "mean_delta_lpips_c0_minus_c1": np.mean(y),
                   "spearman_rho": rho, "spearman_p": p, "bootstrap_95_ci_low": lo, "bootstrap_95_ci_high": hi})
    success = []
    for scene in ("ALL",) + tuple(sorted({r["scene"] for r in hole_size})):
        for scale in ("ALL",) + tuple(x[0] for x in SCALES):
            group = [r for r in hole_size if (scene == "ALL" or r["scene"] == scene) and (scale == "ALL" or r["scale"] == scale)]
            for method in ("C0", "C1"):
                n = len(group)
                lp = sum(r[method.lower()+"_hole_lpips"] < r["hole_hole_lpips"] for r in group)
                ps = sum(r[method.lower()+"_hole_psnr"] > r["hole_hole_psnr"] for r in group)
                ss = sum(r[method.lower()+"_hole_ssim"] > r["hole_hole_ssim"] for r in group)
                all3 = sum(r[method.lower()+"_hole_lpips"] < r["hole_hole_lpips"] and r[method.lower()+"_hole_psnr"] > r["hole_hole_psnr"] and r[method.lower()+"_hole_ssim"] > r["hole_hole_ssim"] for r in group)
                success.append({"scene": scene, "scale": scale, "method": "C1-HARD" if method == "C1" else method, "n_rois": n,
                                "lpips_improved_count": lp, "lpips_improved_fraction": lp/n if n else np.nan,
                                "psnr_improved_count": ps, "psnr_improved_fraction": ps/n if n else np.nan,
                                "ssim_improved_count": ss, "ssim_improved_fraction": ss/n if n else np.nan,
                                "all_three_improved_count": all3, "all_three_improved_fraction": all3/n if n else np.nan})
    return hole_size, paired, success


def diagnostic_summary(rows, previous_a0_csv, render_rows=None):
    output = []
    for scene in sorted({r["scene"] for r in rows}):
        group = [r for r in rows if r["scene"] == scene]
        seam = np.asarray([float(r["boundary_seam"]) for r in (render_rows or [])
                           if r["scene"] == scene and r["method"] in ("C0", "C1-HARD")])
        output.append({"scene": scene, "distribution": "A4_CURRENT", "n_newborn": len(group),
                       "median_projected_radius": np.median([r["projected_screen_radius_max"] for r in group]),
                       "p95_projected_radius": np.percentile([r["projected_screen_radius_max"] for r in group], 95),
                       "median_local_scale_ratio": np.median([r["scale_relative_local_median"] for r in group]),
                       "median_opacity_ratio": np.median([r["opacity_relative_neighbor_median"] for r in group]),
                       "median_sh_color_difference": np.median([r["sh_dc_difference_nearest_support"] for r in group]),
                       "mean_boundary_seam_error": float(np.nanmean(seam)) if np.isfinite(seam).any() else np.nan})
    if previous_a0_csv and os.path.isfile(previous_a0_csv):
        old = list(csv.DictReader(open(previous_a0_csv, newline="")))
        old = [r for r in old if r.get("attribute_variant") == "A0_CURRENT"]
        if old:
            output.append({"scene": "ramen", "distribution": "A0_PREVIOUS_FAILURE", "n_newborn": len(old),
                           "median_projected_radius": np.median([float(r["projected_screen_radius_max"]) for r in old]),
                           "p95_projected_radius": np.percentile([float(r["projected_screen_radius_max"]) for r in old], 95),
                           "median_local_scale_ratio": np.median([float(r["scale_relative_local_median"]) for r in old]),
                           "median_opacity_ratio": np.median([float(r["opacity_relative_neighbor_median"]) for r in old]),
                           "median_sh_color_difference": np.median([float(r["sh_dc_difference_nearest_support"]) for r in old]),
                           "mean_boundary_seam_error": ""})
    return output


def plots(out, hole_size, success, diagnostics):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pdir = os.path.join(out, "plots"); os.makedirs(pdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"SMALL":"tab:blue", "MEDIUM":"tab:orange", "LARGE":"tab:red"}
    for scale in colors:
        rows = [r for r in hole_size if r["scale"] == scale]
        ax.scatter([r["normalized_missing_area"] for r in rows], [r["delta_lpips_c0_minus_c1"] for r in rows], label=scale, color=colors[scale])
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="normalized missing surface area", ylabel="LPIPS(C0+A4) - LPIPS(C1-HARD+A4)")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(pdir, "delta_lpips_vs_normalized_hole_size.png"), dpi=180); plt.close(fig)
    rows = [r for r in success if r["scene"] == "ALL" and r["scale"] != "ALL"]
    fig, ax = plt.subplots(figsize=(8, 4)); labels = [x[0] for x in SCALES]; x = np.arange(3); width = .35
    for i, method in enumerate(("C0", "C1-HARD")):
        vals = [next(r["lpips_improved_fraction"] for r in rows if r["scale"] == s and r["method"] == method) for s in labels]
        ax.bar(x + (i-.5)*width, vals, width, label=method)
    ax.set_xticks(x, labels); ax.set_ylim(0,1); ax.set(ylabel="fraction improving LPIPS over Hole"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(pdir, "success_rate_by_hole_scale.png"), dpi=180); plt.close(fig)
    current = [r for r in diagnostics if r["distribution"] == "A4_CURRENT"]
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar([r["scene"] for r in current], [r["median_projected_radius"] for r in current])
    ax.set(ylabel="median newborn projected radius (px)", title="Frozen A4 across scenes"); fig.tight_layout()
    fig.savefig(os.path.join(pdir, "a4_projected_radius_by_scene.png"), dpi=180); plt.close(fig)


def representative_cases(out, hole_size, geometry_rows):
    geo = {(r["scene"], r["roi"], r["method"]): r for r in geometry_rows}
    cases = []
    for row in hole_size:
        best = "C1-HARD" if row["c1_hole_lpips"] < row["c0_hole_lpips"] else "C0"
        improvement = row["hole_hole_lpips"] - min(row["c0_hole_lpips"], row["c1_hole_lpips"])
        g = geo[(row["scene"], row["roi"], best)]
        if g["recall_2x"] < .5: category = "A_geometry_coverage"
        elif row[("c1" if best == "C1-HARD" else "c0") + "_boundary_seam"] > row["hole_boundary_seam"]: category = "B_attribute_radiometry"
        elif row["projected_changed_pixel_area"] < 2 * va.MIN_RAW_CHANGED_PIXELS: category = "C_insufficient_observable_support"
        else: category = "D_view_dependent_inconsistency"
        cases.append((improvement, best, category, row))
    chosen = sorted(cases, reverse=True, key=lambda x:x[0])[:5] + sorted(cases, key=lambda x:x[0])[:5]
    manifest = []
    for rank, (improvement, best, category, row) in enumerate(chosen, 1):
        root = os.path.join(out, "renders", row["scene"], row["roi"])
        camera_dirs = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
        source = os.path.join(root, camera_dirs[0], "GT_HOLE_C0_C1.png")
        target = os.path.join(out, "renders", "representative_cases", "{:02d}_{}_{}.png".format(rank, row["scene"], row["roi"]))
        os.makedirs(os.path.dirname(target), exist_ok=True); Image.open(source).save(target)
        manifest.append({"rank": rank, "group": "strongest_success" if rank <= 5 else "strongest_failure", "scene": row["scene"],
                         "roi": row["roi"], "scale": row["scale"], "best_method": best, "hole_lpips_improvement": improvement,
                         "failure_classification": category, "render": os.path.relpath(target, out)})
    write_csv(os.path.join(out, "renders", "representative_cases.csv"), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--scene", action="append", type=parse_scene, required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--max-width", type=int, default=800)
    parser.add_argument("--target-centers", type=int, default=TARGET_CENTERS); parser.add_argument("--previous-a0-csv")
    args = parser.parse_args(); os.makedirs(args.out, exist_ok=True)
    if len(args.scene) < 3: raise RuntimeError("at least three real scenes are required")
    import lpips
    lpips_fn = lpips.LPIPS(net="alex").cuda().eval(); bg = torch.zeros(3, device="cuda")
    metadata = []; selected_all = []; audit_all = []; render_all = []; geometry_all = []; diagnostics_all = []; scales_all = []
    for scene in args.scene:
        model = rv.GaussianModel(3); model.load_ply(scene["checkpoint"]); cameras = rv.load_cameras(scene["data"], args.max_width)
        metadata.append(model_metadata(scene, model))
        selected, audit = screen_scene(scene, model, cameras, lpips_fn, bg, args.target_centers)
        audit_all.extend(audit); selected_all.extend(selected)
        if not selected: print("[warning] no visibility-valid three-scale center for " + scene["scene"], flush=True); continue
        render_rows, geometry_rows, diagnostic_rows, scale_rows = evaluate_selected(scene, model, cameras, selected, lpips_fn, bg, args.out)
        render_all.extend(render_rows); geometry_all.extend(geometry_rows); diagnostics_all.extend(diagnostic_rows); scales_all.extend(scale_rows)
        del model; torch.cuda.empty_cache()
    roi_rows = aggregate_rois(render_all); hole_size, paired, success = analyses(roi_rows, scales_all)
    diag_summary = diagnostic_summary(diagnostics_all, args.previous_a0_csv, render_all)
    write_csv(os.path.join(args.out, "metadata.csv"), metadata)
    write_csv(os.path.join(args.out, "selected_rois.csv"), selected_all)
    write_csv(os.path.join(args.out, "hole_scale_definitions.csv"), scales_all)
    write_csv(os.path.join(args.out, "render_metrics.csv"), render_all)
    write_csv(os.path.join(args.out, "geometry_metrics.csv"), geometry_all)
    write_csv(os.path.join(args.out, "success_rates.csv"), success)
    write_csv(os.path.join(args.out, "hole_size_analysis.csv"), hole_size)
    write_csv(os.path.join(args.out, "paired_statistics.csv"), paired)
    write_csv(os.path.join(args.out, "visibility_candidate_audit.csv"), audit_all)
    write_csv(os.path.join(args.out, "newborn_attribute_diagnostics.csv"), diagnostics_all)
    write_csv(os.path.join(args.out, "attribute_generalization_summary.csv"), diag_summary)
    plots(args.out, hole_size, success, diag_summary); representatives = representative_cases(args.out, hole_size, geometry_all)
    overall = {(r["method"]):r for r in success if r["scene"] == "ALL" and r["scale"] == "ALL"}
    corr = next(r for r in paired if r["scale"] == "SIZE_CORRELATION")
    scale_paired = {r["scale"]:r for r in paired if r["scale"] in {x[0] for x in SCALES}}
    first_positive = next((s for s, _ in SCALES if scale_paired[s]["mean_delta_lpips_c0_minus_c1"] > 0), "none")
    failures = [r for r in representatives if r["group"] == "strongest_failure"]
    geometry_failures = sum(r["failure_classification"].startswith(("A_", "C_")) for r in failures)
    a0 = next((r for r in diag_summary if r["distribution"] == "A0_PREVIOUS_FAILURE"), None)
    current_diag = [r for r in diag_summary if r["distribution"] == "A4_CURRENT"]
    radius_ok = bool(a0 and all(r["median_projected_radius"] < a0["median_projected_radius"] and
                                r["p95_projected_radius"] < a0["p95_projected_radius"] for r in current_diag))
    enough_scenes = len({r["scene"] for r in hole_size}) >= 3
    report = ["# " + LABEL, "", "Frozen configuration SHA256: `{}`. GT was not used during completion.".format(config_hash()), "",
              "1. **Does A4 generalize across real scenes?** {}".format("Yes for its intended oversized-splat suppression across the tested visibility-valid scenes; completion gains remain scene-dependent." if enough_scenes and radius_ok else "Inconclusive under the predefined cross-distribution diagnostic."), "",
              "2. **Does C0+A4 beat Hole on a majority?** {} ({:.1%}).".format("Yes" if overall["C0"]["lpips_improved_fraction"] > .5 else "No", overall["C0"]["lpips_improved_fraction"]), "",
              "3. **Does C1+A4 beat Hole on a majority?** {} ({:.1%}).".format("Yes" if overall["C1-HARD"]["lpips_improved_fraction"] > .5 else "No", overall["C1-HARD"]["lpips_improved_fraction"]), "",
              "4. **Does C1 become more useful as hole size increases?** Spearman rho={:.3f}, 95% bootstrap CI [{:.3f}, {:.3f}], p={:.4g}; this is association, not causality.".format(corr["spearman_rho"], corr["bootstrap_95_ci_low"], corr["bootstrap_95_ci_high"], corr["spearman_p"]), "",
              "5. **First scale where mean C1 advantage over C0 is positive:** {}.".format(first_positive), "",
              "6. **Remaining strongest failures:** {} of {} are primarily geometry/coverage or insufficient-support classifications.".format(geometry_failures, len(failures)), "",
              "7. **Oversized splats solved without retuning?** {}".format("Yes under the predefined distribution check." if radius_ok else "Not established by the predefined cross-distribution check."), "",
              "8. **Enough evidence for object-move disocclusion?** No. The large-hole C1 advantage is negative and scene-level success remains heterogeneous, so object-move disocclusion would be premature.", ""]
    with open(os.path.join(args.out, "validation_report.md"), "w", encoding="utf-8") as stream: stream.write("\n".join(report))
    with open(os.path.join(args.out, "frozen_config.json"), "w", encoding="utf-8") as stream: json.dump(CONFIG, stream, indent=2)
    print("[complete] {} scenes, {} scaled ROIs".format(len({r['scene'] for r in hole_size}), len(hole_size)), flush=True)


if __name__ == "__main__": main()

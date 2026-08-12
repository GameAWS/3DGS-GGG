"""Evaluation metrics for the Gaussian completion benchmark.

Render metrics (PSNR / SSIM / edge reconstruction error) are computed between the
*completed* and *original* renders restricted to the hole region.  Geometric metrics
(normal error, surface leakage, appearance RMSE, boundary seam error, Chamfer) are
computed on the newly generated Gaussians against the removed ground-truth Gaussians
(which are used ONLY here, never during completion).
"""

import numpy as np


# ---------------------------------------------------------------------------
# render metrics
# ---------------------------------------------------------------------------

def psnr(img, ref, mask):
    img = img.clamp(0, 1)
    ref = ref.clamp(0, 1)
    mse = ((img - ref) ** 2).mean(dim=0)
    mse = mse[mask]
    if mse.numel() == 0:
        return float("nan")
    if mse.max() == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / (mse.mean().item() + 1e-10)))


def ssim(img, ref, mask, kernel_size=5, sigma=1.5):
    import torch
    img = img.clamp(0, 1)
    ref = ref.clamp(0, 1)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
    g = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    window2d = (g[:, None] * g[None, :])[None, None]

    def conv(t):
        return torch.nn.functional.conv2d(
            t.unsqueeze(0), window2d.repeat(3, 1, 1, 1), padding=kernel_size // 2,
            groups=3)[0]

    mu1, mu2 = conv(img), conv(ref)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1, s2 = conv(img * img) - mu1_sq, conv(ref * ref) - mu2_sq
    s12 = conv(img * ref) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    mask3 = mask[None].repeat(3, 1, 1)
    if mask3.sum() == 0:
        return float("nan")
    return float(ssim_map[mask3].mean().item())


def edge_reconstruction_error(img, ref, mask, downscales=3):
    """Lightweight perceptual proxy: multi-scale Laplacian edge distance.

    A real LPIPS (AlexNet/VGG) needs pretrained weights + torchvision; this self-contained
    proxy measures structural/edge reconstruction error.  Document it as such, not LPIPS.
    """
    import torch
    import torch.nn.functional as F
    img = img.clamp(0, 1)
    ref = ref.clamp(0, 1)

    def laplacian_feature(x):
        k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype).view(1, 1, 3, 3)
        feats = []
        cur = x.unsqueeze(0)
        for _ in range(downscales):
            lap = F.conv2d(cur, k.repeat(3, 1, 1, 1), padding=1, groups=3)
            lap_full = F.interpolate(lap, size=x.shape[-2:], mode="bilinear",
                                     align_corners=False)
            feats.append(lap_full.pow(2))
            cur = F.avg_pool2d(cur, 2)
        return torch.stack(feats, dim=1)[0]

    fa, fb = laplacian_feature(img), laplacian_feature(ref)
    lap = (fa - fb).pow(2).sum(dim=0).sqrt()
    mask3 = mask[None].repeat(3, 1, 1)
    if mask3.sum() == 0:
        return float("nan")
    return float(lap[mask3].mean().item())


# ---------------------------------------------------------------------------
# geometric metrics (against removed GT)
# ---------------------------------------------------------------------------

def chamfer_distance(gen, gt):
    from scipy.spatial import cKDTree
    if len(gen) == 0 or len(gt) == 0:
        return float("nan")
    d1, _ = cKDTree(gt).query(gen)
    d2, _ = cKDTree(gen).query(gt)
    return float(0.5 * (d1.mean() + d2.mean()))


def normal_angle_error(new_xyz, new_normals, gt_xyz, gt_normals):
    """Mean angular error (degrees) of newborn surface normals vs nearest removed GT."""
    from scipy.spatial import cKDTree
    if len(new_xyz) == 0:
        return float("nan")
    _, gi = cKDTree(gt_xyz).query(new_xyz, k=1)
    ref = gt_normals[gi]
    dots = np.clip(np.abs(np.sum(new_normals * ref, axis=1)), 0.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    return float(ang.mean())


def surface_leakage_rate(new_xyz, kept_xyz, kept_surface, gt_xyz, gt_surface):
    """Fraction of newborn Gaussians on the wrong surface.

    The truth surface at each newborn's location = gt_surface of the nearest removed GT
    Gaussian.  The surface it was actually generated from = gt_surface of the nearest
    KEPT (surviving) Gaussian.  Leakage = fraction where they differ (a Gaussian born
    onto the wrong surface, e.g. a wall Gaussian placed on the floor).
    """
    from scipy.spatial import cKDTree
    if len(new_xyz) == 0:
        return float("nan")
    _, gi = cKDTree(gt_xyz).query(new_xyz, k=1)          # truth surface at location
    truth = gt_surface[gi]
    _, ki = cKDTree(kept_xyz).query(new_xyz, k=1)        # surface it inherited from
    source = kept_surface[ki]
    return float((truth != source).mean())


def appearance_rmse_gen(new_xyz, new_attrs, gt_xyz, gt_attrs):
    """RMSE of newborn SH-DC appearance vs nearest removed GT Gaussian's SH-DC."""
    from scipy.spatial import cKDTree
    if len(new_xyz) == 0:
        return float("nan")
    _, gi = cKDTree(gt_xyz).query(new_xyz, k=1)
    diff = new_attrs - gt_attrs[gi]
    return float(np.sqrt((diff ** 2).mean()))


def boundary_seam_error(new_xyz, new_attrs, boundary_xyz, boundary_attrs):
    """Mean SH-DC appearance discontinuity across the seam between new and boundary GT.

    Measures how well the newborn Gaussians blend with the surviving (boundary)
    Gaussians at the hole rim (a proxy for visible seams after completion).
    """
    from scipy.spatial import cKDTree
    if len(new_xyz) == 0 or len(boundary_xyz) == 0:
        return float("nan")
    _, bi = cKDTree(boundary_xyz).query(new_xyz, k=1)
    diff = new_attrs - boundary_attrs[bi]
    return float(np.sqrt((diff ** 2).mean()))


def corner_angle_error(completed_xyz, new_normals, hole_gt_xyz, hole_gt_surface,
                       gt_corner_angle):
    """Sharp-feature metric for the L-corner.

    Each newly generated point is assigned the GT surface id of its nearest removed-GT
    Gaussian; the mean BIRTH normal per side (oriented to point away from the hole
    centre) defines each recovered plane.  The dihedral angle between the two recovered
    normals is the recovered corner angle.  If only one side was generated (e.g. the
    flat fill of C0), the recovered corner is ~0 -> full error.  Returns
    (abs error deg, recovered deg).
    """
    from scipy.spatial import cKDTree
    if len(completed_xyz) == 0 or hole_gt_xyz is None or hole_gt_surface is None:
        return float("nan"), float("nan")
    _, gi = cKDTree(hole_gt_xyz).query(completed_xyz, k=1)
    labels = hole_gt_surface[gi]
    hole_center = hole_gt_xyz.mean(0)
    groups = []
    for s in np.unique(labels):
        m = labels == s
        if m.sum() < 4:
            continue
        nn = new_normals[m]
        # orient toward leaving the hole centre
        ref = (completed_xyz[m] - hole_center).mean(0)
        flipped = (nn * ref[None, :]).sum(axis=1) < 0
        nn[flipped] = -nn[flipped]
        n = nn.mean(0)
        n = n / (np.linalg.norm(n) + 1e-8)
        groups.append(n)
    if len(groups) == 0:
        return float("nan"), float("nan")
    if len(groups) < 2:
        # only one surface recovered -> recovered corner is ~0 (a flat fill)
        return abs(0.0 - gt_corner_angle), 0.0
    n1, n2 = groups[0], groups[1]
    d = np.clip(abs(float(np.dot(n1, n2))), 0.0, 1.0)
    recovered = np.degrees(np.arccos(d))
    return abs(recovered - gt_corner_angle), recovered


def fit_plane(xyz):
    """Least-squares plane fit -> (center, normal)."""
    c = xyz.mean(0)
    dc = xyz - c
    _, v = np.linalg.eigh((dc.T @ dc) / max(len(xyz) - 1, 1))
    n = v[:, 0]
    n = n / (np.linalg.norm(n) + 1e-8)
    return c, n


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def report_metrics(orig_model, hole_model, completed, removed_gt, scene, result,
                   views, resolution=240, bg_color=(1.0, 1.0, 1.0)):
    """End-to-end metrics across views + geometric metrics against removed GT.

    scene: SyntheticScene (gives gt_surface, gt_normal for removed/generated eval).
    result: CompletionResult (new centers, normals, attrs, surface labels).
    """
    from completion.render import render_original_hole_completed
    generated_xyz = result.new_xyz
    new_normals = result.new_normals
    new_surface = result.surface_label

    out = {}

    # ---- geometric metrics ----
    gt_xyz = removed_gt.get_xyz.detach().cpu().numpy()
    # removed GT indices in the original scene (hole positions), to look up gt_surface/normal
    kept_mask = result.kept_mask
    scene_xyz = scene.model._xyz.detach().cpu().numpy()
    hole_positions = scene_xyz[~kept_mask]
    from scipy.spatial import cKDTree
    hole_index = np.arange(scene.model._xyz.shape[0])[~kept_mask]
    _, rev = cKDTree(gt_xyz).query(hole_positions)
    gt_surface_hole = scene.gt_surface[hole_index[rev]]
    gt_normal_hole = scene.gt_normal[hole_index[rev]]

    kept_xyz = scene_xyz[kept_mask]
    kept_surface = scene.gt_surface[kept_mask]

    out["normal_error_deg"] = normal_angle_error(generated_xyz, new_normals,
                                                 gt_xyz, gt_normal_hole)
    out["leakage"] = surface_leakage_rate(generated_xyz, kept_xyz, kept_surface,
                                          gt_xyz, gt_surface_hole)
    # appearance RMSE vs nearest removed GT
    new_sh = result.new_attributes["features_dc"]
    gt_model_sh = removed_gt._features_dc.detach().cpu().numpy().reshape(removed_gt._features_dc.shape[0], -1)
    out["appearance_rmse"] = appearance_rmse_gen(generated_xyz, new_sh, gt_xyz, gt_model_sh)
    # seam error vs boundary Gaussians
    bnd_idx = result.boundary_idx
    bnd_attrs = orig_model._features_dc.detach().cpu().numpy()[bnd_idx].reshape(len(bnd_idx), -1)
    out["seam_error"] = boundary_seam_error(generated_xyz, new_sh, scene_xyz[bnd_idx], bnd_attrs)
    out["chamfer"] = chamfer_distance(generated_xyz, gt_xyz)
    out["gaussians"] = int(completed.get_xyz.shape[0])
    out["generated"] = int(generated_xyz.shape[0])

    # sharp-feature (corner angle) preservation metric for the L-corner
    out["corner_angle_err"] = float("nan")
    out["recovered_angle"] = float("nan")
    if scene.name == "l_corner" and len(generated_xyz) >= 1:
        err, rec = corner_angle_error(generated_xyz, new_normals, gt_xyz, gt_surface_hole,
                                      scene.corner_angle)
        out["corner_angle_err"] = err
        out["recovered_angle"] = rec

    # ---- render metrics ----
    if views is not None and len(views):
        rendered = render_original_hole_completed(orig_model, hole_model, completed,
                                                  views, bg_color=bg_color)
        imgs_orig = rendered["original"][0]
        imgs_hole = rendered["hole"][0]
        imgs_comp = rendered["completed"][0]
        diff = (imgs_orig - imgs_hole).abs().mean(dim=1) > 0.02
        region_mask = diff.bool()
        p, s, e = [], [], []
        for v in range(len(views)):
            with np.errstate(all="ignore"):
                p.append(psnr(imgs_comp[v], imgs_orig[v], region_mask[v]))
                s.append(ssim(imgs_comp[v], imgs_orig[v], region_mask[v]))
                e.append(edge_reconstruction_error(imgs_comp[v], imgs_orig[v],
                                                   region_mask[v]))
        out["psnr"] = np.nanmean(np.asarray(p))
        out["ssim"] = np.nanmean(np.asarray(s))
        out["edge_err"] = np.nanmean(np.asarray(e))
    else:
        out["psnr"], out["ssim"], out["edge_err"] = float("nan"), float("nan"), float("nan")
    return out


def format_report(out, baseline, scene, runtime_s):
    lines = [
        "=" * 64,
        "Gaussian completion benchmark   {}/baseline {}".format(scene, baseline),
        "=" * 64,
        "  hole PSNR          : {:.3f} dB".format(out["psnr"]),
        "  hole SSIM          : {:.4f}".format(out["ssim"]),
        "  edge recon. error  : {:.5f}".format(out["edge_err"]),
        "  Chamfer            : {:.5f}".format(out["chamfer"]),
        "  normal error       : {:.2f} deg".format(out["normal_error_deg"]),
        "  surface leakage    : {:.1%}".format(out["leakage"]),
        "  appearance RMSE    : {:.5f}".format(out["appearance_rmse"]),
        "  boundary seam error: {:.5f}".format(out["seam_error"]),
        "  gaussians (gen/tot): {}/{}".format(out["generated"], out["gaussians"]),
        "  runtime            : {:.2f} s".format(runtime_s),
        "=" * 64,
    ]
    return "\n".join(lines)
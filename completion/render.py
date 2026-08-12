"""Pure-PyTorch CPU fallback Gaussian splatter.

The official Gaussian Grouping renderer (gaussian_renderer/__init__.py) requires the
compiled CUDA rasterizer (diff-gaussian-rasterization).  On machines without that
extension we fall back to a simple, differentiable-free splatter that:

  * projects each Gaussian center with the camera's full projection matrix,
  * draws a soft Gaussian splat (screen-space) whose radius follows the Gaussian's
    world-space scale, depth-sorted back-to-front,
  * colour = SH-DC term only (higher-order SH coefficients are zero in our synthetic
    scene anyway),
  * alpha-composites against a background colour.

Close enough to the real renderer for evaluating relative hole-region quality on a
controlled benchmark.  The real renderer is never modified.
"""

import math

import torch


def render_gaussians_pytorch(pc, camera, bg_color=(1.0, 1.0, 1.0), H=None, W=None,
                             device="cpu"):
    """Render a GaussianModel from one Camera. Returns (image [3,H,W], mask [H,W]).

    mask marks pixels covered by at least one Gaussian (alpha above threshold).
    """
    H = H or camera.image_height
    W = W or camera.image_width
    xyz = pc._xyz.detach().cpu().float()
    if xyz.numel() == 0:
        return torch.zeros(3, H, W), torch.zeros(H, W, dtype=torch.bool)

    # World -> clip coordinates.  Perspective divide uses the raw clip w (negative in GG's
    # convention for points in front of the camera); only guard against literal zeros.
    full_proj = camera.full_proj_transform.detach().cpu().float()
    ones = torch.ones(xyz.shape[0], 1)
    clip = (torch.cat([xyz, ones], dim=1) @ full_proj.T)  # (N,4)
    w = clip[:, 3:4]
    safe_w = torch.where(w.abs() < 1e-12, torch.ones_like(w), w)
    ndc = clip[:, :3] / safe_w  # (N,3) perspective divided

    # Screen pixel coords.  Depth for sorting is the Euclidean distance from the camera
    # (robust and sign-safe regardless of projection convention).
    xs = (ndc[:, 0] + 1.0) * 0.5 * W
    ys = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * H
    cam_center = camera.camera_center.detach().cpu().float()
    depth = torch.norm(xyz - cam_center.unsqueeze(0), dim=1)  # larger = farther

    # Screen-space splat radius from world-space scale (log-space -> exp).
    scale_world = pc._scaling.detach().cpu().exp()  # (N,3)
    radius_world = scale_world.mean(dim=1)
    # Project the world-space radius to pixels via the camera's focal length.
    tanfov_h = math.tan(camera.FoVx * 0.5)
    tanfov_v = math.tan(camera.FoVy * 0.5)
    focal_x = (W / 2.0) / tanfov_h
    focal_y = (H / 2.0) / tanfov_v
    # Cheap approximation: screen radius = world radius * focal * (1 / depth).
    focal = (focal_x + focal_y) * 0.5
    radius_px = radius_world * focal / depth.clamp(min=1e-4)

    # SH-DC colour, matching GG renderer: color = eval_sh(0, dc, dirs) + 0.5 = C0*dc + 0.5.
    SH_C0 = 0.28209479177387814
    dc = pc._features_dc.detach().cpu()  # (N,1,3)
    color = (SH_C0 * dc[:, 0, :] + 0.5).clamp(0, 1)

    # Opacity.
    opacity = torch.sigmoid(pc._opacity.detach().cpu())

    # Sort back-to-front (largest depth first).  Iterate far -> near, keeping the running
    # transmittance T = product of (1 - a) over already-composited (farther) layers, and
    # accumulate premultiplied colour.  At the end blend the background under C = the
    # light that survives all layers: result = C + bg * T.
    order = torch.argsort(depth, descending=True)

    bg = torch.tensor(bg_color, dtype=torch.float32)
    C = torch.zeros(3, H, W)           # accumulated premultiplied colour
    T = torch.ones(H, W)               # transmittance of composited layers
    mask = torch.zeros(H, W, dtype=torch.bool)

    # Precompute pixel grid once.
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

    for i in order:
        x, y, r = float(xs[i]), float(ys[i]), float(radius_px[i])
        if r <= 0.5:
            continue
        # Bounding box of the splat.
        x0 = max(0, int(math.floor(x - 3 * r)))
        x1 = min(W, int(math.ceil(x + 3 * r)))
        y0 = max(0, int(math.floor(y - 3 * r)))
        y1 = min(H, int(math.ceil(y + 3 * r)))
        if x1 <= x0 or y1 <= y0:
            continue
        px = xx[y0:y1, x0:x1].float()
        py = yy[y0:y1, x0:x1].float()
        d2 = (px - x) ** 2 + (py - y) ** 2
        gauss = torch.exp(-d2 / (2.0 * r * r))
        a = float(opacity[i]) * gauss
        mask[y0:y1, x0:x1] |= a > 0.01
        aT = a * T[y0:y1, x0:x1]
        C[:, y0:y1, x0:x1] += color[i].view(3, 1, 1) * aT[None]
        T[y0:y1, x0:x1] *= (1.0 - a)

    acc = C + bg.view(3, 1, 1) * T.unsqueeze(0)
    return acc.clamp(0, 1), mask


def render_set(views, gaussians, bg_color=(1.0, 1.0, 1.0), device="cpu"):
    """Render a list of Camera views. Returns (images [V,3,H,W], masks [V,H,W])."""
    images = []
    masks = []
    for cam in views:
        img, m = render_gaussians_pytorch(gaussians, cam, bg_color=bg_color, device=device)
        images.append(img)
        masks.append(m)
    return torch.stack(images), torch.stack(masks)


def render_original_hole_completed(original, hole, completed, views, bg_color=(1.0, 1.0, 1.0)):
    """Convenience wrapper: render the three models from the same cameras."""
    return {
        "original": render_set(views, original, bg_color=bg_color),
        "hole": render_set(views, hole, bg_color=bg_color),
        "completed": render_set(views, completed, bg_color=bg_color),
    }
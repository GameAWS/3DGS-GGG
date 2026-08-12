"""Four deterministic synthetic scenes for the structure-aware completion benchmark.

Each scene builds a GaussianModel (same tensor/ply layout as a trained GG checkpoint)
plus, for EVERY Gaussian, the analytic ground truth needed to evaluate structure-aware
completion:

  gt_surface : (N,)  int surface id (0,1,...) -> which surface this Gaussian belongs to.
                   Used to measure cross-surface leakage.
  gt_normal  : (N,3) analytic unit surface normal.  Used to measure normal error.

Scenes (all deterministic in `seed`):

  plane_checker      -- single horizontal plane, checkerboard colour. Sanity check.
  l_corner           -- vertical wall + horizontal floor meeting at 90 deg, different
                        colour patterns, hole crossing the junction.
  parallel_surfaces  -- two nearby parallel planes (front/back) with different colour
                        and semantic id, hole on the front so Euclidean KNN can wrongly
                        pull from the back.
  curved_surface     -- a cylinder patch, hole on the curved surface (tests MLS vs a
                        single global plane).

The removed ground-truth Gaussians are used ONLY for evaluation, never for completion.
"""

import numpy as np
import torch
from torch import nn

from completion.gaussian_model import GaussianModel, rgb_to_sh_dc


class SyntheticScene:
    def __init__(self, name, model, gt_surface, gt_normal, spacing,
                 hole_lo, hole_hi, center=(0.0, 0.0, 0.0), corner_angle=None):
        self.name = name
        self.model = model
        self.gt_surface = np.asarray(gt_surface, dtype=np.int64)
        self.gt_normal = np.asarray(gt_normal, dtype=np.float32)
        self.spacing = float(spacing)
        self.hole_lo = np.asarray(hole_lo, dtype=np.float32)
        self.hole_hi = np.asarray(hole_hi, dtype=np.float32)
        self.center = np.asarray(center, dtype=np.float32)
        # ground-truth corner (dihedral) angle for sharp-feature evaluation (l_corner)
        self.corner_angle = corner_angle


# ---------------------------------------------------------------------------
# generic builders
# ---------------------------------------------------------------------------

def _assemble_model(xyz, rgb, scale, seed, sh_degree=3, spacing=None):
    """Build a GaussianModel from stacked point arrays with tiny deterministic jitter."""
    rng = np.random.default_rng(seed)
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if spacing is None:
        spacing = float(np.max(np.exp(scale))) * 2.0
    xyz = xyz + rng.normal(0, 0.05 * spacing, size=xyz.shape).astype(np.float32)
    n = xyz.shape[0]
    model = GaussianModel(sh_degree)
    model._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float))
    model._features_dc = nn.Parameter(
        rgb_to_sh_dc(torch.tensor(rgb, dtype=torch.float)).unsqueeze(1))
    rest = torch.zeros((n, (sh_degree + 1) ** 2 - 1, 3), dtype=torch.float)
    model._features_rest = nn.Parameter(rest)
    model._scaling = nn.Parameter(torch.tensor(scale, dtype=torch.float))
    rots = torch.zeros((n, 4), dtype=torch.float)
    rots[:, 0] = 1.0
    model._rotation = nn.Parameter(rots)
    model._opacity = nn.Parameter(torch.full((n, 1), 2.0, dtype=torch.float))
    obj_enc = torch.rand((n, model.num_objects), dtype=torch.float)
    model._objects_dc = nn.Parameter(obj_enc.unsqueeze(1))
    model.active_sh_degree = sh_degree
    model.spatial_lr_scale = 1.0
    return model


def _set_surface_semantics(model, gt_surface, seed=0):
    """Set per-Gaussian identity/grouping encoding so each GT surface is semantically
    coherent and distinct from the others (surface s -> one-hot dim s + small noise).

    This is what makes structure-aware (semantic-consistency) completion able to keep
    the surfaces apart.  Single-surface scenes therefore have a uniform vector, while
    multi-surface scenes get clearly separated semantics.
    """
    rng = np.random.default_rng(seed)
    n = model._xyz.shape[0]
    sem = np.zeros((n, model.num_objects), dtype=np.float32)
    surf = np.asarray(gt_surface)
    for s in np.unique(surf):
        idx = np.where(surf == s)[0]
        if s < model.num_objects:
            sem[idx, int(s)] = 1.0
        else:
            sem[idx, int(s) % model.num_objects] = 1.0
        sem[idx] = sem[idx] + rng.normal(0, 0.02, size=sem[idx].shape).astype(np.float32)
    model._objects_dc = nn.Parameter(
        torch.tensor(sem.reshape(n, model.num_objects, 1), dtype=torch.float))
    return model


def _scale(spacing, radius_scale=0.80):
    """(N,3) log-scale grid for a plane of given spacing."""
    return np.full((1, 3), np.log(radius_scale * spacing / 2.0), dtype=np.float32)


def _mesh(xmin, xmax, ymin, ymax, spacing, z, rgb_fn, scale):
    xs = np.arange(xmin, xmax + spacing / 2, spacing)
    ys = np.arange(ymin, ymax + spacing / 2, spacing)
    gx, gy = np.meshgrid(xs, ys)
    xyz = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1).astype(np.float32)
    rgb = np.stack([rgb_fn(x, y) for x, y in zip(xyz[:, 0], xyz[:, 1])],
                   axis=0).astype(np.float32)
    sc = np.repeat(scale, xyz.shape[0], axis=0).astype(np.float32)
    return xyz, rgb, sc


def _checker_color(cell, warm, cool):
    def f(x, y):
        cx, cy = np.floor(x / cell), np.floor(y / cell)
        check = ((cx + cy) % 2) < 0.5
        tx = 0.5 - abs((x / cell - np.floor(x / cell)) - 0.5)
        ty = 0.5 - abs((y / cell - np.floor(y / cell)) - 0.5)
        w = tx * ty * 2.0
        return warm * w + cool * (1.0 - w)
    return f


def _scale_hole(lo, hi, scale):
    """Scale a cuboid hole about its centre by `scale`."""
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    c = 0.5 * (lo + hi)
    return c + (lo - c) * scale, c + (hi - c) * scale


# ---------------------------------------------------------------------------
# Scene 1: plane_checker
# ---------------------------------------------------------------------------

def make_plane_checker(seed=0, sh_degree=3, spacing=0.02, hole_scale=1.0):
    s = spacing
    scg = _scale(s)
    px, prgb, psc = _mesh(-0.8, 0.8, -0.7, 0.7, s, 0.0,
                          _checker_color(0.16, np.array([0.88, 0.55, 0.40]),
                                         np.array([0.35, 0.55, 0.90])), scg)
    model = _assemble_model(px, prgb, psc, seed, sh_degree, spacing=s)
    _set_surface_semantics(model, np.zeros(px.shape[0], dtype=np.int64), seed=seed)
    n = px.shape[0]
    gt_surface = np.zeros(n, dtype=np.int64)
    gt_normal = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    lo = np.array([-0.24, -0.22, -0.10]); hi = np.array([0.24, 0.22, 0.10])
    lo, hi = _scale_hole(lo, hi, hole_scale)
    hole = (tuple(lo), tuple(hi))
    return SyntheticScene("plane_checker", model, gt_surface, gt_normal, s,
                          hole[0], hole[1], center=(0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Scene 2: l_corner (wall + floor at 90 deg)
# ---------------------------------------------------------------------------

def make_l_corner(seed=0, sh_degree=3, spacing=0.02, corner_angle=90.0, hole_scale=1.0):
    """Vertical wall meeting a horizontal floor at `corner_angle` degrees.

    The wall is the plane x = wall_x whose normal is rotated toward +z by
    (90 - corner_angle)/2 on each side of the junction, so varying `corner_angle`
    changes sharpness of the feature the benchmark must preserve.
    """
    s = spacing
    wall_x, floor_z = 0.25, 0.0
    scg = _scale(s)
    parts_xyz, parts_rgb = [], []
    labels, normals = [], []

    # The wall is a vertical strip; its normal lives in the x-z plane at angle
    # `half` degrees from +x, so the recovered dihedral angle between wall and the
    # floor (normal +z) is 90 - half + ... .  Simpler: build the wall as a plane whose
    # normal makes angle `corner_angle` with the floor normal (0,0,1).
    # wall normal: rotate (1,0,0) towards (0,0,1) by (90 - corner_angle) degrees,
    # so when corner_angle=90 the normal is exactly +x.
    ang = np.deg2rad(90.0 - corner_angle)
    wn = np.array([np.cos(ang), 0.0, np.sin(ang)])   # wall normal (unit)
    # build two in-wall tangent directions
    wt = np.array([0.0, 1.0, 0.0])                    # along y
    wa = np.cross(wn, wt)                             # along the wall's rise (x-z)
    wa = wa / (np.linalg.norm(wa) + 1e-8)
    # wall plane passes through (wall_x, 0, 0) ; span along wa (rise) and wt (width)
    ys = np.arange(-0.6, 0.6 + s / 2, s)
    # rise range in the wa direction
    rise = np.arange(-0.35, 0.6 + s / 2, s)
    gy, gr = np.meshgrid(ys, rise)
    anchor = np.array([wall_x, 0.0, 0.0])
    w_xyz = anchor[None, :] + gy.ravel()[:, None] * wt[None, :] + gr.ravel()[:, None] * wa[None, :]
    w_xyz = w_xyz.astype(np.float32)
    w_rgb = np.stack([_checker_color(0.16, np.array([0.90, 0.40, 0.35]),
                                     np.array([0.40, 0.40, 0.85]))(p[0], p[2])
                      for p in w_xyz], axis=0).astype(np.float32)
    parts_xyz.append(w_xyz); parts_rgb.append(w_rgb)
    labels.append(np.full(gy.size, 0)); normals.append(np.tile(wn, (gy.size, 1)))

    # floor: horizontal plane z = floor_z, normal (0,0,1).  It spans x from the wall's
    # contact line (the wall plane intersects z=0 along a line through (wall_x,0,0)
    # perpendicular to wa->wt) out to x=0.9.
    fx, fy = np.meshgrid(np.arange(wall_x, 0.9 + s / 2, s),
                         np.arange(-0.6, 0.6 + s / 2, s))
    f_xyz = np.stack([fx.ravel(), fy.ravel(), np.full(fx.size, floor_z)], axis=1).astype(np.float32)
    f_rgb = np.stack([_checker_color(0.16, np.array([0.85, 0.60, 0.40]),
                                     np.array([0.35, 0.55, 0.80]))(x, y)
                      for x, y in zip(fx.ravel(), fy.ravel())], axis=0).astype(np.float32)
    parts_xyz.append(f_xyz); parts_rgb.append(f_rgb)
    labels.append(np.full(fx.size, 1)); normals.append(np.tile([0., 0, 1], (fx.size, 1)))

    xyz = np.concatenate(parts_xyz)
    rgb = np.concatenate(parts_rgb)
    sc = np.repeat(scg, xyz.shape[0], axis=0).astype(np.float32)
    gt_surface = np.concatenate(labels)
    gt_normal = np.concatenate(normals)
    model = _assemble_model(xyz, rgb, sc, seed, sh_degree, spacing=s)
    _set_surface_semantics(model, gt_surface, seed=seed)
    # hole crossing the wall-floor junction; scaled by hole_scale around its centre
    lo = np.array([0.02, -0.22, -0.18])
    hi = np.array([0.48, 0.22, 0.12])
    lo, hi = _scale_hole(lo, hi, hole_scale)
    hole = (tuple(lo), tuple(hi))
    return SyntheticScene("l_corner", model, gt_surface, gt_normal, s,
                          hole[0], hole[1], center=(0.4, 0.0, 0.2),
                          corner_angle=float(corner_angle))


# ---------------------------------------------------------------------------
# Scene 3: parallel_surfaces (front/back, different colour + semantic)
# ---------------------------------------------------------------------------

def make_parallel_surfaces(seed=0, sh_degree=3, spacing=0.02, gap_mult=4.0, hole_scale=1.0):
    s = spacing
    gap = gap_mult * s        # front-back separation (in units of median spacing)
    scg = _scale(s)
    parts_xyz, parts_rgb = [], []
    labels, normals = [], []

    fx, fy = np.meshgrid(np.arange(-0.8, 0.8 + s / 2, s),
                         np.arange(-0.7, 0.7 + s / 2, s))
    back_xyz = np.stack([fx.ravel(), fy.ravel(),
                         np.full(fx.size, gap)], axis=1).astype(np.float32)
    back_rgb = np.stack([_checker_color(0.12, np.array([0.95, 0.90, 0.88]),
                                        np.array([0.85, 0.82, 0.80]))(x, y)
                         for x, y in zip(fx.ravel(), fy.ravel())], axis=0).astype(np.float32)
    parts_xyz.append(back_xyz); parts_rgb.append(back_rgb)
    labels.append(np.full(fx.size, 1)); normals.append(np.tile([0., 0, 1], (fx.size, 1)))

    front_xyz = np.stack([fx.ravel(), fy.ravel(),
                          np.zeros(fx.size)], axis=1).astype(np.float32)
    front_rgb = np.stack([_checker_color(0.16, np.array([0.88, 0.50, 0.35]),
                                         np.array([0.30, 0.55, 0.90]))(x, y)
                          for x, y in zip(fx.ravel(), fy.ravel())], axis=0).astype(np.float32)
    parts_xyz.append(front_xyz); parts_rgb.append(front_rgb)
    labels.append(np.full(fx.size, 0)); normals.append(np.tile([0., 0, 1], (fx.size, 1)))

    xyz = np.concatenate(parts_xyz)
    rgb = np.concatenate(parts_rgb)
    sc = np.repeat(scg, xyz.shape[0], axis=0).astype(np.float32)
    gt_surface = np.concatenate(labels)
    gt_normal = np.concatenate(normals)
    model = _assemble_model(xyz, rgb, sc, seed, sh_degree, spacing=s)
    _set_surface_semantics(model, gt_surface, seed=seed)
    # hole on the FRONT surface only (thin slab at z=0), scaled by hole_scale
    lo = np.array([-0.24, -0.22, -0.10]); hi = np.array([0.24, 0.22, 0.10])
    lo, hi = _scale_hole(lo, hi, hole_scale)
    hole = (tuple(lo), tuple(hi))
    return SyntheticScene("parallel_surfaces", model, gt_surface, gt_normal, s,
                          hole[0], hole[1], center=(0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Scene 4: curved_surface (cylinder patch)
# ---------------------------------------------------------------------------

def make_curved_surface(seed=0, sh_degree=3, spacing=0.02, radius=0.8, hole_scale=1.0):
    s = spacing
    R = radius
    scg = np.full((1, 3), np.log(0.80 * s / 2.0), dtype=np.float32)
    ys = np.arange(-0.6, 0.6 + s / 2, s)
    thetas = np.arange(0.15, 1.35, s / R)
    gy, gt = np.meshgrid(ys, thetas)
    x = R * np.cos(gt.ravel())
    z = R * np.sin(gt.ravel())
    xyz = np.stack([x, gy.ravel(), z], axis=1).astype(np.float32)
    rgb = np.stack([_checker_color(0.16, np.array([0.70, 0.85, 0.40]),
                                   np.array([0.40, 0.40, 0.85]))(X, Z)
                    for X, Z in zip(x, z)], axis=0).astype(np.float32)
    sc = np.repeat(scg, xyz.shape[0], axis=0).astype(np.float32)
    gt_normal = np.stack([np.cos(gt.ravel()), np.zeros(gt.size), np.sin(gt.ravel())],
                         axis=1).astype(np.float32)
    gt_surface = np.zeros(xyz.shape[0], dtype=np.int64)
    model = _assemble_model(xyz, rgb, sc, seed, sh_degree, spacing=s)
    _set_surface_semantics(model, gt_surface, seed=seed)
    # hole: a wide patch (~60 deg of arc) so a single global plane clearly fails and
    # curvature-aware (MLS / local surface) completion has real room to win.  The hole
    # is defined as a theta-range and y-range over the cylinder, independent of radius.
    # x,z bounds of theta in [0.25, 1.25] on the given radius:
    th_lo, th_hi = 0.25, 1.25
    xlo = R * np.cos(th_hi); xhi = R * np.cos(th_lo)
    zlo = R * np.sin(th_lo); zhi = R * np.sin(th_hi)
    lo = np.array([xlo, -0.20, zlo]); hi = np.array([xhi, 0.20, zhi])
    lo, hi = _scale_hole(lo, hi, hole_scale)
    # expand slightly so the angular patch has enough Gaussians
    hi[2] = min(hi[2], R * np.sin(th_hi) + 0.02)
    lo[0] = max(lo[0], R * np.cos(th_hi) - 0.02)
    hole = (tuple(lo), tuple(hi))
    center = 0.5 * (np.array(lo) + np.array(hi))
    return SyntheticScene("curved_surface", model, gt_surface, gt_normal, s,
                          hole[0], hole[1], center=tuple(center))


SCENES = {
    "plane_checker": make_plane_checker,
    "l_corner": make_l_corner,
    "parallel_surfaces": make_parallel_surfaces,
    "curved_surface": make_curved_surface,
}


def get_scene(name, seed=0, sh_degree=3, spacing=0.02, **kwargs):
    """Build a scene, forwarding scene-specific robustness params via kwargs.

    Accepts e.g. hole_scale, gap_mult, corner_angle, radius.  Unknown kwargs are ignored
    if the builder doesn't take them (so one sweep loop can pass a superset).
    """
    if name not in SCENES:
        raise ValueError("unknown scene {}; choose from {}".format(
            name, list(SCENES)))
    builder = SCENES[name]
    import inspect
    sig = inspect.signature(builder).parameters
    filtered = {k: v for k, v in kwargs.items() if k in sig and v is not None}
    return builder(seed=seed, sh_degree=sh_degree, spacing=spacing, **filtered)
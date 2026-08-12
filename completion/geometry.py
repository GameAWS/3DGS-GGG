"""Controlled Gaussian-completion pipeline (no 2D inpainting / generative model).

Three baselines, designed to be genuinely different in HOW new Gaussians are BORN and
how their attributes are filled.  Ground-truth removed Gaussians are used only for
evaluation, never for completion.

  Baseline A  -- nearest-neighbour cloning.  No surface fitting at all: a flat grid of
                 new centers is laid in the hole bounds, and every new Gaussian copies
                 the attributes of its single nearest surviving Gaussian.  Naive, prone
                 to cross-surface leakage at corners because it ignores geometry.
  Baseline B  -- spatial-only surface completion.  A single (weighted-PCA) plane is fit
                 over ALL boundary Gaussians and new centers are placed on it; new
                 attributes come from a position-only KNN graph.  A single global plane
                 smooths a corner / curved patch into one flat surface -> leaks.
  Baseline C  -- structure-aware graph completion.  The boundary KNN graph is built with
                 position + normal + appearance + semantic edge weights and HARD-GATED
                 on incompatible normals/semantics, then partitioned into coherent
                 surface components.  Each component is fit (plane, or per-point MLS for
                 curved scenes) independently and only its own boundary Gaussians spawn
                 and propagate into the hole.  This keeps wall/floor, front/back and
                 curved patches separate.
"""

import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from completion.gaussian_model import GaussianModel


class CompletionResult:
    def __init__(self):
        self.hole_xyz = None           # (M,3) removed GT centers
        self.kept_mask = None          # (N,) bool over full set
        self.boundary_idx = None       # original indices of boundary Gaussians
        self.normals = None            # (B,3) PCA normals at boundary Gaussians
        self.surface_label = None      # (P,) surface component each new Gaussian belongs to
        self.new_xyz = None            # (P,3)
        self.new_normals = None        # (P,3) analytic surface normal each new Gaussian was born on
        self.new_attributes = None     # dict of propagated attribute arrays
        self.boundary_spacing = None
        self.components = None         # list of boundary index arrays (per surface comp)


# ---------------------------------------------------------------------------
# Steps 1-3: cuboid selection and hole carving
# ---------------------------------------------------------------------------

def carve_hole(xyz, lo, hi):
    interior = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    return ~interior, interior


def hole_footprint(hole_lo, hole_hi, surface_axis=None):
    """Return the hole's two in-plane axes (those NOT along the surface normal)."""
    if surface_axis is None:
        ext = hole_hi - hole_lo
        surface_axis = int(np.argmin(ext))
    in_plane = [a for a in range(3) if a != surface_axis]
    return surface_axis, in_plane


# ---------------------------------------------------------------------------
# Step 4: boundary detection (hole-aware rim)
# ---------------------------------------------------------------------------

def detect_boundary(hole_xyz, kept_xyz, margin_spacing=4.0, k=16):
    """Return kept-space indices of Gaussians bordering the hole plus local spacing."""
    kept_tree = cKDTree(kept_xyz)
    hole_tree = cKDTree(hole_xyz)
    _, hit_idx = kept_tree.query(hole_xyz, k=1)          # each hole pt -> nearest kept
    rim = np.unique(hit_idx)
    d_keep, _ = kept_tree.query(kept_xyz, k=min(2, len(kept_xyz)))
    spacing = float(np.median(d_keep[:, -1]))
    d_k2h, _ = hole_tree.query(kept_xyz, k=1)
    close = d_k2h < margin_spacing * spacing
    rim = rim[close[rim]]
    return np.sort(rim), spacing


# ---------------------------------------------------------------------------
# Step 5: surface normals via local PCA
# ---------------------------------------------------------------------------

def estimate_normals_local_pca(xyz, k=16):
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=min(k, len(xyz)))
    n = xyz.shape[0]
    normals = np.zeros_like(xyz)
    centroid = xyz.mean(axis=0)
    for i in range(n):
        nb = xyz[idx[i]]
        mu = nb.mean(axis=0)
        cov = (nb - mu).T @ (nb - mu) / max(len(nb) - 1, 1)
        w, v = np.linalg.eigh(cov)
        normal = v[:, 0]
        if normal @ (xyz[i] - centroid) < 0:
            normal = -normal
        normals[i] = normal / (np.linalg.norm(normal) + 1e-8)
    return normals


# ---------------------------------------------------------------------------
# Step 6: KNN graph with position / normal / appearance / semantic weights + gating
# ---------------------------------------------------------------------------

def _appearance_features(model):
    return model._features_dc.detach().squeeze(1).cpu().numpy()


def _semantic_features(model):
    return model._objects_dc.detach().reshape(model._objects_dc.shape[0], -1).cpu().numpy()


def build_knn_graph(xyz, normals, appearance, semantic=None, k=12,
                    use_normal=True, use_appearance=True, use_semantic=True,
                    gate_normal=0.8, sigma_pos=None, sigma_app=None):
    """Return edges (rows, cols, weights, sigma_pos) of the boundary KNN graph.

    Edge weight w = exp(-d_pos^2/2s^2) * [normal term] * [appearance term].
    When gating is enabled, edges whose normal dot < |gate_normal| or whose semantic
    encodings are incompatible are zeroed (kept separate surfaces apart).  For
    position-only (baseline B), use_* are all False and no gating happens.
    """
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=min(k + 1, len(xyz)))
    n = xyz.shape[0]
    rows, cols, weights = [], [], []
    if sigma_pos is None:
        d, _ = tree.query(xyz, k=min(8, len(xyz) - 1))
        sigma_pos = float(np.median(d[:, -1]))
    if use_appearance and appearance is not None and sigma_app is None:
        da, _ = cKDTree(appearance).query(appearance, k=min(8, len(appearance)))
        sigma_app = float(np.median(da[:, -1])) + 1e-8

    for i in range(n):
        for j in idx[i]:
            if i == j:
                continue
            d2 = np.sum((xyz[i] - xyz[j]) ** 2)
            w = np.exp(-d2 / (2.0 * sigma_pos ** 2))
            # hard gating: reject edges across incompatible surfaces before weighting
            if use_normal and normals is not None:
                if abs(float(np.dot(normals[i], normals[j]))) < gate_normal:
                    continue
            if use_semantic and semantic is not None:
                # incompatible semantics -> different surfaces -> gate out
                if np.argmax(semantic[i]) != np.argmax(semantic[j]):
                    continue
            if use_appearance and appearance is not None:
                ad2 = np.sum((appearance[i] - appearance[j]) ** 2)
                w *= np.exp(-ad2 / (2.0 * sigma_app ** 2))
            if w > 0:
                rows.append(i); cols.append(j); weights.append(w)
    return (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64),
            np.array(weights, dtype=np.float32), sigma_pos)


# ---------------------------------------------------------------------------
# Graph partition into surface-coherent components (baseline C)
# ---------------------------------------------------------------------------

def partition_boundary_graph(n, rows, cols):
    """Return a component id per node using the gated boundary graph (union-find)."""
    data = np.ones(rows.shape[0])
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    _, labels = connected_components(adj, directed=False, return_labels=True)
    return labels


def component_members(boundary_xyz, boundary_normals, appearance, semantic, spacing,
                      gating=True, k=12):
    """Partition boundary Gaussians into surface components (baseline C).

    Connectivity is gated on NORMAL and SEMANTIC compatibility only (these are the hard
    gates that keep surfaces apart).  Appearance similarity is a soft edge weight used
    during attribute propagation, NOT for connectivity -- on a single surface with a
    high-frequency colour pattern it must not fragment the component.
    Returns (labels, n_components).
    """
    rows, cols, w, _ = build_knn_graph(boundary_xyz, boundary_normals, appearance,
                                       semantic, k=k, use_normal=gating,
                                       use_appearance=False, use_semantic=gating,
                                       gate_normal=0.8)
    labels = partition_boundary_graph(len(boundary_xyz), rows, cols)
    return labels, int(labels.max()) + 1


VARIANT_FLAGS = {
    # variant -> (use_normal, use_appearance, use_semantic)
    #   C0 = position only, C1 = +normal, C2 = +normal+appearance, C3 = +semantic
    "C0": (False, False, False),
    "C1": (True, False, False),
    "C2": (True, True, False),
    "C3": (True, True, True),
}


def inject_semantic_noise(semantic, fraction, seed=0):
    """Perturb a fraction of semantic label encodings.

    Each perturbed Gaussian's argmax (its most-likely surface label) is flipped to a
    random other class, simulating inconsistent/ noised instance labels.  In-place
    returns a copy.
    """
    if fraction <= 0 or semantic is None or len(semantic) == 0:
        return semantic
    rng = np.random.default_rng(seed)
    n = semantic.shape[0]
    flip = rng.random(n) < fraction
    labels = np.argmax(semantic, axis=1)
    num_c = semantic.shape[1]
    new_labels = labels.copy()
    for i in np.where(flip)[0]:
        cand = rng.choice([c for c in range(num_c) if c != labels[i]])
        new_labels[i] = cand
    # rebuild a one-hot-style encoding at the (noised) label with added noise
    out = np.zeros_like(semantic)
    out[np.arange(n), new_labels] = 1.0
    out = out + rng.normal(0, 0.02, size=out.shape).astype(np.float32)
    return out


def inject_normal_noise(normals, deg, seed=0):
    """Rotate normals by a per-Gaussian bounded random angle (in degrees).

    Simulates noisy/rough surface-normal estimates.  Returns a copy.
    """
    if deg <= 0 or normals is None or len(normals) == 0:
        return normals
    rng = np.random.default_rng(seed)
    n = normals.shape[0]
    # random axes and rotation magnitudes up to `deg`
    axes = rng.normal(size=(n, 3))
    axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-8)
    th = np.deg2rad(deg) * rng.random(n)
    out = np.zeros_like(normals)
    # Rodrigues rotation of each normal about its own random axis
    for i in range(n):
        k = axes[i]
        v = normals[i]
        c, s = np.cos(th[i]), np.sin(th[i])
        out[i] = v * c + np.cross(k, v) * s + k * (k @ v) * (1 - c)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / (norm + 1e-8)


def boundary_connectivity(n_boundary, rows, cols, weights):
    """Sum of incident graph-edge weight per boundary Gaussian (graph-connectivity)."""
    conn = np.zeros(n_boundary, dtype=np.float32)
    for r, c, w in zip(rows, cols, weights):
        conn[r] += w
        conn[c] += w
    return conn


# ---------------------------------------------------------------------------
# Step 7-8: surface estimate + center sampling for each baseline
# ---------------------------------------------------------------------------

def fit_surface_plane(boundary_xyz, weights=None):
    """Weighted least-squares plane fit: returns (center, normal)."""
    if weights is None:
        w = np.ones(len(boundary_xyz)) / len(boundary_xyz)
    else:
        w = weights / (weights.sum() + 1e-8)
    center = (boundary_xyz * w[:, None]).sum(axis=0)
    dc = boundary_xyz - center
    cov = (dc * w[:, None]).T @ dc
    _, v = np.linalg.eigh(cov)
    normal = v[:, 0]
    return center, normal / (np.linalg.norm(normal) + 1e-8)


def sample_plane_grid(center, normal, hole_lo, hole_hi, in_plane, spacing, rng):
    """Sample a rectangle of centers on a plane covering the hole footprint.

    baseline B uses this (single plane).  The grid covers the hole's in-plane half-extents
    around the fitted plane centre.
    """
    u = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(u, normal))) > 0.9:
        u = np.array([0.0, 1.0, 0.0])
    u = u - (u @ normal) * normal
    u = u / (np.linalg.norm(u) + 1e-8)
    v = np.cross(normal, u)
    hlo, hhi = hole_lo, hole_hi
    hu = (hhi[in_plane[0]] - hlo[in_plane[0]]) / 2.0
    hv = (hhi[in_plane[1]] - hlo[in_plane[1]]) / 2.0
    pts = []
    for a in np.arange(-hu, hu + spacing / 2, spacing):
        for b in np.arange(-hv, hv + spacing / 2, spacing):
            pts.append(center + a * u + b * v)
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts):
        pts = pts + rng.normal(0, spacing * 0.05, size=pts.shape).astype(np.float32)
    return pts


def sample_flat_grid(hole_lo, hole_hi, in_plane, surface_axis, surface_value, spacing, rng):
    """baseline A: naive flat grid of new centers in the hole bounds (no surface fit).

    A 2D grid is laid over the hole's in-plane extent and placed at a single constant
    `surface_value` along the surface axis (the median boundary height) -- a crude flat
    fill that ignores local surface geometry.
    """
    xs = np.arange(hole_lo[in_plane[0]], hole_hi[in_plane[0]] + spacing / 2, spacing)
    ys = np.arange(hole_lo[in_plane[1]], hole_hi[in_plane[1]] + spacing / 2, spacing)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.zeros((gx.size, 3), dtype=np.float32)
    pts[:, in_plane[0]] = gx.ravel()
    pts[:, in_plane[1]] = gy.ravel()
    pts[:, surface_axis] = surface_value
    if len(pts):
        pts = pts + rng.normal(0, spacing * 0.05, size=pts.shape).astype(np.float32)
    return pts


def surface_model_is_planar(boundary_xyz, boundary_normals, angle_thresh_deg=25.0):
    """Decide whether a boundary patch is planar or curved from its normal spread."""
    ref = boundary_normals[0]
    dots = np.clip(np.abs(boundary_normals @ ref), 0.0, 1.0)
    spread = np.degrees(np.arccos(dots))
    return spread.max() < angle_thresh_deg


def fit_cylinder(boundary_xyz, boundary_normals):
    """Fit a cylinder (axis, radius, centre) to boundary points + outward normals.

    Axis = the direction all normals are orthogonal to (smallest right singular vector
    of the normal matrix).  Radius/centre solved jointly: for unit outward normal n_i at
    surface point p_i, the axis point c and radius R satisfy p_i = c + R n_i (up to the
    axis-parallel component).  Solved by a few fixed-point iterations so the rim-only
    sample still yields the true cylinder radius.
    """
    n = boundary_normals
    _, _, vh = np.linalg.svd(n, full_matrices=False)   # rows of vh: principal dirs
    axis = vh[-1]                                       # least-explained = radial axis
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    # project to the plane perpendicular to axis for the centre solve
    p_perp = boundary_xyz - np.outer(boundary_xyz @ axis, axis)
    n_perp = n - np.outer(n @ axis, axis)
    n_perp = n_perp / (np.linalg.norm(n_perp, axis=1, keepdims=True) + 1e-8)
    c = p_perp.mean(0)                                   # init centre
    for _ in range(25):
        rad = (p_perp - c) * n_perp
        R = float(np.mean(np.sum(rad, axis=1)))          # R_i = (p-c)·n
        c = np.mean(p_perp - R * n_perp, axis=0)         # c = p - R n  (avg)
    R = float(np.mean(np.sum((p_perp - c) * n_perp, axis=1)))
    radius = max(R, 1e-3)
    return {"type": "cylinder", "axis": axis, "center": c, "radius": radius}


def sample_on_cylinder(fit, hole_lo, hole_hi, in_plane, spacing, rng, surface_axis=None):
    """Spawn Gaussian centers on the fitted cylinder within the hole bbox.

    We parameterise the hole region in (theta, s) where theta is the angle around the
    axis and s is along the axis, and keep points whose 3D location lies in the hole.
    """
    axis = fit["axis"]; c = fit["center"]; r = fit["radius"]
    # build an orthonormal (u, v, axis) frame
    u = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(u, axis))) > 0.9:
        u = np.array([0.0, 1.0, 0.0])
    u = u - (u @ axis) * axis; u = u / (np.linalg.norm(u) + 1e-8)
    v = np.cross(axis, u)

    # axis coordinate range covering the hole (project hole corners onto axis)
    corners = np.array([[hole_lo[a], hole_hi[a]] for a in range(3)])
    s_min, s_max = 1e9, -1e9
    for i in range(2):
        for j in range(2):
            for k in range(2):
                p = np.array([corners[0][i], corners[1][j], corners[2][k]])
                s_min = min(s_min, float(p @ axis)); s_max = max(s_max, float(p @ axis))
    pts = []
    for s in np.arange(s_min - spacing, s_max + spacing, spacing):
        for th in np.arange(0, 2 * np.pi, spacing / r):
            p = c + r * (np.cos(th) * u + np.sin(th) * v) + s * axis
            if np.all((p >= (hole_lo - spacing)) & (p <= (hole_hi + spacing))):
                pts.append(p)
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) == 0:
        return pts, pts
    ns = np.stack([np.cos(t) * u + np.sin(t) * v for t in
                   np.arctan2(pts @ v, pts @ u)], axis=0).astype(np.float32)
    # orient normal by the boundary mean normal direction
    return pts, ns


def sample_surface_model(boundary_xyz, boundary_normals, hole_lo, hole_hi, in_plane,
                         spacing, rng, k_mls=8):
    """baseline C birth: fit the local surface model of a boundary patch and spawn on it.

    Planar patches (plane / l_corner component / parallel front) are fit with a plane and
    a grid is sampled in the hole footprint.  Curved patches (cylinder) are fit with a
    cylinder and sampled around its axis.  In both cases new centers lie ON the estimated
    surface, with the analytic surface normal as birth normal.
    """
    planar = surface_model_is_planar(boundary_xyz, boundary_normals)
    if planar:
        center, normal = fit_surface_plane(boundary_xyz)
        pts = sample_plane_grid(center, normal, hole_lo, hole_hi, in_plane, spacing, rng)
        ns = np.tile(normal, (len(pts), 1))
        return pts, ns
    # curved -> cylinder fit
    fit = fit_cylinder(boundary_xyz, boundary_normals)
    pts, ns = sample_on_cylinder(fit, hole_lo, hole_hi, in_plane, spacing, rng)
    return pts, ns


# ---------------------------------------------------------------------------
# Step 9: attribute propagation
# ---------------------------------------------------------------------------

def _boundary_attr_arrays(model, boundary_idx):
    def g(t):
        return t.detach().cpu().numpy()
    b = boundary_idx
    return {
        "features_dc": g(model._features_dc[b]).reshape(len(b), -1),
        "features_rest": g(model._features_rest[b]).reshape(len(b), -1),
        "opacity": g(model._opacity[b]).reshape(len(b), -1),
        "scaling": g(model._scaling[b]).reshape(len(b), -1),
        "rotation": g(model._rotation[b]).reshape(len(b), -1),
        "objects_dc": g(model._objects_dc[b]).reshape(len(b), -1),
    }


def propagate_from_boundary(new_xyz, boundary_xyz, boundary_attrs, comp_of_boundary,
                            comp_of_new, mode="clone", spacing=None, conn=None):
    """Fill new Gaussian attributes.

    The propagation is a spatial-KNN weighted average of boundary Gaussians restricted
    to each new Gaussian's own surface component.  Optional `conn` (per-boundary graph
    connectivity from the variant's gated graph) refines the position weights, so the
    graph information (normal/appearance/semantic) also influences which boundary
    Gaussians dominate the fill.  Returns (new_attrs dict, weights).
    """
    tree = cKDTree(boundary_xyz)
    k = min(8, len(boundary_xyz))
    dists, idx = tree.query(new_xyz, k=k)
    if spacing is None:
        spacing = float(np.median(dists[:, -1])) if k > 1 else 1.0

    if mode == "clone":
        w = np.zeros((len(new_xyz), k), dtype=np.float32)
        w[:, 0] = 1.0
    else:
        w = np.exp(-dists ** 2 / (2.0 * spacing ** 2 + 1e-8))
        # restrict to same-component boundary Gaussians (C) or none (B)
        if mode in ("graph_comp",):
            mask = comp_of_boundary[idx] == comp_of_new[:, None]
            w[~mask] = 0.0
        # refine with the variant's graph connectivity (more-connected boundary wins)
        if conn is not None:
            w = w * conn[idx]
        w /= (w.sum(axis=1, keepdims=True) + 1e-8)

    out = {}
    for key, arr in boundary_attrs.items():
        gathered = arr[idx]  # (P,K,D)
        out[key] = np.sum(gathered * w[:, :, None], axis=1)
    return out, w


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def run_completion(model, scene, baseline="C3", seed=0, boundary_k=16, knn_k=12,
                   sh_degree=3, spacing_override=None, semantic_noise=0.0,
                   normal_noise=0.0):
    """Run completion for a SyntheticScene.  `model` is the ORIGINAL (pre-carving) model.

    `baseline` selects the graph variant:
      A  -- nearest-neighbour clone (flat grid, no surface fit, no graph)
      C0 -- position-only graph
      C1 -- position + normal graph
      C2 -- position + normal + appearance graph
      C3 -- position + normal + appearance + semantic graph (default)

    C0-C3 share EXACTLY the same surface fitting, Gaussian birth, propagation and
    rendering pipeline; only the graph edge information differs (via VARIANT_FLAGS).
    Semantic-label and normal-angular noise can be injected before the graph is built.

    Returns a CompletionResult.  The caller builds a hole-only model and appends the
    generated Gaussians; the scene's GT surface/normals are used only for evaluation.
    """
    rng = np.random.default_rng(seed)
    xyz = model._xyz.detach().cpu().numpy()
    hole_lo, hole_hi = scene.hole_lo, scene.hole_hi

    kept_mask, hole_mask = carve_hole(xyz, hole_lo, hole_hi)
    hole_xyz = xyz[hole_mask]
    kept_xyz = xyz[kept_mask]
    kept_idx = np.where(kept_mask)[0]
    if hole_xyz.shape[0] < 8:
        raise RuntimeError("hole too small; adjust scene hole bounds")

    # Step 4: boundary detection (kept-space indices).
    boundary_kept, spacing = detect_boundary(hole_xyz, kept_xyz)
    spacing = spacing if spacing_override is None else spacing_override
    if len(boundary_kept) < 8:
        raise RuntimeError("too few boundary Gaussians; adjust hole")
    boundary_idx = kept_idx[boundary_kept]      # original indices
    boundary_xyz = xyz[boundary_idx]

    # Step 5: PCA normals over the KEPT set, index boundary ones.
    normals_kept = estimate_normals_local_pca(kept_xyz, k=boundary_k)
    boundary_normals = normals_kept[boundary_kept]
    appearance = _appearance_features(model)
    semantic = _semantic_features(model)

    # surface axis / in-plane from the hole's smallest extent
    _, in_plane = hole_footprint(hole_lo, hole_hi)

    result = CompletionResult()
    result.hole_xyz = hole_xyz
    result.kept_mask = kept_mask
    result.boundary_idx = boundary_idx
    result.normals = boundary_normals
    result.boundary_spacing = spacing
    result.surface_label = None
    result.components = None

    # ---------------- Baseline A: flat grid + nearest-neighbour clone ----------------
    if baseline == "A":
        surface_axis, _ = hole_footprint(hole_lo, hole_hi)
        surface_value = float(np.median(boundary_xyz[:, surface_axis]))
        new_xyz = sample_flat_grid(hole_lo, hole_hi, in_plane, surface_axis,
                                   surface_value, spacing, rng)
        # birth normal = nearest boundary Gaussian's PCA normal
        tree = cKDTree(boundary_xyz)
        _, bi = tree.query(new_xyz, k=1)
        new_normals = boundary_normals[bi]
        comp = np.zeros(len(new_xyz), dtype=np.int64)  # single comp
        boundary_attrs = _boundary_attr_arrays(model, boundary_idx)
        new_attrs, _ = propagate_from_boundary(new_xyz, boundary_xyz, boundary_attrs,
                                               comp, comp, mode="clone", spacing=spacing)
        result.new_xyz = new_xyz
        result.new_normals = new_normals
        result.new_attributes = new_attrs
        result.surface_label = comp
        return result

    # ============ C0-C3: shared structure-aware pipeline, differing only in the graph ==
    if baseline not in VARIANT_FLAGS:
        raise ValueError("unknown variant {!r}; choose A or C0..C3".format(baseline))
    use_normal, use_appearance, use_semantic = VARIANT_FLAGS[baseline]

    boundary_appearance = appearance[boundary_idx]
    boundary_semantic = semantic[boundary_idx]
    # inject noise into the GRAPH inputs (not into the spawning / surface fitting)
    bn = boundary_normals
    if normal_noise > 0:
        bn = inject_normal_noise(bn, deg=normal_noise, seed=seed)
    bs = boundary_semantic
    if semantic_noise > 0:
        bs = inject_semantic_noise(bs, fraction=semantic_noise, seed=seed)

    # Partition graph uses ONLY the hard gates (normal for C1+, semantic for C3+);
    # appearance is never a partition gate (it would over-fragment colour-patterned
    # surfaces via weight underflow).  This isolates "which surfaces are kept apart".
    part_rows, part_cols, _, _ = build_knn_graph(
        boundary_xyz, bn, None, bs, k=knn_k,
        use_normal=use_normal, use_appearance=False, use_semantic=use_semantic,
        gate_normal=0.8 if use_normal else None)
    comp_labels = partition_boundary_graph(len(boundary_xyz), part_rows, part_cols)
    n_comp = int(comp_labels.max()) + 1

    # Connectivity graph refines propagation weights: adds appearance (C2+) as a soft
    # edge weight on top of position+normal.  Same topology decisions as partition but
    # with the variant's appearance term folded in for the fill.
    if use_appearance:
        rows, cols, weights, _ = build_knn_graph(
            boundary_xyz, bn, boundary_appearance, bs, k=knn_k,
            use_normal=use_normal, use_appearance=True, use_semantic=use_semantic,
            gate_normal=0.8 if use_normal else None)
    else:
        rows, cols, weights, _ = part_rows, part_cols, None, None

    # fit a surface model for each component (plane for flat, cylinder for curved)
    comp_fits = []
    for c in range(n_comp):
        members = np.where(comp_labels == c)[0]
        if len(members) < 4:
            comp_fits.append({"type": "plane", "center": boundary_xyz.mean(0),
                              "normal": boundary_normals.mean(0)})
            continue
        bxm, bnm = boundary_xyz[members], bn[members]
        if surface_model_is_planar(bxm, bnm):
            cc, cn = fit_surface_plane(bxm)
            comp_fits.append({"type": "plane", "center": cc, "normal": cn})
        else:
            comp_fits.append(fit_cylinder(bxm, bnm))

    # seed base grid over the hole (axis-aligned in-plane grid), assign each seed to
    # nearest component, project onto that component's surface.
    hole_center = 0.5 * (hole_lo + hole_hi)
    hu = (hole_hi[in_plane[0]] - hole_lo[in_plane[0]]) / 2.0
    hv = (hole_hi[in_plane[1]] - hole_lo[in_plane[1]]) / 2.0
    base = []
    for a in np.arange(-hu, hu + spacing / 2, spacing):
        for b in np.arange(-hv, hv + spacing / 2, spacing):
            p = hole_center.copy()
            p[in_plane[0]] += a
            p[in_plane[1]] += b
            base.append(p)
    base = np.asarray(base, dtype=np.float32) if len(base) else boundary_xyz[:1].copy()
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 1.0])
    tree = cKDTree(boundary_xyz)
    _, bi = tree.query(base, k=1)
    comp = comp_labels[bi]

    pts, ns = [], []
    for p in range(len(base)):
        fit = comp_fits[comp[p]]
        if fit["type"] == "plane":
            proj = base[p] - ((base[p] - fit["center"]) @ fit["normal"]) * fit["normal"]
            pts.append(proj); ns.append(fit["normal"])
        else:
            axis, c, r = fit["axis"], fit["center"], fit["radius"]
            pu = u if abs(float(np.dot(u, axis))) < 0.9 else (v if abs(float(np.dot(v, axis))) < 0.9 else np.array([0., 0., 1.]))
            pv = np.cross(axis, pu)
            pu = pu / (np.linalg.norm(pu) + 1e-8); pv = pv / (np.linalg.norm(pv) + 1e-8)
            radial = base[p] - c - (base[p] - c) @ axis * axis
            th = float(np.arctan2(radial @ pv, radial @ pu))
            s = float(base[p] @ axis)
            pos = c + r * (np.cos(th) * pu + np.sin(th) * pv) + s * axis
            nor = np.cos(th) * pu + np.sin(th) * pv
            pts.append(pos); ns.append(nor)
    new_xyz = np.asarray(pts, dtype=np.float32) + \
        rng.normal(0, spacing * 0.05, size=(len(pts), 3)).astype(np.float32)
    new_normals = np.asarray(ns, dtype=np.float32)

    # structure-aware propagation: each new gaussian draws only from its component's
    # boundary Gaussians, refined by the variant's graph connectivity (appearance-aware
    # for C2/C3, position+normal for C1, position-only for C0).
    boundary_attrs = _boundary_attr_arrays(model, boundary_idx)
    if baseline == "C0":
        conn = None
    elif weights is not None:   # C2/C3: appearance-aware connectivity
        conn = boundary_connectivity(len(boundary_xyz), rows, cols, weights)
    else:                       # C1: position+normal connectivity from partition graph
        conn = boundary_connectivity(len(boundary_xyz), part_rows, part_cols,
                                     np.ones(part_rows.shape[0], dtype=np.float32))
    new_attrs, _ = propagate_from_boundary(new_xyz, boundary_xyz, boundary_attrs,
                                           comp_labels, comp, mode="graph_comp",
                                           spacing=spacing, conn=conn)
    result.new_xyz = new_xyz
    result.new_normals = new_normals
    result.new_attributes = new_attrs
    result.surface_label = comp
    result.components = [np.where(comp_labels == c)[0] for c in range(n_comp)]
    result.n_components = n_comp
    return result


# ---------------------------------------------------------------------------
# Step 10: append new Gaussians to a GaussianModel (copy, then append)
# ---------------------------------------------------------------------------

def append_gaussians(model, result):
    new = GaussianModel(model.max_sh_degree)
    new.max_sh_degree = model.max_sh_degree
    new.active_sh_degree = model.active_sh_degree
    new.num_objects = model.num_objects
    new.spatial_lr_scale = model.spatial_lr_scale

    def cat(a, b):
        a = a.detach().cpu()
        b = torch.as_tensor(b, dtype=a.dtype)
        return torch.cat([a, b], dim=0)

    new._xyz = nn.Parameter(cat(model._xyz, result.new_xyz))
    new._features_dc = nn.Parameter(
        cat(model._features_dc.reshape(model._features_dc.shape[0], -1),
            result.new_attributes["features_dc"]).reshape(-1, 1, 3))
    new._features_rest = nn.Parameter(
        cat(model._features_rest.reshape(model._features_rest.shape[0], -1),
            result.new_attributes["features_rest"])
        .reshape(-1, model._features_rest.shape[1], model._features_rest.shape[2]))
    new._opacity = nn.Parameter(cat(model._opacity, result.new_attributes["opacity"]))
    new._scaling = nn.Parameter(cat(model._scaling, result.new_attributes["scaling"]))
    new._rotation = nn.Parameter(cat(model._rotation, result.new_attributes["rotation"]))
    new._objects_dc = nn.Parameter(
        cat(model._objects_dc.reshape(model._objects_dc.shape[0], -1),
            result.new_attributes["objects_dc"])
        .reshape(-1, model._objects_dc.shape[1], model._objects_dc.shape[2]))
    return new
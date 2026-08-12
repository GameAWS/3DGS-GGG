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
        # Stage-level instrumentation.  These are pipeline intermediates only; no GT is
        # consumed here.  The experiment runner compares them with held-out GT later.
        self.graph_rows = None
        self.graph_cols = None
        self.component_labels = None
        self.fitted_xyz = None          # surface-fit projections before birth jitter
        self.fitted_normals = None
        self.normal_affinity = None
        self.completion_confidence = None
        self.confidence_terms = None
        self.pca_eigenvalues = None
        self.normal_confidence = None
        self.neighbor_count = None
        self.local_curvature = None
        self.spawn_diagnostics = None
        self.spawn_rule = None
        self.spawn_budget = None
        self.spawn_budget_diagnostics = None


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


def detect_boundary_from_region(kept_xyz, hole_lo, hole_hi, margin_spacing=4.0):
    """Detect a hole rim without looking at held-out Gaussian coordinates.

    The distance from every surviving point to the configured selection box is used;
    points within a fixed multiple of the surviving-set spacing form the boundary.
    This keeps removed GT entirely outside the completion path.
    """
    tree = cKDTree(kept_xyz)
    d_keep, _ = tree.query(kept_xyz, k=min(2, len(kept_xyz)))
    spacing = float(np.median(d_keep[:, -1]))
    below = np.maximum(hole_lo[None, :] - kept_xyz, 0.0)
    above = np.maximum(kept_xyz - hole_hi[None, :], 0.0)
    distance = np.linalg.norm(below + above, axis=1)
    rim = np.where(distance < margin_spacing * spacing)[0]
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


def estimate_normals_local_pca_at(xyz, query_idx, k=16, return_diagnostics=False):
    """Estimate PCA normals only at selected support indices.

    Real checkpoints may contain millions of Gaussians, while only the hole boundary
    needs normals.  This avoids the previous O(N) Python loop over the whole scene.
    """
    query_idx = np.asarray(query_idx, dtype=np.int64)
    tree = cKDTree(xyz)
    _, neighbours = tree.query(xyz[query_idx], k=min(k, len(xyz)))
    normals = np.zeros((len(query_idx), 3), dtype=xyz.dtype)
    eigenvalues = np.zeros((len(query_idx), 3), dtype=np.float32)
    centroid = xyz.mean(axis=0)
    for q, original_idx in enumerate(query_idx):
        nb = xyz[np.atleast_1d(neighbours[q])]
        mu = nb.mean(axis=0)
        cov = (nb - mu).T @ (nb - mu) / max(len(nb) - 1, 1)
        values, vec = np.linalg.eigh(cov)
        eigenvalues[q] = values
        normal = vec[:, 0]
        if normal @ (xyz[original_idx] - centroid) < 0:
            normal = -normal
        normals[q] = normal / (np.linalg.norm(normal) + 1e-8)
    if not return_diagnostics:
        return normals
    curvature = eigenvalues[:, 0] / (eigenvalues.sum(axis=1) + 1e-12)
    confidence = 1.0 - curvature
    return normals, {"eigenvalues": eigenvalues, "curvature": curvature,
                     "normal_confidence": confidence,
                     "neighbor_count": np.full(len(query_idx), min(k, len(xyz)), dtype=np.int32)}


# ---------------------------------------------------------------------------
# Step 6: KNN graph with position / normal / appearance / semantic weights + gating
# ---------------------------------------------------------------------------

def _appearance_features(model):
    return model._features_dc.detach().squeeze(1).cpu().numpy()


def _semantic_features(model):
    return model._objects_dc.detach().reshape(model._objects_dc.shape[0], -1).cpu().numpy()


def build_knn_graph(xyz, normals, appearance, semantic=None, k=12,
                    use_normal=True, use_appearance=True, use_semantic=True,
                    gate_normal=0.8, sigma_pos=None, sigma_app=None,
                    normal_affinity="hard", normal_sigma_deg=30.0,
                    normal_edge_min=0.10, adaptive_min_sigma_deg=5.0,
                    semantic_gate="hard", semantic_sigma=0.5):
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

    if normal_affinity not in ("hard", "soft", "adaptive"):
        raise ValueError("normal_affinity must be hard, soft, or adaptive")

    # Self-tuning bandwidth for the adaptive strategy.  It is estimated once from
    # each node's spatial neighbours and is therefore identical across corner angles.
    local_sigma = None
    if use_normal and normals is not None and normal_affinity == "adaptive":
        local_sigma = np.full(n, adaptive_min_sigma_deg, dtype=np.float32)
        for i in range(n):
            js = np.asarray([j for j in np.atleast_1d(idx[i]) if j != i], dtype=np.int64)
            if len(js):
                dots = np.clip(np.abs(normals[js] @ normals[i]), 0.0, 1.0)
                angles = np.degrees(np.arccos(dots))
                # Lower-half neighbours normally lie on the same local surface; using
                # their robust median prevents a second surface from setting bandwidth.
                q = np.sort(angles)[:max(1, len(angles) // 2)]
                local_sigma[i] = max(adaptive_min_sigma_deg,
                                     float(np.median(q)) * 2.0)

    for i in range(n):
        for j in idx[i]:
            if i == j:
                continue
            d2 = np.sum((xyz[i] - xyz[j]) ** 2)
            w = np.exp(-d2 / (2.0 * sigma_pos ** 2))
            # Normal compatibility.  The legacy strategy is a hard dot-product gate.
            # Soft strategies retain a continuous affinity in the edge weight and use
            # one global edge-retention threshold for graph partitioning.
            if use_normal and normals is not None:
                dot = np.clip(abs(float(np.dot(normals[i], normals[j]))), 0.0, 1.0)
                if normal_affinity == "hard":
                    if dot < gate_normal:
                        continue
                else:
                    angle = float(np.degrees(np.arccos(dot)))
                    if normal_affinity == "soft":
                        sigma_n = normal_sigma_deg
                    else:
                        sigma_n = float(np.sqrt(local_sigma[i] * local_sigma[j]))
                    normal_w = float(np.exp(-(angle ** 2) / (2.0 * sigma_n ** 2)))
                    if normal_w < normal_edge_min:
                        continue
                    w *= normal_w
            if use_semantic and semantic is not None:
                if semantic_gate == "hard":
                    if np.argmax(semantic[i]) != np.argmax(semantic[j]):
                        continue
                elif semantic_gate == "soft":
                    si = semantic[i] / (np.linalg.norm(semantic[i]) + 1e-8)
                    sj = semantic[j] / (np.linalg.norm(semantic[j]) + 1e-8)
                    w *= np.exp(-(1.0 - np.clip(si @ sj, -1.0, 1.0)) /
                                max(semantic_sigma, 1e-8))
                else:
                    raise ValueError("semantic_gate must be hard or soft")
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
    _, vec = np.linalg.eigh(boundary_normals.T @ boundary_normals)
    ref = vec[:, -1]
    dots = np.clip(np.abs(boundary_normals @ ref), 0.0, 1.0)
    spread = np.degrees(np.arccos(dots))
    # PCA normals directly beside a junction are mixed by cross-surface neighbours.
    # A robust percentile keeps those few outliers from turning a planar component
    # into a spurious cylinder while still detecting sustained curvature.
    return np.percentile(spread, 90.0) < angle_thresh_deg


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


def _robust_component_spacing(points, fallback):
    """Median inlier nearest-neighbour spacing from observable support only."""
    if len(points) < 3:
        return float(fallback)
    distances, _ = cKDTree(points).query(points, k=2)
    values = distances[:, 1]
    lo, hi = np.percentile(values, [10, 90])
    inliers = values[(values >= lo) & (values <= hi)]
    return float(np.median(inliers)) if len(inliers) else float(fallback)


def _fit_tangent_frame(fit, roi_center):
    if fit["type"] == "plane":
        normal = fit["normal"] / (np.linalg.norm(fit["normal"]) + 1e-8)
        origin = roi_center - ((roi_center - fit["center"]) @ normal) * normal
    else:
        axis, center, radius = fit["axis"], fit["center"], fit["radius"]
        radial = roi_center - center - ((roi_center - center) @ axis) * axis
        if np.linalg.norm(radial) < 1e-8:
            radial = np.array([1., 0., 0.])
            if abs(float(radial @ axis)) > 0.9:
                radial = np.array([0., 0., 1.])
            radial -= (radial @ axis) * axis
        normal = radial / (np.linalg.norm(radial) + 1e-8)
        origin = center + radius * normal + ((roi_center - center) @ axis) * axis
    u = np.array([1., 0., 0.])
    if abs(float(u @ normal)) > 0.9:
        u = np.array([0., 1., 0.])
    u -= (u @ normal) * normal; u /= np.linalg.norm(u) + 1e-8
    v = np.cross(normal, u); v /= np.linalg.norm(v) + 1e-8
    return origin, u, v, normal


def _project_to_fit(seeds, fit):
    if fit["type"] == "plane":
        normal = fit["normal"] / (np.linalg.norm(fit["normal"]) + 1e-8)
        points = seeds - np.outer((seeds - fit["center"]) @ normal, normal)
        return points, np.tile(normal, (len(points), 1))
    axis, center, radius = fit["axis"], fit["center"], fit["radius"]
    axial = np.outer((seeds - center) @ axis, axis)
    radial = seeds - center - axial
    normals = radial / (np.linalg.norm(radial, axis=1, keepdims=True) + 1e-8)
    points = center + axial + radius * normals
    return points, normals


def _poisson_reject(points, normals, min_distance, target_count, rng):
    """Lightweight deterministic dart rejection; no GT-dependent count decisions."""
    if len(points) <= target_count:
        return points, normals
    order = rng.permutation(len(points))
    accepted = []
    for index in order:
        point = points[index]
        if not accepted or np.min(np.linalg.norm(points[np.asarray(accepted)] - point, axis=1)) >= min_distance:
            accepted.append(int(index))
            if len(accepted) >= target_count:
                break
    if len(accepted) < target_count:
        used = set(accepted)
        accepted.extend([int(i) for i in order if int(i) not in used][:target_count-len(accepted)])
    accepted = np.asarray(accepted[:target_count], dtype=np.int64)
    return points[accepted], normals[accepted]


def estimate_method_independent_spawn_budget(boundary_xyz, boundary_normals, scene,
                                             fallback_spacing):
    """Estimate one total count before any C0-C3 graph construction/partition.

    The estimate consumes only surviving boundary support, robust spacing, and one
    graph-independent fitted local surface.  It never sees removed points or labels.
    """
    roi_center = np.asarray(getattr(scene, "roi_center", scene.center), dtype=np.float32)
    roi_radius = float(getattr(scene, "roi_radius",
                               0.5 * float(np.max(scene.hole_hi - scene.hole_lo))))
    spacing = _robust_component_spacing(boundary_xyz, fallback_spacing)
    if surface_model_is_planar(boundary_xyz, boundary_normals):
        center, normal = fit_surface_plane(boundary_xyz)
        fit = {"type": "plane", "center": center, "normal": normal}
    else:
        fit = fit_cylinder(boundary_xyz, boundary_normals)
    origin, u, v, _ = _fit_tangent_frame(fit, roi_center)
    uv = np.stack([(boundary_xyz - origin) @ u, (boundary_xyz - origin) @ v], axis=1)
    hull_area = 0.0
    if len(uv) >= 3:
        try:
            from scipy.spatial import ConvexHull
            hull_area = float(ConvexHull(uv).volume)
        except Exception:
            pass
    cell_area = np.sqrt(3.0) * 0.5 * spacing * spacing
    support_area = max(hull_area, len(boundary_xyz) * cell_area, 1e-10)
    density = len(boundary_xyz) / support_area
    missing_area = np.pi * roi_radius * roi_radius
    raw_budget = density * missing_area
    budget = max(1, int(round(raw_budget)))
    max_by_spacing = max(1, int(missing_area /
        (np.sqrt(3.) * .5 * (0.75 * spacing) ** 2)))
    budget = min(budget, max_by_spacing, max(4, 4 * len(boundary_xyz)))
    return budget, {"reliable_boundary_gaussians": int(len(boundary_xyz)),
                    "robust_spacing": spacing,
                    "observable_support_area": support_area,
                    "boundary_density": density,
                    "estimated_missing_surface_area": missing_area,
                    "raw_budget": raw_budget, "N_budget": int(budget)}


def _exact_component_budgets(weights, total):
    """Largest-remainder integer allocation with an exact method-independent sum."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    if weights.sum() <= 0:
        weights[:] = 1.0
    quotas = total * weights / weights.sum()
    allocation = np.floor(quotas).astype(np.int64)
    remainder = int(total - allocation.sum())
    if remainder:
        order = np.argsort(-(quotas - allocation), kind="stable")
        allocation[order[:remainder]] += 1
    return allocation


def density_aware_surface_spawn(boundary_xyz, boundary_normals, comp_labels, comp_fits,
                                scene, fallback_spacing, rng, total_budget=None):
    """Spawn from observable component density x fitted missing surface area.

    Counts use surviving support only.  The configured ROI defines where geometry is
    missing; removed Gaussian positions/counts are never consumed.  A robust component
    budget prevents fragmented/tiny components from duplicating the full hole area.
    """
    roi_center = np.asarray(getattr(scene, "roi_center", scene.center), dtype=np.float32)
    roi_radius = getattr(scene, "roi_radius", None)
    if roi_radius is None:
        roi_radius = 0.5 * float(np.max(scene.hole_hi - scene.hole_lo))
    roi_radius = float(roi_radius)
    component_sizes = np.bincount(comp_labels, minlength=len(comp_fits)).astype(np.float64)
    total_support = max(float(component_sizes.sum()), 1.0)
    points_all, normals_all, labels_all, diagnostics = [], [], [], []

    # Count-matched allocation weights use only per-component reliable support and its
    # observable tangent footprint.  The total was already frozen before partition.
    fixed_allocations = None
    if total_budget is not None:
        allocation_weights = []
        for component, fit in enumerate(comp_fits):
            members = np.where(comp_labels == component)[0]
            if len(members) == 0:
                allocation_weights.append(0.0); continue
            support = boundary_xyz[members]
            local_spacing = _robust_component_spacing(support, fallback_spacing)
            origin, u, v, _ = _fit_tangent_frame(fit, roi_center)
            uv = np.stack([(support-origin)@u, (support-origin)@v], axis=1)
            area = 0.0
            if len(uv) >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    area = float(ConvexHull(uv).volume)
                except Exception:
                    pass
            area = max(area, len(members) * np.sqrt(3.) * .5 * local_spacing**2, 1e-10)
            component_density = len(members) / area
            component_area_share = np.pi * roi_radius**2 * len(members) / total_support
            allocation_weights.append(component_density * component_area_share)
        fixed_allocations = _exact_component_budgets(allocation_weights, int(total_budget))

    for component, fit in enumerate(comp_fits):
        members = np.where(comp_labels == component)[0]
        if len(members) == 0:
            continue
        support = boundary_xyz[members]
        spacing = _robust_component_spacing(support, fallback_spacing)
        origin, u, v, _ = _fit_tangent_frame(fit, roi_center)
        uv = np.stack([(support - origin) @ u, (support - origin) @ v], axis=1)
        # Observable support area: robust tangent footprint, lower-bounded by one
        # hexagonal cell per reliable support Gaussian to suppress spacing outliers.
        support_area_hull = 0.0
        if len(uv) >= 3:
            try:
                from scipy.spatial import ConvexHull
                support_area_hull = float(ConvexHull(uv).volume)
            except Exception:
                support_area_hull = 0.0
        cell_area = np.sqrt(3.0) * 0.5 * spacing * spacing
        support_area = max(support_area_hull, len(members) * cell_area, 1e-10)
        density = len(members) / support_area

        # Fitted local-surface disk.  Component allocation follows observable support
        # share so graph fragmentation cannot multiply the full missing area.
        missing_area_total = np.pi * roi_radius * roi_radius
        support_fraction = float(len(members) / total_support)
        missing_area = missing_area_total * support_fraction
        raw_count = density * missing_area
        # Robust caps: at least one, no tiny component gets >4x its support, and target
        # spacing cannot become denser than 0.75x the observed robust spacing.
        if fixed_allocations is None:
            target_count = max(1, int(round(raw_count)))
            target_count = min(target_count, max(4, 4 * len(members)))
            max_by_spacing = max(1, int(missing_area / (np.sqrt(3.) * .5 * (0.75 * spacing) ** 2)))
            target_count = min(target_count, max_by_spacing)
        else:
            target_count = int(fixed_allocations[component])
            if target_count == 0:
                diagnostics.append({"component": int(component), "support_gaussians": int(len(members)),
                                    "robust_spacing": spacing, "observable_support_area": support_area,
                                    "boundary_density": density, "estimated_missing_surface_area": missing_area,
                                    "raw_predicted_count": raw_count, "spawned_count": 0,
                                    "resulting_newborn_density": 0.0,
                                    "density_ratio_newborn_boundary": 0.0})
                continue

        candidates_n = max(target_count * 12, 128)
        theta = rng.uniform(0.0, 2.0 * np.pi, candidates_n)
        radial = roi_radius * np.sqrt(rng.uniform(0.0, 1.0, candidates_n))
        seeds = origin + np.outer(radial * np.cos(theta), u) + np.outer(radial * np.sin(theta), v)
        candidates, candidate_normals = _project_to_fit(seeds, fit)
        points, normals = _poisson_reject(candidates, candidate_normals,
                                          min_distance=0.75 * spacing,
                                          target_count=target_count, rng=rng)
        points_all.append(points); normals_all.append(normals)
        labels_all.append(np.full(len(points), component, dtype=np.int64))
        diagnostics.append({"component": int(component), "support_gaussians": int(len(members)),
                            "robust_spacing": spacing, "observable_support_area": support_area,
                            "boundary_density": density, "estimated_missing_surface_area": missing_area,
                            "raw_predicted_count": raw_count, "spawned_count": int(len(points)),
                            "resulting_newborn_density": len(points) / max(missing_area, 1e-10),
                            "density_ratio_newborn_boundary": (len(points) / max(missing_area, 1e-10)) / density})
    if not points_all:
        raise RuntimeError("density-aware spawning found no supported components")
    return (np.concatenate(points_all).astype(np.float32),
            np.concatenate(normals_all).astype(np.float32),
            np.concatenate(labels_all), diagnostics)


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
                   normal_noise=0.0, normal_affinity="hard",
                   hole_mask_override=None, semantic_gate="hard",
                   spawn_rule="density_aware"):
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

    if hole_mask_override is None:
        kept_mask, hole_mask = carve_hole(xyz, hole_lo, hole_hi)
    else:
        hole_mask = np.asarray(hole_mask_override, dtype=bool)
        if hole_mask.shape != (len(xyz),):
            raise ValueError("hole_mask_override must have shape ({},)".format(len(xyz)))
        kept_mask = ~hole_mask
    hole_xyz = xyz[hole_mask]
    kept_xyz = xyz[kept_mask]
    kept_idx = np.where(kept_mask)[0]
    if hole_xyz.shape[0] < 8:
        raise RuntimeError("hole too small; adjust scene hole bounds")

    # Step 4: boundary detection (kept-space indices).
    boundary_kept, spacing = detect_boundary_from_region(kept_xyz, hole_lo, hole_hi)
    spacing = spacing if spacing_override is None else spacing_override
    if len(boundary_kept) < 8:
        raise RuntimeError("too few boundary Gaussians; adjust hole")
    boundary_idx = kept_idx[boundary_kept]      # original indices
    boundary_xyz = xyz[boundary_idx]

    # Step 5: PCA normals only where they are consumed (the boundary).  This is
    # equivalent to indexing the full estimate but scales to real checkpoints.
    boundary_normals, normal_diag = estimate_normals_local_pca_at(
        kept_xyz, boundary_kept, k=boundary_k, return_diagnostics=True)
    appearance = _appearance_features(model)
    semantic = _semantic_features(model)

    # surface axis / in-plane from the hole's smallest extent
    _, in_plane = hole_footprint(hole_lo, hole_hi)

    result = CompletionResult()
    # Deliberately do not retain held-out positions in the completion result.  They are
    # reconstructed by evaluation code from kept_mask after completion has finished.
    result.hole_xyz = None
    result.kept_mask = kept_mask
    result.boundary_idx = boundary_idx
    result.normals = boundary_normals
    result.boundary_spacing = spacing
    result.surface_label = None
    result.components = None
    result.normal_affinity = normal_affinity
    result.spawn_rule = spawn_rule
    result.pca_eigenvalues = normal_diag["eigenvalues"]
    result.normal_confidence = normal_diag["normal_confidence"]
    result.neighbor_count = normal_diag["neighbor_count"]
    result.local_curvature = normal_diag["curvature"]

    # Count-matched total is frozen here, before variant flags, graph edges, semantic
    # gates, or graph partition are applied.
    fixed_spawn_budget = None
    if spawn_rule == "count_matched":
        fixed_spawn_budget, budget_diag = estimate_method_independent_spawn_budget(
            boundary_xyz, boundary_normals, scene, spacing)
        result.spawn_budget = fixed_spawn_budget
        result.spawn_budget_diagnostics = budget_diag

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
        gate_normal=0.8 if use_normal else None, normal_affinity=normal_affinity,
        semantic_gate=semantic_gate)
    comp_labels = partition_boundary_graph(len(boundary_xyz), part_rows, part_cols)
    n_comp = int(comp_labels.max()) + 1

    # Connectivity graph refines propagation weights: adds appearance (C2+) as a soft
    # edge weight on top of position+normal.  Same topology decisions as partition but
    # with the variant's appearance term folded in for the fill.
    if use_appearance:
        rows, cols, weights, _ = build_knn_graph(
            boundary_xyz, bn, boundary_appearance, bs, k=knn_k,
            use_normal=use_normal, use_appearance=True, use_semantic=use_semantic,
            gate_normal=0.8 if use_normal else None, normal_affinity=normal_affinity,
            semantic_gate=semantic_gate)
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

    # Spawn directly on fitted surfaces.  The legacy AABB grid remains callable only
    # for controlled old-vs-new diagnostics.
    hole_center = 0.5 * (hole_lo + hole_hi)
    hu = (hole_hi[in_plane[0]] - hole_lo[in_plane[0]]) / 2.0
    hv = (hole_hi[in_plane[1]] - hole_lo[in_plane[1]]) / 2.0
    if spawn_rule in ("density_aware", "count_matched"):
        fitted_xyz, fitted_normals, comp, spawn_diag = density_aware_surface_spawn(
            boundary_xyz, bn, comp_labels, comp_fits, scene, spacing, rng,
            total_budget=fixed_spawn_budget)
        pts = list(fitted_xyz); ns = list(fitted_normals)
        result.spawn_diagnostics = spawn_diag
        base = None
    elif spawn_rule == "legacy":
        base = []
        for a in np.arange(-hu, hu + spacing / 2, spacing):
            for b in np.arange(-hv, hv + spacing / 2, spacing):
                p = hole_center.copy(); p[in_plane[0]] += a; p[in_plane[1]] += b
                base.append(p)
        base = np.asarray(base, dtype=np.float32) if len(base) else boundary_xyz[:1].copy()
    else:
        raise ValueError("spawn_rule must be density_aware, count_matched, or legacy")
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 1.0])
    if spawn_rule == "legacy":
        tree = cKDTree(boundary_xyz); _, bi = tree.query(base, k=1); comp = comp_labels[bi]
        pts, ns = [], []
        for p in range(len(base)):
            fit = comp_fits[comp[p]]
            projected, projected_normals = _project_to_fit(base[p:p+1], fit)
            pts.append(projected[0]); ns.append(projected_normals[0])
    fitted_xyz = np.asarray(pts, dtype=np.float32)
    fitted_normals = np.asarray(ns, dtype=np.float32)
    new_xyz = fitted_xyz + \
        rng.normal(0, spacing * 0.05, size=(len(pts), 3)).astype(np.float32)
    new_normals = fitted_normals.copy()

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
    new_attrs, propagation_weights = propagate_from_boundary(new_xyz, boundary_xyz, boundary_attrs,
                                           comp_labels, comp, mode="graph_comp",
                                           spacing=spacing, conn=conn)
    result.new_xyz = new_xyz
    result.new_normals = new_normals
    result.new_attributes = new_attrs
    result.surface_label = comp
    result.components = [np.where(comp_labels == c)[0] for c in range(n_comp)]
    result.n_components = n_comp
    result.graph_rows = part_rows
    result.graph_cols = part_cols
    result.component_labels = comp_labels
    result.fitted_xyz = fitted_xyz
    result.fitted_normals = fitted_normals
    # Observable-only completion confidence: local geometry quality, normalized
    # propagation support, and semantic consistency.  GT is never consulted.
    _, nearest_boundary = cKDTree(boundary_xyz).query(new_xyz, k=1)
    geometry_conf = np.clip(result.normal_confidence[nearest_boundary], 0.0, 1.0)
    support_conf = np.clip(np.max(propagation_weights, axis=1) *
                           np.count_nonzero(propagation_weights, axis=1), 0.0, 1.0)
    semantic_conf = np.ones(len(new_xyz), dtype=np.float32)
    if use_semantic and boundary_semantic is not None and boundary_semantic.shape[1] > 0:
        sem = np.abs(boundary_semantic[nearest_boundary])
        semantic_conf = np.max(sem, axis=1) / (np.sum(sem, axis=1) + 1e-8)
    result.confidence_terms = {"geometry": geometry_conf, "support": support_conf,
                               "semantic": semantic_conf}
    result.completion_confidence = geometry_conf * support_conf * semantic_conf
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

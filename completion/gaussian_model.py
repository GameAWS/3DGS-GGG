"""GPU- and CUDA-free GaussianModel + camera + projective helpers for the benchmark.

Gaussian Grouping's own `scene/gaussian_model.py` imports `simple_knn._C.distCUDA2`
(a CUDA extension) at module import time, which cannot be compiled on every machine.
This module re-implements the *exact* field layout and ply format of that class so
that real Gaussian Grouping checkpoints can be loaded and saved without the CUDA
rasterizer.  It is used in place of `scene.gaussian_model.GaussianModel`:

  _xyz          (N, 3)           positions
  _features_dc  (N, 1, 3)        SH DC coefficients
  _features_rest(N, M, 3)        SH higher-degree coefficients
  _opacity      (N, 1)           logit opacity
  _scaling      (N, 3)           log-scale
  _rotation     (N, 4)           quaternion
  _objects_dc   (N, 1, num_obj)  identity/grouping encoding

The ply attribute order matches create_from_pcd's output exactly:
  x y z nx ny nz f_dc_* f_rest_* opacity scale_* rot_* obj_dc_*
"""

import math
import os

import numpy as np
import torch
from torch import nn
from plyfile import PlyData, PlyElement


class GaussianModel:
    def __init__(self, sh_degree=3):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._objects_dc = torch.empty(0)
        self.num_objects = 16
        self.spatial_lr_scale = 1.0

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat([self._features_dc.reshape(self._features_dc.shape[0], -1),
                          self._features_rest.reshape(self._features_rest.shape[0], -1)],
                         dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_objects(self):
        return self._objects_dc.reshape(self._objects_dc.shape[0], -1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_centers(self):
        return self._xyz

    # ---- ply serialization (mirrors GG layout) --------------------------------

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l += ['f_dc_{}'.format(i) for i in range(self._features_dc.shape[1] * self._features_dc.shape[2])]
        l += ['f_rest_{}'.format(i) for i in range(self._features_rest.shape[1] * self._features_rest.shape[2])]
        l.append('opacity')
        l += ['scale_{}'.format(i) for i in range(self._scaling.shape[1])]
        l += ['rot_{}'.format(i) for i in range(self._rotation.shape[1])]
        l += ['obj_dc_{}'.format(i) for i in range(self._objects_dc.shape[1] * self._objects_dc.shape[2])]
        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().cpu().reshape(self._features_dc.shape[0], -1).numpy()
        f_rest = self._features_rest.detach().cpu().reshape(self._features_rest.shape[0], -1).numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        obj_dc = self._objects_dc.detach().cpu().reshape(self._objects_dc.shape[0], -1).numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack([np.asarray(plydata.elements[0][n], dtype=np.float32)
                        for n in ("x", "y", "z")], axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"], dtype=np.float32)[:, None]

        features_dc = np.zeros((xyz.shape[0], 1, 3), dtype=np.float32)
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 0, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 0, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = sorted(
            (p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")),
            key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3, \
            "ply has {} f_rest_ fields, expected {} for sh_degree {}".format(
                len(extra_f_names), 3 * (self.max_sh_degree + 1) ** 2 - 3, self.max_sh_degree)
        features_extra = np.stack([np.asarray(plydata.elements[0][n], dtype=np.float32)
                                   for n in extra_f_names], axis=1)
        features_extra = features_extra.reshape(xyz.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3)

        scale_names = sorted((p.name for p in plydata.elements[0].properties
                              if p.name.startswith("scale_")),
                             key=lambda x: int(x.split("_")[-1]))
        scales = np.stack([np.asarray(plydata.elements[0][n], dtype=np.float32)
                           for n in scale_names], axis=1)

        rot_names = sorted((p.name for p in plydata.elements[0].properties
                            if p.name.startswith("rot") and p.name != "rotation"),
                           key=lambda x: int(x.split("_")[-1]))
        rots = np.stack([np.asarray(plydata.elements[0][n], dtype=np.float32)
                         for n in rot_names], axis=1)

        obj_names = sorted((p.name for p in plydata.elements[0].properties
                            if p.name.startswith("obj_dc_")), key=lambda x: int(x.split("_")[-1]))
        # Non-Grouping 3DGS PLYs may omit identity features.  Keep one constant zero
        # channel so downstream tensor shapes remain valid; scene inspection detects
        # its zero variance and reports semantic_identity.exists=false.  C3 then
        # degrades gracefully to the same topology as no semantic signal.
        if obj_names:
            self.num_objects = len(obj_names)
            objects_dc = np.stack([np.asarray(plydata.elements[0][n], dtype=np.float32)
                                   for n in obj_names], axis=1)  # (N, num_obj)
        else:
            self.num_objects = 1
            objects_dc = np.zeros((xyz.shape[0], 1), dtype=np.float32)

        self._xyz = nn.Parameter(torch.tensor(xyz))
        self._features_dc = nn.Parameter(torch.tensor(features_dc.reshape(xyz.shape[0], 1, 3)))
        self._features_rest = nn.Parameter(torch.tensor(features_extra.reshape(
            xyz.shape[0], self._features_rest.shape[1] if self._features_rest.numel() else (self.max_sh_degree + 1) ** 2 - 1, 3)))
        self._opacity = nn.Parameter(torch.tensor(opacities))
        self._scaling = nn.Parameter(torch.tensor(scales))
        self._rotation = nn.Parameter(torch.tensor(rots))
        self._objects_dc = nn.Parameter(torch.tensor(objects_dc.reshape(xyz.shape[0], self.num_objects, 1)))
        self.active_sh_degree = self.max_sh_degree


def rgb_to_sh_dc(rgb):
    """Convert RGB (shape [..., 3]) to SH DC coefficients (SH_C0 * rgb)."""
    SH_C0 = 0.28209479177387814
    return rgb * SH_C0


def make_cameras_from_poses(poses, height=360, width=480, fov_deg=50.0):
    """Build camera objects from (R, T) world-to-camera poses (GG convention)."""
    FoVx = np.deg2rad(fov_deg)
    FoVy = np.deg2rad(fov_deg)
    znear, zfar = 0.01, 100.0
    cams = []
    for i, (R, T) in enumerate(poses):
        world_view_transform = torch.tensor(get_world_to_view_2(R, T)).transpose(0, 1).float()
        projection_matrix = get_projection_matrix(znear, zfar, FoVx, FoVy).transpose(0, 1)
        full_proj = (world_view_transform.unsqueeze(0).bmm(
            projection_matrix.unsqueeze(0))).squeeze(0)
        cam = Camera(i, R, T, FoVx, FoVy, height, width)
        cam.world_view_transform = world_view_transform
        cam.projection_matrix = projection_matrix
        cam.full_proj_transform = full_proj
        cam.camera_center = world_view_transform.inverse()[3, :3]
        cams.append(cam)
    return cams


class Camera:
    """Minimal camera holder matching the fields the render path uses."""
    def __init__(self, uid, R, T, FoVx, FoVy, height, width, image_name="cam"):
        self.uid = uid
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_height = height
        self.image_width = width
        self.image_name = image_name
        self.znear = 0.01
        self.zfar = 100.0


def make_orbit_poses(radius=2.2, height=0.9, n=3, center=(0.0, 0.0, 0.0), fov_deg=50.0,
                     surface_axis=2, elevation_deg=30.0):
    """Camera poses looking at `center` from an elevated ring.

    `surface_axis` is the axis perpendicular to the target surface (2 = the plane is at
    constant z, so "above" is +z).  Cameras sit on a ring of radius `radius` at an
    elevation above the surface and look down at `center`, so the surface (and any hole
    carved into it) is seen obliquely rather than edge-on.
    """
    center = np.asarray(center, dtype=np.float32)
    poses = []
    elev = np.deg2rad(elevation_deg)
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        eye = np.array(center, dtype=np.float32)
        # Place the camera on the elevated ring in the plane(s) perpendicular to
        # `surface_axis`.
        if surface_axis == 2:  # surface normal is +z -> camera ring above (z+)
            eye[0] += radius * np.cos(elev) * np.cos(theta)
            eye[1] += radius * np.cos(elev) * np.sin(theta)
            eye[2] += radius * np.sin(elev)
        elif surface_axis == 1:  # surface normal is +y -> camera ring above (y+)
            eye[0] += radius * np.cos(elev) * np.cos(theta)
            eye[2] += radius * np.cos(elev) * np.sin(theta)
            eye[1] += radius * np.sin(elev)
        else:  # surface normal is +x -> camera ring above (x+)
            eye[1] += radius * np.cos(elev) * np.cos(theta)
            eye[2] += radius * np.cos(elev) * np.sin(theta)
            eye[0] += radius * np.sin(elev)

        fwd = center - eye
        fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        # Guard against a degenerate up vector (camera pointing straight along +y).
        if abs(float(np.dot(fwd, up))) > 0.99:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(fwd, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, fwd)
        R = np.stack([right, up, -fwd], axis=0)  # rows = camera axes
        T = -R @ eye
        poses.append((R, T))
    return poses


def get_world_to_view_2(R, t):
    """world->view (same as GG utils.graphics_utils.getWorld2View2)."""
    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    C2W[:3, 3] = C2W[:3, 3]  # no translate/scale applied
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def get_projection_matrix(znear, zfar, fovX, fovY):
    tan_half_fov_y = math.tan(fovY / 2.0)
    tan_half_fov_x = math.tan(fovX / 2.0)

    top = tan_half_fov_y * znear
    bottom = -top
    right = tan_half_fov_x * znear
    left = -right

    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

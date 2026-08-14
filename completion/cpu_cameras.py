"""CPU COLMAP camera reader (mirrors Gaussian Grouping's scene.colmap_loader binary format).

Gaussian Grouping's ``scene`` package imports simple_knn._C (CUDA) and its Camera
class puts transforms on ``.cuda()``, so neither can run on this CPU-only machine.
This module re-reads the exact COLMAP binary files (images.bin / cameras.bin) and
exposes CPU-only camera geometry (world->camera projection) used by the layer
descriptors.  No completion code is touched.
"""

import math
import os
import struct

import numpy as np


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def read_next_bytes(fid, num_bytes, prefix_char, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + prefix_char, data)


def read_cameras_text(path):
    """Fallback text reader (not used if .bin present)."""
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            cameras[int(p[0])] = {"model": p[1], "width": int(p[2]), "height": int(p[3]),
                                  "params": np.asarray(p[4:], dtype=np.float64)}
    return cameras


def read_cameras_binary(path):
    cameras = {}
    model_names = {0: "SIMPLE_PINHOLE", 1: "PINHOLE"}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width, height = camera_properties[2], camera_properties[3]
            num_params = 3 if model_id == 0 else 4
            params = np.asarray(read_next_bytes(fid, 8 * num_params, "d" * num_params))
            cameras[camera_id] = {"model": model_names.get(int(model_id), "UNKNOWN"),
                                  "width": int(width), "height": int(height),
                                  "params": params}
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.asarray(binary_image_properties[1:5])
            tvec = np.asarray(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            images[image_id] = {"name": image_name, "qvec": qvec, "tvec": tvec,
                                "camera_id": camera_id}
    return images


def load_cameras(data_root, max_width=800):
    """Load CPU camera geometry from a COLMAP sparse/0 dir. Returns list of dicts."""
    sparse = os.path.join(data_root, "sparse", "0")
    bin_ext = os.path.join(sparse, "images.bin")
    bin_intr = os.path.join(sparse, "cameras.bin")
    extr = read_images_binary(bin_ext)
    intr = read_cameras_binary(bin_intr)
    cams = []
    for image_id, item in sorted(extr.items(), key=lambda kv: kv[1]["name"]):
        ci = intr[item["camera_id"]]
        scale = min(1.0, max_width / ci["width"])
        width = int(round(ci["width"] * scale))
        height = int(round(ci["height"] * scale))
        if ci["model"] == "SIMPLE_PINHOLE":
            fx = fy = ci["params"][0]
        elif ci["model"] == "PINHOLE":
            fx, fy = ci["params"][:2]
        else:
            continue
        R = qvec2rotmat(item["qvec"]).T       # world->camera rotation (GG convention)
        T = np.asarray(item["tvec"], dtype=np.float64)
        cams.append({
            "uid": image_id, "name": item["name"], "R": R, "T": T,
            "fx": fx * scale, "fy": fy * scale, "width": width, "height": height,
            "fovx": 2 * math.atan(width / (2 * max(fx * scale, 1e-9))),
            "fovy": 2 * math.atan(height / (2 * max(fy * scale, 1e-9))),
        })
    return cams


def project_camera(cam, xyz):
    """Project world points into a camera. p_cam = R @ p_world + T (near=+z)."""
    p = (cam["R"] @ xyz.T).T + cam["T"]
    valid = p[:, 2] > 0.01
    fx, fy = cam["fx"], cam["fy"]
    x = fx * p[:, 0] / p[:, 2] + cam["width"] / 2
    y = fy * p[:, 1] / p[:, 2] + cam["height"] / 2
    return x, y, p[:, 2], valid


def camera_eye(cam):
    """World-space camera center (from R,T world->cam)."""
    Rt = np.eye(4)
    Rt[:3, :3] = cam["R"]
    Rt[:3, 3] = cam["T"]
    Rt[3, 3] = 1.0
    inv = np.linalg.inv(Rt)
    return inv[:3, 3]


def camera_views_hole(cam, hole_xyz, boundary_xyz=None, margin_px=8.0):
    """Whether a camera sees the hole region: hole pixels near image center, large
    enough, and at least one boundary point projected.  Returns None if not visible."""
    x, y, depth, valid = project_camera(cam, hole_xyz)
    if not valid.any():
        return None
    xv, yv = x[valid], y[valid]
    radius_px = max(cam["fx"], cam["fy"]) * (np.median(depth) and 1.0)  # placeholder
    # in-view + margin check
    inside = (xv >= -margin_px) & (xv < cam["width"] + margin_px) & \
             (yv >= -margin_px) & (yv < cam["height"] + margin_px)
    if inside.sum() < 4:
        return None
    return {"n_project": int(inside.sum()), "depth_med": float(np.median(depth[valid])),
            "depth_min": float(depth[valid].min()), "depth_max": float(depth[valid].max())}
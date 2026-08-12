"""Controlled Gaussian completion benchmark (no 2D inpainting / generative model).

Runs the 13-step completion pipeline from geometry.py over a synthetic Gaussian
Grouping scene (synthetic_scene.py), renders with the CPU fallback (render.py), and
reports hole-region PSNR/SSIM/LPIPS + Chamfer + counts + runtime (metrics.py).
"""

from . import synthetic_scene, geometry, render, metrics

__all__ = ["synthetic_scene", "geometry", "render", "metrics"]
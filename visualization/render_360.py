# -*- coding: utf-8 -*-
"""
visualization/render_360.py — Renders 360-degree orbit video of 3D meshes
"""

import os
import sys
import math
import numpy as np
import cv2
from pathlib import Path

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    o3d = None
    _O3D_AVAILABLE = False

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

MESH_EXTENSIONS = {".ply", ".obj", ".glb", ".gltf", ".stl", ".off"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}


def create_360_trajectory(center, radius, num_frames, height_offset=0.2, look_at_offset=[0, 0, 0]):
    """
    Generate a sequence of camera positions orbiting 360 degrees around a center point.
    """
    poses = []
    center = np.array(center)
    target = center + np.array(look_at_offset)
    up = np.array([0.0, 1.0, 0.0])

    for i in range(num_frames):
        angle = 2.0 * math.pi * i / num_frames
        eye_x = center[0] + radius * math.cos(angle)
        eye_z = center[2] + radius * math.sin(angle)
        eye_y = center[1] + height_offset
        eye = np.array([eye_x, eye_y, eye_z])
        poses.append((eye, target, up))

    return poses


if __name__ == "__main__":
    if not _O3D_AVAILABLE:
        sys.exit(
            "[ERROR] open3d is required for render_360.py but is not installed.\n"
            "  On Kaggle/headless: apt-get install -y libgl1-mesa-glx && pip install open3d\n"
            "  On desktop:         pip install open3d"
        )
    if len(sys.argv) < 2:
        sys.exit("[ERROR] Please specify a 3D mesh or point cloud file (.ply, .obj, .glb) on the command line.\nUsage: python visualization/render_360.py <path_to_mesh_or_pcd>")

    mesh_file = Path(sys.argv[1])
    if not mesh_file.exists():
        sys.exit(f"[ERROR] Specified file does not exist: {mesh_file}")

    if mesh_file.suffix.lower() in VIDEO_EXTENSIONS:
        sys.exit(
            f"[ERROR] '{mesh_file.name}' is a video file.\n"
            f"render_360.py renders 3D geometry files (.ply, .obj, .glb), not video files."
        )

    out_video = config.OUTPUT_DIR / f"{mesh_file.stem}_360.mp4"
    print(f"[+] Output 360 orbit video target: {out_video}")

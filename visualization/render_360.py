# -*- coding: utf-8 -*-
"""
visualization/render_360.py — Renders 360-degree orbit video of 3D meshes and point clouds.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from visualization.local_renderer import render_360_orbit_video

if __name__ == "__main__":
    if len(sys.argv) < 2:
        geometry = config.OUTPUT_DIR / "world_pointcloud.ply"
        if not geometry.exists():
            sys.exit(
                "[ERROR] Please specify a 3D mesh or point cloud file (.ply, .obj, .glb).\n"
                "Usage: python visualization/render_360.py <path_to_mesh_or_pcd> [out_video.mp4]"
            )
    else:
        geometry = Path(sys.argv[1])

    out_video = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
    render_360_orbit_video(geometry, output_video_path=out_video)

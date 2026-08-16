# -*- coding: utf-8 -*-
"""
spatial package — 3D Spatial Reconstruction Engine.
"""

from spatial.room_builder import RoomBuilder, detect_architectural_planes, build_room_background
from spatial.object_estimator import ObjectEstimator, backproject_mask_to_3d
from spatial.mesh_placer import MeshPlacer, snap_mesh_to_surface, snap_mesh_to_wall, assemble_full_scene

__all__ = [
    "RoomBuilder",
    "detect_architectural_planes",
    "build_room_background",
    "ObjectEstimator",
    "backproject_mask_to_3d",
    "MeshPlacer",
    "snap_mesh_to_surface",
    "snap_mesh_to_wall",
    "assemble_full_scene",
]

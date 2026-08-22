# -*- coding: utf-8 -*-
"""
spatial package — 3D Spatial Reconstruction Engine with Mask3D Instance Segmentation.
"""

from spatial.room_builder import (
    RoomBuilder,
    detect_architectural_planes,
    build_room_background,
    inpaint_room_structural_planes,
)
from spatial.object_extractor import (
    Mask3DExtractor,
    Mask3DRunner,
    Mask3DPreprocessor,
    ObjectExtractor,
    extract_object_pointclouds,
    extract_object_points_from_world_pcd_view,
)
from spatial.pointcloud_completer import (
    PointCloudCompleter,
    complete_object_pointclouds,
    complete_single_pointcloud,
)
from spatial.object_mesher import (
    ObjectMesher,
    reconstruct_object_mesh,
    reconstruct_object_meshes,
)
from spatial.object_estimator import (
    ObjectEstimator,
    backproject_mask_to_3d,
    process_object_detections,
)
from spatial.mesh_placer import (
    MeshPlacer,
    snap_mesh_to_surface,
    snap_mesh_to_wall,
    align_and_place_object_meshes,
    assemble_full_scene,
)

__all__ = [
    "RoomBuilder",
    "detect_architectural_planes",
    "build_room_background",
    "inpaint_room_structural_planes",
    "Mask3DExtractor",
    "Mask3DRunner",
    "Mask3DPreprocessor",
    "ObjectExtractor",
    "extract_object_pointclouds",
    "extract_object_points_from_world_pcd_view",
    "PointCloudCompleter",
    "complete_object_pointclouds",
    "complete_single_pointcloud",
    "ObjectMesher",
    "reconstruct_object_mesh",
    "reconstruct_object_meshes",
    "ObjectEstimator",
    "backproject_mask_to_3d",
    "process_object_detections",
    "MeshPlacer",
    "snap_mesh_to_surface",
    "snap_mesh_to_wall",
    "align_and_place_object_meshes",
    "assemble_full_scene",
]

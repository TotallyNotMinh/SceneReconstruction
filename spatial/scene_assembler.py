# -*- coding: utf-8 -*-
"""
spatial/scene_assembler.py — Main 3D Spatial Reconstruction & Scene Assembler Engine
"""

import sys
import io
import numpy as np
import trimesh
from pathlib import Path

import config
from core.data_loader import DataLoader
from spatial.room_builder import RoomBuilder
from spatial.object_estimator import ObjectEstimator
from spatial.mesh_placer import MeshPlacer
from detection.sam_segmentor import SAMSegmentor


def _check_inputs():
    required = [
        config.PROCESSED_DATA_DIR / "ar_metadata.json",
        config.PROCESSED_DATA_DIR / "world_pointcloud.ply",
        config.PROCESSED_DATA_DIR / "room_layout.obj",
        config.PROCESSED_DATA_DIR / "detections.json",
        config.PROCESSED_DATA_DIR / "depth_maps.npz",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("[Pipeline] ERROR: Required input files are missing:")
        for m in missing:
            print(f"  - {m}")
        print("\nRun pointcloud and detection pipelines first:")
        print("  python pointcloud/depth_inference.py <video>")
        print("  python detection/reid_tracker.py <video>")
        sys.exit(1)


def run_pipeline():
    print("\n=== STARTING SPATIAL ENGINE PIPELINE ===\n")

    _check_inputs()

    ar_json    = config.PROCESSED_DATA_DIR / "ar_metadata.json"
    ply_file   = config.PROCESSED_DATA_DIR / "world_pointcloud.ply"
    det_json   = config.PROCESSED_DATA_DIR / "detections.json"
    depth_npz  = config.PROCESSED_DATA_DIR / "depth_maps.npz"
    room_file  = config.PROCESSED_DATA_DIR / "room_layout.obj"

    intrinsics, valid_frames = DataLoader.load_ar_metadata(ar_json)
    pcd_pts      = DataLoader.load_point_cloud(ply_file, config.VOXEL_SIZE_PCD)
    detections   = DataLoader.load_detections(det_json)
    depth_maps   = DataLoader.load_depth_maps(depth_npz)

    room_scene       = RoomBuilder.extract_clean_architectural_room(room_file)
    support_surfaces = RoomBuilder.detect_support_surfaces_ransac(pcd_pts)

    device = "cuda" if _cuda_available() else "cpu"
    sam = SAMSegmentor(config.SAM_CHECKPOINT_PATH, device=device)

    all_frame_ids = set()
    for b_data in detections.values():
        for view in b_data.get("associated_views", []):
            all_frame_ids.add(view["frame_id"])

    video_path = None
    for arg in sys.argv[1:]:
        if arg.startswith("--video="):
            video_path = Path(arg.split("=", 1)[1])
        elif arg.endswith((".mp4", ".mov", ".avi", ".mkv")):
            video_path = Path(arg)

    if video_path is None or not Path(video_path).exists():
        candidates = list(config.RAW_DATA_DIR.glob("*.mp4")) + list(config.RAW_DATA_DIR.glob("*.mov")) \
                     + list((config.DATA_DIR).rglob("*.mov")) + list((config.DATA_DIR).rglob("*.mp4"))
        if candidates:
            video_path = candidates[0]

    rgb_frames: dict = {}
    if video_path is not None and Path(video_path).exists() and all_frame_ids:
        print(f"[Pipeline] Loading RGB frames from real video: {video_path}")
        rgb_frames = DataLoader.load_rgb_frames(Path(video_path), list(all_frame_ids))
    else:
        print("[Pipeline] WARNING: source video not found — SAM will fall back to bbox-fill masks.")

    placed_objects = {}

    for obj_id, b_data in detections.items():
        print(f"\n[Pipeline] Processing object: {obj_id}")

        obj_info = ObjectEstimator.estimate_object_3d_properties(
            associated_views  = b_data["associated_views"],
            valid_frames_dict = valid_frames,
            intrinsics        = intrinsics,
            depth_maps        = depth_maps,
            sam_segmentor     = sam,
            rgb_frames        = rgb_frames,
            alpha             = config.ALPHA_SHAPE_ALPHA,
            max_faces         = config.ALPHA_MAX_FACES,
            voxel_size        = config.VOXEL_SIZE_PCD,
        )

        if obj_info is None:
            print(f"[Pipeline] WARNING: insufficient 3D points for '{obj_id}' — skipping.")
            continue

        source_mesh = obj_info.get("mesh")
        if source_mesh is None:
            dx, dy, dz = obj_info["size"]
            source_mesh = trimesh.creation.box(extents=[max(dx, 0.1),
                                                         max(dy, 0.1),
                                                         max(dz, 0.1)])
            print(f"[Pipeline] Alpha-shape unavailable for '{obj_id}' — using proxy box.")
        else:
            print(f"[Pipeline] Alpha-shape mesh: {len(source_mesh.faces)} faces.")

        aligned_mesh = MeshPlacer.align_and_place_mesh(
            triposr_mesh          = source_mesh,
            obj_3d_info           = obj_info,
            support_surfaces      = support_surfaces,
            max_distortion_thresh = config.MAX_DISTORTION_THRESH,
        )

        placed_objects[obj_id] = aligned_mesh

    final_scene = trimesh.Scene()

    for node_name, geom in room_scene.geometry.items():
        final_scene.add_geometry(geom, node_name=node_name)

    for obj_id, mesh in placed_objects.items():
        final_scene.add_geometry(mesh, node_name=f"Editable_Object_{obj_id}")

    output_path = config.OUTPUT_DIR / "digital_twin_scene.glb"
    final_scene.export(str(output_path))

    print(
        "\n=================================================================="
    )
    print(f" SUCCESS: Digital twin exported to:\n {output_path}")
    print(
        "==================================================================\n"
    )


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    run_pipeline()

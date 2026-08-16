# -*- coding: utf-8 -*-
"""
spatial/object_estimator.py — Unified Phase 2 3D Object Reconstruction Pipeline.

Orchestrates Phase 2A (spatial/object_extractor.py) and Phase 2B (spatial/object_mesher.py)
for complete end-to-end 3D point cloud extraction and surface mesh reconstruction.

Maintains 100% backward compatibility for existing callers and test suites.
"""

import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

from spatial.object_extractor import (
    ObjectExtractor,
    extract_object_pointclouds,
    extract_object_points_from_world_pcd_view,
    filter_object_pointcloud_dbscan,
    load_world_pointcloud,
    _filter_plane_inliers,
    _build_2d_mask,
)

from spatial.object_mesher import (
    ObjectMesher,
    reconstruct_object_mesh,
    reconstruct_object_meshes,
)

# Backward compatibility alias
reconstruct_object_mesh_alpha_shape = reconstruct_object_mesh


def backproject_mask_to_3d(
    mask_2d: np.ndarray,
    depth_map: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    rgb_img: Optional[np.ndarray] = None,
    depth_min: float = config.DEPTH_METRIC_MIN,
    depth_max: float = config.DEPTH_METRIC_MAX,
    foreground_margin: float = config.OBJECT_DEPTH_FOREGROUND_MARGIN,
    stride: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Back-project 2D mask pixels with depth into a 3D world-space point cloud.
    Maintained for direct single-frame back-projection utilities and backward compatibility.
    """
    H, W = depth_map.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Valid mask & depth condition
    valid_mask = (mask_2d > 0) & (depth_map >= depth_min) & (depth_map <= depth_max) & np.isfinite(depth_map)
    if not np.any(valid_mask):
        return np.zeros((0, 3), dtype=np.float64), None

    # Adaptive Foreground Depth Gating
    if foreground_margin > 0:
        masked_depths = depth_map[valid_mask]
        if len(masked_depths) > 10:
            near_z = float(np.percentile(masked_depths, 10.0))
            effective_margin = max(foreground_margin, 0.85)
            max_allowed_z = near_z + effective_margin
            valid_mask = valid_mask & (depth_map <= max_allowed_z)
            if not np.any(valid_mask):
                return np.zeros((0, 3), dtype=np.float64), None

    v_coords, u_coords = np.where(valid_mask)
    if stride > 1:
        v_coords = v_coords[::stride]
        u_coords = u_coords[::stride]

    Z = depth_map[v_coords, u_coords].astype(np.float64)
    X_c = (u_coords - cx) * Z / fx
    Y_c = (v_coords - cy) * Z / fy

    # 3D points in camera space (OpenCV +Y Down, +Z Forward)
    pts_cam = np.column_stack([X_c, Y_c, Z])
    pts_cam_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1), dtype=np.float64)])

    # Transform to World coordinates (+Y Up)
    pts_world = (c2w @ pts_cam_h.T).T[:, :3]

    if rgb_img is not None:
        if rgb_img.shape[:2] != (H, W):
            rgb_img = cv2.resize(rgb_img, (W, H), interpolation=cv2.INTER_LINEAR)
        colors = rgb_img[v_coords, u_coords].astype(np.uint8)
    else:
        colors = None

    return pts_world, colors


def process_object_detections(
    detections_path: Optional[Path | str] = None,
    raw_depths_path: Optional[Path | str] = None,
    ar_metadata_path: Optional[Path | str] = None,
    world_pcd_path: Optional[Path | str] = None,
    plane_data_path: Optional[Path | str] = None,
    out_dir: Optional[Path | str] = None,
    filter_planes: bool = False,
) -> Dict[str, Any]:
    """
    Unified Phase 2 Pipeline:
    1. Phase 2A: Exact Point Cloud Extraction (spatial/object_extractor.py)
    2. Phase 2B: Surface Mesh Reconstruction (spatial/object_mesher.py)
    """
    if out_dir is None:
        out_dir = config.PROCESSED_DATA_DIR / "objects"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[ObjectEstimator] Stage 2A: Extracting exact 3D point clouds for detected objects...")
    extracted_manifest = extract_object_pointclouds(
        detections_path=detections_path,
        raw_depths_path=raw_depths_path,
        ar_metadata_path=ar_metadata_path,
        world_pcd_path=world_pcd_path,
        plane_data_path=plane_data_path,
        out_dir=out_dir,
        filter_planes=filter_planes,
    )

    if not extracted_manifest:
        print("[ObjectEstimator] No object point clouds extracted. Skipping meshing stage.")
        return {}

    print(f"[ObjectEstimator] Stage 2B: Reconstructing 3D surface meshes from point clouds ({len(extracted_manifest)} objects)...")
    mesh_manifest = reconstruct_object_meshes(
        objects_dir=out_dir,
        manifest_path=out_dir / "objects_manifest.json",
        out_dir=out_dir,
    )

    return mesh_manifest


class ObjectEstimator:
    """Class wrapper for complete Phase 2 3D Object Reconstruction."""

    def __init__(
        self,
        detections_path: Optional[Path | str] = None,
        raw_depths_path: Optional[Path | str] = None,
        ar_metadata_path: Optional[Path | str] = None,
        world_pcd_path: Optional[Path | str] = None,
        plane_data_path: Optional[Path | str] = None,
    ):
        self.detections_path = Path(detections_path) if detections_path else config.PROCESSED_DATA_DIR / "detections.json"
        self.raw_depths_path = Path(raw_depths_path) if raw_depths_path is not None else None
        self.ar_metadata_path = Path(ar_metadata_path) if ar_metadata_path else config.PROCESSED_DATA_DIR / "ar_metadata.json"
        self.world_pcd_path = Path(world_pcd_path) if world_pcd_path else config.PROCESSED_DATA_DIR / "world_pointcloud.ply"
        self.plane_data_path = Path(plane_data_path) if plane_data_path else config.PROCESSED_DATA_DIR / "detected_planes.json"

    def run(self) -> Dict[str, Any]:
        return process_object_detections(
            self.detections_path,
            raw_depths_path=self.raw_depths_path,
            ar_metadata_path=self.ar_metadata_path,
            world_pcd_path=self.world_pcd_path,
            plane_data_path=self.plane_data_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Object 3D Point Cloud Extraction & Surface Meshing")
    parser.add_argument("--detections", type=str, default=str(config.PROCESSED_DATA_DIR / "detections.json"),
                        help="Path to detections.json file")
    parser.add_argument("--depths", type=str, default=str(config.PROCESSED_DATA_DIR / "raw_depths.npz"),
                        help="Path to raw_depths.npz file")
    parser.add_argument("--world-pcd", type=str, default=str(config.PROCESSED_DATA_DIR / "world_pointcloud.ply"),
                        help="Path to world_pointcloud.ply file")
    parser.add_argument("--planes-json", type=str, default=str(config.PROCESSED_DATA_DIR / "detected_planes.json"),
                        help="Path to detected_planes.json file")
    parser.add_argument("--out-dir", type=str, default=str(config.PROCESSED_DATA_DIR / "objects"),
                        help="Output directory for reconstructed object 3D meshes")
    args = parser.parse_args()

    process_object_detections(
        detections_path=args.detections,
        raw_depths_path=args.depths,
        world_pcd_path=args.world_pcd,
        plane_data_path=args.planes_json,
        out_dir=args.out_dir,
    )

# -*- coding: utf-8 -*-
"""
spatial/room_builder.py — Phase 1: Architectural Plane Detection via RANSAC.

Reads a 3D point cloud (world_pointcloud.ply), extracts dominant planar surfaces
(Floor, Tabletop/Support surfaces) using RANSAC plane fitting, and exports:
1. data/processed/room_layout.obj (Visual layout mesh)
2. data/processed/detected_planes.json (Plane equations ax + by + cz + d = 0 and bounds)
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def _generate_synthetic_room_pcd(out_ply: Path, num_points: int = 10000) -> np.ndarray:
    """Generate a synthetic room point cloud for testing when no real PLY exists."""
    print(f"[RoomBuilder] Generating synthetic room point cloud for testing ({num_points} points)...")
    rng = np.random.default_rng(42)

    # Floor at Y = 0.0, bounds X: [-3, 3], Z: [-3, 3]
    n_floor = int(num_points * 0.5)
    floor_x = rng.uniform(-3.0, 3.0, n_floor)
    floor_z = rng.uniform(-3.0, 3.0, n_floor)
    floor_y = rng.normal(0.0, 0.01, n_floor)
    floor_pts = np.column_stack([floor_x, floor_y, floor_z])
    floor_cols = np.tile([180, 180, 180], (n_floor, 1)).astype(np.uint8)

    # Tabletop at Y = 0.75, bounds X: [-0.8, 0.8], Z: [-0.5, 0.5]
    n_table = int(num_points * 0.25)
    table_x = rng.uniform(-0.8, 0.8, n_table)
    table_z = rng.uniform(-0.5, 0.5, n_table)
    table_y = rng.normal(0.75, 0.008, n_table)
    table_pts = np.column_stack([table_x, table_y, table_z])
    table_cols = np.tile([130, 90, 50], (n_table, 1)).astype(np.uint8)

    # Object / Chair clutter points
    n_obj = num_points - n_floor - n_table
    obj_x = rng.uniform(-1.5, 1.5, n_obj)
    obj_z = rng.uniform(-1.5, 1.5, n_obj)
    obj_y = rng.uniform(0.1, 1.2, n_obj)
    obj_pts = np.column_stack([obj_x, obj_y, obj_z])
    obj_cols = np.tile([50, 120, 200], (n_obj, 1)).astype(np.uint8)

    pts = np.vstack([floor_pts, table_pts, obj_pts])
    cols = np.vstack([floor_cols, table_cols, obj_cols])

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    if HAS_TRIMESH:
        cloud = trimesh.PointCloud(vertices=pts, colors=cols)
        cloud.export(str(out_ply))
    elif HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols / 255.0)
        o3d.io.write_point_cloud(str(out_ply), pcd)

    print(f"[RoomBuilder] Saved synthetic point cloud -> {out_ply}")
    return pts


def detect_architectural_planes(
    ply_path: Optional[Path | str] = None,
    distance_threshold: float = config.RANSAC_DISTANCE_THRESH,
    ransac_n: int = config.RANSAC_N,
    num_iterations: int = config.RANSAC_NUM_ITERATIONS,
    max_planes: int = 5,
    min_inliers: int = 100,
    out_obj: Optional[Path | str] = None,
    out_json: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Detect dominant architectural planes (Floor & Tabletop surfaces) using RANSAC.

    Parameters
    ----------
    ply_path : Path to input point cloud (.ply). If None or missing, auto-generates test data.
    distance_threshold : Max distance in meters for a point to be an inlier of a plane.
    ransac_n : Number of sampled points to estimate plane equation.
    num_iterations : Maximum RANSAC iterations.
    max_planes : Maximum number of sequential RANSAC planes to extract.
    min_inliers : Minimum number of points required to form a valid plane.
    out_obj : Output path for visual layout mesh (.obj).
    out_json : Output path for plane equations JSON (.json).

    Returns
    -------
    Dict containing:
      - 'floor': dict with plane equation [a,b,c,d], height, inlier_count, bounds
      - 'tables': list of dicts for detected tabletop planes
      - 'all_planes': list of all raw detected RANSAC planes
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for RANSAC plane fitting. Install via: pip install open3d")

    if ply_path is None:
        ply_path = config.PROCESSED_DATA_DIR / "world_pointcloud.ply"
    ply_path = Path(ply_path)

    if not ply_path.exists():
        print(f"[RoomBuilder] Point cloud file not found: {ply_path}")
        _generate_synthetic_room_pcd(ply_path)

    print(f"[RoomBuilder] Loading point cloud for RANSAC plane detection: '{ply_path.name}'...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        raise ValueError(f"[RoomBuilder] Point cloud at {ply_path} contains 0 points.")

    print(f"             Total points: {len(pcd.points):,}")

    remaining_pcd = pcd
    raw_planes: List[Dict[str, Any]] = []

    for idx in range(max_planes):
        if len(remaining_pcd.points) < min_inliers:
            break

        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

        if len(inliers) < min_inliers:
            break

        a, b, c, d = plane_model
        normal = np.array([a, b, c], dtype=np.float64)
        norm_val = np.linalg.norm(normal)
        if norm_val > 1e-6:
            normal = normal / norm_val
            d = d / norm_val

        inlier_pcd = remaining_pcd.select_by_index(inliers)
        inlier_pts = np.asarray(inlier_pcd.points)
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

        bbox = inlier_pcd.get_axis_aligned_bounding_box()
        min_b = bbox.get_min_bound().tolist()
        max_b = bbox.get_max_bound().tolist()
        mean_y = float(np.mean(inlier_pts[:, 1]))

        # Check if plane is horizontal (normal vector aligned with Y axis, i.e. |normal[1]| > tolerance)
        is_horizontal = abs(normal[1]) >= config.ROOM_FLOOR_NORMAL_TOLERANCE

        raw_planes.append({
            "id": idx,
            "equation": [float(normal[0]), float(normal[1]), float(normal[2]), float(d)],
            "normal": normal.tolist(),
            "mean_y": mean_y,
            "is_horizontal": bool(is_horizontal),
            "inlier_count": int(len(inliers)),
            "min_bound": min_b,
            "max_bound": max_b,
        })

    if not raw_planes:
        print("[RoomBuilder] WARNING: No RANSAC planes were detected.")
        return {"floor": None, "tables": [], "all_planes": []}

    # Filter horizontal planes to separate Floor vs Tabletop
    horizontal_planes = [p for p in raw_planes if p["is_horizontal"]]
    if not horizontal_planes:
        # Fallback to all planes sorted by Y height
        horizontal_planes = sorted(raw_planes, key=lambda p: p["mean_y"])

    # Floor identification: Among the lowest horizontal planes (bottom 40% height range or lowest 3),
    # pick the plane with the largest inlier count (dominant support surface)
    horizontal_planes.sort(key=lambda p: p["mean_y"])
    lowest_y = horizontal_planes[0]["mean_y"]
    floor_candidates = [p for p in horizontal_planes if (p["mean_y"] - lowest_y) <= 0.20]
    floor_plane = max(floor_candidates, key=lambda p: p["inlier_count"])

    # Table planes: Horizontal planes at least 0.25m above the floor level
    table_planes = [
        p for p in horizontal_planes
        if (p["mean_y"] - floor_plane["mean_y"]) >= 0.25
    ]

    print(f"[RoomBuilder] RANSAC Plane Detection complete:")
    print(f"             - Floor Plane Detected  : Y = {floor_plane['mean_y']:.3f}m (Inliers: {floor_plane['inlier_count']:,})")
    for t_idx, tp in enumerate(table_planes):
        print(f"             - Tabletop Plane #{t_idx+1}      : Y = {tp['mean_y']:.3f}m (Inliers: {tp['inlier_count']:,})")

    # Export detected plane JSON metadata
    result = {
        "floor": floor_plane,
        "tables": table_planes,
        "all_planes": raw_planes,
    }

    if out_json is None:
        out_json = config.PROCESSED_DATA_DIR / "detected_planes.json"
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[RoomBuilder] Plane metadata saved -> {out_json}")

    # Generate 3D visual layout mesh for Floor and Tables (.obj)
    if out_obj is None:
        out_obj = config.PROCESSED_DATA_DIR / "room_layout.obj"
    out_obj = Path(out_obj)
    out_obj.parent.mkdir(parents=True, exist_ok=True)

    _export_room_layout_mesh(result, out_obj)

    return result


def _export_room_layout_mesh(plane_data: Dict[str, Any], out_obj: Path):
    """Create lightweight box meshes for Floor and Table surfaces for visualization."""
    if not HAS_TRIMESH and not HAS_OPEN3D:
        print("[RoomBuilder] Neither trimesh nor open3d available; skipping layout mesh export.")
        return

    if HAS_TRIMESH:
        meshes: List[trimesh.Trimesh] = []

        floor = plane_data.get("floor")
        if floor:
            min_b = floor["min_bound"]
            max_b = floor["max_bound"]
            size_x = max(0.5, max_b[0] - min_b[0])
            size_z = max(0.5, max_b[2] - min_b[2])
            size_y = 0.02  # 2cm slab thickness

            box = trimesh.creation.box(extents=[size_x, size_y, size_z])
            center_x = (min_b[0] + max_b[0]) / 2.0
            center_z = (min_b[2] + max_b[2]) / 2.0
            center_y = floor["mean_y"] - (size_y / 2.0)
            box.apply_translation([center_x, center_y, center_z])
            box.visual.vertex_colors = np.tile([200, 70, 70, 255], (len(box.vertices), 1)).astype(np.uint8)
            meshes.append(box)

        tables = plane_data.get("tables", [])
        for tp in tables:
            min_b = tp["min_bound"]
            max_b = tp["max_bound"]
            size_x = max(0.3, max_b[0] - min_b[0])
            size_z = max(0.3, max_b[2] - min_b[2])
            size_y = 0.03  # 3cm tabletop slab thickness

            box = trimesh.creation.box(extents=[size_x, size_y, size_z])
            center_x = (min_b[0] + max_b[0]) / 2.0
            center_z = (min_b[2] + max_b[2]) / 2.0
            center_y = tp["mean_y"] - (size_y / 2.0)
            box.apply_translation([center_x, center_y, center_z])
            box.visual.vertex_colors = np.tile([70, 150, 220, 255], (len(box.vertices), 1)).astype(np.uint8)
            meshes.append(box)

        if meshes:
            combined = trimesh.util.concatenate(meshes)
            combined.export(str(out_obj))
            print(f"[RoomBuilder] Visual room layout mesh exported -> {out_obj}")
    elif HAS_OPEN3D:
        combined_o3d = o3d.geometry.TriangleMesh()
        floor = plane_data.get("floor")
        if floor:
            min_b = floor["min_bound"]
            max_b = floor["max_bound"]
            size_x = max(0.5, max_b[0] - min_b[0])
            size_z = max(0.5, max_b[2] - min_b[2])
            size_y = 0.02
            box = o3d.geometry.TriangleMesh.create_box(width=size_x, height=size_y, depth=size_z)
            center_x = (min_b[0] + max_b[0]) / 2.0 - (size_x / 2.0)
            center_z = (min_b[2] + max_b[2]) / 2.0 - (size_z / 2.0)
            center_y = floor["mean_y"] - size_y
            box.translate([center_x, center_y, center_z])
            box.paint_uniform_color([0.78, 0.27, 0.27])
            combined_o3d += box

        if len(combined_o3d.vertices) > 0:
            o3d.io.write_triangle_mesh(str(out_obj), combined_o3d)
            print(f"[RoomBuilder] Visual room layout mesh exported via Open3D -> {out_obj}")


class RoomBuilder:
    """Class wrapper for Architectural Plane Detection."""

    def __init__(self, ply_path: Optional[Path | str] = None):
        self.ply_path = Path(ply_path) if ply_path else config.PROCESSED_DATA_DIR / "world_pointcloud.ply"
        self.plane_data: Optional[Dict[str, Any]] = None

    def run(self) -> Dict[str, Any]:
        self.plane_data = detect_architectural_planes(self.ply_path)
        return self.plane_data

    def get_floor_height(self) -> float:
        if self.plane_data is None:
            self.run()
        floor = self.plane_data.get("floor") if self.plane_data else None
        return floor["mean_y"] if floor else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Architectural Plane Detection via RANSAC")
    parser.add_argument("ply", type=str, nargs="?", default=str(config.PROCESSED_DATA_DIR / "world_pointcloud.ply"),
                        help="Input .ply point cloud file")
    parser.add_argument("--distance-thresh", type=float, default=config.RANSAC_DISTANCE_THRESH,
                        help="Max distance threshold for plane inliers in meters (default: 0.03)")
    parser.add_argument("--max-planes", type=int, default=5,
                        help="Max RANSAC planes to extract (default: 5)")
    parser.add_argument("--out-obj", type=str, default=str(config.PROCESSED_DATA_DIR / "room_layout.obj"),
                        help="Output path for layout mesh .obj file")
    args = parser.parse_args()

    detect_architectural_planes(
        ply_path=args.ply,
        distance_threshold=args.distance_thresh,
        max_planes=args.max_planes,
        out_obj=args.out_obj,
    )

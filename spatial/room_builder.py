# -*- coding: utf-8 -*-
"""
spatial/room_builder.py — Phase 1: Architectural Plane Detection via RANSAC.

Reads a 3D point cloud (world_pointcloud.ply), extracts dominant planar surfaces
(Floor, Tabletop, and Vertical Walls) using RANSAC plane fitting, and exports:
1. data/processed/room_layout.obj (Visual layout mesh with Oriented Bounding Boxes)
2. data/processed/detected_planes.json (Plane equations ax + by + cz + d = 0 and oriented bounds)
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


def _compute_oriented_wall_geometry(
    inlier_pts: np.ndarray,
    normal: np.ndarray,
    floor_y: float = 0.0,
    thickness: float = config.WALL_THICKNESS,
) -> Dict[str, Any]:
    """
    Compute Oriented Bounding Box (OBB) parameters for a vertical wall plane.

    Parameters
    ----------
    inlier_pts : (N, 3) points belonging to the wall plane.
    normal : (3,) unit normal vector of the plane [nx, ny, nz].
    floor_y : Ground floor level Y coordinate (m).
    thickness : Wall slab thickness in meters.

    Returns
    -------
    Dict containing oriented wall geometry (u_tangent, length, height, center, transform_matrix).
    """
    # Project normal to horizontal X-Z plane
    n_xz = np.array([normal[0], 0.0, normal[2]], dtype=np.float64)
    norm_len = np.linalg.norm(n_xz)
    if norm_len < 1e-6:
        n_xz = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n_xz = n_xz / norm_len

    # Horizontal tangent vector u along the wall surface length
    u_tangent = np.array([-n_xz[2], 0.0, n_xz[0]], dtype=np.float64)
    v_vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Ensure right-handed coordinate frame: det([u, v, n]) == +1
    R_mat = np.column_stack([u_tangent, v_vertical, n_xz])
    if np.linalg.det(R_mat) < 0:
        u_tangent = -u_tangent
        R_mat = np.column_stack([u_tangent, v_vertical, n_xz])

    # Project inlier points onto horizontal tangent u and normal n_xz
    u_coords = inlier_pts @ u_tangent
    n_coords = inlier_pts @ n_xz
    y_coords = inlier_pts[:, 1]

    u_min, u_max = float(np.min(u_coords)), float(np.max(u_coords))
    length = max(0.40, u_max - u_min)
    u_mid = (u_min + u_max) / 2.0
    n_mid = float(np.mean(n_coords))

    # Height span: clamp wall base to floor level if points are near floor (<= 0.20m)
    observed_min_y = float(np.min(y_coords))
    observed_max_y = float(np.max(y_coords))
    base_y = floor_y if abs(observed_min_y - floor_y) <= 0.20 else observed_min_y
    top_y = max(base_y + 0.60, observed_max_y)
    height = top_y - base_y
    center_y = base_y + (height / 2.0)

    center_xz = (u_mid * u_tangent) + (n_mid * n_xz)
    center_3d = np.array([center_xz[0], center_y, center_xz[2]], dtype=np.float64)

    # 4x4 Transformation Matrix
    T_mat = np.eye(4, dtype=np.float64)
    T_mat[:3, :3] = R_mat
    T_mat[:3, 3] = center_3d

    return {
        "length": length,
        "height": height,
        "thickness": thickness,
        "center": center_3d.tolist(),
        "u_tangent": u_tangent.tolist(),
        "normal_xz": n_xz.tolist(),
        "transform_matrix": T_mat.tolist(),
    }


def detect_architectural_planes(
    ply_path: Optional[Path | str] = None,
    distance_threshold: float = config.RANSAC_DISTANCE_THRESH,
    ransac_n: int = config.RANSAC_N,
    num_iterations: int = config.RANSAC_NUM_ITERATIONS,
    max_planes: int = getattr(config, "RANSAC_MAX_PLANES", 12),
    min_inliers: int = 150,
    out_obj: Optional[Path | str] = None,
    out_json: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Detect dominant architectural planes (Floor, Tables, Walls, Ceilings) using RANSAC.

    Parameters
    ----------
    ply_path : Path to input point cloud (.ply).
    distance_threshold : Max distance in meters for a point to be an inlier of a plane.
    ransac_n : Number of sampled points to estimate plane hypothesis.
    num_iterations : Maximum RANSAC iterations.
    max_planes : Maximum number of sequential RANSAC planes to extract.
    min_inliers : Minimum number of points required to form a valid plane.
    out_obj : Output path for visual layout mesh (.obj).
    out_json : Output path for plane equations JSON (.json).

    Returns
    -------
    Dict containing detected floor, tables, walls, ceilings, and all_planes metadata.
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for RANSAC plane fitting. Install via: pip install open3d")

    if ply_path is None:
        ply_path = config.PROCESSED_DATA_DIR / "world_pointcloud.ply"
    ply_path = Path(ply_path)

    if not ply_path.exists():
        raise FileNotFoundError(
            f"[RoomBuilder] Point cloud file not found: {ply_path}\n"
            "              Please run Stage 1 (pointcloud_builder.py) first to generate world_pointcloud.ply."
        )

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

        # Check if plane is horizontal (normal vector aligned with Y axis) or vertical (normal in X-Z plane)
        is_horizontal = abs(normal[1]) >= config.ROOM_FLOOR_NORMAL_TOLERANCE
        is_vertical = (not is_horizontal) and (abs(normal[1]) <= config.ROOM_WALL_NORMAL_TOLERANCE)

        raw_planes.append({
            "id": idx,
            "equation": [float(normal[0]), float(normal[1]), float(normal[2]), float(d)],
            "normal": normal.tolist(),
            "mean_y": mean_y,
            "is_horizontal": bool(is_horizontal),
            "is_vertical": bool(is_vertical),
            "inlier_count": int(len(inliers)),
            "inlier_pts": inlier_pts,
            "min_bound": min_b,
            "max_bound": max_b,
        })

    if not raw_planes:
        print("[RoomBuilder] WARNING: No RANSAC planes were detected.")
        return {"floor": None, "tables": [], "walls": [], "ceilings": [], "all_planes": []}

    # Separate horizontal planes (Floor & Tabletop & Ceiling)
    horizontal_planes = [p for p in raw_planes if p["is_horizontal"]]
    if not horizontal_planes:
        horizontal_planes = sorted(
            [p for p in raw_planes if not p.get("is_vertical", False)],
            key=lambda p: p["mean_y"]
        )
        if not horizontal_planes:
            horizontal_planes = sorted(raw_planes, key=lambda p: p["mean_y"])
            print("[RoomBuilder] WARNING: All RANSAC planes appear vertical — floor detection may be inaccurate.")

    # Floor identification: Select dominant support surface among the lowest horizontal planes
    horizontal_planes.sort(key=lambda p: p["mean_y"])
    lowest_y = horizontal_planes[0]["mean_y"]
    floor_candidates = [p for p in horizontal_planes if (p["mean_y"] - lowest_y) <= 0.25]
    floor_plane = max(floor_candidates, key=lambda p: p["inlier_count"])
    floor_y = float(floor_plane["mean_y"])

    # Table planes: Horizontal planes within standard architectural furniture height above floor level
    min_table_h = getattr(config, "TABLE_MIN_HEIGHT", 0.30)
    max_table_h = getattr(config, "TABLE_MAX_HEIGHT", 1.40)
    table_planes = [
        p for p in horizontal_planes
        if min_table_h <= (p["mean_y"] - floor_y) <= max_table_h
    ]

    # Ceiling planes: Horizontal planes high above the floor (higher than tabletop range)
    ceiling_planes = [
        p for p in horizontal_planes
        if (p["mean_y"] - floor_y) > max_table_h
    ]

    # Wall planes: Vertical planes with sufficient inliers
    min_wall_inliers = getattr(config, "ROOM_MIN_WALL_INLIERS", 250)
    raw_wall_planes = [
        p for p in raw_planes
        if p.get("is_vertical", False) and p["inlier_count"] >= min_wall_inliers
    ]

    # Compute Oriented Bounding Box geometry for each wall plane
    wall_planes: List[Dict[str, Any]] = []
    for wp in raw_wall_planes:
        inlier_pts = wp["inlier_pts"]
        norm = np.array(wp["normal"], dtype=np.float64)
        obb_meta = _compute_oriented_wall_geometry(inlier_pts, norm, floor_y=floor_y)
        wp_clean = {
            "id": wp["id"],
            "equation": wp["equation"],
            "normal": wp["normal"],
            "mean_y": wp["mean_y"],
            "inlier_count": wp["inlier_count"],
            "min_bound": wp["min_bound"],
            "max_bound": wp["max_bound"],
            "oriented_box": obb_meta,
        }
        wall_planes.append(wp_clean)

    # Clean inlier_pts from raw_planes before export
    all_planes_clean = []
    for p in raw_planes:
        p_copy = dict(p)
        p_copy.pop("inlier_pts", None)
        all_planes_clean.append(p_copy)

    floor_clean = dict(floor_plane)
    floor_clean.pop("inlier_pts", None)

    tables_clean = []
    for tp in table_planes:
        tp_copy = dict(tp)
        tp_copy.pop("inlier_pts", None)
        tables_clean.append(tp_copy)

    ceilings_clean = []
    for cp in ceiling_planes:
        cp_copy = dict(cp)
        cp_copy.pop("inlier_pts", None)
        ceilings_clean.append(cp_copy)

    print(f"[RoomBuilder] RANSAC Plane Detection complete:")
    print(f"             - Floor Plane Detected  : Y = {floor_y:.3f}m (Inliers: {floor_plane['inlier_count']:,})")
    for t_idx, tp in enumerate(tables_clean):
        print(f"             - Tabletop Plane #{t_idx+1}      : Y = {tp['mean_y']:.3f}m (Inliers: {tp['inlier_count']:,})")
    for c_idx, cp in enumerate(ceilings_clean):
        print(f"             - Ceiling Plane #{c_idx+1}       : Y = {cp['mean_y']:.3f}m (Inliers: {cp['inlier_count']:,})")
    for w_idx, wp in enumerate(wall_planes):
        obb = wp["oriented_box"]
        print(f"             - Wall Plane #{w_idx+1} (Oriented): Length={obb['length']:.2f}m, Height={obb['height']:.2f}m, Normal={wp['normal']}")

    result = {
        "floor": floor_clean,
        "tables": tables_clean,
        "ceilings": ceilings_clean,
        "walls": wall_planes,
        "all_planes": all_planes_clean,
    }

    if out_json is None:
        out_json = config.PROCESSED_DATA_DIR / "detected_planes.json"
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[RoomBuilder] Plane metadata saved -> {out_json}")

    # Generate visual layout mesh (.obj)
    if out_obj is None:
        out_obj = config.PROCESSED_DATA_DIR / "room_layout.obj"
    out_obj = Path(out_obj)
    out_obj.parent.mkdir(parents=True, exist_ok=True)

    _export_room_layout_mesh(result, out_obj)

    return result


def _export_room_layout_mesh(plane_data: Dict[str, Any], out_obj: Path):
    """Create lightweight oriented box meshes for Floor, Tables, and Walls for visualization."""
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

        walls = plane_data.get("walls", [])
        for wp in walls:
            obb = wp.get("oriented_box")
            if obb:
                length = obb["length"]
                height = obb["height"]
                thickness = obb.get("thickness", config.WALL_THICKNESS)
                T_mat = np.array(obb["transform_matrix"], dtype=np.float64)

                # Oriented box centered at origin with local axes (Length: X, Height: Y, Thickness: Z)
                wall_box = trimesh.creation.box(extents=[length, height, thickness])
                wall_box.apply_transform(T_mat)
                wall_box.visual.vertex_colors = np.tile([180, 180, 100, 220], (len(wall_box.vertices), 1)).astype(np.uint8)
                meshes.append(wall_box)

        if meshes:
            combined = trimesh.util.concatenate(meshes)
            combined.export(str(out_obj))
            print(f"[RoomBuilder] Visual room layout mesh (OBB Oriented) exported -> {out_obj}")

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

        walls = plane_data.get("walls", [])
        for wp in walls:
            obb = wp.get("oriented_box")
            if obb:
                length = obb["length"]
                height = obb["height"]
                thickness = obb.get("thickness", config.WALL_THICKNESS)
                T_mat = np.array(obb["transform_matrix"], dtype=np.float64)

                wall_box = o3d.geometry.TriangleMesh.create_box(width=length, height=height, depth=thickness)
                # Center Open3D box at origin
                wall_box.translate([-length / 2.0, -height / 2.0, -thickness / 2.0])
                wall_box.transform(T_mat)
                wall_box.paint_uniform_color([0.70, 0.70, 0.40])
                combined_o3d += wall_box

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
    parser.add_argument("--max-planes", type=int, default=getattr(config, "RANSAC_MAX_PLANES", 12),
                        help="Max RANSAC planes to extract (default: 12)")
    parser.add_argument("--out-obj", type=str, default=str(config.PROCESSED_DATA_DIR / "room_layout.obj"),
                        help="Output path for layout mesh .obj file")
    args = parser.parse_args()

    detect_architectural_planes(
        ply_path=args.ply,
        distance_threshold=args.distance_thresh,
        max_planes=args.max_planes,
        out_obj=args.out_obj,
    )

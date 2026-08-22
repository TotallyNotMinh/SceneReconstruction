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
from typing import Dict, List, Optional, Tuple, Any, Union

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
    detections_path: Optional[Path | str] = None,
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
    detections_path : Optional path to detections.json for semantic plane verification.
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

    # Table planes & Ceiling planes separation with semantic & span verification
    min_table_h = getattr(config, "TABLE_MIN_HEIGHT", 0.30)
    max_table_h = getattr(config, "TABLE_MAX_HEIGHT", 1.40)
    
    table_planes = []
    ceiling_planes = []

    for p in horizontal_planes:
        if p["id"] == floor_plane["id"]:
            continue
        h_diff = p["mean_y"] - floor_y
        if h_diff > max_table_h:
            ceiling_planes.append(p)
        elif min_table_h <= h_diff <= max_table_h:
            # Check inlier footprint span to distinguish genuine tables from small chair cushions or noise
            span_x = p["max_bound"][0] - p["min_bound"][0]
            span_z = p["max_bound"][2] - p["min_bound"][2]
            # Standard desk/table height >= 0.58m or low coffee table with broad surface area
            if h_diff >= 0.58 or (span_x >= 0.35 and span_z >= 0.35) or p["inlier_count"] >= 300:
                table_planes.append(p)

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

    # Generate visual layout CAD bounding box slabs if enabled
    if getattr(config, "EXPORT_ROOM_CAD_SLABS", True):
        if out_obj is None:
            out_obj = config.PROCESSED_DATA_DIR / "room_layout.obj"
        out_obj = Path(out_obj)
        out_obj.parent.mkdir(parents=True, exist_ok=True)
        _export_room_layout_mesh(result, out_obj)

    return result


def build_room_background(
    world_ply_path: Optional[Path | str] = None,
    objects_dir: Optional[Path | str] = None,
    out_pcd_path: Optional[Path | str] = None,
    out_mesh_path: Optional[Path | str] = None,
    plane_data_path: Optional[Path | str] = None,
    subtraction_radius: Optional[float] = None,
    method: Optional[str] = None,
    depth: Optional[int] = None,
    density_trim: Optional[float] = None,
) -> Dict[str, Any]:

    """
    Orchestrate video-accurate room background extraction and 3D surface reconstruction.
    Should be called AFTER Phase 2 (Object Detection & Point Cloud Extraction) has produced objects.

    Parameters
    ----------
    world_ply_path : Path to input world_pointcloud.ply.
    objects_dir : Directory containing segmented object point clouds and objects_manifest.json.
    out_pcd_path : Output path for room_background_pointcloud.ply.
    out_mesh_path : Output path for room_background_mesh.ply.
    subtraction_radius : Spatial radius in meters around object points to prune from world point cloud.
    method : Surface reconstruction algorithm ("poisson", "bpa", "alpha").
    depth : Octree depth for Poisson reconstruction.
    density_trim : Percentile of low-density vertices to trim.

    Returns
    -------
    Dict containing 'room_background_pcd', 'room_background_mesh', 'mesh', 'point_count'.
    """
    if subtraction_radius is None:
        subtraction_radius = getattr(config, "ROOM_OBJECT_SUBTRACTION_RADIUS", 0.05)
    if method is None:
        method = getattr(config, "ROOM_BACKGROUND_MESHING_METHOD", "poisson")
    if depth is None:
        depth = getattr(config, "ROOM_POISSON_DEPTH", 9)
    if density_trim is None:
        density_trim = getattr(config, "ROOM_POISSON_DENSITY_TRIM", 5.0)
    if world_ply_path is None:
        world_ply_path = config.PROCESSED_DATA_DIR / "world_pointcloud.ply"
    world_ply_path = Path(world_ply_path)

    if objects_dir is None:
        objects_dir = config.PROCESSED_DATA_DIR / "objects"
    objects_dir = Path(objects_dir)

    if out_pcd_path is None:
        out_pcd_path = config.PROCESSED_DATA_DIR / "room_background_pointcloud.ply"
    out_pcd_path = Path(out_pcd_path)

    if out_mesh_path is None:
        out_mesh_path = config.PROCESSED_DATA_DIR / "room_background_mesh.ply"
    out_mesh_path = Path(out_mesh_path)

    room_pts, room_cols, saved_pcd_path = extract_room_background_pointcloud(
        world_ply_path=world_ply_path,
        objects_dir=objects_dir,
        out_pcd_path=out_pcd_path,
        subtraction_radius=subtraction_radius,
    )

    # 2. Architectural Inpainting: Fill floor and wall voids where objects were subtracted
    if getattr(config, "ENABLE_ROOM_PLANE_INPAINTING", True):
        cand1 = Path(objects_dir).parent / "detected_planes.json" if objects_dir else None
        p_path = plane_data_path or (cand1 if (cand1 and cand1.exists()) else None)
        if p_path and Path(p_path).exists():
            room_pts, room_cols = inpaint_room_structural_planes(
                room_pts=room_pts,
                room_cols=room_cols,
                plane_data_path=p_path,
            )
            if HAS_TRIMESH:
                pcd_tri = trimesh.PointCloud(vertices=room_pts, colors=room_cols) if room_cols is not None else trimesh.PointCloud(vertices=room_pts)
                pcd_tri.export(str(saved_pcd_path))


    room_mesh = reconstruct_room_background_mesh(
        room_pts=room_pts,
        room_cols=room_cols,
        out_mesh_path=out_mesh_path,
        method=method,
        depth=depth,
        density_trim=density_trim,
    )

    return {
        "room_background_pcd": str(saved_pcd_path),
        "room_background_mesh": str(out_mesh_path),
        "mesh": room_mesh,
        "point_count": len(room_pts),
    }



def extract_room_background_pointcloud(
    world_ply_path: Optional[Path | str] = None,
    objects_dir: Optional[Path | str] = None,
    out_pcd_path: Optional[Path | str] = None,
    subtraction_radius: float = getattr(config, "ROOM_OBJECT_SUBTRACTION_RADIUS", 0.05),
) -> Tuple[np.ndarray, Optional[np.ndarray], Path]:
    """
    Extract the room background point cloud by removing all points belonging to segmented objects.
    """
    if world_ply_path is None:
        world_ply_path = config.PROCESSED_DATA_DIR / "world_pointcloud.ply"
    world_ply_path = Path(world_ply_path)

    if not world_ply_path.exists():
        raise FileNotFoundError(f"[RoomBuilder] World point cloud not found: {world_ply_path}")

    if objects_dir is None:
        objects_dir = config.PROCESSED_DATA_DIR / "objects"
    objects_dir = Path(objects_dir)

    if out_pcd_path is None:
        out_pcd_path = config.PROCESSED_DATA_DIR / "room_background_pointcloud.ply"
    out_pcd_path = Path(out_pcd_path)
    out_pcd_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load world point cloud with safe vertex colors
    world_pts = None
    world_cols = None
    if HAS_TRIMESH:
        try:
            cloud = trimesh.load(str(world_ply_path))
            if isinstance(cloud, trimesh.Scene):
                all_v = []
                all_c = []
                for g in cloud.geometry.values():
                    if hasattr(g, "vertices") and len(g.vertices) > 0:
                        v = np.asarray(g.vertices, dtype=np.float64)
                        all_v.append(v)
                        if hasattr(g, "colors") and g.colors is not None and len(g.colors) == len(v):
                            c = np.asarray(g.colors)[:, :3].astype(np.uint8)
                        elif hasattr(g, "visual") and hasattr(g.visual, "vertex_colors") and g.visual.vertex_colors is not None and len(g.visual.vertex_colors) == len(v):
                            c = np.asarray(g.visual.vertex_colors)[:, :3].astype(np.uint8)
                        else:
                            c = np.tile([180, 180, 180], (len(v), 1)).astype(np.uint8)
                        all_c.append(c)
                if all_v:
                    world_pts = np.vstack(all_v)
                    world_cols = np.vstack(all_c)
            elif isinstance(cloud, trimesh.PointCloud):
                world_pts = np.asarray(cloud.vertices, dtype=np.float64)
                if hasattr(cloud, "colors") and cloud.colors is not None and len(cloud.colors) > 0:
                    world_cols = np.asarray(cloud.colors)[:, :3].astype(np.uint8)
            elif isinstance(cloud, trimesh.Trimesh):
                world_pts = np.asarray(cloud.vertices, dtype=np.float64)
                if hasattr(cloud.visual, "vertex_colors") and cloud.visual.vertex_colors is not None:
                    world_cols = np.asarray(cloud.visual.vertex_colors)[:, :3].astype(np.uint8)
        except Exception:
            world_pts = None

    if (world_pts is None or len(world_pts) == 0) and HAS_OPEN3D:
        try:
            pcd = o3d.io.read_point_cloud(str(world_ply_path))
            if len(pcd.points) > 0:
                world_pts = np.asarray(pcd.points, dtype=np.float64)
                world_cols = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
        except Exception:
            pass

    if world_pts is None or len(world_pts) == 0:
        raise ValueError(f"[RoomBuilder] Failed to load points from: {world_ply_path}")

    # 2. Gather object points from objects_dir using manifest or strict pointcloud files
    all_obj_pts_list = []
    manifest_path = objects_dir / "objects_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
            for obj_id, obj_meta in manifest.items():
                p_file = obj_meta.get("pcd_path")
                if not p_file or not Path(p_file).exists():
                    p_file = obj_meta.get("mesh_path")
                if p_file and Path(p_file).exists():
                    try:
                        c = trimesh.load(str(p_file)) if HAS_TRIMESH else None
                        if c is not None and hasattr(c, "vertices") and len(c.vertices) > 0:
                            all_obj_pts_list.append(np.asarray(c.vertices, dtype=np.float64))
                        elif HAS_OPEN3D:
                            c_o3d = o3d.io.read_point_cloud(str(p_file))
                            if len(c_o3d.points) > 0:
                                all_obj_pts_list.append(np.asarray(c_o3d.points, dtype=np.float64))
                    except Exception:
                        pass
        except Exception as exc:
            print(f"[RoomBuilder] WARNING: Failed reading {manifest_path}: {exc}")

    if not all_obj_pts_list and objects_dir.exists():
        pcd_files = list(objects_dir.glob("*_pointcloud.ply"))
        if not pcd_files:
            # Fallback to meshes if point clouds are not found, excluding layout or scene meshes
            pcd_files = [
                f for f in objects_dir.glob("*.ply")
                if not f.name.endswith("_pointcloud.ply")
                and not f.name.startswith("room_")
                and not f.name.startswith("full_scene")
            ]

        for pf in pcd_files:
            try:
                if HAS_TRIMESH:
                    c = trimesh.load(str(pf))
                    if hasattr(c, "vertices") and len(c.vertices) > 0:
                        all_obj_pts_list.append(np.asarray(c.vertices, dtype=np.float64))
                elif HAS_OPEN3D:
                    c = o3d.io.read_point_cloud(str(pf))
                    if len(c.points) > 0:
                        all_obj_pts_list.append(np.asarray(c.points, dtype=np.float64))
            except Exception:
                pass

    if all_obj_pts_list:
        combined_obj_pts = np.vstack(all_obj_pts_list)
        from scipy.spatial import cKDTree
        tree = cKDTree(combined_obj_pts)
        distances, _ = tree.query(world_pts, k=1)
        keep_mask = distances > subtraction_radius
        room_pts = world_pts[keep_mask]
        room_cols = world_cols[keep_mask] if world_cols is not None else None
        excluded_count = int(np.sum(~keep_mask))
        print(f"[RoomBuilder] Subtracted {excluded_count:,} object points from room point cloud (radius={subtraction_radius*100:.1f}cm).")
    else:
        room_pts = world_pts
        room_cols = world_cols
    if HAS_TRIMESH:
        if room_cols is not None:
            pcd_tri = trimesh.PointCloud(vertices=room_pts, colors=room_cols)
        else:
            pcd_tri = trimesh.PointCloud(vertices=room_pts)
        pcd_tri.export(str(out_pcd_path))
    elif HAS_OPEN3D:
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(room_pts)
        if room_cols is not None:
            pcd_o3d.colors = o3d.utility.Vector3dVector(room_cols / 255.0)
        o3d.io.write_point_cloud(str(out_pcd_path), pcd_o3d)

    print(f"[RoomBuilder] Room background point cloud saved ({len(room_pts):,} pts) -> {out_pcd_path}")
    return room_pts, room_cols, out_pcd_path



def inpaint_room_structural_planes(
    room_pts: np.ndarray,
    room_cols: Optional[np.ndarray],
    planes_data: Optional[Dict[str, Any]] = None,
    plane_data_path: Optional[Union[Path, str]] = None,
    grid_step: float = getattr(config, "ROOM_INPAINTING_GRID_STEP", 0.025),
    gap_threshold: float = getattr(config, "ROOM_INPAINTING_GAP_THRESHOLD", 0.04),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Inpaint dense planar points on the floor and walls where object points were subtracted.

    Prevents sagging and holes in the room point cloud and surface mesh.
    """
    if len(room_pts) == 0:
        return room_pts, room_cols

    if planes_data is None:
        if plane_data_path is None:
            plane_data_path = config.PROCESSED_DATA_DIR / "detected_planes.json"
        plane_data_path = Path(plane_data_path)
        if plane_data_path.exists():
            try:
                with open(plane_data_path, "r", encoding="utf-8") as pf:
                    planes_data = json.load(pf)
            except Exception:
                planes_data = {}
        else:
            planes_data = {}

    if not planes_data:
        return room_pts, room_cols

    from scipy.spatial import cKDTree
    room_tree = cKDTree(room_pts)

    inpaint_pts_list: List[np.ndarray] = []
    inpaint_cols_list: List[np.ndarray] = []

    # 1. Inpaint Floor Plane
    floor_info = planes_data.get("floor")
    if floor_info:
        floor_y = float(floor_info.get("mean_y", 0.0))
        min_b = floor_info.get("min_bound", [-3.0, floor_y, -3.0])
        max_b = floor_info.get("max_bound", [3.0, floor_y, 3.0])

        # Find existing floor points to determine median floor color
        floor_mask = np.abs(room_pts[:, 1] - floor_y) <= 0.04
        if np.any(floor_mask) and room_cols is not None and len(room_cols) == len(room_pts):
            floor_color = np.median(room_cols[floor_mask], axis=0).astype(np.uint8)
        elif room_cols is not None and len(room_cols) > 0:
            floor_color = np.median(room_cols, axis=0).astype(np.uint8)
        else:
            floor_color = np.array([190, 190, 190], dtype=np.uint8)

        # Create floor grid
        xs = np.arange(min_b[0], max_b[0], grid_step)
        zs = np.arange(min_b[2], max_b[2], grid_step)
        if len(xs) > 0 and len(zs) > 0:
            xg, zg = np.meshgrid(xs, zs)
            yg = np.full_like(xg, floor_y)
            floor_grid_pts = np.column_stack([xg.flatten(), yg.flatten(), zg.flatten()])

            # Query distance to nearest room point
            dists, _ = room_tree.query(floor_grid_pts, k=1)
            # A grid point is an empty void if dists >= gap_threshold
            void_mask = (dists >= gap_threshold) & (dists <= 0.80)
            if np.any(void_mask):
                new_floor_pts = floor_grid_pts[void_mask]
                inpaint_pts_list.append(new_floor_pts)
                inpaint_cols_list.append(np.tile(floor_color, (len(new_floor_pts), 1)))

    # 2. Inpaint Vertical Wall Planes
    walls_info = planes_data.get("walls", [])
    for wall in walls_info:
        length = float(wall.get("length", 1.0))
        height = float(wall.get("height", 2.0))
        center = np.array(wall.get("center", [0, 1, 0]), dtype=np.float64)
        u_tangent = np.array(wall.get("u_tangent", [1, 0, 0]), dtype=np.float64)

        # Wall color
        if room_cols is not None and len(room_cols) == len(room_pts):
            wall_nearby = np.linalg.norm(room_pts - center, axis=1) <= 1.0
            if np.any(wall_nearby):
                wall_color = np.median(room_cols[wall_nearby], axis=0).astype(np.uint8)
            else:
                wall_color = np.array([210, 210, 205], dtype=np.uint8)
        else:
            wall_color = np.array([210, 210, 205], dtype=np.uint8)

        # Create wall grid
        u_vals = np.arange(-length / 2.0, length / 2.0, grid_step)
        y_vals = np.arange(-height / 2.0, height / 2.0, grid_step)
        if len(u_vals) > 0 and len(y_vals) > 0:
            ug, yg = np.meshgrid(u_vals, y_vals)
            ug_f = ug.flatten()
            yg_f = yg.flatten()

            wall_grid_pts = center + (ug_f[:, None] * u_tangent) + np.column_stack([np.zeros_like(yg_f), yg_f, np.zeros_like(yg_f)])

            dists, _ = room_tree.query(wall_grid_pts, k=1)
            void_mask = (dists >= gap_threshold) & (dists <= 0.60)
            if np.any(void_mask):
                new_wall_pts = wall_grid_pts[void_mask]
                inpaint_pts_list.append(new_wall_pts)
                inpaint_cols_list.append(np.tile(wall_color, (len(new_wall_pts), 1)))

    if inpaint_pts_list:
        all_inpaint_pts = np.vstack(inpaint_pts_list)
        fused_room_pts = np.vstack([room_pts, all_inpaint_pts])
        if room_cols is not None:
            all_inpaint_cols = np.vstack(inpaint_cols_list)
            fused_room_cols = np.vstack([room_cols, all_inpaint_cols])
        else:
            fused_room_cols = None
        print(f"[RoomBuilder] Architectural Inpainting: Added {len(all_inpaint_pts):,} planar points to seal floor/wall voids.")
        return fused_room_pts, fused_room_cols

    return room_pts, room_cols



def reconstruct_room_background_mesh(
    room_pts: np.ndarray,
    room_cols: Optional[np.ndarray] = None,
    out_mesh_path: Optional[Path | str] = None,
    method: str = getattr(config, "ROOM_BACKGROUND_MESHING_METHOD", "poisson"),
    depth: int = getattr(config, "ROOM_POISSON_DEPTH", 9),
    density_trim: float = getattr(config, "ROOM_POISSON_DENSITY_TRIM", 5.0),
) -> Any:
    """
    Reconstruct a continuous 3D surface mesh for the video-accurate room background.
    Uses Screened Poisson Surface Reconstruction to smoothly inpaint and close occlusion gaps.
    """
    if len(room_pts) < 10:
        print("[RoomBuilder] WARNING: Insufficient room points for surface meshing.")
        return None

    if out_mesh_path is None:
        out_mesh_path = config.PROCESSED_DATA_DIR / "room_background_mesh.ply"
    out_mesh_path = Path(out_mesh_path)
    out_mesh_path.parent.mkdir(parents=True, exist_ok=True)

    from pointcloud.mesh_reconstructor import post_process_mesh

    mesh_o3d = None
    if HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(room_pts)
        if room_cols is not None:
            pcd.colors = o3d.utility.Vector3dVector(room_cols / 255.0)

        # Estimate surface normals
        try:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=min(30, len(room_pts))))
            pcd.orient_normals_consistent_tangent_plane(k=min(30, len(room_pts)))
        except Exception:
            pass

        if method == "poisson":
            try:
                mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=depth, scale=1.1, linear_fit=False
                )
                densities = np.asarray(densities)
                if len(densities) > 0 and density_trim > 0:
                    density_thresh = np.percentile(densities, density_trim)
                    mesh_o3d.remove_vertices_by_mask(densities < density_thresh)
            except Exception as exc:
                print(f"[RoomBuilder] Poisson reconstruction failed ({exc}), trying BPA.")
                mesh_o3d = None

        if mesh_o3d is None or len(mesh_o3d.triangles) < 4:
            # Fallback / method="bpa"
            try:
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = float(np.median(distances)) if len(distances) > 0 else 0.03
                avg_dist = max(avg_dist, 0.005)
                radii_mult = getattr(config, "ROOM_BPA_RADII_MULTIPLIER", [0.8, 1.5, 3.0, 6.0])
                radii = [avg_dist * m for m in radii_mult]
                mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )
            except Exception:
                mesh_o3d = None

        if mesh_o3d is not None and len(mesh_o3d.vertices) > 0 and len(mesh_o3d.triangles) > 0:
            mesh_o3d.remove_degenerate_triangles()
            mesh_o3d.remove_duplicated_triangles()
            mesh_o3d.remove_duplicated_vertices()
            mesh_o3d.remove_non_manifold_edges()

            # Apply post-processing (hole sealing and Taubin smoothing)
            mesh_o3d = post_process_mesh(
                mesh_o3d,
                fill_holes=getattr(config, "FILL_MESH_HOLES", True),
                smooth=True,
                iterations=getattr(config, "MESH_TAUBIN_ITERATIONS", 12),
            )

            if room_cols is not None and len(mesh_o3d.vertices) > 0 and len(room_pts) == len(room_cols):
                mesh_verts = np.asarray(mesh_o3d.vertices)
                finite_v = np.all(np.isfinite(mesh_verts), axis=1)
                finite_pts = np.all(np.isfinite(room_pts), axis=1)
                if np.any(finite_pts) and np.any(finite_v):
                    from scipy.spatial import cKDTree
                    tree = cKDTree(room_pts[finite_pts])
                    valid_query_verts = np.where(np.isfinite(mesh_verts), mesh_verts, 0.0)
                    _, indices = tree.query(valid_query_verts, k=1)
                    v_cols = (room_cols[finite_pts])[indices] / 255.0
                    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(v_cols)

    if mesh_o3d is not None and HAS_TRIMESH and len(mesh_o3d.vertices) > 0 and len(mesh_o3d.triangles) > 0:
        verts = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        v_cols = (np.asarray(mesh_o3d.vertex_colors) * 255).astype(np.uint8) if mesh_o3d.has_vertex_colors() else room_cols
        tri = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=v_cols)
        tri.export(str(out_mesh_path))
        print(f"[RoomBuilder] Video-accurate room background mesh saved ({method.upper()}) -> {out_mesh_path}")
        return tri

    return mesh_o3d


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
    """Class wrapper for Architectural Plane Detection & Room Reconstruction."""

    def __init__(self, ply_path: Optional[Path | str] = None, objects_dir: Optional[Path | str] = None):
        self.ply_path = Path(ply_path) if ply_path else config.PROCESSED_DATA_DIR / "world_pointcloud.ply"
        self.objects_dir = Path(objects_dir) if objects_dir else config.PROCESSED_DATA_DIR / "objects"
        self.plane_data: Optional[Dict[str, Any]] = None

    def run(self) -> Dict[str, Any]:
        self.plane_data = detect_architectural_planes(self.ply_path)
        return self.plane_data

    def build_background(
        self,
        out_pcd_path: Optional[Path | str] = None,
        out_mesh_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Build video-accurate room background by subtracting segmented objects."""
        return build_room_background(
            world_ply_path=self.ply_path,
            objects_dir=self.objects_dir,
            out_pcd_path=out_pcd_path,
            out_mesh_path=out_mesh_path,
        )

    def get_floor_height(self) -> float:
        if self.plane_data is None:
            self.run()
        floor = self.plane_data.get("floor") if self.plane_data else None
        return floor["mean_y"] if floor else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Architectural Plane Detection & Room Reconstruction")
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

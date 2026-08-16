# -*- coding: utf-8 -*-
"""
spatial/object_extractor.py — Phase 2A: 3D Object Point Cloud Extraction & Segmentation.

Extracts exact 3D point cloud clusters for detected objects directly from the reconstructed
world point cloud (world_pointcloud.ply) using 2D instance masks, camera intrinsics, poses,
and depth-consistency gating.

Guarantees 100% fidelity:
- Only extracts existing points from world_pointcloud.ply.
- Zero synthetic, interpolated, or resampled points are created.
- Retains all original point coordinates (X, Y, Z), colors, and attributes.
- Outputs individual object point clouds (*_pointcloud.ply) and an extraction manifest.
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

from sklearn.cluster import DBSCAN


def _build_2d_mask(view: Dict[str, Any], H: int, W: int) -> np.ndarray:
    """Build binary 2D mask from polygon or bounding box."""
    mask_2d = np.zeros((H, W), dtype=np.uint8)
    if "mask" in view and isinstance(view["mask"], (list, np.ndarray)) and len(view["mask"]) >= 3:
        raw_mask = np.array(view["mask"], dtype=np.int32)
        if raw_mask.ndim == 1:
            poly_pts = raw_mask.reshape(-1, 1, 2)
        elif raw_mask.ndim == 2 and raw_mask.shape[1] == 2:
            poly_pts = raw_mask.reshape(-1, 1, 2)
        else:
            poly_pts = raw_mask.astype(np.int32)
        cv2.fillPoly(mask_2d, [poly_pts], 255)
    else:
        bbox = view.get("bbox", [0, 0, W, H])
        xmin, ymin, xmax, ymax = map(int, bbox)
        mask_2d[max(0, ymin):min(H, ymax), max(0, xmin):min(W, xmax)] = 255
    return mask_2d


def extract_object_points_from_world_pcd_view(
    world_pts: np.ndarray,
    world_cols: Optional[np.ndarray],
    mask_2d: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    depth_map: Optional[np.ndarray] = None,
    depth_tolerance: float = getattr(config, "OBJECT_DEPTH_CONSISTENCY_TOLERANCE", 0.10),
    foreground_margin: float = getattr(config, "OBJECT_DEPTH_FOREGROUND_MARGIN", 0.85),
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Project 3D world points onto a 2D camera view and select points falling inside the 2D mask
    with depth-consistency verification.

    Parameters
    ----------
    world_pts : (N, 3) 3D coordinates in world space.
    world_cols : (N, 3) RGB colors (0-255) or None.
    mask_2d : (H, W) uint8 binary mask where >0 indicates object pixels.
    K : (3, 3) intrinsic camera matrix.
    c2w : (4, 4) camera-to-world pose matrix.
    depth_map : Optional (H, W) float depth map in meters for depth consistency gating.
    depth_tolerance : Max allowable |Z_cam - Z_depth| delta in meters when depth map is present.
    foreground_margin : Fallback depth delta when depth map is absent.

    Returns
    -------
    (selected_pts [M, 3], selected_cols [M, 3] or None, matched_indices [M])
    """
    if len(world_pts) == 0:
        return np.zeros((0, 3)), None, np.zeros(0, dtype=np.int64)

    H, W = mask_2d.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Transform world points to camera space
    try:
        w2c = np.linalg.inv(c2w)
    except np.linalg.LinAlgError:
        w2c = np.linalg.pinv(c2w)

    pts_h = np.hstack([world_pts, np.ones((len(world_pts), 1), dtype=np.float64)])
    pts_cam = (w2c @ pts_h.T).T[:, :3]

    Z_c = pts_cam[:, 2]
    # Filter points strictly in front of the camera
    front_mask = Z_c > 0.1
    if not np.any(front_mask):
        return np.zeros((0, 3)), None, np.zeros(0, dtype=np.int64)

    # Perspective projection
    u = np.round((pts_cam[:, 0] * fx / np.maximum(Z_c, 1e-6)) + cx).astype(np.int64)
    v = np.round((pts_cam[:, 1] * fy / np.maximum(Z_c, 1e-6)) + cy).astype(np.int64)

    # Check image bounds
    in_bounds = front_mask & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_bounds):
        return np.zeros((0, 3)), None, np.zeros(0, dtype=np.int64)

    valid_idx = np.where(in_bounds)[0]
    u_valid = u[valid_idx]
    v_valid = v[valid_idx]

    # Check 2D mask inclusion
    in_mask_subset = mask_2d[v_valid, u_valid] > 0
    matched_idx = valid_idx[in_mask_subset]

    if len(matched_idx) == 0:
        return np.zeros((0, 3)), None, np.zeros(0, dtype=np.int64)

    # Depth Consistency Gating:
    # 1. If depth map is available for this frame, enforce |Z_cam - depth_map[v, u]| <= depth_tolerance
    if depth_map is not None and depth_map.shape[:2] == (H, W) and depth_tolerance > 0:
        u_matched = u[matched_idx]
        v_matched = v[matched_idx]
        z_matched = Z_c[matched_idx]
        d_obs = depth_map[v_matched, u_matched]
        valid_depth = np.isfinite(d_obs) & (d_obs > 0)
        
        depth_diff = np.abs(z_matched - d_obs)
        consistent_mask = (~valid_depth) | (depth_diff <= depth_tolerance)
        if np.any(consistent_mask):
            matched_idx = matched_idx[consistent_mask]

    # 2. Fallback Adaptive Foreground Gating: Prune far background bleed while preserving full object depth
    elif foreground_margin > 0 and len(matched_idx) > 10:
        matched_z = Z_c[matched_idx]
        near_z = float(np.percentile(matched_z, 10.0))
        effective_margin = max(foreground_margin, 0.85)
        fg_mask = matched_z <= (near_z + effective_margin)
        matched_idx = matched_idx[fg_mask]

    selected_pts = world_pts[matched_idx]
    selected_cols = world_cols[matched_idx] if world_cols is not None else None
    return selected_pts, selected_cols, matched_idx


def _get_dbscan_mask(
    pts: np.ndarray,
    eps: float = getattr(config, "OBJECT_DBSCAN_EPS", 0.06),
    min_samples: int = getattr(config, "OBJECT_DBSCAN_MIN_SAMPLES", 4),
    min_cluster_size: int = getattr(config, "OBJECT_DBSCAN_MIN_CLUSTER_SIZE", 10),
    max_merge_dist: Optional[float] = None,
) -> np.ndarray:
    """Internal helper: return boolean mask of DBSCAN inliers."""
    n_pts = len(pts)
    if n_pts < min_samples:
        return np.ones(n_pts, dtype=bool)

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(pts)
    labels = db.labels_

    valid_mask_labels = labels >= 0
    if not np.any(valid_mask_labels):
        return np.ones(n_pts, dtype=bool)

    unique_labels, counts = np.unique(labels[valid_mask_labels], return_counts=True)
    size_filter = counts >= min_cluster_size
    valid_clusters = unique_labels[size_filter]
    valid_counts = counts[size_filter]

    if len(valid_clusters) == 0:
        if len(unique_labels) > 0:
            dominant_label = unique_labels[np.argmax(counts)]
            selected_clusters = [dominant_label]
        else:
            return np.ones(n_pts, dtype=bool)
    else:
        sort_order = np.argsort(-valid_counts)
        dominant_label = valid_clusters[sort_order[0]]
        dominant_pts = pts[labels == dominant_label]
        dominant_centroid = np.mean(dominant_pts, axis=0)
        dominant_span = float(np.linalg.norm(np.max(dominant_pts, axis=0) - np.min(dominant_pts, axis=0)))
        if max_merge_dist is None:
            max_merge_dist = max(0.45, dominant_span * 0.50)

        selected_clusters = [dominant_label]
        from scipy.spatial import cKDTree
        dom_tree = cKDTree(dominant_pts)

        for c_idx in sort_order[1:]:
            c_label = valid_clusters[c_idx]
            c_pts = pts[labels == c_label]
            c_centroid = np.mean(c_pts, axis=0)
            centroid_dist = np.linalg.norm(c_centroid - dominant_centroid)
            if centroid_dist < max_merge_dist:
                selected_clusters.append(c_label)
            else:
                min_dists, _ = dom_tree.query(c_pts, k=1)
                if np.min(min_dists) <= (eps * 2.5):
                    selected_clusters.append(c_label)

    return np.isin(labels, selected_clusters)


def filter_object_pointcloud_dbscan(
    pts: np.ndarray,
    colors: Optional[np.ndarray] = None,
    eps: float = getattr(config, "OBJECT_DBSCAN_EPS", 0.06),
    min_samples: int = getattr(config, "OBJECT_DBSCAN_MIN_SAMPLES", 4),
    min_cluster_size: int = getattr(config, "OBJECT_DBSCAN_MIN_CLUSTER_SIZE", 10),
    max_merge_dist: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Remove noise points and isolate the dominant object cluster using core-first DBSCAN.
    Recovers valid sub-clusters (caster wheels, armrests, legs) located within the object envelope.

    Returns
    -------
    (pts_clean, cols_clean)
    """
    mask = _get_dbscan_mask(pts, eps=eps, min_samples=min_samples, min_cluster_size=min_cluster_size, max_merge_dist=max_merge_dist)
    pts_clean = pts[mask]
    cols_clean = colors[mask] if colors is not None else None
    return pts_clean, cols_clean


def _get_plane_inlier_mask(
    pts: np.ndarray,
    label: str,
    plane_data: Optional[Dict[str, Any]],
    margin: float = getattr(config, "PLANE_SUBTRACTION_MARGIN", 0.015),
) -> np.ndarray:
    """Internal helper: return boolean mask of non-plane points."""
    n_pts = len(pts)
    if plane_data is None or n_pts == 0:
        return np.ones(n_pts, dtype=bool)

    label_lower = label.lower()
    if label_lower in {"table", "desk", "floor", "rug", "carpet"}:
        return np.ones(n_pts, dtype=bool)

    keep_mask = np.ones(n_pts, dtype=bool)
    obj_height = float(pts[:, 1].max() - pts[:, 1].min())

    floor = plane_data.get("floor")
    if floor and obj_height > 0.15:
        floor_y = float(floor.get("mean_y", 0.0))
        floor_pts_mask = pts[:, 1] <= (floor_y + margin)
        if np.sum(~floor_pts_mask) >= 15:
            keep_mask &= ~floor_pts_mask

    tables = plane_data.get("tables", [])
    for tp in tables:
        t_y = float(tp.get("mean_y", 0.0))
        min_b = tp.get("min_bound", [-1e5, t_y, -1e5])
        max_b = tp.get("max_bound", [1e5, t_y, 1e5])

        in_table_x = (min_b[0] - 0.05) <= pts[:, 0]
        in_table_x &= pts[:, 0] <= (max_b[0] + 0.05)
        in_table_z = (min_b[2] - 0.05) <= pts[:, 2]
        in_table_z &= pts[:, 2] <= (max_b[2] + 0.05)
        in_table_y = np.abs(pts[:, 1] - t_y) <= margin

        table_slab_mask = in_table_x & in_table_z & in_table_y
        if np.sum(table_slab_mask) > 0 and np.sum(~table_slab_mask & keep_mask) >= 15:
            keep_mask &= ~table_slab_mask

    return keep_mask


def _filter_plane_inliers(
    pts: np.ndarray,
    cols: Optional[np.ndarray],
    label: str,
    plane_data: Optional[Dict[str, Any]],
    margin: float = getattr(config, "PLANE_SUBTRACTION_MARGIN", 0.015),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Remove floor and tabletop plane points that may have been accidentally included.
    Returns (pts_filtered, cols_filtered).
    """
    keep_mask = _get_plane_inlier_mask(pts, label, plane_data, margin=margin)
    pts_filtered = pts[keep_mask]
    cols_filtered = cols[keep_mask] if cols is not None else None
    return pts_filtered, cols_filtered


def load_world_pointcloud(world_pcd_path: Path | str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load world point cloud coordinates and RGB colors.
    Preserves exact coordinates and colors without modification.
    """
    world_pcd_path = Path(world_pcd_path)
    if not world_pcd_path.exists():
        raise FileNotFoundError(f"[ObjectExtractor] World point cloud not found: {world_pcd_path}")

    world_pts = None
    world_cols = None

    if HAS_TRIMESH:
        try:
            cloud = trimesh.load(str(world_pcd_path))
            if isinstance(cloud, trimesh.Scene):
                all_v = [np.asarray(g.vertices, dtype=np.float64) for g in cloud.geometry.values() if hasattr(g, "vertices") and len(g.vertices) > 0]
                if all_v:
                    world_pts = np.vstack(all_v)
                all_c = [np.asarray(g.colors)[:, :3].astype(np.uint8) for g in cloud.geometry.values() if hasattr(g, "colors") and g.colors is not None and len(g.colors) == len(g.vertices)]
                if len(all_c) == len(all_v) and len(all_c) > 0:
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
            pcd = o3d.io.read_point_cloud(str(world_pcd_path))
            if len(pcd.points) > 0:
                world_pts = np.asarray(pcd.points, dtype=np.float64)
                world_cols = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
        except Exception:
            pass

    if world_pts is None or len(world_pts) == 0:
        raise ValueError(f"[ObjectExtractor] Failed to load 3D points from {world_pcd_path}")

    return world_pts, world_cols


def extract_object_pointclouds(
    detections_path: Optional[Path | str] = None,
    raw_depths_path: Optional[Path | str] = None,
    ar_metadata_path: Optional[Path | str] = None,
    world_pcd_path: Optional[Path | str] = None,
    plane_data_path: Optional[Path | str] = None,
    out_dir: Optional[Path | str] = None,
    filter_planes: bool = False,
    enable_dbscan: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Extract exact 3D point cloud clusters for all detected objects from world_pointcloud.ply.

    Guarantees:
    - 100% of extracted points come from world_pointcloud.ply (zero synthetic points).
    - Preserves all original point coordinates (X, Y, Z), colors, and spatial positioning.

    Returns
    -------
    Dict mapping obj_id to metadata dictionary (label, pcd_path, point_count, bounds_min, bounds_max, etc.)
    """
    if detections_path is None:
        detections_path = config.PROCESSED_DATA_DIR / "detections.json"
    detections_path = Path(detections_path)

    if not detections_path.exists():
        raise FileNotFoundError(
            f"[ObjectExtractor] Detections file not found: {detections_path}\n"
            "                  Please run detection stage first to produce detections.json."
        )

    # Resolve world_pcd_path
    if world_pcd_path is None:
        cand_pcd = detections_path.parent / "world_pointcloud.ply"
        world_pcd_path = cand_pcd if cand_pcd.exists() else config.PROCESSED_DATA_DIR / "world_pointcloud.ply"
    world_pcd_path = Path(world_pcd_path)

    if not world_pcd_path.exists():
        raise FileNotFoundError(
            f"[ObjectExtractor] World point cloud not found: {world_pcd_path}\n"
            "                  Please run pointcloud_builder.py first."
        )

    # Resolve raw_depths_path
    if raw_depths_path is not None:
        raw_depths_path = Path(raw_depths_path)
        if not raw_depths_path.exists():
            raise FileNotFoundError(f"[ObjectExtractor] Raw depths file not found: {raw_depths_path}")
    else:
        cand_npz1 = detections_path.parent / "raw_depths.npz"
        cand_npz2 = world_pcd_path.parent / "raw_depths.npz"
        if cand_npz1.exists():
            raw_depths_path = cand_npz1
        elif cand_npz2.exists():
            raw_depths_path = cand_npz2
        elif detections_path.parent == config.PROCESSED_DATA_DIR and (config.PROCESSED_DATA_DIR / "raw_depths.npz").exists():
            raw_depths_path = config.PROCESSED_DATA_DIR / "raw_depths.npz"
        else:
            raw_depths_path = None

    # Resolve ar_metadata_path
    if ar_metadata_path is not None:
        ar_metadata_path = Path(ar_metadata_path)
    else:
        cand_meta1 = detections_path.parent / "ar_metadata.json"
        cand_meta2 = world_pcd_path.parent / "ar_metadata.json"
        if cand_meta1.exists():
            ar_metadata_path = cand_meta1
        elif cand_meta2.exists():
            ar_metadata_path = cand_meta2
        elif detections_path.parent == config.PROCESSED_DATA_DIR and (config.PROCESSED_DATA_DIR / "ar_metadata.json").exists():
            ar_metadata_path = config.PROCESSED_DATA_DIR / "ar_metadata.json"
        else:
            ar_metadata_path = None


    # Resolve plane_data_path
    if plane_data_path is None:
        cand_planes1 = detections_path.parent / "detected_planes.json"
        cand_planes2 = world_pcd_path.parent / "detected_planes.json"
        default_planes = config.PROCESSED_DATA_DIR / "detected_planes.json"
        if cand_planes1.exists():
            plane_data_path = cand_planes1
        elif cand_planes2.exists():
            plane_data_path = cand_planes2
        else:
            plane_data_path = default_planes
    plane_data_path = Path(plane_data_path)

    plane_data = None
    if filter_planes and plane_data_path.exists():
        try:
            with open(plane_data_path, "r", encoding="utf-8") as pf:
                plane_data = json.load(pf)
        except Exception:
            plane_data = None

    if out_dir is None:
        out_dir = detections_path.parent / "objects" if detections_path.parent != config.PROCESSED_DATA_DIR else config.PROCESSED_DATA_DIR / "objects"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if enable_dbscan is None:
        enable_dbscan = getattr(config, "OBJECT_ENABLE_DBSCAN", True)

    with open(detections_path, "r", encoding="utf-8") as f:
        detections = json.load(f)

    print(f"[ObjectExtractor] Loaded {len(detections)} object detections.")
    print(f"[ObjectExtractor] Loading world point cloud from '{world_pcd_path.name}'...")
    world_pts, world_cols = load_world_pointcloud(world_pcd_path)
    print(f"[ObjectExtractor] Loaded {len(world_pts):,} source points. Extracting objects...")

    npz = np.load(str(raw_depths_path)) if raw_depths_path is not None else {}
    frames_meta = {}
    if ar_metadata_path is not None and ar_metadata_path.exists():
        try:
            with open(ar_metadata_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            frames_meta = {f["index"]: f for f in meta.get("frames", [])}
        except Exception:
            frames_meta = {}


    extracted_objects: Dict[str, Any] = {}

    for obj_id, obj_info in detections.items():
        label = obj_info.get("label", "object")
        views = obj_info.get("associated_views", [])
        if not views:
            continue

        matched_indices_list: List[np.ndarray] = []

        for view in views:
            f_idx = view.get("frame_index", 0)
            d_key = f"depth_{f_idx}"

            # Determine frame dimensions & depth map
            if d_key in npz:
                depth_map = npz[d_key]
                H, W = depth_map.shape[:2]
            elif f_idx in frames_meta:
                f_meta = frames_meta[f_idx]
                H = int(f_meta.get("h", 720))
                W = int(f_meta.get("w", 1280))
                depth_map = None
            else:
                view_w = int(view.get("image_width", view.get("w", 0)))
                view_h = int(view.get("image_height", view.get("h", 0)))
                bbox = view.get("bbox", [0, 0, 1280, 720])
                xmin, ymin, xmax, ymax = (int(b) for b in bbox[:4]) if len(bbox) >= 4 else (0, 0, 1280, 720)
                W = max(view_w, xmax, 1280)
                H = max(view_h, ymax, 720)
                depth_map = None

            # Determine Camera Intrinsics K
            if f"ixt_{f_idx}" in npz:
                K = npz[f"ixt_{f_idx}"].astype(np.float64)
            elif f_idx in frames_meta and "fl_x" in frames_meta[f_idx]:
                f_meta = frames_meta[f_idx]
                K = np.array([
                    [f_meta.get("fl_x", 1.2 * max(W, H)), 0.0, f_meta.get("cx", W / 2.0)],
                    [0.0, f_meta.get("fl_y", 1.2 * max(W, H)), f_meta.get("cy", H / 2.0)],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64)
            else:
                K = np.array([[1.2 * max(W, H), 0, W / 2], [0, 1.2 * max(W, H), H / 2], [0, 0, 1]], dtype=np.float64)

            # Determine Camera-to-World Pose c2w
            if f"ext_{f_idx}" in npz:
                w2c = npz[f"ext_{f_idx}"].astype(np.float64)
                if w2c.shape == (3, 4):
                    H_mat = np.eye(4, dtype=np.float64)
                    H_mat[:3, :4] = w2c
                    w2c = H_mat
                c2w = np.linalg.pinv(w2c)
                c2w = np.diag([1.0, -1.0, -1.0, 1.0]) @ c2w
            elif f_idx in frames_meta:
                c2w = np.array(frames_meta[f_idx]["pose_matrix"], dtype=np.float64)
            else:
                c2w = np.eye(4, dtype=np.float64)

            mask_2d = _build_2d_mask(view, H, W)

            # Exact point selection from world point cloud via 2D projection
            _, _, m_idx = extract_object_points_from_world_pcd_view(
                world_pts, world_cols, mask_2d, K, c2w, depth_map=depth_map
            )
            if len(m_idx) > 0:
                matched_indices_list.append(m_idx)

        if not matched_indices_list:
            print(f"[ObjectExtractor] WARNING: Object '{obj_id}' ({label}) has 0 matching 3D points; skipping.")
            continue

        # Multi-View Consensus Aggregation
        all_idx_concat = np.concatenate(matched_indices_list)
        unique_idx, counts = np.unique(all_idx_concat, return_counts=True)
        num_views = len(matched_indices_list)

        consensus_ratio = getattr(config, "OBJECT_VIEW_CONSENSUS_RATIO", 0.30)
        if num_views >= 3:
            min_votes = max(2, int(np.ceil(num_views * consensus_ratio)))
            consensus_mask = counts >= min_votes
            if np.sum(consensus_mask) >= 10:
                merged_idx = unique_idx[consensus_mask]
            else:
                merged_idx = unique_idx
        else:
            merged_idx = unique_idx

        # Directly slice exact points from source point cloud
        cand_pts = world_pts[merged_idx]

        # Optional Plane Inlier Filter
        if filter_planes and plane_data is not None:
            plane_keep = _get_plane_inlier_mask(cand_pts, label, plane_data)
            merged_idx = merged_idx[plane_keep]
            cand_pts = world_pts[merged_idx]

        # DBSCAN Outlier Removal (preserves exact coordinates & attributes)
        if enable_dbscan and len(cand_pts) >= 4:
            dbscan_keep = _get_dbscan_mask(cand_pts)
            merged_idx = merged_idx[dbscan_keep]
            cand_pts = world_pts[merged_idx]

        if len(cand_pts) < 4:
            print(f"[ObjectExtractor] WARNING: Object '{obj_id}' ({label}) has insufficient 3D points ({len(cand_pts)}) after filtering; skipping.")
            continue

        # Final exact point cloud extraction
        final_pts = world_pts[merged_idx]
        final_cols = world_cols[merged_idx] if world_cols is not None else None

        # Export individual object point cloud (.ply)
        obj_pcd_path = out_dir / f"{obj_id}_{label}_pointcloud.ply"
        if HAS_TRIMESH:
            if final_cols is not None:
                pcd_tri = trimesh.PointCloud(vertices=final_pts, colors=final_cols)
            else:
                pcd_tri = trimesh.PointCloud(vertices=final_pts)
            pcd_tri.export(str(obj_pcd_path))
        elif HAS_OPEN3D:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(final_pts)
            if final_cols is not None:
                pcd_o3d.colors = o3d.utility.Vector3dVector(final_cols / 255.0)
            o3d.io.write_point_cloud(str(obj_pcd_path), pcd_o3d)

        print(f"[ObjectExtractor] Extracted object '{obj_id}' ({label}): {len(final_pts):,} points -> {obj_pcd_path.name}")

        extracted_objects[obj_id] = {
            "label": label,
            "pcd_path": str(obj_pcd_path),
            "mesh_path": str(out_dir / f"{obj_id}_{label}.ply"),
            "point_count": len(final_pts),
            "bounds_min": final_pts.min(axis=0).tolist(),
            "bounds_max": final_pts.max(axis=0).tolist(),
            "centroid": final_pts.mean(axis=0).tolist(),
        }

    # Save extraction manifest
    extracted_manifest_path = out_dir / "extracted_objects_manifest.json"
    with open(extracted_manifest_path, "w", encoding="utf-8") as f:
        json.dump(extracted_objects, f, indent=2)

    # Also save/update objects_manifest.json for downstream compatibility
    summary_path = out_dir / "objects_manifest.json"
    existing_manifest = {}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
        except Exception:
            existing_manifest = {}
    existing_manifest.update(extracted_objects)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(existing_manifest, f, indent=2)

    print(f"[ObjectExtractor] Extraction complete: {len(extracted_objects)} objects saved -> {extracted_manifest_path}")

    try:
        if hasattr(npz, "close"):
            npz.close()
    except Exception:
        pass

    return extracted_objects


class ObjectExtractor:
    """Class wrapper for 3D Object Point Cloud Extraction."""

    def __init__(
        self,
        detections_path: Optional[Path | str] = None,
        raw_depths_path: Optional[Path | str] = None,
        ar_metadata_path: Optional[Path | str] = None,
        world_pcd_path: Optional[Path | str] = None,
        plane_data_path: Optional[Path | str] = None,
        out_dir: Optional[Path | str] = None,
    ):
        self.detections_path = Path(detections_path) if detections_path else config.PROCESSED_DATA_DIR / "detections.json"
        self.raw_depths_path = Path(raw_depths_path) if raw_depths_path is not None else None
        self.ar_metadata_path = Path(ar_metadata_path) if ar_metadata_path else None
        self.world_pcd_path = Path(world_pcd_path) if world_pcd_path else None
        self.plane_data_path = Path(plane_data_path) if plane_data_path else None
        self.out_dir = Path(out_dir) if out_dir else None

    def run(self) -> Dict[str, Any]:
        return extract_object_pointclouds(
            detections_path=self.detections_path,
            raw_depths_path=self.raw_depths_path,
            ar_metadata_path=self.ar_metadata_path,
            world_pcd_path=self.world_pcd_path,
            plane_data_path=self.plane_data_path,
            out_dir=self.out_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2A: 3D Object Point Cloud Extraction & Segmentation")
    parser.add_argument("--detections", type=str, default=str(config.PROCESSED_DATA_DIR / "detections.json"),
                        help="Path to detections.json file")
    parser.add_argument("--depths", type=str, default=str(config.PROCESSED_DATA_DIR / "raw_depths.npz"),
                        help="Path to raw_depths.npz file")
    parser.add_argument("--world-pcd", type=str, default=str(config.PROCESSED_DATA_DIR / "world_pointcloud.ply"),
                        help="Path to world_pointcloud.ply file")
    parser.add_argument("--planes-json", type=str, default=str(config.PROCESSED_DATA_DIR / "detected_planes.json"),
                        help="Path to detected_planes.json file")
    parser.add_argument("--out-dir", type=str, default=str(config.PROCESSED_DATA_DIR / "objects"),
                        help="Output directory for extracted object point clouds")
    args = parser.parse_args()

    extract_object_pointclouds(
        detections_path=args.detections,
        raw_depths_path=args.depths,
        world_pcd_path=args.world_pcd,
        plane_data_path=args.planes_json,
        out_dir=args.out_dir,
    )

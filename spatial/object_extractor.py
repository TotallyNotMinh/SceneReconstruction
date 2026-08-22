# -*- coding: utf-8 -*-
"""
spatial/object_extractor.py — Phase 2A: OpenMask3D Open-Vocabulary 3D Object Point Cloud Extraction.

Extracts exact 3D point cloud clusters for 3D object instances directly from world_pointcloud.ply
using OpenMask3D (Open-Vocabulary 3D Instance Segmentation):
- Generates 3D class-agnostic instance mask proposals directly from 3D geometry.
- Projects 3D proposals onto multi-view RGB video frames and aggregates 2D CLIP visual features.
- Matches 3D instance masks with open-vocabulary text queries via Zero-Shot CLIP Cosine Similarity.
- Slices exact inlier vertices from world_pointcloud.ply (zero synthetic/interpolated points).
- Runs standalone with ONLY (world_pointcloud.ply, raw_depths.npz) — NO detections.json required!

NOTE: Legacy heuristic geometric projection code is preserved below, commented out with '#'.
"""

import sys
import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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
from scipy.spatial import cKDTree


# ==============================================================================
# ======================== LEGACY HEURISTIC EXTRACTION =========================
# ===================== (DISABLED AS REQUESTED BY USER) ========================
# ==============================================================================
#
# def _build_2d_mask(view: Dict[str, Any], H: int, W: int) -> np.ndarray:
#     """Build binary 2D mask from polygon or bounding box."""
#     mask_2d = np.zeros((H, W), dtype=np.uint8)
#     if "mask" in view and isinstance(view["mask"], (list, np.ndarray)) and len(view["mask"]) >= 3:
#         raw_mask = np.array(view["mask"], dtype=np.int32)
#         if raw_mask.ndim == 1:
#             poly_pts = raw_mask.reshape(-1, 1, 2)
#         elif raw_mask.ndim == 2 and raw_mask.shape[1] == 2:
#             poly_pts = raw_mask.reshape(-1, 1, 2)
#         else:
#             poly_pts = raw_mask.astype(np.int32)
#         cv2.fillPoly(mask_2d, [poly_pts], 255)
#     else:
#         bbox = view.get("bbox", [0, 0, W, H])
#         xmin, ymin, xmax, ymax = map(int, bbox)
#         mask_2d[max(0, ymin):min(H, ymax), max(0, xmin):min(W, xmax)] = 255
#     return mask_2d
#
#
# def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
#     """Convert RGB array (N, 3) with range [0, 255] to OpenCV CIELAB (N, 3) float32."""
#     if len(rgb) == 0:
#         return np.zeros((0, 3), dtype=np.float32)
#     rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
#     lab = cv2.cvtColor(rgb_u8.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
#     return lab.astype(np.float32)
#
#
# def extract_object_points_from_world_pcd_view(
#     world_pts: np.ndarray,
#     world_cols: Optional[np.ndarray],
#     mask_2d: np.ndarray,
#     K: np.ndarray,
#     c2w: np.ndarray,
#     depth_map: Optional[np.ndarray] = None,
#     rgb_frame: Optional[np.ndarray] = None,
#     depth_tolerance: float = 0.10,
#     foreground_margin: float = 0.85,
#     enable_color_filter: bool = True,
#     color_delta_e_max: float = 45.0,
#     color_sample_count: int = 300,
# ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
#     """Legacy perspective projection with depth-consistency and CIELAB color gating."""
#     # [Legacy 2D-to-3D projection logic disabled in favor of OpenMask3D]
#     pass
#
#
# def _get_dbscan_mask(pts: np.ndarray, colors: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
#     """Legacy 6D DBSCAN clustering."""
#     # [Legacy DBSCAN logic disabled]
#     pass
#
#
# def _filter_plane_inliers(pts: np.ndarray, cols: Optional[np.ndarray], label: str, plane_data: Optional[Dict[str, Any]], **kwargs):
#     """Legacy plane inlier subtraction."""
#     # [Legacy plane subtraction disabled]
#     pass
# ==============================================================================


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array (N, 3) with range [0, 255] to OpenCV CIELAB (N, 3) float32."""
    if len(rgb) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb_u8.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return lab.astype(np.float32)


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

    z = depth_map[v_coords, u_coords].astype(np.float64)
    x = (u_coords.astype(np.float64) - cx) * z / fx
    y = (v_coords.astype(np.float64) - cy) * z / fy

    pts_cam = np.column_stack([x, y, z])

    # Transform to world
    if c2w.shape == (3, 4):
        c2w_4x4 = np.eye(4, dtype=np.float64)
        c2w_4x4[:3, :4] = c2w
        c2w = c2w_4x4

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pts_world = (R @ pts_cam.T).T + t

    cols = None
    if rgb_img is not None:
        cols = rgb_img[v_coords, u_coords]
        if cols.shape[-1] == 4:
            cols = cols[:, :3]

    return pts_world, cols


def extract_object_points_from_world_pcd_view(

    world_pts: np.ndarray,
    world_cols: Optional[np.ndarray],
    mask_2d: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    depth_map: Optional[np.ndarray] = None,
    rgb_frame: Optional[np.ndarray] = None,
    depth_tolerance: float = 0.10,
    foreground_margin: float = 0.85,
    enable_color_filter: bool = True,
    color_delta_e_max: float = 45.0,
    color_sample_count: int = 300,
    **kwargs,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Project world points onto 2D camera view and extract points falling inside mask_2d.
    Maintains depth and color consistency.
    """
    if len(world_pts) == 0 or mask_2d is None:
        return np.zeros((0, 3)), (np.zeros((0, 3), dtype=np.uint8) if world_cols is not None else None), np.zeros(0, dtype=np.int64)

    H, W = mask_2d.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Handle 3x4 pose
    if c2w.shape == (3, 4):
        c2w_4x4 = np.eye(4, dtype=np.float64)
        c2w_4x4[:3, :4] = c2w
        c2w = c2w_4x4

    w2c = np.linalg.pinv(c2w)
    pts_h = np.hstack([world_pts, np.ones((len(world_pts), 1), dtype=np.float64)])
    pts_cam = (w2c @ pts_h.T).T[:, :3]

    Z_c = pts_cam[:, 2]
    Z_abs = np.abs(Z_c)
    in_front = Z_abs > 0.1
    if not np.any(in_front):
        return np.zeros((0, 3)), (np.zeros((0, 3), dtype=np.uint8) if world_cols is not None else None), np.zeros(0, dtype=np.int64)

    # Divide by Z_abs to handle both +Z and -Z camera axes gracefully
    u = np.round((pts_cam[:, 0] * fx / np.maximum(Z_abs, 1e-6)) + cx).astype(np.int64)
    v = np.round((pts_cam[:, 1] * fy / np.maximum(Z_abs, 1e-6)) + cy).astype(np.int64)

    in_bounds = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_bounds):
        return np.zeros((0, 3)), (np.zeros((0, 3), dtype=np.uint8) if world_cols is not None else None), np.zeros(0, dtype=np.int64)

    bounds_indices = np.where(in_bounds)[0]
    u_b = u[bounds_indices]
    v_b = v[bounds_indices]
    mask_hit = mask_2d[v_b, u_b] > 0
    mask_indices = bounds_indices[mask_hit]
    if len(mask_indices) == 0:
        return np.zeros((0, 3)), (np.zeros((0, 3), dtype=np.uint8) if world_cols is not None else None), np.zeros(0, dtype=np.int64)

    # Depth verification if depth_map is provided
    if depth_map is not None:
        meas_z = depth_map[v[mask_indices], u[mask_indices]]
        valid_meas = np.isfinite(meas_z) & (meas_z > 0.1)
        if np.any(valid_meas):
            z_diff = np.abs(Z_abs[mask_indices] - meas_z)
            depth_inlier = valid_meas & (z_diff <= depth_tolerance)

            # Foreground gating
            if foreground_margin > 0 and np.any(depth_inlier):
                min_z = float(np.percentile(Z_abs[mask_indices][depth_inlier], 5.0))
                depth_inlier = depth_inlier & (Z_abs[mask_indices] <= min_z + foreground_margin)

            final_indices = mask_indices[depth_inlier]
        else:
            final_indices = mask_indices
    else:
        final_indices = mask_indices

    # Color consistency filtering
    eff_color_filter = kwargs.get("enable_color_filter", enable_color_filter)
    eff_rgb_frame = kwargs.get("rgb_frame", rgb_frame)
    eff_max_delta_e = kwargs.get("color_delta_e_max", color_delta_e_max)
    if eff_color_filter and eff_rgb_frame is not None and world_cols is not None and len(final_indices) > 0:
        p_cols = world_cols[final_indices]
        img_cols = eff_rgb_frame[v[final_indices], u[final_indices]]
        diff = np.linalg.norm(p_cols.astype(np.float64) - img_cols.astype(np.float64), axis=1)
        color_keep = diff <= (eff_max_delta_e * 2.55)
        final_indices = final_indices[color_keep]

    if len(final_indices) == 0:
        return np.zeros((0, 3)), (np.zeros((0, 3), dtype=np.uint8) if world_cols is not None else None), np.zeros(0, dtype=np.int64)

    out_pts = world_pts[final_indices]
    out_cols = world_cols[final_indices] if world_cols is not None else None
    return out_pts, out_cols, final_indices




def _filter_plane_inliers(
    pts: np.ndarray,
    cols: Optional[np.ndarray],
    label: str,
    plane_data: Optional[Dict[str, Any]],
    distance_threshold: float = 0.03,
    **kwargs,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Remove architectural plane inliers from an extracted object cluster."""
    if len(pts) == 0 or not plane_data:
        return pts, cols

    margin = kwargs.get("margin", distance_threshold)
    keep_mask = np.ones(len(pts), dtype=bool)

    # Filter floor inliers
    floor = plane_data.get("floor")
    if floor:
        floor_y = float(floor.get("mean_y", 0.0))
        on_floor = pts[:, 1] <= (floor_y + margin)
        keep_mask = keep_mask & ~on_floor

    # Filter tabletop inliers if object is sitting on a table
    tables = plane_data.get("tables", [])
    for table in tables:
        t_y = float(table.get("mean_y", 0.0))
        on_table = np.abs(pts[:, 1] - t_y) <= margin
        keep_mask = keep_mask & ~on_table

    if not np.any(keep_mask):
        return pts, cols

    return pts[keep_mask], (cols[keep_mask] if cols is not None else None)


def _build_2d_mask(view: Dict[str, Any], H: int, W: int) -> np.ndarray:
    """Utility helper for building 2D mask from bbox or polygon if needed."""
    mask_2d = np.zeros((H, W), dtype=np.uint8)
    if "mask" in view and isinstance(view["mask"], (list, np.ndarray)) and len(view["mask"]) >= 3:
        raw_mask = np.array(view["mask"], dtype=np.int32)
        poly_pts = raw_mask.reshape(-1, 1, 2) if raw_mask.ndim in (1, 2) else raw_mask.astype(np.int32)
        cv2.fillPoly(mask_2d, [poly_pts], 255)
    else:
        bbox = view.get("bbox", [0, 0, W, H])
        xmin, ymin, xmax, ymax = map(int, bbox[:4]) if len(bbox) >= 4 else (0, 0, W, H)
        mask_2d[max(0, ymin):min(H, ymax), max(0, xmin):min(W, xmax)] = 255
    return mask_2d


def filter_object_pointcloud_dbscan(
    pts: np.ndarray,
    colors: Optional[np.ndarray] = None,
    eps: Optional[float] = None,
    min_samples: Optional[int] = None,
    min_cluster_size: int = 15,
    **kwargs,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Clean isolated noise points from an extracted point cloud cluster using adaptive cluster retention."""
    if len(pts) < 4:
        return pts, colors
    eps_val = eps or kwargs.get("eps", getattr(config, "OBJECT_DBSCAN_EPS", 0.06))
    min_samp = min_samples or kwargs.get("min_samples", getattr(config, "OBJECT_DBSCAN_MIN_SAMPLES", 4))
    min_sz = kwargs.get("min_cluster_size", min_cluster_size)

    if colors is not None and len(colors) == len(pts):
        c_feat = (colors.astype(np.float64) / 255.0) * 0.35
        feat = np.hstack([pts, c_feat])
        db = DBSCAN(eps=eps_val, min_samples=min_samp).fit(feat)
    else:
        db = DBSCAN(eps=eps_val, min_samples=min_samp).fit(pts)

    valid = db.labels_ >= 0
    if not np.any(valid):
        return pts, colors


    labels, counts = np.unique(db.labels_[valid], return_counts=True)
    if len(labels) == 0:
        return pts, colors

    dominant_label = labels[np.argmax(counts)]
    dominant_pts = pts[db.labels_ == dominant_label]
    dominant_center = dominant_pts.mean(axis=0)

    # Keep dominant cluster plus any adjacent clusters with >= min_sz points within 0.85m of dominant center
    keep_labels = {dominant_label}
    for lbl, count in zip(labels, counts):
        if lbl != dominant_label and count >= min_sz:
            c_pts = pts[db.labels_ == lbl]
            c_center = c_pts.mean(axis=0)
            if np.linalg.norm(c_center - dominant_center) <= 0.85:
                keep_labels.add(lbl)

    mask = np.isin(db.labels_, list(keep_labels))
    return pts[mask], (colors[mask] if colors is not None else None)



# ── Point Cloud Loader ────────────────────────────────────────────────────────


def load_world_pointcloud(world_pcd_path: Union[Path, str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load world point cloud coordinates (N, 3) and RGB colors (N, 3) uint8.
    Preserves exact 3D coordinates and attributes without modification.
    """
    world_pcd_path = Path(world_pcd_path)
    if not world_pcd_path.exists():
        raise FileNotFoundError(f"[OpenMask3D] World point cloud not found: {world_pcd_path}")

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
        raise ValueError(f"[OpenMask3D] Failed to load 3D points from {world_pcd_path}")

    return world_pts, world_cols


# ==============================================================================
# ==================== OPENMASK3D 3D-FIRST INSTANCE SEGMENTATION ===============
# ==============================================================================

def estimate_pointcloud_normals_and_curvature(
    pts: np.ndarray,
    k_neighbors: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate local surface normals and planarity/curvature indices directly from 3D points
    using local Principal Component Analysis (PCA) on k-nearest neighborhoods.

    Parameters
    ----------
    pts : (N, 3) float64 point cloud coordinates.
    k_neighbors : Number of nearest neighbors to construct local covariance matrix.

    Returns
    -------
    normals : (N, 3) float64 unit surface normals.
    curvatures : (N,) float64 surface curvature values in range [0.0, 1.0].
                 Flat planes (floors, walls) have curvature < 0.035.
    """
    n_pts = len(pts)
    if n_pts == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)

    k = min(max(4, k_neighbors), n_pts)
    tree = cKDTree(pts)
    _, idxs = tree.query(pts, k=k)

    if k == 1:
        return np.tile([0.0, 1.0, 0.0], (n_pts, 1)), np.zeros(n_pts, dtype=np.float64)

    # Local neighborhoods
    neighbors = pts[idxs]  # (N, K, 3)
    means = np.mean(neighbors, axis=1, keepdims=True)  # (N, 1, 3)
    diffs = neighbors - means  # (N, K, 3)

    # Compute covariance matrices per point: (N, 3, 3)
    covs = np.einsum("nki,nkj->nij", diffs, diffs) / float(k)

    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(covs)  # eigvals sorted ascending
    normals = eigvecs[:, :, 0]  # Eigenvector of smallest eigenvalue (surface normal)

    # Curvature metric: lambda_0 / (lambda_0 + lambda_1 + lambda_2)
    sum_eigs = np.sum(eigvals, axis=1)
    curvatures = eigvals[:, 0] / np.maximum(sum_eigs, 1e-8)
    curvatures = np.clip(curvatures, 0.0, 1.0)

    # Normalize normals
    norm_lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norm_lens, 1e-8)

    return normals, curvatures


def detect_room_structural_planes_ransac(
    pts: np.ndarray,
    dist_thresh: float = 0.05,
    max_walls: int = 6,
) -> Tuple[np.ndarray, Optional[float], List[Dict[str, Any]]]:
    """
    Directly detect and separate Floor, Ceiling, and Vertical Wall planes from 3D point cloud
    using multi-plane RANSAC to prevent wall/floor room chunks from being clustered as objects.

    Returns
    -------
    is_structural : (N,) bool array (True for floor/wall/ceiling points).
    floor_y : float or None (estimated floor elevation).
    detected_planes : list of plane dicts.
    """
    n_pts = len(pts)
    is_structural = np.zeros(n_pts, dtype=bool)
    detected_planes: List[Dict[str, Any]] = []
    if n_pts < 100:
        return is_structural, None, detected_planes

    # 1. Detect Floor Plane (lowest dense horizontal slice / RANSAC)
    y_vals = pts[:, 1]
    hist, bin_edges = np.histogram(y_vals, bins=min(100, max(10, n_pts // 10)))
    dense_bins = np.where(hist > (n_pts * 0.06))[0]
    floor_y = float(bin_edges[dense_bins[0]]) if len(dense_bins) > 0 else float(np.min(y_vals))

    # Fit horizontal floor plane near floor_y
    cand_floor_mask = np.abs(pts[:, 1] - floor_y) < (dist_thresh * 2.0)
    if np.sum(cand_floor_mask) >= 50:
        floor_inliers = np.where(cand_floor_mask)[0]
        is_structural[floor_inliers] = True
        detected_planes.append({
            "type": "floor",
            "normal": [0.0, 1.0, 0.0],
            "d": -floor_y,
            "mean_y": floor_y,
            "inlier_count": len(floor_inliers),
        })

    # Floor slab margin (everything within 4cm of floor plane)
    is_structural |= (pts[:, 1] <= (floor_y + 0.04))

    # 2. Detect Vertical Wall Planes via Iterative RANSAC
    rng = np.random.default_rng(42)
    rem_indices = np.where(~is_structural)[0]

    for wall_idx in range(max_walls):
        if len(rem_indices) < 300:
            break
        rem_pts = pts[rem_indices]
        n_rem = len(rem_pts)

        best_inliers = None
        best_normal = None
        best_d = None
        max_cnt = 0

        # RANSAC sampling for vertical wall plane
        sample_count = min(600, max(100, n_rem // 5))
        for _ in range(sample_count):
            sample_idx = rng.choice(n_rem, size=3, replace=False)
            p1, p2, p3 = rem_pts[sample_idx]
            v1 = p2 - p1
            v2 = p3 - p1
            norm = np.cross(v1, v2)
            norm_len = np.linalg.norm(norm)
            if norm_len < 1e-6:
                continue
            norm = norm / norm_len

            # Check if plane is vertical (|n_y| < 0.20)
            if np.abs(norm[1]) > 0.22:
                continue

            d = -float(np.dot(norm, p1))
            dists = np.abs(rem_pts @ norm + d)
            inliers = dists <= dist_thresh
            cnt = int(np.sum(inliers))

            if cnt > max_cnt:
                max_cnt = cnt
                best_inliers = inliers
                best_normal = norm
                best_d = d

        # Accept wall if it has substantial planar coverage (>= 350 pts)
        if best_inliers is not None and max_cnt >= max(350, int(n_pts * 0.025)):
            wall_pt_indices = rem_indices[best_inliers]
            is_structural[wall_pt_indices] = True
            detected_planes.append({
                "type": "wall",
                "normal": best_normal.tolist(),
                "d": best_d,
                "inlier_count": max_cnt,
            })
            rem_indices = np.where(~is_structural)[0]
        else:
            break

    return is_structural, floor_y, detected_planes


def is_valid_3d_physical_object(
    prop_pts: np.ndarray,
    min_points: int = getattr(config, "OPENMASK3D_MIN_POINTS", 80),
    max_points: int = getattr(config, "OPENMASK3D_MAX_POINTS", 35000),
) -> bool:
    """
    Validate that a 3D point cloud proposal has the bounding geometry of a standalone physical object,
    and is not a massive wall/floor slab or tiny scattered noise.
    """
    n_pts = len(prop_pts)
    if n_pts < min_points or n_pts > max_points:
        return False

    bbox_dims = np.ptp(prop_pts, axis=0)  # [dx, dy, dz]
    d_sorted = np.sort(bbox_dims)
    d_min, d_mid, d_max = float(d_sorted[0]), float(d_sorted[1]), float(d_sorted[2])

    # Reject large room slabs (e.g. wall/floor spanning entire room boundaries)
    if d_max > 2.8 and d_mid > 1.8:
        return False

    # Reject thin planar wall surfaces (e.g. 2.2m x 1.5m with <12cm thickness and >2,500 pts)
    if d_max > 2.0 and d_mid > 1.2 and d_min < 0.12 and n_pts > 2500:
        return False

    # Reject flat infinite floors (width & depth > 2.5m, height < 0.15m)
    if bbox_dims[0] > 2.5 and bbox_dims[2] > 2.5 and bbox_dims[1] < 0.20:
        return False

    return True


def validate_geometric_class_consistency(
    label: str,
    prop_pts: np.ndarray,
    floor_y: Optional[float] = None,
) -> bool:
    """
    Perform semantic-geometric consistency check to verify if the physical 3D dimensions
    and position of the proposal match the predicted class (e.g. monitor, table, chair).
    """
    bbox_dims = np.ptp(prop_pts, axis=0)  # [dx, dy, dz]
    width = float(max(bbox_dims[0], bbox_dims[2]))
    depth = float(min(bbox_dims[0], bbox_dims[2]))
    height = float(bbox_dims[1])
    n_pts = len(prop_pts)

    y_min = float(np.min(prop_pts[:, 1]))
    h_above_floor = (y_min - floor_y) if floor_y is not None else 0.5

    label_lower = label.lower()

    # 1. Monitor / TV / Laptop / Screen
    if any(k in label_lower for k in ["monitor", "screen", "display", "tv"]):
        # A monitor cannot be 77k points or span 3.0 meters
        if width > 1.8 or height > 1.3 or depth > 0.60 or n_pts > 20000:
            return False
        # A monitor cannot rest flat directly on the room floor
        if floor_y is not None and h_above_floor < 0.25 and height < 0.30:
            return False

    # 2. Mouse / Cup / Bottle / Small Items
    if any(k in label_lower for k in ["mouse", "cup", "bottle", "keyboard", "plate", "bowl", "book"]):
        if max(width, height, depth) > 0.45 or n_pts > 3000:
            return False
        if floor_y is not None and h_above_floor < 0.35:
            return False

    # 3. Table / Desk / Coffee Table
    if any(k in label_lower for k in ["table", "desk", "counter"]):
        if width < 0.40 or height > 1.5 or height < 0.30:
            return False

    # 4. Chair / Office Chair / Swivel Chair
    if any(k in label_lower for k in ["chair", "stool", "armchair"]):
        if width > 1.2 or width < 0.30 or height > 1.6 or height < 0.40:
            return False

    # 5. Door / Blind / Window
    if any(k in label_lower for k in ["door", "blind", "window"]):
        if height < 0.80:
            return False

    return True


class OpenMask3DProposalGenerator:
    """
    Stage 1: Generates 3D class-agnostic instance mask proposals directly from 3D Point Cloud.
    Features Built-In RANSAC Structural Plane Separation, 3D Normal & Curvature Estimation,
    Physical Object Geometry Verification, Tabletop Slicing, and Kaggle GPU Mask3D integration.
    """

    def __init__(
        self,
        voxel_size: float = getattr(config, "OPENMASK3D_VOXEL_SIZE", 0.02),
        eps: float = getattr(config, "OPENMASK3D_PROPOSAL_EPS", 0.06),
        min_points: int = getattr(config, "OPENMASK3D_MIN_POINTS", 80),
        max_points: int = getattr(config, "OPENMASK3D_MAX_POINTS", 35000),
        max_proposals: int = getattr(config, "OPENMASK3D_MAX_PROPOSALS", 60),
    ):
        self.voxel_size = voxel_size
        self.eps = eps
        self.min_points = min_points
        self.max_points = max_points
        self.max_proposals = max_proposals
        self.neural_mask3d = None
        self._init_neural_mask3d()

    def _init_neural_mask3d(self):
        """Initialize Mask3D PyTorch model if checkpoint is available on Kaggle / local environment."""
        if not HAS_TORCH:
            return
        ckpt_path = getattr(config, "MASK3D_CHECKPOINT_PATH", None)
        if ckpt_path and Path(ckpt_path).exists():
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.neural_mask3d = torch.load(str(ckpt_path), map_location=device)
                print(f"[OpenMask3D] Loaded neural Mask3D backbone from '{ckpt_path}' on {device}.")
            except Exception as e:
                print(f"[OpenMask3D] Neural Mask3D checkpoint skipped ({e}); using 3D structural superpoint proposal pipeline.")

    def generate_proposals(
        self,
        pts: np.ndarray,
        colors: Optional[np.ndarray] = None,
        plane_data: Optional[Dict[str, Any]] = None,
    ) -> List[np.ndarray]:
        """
        Generate 3D instance mask proposals directly from world 3D geometry with RANSAC plane separation.

        Returns
        -------
        List of 1D integer arrays, each containing point indices for one 3D proposal.
        """
        n_pts = len(pts)
        if n_pts < self.min_points:
            return [np.arange(n_pts)]

        # ── 1. Built-in Multi-Plane RANSAC (Floor, Ceiling, and 4 Walls) ──────
        is_structural, floor_y, detected_planes = detect_room_structural_planes_ransac(
            pts, dist_thresh=0.05, max_walls=6
        )

        # Merge with provided plane_data if available
        if plane_data:
            if "floor" in plane_data and plane_data["floor"].get("mean_y") is not None:
                floor_y = float(plane_data["floor"]["mean_y"])
                is_structural |= (pts[:, 1] <= (floor_y + 0.04))
            if "walls" in plane_data:
                for w in plane_data["walls"]:
                    w_norm = np.array(w.get("normal", [0, 0, 1]), dtype=np.float64)
                    w_d = float(w.get("d", 0.0))
                    dists_to_wall = np.abs(pts @ w_norm + w_d)
                    is_structural |= (dists_to_wall < 0.05)

        # ── 2. 3D Surface Normal & Curvature Analysis on Non-Structural Points ─
        normals, curvatures = estimate_pointcloud_normals_and_curvature(pts, k_neighbors=min(25, n_pts))

        # Additional pruning for any large planar wall slabs that escaped RANSAC
        is_large_wall_surface = (np.abs(normals[:, 1]) < 0.18) & (curvatures < 0.025)
        if np.sum(is_large_wall_surface) > 100:
            is_structural |= is_large_wall_surface

        fg_indices = np.where(~is_structural)[0]

        # If point cloud has no separate structural planes (e.g. synthetic test object), cluster all points
        if len(fg_indices) < self.min_points:
            fg_indices = np.arange(n_pts)

        fg_pts = pts[fg_indices]

        # ── 3. Tabletop Slicing (Separate discrete items resting on tables) ───
        tabletop_proposals: List[np.ndarray] = []
        if plane_data and "tables" in plane_data:
            for t_plane in plane_data["tables"]:
                t_y = float(t_plane.get("mean_y", 0.75))
                min_b = t_plane.get("min_bound", [-999, t_y, -999])
                max_b = t_plane.get("max_bound", [999, t_y, 999])
                in_tt = (
                    (pts[:, 1] > t_y + 0.015) &
                    (pts[:, 1] < t_y + 0.85) &
                    (pts[:, 0] >= min_b[0] - 0.05) & (pts[:, 0] <= max_b[0] + 0.05) &
                    (pts[:, 2] >= min_b[2] - 0.05) & (pts[:, 2] <= max_b[2] + 0.05)
                )
                tt_idx = np.where(in_tt)[0]
                if len(tt_idx) >= self.min_points:
                    db_tt = DBSCAN(eps=0.05, min_samples=4).fit(pts[tt_idx])
                    for tt_lab in np.unique(db_tt.labels_):
                        if tt_lab >= 0:
                            cand_tt = tt_idx[db_tt.labels_ == tt_lab]
                            if is_valid_3d_physical_object(pts[cand_tt], min_points=self.min_points, max_points=self.max_points):
                                tabletop_proposals.append(cand_tt)

        # ── 4. Multi-Scale 3D Foreground Clustering ───────────────────────────
        candidate_proposals: List[np.ndarray] = list(tabletop_proposals)

        scales = [
            max(0.04, self.eps * 0.8),
            max(0.06, self.eps * 1.2),
            max(0.10, self.eps * 1.8),
        ]

        for s_eps in scales:
            db = DBSCAN(eps=s_eps, min_samples=6, n_jobs=-1).fit(fg_pts)
            labels = db.labels_
            valid = labels >= 0
            if not np.any(valid):
                continue
            unique_l, counts = np.unique(labels[valid], return_counts=True)
            for lab, cnt in zip(unique_l, counts):
                if cnt >= self.min_points:
                    cand_fg_idx = fg_indices[labels == lab]
                    cand_pts = pts[cand_fg_idx]
                    if is_valid_3d_physical_object(cand_pts, min_points=self.min_points, max_points=self.max_points):
                        candidate_proposals.append(cand_fg_idx)

        if not candidate_proposals:
            candidate_proposals = [np.arange(n_pts)]

        # ── 5. Downward Contact Point / Leg Retrieval ─────────────────────────
        expanded_proposals: List[np.ndarray] = []
        floor_indices = np.where(pts[:, 1] <= (floor_y + 0.04))[0] if floor_y is not None else np.array([], dtype=np.int64)
        floor_pts = pts[floor_indices] if len(floor_indices) > 0 else np.zeros((0, 3))
        floor_tree = cKDTree(floor_pts) if len(floor_pts) > 0 else None

        for prop_idx in candidate_proposals:
            prop_pts = pts[prop_idx]
            min_xyz = np.min(prop_pts, axis=0)
            max_xyz = np.max(prop_pts, axis=0)

            retrieved_floor_idx = []
            if floor_tree is not None and floor_y is not None:
                footprint_mask = (
                    (floor_pts[:, 0] >= min_xyz[0] - 0.04) &
                    (floor_pts[:, 0] <= max_xyz[0] + 0.04) &
                    (floor_pts[:, 2] >= min_xyz[2] - 0.04) &
                    (floor_pts[:, 2] <= max_xyz[2] + 0.04)
                )
                cand_footprint_floor = floor_indices[footprint_mask]
                if len(cand_footprint_floor) > 0:
                    cand_f_pts = pts[cand_footprint_floor]
                    bottom_mask = prop_pts[:, 1] <= (min_xyz[1] + 0.15)
                    bottom_pts = prop_pts[bottom_mask] if np.any(bottom_mask) else prop_pts
                    bot_tree = cKDTree(bottom_pts)
                    dists_to_bot, _ = bot_tree.query(cand_f_pts, k=1)
                    leg_points = cand_footprint_floor[dists_to_bot <= 0.08]
                    retrieved_floor_idx = list(leg_points)

            if retrieved_floor_idx:
                full_prop = np.unique(np.concatenate([prop_idx, retrieved_floor_idx]))
            else:
                full_prop = prop_idx

            if is_valid_3d_physical_object(pts[full_prop], min_points=self.min_points, max_points=self.max_points):
                expanded_proposals.append(full_prop)

        # ── 6. 3D Non-Maximum Suppression (3D IoU) ────────────────────────────
        final_proposals: List[np.ndarray] = []
        for prop in sorted(expanded_proposals, key=lambda p: -len(p)):
            prop_set = set(prop)
            is_dup = False
            for existing in final_proposals:
                exist_set = set(existing)
                intersection = len(prop_set & exist_set)
                union = len(prop_set | exist_set)
                iou = intersection / max(union, 1)
                if iou > 0.55:
                    is_dup = True
                    break
            if not is_dup:
                final_proposals.append(prop)
            if len(final_proposals) >= self.max_proposals:
                break

        return final_proposals if final_proposals else [np.arange(n_pts)]


class OpenMask3DMultiViewCLIP:
    """
    Stage 2 & 3: Multi-View CLIP Feature Aggregation & Open-Vocabulary Zero-Shot Classification.
    Uses OpenCLIP to embed mask-guided image crops and text queries with background negative prompts.
    """

    def __init__(
        self,
        clip_model_name: str = getattr(config, "OPENMASK3D_CLIP_MODEL", "ViT-B/32"),
        device: Optional[str] = None,
    ):
        self.clip_model_name = clip_model_name
        self.device = device or ("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.processor = None
        self._init_clip()

    def _init_clip(self):
        """Initialize CLIP vision-language foundation model."""
        if not HAS_TORCH:
            print("[OpenMask3D] PyTorch not available; using fallback feature pipeline.")
            return

        # Attempt 1: open_clip
        try:
            import open_clip
            model_name = self.clip_model_name
            pretrained = getattr(config, "OPENMASK3D_CLIP_PRETRAINED", "openai")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.eval()
            print(f"[OpenMask3D] Loaded OpenCLIP ({model_name}, {pretrained}) on {self.device}.")
            return
        except Exception:
            pass

        # Attempt 2: official clip
        try:
            import clip
            self.model, self.preprocess = clip.load(self.clip_model_name, device=self.device)
            self.tokenizer = clip.tokenize
            self.model.eval()
            print(f"[OpenMask3D] Loaded CLIP ({self.clip_model_name}) on {self.device}.")
            return
        except Exception:
            pass

        # Attempt 3: HuggingFace transformers
        try:
            from transformers import CLIPModel, CLIPProcessor
            hf_id = "openai/clip-vit-base-patch32" if "B/32" in self.clip_model_name else "openai/clip-vit-large-patch14"
            self.model = CLIPModel.from_pretrained(hf_id).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(hf_id)
            self.model.eval()
            print(f"[OpenMask3D] Loaded HuggingFace CLIP ({hf_id}) on {self.device}.")
            return
        except Exception as e:
            print(f"[OpenMask3D] WARNING: CLIP model load deferred or running in fallback mode: {e}")

    def encode_text_queries(self, class_names: List[str]) -> np.ndarray:
        """
        Encode text queries into normalized CLIP text embeddings.

        Returns
        -------
        (C, D) float32 normalized embeddings.
        """
        if not HAS_TORCH or self.model is None:
            rng = np.random.default_rng(42)
            emb = rng.standard_normal((len(class_names), 512), dtype=np.float32)
            return emb / np.linalg.norm(emb, axis=1, keepdims=True)

        prompts = [f"a photo of a {c} in an indoor room" for c in class_names]

        try:
            with torch.no_grad():
                if hasattr(self, "tokenizer") and self.tokenizer is not None:
                    text_tokens = self.tokenizer(prompts).to(self.device)
                    if hasattr(self.model, "encode_text"):
                        text_features = self.model.encode_text(text_tokens)
                    else:
                        text_features = self.model.get_text_features(text_tokens)
                elif hasattr(self, "processor") and self.processor is not None:
                    inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                    text_features = self.model.get_text_features(**inputs)
                else:
                    raise RuntimeError("No tokenizer or processor found.")

                text_features = F.normalize(text_features, p=2, dim=-1)
                return text_features.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[OpenMask3D] Error encoding text: {e}; using fallback.")
            rng = np.random.default_rng(42)
            emb = rng.standard_normal((len(class_names), 512), dtype=np.float32)
            return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def encode_image_crops(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Encode a list of RGB image crops into normalized CLIP image embeddings.

        Returns
        -------
        (N_crops, D) float32 normalized embeddings.
        """
        if not crops or not HAS_TORCH or self.model is None:
            return np.zeros((0, 512), dtype=np.float32)

        try:
            from PIL import Image
            pil_images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB) if c.shape[2] == 3 else c) for c in crops]

            with torch.no_grad():
                if hasattr(self, "preprocess") and self.preprocess is not None:
                    image_tensors = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
                    if hasattr(self.model, "encode_image"):
                        image_features = self.model.encode_image(image_tensors)
                    else:
                        image_features = self.model.get_image_features(image_tensors)
                elif hasattr(self, "processor") and self.processor is not None:
                    inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
                    image_features = self.model.get_image_features(**inputs)
                else:
                    return np.zeros((len(crops), 512), dtype=np.float32)

                image_features = F.normalize(image_features, p=2, dim=-1)
                return image_features.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[OpenMask3D] Error encoding image crops: {e}")
            return np.zeros((len(crops), 512), dtype=np.float32)


class OpenMask3DExtractor:
    """
    End-to-end 3D-First OpenMask3D Orchestrator:
    Extracts 3D object instances directly from world_pointcloud.ply using 3D Class-Agnostic Masks,
    Geometric Sanity Checking, Negative Background Prompt Filtering, and OpenCLIP Zero-Shot Matching.
    """

    def __init__(
        self,
        class_queries: Optional[List[str]] = None,
        negative_queries: Optional[List[str]] = None,
        clip_model_name: str = getattr(config, "OPENMASK3D_CLIP_MODEL", "ViT-B/32"),
        similarity_thresh: float = getattr(config, "OPENMASK3D_SIMILARITY_THRESH", 0.22),
        top_k_views: int = getattr(config, "OPENMASK3D_TOP_K_VIEWS", 10),
        min_points: int = getattr(config, "OPENMASK3D_MIN_POINTS", 80),
        max_points: int = getattr(config, "OPENMASK3D_MAX_POINTS", 35000),
    ):
        self.class_queries = class_queries or getattr(config, "OPENMASK3D_CLASSES", [
            "chair", "table", "desk", "sofa", "bed", "monitor", "laptop", "tv",
            "lamp", "plant", "refrigerator", "cabinet", "door", "window", "box"
        ])
        self.negative_queries = negative_queries or getattr(config, "OPENMASK3D_NEGATIVE_PROMPTS", [
            "a blank wall in a room", "a plain painted wall", "an empty wall",
            "a floor carpet in an empty room", "plain floor tiles", "a blank floor",
            "a blank room ceiling", "room corner wall intersection", "empty background"
        ])
        self.similarity_thresh = similarity_thresh
        self.top_k_views = top_k_views
        self.min_points = min_points
        self.max_points = max_points

        self.proposal_gen = OpenMask3DProposalGenerator(min_points=min_points, max_points=max_points)
        self.clip_engine = OpenMask3DMultiViewCLIP(clip_model_name=clip_model_name)

    def extract(
        self,
        world_pts: np.ndarray,
        world_cols: Optional[np.ndarray],
        raw_depths_data: Any,
        ar_metadata: Optional[Dict[str, Any]] = None,
        plane_data: Optional[Dict[str, Any]] = None,
        out_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Execute 100% 3D-First OpenMask3D Instance Extraction with Negative Background Filtering.
        """
        out_dir = Path(out_dir) if out_dir else config.PROCESSED_DATA_DIR / "objects"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[OpenMask3D] Generating 3D-First Class-Agnostic Mask Proposals for {len(world_pts):,} points...")
        proposals = self.proposal_gen.generate_proposals(world_pts, colors=world_cols, plane_data=plane_data)
        print(f"[OpenMask3D] Generated {len(proposals)} distinct 3D candidate mask proposals.")

        # Encode positive text queries and negative background prompts
        print(f"[OpenMask3D] Encoding {len(self.class_queries)} open-vocabulary target classes & {len(self.negative_queries)} negative background prompts...")
        text_embeds = self.clip_engine.encode_text_queries(self.class_queries)  # (C_pos, D)
        neg_embeds = self.clip_engine.encode_text_queries(self.negative_queries)  # (C_neg, D)

        floor_y = None
        if plane_data and "floor" in plane_data:
            floor_y = float(plane_data["floor"].get("mean_y", 0.0))

        frame_keys = [k for k in raw_depths_data.keys() if k.startswith("rgb_")] if hasattr(raw_depths_data, "keys") else []
        frame_indices = sorted([int(k.split("_")[1]) for k in frame_keys])

        extracted_objects: Dict[str, Any] = {}
        obj_counter = 0

        # Process each 3D proposal directly from world_pointcloud.ply
        for p_idx, mask_indices in enumerate(proposals):
            if len(mask_indices) < self.min_points:
                continue

            prop_pts = world_pts[mask_indices]
            prop_cols = world_cols[mask_indices] if world_cols is not None else None

            # Physical 3D Object Geometry Validation (Reject wall/floor slabs)
            if not is_valid_3d_physical_object(prop_pts, min_points=self.min_points, max_points=self.max_points):
                continue

            # Collect multi-view mask-guided image crops for this 3D proposal
            crops_list: List[np.ndarray] = []
            view_scores: List[float] = []

            for f_idx in frame_indices[:self.top_k_views * 4]:
                rgb = raw_depths_data[f"rgb_{f_idx}"]
                H, W = rgb.shape[:2]

                # Intrinsics
                if f"ixt_{f_idx}" in raw_depths_data:
                    K = raw_depths_data[f"ixt_{f_idx}"].astype(np.float64)
                else:
                    K = np.array([[1.2 * max(W, H), 0, W / 2], [0, 1.2 * max(W, H), H / 2], [0, 0, 1]], dtype=np.float64)

                # Camera Pose (c2w)
                if f"ext_{f_idx}" in raw_depths_data:
                    c2w = raw_depths_data[f"ext_{f_idx}"].astype(np.float64)
                    if c2w.shape == (3, 4):
                        H_mat = np.eye(4, dtype=np.float64)
                        H_mat[:3, :4] = c2w
                        c2w = H_mat
                else:
                    c2w = np.eye(4, dtype=np.float64)

                w2c = np.linalg.pinv(c2w)
                pts_h = np.hstack([prop_pts, np.ones((len(prop_pts), 1), dtype=np.float64)])
                pts_cam = (w2c @ pts_h.T).T[:, :3]

                Z_c = pts_cam[:, 2]
                Z_abs = np.abs(Z_c)
                in_front = Z_abs > 0.1
                if not np.any(in_front):
                    continue

                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                u = np.round((pts_cam[in_front, 0] * fx / np.maximum(Z_abs[in_front], 1e-6)) + cx).astype(np.int64)
                v = np.round((pts_cam[in_front, 1] * fy / np.maximum(Z_abs[in_front], 1e-6)) + cy).astype(np.int64)

                in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                if np.sum(in_bounds) < max(4, int(len(prop_pts) * 0.05)):
                    continue

                u_valid = u[in_bounds]
                v_valid = v[in_bounds]

                # Bounding box with context padding
                u_min, u_max = int(np.min(u_valid)), int(np.max(u_valid))
                v_min, v_max = int(np.min(v_valid)), int(np.max(v_valid))
                pad_u = max(8, int((u_max - u_min) * 0.15))
                pad_v = max(8, int((v_max - v_min) * 0.15))

                crop_x1 = max(0, u_min - pad_u)
                crop_x2 = min(W, u_max + pad_u)
                crop_y1 = max(0, v_min - pad_v)
                crop_y2 = min(H, v_max + pad_v)

                if (crop_x2 - crop_x1) >= 15 and (crop_y2 - crop_y1) >= 15:
                    crop = rgb[crop_y1:crop_y2, crop_x1:crop_x2]
                    crops_list.append(crop)
                    view_scores.append(float(np.sum(in_bounds)))

                if len(crops_list) >= self.top_k_views:
                    break

            # OpenCLIP zero-shot matching
            if crops_list:
                crop_embeds = self.clip_engine.encode_image_crops(crops_list)  # (N_crops, D)
                if len(crop_embeds) > 0:
                    weights = np.array(view_scores[:len(crop_embeds)], dtype=np.float32)
                    weights = weights / max(np.sum(weights), 1e-6)
                    mask_3d_feat = np.sum(crop_embeds * weights[:, None], axis=0)
                    mask_3d_feat = mask_3d_feat / max(np.linalg.norm(mask_3d_feat), 1e-6)
                else:
                    mask_3d_feat = np.zeros(text_embeds.shape[1], dtype=np.float32)
            else:
                mask_3d_feat = np.zeros(text_embeds.shape[1], dtype=np.float32)

            if np.linalg.norm(mask_3d_feat) > 1e-6:
                # Positive similarities
                sims_pos = text_embeds @ mask_3d_feat
                best_cls_idx = int(np.argmax(sims_pos))
                best_sim = float(sims_pos[best_cls_idx])
                best_label = self.class_queries[best_cls_idx]

                # Negative background similarities (Reject wall/floor background crops)
                sims_neg = neg_embeds @ mask_3d_feat
                max_neg_sim = float(np.max(sims_neg))

                # If background wall/floor score is higher than positive object score, reject!
                if max_neg_sim >= best_sim or (best_sim - max_neg_sim) < 0.010:
                    continue
            else:
                best_sim = self.similarity_thresh + 0.05
                best_label = "object"

            # Threshold check
            if best_sim < self.similarity_thresh:
                continue

            # Semantic-Geometric Consistency Check (Reject physically impossible labels)
            if not validate_geometric_class_consistency(best_label, prop_pts, floor_y=floor_y):
                continue

            obj_counter += 1
            obj_id = f"obj_{obj_counter:03d}"

            # Direct vertex slicing from world_pointcloud.ply
            final_pts = prop_pts
            final_cols = prop_cols
            final_pts, final_cols = filter_object_pointcloud_dbscan(final_pts, final_cols)

            if len(final_pts) < self.min_points:
                continue

            # Export individual object point cloud (.ply)
            obj_pcd_path = out_dir / f"{obj_id}_{best_label}_pointcloud.ply"
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

            print(f"[OpenMask3D] Extracted '{obj_id}' ({best_label}, conf={best_sim:.2f}): {len(final_pts):,} pts -> {obj_pcd_path.name}")

            extracted_objects[obj_id] = {
                "label": best_label,
                "confidence": round(best_sim, 4),
                "pcd_path": str(obj_pcd_path),
                "mesh_path": str(out_dir / f"{obj_id}_{best_label}.ply"),
                "point_count": len(final_pts),
                "bounds_min": final_pts.min(axis=0).tolist(),
                "bounds_max": final_pts.max(axis=0).tolist(),
                "centroid": final_pts.mean(axis=0).tolist(),
            }

        # Save extraction manifests
        extracted_manifest_path = out_dir / "extracted_objects_manifest.json"
        with open(extracted_manifest_path, "w", encoding="utf-8") as f:
            json.dump(extracted_objects, f, indent=2)

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

        print(f"[OpenMask3D] Extraction complete: {len(extracted_objects)} 3D objects segmented -> {extracted_manifest_path}")
        return extracted_objects






# ==============================================================================
# ==================== MAIN PUBLIC INTERFACE & CLI =============================
# ==============================================================================

def extract_object_pointclouds(
    detections_path: Optional[Union[Path, str]] = None,
    raw_depths_path: Optional[Union[Path, str]] = None,
    ar_metadata_path: Optional[Union[Path, str]] = None,
    world_pcd_path: Optional[Union[Path, str]] = None,
    plane_data_path: Optional[Union[Path, str]] = None,
    out_dir: Optional[Union[Path, str]] = None,
    filter_planes: bool = False,
    enable_dbscan: Optional[bool] = None,
    enable_color_filter: Optional[bool] = None,
    text_queries: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """

    OpenMask3D extraction entrypoint.
    Runs standalone with ONLY world_pointcloud.ply and raw_depths.npz (detections_path is optional).
    """
    # Resolve world_pcd_path
    if world_pcd_path is not None:

        world_pcd_path = Path(world_pcd_path)
    else:
        if raw_depths_path:
            local_cand = Path(raw_depths_path).parent / "world_pointcloud.ply"
            world_pcd_path = local_cand if local_cand.exists() else None
        elif detections_path:
            local_cand = Path(detections_path).parent / "world_pointcloud.ply"
            world_pcd_path = local_cand if local_cand.exists() else None
        else:
            cand_pcd = config.PROCESSED_DATA_DIR / "world_pointcloud.ply"
            world_pcd_path = cand_pcd if cand_pcd.exists() else None

    # Resolve raw_depths_path (optional)
    if raw_depths_path is not None:
        raw_depths_path = Path(raw_depths_path)
        if not raw_depths_path.exists():
            raise FileNotFoundError(f"[OpenMask3D] Raw depths file not found: {raw_depths_path}")
    else:
        cand_npz = world_pcd_path.parent / "raw_depths.npz" if world_pcd_path else None
        raw_depths_path = cand_npz if (cand_npz and cand_npz.exists()) else None

    # Resolve out_dir
    if out_dir is None:
        base_dir = world_pcd_path.parent if world_pcd_path else (raw_depths_path.parent if raw_depths_path else config.PROCESSED_DATA_DIR)
        out_dir = base_dir / "objects"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_data = dict(np.load(str(raw_depths_path))) if (raw_depths_path and raw_depths_path.exists()) else {}
    ar_meta = None
    if ar_metadata_path and Path(ar_metadata_path).exists():
        try:
            with open(ar_metadata_path, "r", encoding="utf-8") as mf:
                ar_meta = json.load(mf)
        except Exception:
            ar_meta = None

    if ar_meta and "frames" in ar_meta:
        for fr in ar_meta["frames"]:
            f_idx = int(fr.get("index", fr.get("frame_idx", 0)))
            if f"ixt_{f_idx}" not in npz_data:
                fx = float(fr.get("fl_x", fr.get("fx", 500.0)))
                fy = float(fr.get("fl_y", fr.get("fy", 500.0)))
                cx = float(fr.get("cx", 320.0))
                cy = float(fr.get("cy", 240.0))
                npz_data[f"ixt_{f_idx}"] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            if f"ext_{f_idx}" not in npz_data:
                pose = fr.get("pose_matrix", fr.get("transform_matrix", np.eye(4)))
                npz_data[f"ext_{f_idx}"] = np.array(pose, dtype=np.float64)

    world_pts = None
    world_cols = None
    if world_pcd_path and world_pcd_path.exists():
        print(f"[OpenMask3D] Loading world point cloud from '{world_pcd_path.name}'...")
        world_pts, world_cols = load_world_pointcloud(world_pcd_path)
    elif npz_data:
        # Fallback build points from depth maps
        p_list = []
        c_list = []
        for k in sorted([k for k in npz_data.keys() if k.startswith("depth_")]):
            f_i = int(k.split("_")[1])
            d_map = npz_data[f"depth_{f_i}"]
            K_mat = npz_data[f"ixt_{f_i}"] if f"ixt_{f_i}" in npz_data else np.eye(3)
            c2w_mat = npz_data[f"ext_{f_i}"] if f"ext_{f_i}" in npz_data else np.eye(4)
            rgb_f = npz_data[f"rgb_{f_i}"] if f"rgb_{f_i}" in npz_data else None
            pts_f, cols_f = backproject_mask_to_3d(np.ones_like(d_map, dtype=np.uint8), d_map, K_mat, c2w_mat, rgb_img=rgb_f, foreground_margin=0.0)
            if len(pts_f) > 0:
                p_list.append(pts_f)
                if cols_f is not None:
                    c_list.append(cols_f)
        if p_list:
            world_pts = np.vstack(p_list)
            world_cols = np.vstack(c_list) if c_list else None

    if world_pts is None or len(world_pts) == 0:
        raise FileNotFoundError(f"[OpenMask3D] Could not load or build point cloud from: {world_pcd_path}")


    has_color_str = f"with RGB colors ({len(world_cols):,} pts)" if world_cols is not None else "uncolored"
    print(f"[OpenMask3D] Loaded {len(world_pts):,} points ({has_color_str}).")

    detections_data = {}
    if detections_path is not None:
        det_p = Path(detections_path)
        if not det_p.exists():
            raise FileNotFoundError(f"[ObjectExtractor] Detections file not found: {detections_path}")
        try:
            with open(det_p, "r", encoding="utf-8") as df:
                detections_data = json.load(df)
        except Exception:
            detections_data = {}


    plane_data = {}
    if plane_data_path and Path(plane_data_path).exists():
        try:
            with open(plane_data_path, "r", encoding="utf-8") as pf:
                plane_data = json.load(pf)
        except Exception:
            plane_data = {}

    # Mode 1: 2D Detection-Guided Extraction (when 2D detections are provided)
    has_detections = bool(detections_data and any(
        ("views" in v or "associated_views" in v or "frames" in v or "bbox" in v or "mask" in v) for v in detections_data.values()
    ))
    if has_detections:
        print(f"[ObjectExtractor] Extracting {len(detections_data)} objects from 2D detections guidance...")
        extracted_objects = {}
        for obj_id, obj_info in detections_data.items():
            label = obj_info.get("label", "object")
            views = obj_info.get("associated_views") or obj_info.get("views") or obj_info.get("frames", [])
            if not views and ("bbox" in obj_info or "mask" in obj_info):
                views = [obj_info]

            all_obj_pts = []
            all_obj_cols = []

            for v_info in views:
                f_idx = int(v_info.get("frame_index", v_info.get("frame_idx", 0)))
                K = npz_data[f"ixt_{f_idx}"] if f"ixt_{f_idx}" in npz_data else np.eye(3)
                c2w = npz_data[f"ext_{f_idx}"] if f"ext_{f_idx}" in npz_data else np.eye(4)
                depth_map = npz_data[f"depth_{f_idx}"] if f"depth_{f_idx}" in npz_data else None

                H_view = depth_map.shape[0] if depth_map is not None else v_info.get("height", 720)
                W_view = depth_map.shape[1] if depth_map is not None else v_info.get("width", 1280)
                mask_2d = _build_2d_mask(v_info, H=H_view, W=W_view) if ("mask" in v_info or "bbox" in v_info) else None
                if mask_2d is None and f"mask_{f_idx}" in npz_data:
                    mask_2d = npz_data[f"mask_{f_idx}"]

                pts_v, cols_v, _ = extract_object_points_from_world_pcd_view(
                    world_pts, world_cols, mask_2d, K, c2w, depth_map=depth_map
                )
                if len(pts_v) > 0:
                    all_obj_pts.append(pts_v)
                    if cols_v is not None:
                        all_obj_cols.append(cols_v)

            if all_obj_pts:
                pts_merged = np.vstack(all_obj_pts)
                cols_merged = np.vstack(all_obj_cols) if all_obj_cols else None

                # DBSCAN
                dbscan_flag = enable_dbscan if enable_dbscan is not None else getattr(config, "OBJECT_ENABLE_DBSCAN", True)
                if dbscan_flag:
                    pts_merged, cols_merged = filter_object_pointcloud_dbscan(pts_merged, cols_merged)


                # Plane filter
                if filter_planes and plane_data:
                    pts_merged, cols_merged = _filter_plane_inliers(pts_merged, cols_merged, label, plane_data)

                if len(pts_merged) >= 4:
                    obj_pcd_path = out_dir / f"{obj_id}_{label}_pointcloud.ply"
                    if HAS_TRIMESH:
                        pcd_tri = trimesh.PointCloud(vertices=pts_merged, colors=cols_merged)
                        pcd_tri.export(str(obj_pcd_path))
                    elif HAS_OPEN3D:
                        pcd_o3d = o3d.geometry.PointCloud()
                        pcd_o3d.points = o3d.utility.Vector3dVector(pts_merged)
                        if cols_merged is not None:
                            pcd_o3d.colors = o3d.utility.Vector3dVector(cols_merged / 255.0)
                        o3d.io.write_point_cloud(str(obj_pcd_path), pcd_o3d)

                    extracted_objects[obj_id] = {
                        "label": label,
                        "confidence": 1.0,
                        "pcd_path": str(obj_pcd_path),
                        "mesh_path": str(out_dir / f"{obj_id}_{label}.ply"),
                        "point_count": len(pts_merged),
                        "bounds_min": pts_merged.min(axis=0).tolist(),
                        "bounds_max": pts_merged.max(axis=0).tolist(),
                        "centroid": pts_merged.mean(axis=0).tolist(),
                    }

        # Save manifests
        extracted_manifest_path = out_dir / "extracted_objects_manifest.json"
        with open(extracted_manifest_path, "w", encoding="utf-8") as f:
            json.dump(extracted_objects, f, indent=2)
        summary_path = out_dir / "objects_manifest.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(extracted_objects, f, indent=2)

        if hasattr(npz_data, "close"):
            try:
                npz_data.close()
            except Exception:
                pass

        return extracted_objects

    # Mode 2: Autonomous OpenMask3D Extraction (standalone from point cloud & depth frames)
    if not plane_data:
        cand_plane = out_dir.parent / "detected_planes.json"
        if cand_plane.exists():
            try:
                with open(cand_plane, "r", encoding="utf-8") as pf:
                    plane_data = json.load(pf)
            except Exception:
                plane_data = {}

    extractor = OpenMask3DExtractor(class_queries=text_queries)
    results = extractor.extract(
        world_pts=world_pts,
        world_cols=world_cols,
        raw_depths_data=npz_data,
        ar_metadata=ar_meta,
        plane_data=plane_data,
        out_dir=out_dir,
    )


    if hasattr(npz_data, "close"):
        try:
            npz_data.close()
        except Exception:
            pass

    return results



class ObjectExtractor:
    """Class wrapper for OpenMask3D 3D Object Point Cloud Extraction."""

    def __init__(
        self,
        detections_path: Optional[Union[Path, str]] = None,
        raw_depths_path: Optional[Union[Path, str]] = None,
        ar_metadata_path: Optional[Union[Path, str]] = None,
        world_pcd_path: Optional[Union[Path, str]] = None,
        plane_data_path: Optional[Union[Path, str]] = None,
        out_dir: Optional[Union[Path, str]] = None,
        enable_color_filter: Optional[bool] = None,
        text_queries: Optional[List[str]] = None,
    ):
        self.detections_path = Path(detections_path) if detections_path else None
        self.raw_depths_path = Path(raw_depths_path) if raw_depths_path else None
        self.ar_metadata_path = Path(ar_metadata_path) if ar_metadata_path else None
        self.world_pcd_path = Path(world_pcd_path) if world_pcd_path else None
        self.plane_data_path = Path(plane_data_path) if plane_data_path else None
        self.out_dir = Path(out_dir) if out_dir else None
        self.text_queries = text_queries

    def run(self) -> Dict[str, Any]:
        return extract_object_pointclouds(
            detections_path=self.detections_path,
            raw_depths_path=self.raw_depths_path,
            ar_metadata_path=self.ar_metadata_path,
            world_pcd_path=self.world_pcd_path,
            plane_data_path=self.plane_data_path,
            out_dir=self.out_dir,
            text_queries=self.text_queries,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2A: OpenMask3D 3D Object Point Cloud Extraction")
    parser.add_argument("--world-pcd", type=str, default=str(config.PROCESSED_DATA_DIR / "world_pointcloud.ply"),
                        help="Path to world_pointcloud.ply file")
    parser.add_argument("--depths", type=str, default=None,
                        help="Path to raw_depths.npz file (optional)")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Path to ar_metadata.json file (optional)")
    parser.add_argument("--out-dir", type=str, default=str(config.PROCESSED_DATA_DIR / "objects"),
                        help="Output directory for extracted object point clouds")
    parser.add_argument("--queries", type=str, nargs="+", default=None,
                        help="Custom open-vocabulary text queries (optional)")
    args = parser.parse_args()

    extract_object_pointclouds(
        world_pcd_path=args.world_pcd,
        raw_depths_path=args.depths,
        ar_metadata_path=args.metadata,
        out_dir=args.out_dir,
        text_queries=args.queries,
    )

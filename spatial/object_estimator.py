# -*- coding: utf-8 -*-
"""
spatial/object_estimator.py — Phase 2: 3D Back-Projection, DBSCAN & Alpha Shape Meshing.

Lifts 2D object instance masks (or bounding boxes) & depth maps into 3D camera/world space rays,
filters background bleed and noise via foreground gating & DBSCAN, and fits 3D Alpha-Shape surface meshes.
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

    Parameters
    ----------
    mask_2d : (H, W) uint8 or bool array where >0 indicates object pixels.
    depth_map : (H, W) float64/float32 metric depth values in meters.
    K : (3, 3) intrinsic camera matrix.
    c2w : (4, 4) camera-to-world pose matrix.
    rgb_img : Optional (H, W, 3) uint8 RGB image.
    depth_min, depth_max : Depth range clipping limits.
    foreground_margin : Maximum depth delta beyond median object depth to prune background bleed.
    stride : Pixel sampling stride (1 = full resolution).

    Returns
    -------
    (pts_world [N, 3], colors [N, 3] or None)
    """
    H, W = depth_map.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Valid mask & depth condition
    valid_mask = (mask_2d > 0) & (depth_map >= depth_min) & (depth_map <= depth_max) & np.isfinite(depth_map)
    if not np.any(valid_mask):
        return np.zeros((0, 3), dtype=np.float64), None

    # Foreground Depth Gating: Prune background bleed (floor/wall behind the object)
    if foreground_margin > 0:
        masked_depths = depth_map[valid_mask]
        if len(masked_depths) > 10:
            median_z = float(np.median(masked_depths))
            max_allowed_z = median_z + foreground_margin
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


def filter_object_pointcloud_dbscan(
    pts: np.ndarray,
    colors: Optional[np.ndarray] = None,
    eps: float = config.DBSCAN_EPS,
    min_samples: int = config.DBSCAN_MIN_SAMPLES,
    min_cluster_size: int = config.DBSCAN_MIN_CLUSTER_SIZE,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Remove noise points and isolate the dominant object cluster using DBSCAN.
    Prevents merging disjoint neighboring objects into a single mesh.
    """
    if len(pts) < min_samples:
        return pts, colors

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(pts)
    labels = db.labels_

    valid_mask_labels = labels >= 0
    if not np.any(valid_mask_labels):
        return pts, colors

    unique_labels, counts = np.unique(labels[valid_mask_labels], return_counts=True)
    size_filter = counts >= min_cluster_size
    valid_clusters = unique_labels[size_filter]
    valid_counts = counts[size_filter]

    if len(valid_clusters) == 0:
        if len(unique_labels) > 0:
            # Fallback: keep the single largest cluster
            dominant_label = unique_labels[np.argmax(counts)]
            selected_clusters = [dominant_label]
        else:
            return pts, colors
    else:
        # Dominant Cluster Isolation: select the primary largest cluster and closely adjacent sub-clusters
        sort_order = np.argsort(-valid_counts)
        dominant_label = valid_clusters[sort_order[0]]
        dominant_pts = pts[labels == dominant_label]
        dominant_centroid = np.mean(dominant_pts, axis=0)
        dominant_span = float(np.linalg.norm(np.max(dominant_pts, axis=0) - np.min(dominant_pts, axis=0)))
        max_merge_dist = max(0.35, dominant_span * 0.40)

        selected_clusters = [dominant_label]
        for c_idx in sort_order[1:]:
            c_label = valid_clusters[c_idx]
            c_pts = pts[labels == c_label]
            c_centroid = np.mean(c_pts, axis=0)
            # Merge if centroid is within adaptive distance threshold
            if np.linalg.norm(c_centroid - dominant_centroid) < max_merge_dist:
                selected_clusters.append(c_label)

    mask = np.isin(labels, selected_clusters)
    pts_clean = pts[mask]
    cols_clean = colors[mask] if colors is not None else None
    return pts_clean, cols_clean


def reconstruct_object_mesh_alpha_shape(
    pts: np.ndarray,
    colors: Optional[np.ndarray] = None,
    alpha: float = config.ALPHA_SHAPE_ALPHA,
    out_path: Optional[Path | str] = None,
) -> Any:
    """Fit a 3D Alpha-Shape surface mesh around object point cloud with Convex Hull fallback."""
    if len(pts) < 4:
        raise ValueError(f"[ObjectEstimator] At least 4 points needed for 3D meshing, got {len(pts)}.")

    mesh_o3d = None
    if HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        try:
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=alpha)
            mesh_o3d.remove_degenerate_triangles()
            mesh_o3d.remove_duplicated_triangles()
            mesh_o3d.remove_duplicated_vertices()
            mesh_o3d.remove_non_manifold_edges()

            if colors is not None and len(pcd.points) > 0 and len(mesh_o3d.vertices) > 0:
                from scipy.spatial import cKDTree
                tree = cKDTree(pts)
                mesh_verts = np.asarray(mesh_o3d.vertices)
                _, indices = tree.query(mesh_verts, k=1)
                v_cols = colors[indices] / 255.0
                mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(v_cols)
        except Exception as exc:
            print(f"[ObjectEstimator] Alpha Shape failed ({exc}), falling back to Convex Hull.")
            mesh_o3d = None

    if mesh_o3d is not None and HAS_TRIMESH and out_path is not None and len(mesh_o3d.vertices) > 0 and len(mesh_o3d.triangles) > 0:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        verts = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        v_cols = (np.asarray(mesh_o3d.vertex_colors) * 255).astype(np.uint8) if mesh_o3d.has_vertex_colors() else colors
        tri = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=v_cols)
        tri.export(str(out_path))
        print(f"[ObjectEstimator] Object 3D mesh saved -> {out_path}")
        return tri

    if HAS_TRIMESH:
        tri = trimesh.convex.convex_hull(pts)
        if colors is not None and len(tri.vertices) > 0:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            _, indices = tree.query(tri.vertices, k=1)
            tri.visual.vertex_colors = colors[indices]

        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tri.export(str(out_path))
            print(f"[ObjectEstimator] Object 3D mesh saved (Convex Hull) -> {out_path}")
        return tri

    return mesh_o3d


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
    foreground_margin: float = config.OBJECT_DEPTH_FOREGROUND_MARGIN,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Project 3D world points onto a 2D camera view and select points falling inside the 2D mask.

    Parameters
    ----------
    world_pts : (N, 3) float64 array of world-space points.
    world_cols : Optional (N, 3) uint8 RGB colors.
    mask_2d : (H, W) uint8 binary mask.
    K : (3, 3) intrinsic matrix.
    c2w : (4, 4) camera-to-world pose matrix.
    foreground_margin : Max depth beyond median to prune background points.

    Returns
    -------
    (selected_pts [M, 3], selected_cols [M, 3], inlier_indices [M])
    """
    if len(world_pts) == 0:
        return np.zeros((0, 3)), None, np.zeros(0, dtype=int)

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
        return np.zeros((0, 3)), None, np.zeros(0, dtype=int)

    # Perspective projection
    u = np.round((pts_cam[:, 0] * fx / np.maximum(Z_c, 1e-6)) + cx).astype(np.int64)
    v = np.round((pts_cam[:, 1] * fy / np.maximum(Z_c, 1e-6)) + cy).astype(np.int64)

    # Check bounds
    in_bounds = front_mask & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_bounds):
        return np.zeros((0, 3)), None, np.zeros(0, dtype=int)

    valid_idx = np.where(in_bounds)[0]
    in_mask_subset = mask_2d[v[valid_idx], u[valid_idx]] > 0
    matched_idx = valid_idx[in_mask_subset]

    if len(matched_idx) == 0:
        return np.zeros((0, 3)), None, np.zeros(0, dtype=int)

    # Foreground Depth Gating: Prune background surfaces behind the foreground object
    if foreground_margin > 0 and len(matched_idx) > 10:
        matched_z = Z_c[matched_idx]
        median_z = float(np.median(matched_z))
        fg_mask = matched_z <= (median_z + foreground_margin)
        matched_idx = matched_idx[fg_mask]

    selected_pts = world_pts[matched_idx]
    selected_cols = world_cols[matched_idx] if world_cols is not None else None
    return selected_pts, selected_cols, matched_idx


def process_object_detections(
    detections_path: Optional[Path | str] = None,
    raw_depths_path: Optional[Path | str] = None,
    ar_metadata_path: Optional[Path | str] = None,
    world_pcd_path: Optional[Path | str] = None,
    out_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Process object detections and reconstruct individual 3D meshes.
    Prioritizes extracting points directly from world_pointcloud.ply via 2D-guided Forward Projection,
    with automatic fallback to multi-view back-projection from raw_depths.npz.
    """
    if detections_path is None:
        detections_path = config.PROCESSED_DATA_DIR / "detections.json"
    detections_path = Path(detections_path)

    if not detections_path.exists():
        raise FileNotFoundError(
            f"[ObjectEstimator] Detections file not found: {detections_path}\n"
            "                   Please run Stage 2 (detection/reid_tracker.py) first to produce detections.json."
        )

    if raw_depths_path is None:
        raw_depths_path = config.PROCESSED_DATA_DIR / "raw_depths.npz"
    raw_depths_path = Path(raw_depths_path)

    if not raw_depths_path.exists():
        raise FileNotFoundError(
            f"[ObjectEstimator] Raw depths file not found: {raw_depths_path}\n"
            "                   Please run Stage 1 (pointcloud/depth_inference.py) first to produce raw_depths.npz."
        )

    if ar_metadata_path is None:
        ar_metadata_path = config.PROCESSED_DATA_DIR / "ar_metadata.json"
    ar_metadata_path = Path(ar_metadata_path)

    if world_pcd_path is None:
        world_pcd_path = config.PROCESSED_DATA_DIR / "world_pointcloud.ply"
    world_pcd_path = Path(world_pcd_path)

    if out_dir is None:
        out_dir = config.PROCESSED_DATA_DIR / "objects"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(detections_path, "r", encoding="utf-8") as f:
        detections = json.load(f)

    print(f"[ObjectEstimator] Loaded {len(detections)} object detections for 3D reconstruction.")

    npz = np.load(str(raw_depths_path))
    frames_meta = {}
    if ar_metadata_path.exists():
        with open(ar_metadata_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
        frames_meta = {f["index"]: f for f in meta.get("frames", [])}

    # Check if we should extract directly from world_pointcloud.ply
    use_world_pcd = getattr(config, "OBJECT_EXTRACT_FROM_WORLD_PCD", True) and world_pcd_path.exists()
    world_pts = None
    world_cols = None

    if use_world_pcd:
        print(f"[ObjectEstimator] Extracting 3D object point clouds directly from '{world_pcd_path.name}'...")
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
            except Exception as e:
                world_pts = None

        # Fallback to Open3D if trimesh failed or returned 0 points
        if (world_pts is None or len(world_pts) == 0) and HAS_OPEN3D:
            try:
                pcd = o3d.io.read_point_cloud(str(world_pcd_path))
                if len(pcd.points) > 0:
                    world_pts = np.asarray(pcd.points, dtype=np.float64)
                    world_cols = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
            except Exception as e:
                pass

        if world_pts is None or len(world_pts) == 0:
            print("[ObjectEstimator] WARNING: Could not load world point cloud; falling back to depth back-projection.")
            use_world_pcd = False
        else:
            print(f"[ObjectEstimator] Loaded {len(world_pts):,} points from world point cloud.")

    reconstructed_objects: Dict[str, Any] = {}

    for obj_id, obj_info in detections.items():
        label = obj_info.get("label", "object")
        views = obj_info.get("associated_views", [])
        if not views:
            continue

        all_obj_pts: List[np.ndarray] = []
        all_obj_cols: List[np.ndarray] = []
        matched_indices_list: List[np.ndarray] = []

        for view in views:
            f_idx = view.get("frame_index", 0)
            d_key = f"depth_{f_idx}"
            if d_key not in npz:
                continue

            depth_map = npz[d_key]
            H, W = depth_map.shape[:2]

            if f"ixt_{f_idx}" in npz:
                K = npz[f"ixt_{f_idx}"].astype(np.float64)
            else:
                K = np.array([[1.2 * max(W, H), 0, W / 2], [0, 1.2 * max(W, H), H / 2], [0, 0, 1]], dtype=np.float64)

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

            if use_world_pcd and world_pts is not None:
                # 2D-guided 3D point cloud extraction
                sel_pts, sel_cols, m_idx = extract_object_points_from_world_pcd_view(
                    world_pts, world_cols, mask_2d, K, c2w
                )
                if len(m_idx) > 0:
                    matched_indices_list.append(m_idx)
            else:
                # Fallback: depth back-projection
                rgb_img = npz[f"rgb_{f_idx}"] if f"rgb_{f_idx}" in npz else None
                pts_w, cols_w = backproject_mask_to_3d(mask_2d, depth_map, K, c2w, rgb_img=rgb_img)
                if len(pts_w) > 0:
                    all_obj_pts.append(pts_w)
                    if cols_w is not None:
                        all_obj_cols.append(cols_w)

        if use_world_pcd and matched_indices_list:
            # Combine unique indices extracted across views
            merged_idx = np.unique(np.concatenate(matched_indices_list))
            concat_pts = world_pts[merged_idx]
            concat_cols = world_cols[merged_idx] if world_cols is not None else None
        elif all_obj_pts:
            concat_pts = np.vstack(all_obj_pts)
            concat_cols = np.vstack(all_obj_cols) if all_obj_cols else None
        else:
            print(f"[ObjectEstimator] WARNING: Object '{obj_id}' ({label}) has 0 valid 3D points; skipping.")
            continue

        # Apply DBSCAN Outlier Removal & Dominant Cluster Isolation if enabled
        if getattr(config, "ENABLE_DBSCAN", True):
            clean_pts, clean_cols = filter_object_pointcloud_dbscan(concat_pts, concat_cols)
        else:
            clean_pts, clean_cols = concat_pts, concat_cols

        if clean_pts is None or len(clean_pts) < 4:
            print(f"[ObjectEstimator] WARNING: Object '{obj_id}' ({label}) has insufficient 3D points ({len(clean_pts) if clean_pts is not None else 0}) after DBSCAN; skipping.")
            continue

        obj_mesh_path = out_dir / f"{obj_id}_{label}.ply"
        mesh = reconstruct_object_mesh_alpha_shape(clean_pts, clean_cols, out_path=obj_mesh_path)

        reconstructed_objects[obj_id] = {
            "label": label,
            "mesh_path": str(obj_mesh_path),
            "point_count": len(clean_pts),
            "bounds_min": clean_pts.min(axis=0).tolist(),
            "bounds_max": clean_pts.max(axis=0).tolist(),
        }

    # Save summary manifest
    summary_path = out_dir / "objects_manifest.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(reconstructed_objects, f, indent=2)
    print(f"[ObjectEstimator] Successfully processed {len(reconstructed_objects)} objects -> {summary_path}")

    return reconstructed_objects


class ObjectEstimator:
    """Class wrapper for Object 3D Reconstruction."""

    def __init__(
        self,
        detections_path: Optional[Path | str] = None,
        raw_depths_path: Optional[Path | str] = None,
        world_pcd_path: Optional[Path | str] = None,
    ):
        self.detections_path = Path(detections_path) if detections_path else config.PROCESSED_DATA_DIR / "detections.json"
        self.raw_depths_path = Path(raw_depths_path) if raw_depths_path else config.PROCESSED_DATA_DIR / "raw_depths.npz"
        self.world_pcd_path = Path(world_pcd_path) if world_pcd_path else config.PROCESSED_DATA_DIR / "world_pointcloud.ply"

    def run(self) -> Dict[str, Any]:
        return process_object_detections(
            self.detections_path,
            raw_depths_path=self.raw_depths_path,
            world_pcd_path=self.world_pcd_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Object 3D Back-Projection & Alpha Shape Meshing")
    parser.add_argument("--detections", type=str, default=str(config.PROCESSED_DATA_DIR / "detections.json"),
                        help="Path to detections.json file")
    parser.add_argument("--depths", type=str, default=str(config.PROCESSED_DATA_DIR / "raw_depths.npz"),
                        help="Path to raw_depths.npz file")
    parser.add_argument("--world-pcd", type=str, default=str(config.PROCESSED_DATA_DIR / "world_pointcloud.ply"),
                        help="Path to world_pointcloud.ply file")
    parser.add_argument("--out-dir", type=str, default=str(config.PROCESSED_DATA_DIR / "objects"),
                        help="Output directory for reconstructed object 3D meshes")
    args = parser.parse_args()

    process_object_detections(
        detections_path=args.detections,
        raw_depths_path=args.depths,
        world_pcd_path=args.world_pcd,
        out_dir=args.out_dir,
    )

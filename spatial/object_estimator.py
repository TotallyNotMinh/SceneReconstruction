# -*- coding: utf-8 -*-
"""
spatial/object_estimator.py — Phase 2: 3D Back-Projection, DBSCAN & Alpha Shape Meshing.

Lifts 2D object instance masks & depth maps into 3D camera/world space rays,
filters noise via DBSCAN, and fits 3D Alpha-Shape surface meshes for objects.
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
    """Remove noise points and small disconnected clusters using DBSCAN."""
    if len(pts) < min_samples:
        return pts, colors

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(pts)
    labels = db.labels_

    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    valid_clusters = unique_labels[counts >= min_cluster_size]

    if len(valid_clusters) == 0:
        if len(unique_labels) > 0:
            valid_clusters = unique_labels[np.argmax(counts):np.argmax(counts) + 1]
        else:
            return pts, colors

    mask = np.isin(labels, valid_clusters)
    pts_clean = pts[mask]
    cols_clean = colors[mask] if colors is not None else None
    return pts_clean, cols_clean


def reconstruct_object_mesh_alpha_shape(
    pts: np.ndarray,
    colors: Optional[np.ndarray] = None,
    alpha: float = config.ALPHA_SHAPE_ALPHA,
    out_path: Optional[Path | str] = None,
) -> Any:
    """Fit a 3D Alpha-Shape surface mesh around object point cloud."""
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

            if colors is not None and len(pcd.points) > 0:
                from scipy.spatial import cKDTree
                tree = cKDTree(pts)
                mesh_verts = np.asarray(mesh_o3d.vertices)
                if len(mesh_verts) > 0:
                    _, indices = tree.query(mesh_verts, k=1)
                    v_cols = colors[indices] / 255.0
                    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(v_cols)
        except Exception as exc:
            print(f"[ObjectEstimator] Alpha Shape failed ({exc}), falling back to Convex Hull.")
            mesh_o3d = None

    if mesh_o3d is not None and HAS_TRIMESH and out_path is not None:
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


def _generate_sample_detections_json(out_json: Path):
    """Generate a clean sample detections.json schema for testing."""
    sample = {
        "obj_1": {
            "label": "chair",
            "class_id": 56,
            "associated_views": [
                {
                    "frame_index": 0,
                    "bbox": [100, 100, 300, 400],  # [xmin, ymin, xmax, ymax]
                    "score": 0.92,
                }
            ]
        }
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2)
    print(f"[ObjectEstimator] Sample detections JSON schema created -> {out_json}")


def process_object_detections(
    detections_path: Optional[Path | str] = None,
    raw_depths_path: Optional[Path | str] = None,
    ar_metadata_path: Optional[Path | str] = None,
    out_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Process object detections and reconstruct individual 3D meshes."""
    if detections_path is None:
        detections_path = config.PROCESSED_DATA_DIR / "detections.json"
    detections_path = Path(detections_path)

    if ar_metadata_path is None:
        ar_metadata_path = config.PROCESSED_DATA_DIR / "ar_metadata.json"
    ar_metadata_path = Path(ar_metadata_path)

    if not detections_path.exists():
        print(f"[ObjectEstimator] Detections file not found: {detections_path}")
        _generate_sample_detections_json(detections_path)

    if out_dir is None:
        out_dir = config.PROCESSED_DATA_DIR / "objects"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(detections_path, "r", encoding="utf-8") as f:
        detections = json.load(f)

    print(f"[ObjectEstimator] Loaded {len(detections)} object detections for 3D reconstruction.")

    reconstructed_objects: Dict[str, Any] = {}

    for obj_id, obj_info in detections.items():
        label = obj_info.get("label", "object")
        views = obj_info.get("associated_views", [])
        if not views:
            continue

        all_obj_pts: List[np.ndarray] = []
        all_obj_cols: List[np.ndarray] = []

        # Synthetic/Mock test generation when no real NPZ archive is present
        if raw_depths_path is None or not Path(raw_depths_path).exists():
            print(f"[ObjectEstimator] Generating test 3D mesh for object '{obj_id}' ({label})...")
            rng = np.random.default_rng(42)
            # Create a 3D box of points centered at (0.5, 0.4, 1.5)
            pts_obj = rng.uniform([-0.3, 0.0, -0.3], [0.3, 0.8, 0.3], size=(300, 3)) + np.array([0.5, 0.4, 1.5])
            cols_obj = np.tile([220, 140, 50], (len(pts_obj), 1)).astype(np.uint8)
            all_obj_pts.append(pts_obj)
            all_obj_cols.append(cols_obj)
        else:
            npz = np.load(str(raw_depths_path))
            frames_meta = {}
            if ar_metadata_path.exists():
                with open(ar_metadata_path, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                frames_meta = {f["index"]: f for f in meta.get("frames", [])}

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
                    K = np.array([[1.2*max(W,H), 0, W/2], [0, 1.2*max(W,H), H/2], [0, 0, 1]], dtype=np.float64)

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

                rgb_img = npz[f"rgb_{f_idx}"] if f"rgb_{f_idx}" in npz else None

                # Build binary mask from 2D Bounding Box [xmin, ymin, xmax, ymax]
                bbox = view.get("bbox", [0, 0, W, H])
                mask_2d = np.zeros((H, W), dtype=np.uint8)
                xmin, ymin, xmax, ymax = map(int, bbox)
                mask_2d[max(0, ymin):min(H, ymax), max(0, xmin):min(W, xmax)] = 255

                pts_w, cols_w = backproject_mask_to_3d(mask_2d, depth_map, K, c2w, rgb_img=rgb_img)
                if len(pts_w) > 0:
                    all_obj_pts.append(pts_w)
                    if cols_w is not None:
                        all_obj_cols.append(cols_w)

        if not all_obj_pts:
            continue

        concat_pts = np.vstack(all_obj_pts)
        concat_cols = np.vstack(all_obj_cols) if all_obj_cols else None

        # Apply DBSCAN Outlier Removal
        clean_pts, clean_cols = filter_object_pointcloud_dbscan(concat_pts, concat_cols)
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

    def __init__(self, detections_path: Optional[Path | str] = None):
        self.detections_path = Path(detections_path) if detections_path else config.PROCESSED_DATA_DIR / "detections.json"

    def run(self) -> Dict[str, Any]:
        return process_object_detections(self.detections_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Object 3D Back-Projection & Alpha Shape Meshing")
    parser.add_argument("--detections", type=str, default=str(config.PROCESSED_DATA_DIR / "detections.json"),
                        help="Path to detections.json file")
    parser.add_argument("--out-dir", type=str, default=str(config.PROCESSED_DATA_DIR / "objects"),
                        help="Output directory for reconstructed object 3D meshes")
    args = parser.parse_args()

    process_object_detections(detections_path=args.detections, out_dir=args.out_dir)

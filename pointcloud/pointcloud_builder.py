# -*- coding: utf-8 -*-
"""
pointcloud/pointcloud_builder.py  —  Pass 2: global depth normalisation → 3-D point cloud
"""

import sys
import io
import json
import numpy as np
import cv2
import trimesh
from pathlib import Path
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from pointcloud.filters import (
    edge_filter_depth_map,
    grazing_angle_filter_depth_map,
    free_space_violation_filter,
    tsdf_fuse,
)


DEFAULT_NPZ_PATH = config.PROCESSED_DATA_DIR / "raw_depths.npz"


def _voxel_downsample_centroid(
    pts: np.ndarray,
    colors: np.ndarray | None = None,
    voxel_size: float = 0.02,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute mean 3D position and mean RGB color per 3D voxel cell."""
    if len(pts) == 0:
        return pts, colors

    voxel_coords = np.floor(pts / voxel_size).astype(np.int64)
    unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
    m_voxels = len(unique_voxels)

    counts = np.bincount(inverse_indices, minlength=m_voxels)[:, None]
    x_sum = np.bincount(inverse_indices, weights=pts[:, 0], minlength=m_voxels)[:, None]
    y_sum = np.bincount(inverse_indices, weights=pts[:, 1], minlength=m_voxels)[:, None]
    z_sum = np.bincount(inverse_indices, weights=pts[:, 2], minlength=m_voxels)[:, None]
    pts_mean = np.hstack([x_sum, y_sum, z_sum]) / np.maximum(counts, 1)

    if colors is not None:
        r_sum = np.bincount(inverse_indices, weights=colors[:, 0].astype(np.float64), minlength=m_voxels)[:, None]
        g_sum = np.bincount(inverse_indices, weights=colors[:, 1].astype(np.float64), minlength=m_voxels)[:, None]
        b_sum = np.bincount(inverse_indices, weights=colors[:, 2].astype(np.float64), minlength=m_voxels)[:, None]
        cols_mean = np.hstack([r_sum, g_sum, b_sum]) / np.maximum(counts, 1)
        cols_mean = np.clip(cols_mean, 0, 255).astype(np.uint8)
    else:
        cols_mean = None

    return pts_mean, cols_mean


def _statistical_outlier_removal(
    pts: np.ndarray,
    cols: np.ndarray | None = None,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Remove points whose mean k-NN distance exceeds mean + std_ratio * std."""
    if len(pts) <= nb_neighbors:
        return pts, cols
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=nb_neighbors + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    mask = mean_dists <= threshold
    pts_out = pts[mask]
    cols_out = cols[mask] if cols is not None else None
    return pts_out, cols_out


def _radius_outlier_removal(
    pts: np.ndarray,
    cols: np.ndarray | None = None,
    radius: float = 0.05,
    min_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Remove points that have fewer than min_neighbors within the search radius."""
    if len(pts) <= min_neighbors:
        return pts, cols
    tree = KDTree(pts)
    counts = tree.query_ball_point(pts, r=radius, return_length=True)
    mask = counts >= (min_neighbors + 1)
    pts_out = pts[mask]
    cols_out = cols[mask] if cols is not None else None
    return pts_out, cols_out


def _cluster_outlier_removal(
    pts: np.ndarray,
    cols: np.ndarray | None = None,
    eps: float = 0.05,
    min_samples: int = 10,
    min_cluster_size: int = 50,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Remove DBSCAN noise points and clusters smaller than min_cluster_size.

    Points with label -1 (noise) and points belonging to clusters with fewer
    than min_cluster_size members are discarded as outliers.
    """
    if len(pts) < min_samples:
        return pts, cols
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(pts)
    unique_labels, label_counts = np.unique(labels[labels >= 0], return_counts=True)
    valid_clusters = unique_labels[label_counts >= min_cluster_size]
    if len(valid_clusters) == 0:
        # Fallback: keep the single largest cluster to avoid wiping everything
        if len(unique_labels) > 0:
            valid_clusters = unique_labels[np.argmax(label_counts):np.argmax(label_counts) + 1]
        else:
            return pts, cols
    mask = np.isin(labels, valid_clusters)
    pts_out = pts[mask]
    cols_out = cols[mask] if cols is not None else None
    return pts_out, cols_out


def build_pointcloud_from_npz(
    npz_path: Path | str,
    point_step: int = getattr(config, "POINTCLOUD_POINT_STEP", 2),
    return_depth_maps: bool = False,
    depth_metric_min: float = config.DEPTH_METRIC_MIN,
    depth_metric_max: float = config.DEPTH_METRIC_MAX,
    voxel_size: float = config.VOXEL_SIZE_PCD,
    conf_percentile: float = 0.0,
    sor_neighbors: int = 15,
    sor_std_ratio: float = 2.5,
    enable_ror: bool = config.ENABLE_ROR,
    ror_radius: float = config.ROR_RADIUS,
    ror_min_neighbors: int = config.ROR_MIN_NEIGHBORS,

    enable_dbscan: bool = config.ENABLE_DBSCAN,
    dbscan_eps: float = config.DBSCAN_EPS,
    dbscan_min_samples: int = config.DBSCAN_MIN_SAMPLES,
    dbscan_min_cluster_size: int = config.DBSCAN_MIN_CLUSTER_SIZE,

    enable_edge_filter: bool = config.ENABLE_EDGE_FILTER,
    edge_filter_alpha: float = config.EDGE_FILTER_ALPHA,
    edge_dilate_iters: int = config.EDGE_DILATE_ITERS,
    enable_grazing_filter: bool = config.ENABLE_GRAZING_FILTER,
    grazing_max_angle_deg: float = config.GRAZING_MAX_ANGLE_DEG,
    use_tsdf: bool = config.USE_TSDF,
    tsdf_sdf_trunc: float | None = config.TSDF_SDF_TRUNC,
    enable_free_space_check: bool = config.ENABLE_FREE_SPACE_CHECK,

    fsv_margin: float = config.FSV_MARGIN,
    fsv_violation_ratio: float = config.FSV_VIOLATION_RATIO,
    no_clean: bool = False,
    out_ply: Path | str | None = None,
):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        sys.exit(f"[ERROR] Raw depths file not found: {npz_path}\n"
                 "        Run Pass 1 first: python pointcloud/depth_inference.py <video>")

    print(f"[+] Pass 2 — loading raw depths from '{npz_path.name}'...")
    npz = np.load(str(npz_path), allow_pickle=False)

    depth_keys = sorted(
        (k for k in npz.files if k.startswith("depth_")),
        key=lambda k: int(k.split("_", 1)[1]),
    )
    if not depth_keys:
        sys.exit("[ERROR] No depth arrays found in the .npz file.")

    frame_indices = [int(k.split("_", 1)[1]) for k in depth_keys]
    n_frames = len(depth_keys)
    raw_depths = [npz[k] for k in depth_keys]

    # Safe metadata extraction with fallback to raw frame shapes
    if "video_w" in npz and "video_h" in npz:
        w = int(npz["video_w"])
        h = int(npz["video_h"])
    else:
        h, w = raw_depths[0].shape[:2]

    if "intrinsics" in npz:
        intrinsics = npz["intrinsics"].tolist()
        fx, fy, cx, cy = intrinsics
    else:
        fx = fy = 1.2 * max(w, h)
        cx, cy = w / 2.0, h / 2.0
        intrinsics = [float(fx), float(fy), float(cx), float(cy)]

    if "frames_meta" in npz:
        frames_meta: list = json.loads(npz["frames_meta"].tobytes().decode("utf-8"))
    else:
        frames_meta = []
    ext_keys   = [f"ext_{idx}" for idx in frame_indices if f"ext_{idx}" in npz]
    ixt_keys   = [f"ixt_{idx}" for idx in frame_indices if f"ixt_{idx}" in npz]
    rgb_keys   = [f"rgb_{idx}" for idx in frame_indices if f"rgb_{idx}" in npz]
    conf_keys  = [f"conf_{idx}" for idx in frame_indices if f"conf_{idx}" in npz]

    print(f"[+] {n_frames} frames loaded from archive.")

    has_predicted_poses = len(ext_keys) == n_frames
    has_colors          = len(rgb_keys) == n_frames

    print("[+] Processing depth maps...")
    depth_maps_out: dict   = {}
    all_world_points: list = []
    all_colors: list       = []
    frames_metadata_out: list = []
    frames_info: list      = []

    for i in range(n_frames):
        frame_idx = frame_indices[i]
        depth_map = raw_depths[i].astype(np.float64)

        # Edge-Aware Depth Map Filtering
        if enable_edge_filter and edge_filter_alpha > 0:
            depth_map = edge_filter_depth_map(
                depth_map,
                alpha=edge_filter_alpha,
                dilate_iters=edge_dilate_iters,
            )

        depth_maps_out[frame_idx] = depth_map
        H, W = depth_map.shape

        if has_predicted_poses and f"ixt_{frame_idx}" in npz:
            K_i = npz[f"ixt_{frame_idx}"].astype(np.float64)
        else:
            K_i = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # Grazing-Angle Depth Map Filtering
        if enable_grazing_filter and grazing_max_angle_deg < 90.0:
            depth_map = grazing_angle_filter_depth_map(
                depth_map,
                K=K_i,
                max_angle_deg=grazing_max_angle_deg,
            )

        depth_maps_out[frame_idx] = depth_map


        if has_predicted_poses and f"ext_{frame_idx}" in npz:
            ext_w2c = npz[f"ext_{frame_idx}"].astype(np.float64)
            if ext_w2c.shape == (3, 4):
                H_mat = np.eye(4, dtype=np.float64)
                H_mat[:3, :4] = ext_w2c
                ext_w2c = H_mat
            c2w = np.linalg.pinv(ext_w2c)
            # Convert OpenCV world coordinates (+Y Down) to Y-Up World coordinates (+Y Up)
            R_flip = np.diag([1.0, -1.0, -1.0, 1.0])
            c2w = R_flip @ c2w
        else:
            pose = np.array(frames_meta[i]["pose_matrix"]) if i < len(frames_meta) else np.eye(4)
            c2w = pose

        rgb_map = npz[f"rgb_{frame_idx}"] if (has_colors and f"rgb_{frame_idx}" in npz) else None
        frames_info.append({"depth": depth_map, "rgb": rgb_map, "c2w": c2w, "K": K_i})

        f_meta = dict(frames_meta[i]) if i < len(frames_meta) else {}
        f_dict = {
            "index": int(frame_idx),
            "pose_matrix": c2w.tolist(),
            "fl_x": float(K_i[0, 0]),
            "fl_y": float(K_i[1, 1]),
            "cx": float(K_i[0, 2]),
            "cy": float(K_i[1, 2]),
            "w": int(W),
            "h": int(H),
        }
        f_meta.update(f_dict)
        frames_metadata_out.append(f_dict)

        if not use_tsdf or no_clean:
            us = np.arange(0, W, point_step)
            vs = np.arange(0, H, point_step)
            uu, vv = np.meshgrid(us, vs)

            d_vals = depth_map[vv, uu].ravel()
            valid = np.isfinite(d_vals) & (d_vals > 0)

            if len(conf_keys) == n_frames and conf_percentile > 0 and f"conf_{frame_idx}" in npz:
                conf_map = npz[f"conf_{frame_idx}"]
                conf_vals = conf_map[vv, uu].ravel()
                conf_thr = np.percentile(conf_vals, float(conf_percentile))
                valid &= (conf_vals >= conf_thr)

            if np.any(valid):
                uu_v = uu.ravel()[valid]
                vv_v = vv.ravel()[valid]
                d_v  = d_vals[valid]

                pix = np.stack([uu_v, vv_v, np.ones_like(uu_v)], axis=-1)
                K_inv = np.linalg.pinv(K_i)
                rays = (K_inv @ pix.T)

                Xc = rays * d_v[None, :]
                Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])

                if has_predicted_poses and f"ext_{frame_idx}" in npz:
                    Xw = (c2w @ Xc_h)[:3].T
                else:
                    pts_cam_arkit = Xc.T.copy()
                    pts_cam_arkit[:, 2] *= -1
                    pts_cam_arkit[:, 1] *= -1
                    pts_homo = np.hstack([pts_cam_arkit, np.ones((len(pts_cam_arkit), 1))])
                    Xw = (c2w @ pts_homo.T).T[:, :3]

                all_world_points.append(Xw)

                if rgb_map is not None:
                    if rgb_map.shape[:2] != (H, W):
                        rgb_map = cv2.resize(rgb_map, (W, H))
                    colors_v = rgb_map[vv_v, uu_v]
                    all_colors.append(colors_v)

    # Point Cloud Fusion: Open3D TSDF Fusion vs. Centroid Voxel Downsampling
    if use_tsdf and not no_clean:
        if tsdf_sdf_trunc is None or tsdf_sdf_trunc < (voxel_size * 1.5):
            tsdf_sdf_trunc = voxel_size * 2.5
        print(f"[+] Running Open3D TSDF Volumetric Fusion (voxel={voxel_size}m, sdf_trunc={tsdf_sdf_trunc}m, ratio={tsdf_sdf_trunc/voxel_size:.1f}x)...")
        pts_clean, cols_clean = tsdf_fuse(
            frames_info,
            voxel_length=voxel_size,
            sdf_trunc=tsdf_sdf_trunc,
            depth_max=depth_metric_max,
        )

    else:
        if not all_world_points:
            sys.exit("[ERROR] Failed to back-project any 3-D points.")
        pts_concat = np.vstack(all_world_points).astype(np.float64)
        print(f"[+] Total raw 3-D points back-projected: {len(pts_concat):,}")
        cols_concat = np.vstack(all_colors).astype(np.uint8) if all_colors else None

        if no_clean:
            pts_clean = pts_concat
            cols_clean = cols_concat
        else:
            pts_clean, cols_clean = _voxel_downsample_centroid(pts_concat, cols_concat, voxel_size=voxel_size)

    # Multi-View Free-Space Consistency Check
    if enable_free_space_check and fsv_margin > 0 and not no_clean and len(pts_clean) > 0:
        print(f"[+] Applying Multi-View Free-Space Consistency Check (margin={fsv_margin}m, ratio={fsv_violation_ratio})...")
        pts_clean, cols_clean = free_space_violation_filter(
            pts_clean,
            cols_clean,
            frames_info,
            margin=fsv_margin,
            violation_ratio=fsv_violation_ratio,
        )

    # Post-filtering cleanup passes
    if not no_clean:
        if sor_neighbors > 0 and sor_std_ratio > 0:
            n_before = len(pts_clean)
            print(f"[+] Applying Statistical Outlier Removal (k={sor_neighbors}, std_ratio={sor_std_ratio})...")
            pts_clean, cols_clean = _statistical_outlier_removal(
                pts_clean, cols_clean, nb_neighbors=sor_neighbors, std_ratio=sor_std_ratio
            )
            print(f"    Removed {n_before - len(pts_clean):,} points.")
        if enable_ror and ror_radius > 0 and ror_min_neighbors > 0:
            n_before = len(pts_clean)
            print(f"[+] Applying Radius Outlier Removal (radius={ror_radius}, min_neighbors={ror_min_neighbors})...")
            pts_clean, cols_clean = _radius_outlier_removal(
                pts_clean, cols_clean, radius=ror_radius, min_neighbors=ror_min_neighbors
            )
            print(f"    Removed {n_before - len(pts_clean):,} points.")

        if enable_dbscan and dbscan_eps > 0:
            n_before = len(pts_clean)
            print(f"[+] Applying DBSCAN Cluster Outlier Removal (eps={dbscan_eps}, min_samples={dbscan_min_samples}, min_cluster_size={dbscan_min_cluster_size})...")
            pts_clean, cols_clean = _cluster_outlier_removal(
                pts_clean, cols_clean,
                eps=dbscan_eps,
                min_samples=dbscan_min_samples,
                min_cluster_size=dbscan_min_cluster_size,
            )
            print(f"    Removed {n_before - len(pts_clean):,} points.")


    print(f"[+] Points after cleaning: {len(pts_clean):,}")

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if cols_clean is not None:
        cloud = trimesh.PointCloud(vertices=pts_clean, colors=cols_clean)
    else:
        cloud = trimesh.PointCloud(vertices=pts_clean)
    
    target_ply_paths = [
        config.PROCESSED_DATA_DIR / "world_pointcloud.ply",
        config.OUTPUT_DIR / "world_pointcloud.ply"
    ]
    if out_ply is not None:
        target_ply_paths.append(Path(out_ply))

    for ply_path in target_ply_paths:
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        cloud.export(str(ply_path))
    print(f"[+] OK Point cloud saved ({len(pts_clean):,} pts) -> {config.OUTPUT_DIR / 'world_pointcloud.ply'}")

    orig_intrinsics = npz["orig_intrinsics"].tolist() if "orig_intrinsics" in npz else intrinsics
    scale_x = float(npz["scale_x"]) if "scale_x" in npz else 1.0
    scale_y = float(npz["scale_y"]) if "scale_y" in npz else 1.0

    metadata = {
        "intrinsics": intrinsics,
        "orig_intrinsics": orig_intrinsics,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "frames": frames_metadata_out,
    }
    for meta_path in (config.PROCESSED_DATA_DIR / "ar_metadata.json",
                      config.OUTPUT_DIR / "ar_metadata.json"):
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    print(f"[+] OK Camera metadata saved ({len(frames_metadata_out)} frames) "
          f"-> {config.PROCESSED_DATA_DIR / 'ar_metadata.json'}")

    if return_depth_maps:
        return pts_clean, metadata, depth_maps_out
    return pts_clean, metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 Pass 2: 3D Point Cloud Construction")
    parser.add_argument("npz", type=str, nargs="?", default=str(DEFAULT_NPZ_PATH), help="Path to raw_depths.npz file")
    parser.add_argument("--step", "--point-step", type=int, default=getattr(config, "POINTCLOUD_POINT_STEP", 2), dest="step", help="Pixel stride per frame (default: 2, 1 = full resolution)")
    parser.add_argument("--voxel-size", type=float, default=config.VOXEL_SIZE_PCD, help="Voxel downsampling size in meters (default: 0.015)")
    parser.add_argument("--conf-percentile", type=float, default=0.0, help="Confidence percentile threshold (default: 0.0 = no clipping, 30.0 = drop bottom 30%%)")
    parser.add_argument("--sor-neighbors", type=int, default=15, help="Statistical outlier removal number of neighbors (default: 15)")
    parser.add_argument("--sor-std-ratio", type=float, default=2.5, help="Statistical outlier removal standard deviation ratio (default: 2.5)")
    parser.add_argument("--ror", "--radius-removal", action=argparse.BooleanOptionalAction, default=config.ENABLE_ROR, help="Enable/disable Radius Outlier Removal (ROR)")
    parser.add_argument("--ror-radius", type=float, default=config.ROR_RADIUS, help="Radius Outlier Removal search radius in meters (default: 0.05)")
    parser.add_argument("--ror-min-neighbors", type=int, default=config.ROR_MIN_NEIGHBORS, help="Radius Outlier Removal minimum neighbors inside radius (default: 4)")

    parser.add_argument("--dbscan", action=argparse.BooleanOptionalAction, default=config.ENABLE_DBSCAN, help="Enable/disable DBSCAN cluster outlier removal")
    parser.add_argument("--dbscan-eps", type=float, default=config.DBSCAN_EPS, help="DBSCAN cluster outlier removal epsilon radius in meters (default: 0.05)")
    parser.add_argument("--dbscan-min-samples", type=int, default=config.DBSCAN_MIN_SAMPLES, help="DBSCAN minimum samples to form a core point (default: 10)")
    parser.add_argument("--dbscan-min-cluster-size", type=int, default=config.DBSCAN_MIN_CLUSTER_SIZE, help="DBSCAN minimum cluster size to keep (default: 50)")

    parser.add_argument("--edge-filter", action=argparse.BooleanOptionalAction, default=config.ENABLE_EDGE_FILTER, help="Enable/disable edge-aware depth map filtering")
    parser.add_argument("--edge-alpha", type=float, default=config.EDGE_FILTER_ALPHA, help="Edge-aware depth map filter alpha threshold (default: 0.05)")
    parser.add_argument("--edge-dilate-iters", type=int, default=config.EDGE_DILATE_ITERS, help="Edge-aware depth map filter dilation iterations (default: 1)")

    parser.add_argument("--grazing-filter", action=argparse.BooleanOptionalAction, default=config.ENABLE_GRAZING_FILTER, help="Enable/disable grazing-angle depth map filtering")
    parser.add_argument("--grazing-max-angle", type=float, default=config.GRAZING_MAX_ANGLE_DEG, help="Grazing-angle filter max viewing angle in degrees (default: 75.0)")

    parser.add_argument("--tsdf", action=argparse.BooleanOptionalAction, default=config.ENABLE_TSDF_FUSION, help="Enable/disable Open3D TSDF Volumetric Fusion")
    parser.add_argument("--tsdf-sdf-trunc", type=float, default=config.TSDF_SDF_TRUNC, help="TSDF SDF truncation distance in meters (default: 0.05)")

    parser.add_argument("--free-space-check", "--fsv", action=argparse.BooleanOptionalAction, default=config.ENABLE_FREE_SPACE_CHECK, help="Enable/disable multi-view free-space consistency check")
    parser.add_argument("--fsv-margin", type=float, default=config.FSV_MARGIN, help="Free-space violation depth margin in meters (default: 0.06)")
    parser.add_argument("--fsv-violation-ratio", type=float, default=config.FSV_VIOLATION_RATIO, help="Free-space violation ratio threshold (default: 0.20)")


    parser.add_argument("--no-clean", action="store_true", help="Skip voxel downsampling and save raw unmerged points")
    parser.add_argument("--out", "--output", type=str, default=None, dest="out", help="Custom output .ply path")
    args = parser.parse_args()

    _npz = Path(args.npz)
    build_pointcloud_from_npz(
        _npz,
        point_step=args.step,
        voxel_size=args.voxel_size,
        conf_percentile=args.conf_percentile,
        sor_neighbors=args.sor_neighbors,
        sor_std_ratio=args.sor_std_ratio,
        enable_ror=args.ror,
        ror_radius=args.ror_radius,
        ror_min_neighbors=args.ror_min_neighbors,
        enable_dbscan=args.dbscan,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        dbscan_min_cluster_size=args.dbscan_min_cluster_size,
        enable_edge_filter=args.edge_filter,
        edge_filter_alpha=args.edge_alpha,
        edge_dilate_iters=args.edge_dilate_iters,
        enable_grazing_filter=args.grazing_filter,
        grazing_max_angle_deg=args.grazing_max_angle,
        use_tsdf=args.tsdf,
        tsdf_sdf_trunc=args.tsdf_sdf_trunc,
        enable_free_space_check=args.free_space_check,
        fsv_margin=args.fsv_margin,
        fsv_violation_ratio=args.fsv_violation_ratio,
        no_clean=args.no_clean,
        out_ply=args.out,
    )





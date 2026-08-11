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

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

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


def build_pointcloud_from_npz(
    npz_path: Path | str,
    point_step: int = 4,
    return_depth_maps: bool = False,
    depth_metric_min: float = config.DEPTH_METRIC_MIN,
    depth_metric_max: float = config.DEPTH_METRIC_MAX,
    voxel_size: float = config.VOXEL_SIZE_PCD,
    conf_percentile: float = 0.0,
    sor_neighbors: int = 20,
    sor_std_ratio: float = 2.0,
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
    global_min  = float(min(d.min() for d in raw_depths))
    global_max  = float(max(d.max() for d in raw_depths))
    depth_range = (global_max - global_min) if global_max > global_min else 1.0
    metric_span = depth_metric_max - depth_metric_min

    def _to_metric(raw: np.ndarray) -> np.ndarray:
        normalised = (raw - global_min) / depth_range
        return depth_metric_min + normalised * metric_span

    print("[+] Back-projecting 3-D points with predicted camera poses & colors...")
    depth_maps_out: dict   = {}
    all_world_points: list = []
    all_colors: list       = []
    frames_metadata_out: list = []

    for i in range(n_frames):
        frame_idx = frame_indices[i]
        depth_map = raw_depths[i].astype(np.float64)
        depth_maps_out[frame_idx] = depth_map

        H, W = depth_map.shape

        if has_predicted_poses and f"ixt_{frame_idx}" in npz:
            K_i = npz[f"ixt_{frame_idx}"].astype(np.float64)
        else:
            K_i = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        if has_predicted_poses and f"ext_{frame_idx}" in npz:
            ext_w2c = npz[f"ext_{frame_idx}"].astype(np.float64)
            if ext_w2c.shape == (3, 4):
                H_mat = np.eye(4, dtype=np.float64)
                H_mat[:3, :4] = ext_w2c
                ext_w2c = H_mat
            c2w = np.linalg.pinv(ext_w2c)
        else:
            pose = np.array(frames_meta[i]["pose_matrix"]) if i < len(frames_meta) else np.eye(4)
            c2w = pose

        if i < len(frames_meta):
            frames_metadata_out.append(frames_meta[i])
        else:
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
            frames_metadata_out.append(f_dict)

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

        if not np.any(valid):
            continue

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

        if has_colors and f"rgb_{frame_idx}" in npz:
            rgb_map = npz[f"rgb_{frame_idx}"]
            if rgb_map.shape[:2] != (H, W):
                rgb_map = cv2.resize(rgb_map, (W, H))
            colors_v = rgb_map[vv_v, uu_v]
            all_colors.append(colors_v)

    if not all_world_points:
        sys.exit("[ERROR] Failed to back-project any 3-D points.")

    pts_concat = np.vstack(all_world_points).astype(np.float64)
    print(f"[+] Total raw 3-D points back-projected: {len(pts_concat):,}")

    if all_colors:
        cols_concat = np.vstack(all_colors).astype(np.uint8)
    else:
        cols_concat = None

    if no_clean:
        pts_clean = pts_concat
        cols_clean = cols_concat
    else:
        pts_clean, cols_clean = _voxel_downsample_centroid(pts_concat, cols_concat, voxel_size=voxel_size)
        if sor_neighbors > 0 and sor_std_ratio > 0:
            print(f"[+] Applying Statistical Outlier Removal (k={sor_neighbors}, std_ratio={sor_std_ratio})...")
            pts_clean, cols_clean = _statistical_outlier_removal(
                pts_clean, cols_clean, nb_neighbors=sor_neighbors, std_ratio=sor_std_ratio
            )

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
    parser.add_argument("--step", "--point-step", type=int, default=4, dest="step", help="Pixel stride per frame (default: 4, 1 = full resolution)")
    parser.add_argument("--voxel-size", type=float, default=config.VOXEL_SIZE_PCD, help="Voxel downsampling size in meters (default: 0.02)")
    parser.add_argument("--conf-percentile", type=float, default=0.0, help="Confidence percentile threshold (default: 0.0 = no clipping, 30.0 = drop bottom 30%%)")
    parser.add_argument("--no-clean", action="store_true", help="Skip voxel downsampling and save raw unmerged points")
    parser.add_argument("--out", "--output", type=str, default=None, dest="out", help="Custom output .ply path")
    args = parser.parse_args()

    _npz = Path(args.npz)
    build_pointcloud_from_npz(
        _npz,
        point_step=args.step,
        voxel_size=args.voxel_size,
        conf_percentile=args.conf_percentile,
        no_clean=args.no_clean,
        out_ply=args.out,
    )

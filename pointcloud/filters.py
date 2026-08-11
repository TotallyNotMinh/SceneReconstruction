# -*- coding: utf-8 -*-
"""
pointcloud/filters.py — Edge filtering, Grazing-Angle filtering, Free-Space Violation filtering, and TSDF Volumetric Fusion.
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional, Any

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def edge_filter_depth_map(
    depth_map: np.ndarray,
    alpha: float = 0.05,
    dilate_iters: int = 1,
) -> np.ndarray:
    """Invalidate depth pixels at object silhouette boundaries.

    Computes depth gradient magnitude relative to depth value (grad_mag > alpha * D)
    and dilates the resulting edge mask to remove flying pixels near depth discontinuities.

    Args:
        depth_map: 2D numpy array (H, W) of depth values.
        alpha: Relative gradient magnitude threshold factor.
        dilate_iters: Number of dilation iterations with 3x3 kernel.

    Returns:
        Filtered depth map copy with invalidated edge pixels set to 0.0.
    """
    if alpha <= 0 or not np.any(depth_map > 0):
        return depth_map

    D = depth_map.astype(np.float64)
    gy, gx = np.gradient(D)
    grad_mag = np.abs(gx) + np.abs(gy)

    # Relative thresholding
    edge_mask = (grad_mag > (alpha * D)) & (D > 0)

    if dilate_iters > 0 and np.any(edge_mask):
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel=kernel, iterations=dilate_iters).astype(bool)

    filtered_D = depth_map.copy()
    filtered_D[edge_mask] = 0.0
    return filtered_D


def grazing_angle_filter_depth_map(
    depth_map: np.ndarray,
    K: np.ndarray,
    max_angle_deg: float = 75.0,
) -> np.ndarray:
    """Invalidate pixels observed at steep grazing angles (> max_angle_deg).

    Computes 3D camera-space positions from depth and intrinsic matrix K,
    calculates surface normals via spatial gradient cross-products, and
    filters out pixels where the viewing ray angle from the surface normal
    exceeds max_angle_deg.

    Args:
        depth_map: 2D numpy array (H, W) of depth values in meters.
        K: (3, 3) intrinsic matrix.
        max_angle_deg: Maximum allowed viewing angle from surface normal (degrees).

    Returns:
        Filtered depth map copy with steep grazing angle pixels set to 0.0.
    """
    if max_angle_deg >= 90.0 or not np.any(depth_map > 0):
        return depth_map

    H, W = depth_map.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    Z = depth_map.astype(np.float64)
    valid = Z > 0

    if not np.any(valid):
        return depth_map

    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    # 3D points in camera space (H, W, 3)
    P = np.dstack([X, Y, Z])

    # Spatial partial derivatives (central differences)
    dP_du = np.zeros_like(P)
    dP_du[:, 1:-1, :] = (P[:, 2:, :] - P[:, :-2, :]) / 2.0
    dP_du[:, 0, :] = P[:, 1, :] - P[:, 0, :]
    dP_du[:, -1, :] = P[:, -1, :] - P[:, -2, :]

    dP_dv = np.zeros_like(P)
    dP_dv[1:-1, :, :] = (P[2:, :, :] - P[:-2, :, :]) / 2.0
    dP_dv[0, :, :] = P[1, :, :] - P[0, :, :]
    dP_dv[-1, :, :] = P[-1, :, :] - P[-2, :, :]

    # Surface normal N = dP_du x dP_dv
    N = np.cross(dP_du, dP_dv)
    norm_N = np.linalg.norm(N, axis=-1, keepdims=True)
    norm_N = np.maximum(norm_N, 1e-6)
    N_unit = N / norm_N

    # Ray direction from camera origin to 3D point
    norm_P = np.linalg.norm(P, axis=-1, keepdims=True)
    norm_P = np.maximum(norm_P, 1e-6)
    V_unit = P / norm_P

    # Cosine of angle between ray vector and surface normal
    cos_theta = np.abs(np.sum(V_unit * N_unit, axis=-1))

    min_cos = np.cos(np.radians(max_angle_deg))
    grazing_mask = (cos_theta < min_cos) & valid

    filtered_D = depth_map.copy()
    filtered_D[grazing_mask] = 0.0
    return filtered_D


def free_space_violation_filter(
    pts: np.ndarray,
    cols: Optional[np.ndarray],
    frames_info: List[Dict[str, Any]],
    margin: float = 0.06,
    violation_ratio: float = 0.20,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Filter 3D points that violate free space as observed by other camera views.

    For each 3D point P, project P into each camera view j. If P is behind the observed
    depth D_j(u, v) by more than margin (P_cam.z > D_j(u,v) + margin), tally a violation.
    Points exceeding the violation ratio across valid observing views are discarded.

    Args:
        pts: (N, 3) numpy array of candidate 3D points.
        cols: (N, 3) numpy array of RGB colors or None.
        frames_info: List of dicts containing 'depth', 'c2w', 'K'.
        margin: Depth violation margin in meters.
        violation_ratio: Ratio threshold of violating views to total valid views.

    Returns:
        Tuple of (filtered_pts, filtered_cols).
    """
    if len(pts) == 0 or not frames_info or margin <= 0:
        return pts, cols

    N = len(pts)
    pts_h = np.hstack([pts, np.ones((N, 1), dtype=pts.dtype)])  # (N, 4)

    observed_counts = np.zeros(N, dtype=np.int32)
    violation_counts = np.zeros(N, dtype=np.int32)

    for frame in frames_info:
        c2w = frame["c2w"]
        K = frame["K"]
        D = frame["depth"]
        H, W = D.shape[:2]

        w2c = np.linalg.pinv(c2w)
        P_cam_h = (w2c @ pts_h.T).T  # (N, 4)
        P_cam = P_cam_h[:, :3]        # (N, 3)

        z = P_cam[:, 2]
        valid_z = z > 0.1

        if not np.any(valid_z):
            continue

        # Project to pixel coordinates
        uv_h = (K @ P_cam.T).T        # (N, 3)
        z_safe = np.maximum(uv_h[:, 2], 1e-6)
        u = np.round(uv_h[:, 0] / z_safe).astype(np.int32)
        v = np.round(uv_h[:, 1] / z_safe).astype(np.int32)

        valid_uv = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(valid_uv):
            continue

        indices = np.where(valid_uv)[0]
        u_valid = u[indices]
        v_valid = v[indices]
        z_valid = z[indices]

        d_obs = D[v_valid, u_valid]
        has_depth = d_obs > 0

        indices_obs = indices[has_depth]
        d_obs_valid = d_obs[has_depth]
        z_obs_valid = z_valid[has_depth]

        observed_counts[indices_obs] += 1
        is_violating = z_obs_valid > (d_obs_valid + margin)
        violation_counts[indices_obs[is_violating]] += 1

    # Keep points with no observations or violation ratio <= threshold
    has_obs = observed_counts > 0
    ratios = np.zeros(N, dtype=np.float32)
    ratios[has_obs] = violation_counts[has_obs] / observed_counts[has_obs]

    keep_mask = (~has_obs) | (ratios <= violation_ratio)
    n_removed = N - np.count_nonzero(keep_mask)

    if n_removed > 0:
        print(f"[+] Free-Space Violation Filter: Removed {n_removed:,} / {N:,} points violating free space.")

    pts_out = pts[keep_mask]
    cols_out = cols[keep_mask] if cols is not None else None
    return pts_out, cols_out


def tsdf_fuse(
    frames_info: List[Dict[str, Any]],
    voxel_length: float = 0.02,
    sdf_trunc: Optional[float] = None,
    depth_max: float = 5.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """TSDF Volumetric Fusion using Open3D ScalableTSDFVolume.

    Args:
        frames_info: List of dicts containing 'depth', 'rgb' (optional), 'c2w', 'K'.
        voxel_length: Size of voxel grid in meters (default: 0.02m).
        sdf_trunc: Truncation distance for signed distance field in meters (default: 2.5 * voxel_length).
        depth_max: Maximum depth cutoff in meters.

    Returns:
        Tuple of (pts, cols).
    """
    if not HAS_OPEN3D:
        raise ImportError(
            "[ERROR] open3d is required for TSDF Volumetric Fusion.\n"
            "        Install it with: pip install open3d"
        )

    if voxel_length <= 0:
        raise ValueError(f"[ERROR] TSDF voxel_length must be > 0 meters, got {voxel_length}")

    if sdf_trunc is None or sdf_trunc <= 0:
        sdf_trunc = voxel_length * 2.5

    ratio_val = (sdf_trunc / voxel_length) if voxel_length > 0 else 0.0

    # Compute overall max observed depth to ensure depth_trunc does not clip valid geometry
    max_observed_depth = depth_max
    for frame in frames_info:
        d = frame["depth"]
        if np.any(d > 0):
            max_observed_depth = max(max_observed_depth, float(np.max(d)))

    print(f"[+] Initializing ScalableTSDFVolume (voxel={voxel_length}m, sdf_trunc={sdf_trunc}m, ratio={ratio_val:.1f}x)...")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i, frame in enumerate(frames_info):
        depth = frame["depth"].astype(np.float32)
        H, W = depth.shape[:2]
        c2w = frame["c2w"]
        K = frame["K"]

        if "rgb" in frame and frame["rgb"] is not None:
            rgb = frame["rgb"].astype(np.uint8)
            if rgb.shape[:2] != (H, W):
                rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            rgb = np.full((H, W, 3), 128, dtype=np.uint8)

        depth_o3d = o3d.geometry.Image(depth)
        color_o3d = o3d.geometry.Image(rgb)

        # depth_scale=1.0 matches float32 metric depth in meters exactly
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=max_observed_depth,
            convert_rgb_to_intensity=False,
        )

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=W,
            height=H,
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
        )

        w2c = np.linalg.pinv(c2w)
        volume.integrate(rgbd, intrinsic_o3d, w2c)

    pcd = volume.extract_point_cloud()
    pts = np.asarray(pcd.points).astype(np.float64)
    if len(pcd.colors) > 0:
        cols = (np.asarray(pcd.colors) * 255.0).astype(np.uint8)
    else:
        cols = None

    print(f"[+] TSDF Volumetric Fusion complete: extracted {len(pts):,} points.")
    return pts, cols

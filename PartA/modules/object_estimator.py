# modules/object_estimator.py
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

class ObjectEstimator:

  @staticmethod
  def estimate_object_3d_properties(
      global_pcd_pts: np.ndarray,
      associated_views: list,
      valid_frames_dict: dict,
      intrinsics: list,
      min_consensus: float = 0.5,
      dbscan_eps: float = 0.08,
      dbscan_min_samples: int = 8,
  ):
    fx, fy, cx, cy = intrinsics
    N = len(global_pcd_pts)
    pts_homo = np.hstack([global_pcd_pts, np.ones((N, 1))])

    inside_counts = np.zeros(N)
    valid_view_counts = np.zeros(N)
    cam_directions = []

    # 1. VECTORIZED BACK-PROJECTION & CAMERA VECTOR ACCUMULATION
    valid_views_count = 0
    for view in associated_views:
      frame_id = view["frame_id"]
      if frame_id not in valid_frames_dict:
        continue

      pose = valid_frames_dict[frame_id]["pose"]
      xmin, ymin, xmax, ymax = view["bbox_px"]

      # Hướng nhìn (+Z Forward) của Camera ở World space
      cam_dir = pose[0:3, 2]
      cam_dir[1] = 0  # Chỉ xét mặt phẳng ngang XZ
      if np.linalg.norm(cam_dir) > 0:
        cam_directions.append(cam_dir / np.linalg.norm(cam_dir))

      # World -> Camera Coordinates Transformation
      pts_cam = (np.linalg.inv(pose) @ pts_homo.T).T[:, :3]

      # Frustum Culling (Z > 0.1m)
      in_frustum = pts_cam[:, 2] > 0.1
      z_safe = np.where(in_frustum, pts_cam[:, 2], 1.0)

      # Projection sang Pixel (Pinhole Model)
      x_pix = (pts_cam[:, 0] * fx / z_safe) + cx
      y_pix = (pts_cam[:, 1] * fy / z_safe) + cy

      in_bbox = (
          in_frustum
          & (x_pix >= xmin)
          & (x_pix <= xmax)
          & (y_pix >= ymin)
          & (y_pix <= ymax)
      )

      valid_view_counts[in_frustum] += 1
      inside_counts[in_bbox] += 1
      valid_views_count += 1

    if valid_views_count == 0 or len(cam_directions) == 0:
      return None

    # 2. OCCLUSION SCORE VOTING
    valid_mask = (valid_view_counts > 0) & (
        (inside_counts / np.maximum(valid_view_counts, 1)) >= min_consensus
    )
    cropped_pts = global_pcd_pts[valid_mask]

    if len(cropped_pts) < 10:
      return None

    # 3. BACKGROUND BLEEDING REMOVAL (DBSCAN 3D)
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(cropped_pts)
    labels = db.labels_
    valid_labels = labels[labels != -1]

    if len(valid_labels) > 0:
      main_cluster_id = np.bincount(valid_labels).argmax()
      clean_pts = cropped_pts[labels == main_cluster_id]
    else:
      clean_pts = cropped_pts

    # Guard clause: Cần ít nhất 4 điểm để chạy PCA và tính BBox
    if len(clean_pts) < 4:
      return None

    # 4. ORIENTED BOUNDING BOX (OBB) & PCA YAW
    pts_xz = clean_pts[:, [0, 2]]
    pca = PCA(n_components=2).fit(pts_xz)
    main_axis = pca.components_[0]
    yaw_angle = np.arctan2(main_axis[1], main_axis[0])

    # 5. KHỬ MÙ GÓC XOAY 180 DEGREES
    avg_cam_dir = np.mean(cam_directions, axis=0)
    avg_cam_dir /= np.linalg.norm(avg_cam_dir)
    pca_dir = np.array([np.cos(yaw_angle), 0, np.sin(yaw_angle)])

    if np.dot(pca_dir, avg_cam_dir) < 0:
      yaw_angle += np.pi

    # 6. UN-ROTATE ĐIỂM VỀ LOCAL FRAME (FIXED MATH BUG)
    cos_y, sin_y = np.cos(yaw_angle), np.sin(yaw_angle)
    R_unrotate = np.array([
        [cos_y, 0, -sin_y],
        [0, 1, 0],
        [sin_y, 0, cos_y]
    ])

    local_pts = clean_pts @ R_unrotate.T
    min_b = np.min(local_pts, axis=0)
    max_b = np.max(local_pts, axis=0)

    real_dx, real_dy, real_dz = max_b - min_b
    center_3d = np.mean(clean_pts, axis=0)

    return {
        "center": center_3d,
        "size": (float(real_dx), float(real_dy), float(real_dz)),
        "yaw": float(yaw_angle),
        "min_y": float(np.min(clean_pts[:, 1])),
    }
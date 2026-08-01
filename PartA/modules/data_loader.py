import json
from pathlib import Path
from adapters.coordinate_adapter import CoordinateAdapter
import numpy as np
import open3d as o3d


class DataLoader:

  @staticmethod
  def load_ar_metadata(json_path: Path):
    if not json_path.exists():
      raise FileNotFoundError(f"Không tìm thấy file metadata tại: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
      data = json.load(f)

    intrinsics = data["intrinsics"]  # [fx, fy, cx, cy]
    valid_frames = {}

    for frame in data["frames"]:
      if frame.get("tracking_state") == "TRACKING":
        raw_pose = np.array(frame["pose_matrix"], dtype=np.float64)
        world_pose = CoordinateAdapter.arkit_to_world(raw_pose)
        valid_frames[frame["frame_id"]] = {
            "timestamp": frame["timestamp_ns"],
            "pose": world_pose,
        }

    print(f"[DataLoader] Đã nạp {len(valid_frames)} frames valid (TRACKING).")
    return intrinsics, valid_frames

  @staticmethod
  def load_point_cloud(ply_path: Path, voxel_size: float = 0.02):
    if not ply_path.exists():
      raise FileNotFoundError(
          f"Không tìm thấy file Point Cloud tại: {ply_path}"
      )

    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    if voxel_size > 0:
      if hasattr(pcd, "voxel_downsample"):
        pcd = pcd.voxel_downsample(voxel_size=voxel_size)
      elif hasattr(pcd, "voxel_down_sample"):
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Lọc outlier nhiễu
    if hasattr(pcd, "remove_statistical_outlier"):
      pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    elif hasattr(pcd, "remove_statistical_outliers"):
      pcd, _ = pcd.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)

    print(
        "[DataLoader] Đã load Point Cloud với"
        f" {len(pcd.points)} điểm sau khi dọn nhiễu."
    )
    return pcd

  @staticmethod
  def load_detections_from_person_b(json_path: Path):
    if not json_path.exists():
      raise FileNotFoundError(
          f"Không tìm thấy file detection từ B tại: {json_path}"
      )

    with open(json_path, "r", encoding="utf-8") as f:
      return json.load(f)
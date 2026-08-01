# modules/room_builder.py
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh

class RoomBuilder:

  @staticmethod
  def extract_clean_architectural_room(room_file_path: Path) -> trimesh.Scene:
    scene = trimesh.load(str(room_file_path))
    clean_room = trimesh.Scene()
    allowed_structs = ["wall", "floor", "ceiling", "door", "window"]

    if isinstance(scene, trimesh.Trimesh):
      clean_room.add_geometry(scene, node_name="Architecture_Main")
      return clean_room

    for node_name, geometry in scene.geometry.items():
      name_lower = node_name.lower()
      if any(struct in name_lower for struct in allowed_structs):
        clean_room.add_geometry(geometry, node_name=f"Arch_{node_name}")

    return clean_room

  @staticmethod
  def detect_support_surfaces_ransac(
      pcd: o3d.geometry.PointCloud, max_planes: int = 8
  ):
    surfaces = []
    pcd_working = o3d.geometry.PointCloud(pcd)
    horizontal_planes = []

    for i in range(max_planes):
      if len(pcd_working.points) < 100:
        break

      plane_model, inliers = pcd_working.segment_plane(
          distance_threshold=0.03, ransac_n=3, num_iterations=500
      )
      
      # Guard clause: Bỏ qua nếu không đủ inliers
      if len(inliers) < 20:
        break

      [A, B, C, D] = plane_model

      # Mặt phẳng nằm ngang (Normal vector song song trục Y -> |B| lớn)
      if abs(B) > 0.80:
        y_level = -D / B
        inlier_pts = np.asarray(pcd_working.select_by_index(inliers).points)
        
        if len(inlier_pts) > 0:
          min_x, max_x = np.min(inlier_pts[:, 0]), np.max(inlier_pts[:, 0])
          min_z, max_z = np.min(inlier_pts[:, 2]), np.max(inlier_pts[:, 2])

          horizontal_planes.append({
              "y_level": float(y_level),
              "bounds_xz": (min_x, max_x, min_z, max_z),
              "num_pts": len(inliers)
          })

      pcd_working = pcd_working.select_by_index(inliers, invert=True)

    if not horizontal_planes:
      print("[RoomBuilder] ⚠️ Không tìm thấy mặt phẳng bằng RANSAC. Dùng Y=0.0 mặc định.")
      return [{"type": "floor", "y_level": 0.0}]

    horizontal_planes.sort(key=lambda x: x["y_level"])

    # Mặt phẳng thấp nhất là Mặt Sàn
    actual_floor = horizontal_planes[0]
    surfaces.append({
        "type": "floor",
        "y_level": actual_floor["y_level"]
    })
    print(f"[RoomBuilder]  Phát hiện Mặt Sàn thực tế tại Y = {actual_floor['y_level']:.2f}m")

    # Các mặt phẳng cao hơn sàn từ 0.35m -> 1.30m là Mặt Bàn
    for hp in horizontal_planes[1:]:
      height_above_floor = hp["y_level"] - actual_floor["y_level"]
      if 0.35 <= height_above_floor <= 1.30:
        surfaces.append({
            "type": "table",
            "y_level": hp["y_level"],
            "bounds_xz": hp["bounds_xz"]
        })
        print(f"[RoomBuilder]  Phát hiện Mặt Bàn tại Y = {hp['y_level']:.2f}m (Cao {height_above_floor:.2f}m so với sàn)")

    return surfaces
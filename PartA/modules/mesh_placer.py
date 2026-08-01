# modules/mesh_placer.py
from adapters.coordinate_adapter import CoordinateAdapter
import numpy as np
import trimesh

class MeshPlacer:

  @staticmethod
  def align_and_place_mesh(
      triposr_mesh: trimesh.Trimesh,
      obj_3d_info: dict,
      support_surfaces: list,
      max_distortion_thresh: float = 1.35,
  ) -> trimesh.Trimesh:

    mesh = CoordinateAdapter.triposr_to_world(triposr_mesh)

    # 1. SCALE ALIGNMENT
    mesh_extent = np.maximum(mesh.extents, 1e-4)
    real_dx, real_dy, real_dz = obj_3d_info["size"]

    scales = [
        real_dx / mesh_extent[0],
        real_dy / mesh_extent[1],
        real_dz / mesh_extent[2],
    ]
    distortion_ratio = np.max(scales) / np.min(scales)

    if distortion_ratio <= max_distortion_thresh:
      # Isotropic scale theo chiều cao Y
      mesh.apply_scale(scales[1])
    else:
      # Anisotropic scale nếu tỉ lệ vật thể dị dạng nhiều
      transform_matrix = np.diag([scales[0], scales[1], scales[2], 1.0])
      mesh.apply_transform(transform_matrix)

    # 2. ROTATION
    R_yaw = trimesh.transformations.rotation_matrix(
        obj_3d_info["yaw"], [0, 1, 0]
    )
    mesh.apply_transform(R_yaw)

    # 3. SUPPORT SURFACE SNAP (Tìm sàn hoặc bàn thích hợp)
    cx, cy, cz = obj_3d_info["center"]
    
    # Lấy cao độ Mặt Sàn đã detect làm mặc định
    floor_surface = next((s for s in support_surfaces if s["type"] == "floor"), None)
    target_y = floor_surface["y_level"] if floor_surface else 0.0

    # Kiểm tra xem có nằm trên Mặt Bàn nào không
    for surface in support_surfaces:
      if surface["type"] == "table":
        is_height_close = abs(cy - surface["y_level"]) < 0.35
        min_x, max_x, min_z, max_z = surface.get("bounds_xz", (-np.inf, np.inf, -np.inf, np.inf))
        is_inside_xz = (min_x <= cx <= max_x) and (min_z <= cz <= max_z)

        if is_height_close and is_inside_xz:
          target_y = surface["y_level"]
          print(f"[MeshPlacer] 🎯 Đã snap vật thể lên Bàn tại Y = {target_y:.2f}m")
          break

    # 4. TRANSLATION (Đặt đáy vật thể tiếp xúc mặt đỡ)
    mesh_min_y = mesh.bounds[0][1]
    translation = [cx, target_y - mesh_min_y, cz]
    mesh.apply_translation(translation)

    return mesh
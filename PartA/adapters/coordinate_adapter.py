# adapters/coordinate_adapter.py
import numpy as np
import trimesh


class CoordinateAdapter:

  @staticmethod
  def arkit_to_world(pose_arkit: np.ndarray) -> np.ndarray:
    """Chuyển ARKit Camera Pose (+X Right, +Y Up, -Z Forward) 
    sang World Pinhole Pose chuẩn (+X Right, +Y Down, +Z Forward).
    Xoay 180 deg quanh trục X (det = +1, giữ nguyên tính chất SO(3)).
    """
    T = np.copy(pose_arkit)
    
    # Ma trận quay 180 độ quanh trục X
    # [1,  0,  0, 0]
    # [0, -1,  0, 0]
    # [0,  0, -1, 0]
    # [0,  0,  0, 1]
    R_cam_flip = np.eye(4)
    R_cam_flip[1, 1] = -1.0
    R_cam_flip[2, 2] = -1.0
    
    world_pose = T @ R_cam_flip
    return world_pose

  @staticmethod
  def triposr_to_world(mesh_triposr: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh_copy = mesh_triposr.copy()
    centroid = mesh_copy.bounding_box.centroid
    mesh_copy.vertices -= centroid

    # TripoSR (Z-up) -> World (Y-up)
    R = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    mesh_copy.apply_transform(R)
    return mesh_copy
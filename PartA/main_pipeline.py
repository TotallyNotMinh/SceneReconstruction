# main_pipeline.py
import config
from modules.data_loader import DataLoader
from modules.mesh_placer import MeshPlacer
from modules.object_estimator import ObjectEstimator
from modules.room_builder import RoomBuilder
import numpy as np
import open3d as o3d
import trimesh

def generate_mock_data_if_missing():
  """Tự động tạo Mock Data chuẩn hình học để test pipeline."""
  if not (config.DATA_DIR / "ar_metadata.json").exists():
    print("[Pipeline] Đang khởi tạo Mock Data mẫu để test...")
    mock_metadata = {
        "intrinsics": [1420.5, 1420.5, 960.0, 540.0],
        "frames": [{
            "frame_id": 102,
            "timestamp_ns": 1711234567890,
            "tracking_state": "TRACKING",
            "pose_matrix": np.eye(4).tolist(),
        }],
    }
    import json

    with open(config.DATA_DIR / "ar_metadata.json", "w") as f:
      json.dump(mock_metadata, f)

    # 1. Tạo Mock Point Cloud (Vật thể Z = -2.0m)
    box = o3d.geometry.TriangleMesh.create_box(0.8, 0.8, 0.8)
    box.translate([-0.4, -0.4, -2.0])
    pcd = box.sample_points_uniformly(4000)
    o3d.io.write_point_cloud(str(config.DATA_DIR / "world_pointcloud.ply"), pcd)

    # 2. Tạo MẶT SÀN DÀY (Sàn gỗ dày 10cm tại Y = -0.40m)
    room_floor = trimesh.creation.box(extents=[4.0, 0.1, 5.0])
    room_floor.apply_translation([0, -0.45, -2.0])
    room_floor.export(str(config.DATA_DIR / "room_layout.obj"))

    # 3. Tạo Mock Mesh TripoSR chuẩn (Z-up để CoordinateAdapter xoay dựng đứng thành Y-up)
    chair_triposr = trimesh.creation.box(extents=[0.8, 0.8, 0.8])
    R_triposr = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
    chair_triposr.apply_transform(R_triposr)
    chair_triposr.export(str(config.MESH_INPUT_DIR / "chair_01.obj"))

    # 4. BBox 2D
    mock_b = {
        "chair_01": {
            "associated_views": [
                {"frame_id": 102, "bbox_px": [300, 100, 1600, 1000]}
            ]
        }
    }
    with open(config.DATA_DIR / "detections_from_b.json", "w") as f:
      json.dump(mock_b, f)

def run_pipeline():
  print("\n=== STARTING PERSON A SPATIAL ENGINE PIPELINE ===\n")

  generate_mock_data_if_missing()

  ar_json = config.DATA_DIR / "ar_metadata.json"
  ply_file = config.DATA_DIR / "world_pointcloud.ply"
  b_json = config.DATA_DIR / "detections_from_b.json"
  room_file = (
      config.DATA_DIR / "room_layout.usdz"
      if (config.DATA_DIR / "room_layout.usdz").exists()
      else config.DATA_DIR / "room_layout.obj"
  )

  intrinsics, valid_frames = DataLoader.load_ar_metadata(ar_json)
  pcd = DataLoader.load_point_cloud(ply_file, config.VOXEL_SIZE_PCD)
  pcd_pts = np.asarray(pcd.points)
  detections_b = DataLoader.load_detections_from_person_b(b_json)

  room_scene = RoomBuilder.extract_clean_architectural_room(room_file)
  support_surfaces = RoomBuilder.detect_support_surfaces_ransac(pcd)

  placed_objects = {}

  for obj_id, b_data in detections_b.items():
    print(f"\n[Pipeline] Đang xử lý Object ID: {obj_id}...")

    obj_info = ObjectEstimator.estimate_object_3d_properties(
        global_pcd_pts=pcd_pts,
        associated_views=b_data["associated_views"],
        valid_frames_dict=valid_frames,
        intrinsics=intrinsics,
        min_consensus=config.OCCLUSION_MIN_CONSENSUS,
        dbscan_eps=config.DBSCAN_EPS,
        dbscan_min_samples=config.DBSCAN_MIN_SAMPLES,
    )

    if obj_info is None:
      print(
          f"⚠️ Không đủ thông tin điểm 3D để reconstruct cho object: {obj_id}"
      )
      continue

    mesh_c_path = config.MESH_INPUT_DIR / f"{obj_id}.obj"
    if not mesh_c_path.exists():
      mesh_c_path = config.MESH_INPUT_DIR / f"{obj_id}.glb"

    if not mesh_c_path.exists():
      print(f"❌ Không tìm thấy Mesh từ C cho {obj_id} tại {mesh_c_path}")
      continue

    raw_mesh = trimesh.load(str(mesh_c_path))
    if isinstance(raw_mesh, trimesh.Scene):
      raw_mesh = raw_mesh.dump(concatenate=True)

    aligned_mesh = MeshPlacer.align_and_place_mesh(
        triposr_mesh=raw_mesh,
        obj_3d_info=obj_info,
        support_surfaces=support_surfaces,
        max_distortion_thresh=config.MAX_DISTORTION_THRESH,
    )

    placed_objects[obj_id] = aligned_mesh

  final_scene = trimesh.Scene()

  for node_name, geom in room_scene.geometry.items():
    final_scene.add_geometry(geom, node_name=node_name)

  for obj_id, mesh in placed_objects.items():
    final_scene.add_geometry(mesh, node_name=f"Editable_Object_{obj_id}")

  output_path = config.OUTPUT_DIR / "digital_twin_scene.glb"
  final_scene.export(str(output_path))

  print(
      "\n=================================================================="
  )
  print(f" SUCCESS! File Digital Twin đã xuất thành công tại:\n {output_path}")
  print(
      "==================================================================\n"
  )

if __name__ == "__main__":
  run_pipeline()
# -*- coding: utf-8 -*-
"""
tests/test_pipeline.py — Unit test suite for Scene Reconstruction pipeline components.
"""

import sys
import unittest
import tempfile
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.coordinate_adapter import CoordinateAdapter
from core.data_loader import DataLoader, _voxel_downsample, _remove_statistical_outliers
from core.video_normalizer import get_scaled_resolution, rescale_intrinsics, rotate_intrinsics
from pointcloud.pointcloud_builder import build_pointcloud_from_npz, _radius_outlier_removal, _cluster_outlier_removal


class TestCoordinateAdapter(unittest.TestCase):

    def test_arkit_to_world(self):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = 1.0
        pose[1, 3] = 2.0
        pose[2, 3] = 3.0

        world_pose = CoordinateAdapter.arkit_to_world(pose)
        self.assertEqual(world_pose.shape, (4, 4))
        # Determinant of 3x3 rotation block should equal +1.0 (SO(3))
        det = np.linalg.det(world_pose[:3, :3])
        self.assertAlmostEqual(det, 1.0, places=5)
        # Check translation coordinates (camera center in world space is preserved)
        self.assertAlmostEqual(world_pose[0, 3], 1.0)
        self.assertAlmostEqual(world_pose[1, 3], 2.0)
        self.assertAlmostEqual(world_pose[2, 3], 3.0)


class TestVoxelDownsample(unittest.TestCase):

    def test_voxel_downsample_positive_and_negative_coords(self):
        # Create points in positive and negative coordinate quadrants
        pts = np.array([
            [0.001, 0.001, 0.001],
            [0.005, 0.005, 0.005],  # Should merge into same 0.02m voxel as above
            [-0.005, -0.005, -0.005],
            [-0.008, -0.008, -0.008], # Should merge into same negative voxel as above
            [1.0, 1.0, 1.0],         # Distinct voxel
        ], dtype=np.float64)

        clean = _voxel_downsample(pts, voxel_size=0.02)
        self.assertEqual(len(clean), 3)

    def test_voxel_downsample_empty(self):
        pts = np.zeros((0, 3), dtype=np.float64)
        clean = _voxel_downsample(pts, voxel_size=0.02)
        self.assertEqual(len(clean), 0)


class TestDataLoader(unittest.TestCase):

    def test_load_depth_maps_with_prefixed_keys(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            npz_path = Path(tmp_dir) / "test_depths.npz"
            d0 = np.ones((10, 10), dtype=np.float32) * 1.5
            d5 = np.ones((10, 10), dtype=np.float32) * 2.5
            np.savez(str(npz_path), depth_0=d0, depth_5=d5)

            depth_maps = DataLoader.load_depth_maps(npz_path)
            self.assertIn(0, depth_maps)
            self.assertIn(5, depth_maps)
            self.assertEqual(len(depth_maps), 2)
            np.testing.assert_array_almost_equal(depth_maps[0], d0)
            np.testing.assert_array_almost_equal(depth_maps[5], d5)


class TestVideoNormalizerHelpers(unittest.TestCase):

    def test_get_scaled_resolution(self):
        w, h, sx, sy = get_scaled_resolution(1920, 1080, target_long_edge=720)
        self.assertEqual(w, 720)
        self.assertEqual(h % 2, 0)  # Must be even
        self.assertAlmostEqual(sx, 720 / 1920, places=5)
        self.assertAlmostEqual(sy, h / 1080, places=5)

    def test_rescale_intrinsics(self):
        k_orig = [1000.0, 1000.0, 500.0, 500.0]
        k_rescaled = rescale_intrinsics(k_orig, 0.5, 0.5)
        self.assertEqual(k_rescaled, [500.0, 500.0, 250.0, 250.0])


class TestPointcloudBuilder(unittest.TestCase):

    def test_build_pointcloud_from_npz_non_contiguous(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            npz_path = Path(tmp_dir) / "test_raw_depths.npz"
            d0 = np.ones((20, 20), dtype=np.float32) * 2.0
            d8 = np.ones((20, 20), dtype=np.float32) * 3.0
            ext0 = np.eye(4, dtype=np.float32)
            ext8 = np.eye(4, dtype=np.float32)
            ext8[0, 3] = 0.5

            np.savez(
                str(npz_path),
                video_w=np.int64(20),
                video_h=np.int64(20),
                intrinsics=np.array([20.0, 20.0, 10.0, 10.0]),
                depth_0=d0,
                depth_8=d8,
                ext_0=ext0,
                ext_8=ext8,
            )

            out_ply = Path(tmp_dir) / "output.ply"
            pts, meta = build_pointcloud_from_npz(
                npz_path,
                point_step=2,
                voxel_size=0.05,
                out_ply=out_ply,
            )
            self.assertGreater(len(pts), 0)
            self.assertTrue(out_ply.exists())
            self.assertEqual(len(meta["frames"]), 2)


class TestRadiusOutlierRemoval(unittest.TestCase):

    def test_radius_outlier_removal_basic(self):
        # A dense cluster of 6 points, and 1 isolated point far away
        pts = np.array([
            [0.0, 0.0, 0.0],
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.01],
            [5.0, 5.0, 5.0]  # Isolated point
        ], dtype=np.float64)
        
        # cols: dummy rgb values
        cols = np.ones((7, 3), dtype=np.uint8) * 255

        # If search radius = 0.1m, and min_neighbors = 5
        # The 6 points near origin each have 5 other points as neighbors (count >= 6).
        # The isolated point has 0 neighbors (count = 1).
        clean_pts, clean_cols = _radius_outlier_removal(pts, cols, radius=0.1, min_neighbors=5)

        self.assertEqual(len(clean_pts), 6)
        self.assertEqual(len(clean_cols), 6)
        # Verify the isolated point was removed
        for pt in clean_pts:
            self.assertNotEqual(pt[0], 5.0)

    def test_radius_outlier_removal_empty(self):
        pts = np.zeros((0, 3), dtype=np.float64)
        clean_pts, _ = _radius_outlier_removal(pts, None, radius=0.1, min_neighbors=5)
        self.assertEqual(len(clean_pts), 0)


class TestClusterOutlierRemoval(unittest.TestCase):

    def test_dbscan_removes_isolated_noise(self):
        rng = np.random.default_rng(42)
        # Dense cluster of 200 tightly packed points around origin
        dense = rng.uniform(-0.1, 0.1, size=(200, 3))
        # 5 isolated noise points far away
        noise = np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [-5.0, -5.0, -5.0],
                          [7.0, 0.0, 0.0], [0.0, 7.0, 0.0]], dtype=np.float64)
        pts = np.vstack([dense, noise])
        cols = (np.ones((len(pts), 3)) * 128).astype(np.uint8)

        clean_pts, clean_cols = _cluster_outlier_removal(
            pts, cols, eps=0.3, min_samples=5, min_cluster_size=50
        )
        # Noise points should be removed; the 200-point cluster should survive
        self.assertGreaterEqual(len(clean_pts), 150)
        # None of the isolated noise points should survive
        for pt in clean_pts:
            self.assertLess(np.max(np.abs(pt)), 3.0)
        self.assertEqual(len(clean_pts), len(clean_cols))

    def test_dbscan_empty(self):
        pts = np.zeros((0, 3), dtype=np.float64)
        clean_pts, _ = _cluster_outlier_removal(pts, None, eps=0.1, min_samples=5, min_cluster_size=10)
        self.assertEqual(len(clean_pts), 0)


from pointcloud.filters import (
    edge_filter_depth_map,
    grazing_angle_filter_depth_map,
    free_space_violation_filter,
    tsdf_fuse,
)


class TestFilters(unittest.TestCase):

    def test_edge_filter_depth_map(self):
        # Create a depth map with a sharp step boundary (depth discontinuity)
        D = np.ones((50, 50), dtype=np.float64) * 2.0
        D[:, 25:] = 5.0  # Sharp step from 2.0 to 5.0 at column 25

        filtered_D = edge_filter_depth_map(D, alpha=0.05, dilate_iters=1)
        # Edge pixels around column 24-26 should be invalidated (set to 0)
        self.assertEqual(filtered_D[:, 24].sum(), 0.0)
        self.assertEqual(filtered_D[:, 25].sum(), 0.0)
        # Smooth regions away from boundary should remain unmasked
        self.assertTrue(np.all(filtered_D[:, 5] == 2.0))
        self.assertTrue(np.all(filtered_D[:, 40] == 5.0))

    def test_grazing_angle_filter_depth_map(self):
        # Create a depth map of a sphere/cylinder where edges have steep grazing angles relative to camera ray
        H, W = 100, 100
        K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        
        # A hemisphere at center
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)
        r_sq = (uu - 50)**2 + (vv - 50)**2
        R_sq = 40**2
        mask = r_sq < R_sq
        
        D = np.zeros((H, W), dtype=np.float64)
        D[mask] = 2.0 - np.sqrt(R_sq - r_sq[mask]) * 0.02
        
        filtered_D = grazing_angle_filter_depth_map(D, K, max_angle_deg=60.0)
        # Center of hemisphere has normal pointing back at camera (low grazing angle) -> kept
        self.assertGreater(filtered_D[50, 50], 0.0)
        # Outer boundary of hemisphere has steep grazing angle -> filtered out (set to 0)
        valid_before = np.count_nonzero(D > 0)
        valid_after = np.count_nonzero(filtered_D > 0)
        self.assertLess(valid_after, valid_before)

    def test_free_space_violation_filter(self):

        # Frame at origin looking along +Z (identity pose)
        c2w = np.eye(4, dtype=np.float64)
        K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.ones((100, 100), dtype=np.float64) * 2.0  # Observed depth is at Z=2.0m

        # Point A at Z=2.0 (consistent with observed surface)
        # Point B at Z=4.0 (behind observed surface at Z=2.0 -> free space violation)
        pts = np.array([
            [0.0, 0.0, 2.0],  # Valid point at surface
            [0.0, 0.0, 4.0],  # Flying pixel / streak artifact behind surface
        ], dtype=np.float64)
        cols = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

        frames_info = [{"depth": D, "c2w": c2w, "K": K}]

        clean_pts, clean_cols = free_space_violation_filter(
            pts, cols, frames_info, margin=0.1, violation_ratio=0.5
        )

        # Point B (Z=4.0) should be removed as a free space violation
        self.assertEqual(len(clean_pts), 1)
        np.testing.assert_almost_equal(clean_pts[0], [0.0, 0.0, 2.0])

    def test_tsdf_fuse_integration(self):
        # Create 2 simple synthetic frames looking at a plane at Z=2.0
        c2w_1 = np.eye(4, dtype=np.float64)
        c2w_2 = np.eye(4, dtype=np.float64)
        c2w_2[0, 3] = 0.1  # Slight camera shift on X axis

        K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.ones((100, 100), dtype=np.float32) * 2.0
        RGB = np.ones((100, 100, 3), dtype=np.uint8) * 200

        frames_info = [
            {"depth": D, "rgb": RGB, "c2w": c2w_1, "K": K},
            {"depth": D, "rgb": RGB, "c2w": c2w_2, "K": K},
        ]

        pts, cols = tsdf_fuse(frames_info, voxel_length=0.05, sdf_trunc=0.15, depth_max=4.0)
        self.assertGreater(len(pts), 0)
        self.assertIsNotNone(cols)


from spatial.room_builder import (
    detect_architectural_planes,
    _compute_oriented_wall_geometry,
    extract_room_background_pointcloud,
    reconstruct_room_background_mesh,
    build_room_background,
)
from spatial.object_extractor import (
    ObjectExtractor,
    extract_object_pointclouds,
    extract_object_points_from_world_pcd_view,
    filter_object_pointcloud_dbscan,
    load_world_pointcloud,
    _filter_plane_inliers,
)
from spatial.object_mesher import (
    ObjectMesher,
    reconstruct_object_mesh,
    reconstruct_object_meshes,
)
from spatial.object_estimator import (
    ObjectEstimator,
    backproject_mask_to_3d,
    process_object_detections,
)
from spatial.mesh_placer import (
    MeshPlacer,
    snap_mesh_to_surface,
    snap_mesh_to_wall,
    align_and_place_object_meshes,
    assemble_full_scene,
)

import trimesh
import json


def _create_test_room_pointcloud(out_ply: Path, num_points: int = 8000) -> np.ndarray:
    """Helper fixture to generate a temporary test point cloud for spatial tests."""
    rng = np.random.default_rng(42)

    # Floor at Y = 0.0, bounds X: [-2.5, 2.5], Z: [-2.5, 2.5]
    n_floor = int(num_points * 0.45)
    floor_x = rng.uniform(-2.5, 2.5, n_floor)
    floor_z = rng.uniform(-2.5, 2.5, n_floor)
    floor_y = rng.normal(0.0, 0.005, n_floor)
    floor_pts = np.column_stack([floor_x, floor_y, floor_z])
    floor_cols = np.tile([180, 180, 180], (n_floor, 1)).astype(np.uint8)

    # Vertical Wall at Z = -2.0, bounds X: [-2.0, 2.0], Y: [0.0, 2.2]
    n_wall = int(num_points * 0.25)
    wall_x = rng.uniform(-2.0, 2.0, n_wall)
    wall_y = rng.uniform(0.0, 2.2, n_wall)
    wall_z = rng.normal(-2.0, 0.005, n_wall)
    wall_pts = np.column_stack([wall_x, wall_y, wall_z])
    wall_cols = np.tile([210, 210, 200], (n_wall, 1)).astype(np.uint8)

    # Tabletop at Y = 0.75, bounds X: [-0.6, 0.6], Z: [-0.4, 0.4]
    n_table = int(num_points * 0.15)
    table_x = rng.uniform(-0.6, 0.6, n_table)
    table_z = rng.uniform(-0.4, 0.4, n_table)
    table_y = rng.normal(0.75, 0.005, n_table)
    table_pts = np.column_stack([table_x, table_y, table_z])
    table_cols = np.tile([130, 90, 50], (n_table, 1)).astype(np.uint8)

    # Clutter / Object points
    n_obj = num_points - n_floor - n_wall - n_table
    obj_x = rng.uniform(-1.0, 1.0, n_obj)
    obj_z = rng.uniform(-1.0, 1.0, n_obj)
    obj_y = rng.uniform(0.1, 1.0, n_obj)
    obj_pts = np.column_stack([obj_x, obj_y, obj_z])
    obj_cols = np.tile([50, 120, 200], (n_obj, 1)).astype(np.uint8)

    pts = np.vstack([floor_pts, wall_pts, table_pts, obj_pts])
    cols = np.vstack([floor_cols, wall_cols, table_cols, obj_cols])

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    cloud = trimesh.PointCloud(vertices=pts, colors=cols)
    cloud.export(str(out_ply))
    return pts


class TestSpatialPhase1RoomBuilder(unittest.TestCase):

    def test_detect_architectural_planes_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ply_path = Path(tmp_dir) / "synthetic_room.ply"
            _create_test_room_pointcloud(ply_path)
            out_obj = Path(tmp_dir) / "room_layout.obj"
            out_json = Path(tmp_dir) / "detected_planes.json"

            result = detect_architectural_planes(
                ply_path=ply_path,
                distance_threshold=0.04,
                max_planes=4,
                out_obj=out_obj,
                out_json=out_json,
            )

            self.assertIsNotNone(result["floor"])
            self.assertAlmostEqual(result["floor"]["mean_y"], 0.0, delta=0.1)
            self.assertIn("walls", result)
            self.assertTrue(len(result["walls"]) > 0)
            self.assertTrue(out_obj.exists())
            self.assertTrue(out_json.exists())

            # Verify Oriented Bounding Box geometry
            for wall in result["walls"]:
                self.assertIn("oriented_box", wall)
                obb = wall["oriented_box"]
                self.assertGreater(obb["length"], 0.0)
                self.assertGreater(obb["height"], 0.0)
                self.assertAlmostEqual(obb["thickness"], config.WALL_THICKNESS)
                self.assertEqual(len(obb["transform_matrix"]), 4)

            # Verify plane equation normalization
            for plane in result["all_planes"]:
                eq = plane["equation"]
                norm_len = np.linalg.norm(eq[:3])
                self.assertAlmostEqual(norm_len, 1.0, places=5)

    def test_detect_architectural_planes_ceiling_separation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ply_path = Path(tmp_dir) / "synthetic_room_with_ceiling.ply"
            rng = np.random.default_rng(42)

            # Floor at Y = 0.0
            n_floor = 2000
            floor_x = rng.uniform(-2.5, 2.5, n_floor)
            floor_z = rng.uniform(-2.5, 2.5, n_floor)
            floor_y = rng.normal(0.0, 0.005, n_floor)
            floor_pts = np.column_stack([floor_x, floor_y, floor_z])

            # Ceiling at Y = 2.8m (Should be classified as ceiling, NOT table)
            n_ceil = 1500
            ceil_x = rng.uniform(-2.5, 2.5, n_ceil)
            ceil_z = rng.uniform(-2.5, 2.5, n_ceil)
            ceil_y = rng.normal(2.8, 0.005, n_ceil)
            ceil_pts = np.column_stack([ceil_x, ceil_y, ceil_z])

            # Table at Y = 0.75m
            n_table = 800
            table_x = rng.uniform(-0.6, 0.6, n_table)
            table_z = rng.uniform(-0.4, 0.4, n_table)
            table_y = rng.normal(0.75, 0.005, n_table)
            table_pts = np.column_stack([table_x, table_y, table_z])

            # Wall at Z = -2.0m
            n_wall = 1200
            wall_x = rng.uniform(-2.0, 2.0, n_wall)
            wall_y = rng.uniform(0.0, 2.2, n_wall)
            wall_z = rng.normal(-2.0, 0.005, n_wall)
            wall_pts = np.column_stack([wall_x, wall_y, wall_z])

            all_pts = np.vstack([floor_pts, ceil_pts, table_pts, wall_pts])
            cloud = trimesh.PointCloud(vertices=all_pts)
            cloud.export(str(ply_path))

            result = detect_architectural_planes(
                ply_path=ply_path,
                distance_threshold=0.04,
                max_planes=6,
                min_inliers=150,
            )

            # Sàn và Bàn và Trần nhà phải được tách biệt rõ ràng
            self.assertIsNotNone(result["floor"])
            self.assertAlmostEqual(result["floor"]["mean_y"], 0.0, delta=0.1)

            # Table planes must only include the table at ~0.75m, NOT the ceiling at 2.8m
            self.assertEqual(len(result["tables"]), 1)
            self.assertAlmostEqual(result["tables"][0]["mean_y"], 0.75, delta=0.1)

            # Ceiling must be captured under ceilings
            self.assertIn("ceilings", result)
            self.assertEqual(len(result["ceilings"]), 1)
            self.assertAlmostEqual(result["ceilings"][0]["mean_y"], 2.8, delta=0.1)

    def test_compute_oriented_wall_geometry_clamping(self):
        # Case 1: Wall points starting near floor (min_y = 0.10m, floor_y = 0.0m) -> Clamped to floor_y
        pts_near = np.array([
            [-1.0, 0.10, -2.0],
            [1.0, 0.10, -2.0],
            [0.0, 2.0, -2.0]
        ], dtype=np.float64)
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        obb_near = _compute_oriented_wall_geometry(pts_near, normal, floor_y=0.0)
        # base_y should be snapped to 0.0, center_y = 0.0 + 2.0/2.0 = 1.0
        self.assertAlmostEqual(obb_near["height"], 2.0, places=4)
        self.assertAlmostEqual(obb_near["center"][1], 1.0, places=4)

        # Case 2: Hanging wall divider (min_y = 0.50m, floor_y = 0.0m) -> NOT clamped to floor_y
        pts_hanging = np.array([
            [-1.0, 0.50, -2.0],
            [1.0, 0.50, -2.0],
            [0.0, 2.0, -2.0]
        ], dtype=np.float64)
        obb_hanging = _compute_oriented_wall_geometry(pts_hanging, normal, floor_y=0.0)
        # base_y remains 0.50, height = 2.0 - 0.50 = 1.50
        self.assertAlmostEqual(obb_hanging["height"], 1.50, places=4)
        self.assertAlmostEqual(obb_hanging["center"][1], 1.25, places=4)

    def test_extract_room_background_pointcloud(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            world_ply = Path(tmp_dir) / "world.ply"
            objs_dir = Path(tmp_dir) / "objects"
            objs_dir.mkdir()
            out_pcd = Path(tmp_dir) / "room_bg.ply"

            # World points: 100 room points around (0, 0, 0) + 50 chair points around (2, 2, 2)
            rng = np.random.default_rng(42)
            room_pts = rng.uniform(-1.0, 1.0, size=(100, 3))
            chair_pts = rng.uniform(1.9, 2.1, size=(50, 3))
            all_pts = np.vstack([room_pts, chair_pts])
            trimesh.PointCloud(vertices=all_pts).export(str(world_ply))

            # Save object point cloud
            chair_pcd = objs_dir / "obj_0_chair_pointcloud.ply"
            trimesh.PointCloud(vertices=chair_pts).export(str(chair_pcd))

            bg_pts, _, p_out = extract_room_background_pointcloud(
                world_ply_path=world_ply,
                objects_dir=objs_dir,
                out_pcd_path=out_pcd,
                subtraction_radius=0.05
            )

            self.assertTrue(p_out.exists())
            self.assertEqual(len(bg_pts), 100)
            # Ensure none of the (2, 2, 2) chair points are in the room background
            self.assertTrue(np.all(bg_pts < 1.5))

    def test_reconstruct_room_background_mesh(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_mesh = Path(tmp_dir) / "room_mesh.ply"
            rng = np.random.default_rng(42)
            # Create synthetic sphere points for meshing
            u = rng.uniform(0, 2 * np.pi, 300)
            v = rng.uniform(0, np.pi, 300)
            x = 1.0 * np.sin(v) * np.cos(u)
            y = 1.0 * np.sin(v) * np.sin(u)
            z = 1.0 * np.cos(v)
            pts = np.column_stack([x, y, z])
            cols = rng.integers(50, 200, size=(300, 3), dtype=np.uint8)

            mesh = reconstruct_room_background_mesh(
                room_pts=pts,
                room_cols=cols,
                out_mesh_path=out_mesh,
                method="poisson",
                depth=7,
            )
            self.assertIsNotNone(mesh)
            self.assertTrue(out_mesh.exists())
            self.assertGreater(len(mesh.vertices), 0)

    def test_build_room_background_orchestration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            world_ply = Path(tmp_dir) / "world.ply"
            objs_dir = Path(tmp_dir) / "objects"
            objs_dir.mkdir()
            out_pcd = Path(tmp_dir) / "room_bg.ply"
            out_mesh = Path(tmp_dir) / "room_mesh.ply"

            # World points (sphere)
            rng = np.random.default_rng(42)
            u = rng.uniform(0, 2 * np.pi, 200)
            v = rng.uniform(0, np.pi, 200)
            x = 1.0 * np.sin(v) * np.cos(u)
            y = 1.0 * np.sin(v) * np.sin(u)
            z = 1.0 * np.cos(v)
            room_pts = np.column_stack([x, y, z])
            obj_pts = rng.uniform(2.0, 2.5, size=(30, 3))
            all_pts = np.vstack([room_pts, obj_pts])
            trimesh.PointCloud(vertices=all_pts).export(str(world_ply))

            # Object point cloud
            obj_pcd = objs_dir / "obj_1_chair_pointcloud.ply"
            trimesh.PointCloud(vertices=obj_pts).export(str(obj_pcd))

            res = build_room_background(
                world_ply_path=world_ply,
                objects_dir=objs_dir,
                out_pcd_path=out_pcd,
                out_mesh_path=out_mesh,
                depth=6,
            )
            self.assertTrue(out_pcd.exists())
            self.assertTrue(out_mesh.exists())
            self.assertEqual(res["point_count"], 200)


class TestSpatialPhase2ObjectEstimator(unittest.TestCase):

    def test_backproject_mask_to_3d(self):
        mask_2d = np.zeros((50, 50), dtype=np.uint8)
        mask_2d[10:30, 10:30] = 255
        depth_map = np.ones((50, 50), dtype=np.float64) * 2.0
        K = np.array([[50.0, 0.0, 25.0], [0.0, 50.0, 25.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)

        pts_w, _ = backproject_mask_to_3d(mask_2d, depth_map, K, c2w, foreground_margin=0.0)
        self.assertEqual(len(pts_w), 20 * 20)
        self.assertAlmostEqual(pts_w[:, 2].mean(), 2.0, places=4)

    def test_backproject_foreground_depth_gating(self):
        # Object mask at depth 1.5m, with background wall bleed at depth 4.0m
        mask_2d = np.ones((40, 40), dtype=np.uint8)
        depth_map = np.ones((40, 40), dtype=np.float64) * 1.5
        depth_map[25:, :] = 4.0  # Background bleed
        K = np.array([[50.0, 0.0, 20.0], [0.0, 50.0, 20.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)

        pts_w, _ = backproject_mask_to_3d(mask_2d, depth_map, K, c2w, foreground_margin=0.35)
        # Background depth 4.0m should be pruned because median is ~1.5m
        self.assertGreater(len(pts_w), 0)
        self.assertLess(len(pts_w), 40 * 40)
        self.assertTrue(np.all(pts_w[:, 2] <= 2.0))

    def test_dbscan_filtering(self):
        rng = np.random.default_rng(42)
        dense = rng.uniform(-0.05, 0.05, size=(100, 3))
        noise = np.array([[10.0, 10.0, 10.0]], dtype=np.float64)
        pts = np.vstack([dense, noise])

        clean_pts, _ = filter_object_pointcloud_dbscan(pts, eps=0.1, min_samples=5, min_cluster_size=20)
        self.assertEqual(len(clean_pts), 100)

    def test_process_object_detections_with_3x4_extrinsics_and_polygon_mask(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            npz_path = Path(tmp_dir) / "raw_depths.npz"
            det_path = Path(tmp_dir) / "detections.json"
            out_dir = Path(tmp_dir) / "objects"

            depth_0 = np.ones((60, 60), dtype=np.float32) * 1.8
            ixt_0 = np.array([[50.0, 0.0, 30.0], [0.0, 50.0, 30.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            ext_0 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
            np.savez(str(npz_path), depth_0=depth_0, ixt_0=ixt_0, ext_0=ext_0)

            det_data = {
                "obj_1": {
                    "label": "chair",
                    "class_id": 56,
                    "associated_views": [
                        {
                            "frame_index": 0,
                            "bbox": [15, 15, 45, 45],
                            "mask": [[15, 15], [45, 15], [45, 45], [15, 45]],
                            "score": 0.95,
                        }
                    ]
                }
            }
            with open(det_path, "w", encoding="utf-8") as f:
                json.dump(det_data, f)

            result = process_object_detections(
                detections_path=det_path,
                raw_depths_path=npz_path,
                out_dir=out_dir
            )
            self.assertIn("obj_1", result)
            self.assertGreater(result["obj_1"]["point_count"], 0)
            self.assertTrue(Path(result["obj_1"]["mesh_path"]).exists())

    def test_process_object_detections_with_flat_polygon_mask(self):
        # Test flat mask list: [x1, y1, x2, y2, x3, y3, x4, y4]
        with tempfile.TemporaryDirectory() as tmp_dir:
            npz_path = Path(tmp_dir) / "raw_depths.npz"
            det_path = Path(tmp_dir) / "detections.json"
            out_dir = Path(tmp_dir) / "objects"

            depth_0 = np.ones((60, 60), dtype=np.float32) * 1.8
            ixt_0 = np.array([[50.0, 0.0, 30.0], [0.0, 50.0, 30.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            ext_0 = np.eye(4, dtype=np.float64)
            np.savez(str(npz_path), depth_0=depth_0, ixt_0=ixt_0, ext_0=ext_0)

            det_data = {
                "obj_flat": {
                    "label": "table",
                    "associated_views": [
                        {
                            "frame_index": 0,
                            "bbox": [10, 10, 50, 50],
                            "mask": [10, 10, 50, 10, 50, 50, 10, 50],  # Flat list format
                        }
                    ]
                }
            }
            with open(det_path, "w", encoding="utf-8") as f:
                json.dump(det_data, f)

            result = process_object_detections(
                detections_path=det_path,
                raw_depths_path=npz_path,
                out_dir=out_dir
            )
            self.assertIn("obj_flat", result)
            self.assertGreater(result["obj_flat"]["point_count"], 0)
            self.assertTrue(Path(result["obj_flat"]["mesh_path"]).exists())

    def test_dbscan_dominant_cluster_adaptive_merge(self):
        rng = np.random.default_rng(42)
        # Cluster 1: Primary body (e.g. sofa seat), 150 points in a dense box around (0, 0, 0)
        c1 = rng.uniform(-0.15, 0.15, size=(150, 3))
        # Cluster 2: Armrest adjacent to sofa seat, 50 points in a dense box around (0.35, 0.05, 0)
        c2 = rng.uniform(-0.05, 0.05, size=(50, 3)) + np.array([0.35, 0.05, 0.0])
        # Cluster 3: Isolated noise far away (3.0, 3.0, 3.0)
        noise = rng.uniform(-0.05, 0.05, size=(40, 3)) + np.array([3.0, 3.0, 3.0])

        pts = np.vstack([c1, c2, noise])
        clean_pts, _ = filter_object_pointcloud_dbscan(pts, eps=0.10, min_samples=5, min_cluster_size=20)
        # Both c1 (150) and c2 (50) should be retained, while noise (40) is pruned
        self.assertGreaterEqual(len(clean_pts), 180)
        self.assertTrue(np.all(np.abs(clean_pts) < 1.5))

    def test_missing_detections_raises_filenotfound(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_det = Path(tmp_dir) / "non_existent_det.json"
            npz_path = Path(tmp_dir) / "raw_depths.npz"
            np.savez(str(npz_path), dummy=np.array([1]))

            with self.assertRaises(FileNotFoundError):
                process_object_detections(detections_path=missing_det, raw_depths_path=npz_path)

    def test_extract_object_pointcloud_from_world_pcd(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pcd_path = Path(tmp_dir) / "world_pointcloud.ply"
            npz_path = Path(tmp_dir) / "raw_depths.npz"
            det_path = Path(tmp_dir) / "detections.json"
            out_dir = Path(tmp_dir) / "objects"

            # Create synthetic world point cloud: Chair points at (0.0, 0.0, -2.0) and background wall at (0.0, 0.0, -5.0)
            rng = np.random.default_rng(42)
            chair_pts = rng.uniform(-0.1, 0.1, size=(200, 3)) + np.array([0.0, 0.0, -2.0])
            wall_pts = rng.uniform(-1.0, 1.0, size=(300, 3)) + np.array([0.0, 0.0, -5.0])
            all_pts = np.vstack([chair_pts, wall_pts])
            all_cols = np.tile([100, 150, 200], (len(all_pts), 1)).astype(np.uint8)

            cloud = trimesh.PointCloud(vertices=all_pts, colors=all_cols)
            cloud.export(str(pcd_path))

            # Camera pointing at the chair
            H, W = 60, 60
            depth_0 = np.ones((H, W), dtype=np.float32) * 2.0
            ixt_0 = np.array([[50.0, 0.0, 30.0], [0.0, 50.0, 30.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            ext_0 = np.eye(4, dtype=np.float64)
            np.savez(str(npz_path), depth_0=depth_0, ixt_0=ixt_0, ext_0=ext_0)

            det_data = {
                "obj_chair": {
                    "label": "chair",
                    "associated_views": [
                        {
                            "frame_index": 0,
                            "bbox": [20, 20, 40, 40],
                            "mask": [[20, 20], [40, 20], [40, 40], [20, 40]],
                        }
                    ]
                }
            }
            with open(det_path, "w", encoding="utf-8") as f:
                json.dump(det_data, f)

            result = process_object_detections(
                detections_path=det_path,
                raw_depths_path=npz_path,
                world_pcd_path=pcd_path,
                out_dir=out_dir,
            )

            self.assertIn("obj_chair", result)
            self.assertGreater(result["obj_chair"]["point_count"], 50)
            self.assertTrue(Path(result["obj_chair"]["mesh_path"]).exists())
            self.assertIsNotNone(result["obj_chair"]["pcd_path"])
            self.assertTrue(Path(result["obj_chair"]["pcd_path"]).exists())
            # Ensure background wall points at Z=-5.0m were pruned and only chair points near Z=-2.0m kept
            bounds_min = result["obj_chair"]["bounds_min"]
            self.assertGreater(bounds_min[2], -3.0)

    def test_process_object_detections_with_trimesh_scene(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pcd_path = Path(tmp_dir) / "world_pointcloud.ply"
            npz_path = Path(tmp_dir) / "raw_depths.npz"
            det_path = Path(tmp_dir) / "detections.json"
            out_dir = Path(tmp_dir) / "objects"

            # Create a point cloud for world_pointcloud.ply
            rng = np.random.default_rng(42)
            pts1 = rng.uniform(-0.1, 0.1, size=(200, 3)) + np.array([0.0, 0.0, -2.0])
            pc1 = trimesh.PointCloud(vertices=pts1)
            pc1.export(str(pcd_path))

            H, W = 60, 60
            depth_0 = np.ones((H, W), dtype=np.float32) * 2.0
            ixt_0 = np.array([[50.0, 0.0, 30.0], [0.0, 50.0, 30.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            ext_0 = np.eye(4, dtype=np.float64)
            np.savez(str(npz_path), depth_0=depth_0, ixt_0=ixt_0, ext_0=ext_0)

            det_data = {
                "obj_scene": {
                    "label": "sofa",
                    "associated_views": [
                        {
                            "frame_index": 0,
                            "bbox": [20, 20, 40, 40],
                        }
                    ]
                }
            }
            with open(det_path, "w", encoding="utf-8") as f:
                json.dump(det_data, f)

            result = process_object_detections(
                detections_path=det_path,
                raw_depths_path=npz_path,
                world_pcd_path=pcd_path,
                out_dir=out_dir,
            )

            self.assertIn("obj_scene", result)
            self.assertGreater(result["obj_scene"]["point_count"], 50)
            self.assertTrue(Path(result["obj_scene"]["mesh_path"]).exists())

    def test_process_object_detections_disabled_dbscan(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            npz_path = Path(tmp_dir) / "raw_depths.npz"
            det_path = Path(tmp_dir) / "detections.json"
            out_dir = Path(tmp_dir) / "objects"

            depth_0 = np.ones((60, 60), dtype=np.float32) * 1.8
            ixt_0 = np.array([[50.0, 0.0, 30.0], [0.0, 50.0, 30.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            ext_0 = np.eye(4, dtype=np.float64)
            np.savez(str(npz_path), depth_0=depth_0, ixt_0=ixt_0, ext_0=ext_0)

            det_data = {
                "obj_nodbscan": {
                    "label": "chair",
                    "associated_views": [
                        {"frame_index": 0, "bbox": [20, 20, 40, 40]}
                    ]
                }
            }
            with open(det_path, "w", encoding="utf-8") as f:
                json.dump(det_data, f)

            orig_dbscan = config.OBJECT_ENABLE_DBSCAN
            try:
                config.OBJECT_ENABLE_DBSCAN = False
                result = process_object_detections(
                    detections_path=det_path,
                    raw_depths_path=npz_path,
                    out_dir=out_dir
                )
                self.assertIn("obj_nodbscan", result)
                self.assertGreater(result["obj_nodbscan"]["point_count"], 0)
            finally:
                config.OBJECT_ENABLE_DBSCAN = orig_dbscan

    def test_bpa_reconstruct_object_mesh(self):
        # Test Ball Pivoting Algorithm on a 3D point cloud
        rng = np.random.default_rng(42)
        # Create a sphere surface point cloud
        u = rng.uniform(0, 2 * np.pi, 200)
        v = rng.uniform(0, np.pi, 200)
        x = 0.5 * np.sin(v) * np.cos(u)
        y = 0.5 * np.sin(v) * np.sin(u)
        z = 0.5 * np.cos(v)
        pts = np.column_stack([x, y, z])

        mesh = reconstruct_object_mesh(pts, method="bpa")
        self.assertIsNotNone(mesh)
        self.assertGreater(len(mesh.vertices), 0)
        self.assertGreater(len(mesh.faces if hasattr(mesh, "faces") else mesh.triangles), 0)

    def test_plane_subtraction_excludes_floor_points(self):
        # Create points representing a chair with points on floor
        rng = np.random.default_rng(42)
        chair_pts = rng.uniform(-0.2, 0.2, size=(50, 3))
        chair_pts[:, 1] = rng.uniform(0.05, 0.8, size=50) # Chair body at Y in [0.05, 0.8]

        floor_pts = rng.uniform(-0.2, 0.2, size=(30, 3))
        floor_pts[:, 1] = rng.uniform(-0.01, 0.005, size=30) # Floor points at Y <= 0.005

        all_pts = np.vstack([chair_pts, floor_pts])

        plane_data = {
            "floor": {"mean_y": 0.0},
            "tables": []
        }

        filtered_pts, _ = _filter_plane_inliers(all_pts, None, "chair", plane_data, margin=0.010)
        self.assertEqual(len(filtered_pts), len(chair_pts))
        self.assertTrue(np.all(filtered_pts[:, 1] > 0.010))

    def test_multi_view_consensus_filtering(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pcd_path = Path(tmp_dir) / "world_pointcloud.ply"
            npz_path = Path(tmp_dir) / "raw_depths.npz"
            det_path = Path(tmp_dir) / "detections.json"
            out_dir = Path(tmp_dir) / "objects"

            # 150 points at chair position Z=-2.0m
            rng = np.random.default_rng(42)
            chair_pts = rng.uniform(-0.1, 0.1, size=(150, 3)) + np.array([0.0, 0.0, -2.0])
            pc = trimesh.PointCloud(vertices=chair_pts)
            pc.export(str(pcd_path))

            H, W = 60, 60
            depth_0 = np.ones((H, W), dtype=np.float32) * 2.0
            depth_1 = np.ones((H, W), dtype=np.float32) * 2.0
            depth_2 = np.ones((H, W), dtype=np.float32) * 2.0
            ixt = np.array([[50.0, 0.0, 30.0], [0.0, 50.0, 30.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            ext = np.eye(4, dtype=np.float64)

            np.savez(str(npz_path), depth_0=depth_0, depth_1=depth_1, depth_2=depth_2,
                     ixt_0=ixt, ixt_1=ixt, ixt_2=ixt, ext_0=ext, ext_1=ext, ext_2=ext)

            # Object observed in 3 frames (Majority consensus test)
            det_data = {
                "obj_mv": {
                    "label": "chair",
                    "associated_views": [
                        {"frame_index": 0, "bbox": [20, 20, 40, 40]},
                        {"frame_index": 1, "bbox": [20, 20, 40, 40]},
                        {"frame_index": 2, "bbox": [20, 20, 40, 40]},
                    ]
                }
            }
            with open(det_path, "w", encoding="utf-8") as f:
                json.dump(det_data, f)

            result = process_object_detections(
                detections_path=det_path,
                raw_depths_path=npz_path,
                world_pcd_path=pcd_path,
                out_dir=out_dir
            )
            self.assertIn("obj_mv", result)
            self.assertGreater(result["obj_mv"]["point_count"], 30)

    def test_missing_raw_depths_raises_filenotfound(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            det_path = Path(tmp_dir) / "detections.json"
            with open(det_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
            missing_npz = Path(tmp_dir) / "non_existent.npz"

            with self.assertRaises(FileNotFoundError):
                process_object_detections(detections_path=det_path, raw_depths_path=missing_npz)

    def test_object_estimator_world_pcd_only_extraction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pts = np.random.uniform(-0.2, 0.2, (200, 3))
            pts[:, 2] += 1.5
            pcd = trimesh.PointCloud(vertices=pts)
            pcd_path = Path(tmp_dir) / "world_pointcloud.ply"
            pcd.export(str(pcd_path))

            det_data = {
                "obj_chair": {
                    "label": "chair",
                    "associated_views": [{"frame_index": 0, "bbox": [0, 0, 640, 480], "score": 0.95}],
                }
            }
            det_path = Path(tmp_dir) / "detections.json"
            with open(det_path, "w", encoding="utf-8") as f:
                json.dump(det_data, f)

            meta_data = {
                "frames": [
                    {
                        "index": 0,
                        "pose_matrix": np.eye(4).tolist(),
                        "fl_x": 500.0,
                        "fl_y": 500.0,
                        "cx": 320.0,
                        "cy": 240.0,
                        "w": 640,
                        "h": 480,
                    }
                ]
            }
            meta_path = Path(tmp_dir) / "ar_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f)

            out_dir = Path(tmp_dir) / "objects"
            res = process_object_detections(
                detections_path=det_path,
                raw_depths_path=None,
                ar_metadata_path=meta_path,
                world_pcd_path=pcd_path,
                out_dir=out_dir,
            )
            self.assertIn("obj_chair", res)
            self.assertGreater(res["obj_chair"]["point_count"], 0)


class TestSpatialPhase2AObjectExtractor(unittest.TestCase):

    def test_extract_strictly_from_world_pointcloud_zero_synthetic_points(self):
        """
        Verify that 100% of extracted object points are an exact subset of world_pointcloud.ply
        with identical coordinates and original RGB colors (0 synthetic points created).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            pcd_path = Path(tmp_dir) / "world_pointcloud.ply"
            det_path = Path(tmp_dir) / "detections.json"
            out_dir = Path(tmp_dir) / "objects"

            rng = np.random.default_rng(123)
            # 300 unique points in world point cloud
            pts_world = rng.uniform(-1.0, 1.0, size=(300, 3)).astype(np.float64)
            # Distinct RGB colors
            cols_world = rng.integers(10, 240, size=(300, 3), dtype=np.uint8)

            cloud = trimesh.PointCloud(vertices=pts_world, colors=cols_world)
            cloud.export(str(pcd_path))

            # Object detection with bounding box covering part of the scene
            det_data = {
                "obj_table": {
                    "label": "table",
                    "associated_views": [
                        {
                            "frame_index": 0,
                            "bbox": [0, 0, 640, 480],
                            "score": 0.98,
                        }
                    ]
                }
            }
            with open(det_path, "w", encoding="utf-8") as f:
                json.dump(det_data, f)

            meta_data = {
                "frames": [
                    {
                        "index": 0,
                        "pose_matrix": np.eye(4).tolist(),
                        "fl_x": 500.0,
                        "fl_y": 500.0,
                        "cx": 320.0,
                        "cy": 240.0,
                        "w": 640,
                        "h": 480,
                    }
                ]
            }
            meta_path = Path(tmp_dir) / "ar_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f)

            manifest = extract_object_pointclouds(
                detections_path=det_path,
                ar_metadata_path=meta_path,
                world_pcd_path=pcd_path,
                out_dir=out_dir,
                enable_dbscan=False,
            )

            self.assertIn("obj_table", manifest)
            obj_pcd_file = Path(manifest["obj_table"]["pcd_path"])
            self.assertTrue(obj_pcd_file.exists())

            extracted_cloud = trimesh.load(str(obj_pcd_file))
            ext_pts = np.asarray(extracted_cloud.vertices)
            ext_cols = np.asarray(extracted_cloud.colors)[:, :3]

            self.assertGreater(len(ext_pts), 0)

            # Strict verification: every single extracted point must exist identically in pts_world
            for i, pt in enumerate(ext_pts):
                dists = np.linalg.norm(pts_world - pt, axis=1)
                min_idx = np.argmin(dists)
                self.assertAlmostEqual(dists[min_idx], 0.0, places=5, msg="Extracted point does not exist in source point cloud!")
                # Verify exact RGB colors match
                np.testing.assert_array_equal(ext_cols[i], cols_world[min_idx], err_msg="Point color was altered or interpolated!")

    def test_depth_consistency_gating_rejects_background_bleed(self):
        """
        Verify that background surface points behind the object are cleanly rejected
        by depth-consistency checking against the frame's depth map.
        """
        H, W = 60, 60
        K = np.array([[50.0, 0.0, 30.0], [0.0, 50.0, 30.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)

        # 1 foreground point at Z=1.5m and 1 background point at Z=4.0m projecting to center pixel (30, 30)
        fg_pt = np.array([0.0, 0.0, 1.5])
        bg_pt = np.array([0.0, 0.0, 4.0])
        world_pts = np.vstack([fg_pt, bg_pt])

        mask_2d = np.zeros((H, W), dtype=np.uint8)
        mask_2d[20:40, 20:40] = 255  # Mask covering center

        # Depth map observed foreground surface at 1.5m
        depth_map = np.ones((H, W), dtype=np.float64) * 1.5

        sel_pts, _, sel_idx = extract_object_points_from_world_pcd_view(
            world_pts=world_pts,
            world_cols=None,
            mask_2d=mask_2d,
            K=K,
            c2w=c2w,
            depth_map=depth_map,
            depth_tolerance=0.10,
        )

        # Only the foreground point (index 0) at Z=1.5m should survive
        self.assertEqual(len(sel_idx), 1)
        self.assertEqual(sel_idx[0], 0)
        self.assertAlmostEqual(sel_pts[0, 2], 1.5, places=4)

    def test_extractor_class_interface(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pcd_path = Path(tmp_dir) / "world_pointcloud.ply"
            det_path = Path(tmp_dir) / "detections.json"
            out_dir = Path(tmp_dir) / "objects"

            rng = np.random.default_rng(42)
            pts = rng.uniform(-0.2, 0.2, size=(100, 3)) + np.array([0.0, 0.0, 1.5])
            trimesh.PointCloud(vertices=pts).export(str(pcd_path))

            det_data = {
                "obj_lamp": {
                    "label": "lamp",
                    "associated_views": [{"frame_index": 0, "bbox": [0, 0, 640, 480]}]
                }
            }
            with open(det_path, "w", encoding="utf-8") as f:
                json.dump(det_data, f)

            extractor = ObjectExtractor(
                detections_path=det_path,
                world_pcd_path=pcd_path,
                out_dir=out_dir,
            )
            res = extractor.run()
            self.assertIn("obj_lamp", res)
            self.assertTrue((out_dir / "extracted_objects_manifest.json").exists())

    def test_color_consistency_filter_prunes_wrong_color_points(self):
        """
        Verify that CIELAB color consistency gating cleanly removes wrong-colored background/floor points
        that fall inside the 2D mask while retaining genuine object points.
        """
        # Red object points
        red_pts = np.array([
            [0.0, 0.0, 1.5],
            [0.05, 0.05, 1.5],
            [-0.05, 0.05, 1.5],
            [0.0, -0.05, 1.5],
            [0.05, -0.05, 1.5],
        ], dtype=np.float64)
        red_cols = np.array([[220, 30, 30]] * len(red_pts), dtype=np.uint8)

        # Grey floor/background points in same 2D projection
        grey_pts = np.array([
            [0.02, 0.02, 1.5],
            [-0.02, -0.02, 1.5],
            [0.03, -0.03, 1.5],
        ], dtype=np.float64)
        grey_cols = np.array([[190, 190, 190]] * len(grey_pts), dtype=np.uint8)

        world_pts = np.vstack([red_pts, grey_pts])
        world_cols = np.vstack([red_cols, grey_cols])

        mask_2d = np.ones((100, 100), dtype=np.uint8)
        K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)

        # 2D frame image is red in object mask
        rgb_frame = np.full((100, 100, 3), [220, 30, 30], dtype=np.uint8)

        # Gated extraction with color filtering enabled
        sel_pts, sel_cols, m_idx = extract_object_points_from_world_pcd_view(
            world_pts,
            world_cols,
            mask_2d,
            K,
            c2w,
            rgb_frame=rgb_frame,
            enable_color_filter=True,
            color_delta_e_max=35.0,
        )

        # Red points must be retained (5 points)
        self.assertEqual(len(sel_pts), len(red_pts))
        # All selected points must be red
        self.assertTrue(np.all(sel_cols[:, 0] > 180))
        self.assertTrue(np.all(sel_cols[:, 1] < 60))

    def test_color_filter_graceful_fallback_uncolored(self):
        """
        Verify that extraction gracefully falls back to geometric extraction when colors are absent.
        """
        pts = np.array([[0.0, 0.0, 1.5], [0.05, 0.05, 1.5]], dtype=np.float64)
        mask_2d = np.ones((100, 100), dtype=np.uint8)
        K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)

        # When world_cols is None
        sel_pts, sel_cols, m_idx = extract_object_points_from_world_pcd_view(
            pts, None, mask_2d, K, c2w, enable_color_filter=True
        )
        self.assertEqual(len(sel_pts), 2)
        self.assertIsNone(sel_cols)

    def test_6d_cielab_dbscan_separates_contact_plane(self):
        """
        Verify that 6D (XYZ + CIELAB) DBSCAN cleanly separates touching white floor points
        from a dark chair leg cluster despite 0 spatial distance gap.
        """
        rng = np.random.default_rng(42)
        # 80 points of black chair leg
        chair_pts = rng.uniform(-0.04, 0.04, size=(80, 3))
        chair_cols = np.full((80, 3), [25, 25, 25], dtype=np.uint8)

        # 15 points of white floor touching the bottom of the chair leg (spatial dist ~ 0.01m)
        floor_pts = rng.uniform(-0.04, 0.04, size=(15, 3))
        floor_pts[:, 1] = chair_pts[:, 1].min() - 0.005 # touching base
        floor_cols = np.full((15, 3), [230, 230, 230], dtype=np.uint8)

        all_pts = np.vstack([chair_pts, floor_pts])
        all_cols = np.vstack([chair_cols, floor_cols])

        clean_pts, clean_cols = filter_object_pointcloud_dbscan(
            all_pts, colors=all_cols, eps=0.06, min_samples=4, min_cluster_size=20
        )

        # Chair cluster must be isolated and retained (80 pts), white floor (15 pts) pruned
        self.assertEqual(len(clean_pts), 80)
        self.assertTrue(np.all(clean_cols[:, 0] < 50))

    def test_mask3d_structural_plane_and_multi_object_segmentation(self):
        """
        Verify that Mask3D with structural plane awareness cleanly separates
        floor-connected furniture (chair) and tabletop objects (laptop on table) into distinct proposals.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            pcd_path = Path(tmp_dir) / "world_pointcloud.ply"
            npz_path = Path(tmp_dir) / "raw_depths.npz"
            plane_path = Path(tmp_dir) / "detected_planes.json"
            out_dir = Path(tmp_dir) / "objects"

            rng = np.random.default_rng(42)

            # 1. Floor points at Y=0.0m (500 pts)
            floor_x = rng.uniform(-2.0, 2.0, 500)
            floor_z = rng.uniform(-2.0, 2.0, 500)
            floor_y = np.zeros(500)
            pts_floor = np.column_stack([floor_x, floor_y, floor_z])

            # 2. Chair on floor around (1.0, Y in [0.05, 0.8], 1.0) (150 pts)
            chair_x = rng.uniform(0.8, 1.2, 150)
            chair_y = rng.uniform(0.05, 0.8, 150)
            chair_z = rng.uniform(0.8, 1.2, 150)
            pts_chair = np.column_stack([chair_x, chair_y, chair_z])

            # 3. Table at (-1.0, Y=0.75, -1.0) (150 pts)
            tbl_x = rng.uniform(-1.4, -0.6, 150)
            tbl_y = rng.uniform(0.05, 0.75, 150)
            tbl_z = rng.uniform(-1.4, -0.6, 150)
            pts_table = np.column_stack([tbl_x, tbl_y, tbl_z])

            # 4. Laptop on top of table at (-1.0, Y in [0.77, 0.95], -1.0) (80 pts)
            lp_x = rng.uniform(-1.15, -0.85, 80)
            lp_y = rng.uniform(0.77, 0.95, 80)
            lp_z = rng.uniform(-1.15, -0.85, 80)
            pts_laptop = np.column_stack([lp_x, lp_y, lp_z])

            all_pts = np.vstack([pts_floor, pts_chair, pts_table, pts_laptop])
            all_cols = rng.integers(50, 220, size=(len(all_pts), 3), dtype=np.uint8)

            trimesh.PointCloud(vertices=all_pts, colors=all_cols).export(str(pcd_path))

            # Plane metadata
            plane_data = {
                "floor": {"mean_y": 0.0, "inliers": 500},
                "tables": [
                    {
                        "mean_y": 0.75,
                        "min_bound": [-1.4, 0.75, -1.4],
                        "max_bound": [-0.6, 0.75, -0.6],
                    }
                ],
                "walls": [],
            }
            with open(plane_path, "w", encoding="utf-8") as f:
                json.dump(plane_data, f)

            # Synthetic raw depths
            H, W = 60, 60
            rgb_0 = np.full((H, W, 3), 120, dtype=np.uint8)
            depth_0 = np.ones((H, W), dtype=np.float32) * 2.0
            ixt_0 = np.array([[50.0, 0.0, 30.0], [0.0, 50.0, 30.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            ext_0 = np.eye(4, dtype=np.float64)
            np.savez(str(npz_path), rgb_0=rgb_0, depth_0=depth_0, ixt_0=ixt_0, ext_0=ext_0)

            # Run Autonomous Mask3D extraction with plane metadata
            results = extract_object_pointclouds(
                world_pcd_path=pcd_path,
                raw_depths_path=npz_path,
                plane_data_path=plane_path,
                out_dir=out_dir,
                text_queries=["chair", "table", "laptop", "monitor", "sofa"],
            )

            # Slicing should successfully extract discrete objects without merging into one giant floor mesh
            self.assertGreaterEqual(len(results), 2)
            manifest_file = out_dir / "extracted_objects_manifest.json"
            self.assertTrue(manifest_file.exists())
            with open(manifest_file, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
            self.assertEqual(len(manifest), len(results))
            # Check that each object has a saved point cloud ply
            for obj_k, obj_v in manifest.items():
                self.assertTrue(Path(obj_v["pcd_path"]).exists())
                self.assertGreater(obj_v["point_count"], 20)




class TestSpatialPhase2BObjectMesher(unittest.TestCase):


    def test_reconstruct_object_meshes_from_pointclouds(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            objs_dir = Path(tmp_dir) / "objects"
            objs_dir.mkdir()

            rng = np.random.default_rng(42)
            # Create synthetic point cloud on sphere
            u = rng.uniform(0, 2 * np.pi, 250)
            v = rng.uniform(0, np.pi, 250)
            pts = np.column_stack([0.3 * np.sin(v) * np.cos(u), 0.3 * np.sin(v) * np.sin(u), 0.3 * np.cos(v)])
            cols = rng.integers(50, 220, size=(250, 3), dtype=np.uint8)

            pcd_path = objs_dir / "obj_0_sphere_pointcloud.ply"
            trimesh.PointCloud(vertices=pts, colors=cols).export(str(pcd_path))

            manifest_path = objs_dir / "extracted_objects_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({
                    "obj_0": {
                        "label": "sphere",
                        "pcd_path": str(pcd_path),
                        "point_count": len(pts),
                    }
                }, f)

            mesher = ObjectMesher(objects_dir=objs_dir, manifest_path=manifest_path, method="poisson", depth=6)
            res = mesher.run()

            self.assertIn("obj_0", res)
            out_mesh = Path(res["obj_0"]["mesh_path"])
            self.assertTrue(out_mesh.exists())
            self.assertGreater(res["obj_0"]["vertex_count"], 0)
            self.assertGreater(res["obj_0"]["face_count"], 0)

    def test_intermediate_ai_pointcloud_completion_workflow(self):
        """
        Simulate the two-stage decoupled pipeline where an intermediate AI Point Cloud Completion
        step augments the extracted point cloud before passing it to ObjectMesher.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            objs_dir = Path(tmp_dir) / "objects"
            objs_dir.mkdir()

            # Step 1: Raw extracted sparse point cloud (e.g. 50 points)
            rng = np.random.default_rng(42)
            sparse_pts = rng.uniform(-0.2, 0.2, size=(50, 3))
            raw_pcd = objs_dir / "obj_chair_pointcloud.ply"
            trimesh.PointCloud(vertices=sparse_pts).export(str(raw_pcd))

            # Step 2: Intermediate AI Completion Step (enriches to 300 points)
            dense_completed_pts = rng.uniform(-0.2, 0.2, size=(300, 3))
            completed_pcd = objs_dir / "obj_chair_pointcloud.ply"
            trimesh.PointCloud(vertices=dense_completed_pts).export(str(completed_pcd))

            # Step 3: ObjectMesher takes the completed point cloud
            mesh_summary = reconstruct_object_meshes(
                objects_dir=objs_dir,
                method="poisson",
                depth=6,
            )
            self.assertIn("obj_chair", mesh_summary)
            self.assertEqual(mesh_summary["obj_chair"]["point_count"], 300)
            self.assertTrue(Path(mesh_summary["obj_chair"]["mesh_path"]).exists())

    def test_fallback_pointcloud_discovery_patterns(self):
        """
        Verify that various point cloud naming patterns without manifest are cleanly parsed.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            objs_dir = Path(tmp_dir) / "objects"
            objs_dir.mkdir()

            rng = np.random.default_rng(42)
            pts = rng.uniform(-0.2, 0.2, size=(50, 3))

            # Pattern 1: obj_1_chair_pointcloud.ply -> obj_id='obj_1', label='chair'
            trimesh.PointCloud(vertices=pts).export(str(objs_dir / "obj_1_chair_pointcloud.ply"))
            # Pattern 2: obj_chair_chair_pointcloud.ply -> obj_id='obj_chair', label='chair'
            trimesh.PointCloud(vertices=pts).export(str(objs_dir / "obj_chair_chair_pointcloud.ply"))
            # Pattern 3: table_pointcloud.ply -> obj_id='table', label='table'
            trimesh.PointCloud(vertices=pts).export(str(objs_dir / "table_pointcloud.ply"))

            res = reconstruct_object_meshes(objects_dir=objs_dir, method="poisson", depth=6)
            self.assertIn("obj_1", res)
            self.assertEqual(res["obj_1"]["label"], "chair")
            self.assertIn("obj_chair", res)
            self.assertEqual(res["obj_chair"]["label"], "chair")
            self.assertIn("table", res)



class TestSpatialPhase3MeshPlacer(unittest.TestCase):

    def test_natural_world_placement_default(self):
        """
        Verify that by default (enable_snapping=False), object meshes are placed at their
        exact natural world coordinates (delta = [0, 0, 0]) without artificial shift.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            objs_dir = Path(tmp_dir) / "objects"
            aligned_dir = Path(tmp_dir) / "objects_aligned"
            objs_dir.mkdir()

            box = trimesh.creation.box(extents=[0.4, 0.8, 0.4])
            box.apply_translation([1.2, 0.75, -2.1])
            m_path = objs_dir / "obj_test_box.ply"
            box.export(str(m_path))

            manifest = align_and_place_object_meshes(
                objects_dir=objs_dir,
                out_dir=aligned_dir,
                enable_snapping=False,
            )
            self.assertEqual(len(manifest), 1)
            entry = manifest[0]
            self.assertEqual(entry["placement_type"], "natural_world")
            self.assertEqual(entry["delta_translation"], [0.0, 0.0, 0.0])
            self.assertEqual(entry["delta_y_applied"], 0.0)

            # Placed mesh vertices must be identical to original
            placed_mesh = trimesh.load(entry["aligned_path"])
            np.testing.assert_array_almost_equal(placed_mesh.vertices, box.vertices)


    def test_snap_mesh_to_surface(self):
        box = trimesh.creation.box(extents=[0.4, 0.8, 0.4])
        # Position box so its bottom min_y is at Y = 0.5m
        box.apply_translation([0.0, 0.9, 0.0])
        min_y_before = float(box.vertices[:, 1].min())
        self.assertAlmostEqual(min_y_before, 0.5, places=4)

        snapped_box, delta_y = snap_mesh_to_surface(box, surface_y=0.0, margin=0.0)
        min_y_after = float(snapped_box.vertices[:, 1].min())

        self.assertAlmostEqual(delta_y, -0.5, places=4)
        self.assertAlmostEqual(min_y_after, 0.0, places=4)

    def test_snap_mesh_scene_input(self):
        box = trimesh.creation.box(extents=[0.4, 0.8, 0.4])
        box.apply_translation([0.0, 0.9, 0.0])
        scene = trimesh.Scene(box)

        snapped_mesh, delta_y = snap_mesh_to_surface(scene, surface_y=0.0, margin=0.0)
        self.assertIsInstance(snapped_mesh, trimesh.Trimesh)
        min_y_after = float(snapped_mesh.vertices[:, 1].min())
        self.assertAlmostEqual(min_y_after, 0.0, places=4)

    def test_snap_mesh_to_wall(self):
        # Create a TV mesh (thin box) hanging on a wall at Y = 1.5m, X = 0.0m, Z = -1.5m
        tv_box = trimesh.creation.box(extents=[1.0, 0.6, 0.05])
        tv_box.apply_translation([0.0, 1.5, -1.5])

        wall_plane = {
            "id": 1,
            "equation": [0.0, 0.0, 1.0, 2.0],  # 0*x + 0*y + 1*z + 2.0 = 0 -> Wall at Z = -2.0
            "normal": [0.0, 0.0, 1.0],
            "min_bound": [-3.0, 0.0, -2.05],
            "max_bound": [3.0, 3.0, -1.95],
        }

        snapped_tv, delta_trans = snap_mesh_to_wall(tv_box, wall_plane, margin=0.01)
        verts_snapped = np.asarray(snapped_tv.vertices)

        # 1. Height Y must be preserved exactly
        self.assertAlmostEqual(delta_trans[1], 0.0, places=5)
        self.assertAlmostEqual(float(verts_snapped[:, 1].mean()), 1.5, places=4)

        # 2. Back of TV (min Z) must be shifted close to wall Z = -2.0 + margin 0.01 = -1.99
        min_z_snapped = float(verts_snapped[:, 2].min())
        self.assertAlmostEqual(min_z_snapped, -1.99, places=4)

    def test_align_and_place_wall_mounted_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            objs_dir = Path(tmp_dir) / "objects"
            objs_dir.mkdir()
            planes_json_path = Path(tmp_dir) / "detected_planes.json"
            out_dir = Path(tmp_dir) / "objects_aligned"

            # Create TV mesh
            tv_mesh = trimesh.creation.box(extents=[1.0, 0.6, 0.05])
            tv_mesh.apply_translation([0.0, 1.5, -1.5])
            tv_mesh.export(str(objs_dir / "obj_1_tv.ply"))

            # Create Chair mesh
            chair_mesh = trimesh.creation.box(extents=[0.5, 0.8, 0.5])
            chair_mesh.apply_translation([0.0, 0.8, 0.0])
            chair_mesh.export(str(objs_dir / "obj_2_chair.ply"))

            manifest_data = {
                "obj_1_tv": {"label": "tv"},
                "obj_2_chair": {"label": "chair"}
            }
            with open(objs_dir / "objects_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest_data, f)

            planes_data = {
                "floor": {"mean_y": 0.0, "min_bound": [-3, -0.05, -3], "max_bound": [3, 0.05, 3]},
                "tables": [],
                "walls": [
                    {
                        "id": 0,
                        "equation": [0.0, 0.0, 1.0, 2.0],
                        "normal": [0.0, 0.0, 1.0],
                        "min_bound": [-3, 0, -2.05],
                        "max_bound": [3, 3, -1.95],
                    }
                ]
            }
            with open(planes_json_path, "w", encoding="utf-8") as f:
                json.dump(planes_data, f)

            summary = align_and_place_object_meshes(
                objects_dir=objs_dir,
                plane_data_path=planes_json_path,
                out_dir=out_dir,
                enable_snapping=True,
            )

            self.assertEqual(len(summary), 2)
            summary_dict = {item["name"]: item for item in summary}

            # TV must have placement_type == "wall", delta_y == 0.0
            tv_item = summary_dict["obj_1_tv"]
            self.assertEqual(tv_item["placement_type"], "wall")
            self.assertAlmostEqual(tv_item["delta_y_applied"], 0.0)

            # Chair must have placement_type == "floor", delta_y == -0.4
            chair_item = summary_dict["obj_2_chair"]
            self.assertEqual(chair_item["placement_type"], "floor")
            self.assertAlmostEqual(chair_item["delta_y_applied"], -0.4, places=4)

    def test_align_tabletop_object_with_depth_noise(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            objs_dir = Path(tmp_dir) / "objects"
            objs_dir.mkdir()
            planes_json_path = Path(tmp_dir) / "detected_planes.json"
            out_dir = Path(tmp_dir) / "objects_aligned"

            # Cup/laptop on table: Table is at Y=0.75m.
            # Object bottom has slight depth noise (min_y = 0.58m, lower by 17cm, but center_y is at 0.85m)
            laptop_mesh = trimesh.creation.box(extents=[0.3, 0.2, 0.3])
            # Box height is 0.2m, center at Y=0.68m -> min_y = 0.58m, max_y = 0.78m
            laptop_mesh.apply_translation([0.0, 0.68, 0.0])
            laptop_mesh.export(str(objs_dir / "obj_1_laptop.ply"))

            manifest_data = {
                "obj_1_laptop": {"label": "laptop"}
            }
            with open(objs_dir / "objects_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest_data, f)

            planes_data = {
                "floor": {"mean_y": 0.0, "min_bound": [-3, -0.05, -3], "max_bound": [3, 0.05, 3]},
                "tables": [
                    {
                        "id": 1,
                        "mean_y": 0.75,
                        "min_bound": [-0.6, 0.73, -0.6],
                        "max_bound": [0.6, 0.77, 0.6],
                    }
                ],
                "walls": []
            }
            with open(planes_json_path, "w", encoding="utf-8") as f:
                json.dump(planes_data, f)

            summary = align_and_place_object_meshes(
                objects_dir=objs_dir,
                plane_data_path=planes_json_path,
                out_dir=out_dir,
                enable_snapping=True,
            )

            self.assertEqual(len(summary), 1)
            item = summary[0]
            # Must be placed on table (not dropped to floor)
            self.assertEqual(item["placement_type"], "table")
            # Bottom should be moved from 0.58m up to table surface 0.75m -> delta_y = +0.17m
            self.assertAlmostEqual(item["delta_y_applied"], 0.17, places=3)

    def test_align_tabletop_tv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            objs_dir = Path(tmp_dir) / "objects"
            objs_dir.mkdir()
            planes_json_path = Path(tmp_dir) / "detected_planes.json"
            out_dir = Path(tmp_dir) / "objects_aligned"

            # Desktop TV resting on TV stand table at Y=0.50m
            tv_mesh = trimesh.creation.box(extents=[0.8, 0.5, 0.1])
            tv_mesh.apply_translation([0.0, 0.75, 0.0])  # min_y = 0.50m
            tv_mesh.export(str(objs_dir / "obj_tv.ply"))

            manifest_data = {
                "obj_tv": {"label": "tv"}
            }
            with open(objs_dir / "objects_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest_data, f)

            planes_data = {
                "floor": {"mean_y": 0.0, "min_bound": [-3, -0.05, -3], "max_bound": [3, 0.05, 3]},
                "tables": [
                    {
                        "id": 1,
                        "mean_y": 0.50,
                        "min_bound": [-0.8, 0.48, -0.4],
                        "max_bound": [0.8, 0.52, 0.4],
                    }
                ],
                "walls": [
                    {
                        "id": 0,
                        "equation": [0.0, 0.0, 1.0, 2.0],  # Wall far away at Z=-2.0m
                        "normal": [0.0, 0.0, 1.0],
                        "min_bound": [-3, 0, -2.05],
                        "max_bound": [3, 3, -1.95],
                    }
                ]
            }
            with open(planes_json_path, "w", encoding="utf-8") as f:
                json.dump(planes_data, f)

            summary = align_and_place_object_meshes(
                objects_dir=objs_dir,
                plane_data_path=planes_json_path,
                out_dir=out_dir,
                enable_snapping=True,
            )

            self.assertEqual(len(summary), 1)
            item = summary[0]
            # Must be placed on tabletop
            self.assertEqual(item["placement_type"], "table")

    def test_align_tabletop_object_max_y_branch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            objs_dir = Path(tmp_dir) / "objects"
            objs_dir.mkdir()
            planes_json_path = Path(tmp_dir) / "detected_planes.json"
            out_dir = Path(tmp_dir) / "objects_aligned"

            # Tall object: Table is at Y=0.75m.
            # Object height = 0.40m, center_y = 0.60m -> min_y = 0.40m, max_y = 0.80m
            # Notice obj_center_y (0.60) < t_y - 0.10 (0.65), so Python MUST evaluate obj_max_y > t_y (0.80 > 0.75)
            tall_object = trimesh.creation.box(extents=[0.2, 0.4, 0.2])
            tall_object.apply_translation([0.0, 0.60, 0.0])
            tall_object.export(str(objs_dir / "obj_tall.ply"))

            manifest_data = {
                "obj_tall": {"label": "bottle"}
            }
            with open(objs_dir / "objects_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest_data, f)

            planes_data = {
                "floor": {"mean_y": 0.0, "min_bound": [-3, -0.05, -3], "max_bound": [3, 0.05, 3]},
                "tables": [
                    {
                        "id": 1,
                        "mean_y": 0.75,
                        "min_bound": [-0.6, 0.73, -0.6],
                        "max_bound": [0.6, 0.77, 0.6],
                    }
                ],
                "walls": []
            }
            with open(planes_json_path, "w", encoding="utf-8") as f:
                json.dump(planes_data, f)

            summary = align_and_place_object_meshes(
                objects_dir=objs_dir,
                plane_data_path=planes_json_path,
                out_dir=out_dir,
                enable_snapping=True,
            )


            self.assertEqual(len(summary), 1)
            item = summary[0]
            self.assertEqual(item["placement_type"], "table")
            self.assertAlmostEqual(item["delta_y_applied"], 0.35, places=3)

    def test_align_empty_objects_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_objs_dir = Path(tmp_dir) / "empty_objects"
            empty_objs_dir.mkdir()
            planes_json = Path(tmp_dir) / "planes.json"
            with open(planes_json, "w", encoding="utf-8") as f:
                json.dump({"floor": {"mean_y": 0.0}, "tables": [], "walls": []}, f)
            out_dir = Path(tmp_dir) / "aligned"

            summary = align_and_place_object_meshes(
                objects_dir=empty_objs_dir,
                plane_data_path=planes_json,
                out_dir=out_dir
            )
            self.assertEqual(summary, [])

    def test_assemble_full_scene(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            room_ply = Path(tmp_dir) / "room_bg.ply"
            aligned_dir = Path(tmp_dir) / "aligned"
            aligned_dir.mkdir()
            out_scene = Path(tmp_dir) / "full_scene.ply"

            # Create dummy room mesh box
            room_box = trimesh.creation.box(extents=[4, 3, 4])
            room_box.export(str(room_ply))

            # Create dummy aligned chair mesh box
            chair_box = trimesh.creation.box(extents=[0.5, 0.8, 0.5])
            chair_ply = aligned_dir / "obj_0_chair.ply"
            chair_box.export(str(chair_ply))

            scene = assemble_full_scene(
                room_mesh_path=room_ply,
                aligned_objects_dir=aligned_dir,
                out_scene_path=out_scene
            )
            self.assertIsNotNone(scene)
            self.assertTrue(out_scene.exists())
            self.assertTrue(out_scene.with_suffix(".obj").exists())
            self.assertEqual(len(scene.vertices), len(room_box.vertices) + len(chair_box.vertices))

    def test_align_and_place_ignores_pointcloud_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            objs_dir = Path(tmp_dir) / "objects"
            objs_dir.mkdir()
            out_dir = Path(tmp_dir) / "aligned"
            planes_json = Path(tmp_dir) / "planes.json"

            with open(planes_json, "w", encoding="utf-8") as f:
                json.dump({"floor": {"mean_y": 0.0}, "tables": [], "walls": []}, f)

            # Create 1 mesh file and 1 point cloud file
            chair_mesh = trimesh.creation.box(extents=[0.5, 0.8, 0.5])
            chair_mesh.export(str(objs_dir / "obj_0_chair.ply"))
            pcd = trimesh.PointCloud(vertices=np.random.uniform(-0.2, 0.2, (50, 3)))
            pcd.export(str(objs_dir / "obj_0_chair_pointcloud.ply"))

            summary = align_and_place_object_meshes(
                objects_dir=objs_dir,
                plane_data_path=planes_json,
                out_dir=out_dir
            )

            # Only the mesh should be processed, NOT the pointcloud
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["name"], "obj_0_chair")
            # Point cloud file should NOT be copied to objects_aligned
            self.assertFalse((out_dir / "obj_0_chair_pointcloud.ply").exists())

    def test_full_sequential_phase_1_2_3(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proc_dir = Path(tmp_dir) / "processed"
            proc_dir.mkdir(parents=True, exist_ok=True)
            objs_dir = proc_dir / "objects"
            aligned_dir = proc_dir / "objects_aligned"
            out_dir = Path(tmp_dir) / "output"
            out_dir.mkdir(parents=True, exist_ok=True)

            world_ply = proc_dir / "world_pointcloud.ply"
            raw_depths_npz = proc_dir / "raw_depths.npz"
            det_json = proc_dir / "detections.json"
            planes_json = proc_dir / "detected_planes.json"

            # Create synthetic room + chair
            rng = np.random.default_rng(42)
            n_floor = 1000
            fx = rng.uniform(-2, 2, n_floor)
            fz = rng.uniform(-2, 2, n_floor)
            fy = rng.normal(0.0, 0.005, n_floor)
            floor_pts = np.column_stack([fx, fy, fz])

            n_chair = 200
            cx = rng.uniform(0.5, 0.9, n_chair)
            cz = rng.uniform(0.5, 0.9, n_chair)
            cy = rng.uniform(0.05, 0.65, n_chair)
            chair_pts = np.column_stack([cx, cy, cz])

            all_pts = np.vstack([floor_pts, chair_pts])
            trimesh.PointCloud(vertices=all_pts).export(str(world_ply))

            # Phase 1: Architectural Plane Detection
            plane_meta = detect_architectural_planes(
                ply_path=world_ply,
                distance_threshold=0.04,
                out_json=planes_json,
            )
            self.assertIsNotNone(plane_meta["floor"])
            self.assertTrue(planes_json.exists())

            # Phase 2: Object Estimation
            depth_0 = np.ones((60, 60), dtype=np.float32) * 1.8
            ixt_0 = np.array([[50.0, 0.0, 30.0], [0.0, 50.0, 30.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            ext_0 = np.eye(4, dtype=np.float64)
            np.savez(str(raw_depths_npz), depth_0=depth_0, ixt_0=ixt_0, ext_0=ext_0)

            det_data = {
                "obj_0": {
                    "label": "chair",
                    "associated_views": [
                        {
                            "frame_index": 0,
                            "bbox": [10, 10, 50, 50],
                        }
                    ]
                }
            }
            with open(det_json, "w", encoding="utf-8") as f:
                json.dump(det_data, f)

            objs_meta = process_object_detections(
                detections_path=det_json,
                raw_depths_path=raw_depths_npz,
                world_pcd_path=world_ply,
                plane_data_path=planes_json,
                out_dir=objs_dir,
            )
            self.assertIn("obj_0", objs_meta)
            self.assertTrue(Path(objs_meta["obj_0"]["pcd_path"]).exists())
            self.assertTrue(Path(objs_meta["obj_0"]["mesh_path"]).exists())

            # Phase 3: Mesh Alignment & Full Scene Assembly
            scene_path = out_dir / "full_scene_reconstruction.ply"
            room_bg_ply = proc_dir / "room_background_mesh.ply"

            aligned_summary = align_and_place_object_meshes(
                objects_dir=objs_dir,
                plane_data_path=planes_json,
                out_dir=aligned_dir,
            )
            self.assertEqual(len(aligned_summary), 1)

            scene = assemble_full_scene(
                room_mesh_path=room_bg_ply,
                aligned_objects_dir=aligned_dir,
                out_scene_path=scene_path,
                objects_dir=objs_dir,
                world_ply_path=world_ply,
            )
            self.assertIsNotNone(scene)
            self.assertTrue(scene_path.exists())


from pointcloud.mesh_reconstructor import fill_mesh_holes, smooth_mesh_taubin, post_process_mesh


class TestPipelineEnhancements(unittest.TestCase):

    def test_semantic_and_span_table_plane_verification(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ply_path = Path(tmp_dir) / "room.ply"
            rng = np.random.default_rng(42)

            # Floor at Y = 0.0
            n_floor = 1500
            fx = rng.uniform(-2, 2, n_floor)
            fz = rng.uniform(-2, 2, n_floor)
            fy = rng.normal(0.0, 0.005, n_floor)
            floor_pts = np.column_stack([fx, fy, fz])

            # Small chair cushion patch at Y = 0.45m (narrow span: 0.20m x 0.20m, 50 points -> not a table)
            n_cushion = 50
            cx = rng.uniform(-0.1, 0.1, n_cushion)
            cz = rng.uniform(-0.1, 0.1, n_cushion)
            cy = rng.normal(0.45, 0.005, n_cushion)
            cushion_pts = np.column_stack([cx, cy, cz])

            # Real Desk at Y = 0.75m (broad span: 0.8m x 0.8m, 400 points)
            n_table = 400
            tx = rng.uniform(-0.4, 0.4, n_table)
            tz = rng.uniform(-0.4, 0.4, n_table)
            ty = rng.normal(0.75, 0.005, n_table)
            table_pts = np.column_stack([tx, ty, tz])

            all_pts = np.vstack([floor_pts, cushion_pts, table_pts])
            trimesh.PointCloud(vertices=all_pts).export(str(ply_path))

            res = detect_architectural_planes(
                ply_path=ply_path,
                distance_threshold=0.03,
                max_planes=4,
                min_inliers=40,
            )

            # Verify that the desk at 0.75m is captured as a table plane
            self.assertIsNotNone(res["floor"])
            self.assertTrue(any(abs(tp["mean_y"] - 0.75) < 0.05 for tp in res["tables"]))

    def test_adaptive_foreground_depth_gating_deep_object(self):
        # Deep desk/sofa extending from 1.0m to 1.7m depth (depth delta = 0.7m)
        mask_2d = np.ones((60, 60), dtype=np.uint8)
        depth_map = np.linspace(1.0, 1.7, 60)[:, None] * np.ones((1, 60))
        K = np.array([[50.0, 0.0, 30.0], [0.0, 50.0, 30.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)

        # With adaptive foreground_margin >= 0.85m, entire depth range 1.0m - 1.7m must be captured
        pts_w, _ = backproject_mask_to_3d(mask_2d, depth_map, K, c2w, foreground_margin=0.85)
        self.assertGreater(len(pts_w), 0)
        self.assertAlmostEqual(float(pts_w[:, 2].min()), 1.0, places=1)
        self.assertAlmostEqual(float(pts_w[:, 2].max()), 1.7, places=1)

    def test_mesh_hole_filling_and_taubin_smoothing(self):
        # Create a cube mesh with a hole (missing 2 faces)
        cube = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
        # Remove a face to introduce open boundary
        open_cube = trimesh.Trimesh(vertices=cube.vertices, faces=cube.faces[:-2])
        self.assertFalse(open_cube.is_watertight)

        # Post-process (fill holes + Taubin smoothing)
        repaired = post_process_mesh(open_cube, fill_holes=True, smooth=True, iterations=5)
        self.assertIsNotNone(repaired)
        self.assertGreater(len(repaired.faces), len(open_cube.faces))

    def test_dilated_room_subtraction_purges_chair_legs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            world_ply = Path(tmp_dir) / "world.ply"
            objs_dir = Path(tmp_dir) / "objects"
            objs_dir.mkdir()
            out_pcd = Path(tmp_dir) / "room_bg.ply"

            # World points: Floor points at Y=0.0, plus chair body at (1.0, 0.5, 1.0) and chair wheel at (1.03, 0.02, 1.03)
            rng = np.random.default_rng(42)
            floor_pts = rng.uniform(-1.0, 1.0, size=(100, 3))
            floor_pts[:, 1] = 0.0
            chair_body_pts = rng.uniform(0.9, 1.1, size=(50, 3))
            chair_body_pts[:, 1] = 0.5
            chair_wheel_pts = np.array([[1.03, 0.02, 1.03], [0.98, 0.02, 0.98]])

            all_world = np.vstack([floor_pts, chair_body_pts, chair_wheel_pts])
            trimesh.PointCloud(vertices=all_world).export(str(world_ply))

            # Segmented chair point cloud
            chair_pcd = objs_dir / "obj_1_chair_pointcloud.ply"
            trimesh.PointCloud(vertices=chair_body_pts).export(str(chair_pcd))

            # With dilated subtraction radius = 0.05m (5cm) or proximity
            bg_pts, _, p_out = extract_room_background_pointcloud(
                world_ply_path=world_ply,
                objects_dir=objs_dir,
                out_pcd_path=out_pcd,
                subtraction_radius=0.05,
            )

            self.assertTrue(p_out.exists())
            # Chair body at Y=0.5m must be removed
            self.assertFalse(np.any((bg_pts[:, 1] > 0.4) & (bg_pts[:, 1] < 0.6)))


if __name__ == "__main__":
    unittest.main()




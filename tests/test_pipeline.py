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


from spatial.room_builder import detect_architectural_planes
from spatial.object_estimator import backproject_mask_to_3d, filter_object_pointcloud_dbscan
from spatial.mesh_placer import snap_mesh_to_surface
import trimesh


class TestSpatialPhase1RoomBuilder(unittest.TestCase):

    def test_detect_architectural_planes_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ply_path = Path(tmp_dir) / "synthetic_room.ply"
            out_obj = Path(tmp_dir) / "room_layout.obj"
            out_json = Path(tmp_dir) / "detected_planes.json"

            result = detect_architectural_planes(
                ply_path=ply_path,
                distance_threshold=0.04,
                max_planes=3,
                out_obj=out_obj,
                out_json=out_json,
            )

            self.assertIsNotNone(result["floor"])
            self.assertAlmostEqual(result["floor"]["mean_y"], 0.0, delta=0.1)
            self.assertTrue(out_obj.exists())
            self.assertTrue(out_json.exists())

            # Verify plane equation normalization
            for plane in result["all_planes"]:
                eq = plane["equation"]
                norm_len = np.linalg.norm(eq[:3])
                self.assertAlmostEqual(norm_len, 1.0, places=5)


class TestSpatialPhase2ObjectEstimator(unittest.TestCase):

    def test_backproject_mask_to_3d(self):
        mask_2d = np.zeros((50, 50), dtype=np.uint8)
        mask_2d[10:30, 10:30] = 255
        depth_map = np.ones((50, 50), dtype=np.float64) * 2.0
        K = np.array([[50.0, 0.0, 25.0], [0.0, 50.0, 25.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)

        pts_w, _ = backproject_mask_to_3d(mask_2d, depth_map, K, c2w)
        self.assertEqual(len(pts_w), 20 * 20)
        self.assertAlmostEqual(pts_w[:, 2].mean(), 2.0, places=4)

    def test_dbscan_filtering(self):
        rng = np.random.default_rng(42)
        dense = rng.uniform(-0.05, 0.05, size=(100, 3))
        noise = np.array([[10.0, 10.0, 10.0]], dtype=np.float64)
        pts = np.vstack([dense, noise])

        clean_pts, _ = filter_object_pointcloud_dbscan(pts, eps=0.1, min_samples=5, min_cluster_size=20)
        self.assertEqual(len(clean_pts), 100)


class TestSpatialPhase3MeshPlacer(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()




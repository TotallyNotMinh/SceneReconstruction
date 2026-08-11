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
from pointcloud.pointcloud_builder import build_pointcloud_from_npz


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


if __name__ == "__main__":
    unittest.main()

# core/data_loader.py
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import KDTree

from core.coordinate_adapter import CoordinateAdapter


# ── Pure-numpy helpers (no open3d) ────────────────────────────────────────────

def _voxel_downsample(pts: np.ndarray, voxel_size: float = 0.02) -> np.ndarray:
    """Keep one point per voxel cell using a numpy integer-key hash."""
    if len(pts) == 0:
        return pts
    voxel_ids = np.floor(pts / voxel_size).astype(np.int64)
    shift = voxel_ids.max(axis=0) - voxel_ids.min(axis=0) + 1
    keys = (voxel_ids[:, 0] * shift[1] * shift[2]
            + voxel_ids[:, 1] * shift[2]
            + voxel_ids[:, 2])
    _, first = np.unique(keys, return_index=True)
    return pts[first]


def _remove_statistical_outliers(
    pts: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0
) -> np.ndarray:
    """Remove points whose mean k-NN distance exceeds mean + std_ratio * std."""
    if len(pts) <= nb_neighbors:
        return pts
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=nb_neighbors + 1)   # col-0 is self (dist=0)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    return pts[mean_dists <= threshold]


# ── DataLoader ────────────────────────────────────────────────────────────────

class DataLoader:

    @staticmethod
    def load_ar_metadata(json_path: Path) -> tuple[list, dict]:
        """
        Parse ar_metadata.json and return (intrinsics, valid_frames).

        intrinsics : [fx, fy, cx, cy]
        valid_frames : {frame_id -> {"timestamp": int, "pose": np.ndarray 4x4}}
        """
        if not json_path.exists():
            raise FileNotFoundError(f"AR metadata not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        intrinsics = data["intrinsics"]   # [fx, fy, cx, cy]
        valid_frames: dict = {}

        for frame in data["frames"]:
            if frame.get("tracking_state") != "TRACKING":
                continue
            raw_pose = np.array(frame["pose_matrix"], dtype=np.float64)
            if raw_pose.shape != (4, 4):
                print(f"[DataLoader] WARNING: frame {frame['frame_id']} has malformed pose, skipping.")
                continue
            world_pose = CoordinateAdapter.arkit_to_world(raw_pose)
            valid_frames[frame["frame_id"]] = {
                "timestamp": frame["timestamp_ns"],
                "pose": world_pose,
            }

        print(f"[DataLoader] Loaded {len(valid_frames)} TRACKING frames from {json_path.name}")
        return intrinsics, valid_frames

    @staticmethod
    def load_point_cloud(ply_path: Path, voxel_size: float = 0.02) -> np.ndarray:
        """
        Load a PLY point cloud, voxel-downsample and remove statistical outliers.

        Returns a (N, 3) float64 numpy array of clean 3-D points.
        """
        if not ply_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {ply_path}")

        cloud = trimesh.load(str(ply_path))

        if isinstance(cloud, trimesh.PointCloud):
            pts = np.asarray(cloud.vertices, dtype=np.float64)
        elif isinstance(cloud, trimesh.Trimesh):
            pts = np.asarray(cloud.vertices, dtype=np.float64)
        else:
            raise ValueError(f"Unexpected trimesh type for PLY: {type(cloud)}")

        pts = _voxel_downsample(pts, voxel_size)
        pts = _remove_statistical_outliers(pts, nb_neighbors=20, std_ratio=2.0)

        print(f"[DataLoader] Point cloud loaded: {len(pts):,} points after cleaning.")
        return pts

    @staticmethod
    def load_detections(json_path: Path) -> dict:
        """Load detections.json → {obj_id: {associated_views: [...]}}"""
        if not json_path.exists():
            raise FileNotFoundError(f"Detections file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"[DataLoader] Loaded {len(data)} object detections from {json_path.name}")
        return data

    @staticmethod
    def load_depth_maps(npz_path: Path) -> dict:
        """Load per-frame metric depth maps from a .npz file."""
        if not npz_path.exists():
            raise FileNotFoundError(f"Depth maps file not found: {npz_path}")

        npz = np.load(str(npz_path))
        depth_maps = {int(k): npz[k].astype(np.float64) for k in npz.files if k.isdigit() or k.startswith("depth_")}
        print(f"[DataLoader] Loaded {len(depth_maps)} depth maps from {npz_path.name}")
        return depth_maps

    @staticmethod
    def load_rgb_frames(video_path: Path, frame_ids: list) -> dict:
        """Seek-load required frames from video for SAM input."""
        import cv2 as _cv2
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = _cv2.VideoCapture(str(video_path))
        total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))

        rgb_frames: dict = {}
        for fid in sorted(set(frame_ids)):
            if fid >= total:
                continue
            cap.set(_cv2.CAP_PROP_POS_FRAMES, fid)
            ret, bgr = cap.read()
            if not ret:
                continue
            rgb_frames[fid] = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)

        cap.release()
        print(f"[DataLoader] Loaded {len(rgb_frames)} RGB frames from {video_path.name}")
        return rgb_frames

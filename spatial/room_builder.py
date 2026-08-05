# spatial/room_builder.py
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

import config


def _ransac_plane(
    pts: np.ndarray,
    distance_thresh: float = config.RANSAC_DISTANCE_THRESH,
    num_iterations: int = config.RANSAC_ITERATIONS,
    min_inliers: int = config.RANSAC_MIN_INLIERS,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    n = len(pts)
    if n < 3:
        return None, np.zeros(n, dtype=bool)

    best_mask = np.zeros(n, dtype=bool)
    best_count = 0
    best_model = None

    for _ in range(num_iterations):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            continue
        normal /= norm_len
        d = -np.dot(normal, p0)

        dists = np.abs(pts @ normal + d)
        mask = dists < distance_thresh
        count = mask.sum()
        if count > best_count:
            best_count = count
            best_mask = mask
            best_model = np.append(normal, d)

    if best_count < min_inliers or best_model is None:
        return None, np.zeros(n, dtype=bool)

    inlier_pts = pts[best_mask]
    centroid = inlier_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(inlier_pts - centroid)
    normal = Vt[-1]
    d = -np.dot(normal, centroid)

    dists = np.abs(pts @ normal + d)
    final_mask = dists < distance_thresh

    return np.append(normal, d), final_mask


class RoomBuilder:

    @staticmethod
    def extract_clean_architectural_room(room_file_path: Path) -> trimesh.Scene:
        scene = trimesh.load(str(room_file_path))
        clean_room = trimesh.Scene()
        allowed = {"wall", "floor", "ceiling", "door", "window"}

        if isinstance(scene, trimesh.Trimesh):
            clean_room.add_geometry(scene, node_name="Architecture_Main")
            return clean_room

        for node_name, geometry in scene.geometry.items():
            if any(kw in node_name.lower() for kw in allowed):
                clean_room.add_geometry(geometry, node_name=f"Arch_{node_name}")

        if not clean_room.geometry:
            print("[RoomBuilder] WARNING: no named architectural geometry found — "
                  "including full scene.")
            for node_name, geometry in scene.geometry.items():
                clean_room.add_geometry(geometry, node_name=node_name)

        return clean_room

    @staticmethod
    def detect_support_surfaces_ransac(
        pcd_pts: np.ndarray,
        max_planes: int = 8,
    ) -> list[dict]:
        surfaces: list[dict] = []
        remaining = pcd_pts.copy()
        horizontal_planes: list[dict] = []
        rng = np.random.default_rng(seed=42)

        for _ in range(max_planes):
            if len(remaining) < config.RANSAC_MIN_INLIERS:
                break

            model, mask = _ransac_plane(remaining, rng=rng)
            if model is None:
                break

            a, b, c, d = model

            if abs(b) > config.FLOOR_NORMAL_THRESH:
                y_level = -d / b
                inlier_pts = remaining[mask]
                min_x, max_x = float(inlier_pts[:, 0].min()), float(inlier_pts[:, 0].max())
                min_z, max_z = float(inlier_pts[:, 2].min()), float(inlier_pts[:, 2].max())
                horizontal_planes.append({
                    "y_level":    float(y_level),
                    "bounds_xz":  (min_x, max_x, min_z, max_z),
                    "num_pts":    int(mask.sum()),
                })

            remaining = remaining[~mask]

        if not horizontal_planes:
            print("[RoomBuilder] WARNING: No horizontal planes found via RANSAC — "
                  "defaulting floor to Y=0.0.")
            return [{"type": "floor", "y_level": 0.0}]

        horizontal_planes.sort(key=lambda p: p["y_level"])

        floor = horizontal_planes[0]
        surfaces.append({"type": "floor", "y_level": floor["y_level"]})
        print(f"[RoomBuilder] Floor detected at Y = {floor['y_level']:.3f} m  "
              f"({floor['num_pts']:,} inliers)")

        for hp in horizontal_planes[1:]:
            height = hp["y_level"] - floor["y_level"]
            if config.TABLE_MIN_HEIGHT <= height <= config.TABLE_MAX_HEIGHT:
                surfaces.append({
                    "type":       "table",
                    "y_level":    hp["y_level"],
                    "bounds_xz":  hp["bounds_xz"],
                })
                print(f"[RoomBuilder] Table surface at Y = {hp['y_level']:.3f} m  "
                      f"(+{height:.2f} m above floor,  {hp['num_pts']:,} inliers)")

        return surfaces

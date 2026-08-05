# spatial/mesh_placer.py
from typing import Optional
import numpy as np
import trimesh

import config
from core.coordinate_adapter import CoordinateAdapter


class MeshPlacer:

    @staticmethod
    def align_and_place_mesh(
        triposr_mesh: trimesh.Trimesh,
        obj_3d_info: dict,
        support_surfaces: list[dict],
        max_distortion_thresh: float = config.MAX_DISTORTION_THRESH,
    ) -> trimesh.Trimesh:
        mesh = CoordinateAdapter.triposr_to_world(triposr_mesh)

        mesh_extent = np.maximum(mesh.extents, 1e-4)
        real_dx, real_dy, real_dz = obj_3d_info["size"]

        sx = real_dx / mesh_extent[0]
        sy = real_dy / mesh_extent[1]
        sz = real_dz / mesh_extent[2]
        scales = [sx, sy, sz]

        distortion_ratio = max(scales) / min(scales)
        if distortion_ratio <= max_distortion_thresh:
            mesh.apply_scale(sy)
        else:
            mesh.apply_transform(np.diag([sx, sy, sz, 1.0]))

        R_yaw = trimesh.transformations.rotation_matrix(obj_3d_info["yaw"], [0, 1, 0])
        mesh.apply_transform(R_yaw)

        cx, cy, cz = obj_3d_info["center"]

        floor_surface = next((s for s in support_surfaces if s["type"] == "floor"), None)
        target_y = floor_surface["y_level"] if floor_surface is not None else 0.0

        for surface in support_surfaces:
            if surface["type"] != "table":
                continue
            height_match = abs(cy - surface["y_level"]) < config.TABLE_SNAP_TOLERANCE
            min_x, max_x, min_z, max_z = surface.get(
                "bounds_xz", (-np.inf, np.inf, -np.inf, np.inf)
            )
            footprint_match = (min_x <= cx <= max_x) and (min_z <= cz <= max_z)
            if height_match and footprint_match:
                target_y = surface["y_level"]
                print(f"[MeshPlacer] Snapping object to table at Y = {target_y:.3f} m")
                break

        mesh_bottom = float(mesh.bounds[0][1])
        mesh.apply_translation([cx, target_y - mesh_bottom, cz])

        return mesh

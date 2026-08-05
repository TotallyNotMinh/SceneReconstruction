# spatial/object_estimator.py
"""
Per-object 3-D reconstruction via depth back-projection and alpha-shape meshing.
"""

from typing import Optional
import numpy as np
from sklearn.decomposition import PCA

import config

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False
    print("[ObjectEstimator] WARNING: open3d not found — alpha-shape meshing disabled. "
          "Install with: pip install open3d")

try:
    import trimesh as _trimesh
except ImportError:
    _trimesh = None


class ObjectEstimator:

    @staticmethod
    def estimate_object_3d_properties(
        associated_views: list,
        valid_frames_dict: dict,
        intrinsics: list,
        depth_maps: dict,
        sam_segmentor,
        rgb_frames: dict,
        alpha: float = config.ALPHA_SHAPE_ALPHA,
        max_faces: int = config.ALPHA_MAX_FACES,
        voxel_size: float = config.VOXEL_SIZE_PCD,
    ) -> Optional[dict]:
        fx, fy, cx, cy = intrinsics
        per_frame_clouds: list = []

        for view in associated_views:
            frame_id = view["frame_id"]
            bbox     = view["bbox_px"]

            if frame_id not in valid_frames_dict:
                continue
            if frame_id not in depth_maps:
                continue

            pose      = valid_frames_dict[frame_id]["pose"]
            depth_map = depth_maps[frame_id]

            if sam_segmentor is not None and rgb_frames and frame_id in rgb_frames:
                mask = sam_segmentor.segment_object(rgb_frames[frame_id], bbox)
            else:
                h, w = depth_map.shape
                mask = np.zeros((h, w), dtype=bool)
                x0, y0, x1, y1 = (
                    max(0, int(bbox[0])), max(0, int(bbox[1])),
                    min(w, int(bbox[2])), min(h, int(bbox[3])),
                )
                mask[y0:y1, x0:x1] = True

            pts_world = ObjectEstimator._backproject_frame(
                depth_map, mask, pose, fx, fy, cx, cy
            )
            if len(pts_world) > 0:
                per_frame_clouds.append(pts_world)

        if not per_frame_clouds:
            return None

        fused_pts = ObjectEstimator._fuse_point_clouds(per_frame_clouds, voxel_size)

        if len(fused_pts) < 10:
            return None

        mesh = ObjectEstimator._alpha_shape_mesh(fused_pts, alpha)
        if mesh is not None and len(mesh.faces) > max_faces:
            mesh = ObjectEstimator._decimate_mesh(mesh, max_faces)

        pts_xz    = fused_pts[:, [0, 2]]
        pca       = PCA(n_components=2).fit(pts_xz)
        main_axis = pca.components_[0]
        yaw_angle = float(np.arctan2(main_axis[1], main_axis[0]))

        cos_y = np.cos(yaw_angle)
        sin_y = np.sin(yaw_angle)
        R_unrotate = np.array([
            [ cos_y, 0.0, -sin_y],
            [  0.0,  1.0,   0.0 ],
            [ sin_y, 0.0,  cos_y],
        ])
        local_pts = fused_pts @ R_unrotate.T
        min_b, max_b = local_pts.min(axis=0), local_pts.max(axis=0)
        real_dx, real_dy, real_dz = (max_b - min_b).tolist()

        return {
            "center": fused_pts.mean(axis=0),
            "size":   (float(real_dx), float(real_dy), float(real_dz)),
            "yaw":    float(yaw_angle),
            "min_y":  float(fused_pts[:, 1].min()),
            "mesh":   mesh,
        }

    @staticmethod
    def _backproject_frame(
        depth_map: np.ndarray,
        mask: np.ndarray,
        pose: np.ndarray,
        fx: float, fy: float, cx: float, cy: float,
    ) -> np.ndarray:
        if depth_map.shape != mask.shape:
            import cv2
            depth_map = cv2.resize(depth_map, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        rows, cols = np.where(mask)
        if len(rows) == 0:
            return np.empty((0, 3), dtype=np.float64)

        z_vals = depth_map[rows, cols].astype(np.float64)

        valid = z_vals > 0.01
        rows, cols, z_vals = rows[valid], cols[valid], z_vals[valid]
        if len(z_vals) == 0:
            return np.empty((0, 3), dtype=np.float64)

        x_cam = (cols - cx) * z_vals / fx
        y_cam = (rows - cy) * z_vals / fy
        z_cam = z_vals

        pts_cam_arkit = np.column_stack([x_cam, -y_cam, -z_cam])
        pts_homo = np.hstack([pts_cam_arkit, np.ones((len(pts_cam_arkit), 1))])
        pts_world = (pose @ pts_homo.T).T[:, :3]
        return pts_world.astype(np.float64)

    @staticmethod
    def _fuse_point_clouds(frame_clouds: list, voxel_size: float) -> np.ndarray:
        all_pts = np.vstack(frame_clouds).astype(np.float64)
        if len(all_pts) == 0:
            return all_pts

        voxel_ids = np.floor(all_pts / voxel_size).astype(np.int64)
        shift = voxel_ids.max(axis=0) - voxel_ids.min(axis=0) + 1
        shift = np.maximum(shift, 1)
        keys = (voxel_ids[:, 0] * shift[1] * shift[2]
                + voxel_ids[:, 1] * shift[2]
                + voxel_ids[:, 2])
        _, first = np.unique(keys, return_index=True)
        return all_pts[first]

    @staticmethod
    def _alpha_shape_mesh(pts: np.ndarray, alpha: float):
        if not _O3D_AVAILABLE or _trimesh is None:
            return None

        if len(pts) < 4:
            return None

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha
            )
            mesh_o3d.remove_degenerate_triangles()
            mesh_o3d.remove_duplicated_triangles()
            mesh_o3d.remove_duplicated_vertices()

            verts = np.asarray(mesh_o3d.vertices)
            faces = np.asarray(mesh_o3d.triangles)

            if len(verts) < 4 or len(faces) < 1:
                return None

            return _trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        except Exception as exc:
            print(f"[ObjectEstimator] Alpha shape failed ({exc}); object skipped.")
            return None

    @staticmethod
    def _decimate_mesh(mesh, target_faces: int):
        if _trimesh is None or mesh is None:
            return mesh

        try:
            ratio = target_faces / max(len(mesh.faces), 1)
            if ratio >= 1.0:
                return mesh
            decimated = mesh.simplify_quadric_decimation(target_faces)
            print(f"[ObjectEstimator] Decimated mesh: {len(mesh.faces)} → "
                  f"{len(decimated.faces)} faces.")
            return decimated
        except Exception as exc:
            print(f"[ObjectEstimator] Decimation failed ({exc}); keeping original mesh.")
            return mesh

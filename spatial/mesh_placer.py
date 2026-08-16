# -*- coding: utf-8 -*-
"""
spatial/mesh_placer.py — Phase 3: Support Surface Snapping & Spatial Alignment.

Calculates the bottom minimum Y coordinate of reconstructed 3D object meshes,
finds the nearest detected architectural support plane (Floor or Tabletop),
and translates the object vertically so its base rests snugly on the surface.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def snap_mesh_to_surface(
    mesh: Any,
    surface_y: float,
    margin: float = config.SURFACE_SNAPPING_MARGIN,
) -> Tuple[Any, float]:
    """
    Snap the bottom minimum Y coordinate of a 3D mesh onto a target surface height.

    Parameters
    ----------
    mesh : trimesh.Trimesh or open3d.geometry.TriangleMesh
    surface_y : Target Y coordinate of the support plane in meters.
    margin : Vertical offset margin (m). Default 0.0m.

    Returns
    -------
    (transformed_mesh, delta_y_applied)
    """
    if HAS_TRIMESH:
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump()) if len(mesh.dump()) > 0 else trimesh.Trimesh()

        if isinstance(mesh, trimesh.Trimesh):
            verts = np.asarray(mesh.vertices)
            if len(verts) == 0:
                return mesh, 0.0

            min_y = float(verts[:, 1].min())
            target_y = surface_y + margin
            delta_y = target_y - min_y

            mesh_copy = mesh.copy()
            mesh_copy.apply_translation([0.0, delta_y, 0.0])
            return mesh_copy, delta_y

    if HAS_OPEN3D and isinstance(mesh, o3d.geometry.TriangleMesh):
        verts = np.asarray(mesh.vertices)
        if len(verts) == 0:
            return mesh, 0.0

        min_y = float(verts[:, 1].min())
        target_y = surface_y + margin
        delta_y = target_y - min_y

        import copy
        mesh_copy = copy.deepcopy(mesh)
        mesh_copy.translate([0.0, delta_y, 0.0])
        return mesh_copy, delta_y

    raise TypeError(f"[MeshPlacer] Unsupported mesh type: {type(mesh)}")


def snap_mesh_to_wall(
    mesh: Any,
    wall_plane: Dict[str, Any],
    margin: float = config.WALL_SNAPPING_MARGIN,
) -> Tuple[Any, List[float]]:
    """
    Snap a 3D mesh horizontally against a vertical wall plane, preserving Y elevation.

    Parameters
    ----------
    mesh : trimesh.Trimesh or open3d.geometry.TriangleMesh
    wall_plane : Dict containing 'equation' [a, b, c, d] and 'normal' [nx, ny, nz]
    margin : Offset margin from wall in meters. Default 0.01m.

    Returns
    -------
    (transformed_mesh, delta_translation [dx, dy, dz])
    """
    eq = wall_plane.get("equation", [0.0, 0.0, 1.0, 0.0])
    a, b, c, d = eq
    normal = np.array([a, b, c], dtype=np.float64)
    # Project normal vector to horizontal X-Z plane to ensure no vertical drift
    normal_xz = np.array([normal[0], 0.0, normal[2]], dtype=np.float64)
    norm_len = np.linalg.norm(normal_xz)
    if norm_len < 1e-6:
        return mesh, [0.0, 0.0, 0.0]
    normal_xz = normal_xz / norm_len

    if HAS_TRIMESH:
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump()) if len(mesh.dump()) > 0 else trimesh.Trimesh()

    verts = np.asarray(mesh.vertices) if hasattr(mesh, "vertices") else np.zeros((0, 3))
    if len(verts) == 0:
        return mesh, [0.0, 0.0, 0.0]

    # Signed distances of all vertices to the wall plane
    signed_dists = verts[:, 0] * a + verts[:, 1] * b + verts[:, 2] * c + d
    center_dist = float(np.mean(signed_dists))

    # Identify the side facing the wall and the back point closest to the wall
    if center_dist >= 0:
        back_dist = float(np.min(signed_dists))
        shift_amount = margin - back_dist
    else:
        back_dist = float(np.max(signed_dists))
        shift_amount = -margin - back_dist

    # Horizontal translation vector (purely X-Z)
    delta_vec = [float(shift_amount * normal_xz[0]), 0.0, float(shift_amount * normal_xz[2])]

    if HAS_TRIMESH and isinstance(mesh, trimesh.Trimesh):
        mesh_copy = mesh.copy()
        mesh_copy.apply_translation(delta_vec)
        return mesh_copy, delta_vec

    if HAS_OPEN3D and isinstance(mesh, o3d.geometry.TriangleMesh):
        import copy
        mesh_copy = copy.deepcopy(mesh)
        mesh_copy.translate(delta_vec)
        return mesh_copy, delta_vec

    return mesh, [0.0, 0.0, 0.0]


def align_and_place_object_meshes(
    objects_dir: Optional[Path | str] = None,
    plane_data_path: Optional[Path | str] = None,
    out_dir: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """Align and snap all object meshes onto detected architectural planes (Floors, Tables, Walls)."""
    if objects_dir is None:
        objects_dir = config.PROCESSED_DATA_DIR / "objects"
    objects_dir = Path(objects_dir)

    if plane_data_path is None:
        plane_data_path = config.PROCESSED_DATA_DIR / "detected_planes.json"
    plane_data_path = Path(plane_data_path)

    if out_dir is None:
        out_dir = config.PROCESSED_DATA_DIR / "objects_aligned"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load plane metadata (from Phase 1 RoomBuilder)
    floor_y = 0.0
    table_planes: List[Dict[str, Any]] = []
    wall_planes: List[Dict[str, Any]] = []

    if plane_data_path.exists():
        with open(plane_data_path, "r", encoding="utf-8") as pf:
            planes_json = json.load(pf)
        floor = planes_json.get("floor")
        if floor:
            floor_y = float(floor.get("mean_y", 0.0))
        table_planes = planes_json.get("tables", [])
        wall_planes = planes_json.get("walls", [])
    else:
        print(f"[MeshPlacer] Plane metadata file not found at {plane_data_path}. Defaulting floor_y = 0.0m")

    # Load object manifest if available
    manifest_path = objects_dir / "objects_manifest.json"
    objects_manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as mf:
            objects_manifest = json.load(mf)

    # Find object mesh files (.ply, .obj)
    mesh_files = [
        f for f in list(objects_dir.glob("*.ply")) + list(objects_dir.glob("*.obj"))
        if f.name != "room_layout.obj" and not f.name.endswith(".json")
    ]
    if not mesh_files:
        print(f"[MeshPlacer] No object mesh files found in '{objects_dir}'. Nothing to align.")
        return []

    print(f"[MeshPlacer] Aligning & Snapping {len(mesh_files)} object meshes onto support planes/walls...")
    aligned_summary: List[Dict[str, Any]] = []

    for m_path in mesh_files:
        if m_path.name == "room_layout.obj" or m_path.name.endswith(".json"):
            continue

        if HAS_TRIMESH:
            mesh = trimesh.load(str(m_path))
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump()) if len(mesh.dump()) > 0 else trimesh.Trimesh()
        elif HAS_OPEN3D:
            mesh = o3d.io.read_triangle_mesh(str(m_path))
        else:
            raise ImportError("Either trimesh or open3d is required.")

        # Determine object label / semantic category
        obj_info = objects_manifest.get(m_path.stem, {})
        label = obj_info.get("label", "")
        if not label:
            parts = m_path.stem.split("_")
            if len(parts) > 2:
                # Try joining last 2 parts first (e.g. 'wall_art' from 'obj_3_wall_art')
                candidate = "_".join(parts[-2:]).lower()
                label = candidate if candidate in config.WALL_MOUNTED_CLASSES else parts[-1].lower()
            elif len(parts) > 1:
                label = parts[-1].lower()
            else:
                label = m_path.stem.lower()

        is_wall_mounted = label.lower() in config.WALL_MOUNTED_CLASSES

        verts = np.asarray(mesh.vertices) if hasattr(mesh, "vertices") else np.zeros((0, 3))
        if len(verts) == 0:
            continue

        obj_min_y = float(verts[:, 1].min())
        obj_max_y = float(verts[:, 1].max())
        obj_min_x = float(verts[:, 0].min())
        obj_max_x = float(verts[:, 0].max())
        obj_min_z = float(verts[:, 2].min())
        obj_max_z = float(verts[:, 2].max())
        obj_center_x = (obj_min_x + obj_max_x) / 2.0
        obj_center_y = float(verts[:, 1].mean())
        obj_center_z = (obj_min_z + obj_max_z) / 2.0

        # Check if object is directly resting on a detected tabletop plane
        matching_table_y = None
        for tp in table_planes:
            t_y = float(tp.get("mean_y", 0.0))
            min_b = tp.get("min_bound", [-1e5, t_y, -1e5])
            max_b = tp.get("max_bound", [1e5, t_y, 1e5])

            # Check horizontal footprint overlap with margin
            margin_h = 0.25
            in_x = (min_b[0] - margin_h) <= obj_center_x <= (max_b[0] + margin_h)
            in_z = (min_b[2] - margin_h) <= obj_center_z <= (max_b[2] + margin_h)

            # Robust condition:
            # 1. Object bottom is reasonably near tabletop (>= t_y - 0.35m)
            # 2. Object center or top is at or above tabletop
            is_above_table = (obj_min_y >= t_y - 0.35) and ((obj_center_y >= t_y - 0.10) or (obj_max_y > t_y))
            if in_x and in_z and is_above_table:
                if matching_table_y is None or t_y > matching_table_y:
                    matching_table_y = t_y

        # If object is wall-mounted type (e.g. TV / monitor / picture):
        # If it is situated right on top of a table (like a desktop monitor or tabletop TV), snap to table.
        # Otherwise, snap to nearest vertical wall plane.
        is_mounted_on_wall = is_wall_mounted and (
            matching_table_y is None
            or obj_min_y > matching_table_y + 0.35
            or obj_center_y > matching_table_y + 0.85
        )
        if is_mounted_on_wall:
            # Wall-mounted object: Snap horizontally to nearest vertical wall plane, keep Y elevation
            if wall_planes:
                _cx, _cy, _cz = obj_center_x, obj_center_y, obj_center_z
                best_wall = min(
                    wall_planes,
                    key=lambda wp, cx=_cx, cy=_cy, cz=_cz: abs(
                        wp.get("equation", [0, 0, 1, 0])[0] * cx
                        + wp.get("equation", [0, 0, 1, 0])[1] * cy
                        + wp.get("equation", [0, 0, 1, 0])[2] * cz
                        + wp.get("equation", [0, 0, 1, 0])[3]
                    ),
                )
                snapped_mesh, delta_vec = snap_mesh_to_wall(mesh, best_wall, margin=config.WALL_SNAPPING_MARGIN)
                placement_type = "wall"
                target_surface = f"wall_id_{best_wall.get('id')}"
                delta_y = 0.0
            else:
                snapped_mesh = mesh
                delta_vec = [0.0, 0.0, 0.0]
                placement_type = "wall_unattached"
                target_surface = "none"
                delta_y = 0.0
        else:
            # Floor / Tabletop supported object: Snap vertically to surface beneath object
            target_surface_y = matching_table_y if matching_table_y is not None else floor_y
            snapped_mesh, delta_y = snap_mesh_to_surface(mesh, surface_y=target_surface_y)
            delta_vec = [0.0, delta_y, 0.0]
            placement_type = "table" if target_surface_y > (floor_y + 0.10) else "floor"
            target_surface = f"Y={target_surface_y:.3f}m"

        out_path = out_dir / m_path.name
        if HAS_TRIMESH and isinstance(snapped_mesh, trimesh.Trimesh):
            snapped_mesh.export(str(out_path))
        elif HAS_OPEN3D and isinstance(snapped_mesh, o3d.geometry.TriangleMesh):
            o3d.io.write_triangle_mesh(str(out_path), snapped_mesh)

        aligned_summary.append({
            "name": m_path.stem,
            "label": label,
            "placement_type": placement_type,
            "original_path": str(m_path),
            "aligned_path": str(out_path),
            "target_surface": target_surface,
            "delta_translation": delta_vec,
            "delta_y_applied": delta_y,
        })
        print(f"             - '{m_path.stem}' ({label}): [{placement_type}] -> Trans: {[round(v, 3) for v in delta_vec]} onto {target_surface}")

    manifest_out = out_dir / "aligned_objects_manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(aligned_summary, f, indent=2)
    print(f"[MeshPlacer] Aligned object placement manifest saved -> {manifest_out}")

    return aligned_summary


class MeshPlacer:
    """Class wrapper for Support Surface Snapping."""

    def __init__(self, objects_dir: Optional[Path | str] = None, plane_data_path: Optional[Path | str] = None):
        self.objects_dir = Path(objects_dir) if objects_dir else config.PROCESSED_DATA_DIR / "objects"
        self.plane_data_path = Path(plane_data_path) if plane_data_path else config.PROCESSED_DATA_DIR / "detected_planes.json"

    def run(self) -> List[Dict[str, Any]]:
        return align_and_place_object_meshes(self.objects_dir, self.plane_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Support Surface Snapping & Spatial Alignment")
    parser.add_argument("--objects-dir", type=str, default=str(config.PROCESSED_DATA_DIR / "objects"),
                        help="Input directory containing object 3D meshes")
    parser.add_argument("--planes-json", type=str, default=str(config.PROCESSED_DATA_DIR / "detected_planes.json"),
                        help="Input path for detected_planes.json")
    parser.add_argument("--out-dir", type=str, default=str(config.PROCESSED_DATA_DIR / "objects_aligned"),
                        help="Output directory for aligned & snapped object 3D meshes")
    args = parser.parse_args()

    align_and_place_object_meshes(
        objects_dir=args.objects_dir,
        plane_data_path=args.planes_json,
        out_dir=args.out_dir,
    )

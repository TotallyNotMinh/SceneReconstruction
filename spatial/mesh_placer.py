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


def align_and_place_object_meshes(
    objects_dir: Optional[Path | str] = None,
    plane_data_path: Optional[Path | str] = None,
    out_dir: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """Align and snap all object meshes onto detected architectural planes."""
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

    if plane_data_path.exists():
        with open(plane_data_path, "r", encoding="utf-8") as pf:
            planes_json = json.load(pf)
        floor = planes_json.get("floor")
        if floor:
            floor_y = float(floor.get("mean_y", 0.0))
        table_planes = planes_json.get("tables", [])
    else:
        print(f"[MeshPlacer] Plane metadata file not found at {plane_data_path}. Defaulting floor_y = 0.0m")

    # Find object mesh files (.ply, .obj)
    mesh_files = list(objects_dir.glob("*.ply")) + list(objects_dir.glob("*.obj"))
    if not mesh_files:
        print(f"[MeshPlacer] No object mesh files found in '{objects_dir}'. Creating a test object box...")
        test_mesh_path = objects_dir / "test_chair.ply"
        if HAS_TRIMESH:
            box = trimesh.creation.box(extents=[0.5, 0.8, 0.5])
            box.apply_translation([0.0, 0.6, 0.0])  # Floating at Y = 0.2m min_y
            box.export(str(test_mesh_path))
            mesh_files = [test_mesh_path]

    print(f"[MeshPlacer] Aligning & Snapping {len(mesh_files)} object meshes onto support planes...")
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

        # Determine target support plane
        target_surface_y = floor_y
        verts = np.asarray(mesh.vertices)
        if len(verts) > 0:
            obj_min_y = float(verts[:, 1].min())
            obj_min_x = float(verts[:, 0].min())
            obj_max_x = float(verts[:, 0].max())
            obj_min_z = float(verts[:, 2].min())
            obj_max_z = float(verts[:, 2].max())
            obj_center_x = (obj_min_x + obj_max_x) / 2.0
            obj_center_z = (obj_min_z + obj_max_z) / 2.0

            # Find valid support candidate planes (Floor + qualifying Tables beneath the object footprint)
            candidate_support_y = [floor_y]
            for tp in table_planes:
                t_y = float(tp.get("mean_y", 0.0))
                min_b = tp.get("min_bound", [-1e5, t_y, -1e5])
                max_b = tp.get("max_bound", [1e5, t_y, 1e5])

                # Check horizontal footprint overlap with margin
                margin_h = 0.20
                in_x = (min_b[0] - margin_h) <= obj_center_x <= (max_b[0] + margin_h)
                in_z = (min_b[2] - margin_h) <= obj_center_z <= (max_b[2] + margin_h)

                # Object base must be reasonably above the table surface
                if in_x and in_z and (obj_min_y >= t_y - 0.20):
                    candidate_support_y.append(t_y)

            # Pick the highest qualifying support surface beneath the object
            target_surface_y = max(candidate_support_y)

        snapped_mesh, delta_y = snap_mesh_to_surface(mesh, surface_y=target_surface_y)

        out_path = out_dir / m_path.name
        if HAS_TRIMESH and isinstance(snapped_mesh, trimesh.Trimesh):
            snapped_mesh.export(str(out_path))
        elif HAS_OPEN3D and isinstance(snapped_mesh, o3d.geometry.TriangleMesh):
            o3d.io.write_triangle_mesh(str(out_path), snapped_mesh)

        aligned_summary.append({
            "name": m_path.stem,
            "original_path": str(m_path),
            "aligned_path": str(out_path),
            "target_surface_y": target_surface_y,
            "delta_y_applied": delta_y,
        })
        print(f"             - '{m_path.stem}': Min Y shifted {delta_y:+.3f}m -> Snapped to surface at Y={target_surface_y:.3f}m")

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

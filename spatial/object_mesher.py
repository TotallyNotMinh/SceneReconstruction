# -*- coding: utf-8 -*-
"""
spatial/object_mesher.py — Phase 2B: 3D Object Surface Mesh Generation.

Reconstructs high-fidelity, watertight 3D surface meshes from individual object point clouds.
Supports point clouds produced by Phase 2A (spatial/object_extractor.py) or an intermediate
AI Point Cloud Completion step.

Reconstruction Algorithms:
- Screened Poisson Surface Reconstruction (watertight, smooth, with density trimming).
- Ball Pivoting Algorithm (BPA) with progressive ball radii.
- Alpha-Shape concave hull fitting.
- Automatic hole sealing (fill_mesh_holes) and non-shrinking Taubin smoothing.
- High-fidelity vertex color transfer from source point cloud.

Preserves natural world coordinates (X, Y, Z) so reconstructed meshes fit seamlessly into the full scene.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

from pointcloud.mesh_reconstructor import fill_mesh_holes, smooth_mesh_taubin, post_process_mesh


def reconstruct_object_mesh(
    pts: np.ndarray,
    colors: Optional[np.ndarray] = None,
    method: str = getattr(config, "OBJECT_MESHING_METHOD", "poisson"),
    alpha: Optional[float] = None,
    depth: Optional[int] = None,
    density_trim: Optional[float] = None,
    fill_holes: bool = getattr(config, "FILL_MESH_HOLES", True),
    smooth: bool = True,
    taubin_iterations: int = getattr(config, "MESH_TAUBIN_ITERATIONS", 12),
    out_path: Optional[Path | str] = None,
) -> Any:
    """
    Reconstruct a high-fidelity 3D surface mesh from an object point cloud.

    Parameters
    ----------
    pts : (N, 3) 3D point cloud coordinates in world space.
    colors : Optional (N, 3) RGB uint8 colors (0-255).
    method : Meshing algorithm: "poisson", "bpa", or "alpha".
    alpha : Alpha radius parameter for Alpha Shape meshing.
    depth : Octree depth for Poisson reconstruction (default config.OBJECT_POISSON_DEPTH).
    density_trim : Percentile of low-density vertices to trim (default config.OBJECT_POISSON_DENSITY_TRIM).
    fill_holes : Seal open boundary holes.
    smooth : Apply Taubin smoothing.
    taubin_iterations : Number of Taubin smoothing iterations.
    out_path : Optional file path to export reconstructed mesh (.ply or .obj).

    Returns
    -------
    trimesh.Trimesh or open3d.geometry.TriangleMesh
    """
    if len(pts) < 4:
        raise ValueError(f"[ObjectMesher] At least 4 points needed for 3D meshing, got {len(pts)}.")

    if depth is None:
        depth = getattr(config, "OBJECT_POISSON_DEPTH", 8)
    if density_trim is None:
        density_trim = getattr(config, "OBJECT_POISSON_DENSITY_TRIM", 3.0)

    mesh_o3d = None
    if HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if colors is not None and len(colors) == len(pts):
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors) / 255.0)

        # Estimate surface normals
        try:
            knn_val = min(30, max(4, len(pts) - 1))
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_val))
            pcd.orient_normals_consistent_tangent_plane(k=knn_val)
        except Exception:
            pass

        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = float(np.median(distances)) if len(distances) > 0 else 0.015
        avg_dist = max(avg_dist, 0.005)

        # 1. Screened Poisson Surface Reconstruction
        if method == "poisson":
            try:
                mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=depth, scale=1.1, linear_fit=False
                )
                densities = np.asarray(densities)
                if len(densities) > 0 and density_trim > 0:
                    density_thresh = np.percentile(densities, density_trim)
                    mesh_o3d.remove_vertices_by_mask(densities < density_thresh)
            except Exception as exc:
                print(f"[ObjectMesher] Poisson meshing failed ({exc}), falling back to BPA.")
                mesh_o3d = None

        # 2. Ball Pivoting Algorithm (BPA) Fallback
        if (mesh_o3d is None or len(mesh_o3d.triangles) < 4) and (method == "bpa" or mesh_o3d is None):
            try:
                radii_mult = getattr(config, "OBJECT_BPA_RADII_MULTIPLIER", [0.8, 1.5, 3.0, 6.0])
                radii = [avg_dist * m for m in radii_mult]
                mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )
            except Exception:
                mesh_o3d = None

        # 3. Adaptive Alpha Shape Fallback
        if mesh_o3d is None or len(mesh_o3d.triangles) < 4:
            try:
                effective_alpha = alpha if alpha is not None else max(getattr(config, "ALPHA_SHAPE_ALPHA", 0.04), avg_dist * 2.5)
                mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=effective_alpha)
            except Exception:
                mesh_o3d = None

        # Post-Processing: Clean topology, seal holes, smooth, and transfer vertex colors
        if mesh_o3d is not None and len(mesh_o3d.vertices) > 0 and len(mesh_o3d.triangles) > 0:
            mesh_o3d.remove_degenerate_triangles()
            mesh_o3d.remove_duplicated_triangles()
            mesh_o3d.remove_duplicated_vertices()
            mesh_o3d.remove_non_manifold_edges()

            mesh_o3d = post_process_mesh(
                mesh_o3d,
                fill_holes=fill_holes,
                smooth=smooth,
                iterations=taubin_iterations,
            )

            # High-fidelity vertex color transfer from original point cloud
            if colors is not None and len(mesh_o3d.vertices) > 0 and len(pts) == len(colors):
                mesh_verts = np.asarray(mesh_o3d.vertices)
                finite_v = np.all(np.isfinite(mesh_verts), axis=1)
                finite_pts = np.all(np.isfinite(pts), axis=1)
                if np.any(finite_pts) and np.any(finite_v):
                    from scipy.spatial import cKDTree
                    tree = cKDTree(pts[finite_pts])
                    valid_query_verts = np.where(np.isfinite(mesh_verts), mesh_verts, 0.0)
                    _, indices = tree.query(valid_query_verts, k=1)
                    v_cols = (colors[finite_pts])[indices] / 255.0
                    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(v_cols)

    # Export mesh if out_path is specified
    if mesh_o3d is not None and HAS_TRIMESH and out_path is not None and len(mesh_o3d.vertices) > 0 and len(mesh_o3d.triangles) > 0:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        verts = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        v_cols = (np.asarray(mesh_o3d.vertex_colors) * 255).astype(np.uint8) if mesh_o3d.has_vertex_colors() else colors
        tri = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=v_cols)
        tri.export(str(out_path))
        print(f"[ObjectMesher] Object 3D mesh saved ({method.upper()}) -> {out_path.name}")
        return tri

    return mesh_o3d


def reconstruct_object_meshes(
    objects_dir: Optional[Path | str] = None,
    manifest_path: Optional[Path | str] = None,
    method: Optional[str] = None,
    depth: Optional[int] = None,
    out_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Process all object point clouds in objects_dir and reconstruct 3D surface meshes.

    Parameters
    ----------
    objects_dir : Directory containing object point clouds (*_pointcloud.ply).
    manifest_path : Optional path to objects_manifest.json or extracted_objects_manifest.json.
    method : Meshing method ("poisson", "bpa", "alpha").
    depth : Octree depth for Poisson reconstruction.
    out_dir : Output directory for reconstructed meshes.

    Returns
    -------
    Dict mapping obj_id to updated metadata dictionary.
    """
    if objects_dir is None:
        objects_dir = config.PROCESSED_DATA_DIR / "objects"
    objects_dir = Path(objects_dir)

    if out_dir is None:
        out_dir = objects_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if method is None:
        method = getattr(config, "OBJECT_MESHING_METHOD", "poisson")

    if manifest_path is None:
        m_extracted = objects_dir / "extracted_objects_manifest.json"
        m_summary = objects_dir / "objects_manifest.json"
        manifest_path = m_extracted if m_extracted.exists() else m_summary
    else:
        manifest_path = Path(manifest_path)

    manifest_data: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            manifest_data = {}

    # Discover point cloud files
    pcd_items: List[Tuple[str, str, Path]] = []
    if manifest_data:
        for obj_id, obj_meta in manifest_data.items():
            label = obj_meta.get("label", "object")
            pcd_p = obj_meta.get("pcd_path")
            if pcd_p and Path(pcd_p).exists():
                pcd_items.append((obj_id, label, Path(pcd_p)))
            else:
                candidate1 = objects_dir / f"{obj_id}_{label}_pointcloud.ply"
                candidate2 = objects_dir / f"{obj_id}_pointcloud.ply"
                if candidate1.exists():
                    pcd_items.append((obj_id, label, candidate1))
                elif candidate2.exists():
                    pcd_items.append((obj_id, label, candidate2))

    if not pcd_items:
        raw_pcds = list(objects_dir.glob("*_pointcloud.ply"))
        for p in raw_pcds:
            stem = p.stem.replace("_pointcloud", "")
            parts = stem.split("_")
            if len(parts) >= 3 and parts[0] == "obj" and parts[1].isdigit():
                obj_id = f"{parts[0]}_{parts[1]}"
                label = "_".join(parts[2:])
            elif len(parts) >= 3 and parts[0] == "obj" and parts[-1] == parts[-2]:
                obj_id = "_".join(parts[:-1])
                label = parts[-1]
            elif len(parts) >= 2 and parts[0] == "obj":
                obj_id = stem
                label = "_".join(parts[1:])
            else:
                obj_id = stem
                label = parts[-1] if len(parts) > 1 else stem
            pcd_items.append((obj_id, label, p))

    if not pcd_items:
        print(f"[ObjectMesher] No object point clouds found in '{objects_dir}'. Nothing to reconstruct.")
        return {}

    print(f"[ObjectMesher] Found {len(pcd_items)} object point clouds for 3D surface meshing...")

    reconstructed_summary: Dict[str, Any] = dict(manifest_data)
    num_reconstructed = 0

    for obj_id, label, pcd_path in pcd_items:
        print(f"[ObjectMesher] Reconstructing mesh for '{obj_id}' ({label}) from '{pcd_path.name}'...")
        pts = None
        cols = None

        if HAS_TRIMESH:
            try:
                cloud = trimesh.load(str(pcd_path))
                if isinstance(cloud, trimesh.PointCloud):
                    pts = np.asarray(cloud.vertices, dtype=np.float64)
                    if hasattr(cloud, "colors") and cloud.colors is not None and len(cloud.colors) > 0:
                        cols = np.asarray(cloud.colors)[:, :3].astype(np.uint8)
                elif isinstance(cloud, trimesh.Trimesh):
                    pts = np.asarray(cloud.vertices, dtype=np.float64)
                    if hasattr(cloud.visual, "vertex_colors") and cloud.visual.vertex_colors is not None:
                        cols = np.asarray(cloud.visual.vertex_colors)[:, :3].astype(np.uint8)
            except Exception:
                pts = None

        if (pts is None or len(pts) == 0) and HAS_OPEN3D:
            try:
                pcd_o3d = o3d.io.read_point_cloud(str(pcd_path))
                if len(pcd_o3d.points) > 0:
                    pts = np.asarray(pcd_o3d.points, dtype=np.float64)
                    cols = (np.asarray(pcd_o3d.colors) * 255).astype(np.uint8) if pcd_o3d.has_colors() else None
            except Exception:
                pass

        if pts is None or len(pts) < 4:
            print(f"[ObjectMesher] WARNING: Object '{obj_id}' point cloud has insufficient points ({len(pts) if pts is not None else 0}); skipping.")
            continue

        out_mesh_path = out_dir / f"{obj_id}_{label}.ply"
        mesh = reconstruct_object_mesh(
            pts,
            colors=cols,
            method=method,
            depth=depth,
            out_path=out_mesh_path,
        )

        n_verts = len(mesh.vertices) if hasattr(mesh, "vertices") else 0
        n_faces = len(mesh.faces) if hasattr(mesh, "faces") else (len(mesh.triangles) if hasattr(mesh, "triangles") else 0)

        meta = reconstructed_summary.get(obj_id, {})
        meta.update({
            "label": label,
            "mesh_path": str(out_mesh_path),
            "pcd_path": str(pcd_path),
            "point_count": len(pts),
            "vertex_count": n_verts,
            "face_count": n_faces,
            "bounds_min": pts.min(axis=0).tolist(),
            "bounds_max": pts.max(axis=0).tolist(),
            "centroid": pts.mean(axis=0).tolist(),
        })
        reconstructed_summary[obj_id] = meta
        num_reconstructed += 1

    # Save summary manifest
    summary_path = out_dir / "objects_manifest.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(reconstructed_summary, f, indent=2)
    print(f"[ObjectMesher] Successfully reconstructed {num_reconstructed} object meshes ({len(reconstructed_summary)} total registered) -> {summary_path}")

    return reconstructed_summary


class ObjectMesher:
    """Class wrapper for 3D Object Surface Mesh Reconstruction."""

    def __init__(
        self,
        objects_dir: Optional[Path | str] = None,
        manifest_path: Optional[Path | str] = None,
        method: Optional[str] = None,
        depth: Optional[int] = None,
        out_dir: Optional[Path | str] = None,
    ):
        self.objects_dir = Path(objects_dir) if objects_dir else config.PROCESSED_DATA_DIR / "objects"
        self.manifest_path = Path(manifest_path) if manifest_path else self.objects_dir / "objects_manifest.json"
        self.method = method or getattr(config, "OBJECT_MESHING_METHOD", "poisson")
        self.depth = depth
        self.out_dir = Path(out_dir) if out_dir else self.objects_dir

    def run(self) -> Dict[str, Any]:
        return reconstruct_object_meshes(
            objects_dir=self.objects_dir,
            manifest_path=self.manifest_path,
            method=self.method,
            depth=self.depth,
            out_dir=self.out_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2B: 3D Object Surface Mesh Generation")
    parser.add_argument("--objects-dir", type=str, default=str(config.PROCESSED_DATA_DIR / "objects"),
                        help="Directory containing object point clouds and manifest")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to objects manifest JSON file")
    parser.add_argument("--method", type=str, default=config.OBJECT_MESHING_METHOD,
                        choices=["poisson", "bpa", "alpha"],
                        help="3D surface meshing algorithm")
    parser.add_argument("--depth", type=int, default=config.OBJECT_POISSON_DEPTH,
                        help="Octree depth for Poisson reconstruction")
    parser.add_argument("--out-dir", type=str, default=str(config.PROCESSED_DATA_DIR / "objects"),
                        help="Output directory for reconstructed object meshes")
    args = parser.parse_args()

    reconstruct_object_meshes(
        objects_dir=args.objects_dir,
        manifest_path=args.manifest,
        method=args.method,
        depth=args.depth,
        out_dir=args.out_dir,
    )

# -*- coding: utf-8 -*-
"""
pointcloud/mesh_reconstructor.py — Surface Reconstruction and Voxelization from Point Clouds.

Supports:
- Poisson Surface Reconstruction (watertight, smooth, color-interpolating)
- Ball Pivoting Algorithm (BPA) (exact point-preserving triangulation)
- Alpha Shape (concave bounding hull)
- Voxel Grid Meshing (converts point cloud into 3D voxel box meshes)
"""

import sys
import argparse
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, Any
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False

try:
    import trimesh
    _TRIMESH_AVAILABLE = True
except ImportError:
    _TRIMESH_AVAILABLE = False


def _fill_trimesh_holes_robust(tri_mesh: Any) -> Any:
    """Robust pure NumPy boundary loop closure for trimesh objects."""
    if not hasattr(tri_mesh, "edges_sorted") or len(tri_mesh.faces) == 0:
        return tri_mesh
    edges = tri_mesh.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique[counts == 1]
    if len(boundary_edges) == 0:
        return tri_mesh

    adj: dict[int, list[int]] = {}
    for u, v in boundary_edges:
        adj.setdefault(int(u), []).append(int(v))
        adj.setdefault(int(v), []).append(int(u))

    visited = set()
    new_faces = []
    for start_node in adj:
        if start_node in visited:
            continue
        cycle = [start_node]
        visited.add(start_node)
        curr = start_node
        prev = None
        while True:
            nbrs = [n for n in adj[curr] if n != prev]
            if not nbrs:
                break
            next_node = nbrs[0]
            if next_node == start_node:
                break
            if next_node in visited:
                break
            visited.add(next_node)
            cycle.append(next_node)
            prev = curr
            curr = next_node

        if len(cycle) == 3:
            new_faces.append(cycle)
        elif len(cycle) == 4:
            new_faces.append([cycle[0], cycle[1], cycle[2]])
            new_faces.append([cycle[0], cycle[2], cycle[3]])
        elif 4 < len(cycle) <= 24:
            for i in range(1, len(cycle) - 1):
                new_faces.append([cycle[0], cycle[i], cycle[i+1]])

    if new_faces:
        all_faces = np.vstack([tri_mesh.faces, np.array(new_faces)])
        v_cols = tri_mesh.visual.vertex_colors if (hasattr(tri_mesh, "visual") and hasattr(tri_mesh.visual, "vertex_colors")) else None
        return trimesh.Trimesh(vertices=tri_mesh.vertices, faces=all_faces, vertex_colors=v_cols, process=False)
    return tri_mesh


def fill_mesh_holes(mesh: Any) -> Any:
    """Detect and seal open boundary loops and triangular holes on a 3D mesh."""
    if _TRIMESH_AVAILABLE and isinstance(mesh, trimesh.Trimesh):
        try:
            return _fill_trimesh_holes_robust(mesh)
        except Exception:
            return mesh

    if _O3D_AVAILABLE and isinstance(mesh, o3d.geometry.TriangleMesh):
        try:
            if _TRIMESH_AVAILABLE:
                verts = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                cols = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) if mesh.has_vertex_colors() else None
                tri = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=cols, process=False)
                tri = _fill_trimesh_holes_robust(tri)
                out_o3d = o3d.geometry.TriangleMesh()
                out_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(tri.vertices))
                out_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(tri.faces))
                if tri.visual.vertex_colors is not None and len(tri.visual.vertex_colors) == len(tri.vertices):
                    out_o3d.vertex_colors = o3d.utility.Vector3dVector(np.asarray(tri.visual.vertex_colors)[:, :3] / 255.0)
                return out_o3d
        except Exception:
            pass
    return mesh


def smooth_mesh_taubin(
    mesh: Any,
    iterations: int = getattr(config, "MESH_TAUBIN_ITERATIONS", 12),
    lambda_filter: float = 0.5,
    mu: float = -0.53,
) -> Any:
    """Apply non-shrinking Taubin smoothing to eliminate depth surface noise and roughness."""
    if iterations <= 0:
        return mesh

    if _O3D_AVAILABLE and isinstance(mesh, o3d.geometry.TriangleMesh):
        try:
            if len(mesh.vertices) >= 4 and len(mesh.triangles) >= 4:
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
                smoothed = mesh.filter_smooth_taubin(
                    number_of_iterations=iterations,
                    lambda_filter=lambda_filter,
                    mu=mu,
                )
                s_verts = np.asarray(smoothed.vertices)
                if len(s_verts) > 0 and np.all(np.isfinite(s_verts)):
                    smoothed.compute_vertex_normals()
                    return smoothed
        except Exception:
            return mesh

    if _TRIMESH_AVAILABLE and isinstance(mesh, trimesh.Trimesh):
        try:
            if _O3D_AVAILABLE and len(mesh.vertices) >= 4 and len(mesh.faces) >= 4:
                o3d_m = o3d.geometry.TriangleMesh()
                o3d_m.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
                o3d_m.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
                o3d_m.remove_degenerate_triangles()
                o3d_m.remove_duplicated_triangles()
                o3d_m.remove_duplicated_vertices()
                o3d_m.remove_non_manifold_edges()
                smoothed = o3d_m.filter_smooth_taubin(
                    number_of_iterations=iterations,
                    lambda_filter=lambda_filter,
                    mu=mu,
                )
                s_verts = np.asarray(smoothed.vertices)
                if len(s_verts) > 0 and np.all(np.isfinite(s_verts)):
                    v_cols = mesh.visual.vertex_colors if (hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors")) else None
                    return trimesh.Trimesh(
                        vertices=s_verts,
                        faces=np.asarray(smoothed.triangles),
                        vertex_colors=v_cols,
                        process=False,
                    )
        except Exception:
            return mesh

    return mesh


def post_process_mesh(
    mesh: Any,
    fill_holes: bool = getattr(config, "FILL_MESH_HOLES", True),
    smooth: bool = True,
    iterations: int = getattr(config, "MESH_TAUBIN_ITERATIONS", 12),
) -> Any:
    """Execute complete post-processing pipeline on 3D mesh."""
    if mesh is None:
        return None

    if fill_holes:
        mesh = fill_mesh_holes(mesh)

    if smooth and iterations > 0:
        mesh = smooth_mesh_taubin(mesh, iterations=iterations)

    return mesh


def mesh_pointcloud(
    ply_input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    method: Literal["poisson", "bpa", "alpha", "voxel"] = "poisson",
    depth: int = 9,
    density_trim_percentile: float = 5.0,
    alpha: float = 0.05,
    voxel_size: float = 0.04,
    estimate_normals_knn: int = 30,
) -> Union["trimesh.Trimesh", "o3d.geometry.TriangleMesh"]:
    """
    Load a point cloud (.ply) and reconstruct a 3D surface mesh or voxel grid.

    Args:
        ply_input_path: Path to input point cloud (.ply) file.
        output_path: Optional path to save reconstructed mesh (.ply, .obj, .glb).
        method: Reconstruction method:
            - "poisson": Screened Poisson Reconstruction (smooth, watertight, fills holes)
            - "bpa": Ball Pivoting Algorithm (exact triangulation, no hole filling)
            - "alpha": Alpha Shapes (concave geometric hull)
            - "voxel": Voxel grid of solid 3D cubes with average colors
        depth: Poisson octree tree depth (higher = finer detail, default: 9, max recommended: 11).
        density_trim_percentile: Prune low-density surface floaters in Poisson (e.g. 5.0 = trim bottom 5%).
        alpha: Alpha value for Alpha Shapes (m).
        voxel_size: Voxel size in meters for voxel meshing (default: 0.04m / 4cm).
        estimate_normals_knn: Nearest neighbors for normal estimation.

    Returns:
        trimesh.Trimesh or open3d.geometry.TriangleMesh instance.
    """
    if not _O3D_AVAILABLE and not _TRIMESH_AVAILABLE:
        raise ImportError("Either open3d or trimesh is required for meshing. Install via: pip install open3d trimesh")

    ply_input_path = Path(ply_input_path)
    if not ply_input_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {ply_input_path}")

    print(f"[+] Loading point cloud from: {ply_input_path.name}")
    if _O3D_AVAILABLE:
        pcd = o3d.io.read_point_cloud(str(ply_input_path))
        num_pts = len(pcd.points)
        if num_pts == 0:
            raise ValueError(f"Point cloud at {ply_input_path} contains 0 points.")
        pts = np.asarray(pcd.points)
        cols = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
    elif _TRIMESH_AVAILABLE:
        cloud = trimesh.load(str(ply_input_path))
        if isinstance(cloud, trimesh.Scene):
            raise ValueError(f"Unexpected trimesh type for PLY: {type(cloud)}")
        pts = np.asarray(cloud.vertices)
        num_pts = len(pts)
        if num_pts == 0:
            raise ValueError(f"Point cloud at {ply_input_path} contains 0 points.")
        cols = np.asarray(cloud.visual.vertex_colors)[:, :3] if (hasattr(cloud, "visual") and hasattr(cloud.visual, "vertex_colors") and len(cloud.visual.vertex_colors) > 0) else None
        pcd = None
    else:
        raise ImportError("Either open3d or trimesh is required. Install with: pip install open3d trimesh")

    print(f"    Total points loaded: {num_pts:,}")

    if method in ("poisson", "bpa", "alpha") and not _O3D_AVAILABLE:
        raise ImportError(f"Method '{method}' requires open3d. Install with: pip install open3d or activate your (.venv)")

    # Ensure point normals exist for normal-based methods
    if method in ("poisson", "bpa") and pcd is not None:
        if not pcd.has_normals():
            print(f"[+] Estimating surface normals (k={estimate_normals_knn})...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=estimate_normals_knn)
            )
            # Orient normals consistently
            pcd.orient_normals_consistent_tangent_plane(k=estimate_normals_knn)

    mesh_o3d: Optional[o3d.geometry.TriangleMesh] = None

    if method == "poisson":
        print(f"[+] Running Screened Poisson Surface Reconstruction (octree depth={depth})...")
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )
        densities = np.asarray(densities)

        # Trim low-density surface artifacts / phantom boundary hulls
        if density_trim_percentile > 0.0 and len(densities) > 0:
            density_threshold = np.percentile(densities, density_trim_percentile)
            vertices_to_remove = densities < density_threshold
            mesh_o3d.remove_vertices_by_mask(vertices_to_remove)
            print(f"    Trimmed {vertices_to_remove.sum():,} low-density vertices (< {density_trim_percentile}th percentile)")

    elif method == "bpa":
        print("[+] Running Ball Pivoting Algorithm (BPA)...")
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.median(distances)
        radii = [avg_dist, avg_dist * 2.0, avg_dist * 4.0]
        print(f"    Ball radii: {[round(r, 4) for r in radii]}m")
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

    elif method == "alpha":
        print(f"[+] Running Alpha Shape Reconstruction (alpha={alpha:.3f}m)...")
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha=alpha
        )

    elif method == "voxel":
        print(f"[+] Voxelizing point cloud (voxel_size={voxel_size:.3f}m)...")
        # Voxel indexing
        voxel_coords = np.floor(pts / voxel_size).astype(np.int64)
        unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
        m_voxels = len(unique_voxels)
        centers = (unique_voxels + 0.5) * voxel_size
        s = voxel_size * 0.98

        # 8 base box vertices & 12 triangular faces
        base_v = np.array([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5], [0.5, 0.5,  0.5], [-0.5, 0.5,  0.5],
        ], dtype=np.float64) * s

        base_f = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ], dtype=np.int64)

        all_verts = (base_v[None, :, :] + centers[:, None, :]).reshape(-1, 3)
        all_faces = (base_f[None, :, :] + (np.arange(m_voxels, dtype=np.int64) * 8)[:, None, None]).reshape(-1, 3)

        if cols is not None:
            # Vectorized color aggregation per unique voxel
            counts = np.bincount(inverse_indices, minlength=m_voxels)[:, None]
            r_sum = np.bincount(inverse_indices, weights=cols[:, 0], minlength=m_voxels)[:, None]
            g_sum = np.bincount(inverse_indices, weights=cols[:, 1], minlength=m_voxels)[:, None]
            b_sum = np.bincount(inverse_indices, weights=cols[:, 2], minlength=m_voxels)[:, None]
            avg_cols = np.hstack([r_sum / np.maximum(counts, 1),
                                  g_sum / np.maximum(counts, 1),
                                  b_sum / np.maximum(counts, 1)]).astype(np.uint8)
            all_v_cols = np.repeat(avg_cols, 8, axis=0)
        else:
            all_v_cols = None

        print(f"    Created {m_voxels:,} solid 3D colored voxels")
        tri_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, vertex_colors=all_v_cols)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tri_mesh.export(str(output_path))
            print(f"[+] Colored voxel mesh saved -> {output_path}")

        return tri_mesh

    # Clean and post-process Open3D mesh
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_non_manifold_edges()

    # Apply non-shrinking Taubin smoothing and hole sealing
    mesh_o3d = post_process_mesh(mesh_o3d, fill_holes=getattr(config, "FILL_MESH_HOLES", True), smooth=True)

    # Preserve and transfer RGB vertex colors from input point cloud onto mesh surface
    if pcd.has_colors() and len(mesh_o3d.vertices) > 0:
        print("[+] Preserving point cloud RGB colors onto mesh surface...")
        from scipy.spatial import cKDTree
        pcd_pts = np.asarray(pcd.points)
        pcd_cols = np.asarray(pcd.colors)  # shape (N, 3), range [0.0, 1.0]
        mesh_verts = np.asarray(mesh_o3d.vertices)

        k = min(4, len(pcd_pts))
        tree = cKDTree(pcd_pts)
        dists, indices = tree.query(mesh_verts, k=k)

        if k == 1 or dists.ndim == 1:
            mesh_cols = pcd_cols[indices]
        else:
            weights = 1.0 / np.maximum(dists, 1e-6)
            weights /= weights.sum(axis=1, keepdims=True)
            mesh_cols = (pcd_cols[indices] * weights[:, :, None]).sum(axis=1)

        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(np.clip(mesh_cols, 0.0, 1.0))
        print(f"    RGB colors smoothly mapped onto {len(mesh_cols):,} mesh vertices.")

    print(f"[+] Surface mesh ready: {len(mesh_o3d.vertices):,} vertices, {len(mesh_o3d.triangles):,} faces")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ext = output_path.suffix.lower()

        # For GLB/GLTF/OBJ with vertex colors, use Trimesh for universal 3D viewer compatibility
        if _TRIMESH_AVAILABLE and ext in (".glb", ".gltf", ".obj"):
            verts = np.asarray(mesh_o3d.vertices)
            faces = np.asarray(mesh_o3d.triangles)
            cols = (np.asarray(mesh_o3d.vertex_colors) * 255).astype(np.uint8) if mesh_o3d.has_vertex_colors() else None
            tri = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=cols)
            tri.export(str(output_path))
        else:
            o3d.io.write_triangle_mesh(str(output_path), mesh_o3d, write_vertex_colors=True)
        print(f"[+] Colored mesh saved -> {output_path}")

    return mesh_o3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct 3D Surface Mesh or Voxel Grid from Point Cloud")
    parser.add_argument("pcd_path", type=str, nargs="?", default=str(config.OUTPUT_DIR / "world_pointcloud.ply"),
                        help="Input .ply point cloud file (default: data/output/world_pointcloud.ply)")
    parser.add_argument("--out", "--output", type=str, default=str(config.OUTPUT_DIR / "scene_mesh.ply"),
                        dest="out", help="Output mesh path (.ply, .obj, .glb)")
    parser.add_argument("--method", type=str, default="poisson", choices=["poisson", "bpa", "alpha", "voxel"],
                        help="Reconstruction algorithm (poisson, bpa, alpha, voxel)")
    parser.add_argument("--depth", type=int, default=9,
                        help="Octree depth for Poisson reconstruction (default: 9, 8=fast, 10=fine)")
    parser.add_argument("--trim", type=float, default=5.0,
                        help="Density trim percentile for Poisson (default: 5.0%% to remove boundary floaters)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Alpha radius in meters for alpha shape (default: 0.05)")
    parser.add_argument("--voxel-size", type=float, default=0.04,
                        help="Voxel size in meters for voxel grid meshing (default: 0.04)")

    args = parser.parse_args()

    mesh_pointcloud(
        ply_input_path=args.pcd_path,
        output_path=args.out,
        method=args.method,
        depth=args.depth,
        density_trim_percentile=args.trim,
        alpha=args.alpha,
        voxel_size=args.voxel_size,
    )

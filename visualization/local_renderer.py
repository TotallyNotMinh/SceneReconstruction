# -*- coding: utf-8 -*-
"""
visualization/local_renderer.py — Local 3D Scene Viewer & 360-Degree Video Renderer.

Features:
1. Interactive 3D Viewer: Real-time orbit, zoom, pan for point clouds (.ply) and meshes (.ply, .obj, .glb).
2. 360° Cinematic Orbit Video: Renders smooth 360-degree turntable orbit MP4 videos.
3. Trajectory Visualizer: Overlays predicted camera frustums and trajectory paths.
"""

import sys
import math
import argparse
from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import cv2
from tqdm import tqdm

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


def load_geometry(file_path: Union[str, Path]):
    """
    Load a 3D geometry file (.ply, .obj, .glb, .gltf) as an Open3D or Trimesh geometry.
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"3D file not found: {file_path}")

    ext = file_path.suffix.lower()

    if _O3D_AVAILABLE:
        # Try loading as triangle mesh first
        if ext in (".ply", ".obj", ".stl", ".off", ".gltf", ".glb"):
            try:
                mesh = o3d.io.read_triangle_mesh(str(file_path))
                mesh.remove_non_finite_vertices()
                if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
                    mesh.compute_vertex_normals()
                    return mesh, "mesh"
            except Exception:
                pass

        # Try loading as point cloud
        try:
            pcd = o3d.io.read_point_cloud(str(file_path))
            pcd.remove_non_finite_points(remove_nan=True, remove_infinite=True)
            if len(pcd.points) > 0:
                return pcd, "pointcloud"
        except Exception:
            pass

    if _TRIMESH_AVAILABLE:
        try:
            geo = trimesh.load(str(file_path))
            return geo, "trimesh"
        except Exception as e:
            raise RuntimeError(f"Trimesh failed to load {file_path.name}: {e}")

    raise RuntimeError(f"Failed to load valid 3D geometry from {file_path}. The file may be empty or corrupted.")


def view_scene_interactive(
    geometry_path: Union[str, Path],
    window_name: str = "3D Scene Reconstruction Viewer",
    point_size: float = 3.0,
    background_color: tuple = (0.1, 0.1, 0.12),
):
    """
    Launch a local interactive 3D window to inspect point clouds and meshes.
    Controls:
      - Left Click + Drag: Rotate
      - Right Click + Drag: Pan
      - Mouse Wheel: Zoom
      - Shift + Left Click: Roll
      - Key 'Q' or 'ESC': Close
    """
    geometry_path = Path(geometry_path)
    print(f"[+] Opening interactive 3D viewer for: {geometry_path.name}")
    print("    [Controls] Left-click: Rotate | Right-click: Pan | Scroll: Zoom | Q: Exit")

    if _O3D_AVAILABLE:
        geom, geom_type = load_geometry(geometry_path)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"{window_name} — {geometry_path.name}", width=1280, height=720, visible=True)
        vis.add_geometry(geom)

        opt = vis.get_render_option()
        opt.background_color = np.asarray(background_color)
        opt.point_size = float(point_size)
        opt.mesh_show_back_face = True
        if geom_type == "mesh":
            opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color

        vis.run()
        vis.destroy_window()
    elif _TRIMESH_AVAILABLE:
        geo, _ = load_geometry(geometry_path)
        try:
            if isinstance(geo, trimesh.Scene):
                geo.show()
            else:
                scene = trimesh.Scene(geo)
                scene.show()
        except ImportError as e:
            if "pyglet" in str(e).lower():
                raise ImportError(f"[ERROR] Trimesh requires an older version of pyglet for the interactive viewer.\nRun: pip install \"pyglet<2\"") from e
            raise
    else:
        raise ImportError("Please install open3d or trimesh to use interactive viewer.")


def render_360_orbit_video(
    geometry_path: Union[str, Path],
    output_video_path: Optional[Union[str, Path]] = None,
    num_frames: int = 180,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    point_size: float = 3.5,
    elevation_angle_deg: float = 20.0,
    background_color: tuple = (0.08, 0.08, 0.10),
    radius_scale: float = 1.6,
) -> Path:
    """
    Render a smooth 360-degree turntable orbit MP4 video around the 3D scene.
    """
    if not _O3D_AVAILABLE:
        raise ImportError("open3d is required for 360 orbit video rendering. Install with: pip install open3d")

    geometry_path = Path(geometry_path)
    if output_video_path is None:
        output_video_path = config.OUTPUT_DIR / f"{geometry_path.stem}_360.mp4"
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    geom, geom_type = load_geometry(geometry_path)
    if geom_type == "trimesh":
        raise RuntimeError(f"File {geometry_path.name} was loaded via Trimesh, but 360-orbit rendering strictly requires Open3D. Ensure the file format is supported by Open3D (.ply, .obj, .glb).")
    
    center = geom.get_center()
    bbox = geom.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    max_dim = max(extent)
    radius = max_dim * radius_scale

    print(f"[+] Setting up 360° orbit rendering ({width}x{height} @ {fps} FPS, {num_frames} frames)...")
    print(f"    Target geometry: {geometry_path.name} ({geom_type})")
    print(f"    Bounding center: {[round(c, 2) for c in center]}, extent: {[round(e, 2) for e in extent]}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Offscreen Renderer", width=width, height=height, visible=False)
    vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color)
    opt.point_size = float(point_size)
    opt.mesh_show_back_face = True
    if geom_type == "mesh":
        opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    ctr = vis.get_view_control()
    elev_rad = math.radians(elevation_angle_deg)

    pbar = tqdm(total=num_frames, desc="[Rendering 360° Video]", unit="frame")
    for i in range(num_frames):
        theta = 2.0 * math.pi * (i / num_frames)
        eye_x = center[0] + radius * math.cos(theta) * math.cos(elev_rad)
        eye_z = center[2] + radius * math.sin(theta) * math.cos(elev_rad)
        eye_y = center[1] + radius * math.sin(elev_rad)

        eye = np.array([eye_x, eye_y, eye_z])
        up = np.array([0.0, 1.0, 0.0])

        ctr.set_lookat(center)
        ctr.set_front(center - eye)
        ctr.set_up(up)
        ctr.set_zoom(0.7)

        vis.poll_events()
        vis.update_renderer()

        # Capture RGB image
        img_buffer = vis.capture_screen_float_buffer(do_render=True)
        img_rgb = (np.asarray(img_buffer) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        writer.write(img_bgr)
        pbar.update(1)

    pbar.close()
    writer.release()
    vis.destroy_window()

    print(f"[+] OK 360° orbit video saved -> {output_video_path}")
    return output_video_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local 3D Scene Viewer & 360 Orbit Video Renderer")
    parser.add_argument("geometry", type=str, nargs="?",
                        default=str(config.OUTPUT_DIR / "world_pointcloud.ply"),
                        help="Path to 3D point cloud or mesh (.ply, .obj, .glb)")
    parser.add_argument("--mode", type=str, default="view", choices=["view", "video"],
                        help="'view' for interactive 3D window, 'video' for 360 orbit MP4")
    parser.add_argument("--out", "--output", type=str, default=None, dest="out",
                        help="Output video file path (.mp4)")
    parser.add_argument("--frames", type=int, default=180,
                        help="Total frames for 360 orbit video (default: 180 = 6s at 30fps)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video frame rate (default: 30)")
    parser.add_argument("--res", type=str, default="1280x720",
                        help="Render resolution WxH (default: 1280x720)")
    parser.add_argument("--point-size", type=float, default=3.0,
                        help="Render point size for point clouds (default: 3.0)")
    parser.add_argument("--elevation", type=float, default=20.0,
                        help="Camera elevation angle in degrees for orbit video (default: 20)")

    args = parser.parse_args()

    w, h = map(int, args.res.split("x")) if "x" in args.res else (1280, 720)

    if args.mode == "view":
        view_scene_interactive(
            geometry_path=args.geometry,
            point_size=args.point_size,
        )
    elif args.mode == "video":
        render_360_orbit_video(
            geometry_path=args.geometry,
            output_video_path=args.out,
            num_frames=args.frames,
            fps=args.fps,
            width=w,
            height=h,
            point_size=args.point_size,
            elevation_angle_deg=args.elevation,
        )

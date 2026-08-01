import os
import sys
import glob
import math
import numpy as np
import cv2
import open3d as o3d

def create_360_trajectory(center, radius, num_frames, height_offset=0.2, look_at_offset=[0, 0, 0]):
    """
    Generate a sequence of 4x4 Extrinsic Matrices orbiting 360 degrees around a center point.
    """
    poses = []
    center = np.array(center)
    target = center + np.array(look_at_offset)
    up = np.array([0.0, 0.0, 1.0])  # Z-up coordinate system for PLY meshes

    for i in range(num_frames):
        angle = 2.0 * math.pi * i / num_frames
        
        # Calculate camera eye position on a circle
        eye_x = center[0] + radius * math.cos(angle)
        eye_y = center[1] + radius * math.sin(angle)
        eye_z = center[2] + height_offset
        eye = np.array([eye_x, eye_y, eye_z])

        # Compute camera coordinate axes (Forward, Right, Up)
        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / np.linalg.norm(right)

        true_up = np.cross(right, forward)
        true_up = true_up / np.linalg.norm(true_up)

        # Build 4x4 Extrinsic matrix (World to Camera transform)
        R = np.vstack([right, true_up, -forward])  # 3x3 rotation
        t = -R @ eye                               # 3x1 translation
        
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        poses.append((eye, target, true_up))

    return poses

def render_replica_360(ply_path, output_video_path="replica_360_orbit.mp4", width=1280, height=720, num_frames=120, fps=30):
    """
    Load a Replica mesh.ply, set up an offscreen 360-degree orbit renderer, 
    and export frames directly into an MP4 video file.
    """
    if not os.path.exists(ply_path):
        print(f"[-] Mesh file not found: {ply_path}")
        return

    print(f"[+] Loading 3D mesh: {ply_path} ...")
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()

    # Compute bounding box to determine camera orbit center and distance
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = max(extent[:2]) * 0.8  # Orbit radius proportional to scene size

    print(f"[+] Scene Center: {center}, Radius: {radius:.2f}m")

    # Set up Open3D Visualizer Offscreen Renderer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Replica 360 Renderer", width=width, height=height, visible=False)
    vis.add_geometry(mesh)

    # Render Options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.light_on = True

    ctr = vis.get_view_control()

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"[+] Rendering {num_frames} frames 360-degree orbit video...")
    
    poses = create_360_trajectory(center, radius, num_frames)

    for i, (eye, target, up) in enumerate(poses):
        # Update camera position & orientation
        ctr.set_lookat(target)
        ctr.set_front((eye - target) / np.linalg.norm(eye - target))
        ctr.set_up(up)
        ctr.set_zoom(0.45)

        vis.poll_events()
        vis.update_renderer()

        # Capture RGB image buffer
        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = (np.asarray(image) * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        video_writer.write(image_bgr)

        if (i + 1) % 30 == 0 or (i + 1) == num_frames:
            print(f"  - Rendered frame {i + 1}/{num_frames}")

    video_writer.release()
    vis.destroy_window()
    print(f"\n[SUCCESS] 360 Video generated and saved to: {output_video_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ply_file = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_mesh = os.path.join(script_dir, "..", "Replica-Dataset", "room_0", "mesh.ply")
        ply_file = default_mesh if os.path.exists(default_mesh) else os.path.join("Replica-Dataset", "room_0", "mesh.ply")

    output_mp4 = "replica_room0_360.mp4"
    render_replica_360(ply_file, output_mp4)

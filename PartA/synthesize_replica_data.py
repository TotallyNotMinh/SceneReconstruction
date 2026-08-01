"""
synthesize_replica_data.py

Synthesizes all data required to run PartA/main_pipeline.py from a
downloaded Replica/ReplicaCAD scene.

Given a Replica scene mesh.ply, this script generates:
  1. ar_metadata.json  -- Simulated ARKit camera poses along a smooth
                          orbit trajectory around the scene center.
  2. world_pointcloud.ply -- Sampled 3D point cloud from the mesh surface.
  3. room_layout.obj   -- Clean floor slab extracted by Y-height heuristics.
  4. detections_from_b.json -- Synthetic 2D bounding box detections for
                               furniture objects, projected from 3D clusters
                               using the generated camera poses.
  5. meshes_from_c/<obj_id>.obj -- One representative proxy mesh per
                                   detected furniture object.

Usage:
    python PartA/synthesize_replica_data.py [path/to/replica/room_X/mesh.ply]

If no path is given, it scans Replica-Dataset/ for the first available mesh.ply.

Requirements: numpy, open3d, trimesh, scikit-learn
"""

import os
import sys
import json
import math
import numpy as np
from pathlib import Path

# -- Lazy imports ---------------------------------------------------------------
try:
    import open3d as o3d
except ImportError:
    sys.exit("[ERROR] open3d not installed.  Run: pip install open3d")

try:
    import trimesh
except ImportError:
    sys.exit("[ERROR] trimesh not installed.  Run: pip install trimesh")

try:
    from sklearn.cluster import DBSCAN as _DBSCAN
except ImportError:
    sys.exit("[ERROR] scikit-learn not installed.  Run: pip install scikit-learn")

# -- Paths ----------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_DIR     = SCRIPT_DIR / "data"
MESH_DIR     = DATA_DIR  / "meshes_from_c"
REPLICA_ROOT = SCRIPT_DIR.parent / "Replica-Dataset"

# -- Hyperparameters ------------------------------------------------------------
NUM_ORBIT_FRAMES   = 120        # synthetic camera frames
ORBIT_RADIUS_SCALE = 0.6        # orbit radius as fraction of scene max-extent
ORBIT_HEIGHT_ABOVE = 0.8        # camera height above floor (m)
FX = FY            = 1420.5     # focal length px  (iPhone 13 Pro approx)
IMG_W, IMG_H       = 1920, 1440
CX, CY             = IMG_W / 2.0, IMG_H / 2.0
POINT_CLOUD_SAMPLE = 80_000
FLOOR_THICKNESS    = 0.12       # m
MIN_FURNITURE_PTS  = 200        # min pts per cluster to keep

# ==============================================================================
# Helpers
# ==============================================================================

def find_replica_mesh(hint=None):
    if hint:
        p = Path(hint)
        if p.exists():
            return p
        sys.exit(f"[ERROR] Not found: {hint}")
    for mp in sorted(REPLICA_ROOT.rglob("mesh.ply")):
        if "habitat" not in str(mp):
            print(f"[Synthesizer] Auto-detected mesh: {mp}")
            return mp
    sys.exit(
        "[ERROR] No Replica mesh.ply found under Replica-Dataset/.\n"
        "        Run the download script first:\n"
        f"          {REPLICA_ROOT}\\win_download.bat"
    )


def load_mesh(mesh_path):
    print(f"[Synthesizer] Loading mesh: {mesh_path}")
    raw = trimesh.load(str(mesh_path), process=False)
    if isinstance(raw, trimesh.Scene):
        mesh = raw.dump(concatenate=True)
    else:
        mesh = raw
    print(f"[Synthesizer] Mesh: {len(mesh.vertices):,} verts / {len(mesh.faces):,} faces")
    return mesh


# ==============================================================================
# 1. world_pointcloud.ply
# ==============================================================================

def synthesize_point_cloud(mesh, out_path):
    print(f"[Synthesizer] Sampling {POINT_CLOUD_SAMPLE:,} surface points ...")
    pts, _ = trimesh.sample.sample_surface(mesh, POINT_CLOUD_SAMPLE)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    o3d.io.write_point_cloud(str(out_path), pcd)
    arr = np.asarray(pcd.points)
    print(f"[Synthesizer] OK world_pointcloud.ply  ({len(arr):,} pts) -> {out_path}")
    return arr


# ==============================================================================
# 2. room_layout.obj
# ==============================================================================

def synthesize_room_layout(pcd_pts, out_path):
    print("[Synthesizer] Building floor slab for room_layout.obj ...")
    y_min        = float(pcd_pts[:, 1].min())
    floor_mask   = pcd_pts[:, 1] <= y_min + 0.15
    floor_pts    = pcd_pts[floor_mask]

    if len(floor_pts) < 10:
        print("[Synthesizer] WARN: too few floor pts, using full bounding box.")
        floor_pts = pcd_pts

    min_x = float(floor_pts[:, 0].min()) - 0.1
    max_x = float(floor_pts[:, 0].max()) + 0.1
    min_z = float(floor_pts[:, 2].min()) - 0.1
    max_z = float(floor_pts[:, 2].max()) + 0.1
    fw = max_x - min_x
    fd = max_z - min_z

    slab = trimesh.creation.box(extents=[fw, FLOOR_THICKNESS, fd])
    slab.apply_translation([(min_x + max_x) / 2,
                             y_min - FLOOR_THICKNESS / 2,
                             (min_z + max_z) / 2])
    slab.export(str(out_path))
    print(f"[Synthesizer] OK room_layout.obj  ({fw:.1f} x {fd:.1f} m) -> {out_path}")


# ==============================================================================
# 3. ar_metadata.json
# ==============================================================================

def synthesize_ar_metadata(pcd_pts, out_path):
    print(f"[Synthesizer] Generating {NUM_ORBIT_FRAMES} orbit camera poses ...")
    center     = pcd_pts.mean(axis=0)
    max_extent = float(np.max(pcd_pts.max(axis=0) - pcd_pts.min(axis=0)))
    radius     = max_extent * ORBIT_RADIUS_SCALE
    floor_y    = float(pcd_pts[:, 1].min())
    cam_y      = floor_y + ORBIT_HEIGHT_ABOVE

    frames = []
    for i in range(NUM_ORBIT_FRAMES):
        angle = 2.0 * math.pi * i / NUM_ORBIT_FRAMES
        eye   = np.array([center[0] + radius * math.cos(angle),
                          cam_y,
                          center[2] + radius * math.sin(angle)])
        tgt   = np.array([center[0], floor_y + 0.5, center[2]])

        fwd  = tgt - eye;  fwd /= np.linalg.norm(fwd)
        up0  = np.array([0.0, 1.0, 0.0])
        rgt  = np.cross(fwd, up0)
        if np.linalg.norm(rgt) < 1e-6:
            rgt = np.array([1.0, 0.0, 0.0])
        rgt /= np.linalg.norm(rgt)
        tup  = np.cross(rgt, fwd);  tup /= np.linalg.norm(tup)

        R = np.eye(4)
        R[:3, 0] = rgt
        R[:3, 1] = tup
        R[:3, 2] = -fwd   # ARKit -Z forward
        R[:3, 3] = eye

        frames.append({
            "frame_id":       i,
            "timestamp_ns":   1_711_234_567_890 + i * 33_333_333,
            "tracking_state": "TRACKING",
            "pose_matrix":    R.tolist(),
        })

    metadata = {"intrinsics": [FX, FY, CX, CY], "frames": frames}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Synthesizer] OK ar_metadata.json  ({NUM_ORBIT_FRAMES} frames) -> {out_path}")
    return frames, [FX, FY, CX, CY]


# ==============================================================================
# 4. detections_from_b.json  +  meshes_from_c/
# ==============================================================================

def project_cluster(cluster_pts, frames, fx, fy, cx_px, cy_px):
    """Return list of {frame_id, bbox_px} for frames where cluster is visible."""
    N       = len(cluster_pts)
    homo    = np.hstack([cluster_pts, np.ones((N, 1))])
    views   = []

    for frame in frames:
        pose     = np.array(frame["pose_matrix"])
        try:
            pose_inv = np.linalg.inv(pose)
        except np.linalg.LinAlgError:
            continue

        cam      = (pose_inv @ homo.T).T[:, :3]
        cam[:, 2] *= -1   # ARKit -> Pinhole Z
        cam[:, 1] *= -1   # ARKit -> Pinhole Y

        front   = cam[:, 2] > 0.3
        if front.sum() < 5:
            continue

        z_s     = np.where(front, cam[:, 2], 1.0)
        u       = cam[:, 0] * fx / z_s + cx_px
        v       = cam[:, 1] * fy / z_s + cy_px
        vis     = front & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        if vis.sum() < 5:
            continue

        xmin = int(np.clip(u[vis].min() - 5, 0, IMG_W))
        ymin = int(np.clip(v[vis].min() - 5, 0, IMG_H))
        xmax = int(np.clip(u[vis].max() + 5, 0, IMG_W))
        ymax = int(np.clip(v[vis].max() + 5, 0, IMG_H))

        if (xmax - xmin) < 30 or (ymax - ymin) < 30:
            continue
        views.append({"frame_id": frame["frame_id"],
                      "bbox_px":  [xmin, ymin, xmax, ymax]})
    return views


def synthesize_detections(mesh, pcd_pts, frames, intrinsics, out_path):
    print("[Synthesizer] Clustering furniture objects via DBSCAN ...")
    fx, fy, cx_px, cy_px = intrinsics
    floor_y = float(pcd_pts[:, 1].min())
    ceil_y  = float(pcd_pts[:, 1].max())

    obj_mask = (pcd_pts[:, 1] > floor_y + 0.20) & (pcd_pts[:, 1] < ceil_y - 0.30)
    obj_pts  = pcd_pts[obj_mask]

    if len(obj_pts) < 100:
        print("[Synthesizer] WARN: too few object-height pts -> using fallback object.")
        _fallback_detections(pcd_pts, frames, out_path)
        return

    xz     = obj_pts[:, [0, 2]]
    labels = _DBSCAN(eps=0.45, min_samples=30).fit_predict(xz)
    unique = [l for l in set(labels) if l >= 0]
    print(f"[Synthesizer] Found {len(unique)} clusters.")

    detections = {}
    idx        = 1

    for cid in unique:
        cpts = obj_pts[labels == cid]
        if len(cpts) < MIN_FURNITURE_PTS:
            continue

        obj_id  = f"furniture_{idx:02d}";  idx += 1
        min_b   = cpts.min(axis=0)
        max_b   = cpts.max(axis=0)
        center  = cpts.mean(axis=0)
        dx, dy, dz = float(max_b[0]-min_b[0]), float(max_b[1]-min_b[1]), float(max_b[2]-min_b[2])

        views = project_cluster(cpts, frames, fx, fy, cx_px, cy_px)
        if len(views) < 3:
            continue
        views = views[::3]   # thin out to avoid O(N^2) in ObjectEstimator

        detections[obj_id] = {"associated_views": views}

        proxy = trimesh.creation.box(extents=[dx, dy, dz])
        proxy.apply_translation(center.tolist())
        proxy.export(str(MESH_DIR / f"{obj_id}.obj"))
        print(f"  -> {obj_id}: {len(views)} views | {dx:.2f}x{dy:.2f}x{dz:.2f} m")

    if not detections:
        print("[Synthesizer] WARN: no clusters passed view test -> using fallback.")
        _fallback_detections(pcd_pts, frames, out_path)
        return

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2)
    print(f"[Synthesizer] OK detections_from_b.json  ({len(detections)} objects) -> {out_path}")


def _fallback_detections(pcd_pts, frames, out_path):
    center      = pcd_pts.mean(axis=0).copy()
    center[1]   = float(pcd_pts[:, 1].min()) + 0.4
    obj_id      = "furniture_01"
    proxy       = trimesh.creation.box(extents=[0.8, 0.8, 0.8])
    proxy.apply_translation(center.tolist())
    proxy.export(str(MESH_DIR / f"{obj_id}.obj"))
    views = [{"frame_id": f["frame_id"], "bbox_px": [200, 200, 1700, 1200]}
             for f in frames[:10]]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({obj_id: {"associated_views": views}}, f, indent=2)
    print(f"[Synthesizer] OK detections_from_b.json (fallback) -> {out_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    hint      = sys.argv[1] if len(sys.argv) > 1 else None
    mesh_path = find_replica_mesh(hint)

    print("\n" + "="*65)
    print("  Replica -> Part A Data Synthesizer")
    print("="*65)
    print(f"  Source : {mesh_path}")
    print(f"  Output : {DATA_DIR}")
    print("="*65 + "\n")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MESH_DIR.mkdir(parents=True, exist_ok=True)

    mesh    = load_mesh(mesh_path)
    pcd_pts = synthesize_point_cloud(mesh, DATA_DIR / "world_pointcloud.ply")
    synthesize_room_layout(pcd_pts, DATA_DIR / "room_layout.obj")
    frames, intrinsics = synthesize_ar_metadata(pcd_pts, DATA_DIR / "ar_metadata.json")
    synthesize_detections(mesh, pcd_pts, frames, intrinsics,
                          DATA_DIR / "detections_from_b.json")

    print("\n" + "="*65)
    print("  Synthesis complete.  All Part A inputs are ready.")
    print("="*65)
    print("\nFiles generated:")
    for fp in sorted(DATA_DIR.rglob("*")):
        if fp.is_file():
            print(f"  {fp.relative_to(SCRIPT_DIR)}")
    print("\nRun Part A with:")
    print("  python PartA/main_pipeline.py\n")


if __name__ == "__main__":
    main()

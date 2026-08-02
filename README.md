# SceneReconstruction

A two-part pipeline that reconstructs a furnished indoor scene from video and 3D mesh assets.

```
Part B  →  Object detection & Re-ID from video
Part A  →  3D spatial engine: camera poses + point cloud → placed digital twin
```

---

## Repository Structure

```
SceneReconstruction/
├── PartA/
│   ├── main_pipeline.py            # Part A entry point
│   ├── synthesize_replica_data.py  # Generates Part A inputs from a Replica mesh
│   ├── config.py                   # Voxel sizes, DBSCAN thresholds, paths
│   ├── requirements.txt            # Part A Python dependencies
│   ├── modules/
│   │   ├── data_loader.py          # ARKit metadata + point cloud loader
│   │   ├── object_estimator.py     # Back-projection, PCA, DBSCAN OBB
│   │   ├── mesh_placer.py          # Mesh scale/align to support surface
│   │   └── room_builder.py         # RANSAC floor/wall detection
│   └── adapters/                   # CoordinateAdapter (ARKit → Pinhole)
│
├── PartB/
│   ├── reid_pipeline.py            # Furniture detection + BoT-SORT Re-ID
│   └── render_360_video.py         # 360° walkthrough video renderer
│
├── Replica-Dataset/                # Replica scene files (download separately)
│   ├── win_download.bat
│   └── download.sh
│
└── .venv/                          # Python virtual environment (not committed)
```

---

## Quick Start

### 1 — Clone & create virtual environment

```powershell
git clone <repo-url>
cd SceneReconstruction

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
```

### 2 — Install dependencies

```powershell
# Part A (3D spatial engine)
pip install -r PartA/requirements.txt

# Part B (video Re-ID pipeline)
pip install ultralytics torch torchvision opencv-python tqdm Pillow

# Part C (Construct 3D objects from image)
cd SceneReconstruction/PartC
chmod +x setup.sh
./setup.sh
```

### 3 — Download a Replica scene

```powershell
# Windows
Replica-Dataset\win_download.bat

# Linux / macOS
bash Replica-Dataset/download.sh /path/to/output
```

### 4 — Synthesise Part A inputs from the Replica mesh

```powershell
# Auto-discovers the first mesh.ply under Replica-Dataset/
python PartA/synthesize_replica_data.py

# Or point to a specific scene
python PartA/synthesize_replica_data.py Replica-Dataset\room_0\mesh.ply
```

This produces everything Part A needs inside `PartA/data/`:

| File | Description |
|---|---|
| `ar_metadata.json` | 120 synthetic ARKit camera poses (360° orbit) |
| `world_pointcloud.ply` | 80 k-point surface sample, voxel-cleaned |
| `room_layout.obj` | Floor slab geometry for RANSAC detection |
| `detections_from_b.json` | Per-object 2D bounding boxes per frame |
| `meshes_from_c/*.obj` | Axis-aligned proxy mesh per furniture cluster |

### 5 — Run Part A (3D spatial engine)

```powershell
python PartA/main_pipeline.py
```

Output: `PartA/data/output/digital_twin_scene.glb`

### 6 — Run Part B (video Re-ID pipeline)

Place your input video at `PartB/demo.mp4`, then:

```powershell
python PartB/reid_pipeline.py
```

Outputs land in `PartB/reid_objects_output/`:

| Path | Description |
|---|---|
| `reid_objects_output/demo_tracked.mp4` | Annotated tracking video |
| `reid_objects_output/segmented/` | Per-object segmentation crops |
| `reid_objects_output/bbox_unclipped/` | Full-frame annotated stills |

### 7 — (Optional) Render 360° video from Replica

```powershell
python PartB/render_360_video.py
```

---

## Dependencies at a Glance

| Package | Used by | Purpose |
|---|---|---|
| `open3d` | Part A | Point cloud I/O, voxel sampling, outlier removal |
| `trimesh` | Part A | Mesh loading, surface sampling, proxy mesh export |
| `scikit-learn` | Part A | DBSCAN clustering, PCA OBB |
| `scipy` | Part A | Spatial transforms |
| `pydantic` | Part A | Config validation |
| `ultralytics` | Part B | YOLO11m-seg detection |
| `torch` + `torchvision` | Part B | Re-ID feature embeddings |
| `opencv-python` | Part B | Video I/O, frame processing |
| `tqdm` | Part B | Progress bars |

---

## Notes

- **Model weights** (`*.pt`, `*.pth`) are excluded from git — download via Ultralytics on first run.
- **Replica-Dataset scene files** are excluded — they are multi-GB downloads.
- **Generated data** (`PartA/data/`, `reid_objects_output/`) is excluded — regenerate with the steps above.
- The synthesizer uses a pure-geometry heuristic (Y-band floor detection + XZ DBSCAN). For better furniture isolation, provide your own `detections_from_b.json` from a real Part B run.

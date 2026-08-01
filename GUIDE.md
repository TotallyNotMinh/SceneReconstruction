# Running Guide

Step-by-step instructions to go from a fresh clone to a working digital twin.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.13 | Tested on 3.13 via pyenv |
| Git | any | For cloning |
| Disk space | ~5 GB free | Replica scene + model weights |
| GPU (optional) | CUDA 11.8+ | Part B is faster with a GPU; Part A is CPU-only |

---

## Step 1 — Set up the virtual environment

```powershell
# From the project root
python -m venv .venv
.venv\Scripts\activate
```

> On Linux/macOS: `source .venv/bin/activate`

Verify activation — your prompt should show `(.venv)`.

---

## Step 2 — Install Part A dependencies

```powershell
pip install -r PartA/requirements.txt
```

Installs: `numpy`, `open3d`, `trimesh`, `scikit-learn`, `scipy`, `pygltflib`, `pydantic`.

---

## Step 3 — Install Part B dependencies

```powershell
pip install ultralytics torch torchvision opencv-python tqdm Pillow
```

> YOLO11m-seg weights (`yolo11m-seg.pt`) are downloaded automatically on first run.
> If you already have the `.pt` file in `PartB/`, it will be used directly.

---

## Step 4 — Download a Replica scene

Part A needs a real 3D scene to work with.

**Windows:**
```powershell
Replica-Dataset\win_download.bat
```

**Linux / macOS:**
```bash
bash Replica-Dataset/download.sh /path/to/output
```

This places scene folders like `Replica-Dataset/room_0/mesh.ply` on disk.  
The download is ~2–4 GB per scene.

---

## Step 5 — Synthesise Part A inputs

The synthesiser reads the Replica mesh and produces all five files that
`main_pipeline.py` expects.

```powershell
# Auto mode — picks the first mesh.ply it finds
python PartA/synthesize_replica_data.py

# Manual mode — point to a specific scene
python PartA/synthesize_replica_data.py Replica-Dataset\room_0\mesh.ply
```

### What gets generated

```
PartA/data/
├── ar_metadata.json          ← 120 synthetic ARKit camera poses
├── world_pointcloud.ply      ← 80 k surface points, cleaned
├── room_layout.obj           ← floor slab for RANSAC
├── detections_from_b.json    ← 2D bounding boxes per object per frame
└── meshes_from_c/
    ├── furniture_01.obj
    ├── furniture_02.obj
    └── ...
```

### How the synthesiser works

1. **Point cloud** — Samples 80,000 points from the Replica mesh surface,
   then voxel-downsamples at 2 cm and removes outliers.

2. **Room layout** — Finds the floor by selecting the lowest 15 cm of
   the point cloud and extrudes it into a flat slab.

3. **Camera trajectory** — Places 120 cameras on a smooth 360° orbit at
   80 cm above the floor, each looking at the scene centre.
   Poses are in ARKit 4×4 camera-to-world convention.

4. **Detections** — Filters points between floor+20 cm and ceiling−30 cm
   (furniture height band), runs XZ-plane DBSCAN (eps=0.45 m) to separate
   individual objects, then projects each cluster into every camera frame
   using the pinhole model. Any cluster visible in ≥3 frames becomes a
   detection entry.

5. **Proxy meshes** — One axis-aligned bounding box mesh per detected
   cluster, sized to the real physical extents of the cluster.

---

## Step 6 — Run Part A (3D spatial engine)

```powershell
python PartA/main_pipeline.py
```

The pipeline:
1. Loads `ar_metadata.json` + `world_pointcloud.ply`
2. Back-projects 2D detections into 3D using PCA → oriented bounding boxes
3. Detects floor/wall support surfaces via RANSAC
4. Scales and snaps each proxy mesh onto its nearest support surface
5. Exports the final scene

**Output:** `PartA/data/output/digital_twin_scene.glb`

Open `digital_twin_scene.glb` in any glTF viewer (e.g. Windows 3D Viewer,
online at https://gltf-viewer.donmccurdy.com/).

---

## Step 7 — Run Part B (video Re-ID pipeline)

Place a video file at `PartB/demo.mp4`, then:

```powershell
python PartB/reid_pipeline.py
```

The pipeline:
1. Runs YOLO11m-seg frame-by-frame to detect and segment furniture
2. Tracks objects across frames with BoT-SORT
3. Extracts Re-ID embeddings (ResNet-50 backbone) per crop
4. Clusters embeddings to assign persistent global IDs
5. Writes annotated video + per-object crops

**Outputs:**

| Path | Description |
|---|---|
| `PartB/reid_objects_output/demo_tracked.mp4` | Full annotated tracking video |
| `PartB/reid_objects_output/segmented/` | Per-object masked crops |
| `PartB/reid_objects_output/bbox_unclipped/` | Full-frame annotated stills |

---

## Step 8 — (Optional) Render a 360° walkthrough from Replica

```powershell
python PartB/render_360_video.py
```

Renders a video fly-through of the Replica scene using the same orbit
trajectory the synthesiser generates.

---

## Common Issues

### `ModuleNotFoundError: No module named 'open3d'`
You are not inside the virtual environment.
```powershell
.venv\Scripts\activate
```

### `No Replica mesh.ply found`
Run the download script first (Step 4) or pass an explicit path to the synthesiser.

### YOLO downloads weights every run
The `.pt` file is git-ignored. Copy it from a previous run or let it download once;
it is cached in the `PartB/` directory.

### Part A outputs an empty scene
Check that `PartA/data/detections_from_b.json` has at least one entry with
`associated_views` containing ≥3 frames. Re-run the synthesiser with a
larger scene or lower `MIN_FURNITURE_PTS` in `synthesize_replica_data.py`.

---

## Regenerating data

All generated data is reproducible:

```powershell
# Clean everything
Remove-Item -Recurse -Force PartA\data
Remove-Item -Recurse -Force PartB\reid_objects_output

# Re-synthesise
python PartA/synthesize_replica_data.py
python PartA/main_pipeline.py
```

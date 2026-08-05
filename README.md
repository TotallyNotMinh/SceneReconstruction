# Scene Reconstruction & Digital Twin Pipeline

A modular, computer vision pipeline that reconstructs a furnished 3D indoor scene from a video recording using **Depth Anything V3**, **YOLO11 + BoT-SORT + DINOv2**, **MobileSAM**, and **RANSAC 3D Spatial Assembly**.

---

## Codebase Architecture

```
SceneReconstruction/
├── config.py                     # Centralized configuration & hyperparameters
├── requirements.txt              # Unified root dependencies
│
├── core/                         # Core Utilities & Adapters
│   ├── video_normalizer.py       # Resolution normalization & per-axis intrinsics scaling
│   ├── coordinate_adapter.py     # ARKit/world/TripoSR coordinate transforms
│   └── data_loader.py            # Point cloud, metadata & video frame I/O
│
├── pointcloud/                   # Depth Anything V3 & Point Cloud Map Generator
│   ├── depth_inference.py        # Depth Anything V3 multi-view depth & pose inference
│   └── pointcloud_builder.py     # Global depth normalization & PLY/JSON point cloud exporter
│
├── detection/                    # Object Detection, Tracking & Re-ID
│   ├── reid_tracker.py           # YOLO11 + BoT-SORT + DINOv2 object tracker
│   └── sam_segmentor.py          # MobileSAM per-object pixel segmentation wrapper
│
├── spatial/                      # 3D Spatial Reconstruction Engine
│   ├── object_estimator.py       # Multi-view back-projection & 3D Alpha-Shape meshing
│   ├── room_builder.py           # RANSAC architectural floor/table plane detector
│   ├── mesh_placer.py            # Support surface snapping & spatial alignment
│   └── scene_assembler.py        # Main 3D digital twin scene orchestrator
│
├── visualization/                # Renderers & Visualizer Utilities
│   ├── render_side_by_side.py    # Side-by-side depth video renderer
│   └── render_360.py             # 360-degree mesh orbit video renderer
│
├── weights/                      # Pretrained Model Checkpoints
│   ├── yolo11m-seg.pt
│   └── mobile_sam.pt
│
└── data/                         # Consolidated Data Directory
    ├── raw/                      # Raw input video datasets (40753679/, 41007602/)
    ├── processed/                # Intermediate depth maps, metadata & detections
    └── output/                   # Reconstructed PLY point clouds & GLB digital twin scenes
```

---

## Quick Start

### 1. Environment Setup

```powershell
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies from root requirements.txt
pip install -r requirements.txt
```

### 2. Step 1 — Generate 3D Point Cloud Map (Depth Anything V3)

```powershell
<<<<<<< HEAD
# Part A (3D spatial engine)
pip install -r PartA/requirements.txt

# Part B (video Re-ID pipeline)
pip install ultralytics torch torchvision opencv-python tqdm Pillow

# Part C (Construct 3D objects from image)
cd SceneReconstruction/PartC
chmod +x setup.sh
./setup.sh
=======
python pointcloud/depth_inference.py data/raw/40753679/40753679.mov
>>>>>>> afaf91f2d3e638223820d4d31706241c8e75d502
```

Outputs:
- `data/output/world_pointcloud.ply`
- `data/processed/ar_metadata.json`
- `data/processed/depth_maps.npz`

### 3. Step 2 — Track Objects & Re-ID (YOLO11 + BoT-SORT + DINOv2)

```powershell
python detection/reid_tracker.py data/raw/40753679/40753679.mov
```

Outputs:
- `data/processed/detections.json`
- `data/processed/reid_output/`

### 4. Step 3 — Reconstruct & Assemble 3D Digital Twin Scene

```powershell
python spatial/scene_assembler.py --video=data/raw/40753679/40753679.mov
```

<<<<<<< HEAD
This produces everything Part A needs inside `PartA/data/`:

| File                     | Description                                   |
|--------------------------|-----------------------------------------------|
| `ar_metadata.json`       | 120 synthetic ARKit camera poses (360° orbit) |
| `world_pointcloud.ply`   | 80 k-point surface sample, voxel-cleaned      |
| `room_layout.obj`        | Floor slab geometry for RANSAC detection      |
| `detections_from_b.json` | Per-object 2D bounding boxes per frame        |
| `meshes_from_c/*.obj`    | Axis-aligned proxy mesh per furniture cluster |

### 5 - How to run
#### 5.1 — Run Part A (3D spatial engine)

```powershell
python PartA/main_pipeline.py
```

Output: `PartA/data/output/digital_twin_scene.glb`

#### 5.2 — Run Part B (video Re-ID pipeline)

Place your input video at `PartB/demo.mp4`, then:

```powershell
python PartB/reid_pipeline.py
```

Outputs land in `PartB/reid_objects_output/`:

#### 5.3 - Run Part C (Construct 3D objects from image)

Execute :

``` powershell
python PartC/main.py
```
Output: 'PartC/outputs'


| Path                                   | Description                   |
|----------------------------------------|-------------------------------|
| `reid_objects_output/demo_tracked.mp4` | Annotated tracking video      |
| `reid_objects_output/segmented/`       | Per-object segmentation crops |
| `reid_objects_output/bbox_unclipped/`  | Full-frame annotated stills   |

### 6 — (Optional) Render 360° video from Replica

```powershell
python PartB/render_360_video.py
```
=======
Outputs:
- `data/output/digital_twin_scene.glb`
>>>>>>> afaf91f2d3e638223820d4d31706241c8e75d502

---

## Visualization Tools

<<<<<<< HEAD
| Package                 | Used by | Purpose                                           |
|-------------------------|---------|---------------------------------------------------|
| `open3d`                | Part A  | Point cloud I/O, voxel sampling, outlier removal  |
| `trimesh`               | Part A  | Mesh loading, surface sampling, proxy mesh export |
| `scikit-learn`          | Part A  | DBSCAN clustering, PCA OBB                        |
| `scipy`                 | Part A  | Spatial transforms                                |
| `pydantic`              | Part A  | Config validation                                 |
| `ultralytics`           | Part B  | YOLO11m-seg detection                             |
| `torch` + `torchvision` | Part B  | Re-ID feature embeddings                          |
| `opencv-python`         | Part B  | Video I/O, frame processing                       |
| `tqdm`                  | Part B  | Progress bars                                     |
=======
Render side-by-side RGB vs Depth video:
```powershell
python visualization/render_side_by_side.py data/raw/40753679/40753679.mov
```
>>>>>>> afaf91f2d3e638223820d4d31706241c8e75d502

Render 360° orbit video of reconstructed 3D mesh:
```powershell
python visualization/render_360.py data/output/digital_twin_scene.glb
```

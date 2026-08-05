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
python pointcloud/depth_inference.py data/raw/40753679/40753679.mov
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

Outputs:
- `data/output/digital_twin_scene.glb`

---

## Visualization Tools

Render side-by-side RGB vs Depth video:
```powershell
python visualization/render_side_by_side.py data/raw/40753679/40753679.mov
```

Render 360° orbit video of reconstructed 3D mesh:
```powershell
python visualization/render_360.py data/output/digital_twin_scene.glb
```

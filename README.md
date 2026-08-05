# Scene Reconstruction & Digital Twin Pipeline

A modular, high-precision computer vision and 3D spatial intelligence pipeline that transforms indoor video recordings into interactive 3D digital twin scenes.

---

## Technical Overview & Model Stack

The pipeline operates across four core multi-modal stages:
1. **Depth Estimation & Pose Tracking**: **Depth Anything V3** (joint multi-view depth, camera intrinsics, and pose estimation).
2. **Object Detection & Re-ID**: **YOLO11m-seg** (semantic instance segmentation), **BoT-SORT** (multi-object tracking), and **DINOv2** (feature extraction for visual re-identification across camera viewpoints).
3. **Instance Segmentation Refinement**: **MobileSAM** (zero-shot promptable high-resolution mask refinement).
4. **3D Spatial Reconstruction**: **RANSAC** (architectural plane and support surface fitting), **DBSCAN** (3D point cloud clustering), **3D Alpha-Shape Meshing**, and **Trimesh** (scene assembly and GLB export).

### Pipeline Data Flow

```
 ┌─────────────────────────────────────┐
 │         data/raw/<dataset_id>/      │
 │                                     │
 │  📹 <dataset_id>.mov                │
 │  📍 lowres_wide.traj                │
 │  🔭 lowres_wide_intrinsics/*.pincam │
 └──────────────┬──────────────────────┘
                │
        ┌───────┴────────┬────────────────────────────────────┐
        │                │                                    │
        ▼                ▼                                    │
┌───────────────┐ ┌───────────────┐                          │
│    Stage 1    │ │    Stage 2    │                          │
│ depth_        │ │ reid_         │                          │
│ inference.py  │ │ tracker.py    │                          │
└───────┬───────┘ └───────┬───────┘                          │
        │                 │                                   │
        ▼                 ▼                                   │
┌───────────────────────────────────────┐                    │
│          data/processed/              │                    │
│                                       │                    │
│  📄 ar_metadata.json                  │◀───────────────────┘
│  🗜  depth_maps.npz                   │
│  ☁️  world_pointcloud.ply             │
│  📦 detections.json                   │
│  🏠 room_layout.obj                   │
└──────────────────┬────────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │    Stage 3     │
          │ scene_         │
          │ assembler.py   │
          └────────┬───────┘
                   │
                   ▼
   ┌───────────────────────────────┐
   │         data/output/          │
   │                               │
   │  ☁️  world_pointcloud.ply     │
   │  🌐 digital_twin_scene.glb   │
   └───────────────────────────────┘
```

---

## Codebase Architecture

```
SceneReconstruction/
├── config.py                     # Centralized configuration & hyperparameters
├── requirements.txt              # Project dependency requirements
│
├── core/                         # Core Utilities & Adapters
│   ├── video_normalizer.py       # Resolution normalization & per-axis intrinsics scaling
│   ├── coordinate_adapter.py     # ARKit / World / TripoSR coordinate transforms
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
└── data/                         # Consolidated Data Directory (Folders tracked, contents gitignored)
    ├── raw/                      # Raw video datasets & ARKit capture metadata
    ├── processed/                # Intermediate depth maps, metadata & detections
    └── output/                   # Reconstructed PLY point clouds & GLB digital twin scenes
```

---

## Data Directory Specification

```
data/
├── raw/                          <-- User Provided Inputs & ARKit Dataset Captures
│   └── <dataset_id>/
│       ├── <dataset_id>.mov                  # Primary indoor video recording
│       ├── lowres_wide.traj                  # ARKit camera trajectory / pose metadata file
│       └── lowres_wide_intrinsics/           # ARKit per-frame camera intrinsic parameters
│           └── <dataset_id>_<frame_id>.pincam # Individual pinhole camera calibration files
│
├── processed/                    <-- Pipeline Intermediate State (Auto-generated by Stage 1 & Stage 2)
│   ├── ar_metadata.json          # Camera intrinsic matrix K, scale factors, per-frame poses
│   ├── depth_maps.npz            # Compressed array containing per-frame metric depth maps
│   ├── raw_depths.npz            # Raw uncalibrated depth estimates from Depth Anything V3
│   ├── world_pointcloud.ply      # Intermediate global 3D point cloud used for 3D estimation
│   ├── detections.json           # Tracked object 2D bounding boxes, labels, and frame views
│   ├── room_layout.obj           # Base architectural floor boundary mesh (extracted by reid_tracker)
│   └── reid_output/              # Directory of cropped object instance images & debug tracking videos
│
└── output/                       <-- Final Deliverables (Auto-generated by Stage 1 & Stage 3)
    ├── world_pointcloud.ply      # Cleaned, downsampled global 3D point cloud export
    ├── digital_twin_scene.glb    # Assembled interactive 3D digital twin scene (GLTF Binary format)
    ├── side_by_side_depth.mp4    # (Optional) Side-by-side RGB vs Depth visualization video
    └── orbit_360.mp4             # (Optional) 360-degree turntable orbit rendering of the scene
```

> **Dataset Capture**: The `lowres_wide.traj` and `lowres_wide_intrinsics/` files are ARKit outputs exported from apps such as the [3D Scanner App](https://apps.apple.com/app/3d-scanner-app/id1419913995) (iOS) or [Record3D](https://record3d.app/). Capture your indoor scene using one of these apps and export in ARKit format to obtain the required files alongside the `.mov` recording.

---

## Detailed End-to-End Instructions

Follow these step-by-step instructions to run a complete pass from raw video to a 3D digital twin.

### Prerequisites & Checkpoints

- Ensure pretrained model checkpoints exist under `weights/`:
  - `weights/yolo11m-seg.pt`
  - `weights/mobile_sam.pt`
- Ensure `ffmpeg` is installed and available on your system `PATH` (required by `core/video_normalizer.py` for video preprocessing):
  - **Windows**: `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt install ffmpeg`

---

### Step 1 — Environment Setup

#### 1.1 Create Virtual Environment
Open your terminal in the root directory of the project:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 1.2 Install Dependencies
```bash
pip install -r requirements.txt
```
*Key packages installed:* `torch`, `torchvision`, `open3d`, `trimesh`, `ultralytics` (YOLO11), `transformers` (DINOv2), `depth-anything-3`, `mobile-sam`, `scikit-learn`, `scipy`.

---

### Step 2 — Stage 1: Multi-View Depth & 3D Point Cloud Generation

Run Depth Anything V3 inference on your raw input video:

```bash
python pointcloud/depth_inference.py data/raw/<dataset_id>/<dataset_id>.mov
```

#### Internal Processing Steps:
1. **Video Normalization**: Uses `core/video_normalizer.py` to fix orientation flags, downscale long-edge resolution to **720px** (Lanczos filtering), and update the intrinsic camera matrix $K$ (`scale_x`, `scale_y`).
2. **Depth & Pose Estimation**: Runs Depth Anything V3 Base to estimate multi-view metric depth and frame camera poses.
3. **Point Cloud Export**: Back-projects depth maps into global 3D space, applies voxel grid filtering, and exports the point cloud.

#### Generated Artifacts:
- `data/output/world_pointcloud.ply` — Cleaned global 3D point cloud
- `data/processed/ar_metadata.json` — Per-frame camera poses & rescaled intrinsics
- `data/processed/depth_maps.npz` — Archive of processed depth arrays

---

### Step 3 — Stage 2: Object Detection, Tracking & Re-Identification

Run multi-view object tracking and cross-view re-identification on the same input video:

```bash
python detection/reid_tracker.py data/raw/<dataset_id>/<dataset_id>.mov
```

> **Detected Object Classes**: By default, the tracker only detects the following COCO classes: `chair`, `couch`, `tv`, `microwave`, `oven`, `refrigerator`, `dining table`. To track different objects, update `TARGET_CLASSES` in [`config.py`](config.py).

#### Internal Processing Steps:
1. **Detection & Tracking**: Runs **YOLO11m-seg** with **BoT-SORT** to track target objects across consecutive frames.
2. **Re-ID Feature Extraction**: Extracts **DINOv2** visual embeddings for detected object crops.
3. **Cross-View Re-ID**: Computes cosine similarity matrices to consolidate multi-view trajectories into unique physical 3D object instances.
4. **Room Layout Generation**: Automatically generates initial architectural floor boundary `data/processed/room_layout.obj`.

#### Generated Artifacts:
- `data/processed/detections.json` — Consolidated multi-view object bounding boxes and track histories
- `data/processed/room_layout.obj` — Base architectural floor slab
- `data/processed/reid_output/` — Cropped object instances and annotated tracking video

---

### Step 4 — Stage 3: Spatial Reconstruction & 3D Digital Twin Assembly

> **Note**: Stage 3 requires outputs from both Stage 1 and Stage 2 to be present in `data/processed/`. It will fail with a clear error listing any missing files if either preceding stage has not been run.

Reconstruct individual 3D object meshes, detect architectural support planes, and assemble the scene:

```bash
python spatial/scene_assembler.py
```

#### Internal Processing Steps:
1. **Architectural Plane Detection**: Runs **RANSAC** plane fitting on the point cloud to identify floor and tabletop support surfaces (`spatial/room_builder.py`).
2. **Instance Masking & Back-Projection**: Refines 2D object masks using **MobileSAM** and back-projects 2D pixels into 3D point clusters (`spatial/object_estimator.py`).
3. **3D Alpha-Shape Meshing**: Fits 3D Alpha Shapes (`alpha=0.10`) around point clusters and decimates high-density geometry.
4. **Surface Snapping & Assembly**: Aligns and snaps object meshes onto detected support surfaces (`spatial/mesh_placer.py`) and packages the scene (`spatial/scene_assembler.py`).

#### Final Generated Artifact:
- `data/output/digital_twin_scene.glb` — Assembled, interactive 3D digital twin scene

---

### Step 5 — Visualization & Inspection (Optional)

#### Render Side-by-Side RGB vs Depth Video:
```bash
python visualization/render_side_by_side.py data/raw/<dataset_id>/<dataset_id>.mov
```

#### Render 360° Turntable Orbit Video of the 3D Scene:
```bash
python visualization/render_360.py data/output/digital_twin_scene.glb
```

---

## Configuration & Tuning Parameters

Pipeline parameters can be customized in [`config.py`](config.py):

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `TARGET_CLASSES` | `chair, couch, tv, microwave, oven, refrigerator, dining table` | COCO object classes to detect and reconstruct |
| `DEPTH_METRIC_MIN / MAX` | `0.5m / 5.0m` | Metric depth clipping range |
| `VOXEL_SIZE_PCD` | `0.02m` | Voxel grid downsampling resolution for point clouds |
| `DBSCAN_EPS` | `0.05m` | Clustering neighborhood radius for object point extraction |
| `SIMILARITY_THRESHOLD` | `0.80` | DINOv2 cosine similarity threshold for visual object Re-ID |
| `ALPHA_SHAPE_ALPHA` | `0.10m` | Alpha-Shape concavity parameter for 3D mesh surface generation |
| `RANSAC_DISTANCE_THRESH` | `0.03m` | Max inlier distance for RANSAC floor/table plane extraction |

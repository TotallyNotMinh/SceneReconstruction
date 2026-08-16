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
│   ├── object_extractor.py       # Phase 2A: 3D point cloud extraction & segmentation (0 synthetic points)
│   ├── object_mesher.py          # Phase 2B: High-fidelity 3D surface mesh generation (Poisson / BPA / Alpha)
│   ├── object_estimator.py       # Unified orchestrator wrapper (extract -> mesh)
│   ├── room_builder.py           # RANSAC architectural floor/table plane detector & room background
│   ├── mesh_placer.py            # Natural world placement & support surface snapping

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

---

### Running on Kaggle / Cloud Notebooks

> **Requirements**: Internet **ON** (Settings → Internet → On) + GPU accelerator enabled.

Run the bundled setup script as a notebook cell — it installs all dependencies, clones Depth Anything V3, verifies all imports, and prints a smoke test:

```python
exec(open("/kaggle/working/SceneReconstruction/scripts/kaggle_setup.py").read())
```

Or as a shell cell:
```bash
python /kaggle/working/SceneReconstruction/scripts/kaggle_setup.py
```

#### Kaggle Compatibility Notes

| Component | Status | Notes |
| :--- | :--- | :--- |
| `opencv-python-headless` | ✅ Ready | Used instead of `opencv-python` — no display libs needed |
| `open3d` | ✅ Ready | Requires `libgl1-mesa-glx` (installed by setup script) |
| `ffmpeg` | ✅ Pre-installed | Available in all Kaggle kernels |
| `torch` + CUDA | ✅ Pre-installed | Enable GPU accelerator in notebook settings |
| `depth-anything-3` | ✅ Ready | Installed from source via setup script (no PyPI release) |
| `render_360.py` | ⚠️ Optional | 360° orbit rendering — skippable if `open3d` unavailable |
| Disk space | ⚠️ Watch | DA3 model weights from HuggingFace can be 2–4 GB; Kaggle gives ~20 GB |

---

### Prerequisites & Checkpoints

- Ensure pretrained model checkpoints exist under `weights/`:
  - `weights/yolo11m-seg.pt`
  - `weights/mobile_sam.pt`
- Ensure `ffmpeg` is installed and available on your system `PATH` (required by `core/video_normalizer.py` for video preprocessing):
  - **Windows**: `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt install ffmpeg`
- **Depth Anything V3** is not published on PyPI — it must be installed from source (see [Step 1.3](#step-13--install-depth-anything-v3-from-source) below).

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
*Key packages installed:* `torch`, `torchvision`, `open3d`, `trimesh`, `ultralytics` (YOLO11), `transformers` (DINOv2), `mobile-sam`, `scikit-learn`, `scipy`.

> **Note**: `depth-anything-3` is intentionally excluded from `requirements.txt` — it has no official PyPI release. Install it from source in the next step.

---

#### 1.3 — Install Depth Anything V3 from Source

Depth Anything V3 is published by ByteDance-Seed only as a GitHub repository with no PyPI package. It must be cloned and installed in editable mode **outside** the project directory:

```bash
# From a directory of your choice (e.g. one level above this project)
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
pip install -e .
```

Alternatively, install directly from GitHub without cloning:

```bash
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

After installing, download the pretrained model weights using the provided script:

```bash
# Run from inside the cloned Depth-Anything-3 directory
bash ./scripts/download_weights.sh
```

Verify the install succeeded:

```python
from depth_anything_3.api import DepthAnything3
print("Depth Anything V3 installed successfully")
```

---

### Step 2 — Stage 1 Pass 1: Multi-View Depth & Pose Estimation

Run Depth Anything V3 joint multi-view inference on your raw input video:

```bash
# Standard FP32 inference (recommended for Ampere/Ada RTX 30/40 GPUs):
python pointcloud/depth_inference.py data/raw/<dataset_id>/<dataset_id>.mov

# FP16 mixed precision (required for Turing/Tesla T4 on Kaggle to avoid OOM):
python pointcloud/depth_inference.py data/raw/<dataset_id>/<dataset_id>.mov --fp16
```

Optionally specify a custom output path for the raw depths archive:

```bash
python pointcloud/depth_inference.py data/raw/<dataset_id>/<dataset_id>.mov data/processed/raw_depths.npz --fp16
```

#### Internal Processing Steps:
1. **Video Normalization**: Uses `core/video_normalizer.py` to correct orientation flags, downscale the long edge to `VIDEO_TARGET_LONG_EDGE` (default **720px**) via Lanczos filtering, cap frame rate to `VIDEO_TARGET_FPS` (default **24 FPS**), and propagate exact per-axis `scale_x`/`scale_y` into the intrinsic matrix $K$.
2. **Frame Sampling**: Samples one frame every `sample_stride` frames (default **8**), capped at `max_frames` (default **60**) to bound GPU VRAM usage during joint inference.
3. **Joint Multi-View Depth & Pose Inference**: Feeds all sampled frames simultaneously to **Depth Anything V3** to produce globally consistent metric depth maps and predicted camera extrinsics/intrinsics for each frame.
4. **Archive Export**: Saves all per-frame depth maps, RGB frames, predicted poses, intrinsics, and confidence maps into a compressed `.npz` archive. **No 3D point cloud is generated at this stage.**

#### Generated Artifacts:
- `data/processed/raw_depths.npz` — Compressed archive of per-frame depth maps, predicted camera poses (`ext_i`), intrinsics (`ixt_i`), confidence maps (`conf_i`), and RGB frames (`rgb_i`)

---

### Step 2b — Stage 1 Pass 2: 3D Point Cloud Construction

Back-project the saved depth maps into a global 3D point cloud:

```bash
# Default (step=4, voxel_size=0.02m):
python pointcloud/pointcloud_builder.py data/processed/raw_depths.npz

# Denser point cloud (step=2, samples 4x more pixels):
python pointcloud/pointcloud_builder.py data/processed/raw_depths.npz --step 2

# Maximum resolution (step=1, voxel_size=1cm for ultra-dense point cloud):
python pointcloud/pointcloud_builder.py data/processed/raw_depths.npz --step 1 --voxel-size 0.01
```

#### Internal Processing Steps:
1. **Load & Decode Archive**: Reads the `.npz` from Pass 1, extracting per-frame depth maps, camera poses, per-frame intrinsic matrices, RGB colors, and confidence maps.
2. **Global Depth Normalization**: Computes global `min`/`max` across all frames and linearly maps raw depth values into the metric range `[DEPTH_METRIC_MIN, DEPTH_METRIC_MAX]` (default `0.5m – 5.0m`).
3. **Confidence Filtering**: Discards the bottom 30th percentile of low-confidence pixels per frame before back-projection.
4. **Camera-to-World Back-Projection**: For each sampled pixel at stride `point_step`, lifts the 2D pixel + depth value into a 3D camera-space ray using the per-frame intrinsic $K_i^{-1}$, then transforms to world space using the predicted camera-to-world matrix $[R|t]$.
5. **Voxel Grid Downsampling**: Deduplicates points using a voxel grid of size `VOXEL_SIZE_PCD` (default **0.02m**) while preserving per-point RGB color.
6. **Export**: Saves the cleaned, colored point cloud to both `data/processed/` and `data/output/` as `world_pointcloud.ply`, and writes `ar_metadata.json` with the camera intrinsics and per-frame pose records.

#### Generated Artifacts:
- `data/output/world_pointcloud.ply` — Cleaned, colored, downsampled global 3D point cloud
- `data/processed/world_pointcloud.ply` — Intermediate copy used by downstream stages
- `data/processed/ar_metadata.json` — Camera intrinsic matrix $K$, per-axis scale factors, and per-frame pose matrices

> **Tip**: You can run both passes in one call from Python code using `generate_pcd_from_video()` in `pointcloud/depth_inference.py`:
> ```python
> from pointcloud.depth_inference import generate_pcd_from_video
> pts, metadata = generate_pcd_from_video("data/raw/<id>/<id>.mov")
> ```

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
2. **Object Point Cloud Extraction (Phase 2A)**: Extracts exact 3D point clusters directly from `world_pointcloud.ply` with 100% attribute fidelity and zero synthetic points (`spatial/object_extractor.py`).
3. **Optional Intermediate AI Point Cloud Completion**: Colleague's AI completion step can enrich extracted point clouds before meshing.
4. **3D Surface Meshing (Phase 2B)**: Reconstructs watertight meshes using **Screened Poisson**, BPA, or Alpha Shapes with Taubin smoothing and hole sealing (`spatial/object_mesher.py`).
5. **Natural Placement & Assembly**: Places object meshes directly at their natural world coordinates and packages the full scene (`spatial/mesh_placer.py`).


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
| `VIDEO_TARGET_LONG_EDGE` | `720` | Target long-edge resolution (px) for video normalization |
| `VIDEO_TARGET_FPS` | `24` | Target output frame rate cap for video normalization |
| `DEPTH_MODEL_ID` | `depth-anything/da3-base` | HuggingFace model repo ID for Depth Anything V3 |
| `DEPTH_SAMPLE_STRIDE` | `8` | Sample 1 frame every N frames for DA3 joint inference |
| `DEPTH_MAX_FRAMES` | `60` | Max frames fed simultaneously to DA3 (bounded by GPU VRAM) |
| `DEPTH_USE_FP16` | `False` | Run DA3 in FP16 mixed precision (use `--fp16` on Turing/T4 GPUs) |
| `DEPTH_METRIC_MIN / MAX` | `0.5m / 5.0m` | Metric depth clipping range for global depth normalization |
| `VOXEL_SIZE_PCD` | `0.02m` | Voxel grid downsampling cell size for point cloud deduplication |
| `TARGET_CLASSES` | `chair, couch, tv, microwave, oven, refrigerator, dining table` | COCO object classes to detect and reconstruct |
| `DBSCAN_EPS` | `0.05m` | Clustering neighborhood radius for object point extraction |
| `SIMILARITY_THRESHOLD` | `0.80` | DINOv2 cosine similarity threshold for visual object Re-ID |
| `ALPHA_SHAPE_ALPHA` | `0.10m` | Alpha-Shape concavity parameter for 3D mesh surface generation |
| `RANSAC_DISTANCE_THRESH` | `0.03m` | Max inlier distance for RANSAC floor/table plane extraction |

# -*- coding: utf-8 -*-
"""
config.py — Unified Configuration & Hyperparameters for Scene Reconstruction Pipeline.
"""

from pathlib import Path

# ── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent
DATA_DIR           = BASE_DIR / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR         = DATA_DIR / "output"
WEIGHTS_DIR        = BASE_DIR / "weights"

# Ensure required directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model Weights Paths ──────────────────────────────────────────────────────
YOLO_MODEL_PATH    = WEIGHTS_DIR / "yolo11m-seg.pt"
SAM_CHECKPOINT_PATH = WEIGHTS_DIR / "mobile_sam.pt"

# ── Point-cloud Processing ───────────────────────────────────────────────────
VOXEL_SIZE_PCD     = 0.005      # Voxel grid cell size (m)
DEPTH_METRIC_MIN   = 0.5       # Near clamp for metric depth (m)
DEPTH_METRIC_MAX   = 5.0       # Far clamp for metric depth (m)
ENABLE_ROR         = False     # Enable Radius Outlier Removal by default
ROR_RADIUS         = 0.05      # Radius Outlier Removal search radius (m)
ROR_MIN_NEIGHBORS  = 5         # Radius Outlier Removal min neighbors inside radius


# ── Multi-View Artifact Reduction & TSDF Fusion ──────────────────────────────
ENABLE_EDGE_FILTER     = False  # Enable edge-aware depth map filtering before back-projection
EDGE_FILTER_ALPHA      = 0.05  # Relative gradient threshold for silhouette depth filtering
EDGE_DILATE_ITERS      = 1     # Dilation iterations for edge depth mask

ENABLE_GRAZING_FILTER  = False  # Invalidate depth pixels observed at steep grazing angles
GRAZING_MAX_ANGLE_DEG  = 30  # Max viewing angle from surface normal in degrees

ENABLE_TSDF_FUSION     = True  # Use Open3D TSDF Volumetric Fusion
USE_TSDF               = ENABLE_TSDF_FUSION  # Alias for ENABLE_TSDF_FUSION
TSDF_SDF_TRUNC         = VOXEL_SIZE_PCD * 2.5 # TSDF SDF truncation distance (m) (2.5x voxel size)

ENABLE_FREE_SPACE_CHECK = False  # Enable multi-view free-space consistency check
FSV_MARGIN             = 0.06  # Free-space violation depth margin (m)
FSV_VIOLATION_RATIO    = 0.20  # Ratio threshold for free-space violation filter



# ── ObjectEstimator: Back-projection & Clustering ────────────────────────────
ENABLE_DBSCAN           = False # Enable DBSCAN cluster outlier removal
DBSCAN_EPS              = 0.05  # DBSCAN neighbourhood radius (m)
DBSCAN_MIN_SAMPLES      = 10    # Minimum points to form a cluster
DBSCAN_MIN_CLUSTER_SIZE = 50    # Minimum cluster size to keep
OCCLUSION_MIN_CONSENSUS = 0.60  # Fraction of views a point must be visible in


# ── Video Normalizer ─────────────────────────────────────────────────────────
VIDEO_TARGET_LONG_EDGE = 720    # Resize so the longer edge is this many pixels
VIDEO_TARGET_FPS       = 24     # Cap output frame rate (clamped to source FPS)

# ── Depth Inference (Pass 1) ─────────────────────────────────────────────────
DEPTH_MODEL_ID      = "depth-anything/da3-base"  # HuggingFace model repo ID
DEPTH_SAMPLE_STRIDE = 8    # Sample 1 frame every N frames for DA3 inference
DEPTH_MAX_FRAMES    = 60   # Max frames fed to DA3 joint inference (VRAM bound)


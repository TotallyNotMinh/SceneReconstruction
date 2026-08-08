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
VOXEL_SIZE_PCD     = 0.02      # Voxel grid cell size (m)
DEPTH_METRIC_MIN   = 0.5       # Near clamp for metric depth (m)
DEPTH_METRIC_MAX   = 5.0       # Far clamp for metric depth (m)

# ── ObjectEstimator: Back-projection & Clustering ────────────────────────────
DBSCAN_EPS              = 0.05  # DBSCAN neighbourhood radius (m)
DBSCAN_MIN_SAMPLES      = 10    # Minimum points to form a cluster
OCCLUSION_MIN_CONSENSUS = 0.60  # Fraction of views a point must be visible in

# ── MeshPlacer ───────────────────────────────────────────────────────────────
MAX_DISTORTION_THRESH   = 1.25  # Max anisotropic scale ratio before non-uniform scale

# ── RoomBuilder: RANSAC Plane Detection ──────────────────────────────────────
RANSAC_DISTANCE_THRESH  = 0.03  # Inlier distance from plane (m)
RANSAC_ITERATIONS       = 500   # RANSAC iteration budget
RANSAC_MIN_INLIERS      = 20    # Minimum inliers to accept a plane
FLOOR_NORMAL_THRESH     = 0.80  # |B| threshold for horizontal plane (Y-normal)
TABLE_MIN_HEIGHT        = 0.35  # Min height above floor to be a table surface (m)
TABLE_MAX_HEIGHT        = 1.30  # Max height above floor to be a table surface (m)
TABLE_SNAP_TOLERANCE    = 0.35  # Vertical distance to snap object onto a table (m)

# ── Alpha Shape Meshing ───────────────────────────────────────────────────────
ALPHA_SHAPE_ALPHA   = 0.10      # Tightness of alpha shape (m)
ALPHA_MAX_FACES     = 5_000     # Quadric-decimate threshold if face count exceeds this

# ── Video Normalizer ─────────────────────────────────────────────────────────
VIDEO_TARGET_LONG_EDGE = 720    # Resize so the longer edge is this many pixels
VIDEO_TARGET_FPS       = 24     # Cap output frame rate (clamped to source FPS)

# ── Depth Inference (Pass 1) ─────────────────────────────────────────────────
DEPTH_MODEL_ID      = "depth-anything/da3-base"  # HuggingFace model repo ID
DEPTH_SAMPLE_STRIDE = 8    # Sample 1 frame every N frames for DA3 inference
DEPTH_MAX_FRAMES    = 60   # Max frames fed to DA3 joint inference (VRAM bound)
DEPTH_USE_FP16      = False  # Run inference in FP16 mixed precision (needed for Turing/T4 GPUs)

# ── ReID & Object Detection ──────────────────────────────────────────────────
SIMILARITY_THRESHOLD  = 0.8     # DINOv2 Cosine similarity threshold for object Re-ID
SAMPLE_EVERY_N_FRAMES = 5       # Sample 1 every 5 frames for Re-ID feature extraction
MIN_CROP_SIZE         = 100     # Minimum px size to ignore noise

TARGET_CLASSES = {
    'chair', 'couch', 'tv', 'microwave', 'oven', 'refrigerator', 'dining table'
}

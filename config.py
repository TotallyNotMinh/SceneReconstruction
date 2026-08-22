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
POINTR_MODEL_PATH  = WEIGHTS_DIR / "pointr_shapenet.pth"


# ── Point-cloud Processing ───────────────────────────────────────────────────
VOXEL_SIZE_PCD     = 0.015      # Voxel grid cell size (m) - increased density (1.5cm)
POINTCLOUD_POINT_STEP = 2       # Pixel step for back-projection (2 = 4x denser than 4)
DEPTH_METRIC_MIN   = 0.5        # Near clamp for metric depth (m)
DEPTH_METRIC_MAX   = 6.0        # Far clamp for metric depth (m)
ENABLE_ROR         = False      # Enable Radius Outlier Removal by default
ROR_RADIUS         = 0.05       # Radius Outlier Removal search radius (m)
ROR_MIN_NEIGHBORS  = 4          # Radius Outlier Removal min neighbors inside radius


# ── Multi-View Artifact Reduction & TSDF Fusion ──────────────────────────────
ENABLE_EDGE_FILTER     = False  # Enable edge-aware depth map filtering before back-projection
EDGE_FILTER_ALPHA      = 0.05  # Relative gradient threshold for silhouette depth filtering
EDGE_DILATE_ITERS      = 1     # Dilation iterations for edge depth mask

ENABLE_GRAZING_FILTER  = False  # Invalidate depth pixels observed at steep grazing angles
GRAZING_MAX_ANGLE_DEG  = 30    # Max viewing angle from surface normal in degrees

ENABLE_TSDF_FUSION     = True   # Use Open3D TSDF Volumetric Fusion
USE_TSDF               = ENABLE_TSDF_FUSION  # Alias for ENABLE_TSDF_FUSION
TSDF_SDF_TRUNC         = VOXEL_SIZE_PCD * 2.5 # TSDF SDF truncation distance (m) (2.5x voxel size = 0.0375m)

ENABLE_FREE_SPACE_CHECK = False  # Enable multi-view free-space consistency check
FSV_MARGIN             = 0.06  # Free-space violation depth margin (m)
FSV_VIOLATION_RATIO    = 0.20  # Ratio threshold for free-space violation filter



# ── PointcloudBuilder Whole-Scene Clustering ─────────────────────────────────
ENABLE_DBSCAN                = False  # Disable whole-scene DBSCAN (prevents wiping disparate room components)
DBSCAN_EPS                   = 0.08   # Whole-scene DBSCAN radius (m)
DBSCAN_MIN_SAMPLES           = 5      # Whole-scene DBSCAN min samples
DBSCAN_MIN_CLUSTER_SIZE      = 30     # Whole-scene DBSCAN min cluster size

# ── ObjectExtractor: 3D Point Cloud Extraction & Clustering ────────────────
OBJECT_ENABLE_DBSCAN         = True   # Enable DBSCAN cluster outlier removal for object instances
OBJECT_DBSCAN_EPS            = 0.06   # Object DBSCAN neighbourhood radius (m)
OBJECT_DBSCAN_MIN_SAMPLES    = 4      # Object DBSCAN min samples
OBJECT_DBSCAN_MIN_CLUSTER_SIZE = 10   # Object DBSCAN min cluster size
OBJECT_VIEW_CONSENSUS_RATIO  = 0.30   # Min fraction of multi-view detections a 3D point must be observed in
PLANE_SUBTRACTION_MARGIN     = 0.015  # Distance margin to subtract floor/tabletop plane points from objects (m)
ENABLE_DUAL_SOURCE_OBJECT_EXTRACTION = False # Deprecated: Keep False to guarantee 0 synthetic/interpolated points
OBJECT_DEPTH_CONSISTENCY_TOLERANCE = 0.10   # Max absolute delta |Z_cam - Z_depth| in meters for depth gating (10cm)
ENABLE_OBJECT_COLOR_FILTER   = True   # Enable CIELAB color consistency verification between 2D masks and 3D points
OBJECT_COLOR_DELTA_E_MAX     = 45.0   # Max allowable CIELAB Euclidean color distance (Delta E) to 2D mask appearance
OBJECT_COLOR_MASK_SAMPLE_COUNT = 300  # Number of pixels to sample from 2D mask to model object color manifold
ENABLE_SUPPORT_PLANE_COLOR_CONTRAST = True # Prune base contact points whose color matches underlying plane
OBJECT_COLOR_WEIGHT_IN_DBSCAN = 0.0025 # Color distance weighting factor in 6D XYZ+CIELAB DBSCAN clustering

# ── OpenMask3D: Open-Vocabulary 3D Instance Segmentation ──────────────────────
OPENMASK3D_CLIP_MODEL        = "ViT-B/32"  # Pretrained CLIP model: "ViT-B/32", "ViT-L/14", "ViT-L/14@336px"
OPENMASK3D_CLIP_PRETRAINED   = "openai"    # CLIP pretrained weights source (or "laion2b_s34b_b79k")
OPENMASK3D_SIMILARITY_THRESH = 0.18        # Minimum cosine similarity threshold for zero-shot text-mask matching
OPENMASK3D_MIN_POINTS        = 30          # Minimum point count for a valid 3D object instance
OPENMASK3D_TOP_K_VIEWS       = 10          # Number of best camera viewpoints to aggregate CLIP embeddings per mask
OPENMASK3D_VOXEL_SIZE        = 0.02        # Voxel grid size (m) for 3D mask proposal downsampling
OPENMASK3D_PROPOSAL_EPS      = 0.06        # Neighborhood distance for 3D point grouping proposals (m)
OPENMASK3D_MAX_PROPOSALS     = 60          # Maximum 3D candidate proposals to process
OPENMASK3D_CLASSES           = [           # Open-vocabulary target indoor / scene classes
    "chair", "armchair", "office chair", "swivel chair", "stool",
    "table", "desk", "coffee table", "dining table", "nightstand",
    "sofa", "couch", "cushion", "bed", "mattress",
    "monitor", "computer", "laptop", "tv", "screen", "keyboard", "mouse",
    "lamp", "floor lamp", "desk lamp", "light",
    "plant", "potted plant", "vase", "flower",
    "refrigerator", "microwave", "oven", "sink", "trash can", "bin",
    "cabinet", "shelf", "bookshelf", "drawer", "wardrobe", "cupboard",
    "door", "window", "curtain", "blind",
    "box", "backpack", "bag", "pillow", "book", "bottle", "cup", "mug"
]

# ── Point Cloud Completion (PoinTr Transformer) ──────────────────────────────
ENABLE_POINTCLOUD_COMPLETION = True        # Enable ML-based 3D point cloud shape completion before meshing
POINTR_NUM_INPUT_POINTS      = 2048        # Number of input points sampled for PoinTr canonical input (FPS)
POINTR_NUM_OUTPUT_POINTS     = 8192        # Number of dense completed points generated by PoinTr
POINTR_PRESERVE_ORIGINAL     = True        # Fuse 100% of original observed points with completed points
POINTR_KNN_COLOR_K           = 3           # Number of nearest observed neighbors for color propagation
POINTR_CANONICAL_SCALE       = 1.0         # Normalization target scale factor


# ── RoomBuilder & RANSAC Plane Detection ──────────────────────────────────────

RANSAC_DISTANCE_THRESH      = 0.03   # Max inlier distance for RANSAC floor/table plane extraction (m)
RANSAC_N                    = 3      # Number of points sampled per RANSAC plane hypothesis
RANSAC_NUM_ITERATIONS       = 1000   # Number of RANSAC iterations
RANSAC_MAX_PLANES           = 12     # Maximum RANSAC planes to extract (ensures tabletop planes are found)
ROOM_FLOOR_NORMAL_TOLERANCE = 0.85   # Min abs(dot(normal, gravity_up)) to qualify as horizontal plane
ROOM_WALL_NORMAL_TOLERANCE  = 0.25   # Max abs(dot(normal, gravity_up)) to qualify as vertical wall plane
TABLE_MIN_HEIGHT            = 0.30   # Minimum height above floor for a surface to be considered a table (m)
TABLE_MAX_HEIGHT            = 1.40   # Maximum height above floor for a surface to be considered a table (m)
ENABLE_SEMANTIC_PLANE_VERIFICATION = True # Cross-verify horizontal planes with 2D/3D detections

# ── ObjectEstimator & 3D Surface Meshing ─────────────────────────────────────
OBJECT_MESHING_METHOD          = "poisson" # 3D Meshing algorithm: "poisson" (Smooth watertight), "bpa", "alpha"
OBJECT_POISSON_DEPTH           = 9      # Octree depth for object Screened Poisson (9 = fine sub-centimeter resolution)
OBJECT_POISSON_DENSITY_TRIM    = 1.5    # Gentle density trimming percentile (<= 1.5% preserves thin edges & internal carvings)
OBJECT_BPA_RADII_MULTIPLIER    = [0.6, 1.2, 2.5, 5.0, 10.0] # 5-tier progressive ball radii multipliers for exact triangulation
ALPHA_SHAPE_ALPHA              = 0.035  # Concavity parameter for fallback meshing (m)
OBJECT_DEPTH_FOREGROUND_MARGIN = 0.85   # Adaptive max depth delta beyond near depth to prune background bleed (m)
OBJECT_EXTRACT_FROM_WORLD_PCD  = True   # Extract object point clouds directly from world_pointcloud.ply via 2D guidance
SAVE_OBJECT_POINTCLOUDS        = True   # Export individual object point clouds (.ply) for visual inspection
MESH_SMOOTHING_METHOD          = "taubin" # Non-shrinking Taubin mesh smoothing (eliminates ripples without volume loss)
MESH_TAUBIN_ITERATIONS         = 10     # Number of Taubin smoothing iterations
MESH_TAUBIN_LAMBDA             = 0.45   # Taubin positive smoothing factor
MESH_TAUBIN_MU                 = -0.48  # Taubin negative inflation factor (preserves carved concavities)
FILL_MESH_HOLES                = True   # Automatically detect and seal small open triangle boundary loops
MAX_HOLE_VERTICES              = 16     # Maximum boundary loop vertex count to fill (preserves large structural openings)
ENABLE_ADAPTIVE_POISSON_DEPTH  = True   # Automatically adapt depth 8-10 based on object point cloud scale


# ── RoomBuilder & Background Room Meshing ────────────────────────────────────
ROOM_RECONSTRUCTION_METHOD     = "background_mesh" # "background_mesh" (Video-accurate), "cad_slabs" (Bounding box slabs), or "both"
ROOM_BACKGROUND_MESHING_METHOD = "poisson"         # "poisson" (Smooth inpainting of occlusion holes), "bpa", "alpha"
ROOM_BPA_RADII_MULTIPLIER      = [0.8, 1.5, 3.0, 6.0] # 4-tier progressive ball radii multipliers for room BPA
ROOM_OBJECT_SUBTRACTION_RADIUS = 0.05              # Spatial radius (m) around object points to prune from world point cloud (5cm)
ENABLE_ROOM_PLANE_INPAINTING   = True              # Inpaint planar points on floor & walls where objects were subtracted
ROOM_INPAINTING_GRID_STEP      = 0.025             # 2.5cm grid step for dense planar floor & wall inpainting
ROOM_INPAINTING_GAP_THRESHOLD  = 0.04              # Distance threshold (4cm) to identify and fill empty voids
ROOM_POISSON_DEPTH             = 9                 # Octree depth for Screened Poisson reconstruction (fine room geometry)
ROOM_POISSON_DENSITY_TRIM      = 3.0               # Percentile of low-density vertices to trim from Poisson surface
WALL_THICKNESS                 = 0.05              # Thickness of generated wall slab meshes in meters
ROOM_MIN_WALL_INLIERS          = 250               # Minimum inlier points required to qualify as an architectural wall plane
EXPORT_ROOM_CAD_SLABS          = True              # Export room_layout.obj bounding box slabs alongside background mesh


# ── MeshPlacer & Scene Assembly ──────────────────────────────────────────────
ENABLE_SURFACE_SNAPPING     = False  # Default False: Place objects directly at natural extracted world coordinates (no artificial shift)
SURFACE_SNAPPING_MARGIN     = 0.00   # Vertical offset margin when snapping mesh bottom to surface (m)
WALL_SNAPPING_MARGIN        = 0.01   # Margin offset when snapping mesh onto vertical wall surface (m)
EXPORT_FULL_SCENE           = True   # Automatically assemble full scene (room background + aligned objects) into data/output
WALL_MOUNTED_CLASSES        = {      # Semantic labels for objects mounted on walls
    "tv", "tvmonitor", "picture", "clock", "mirror", "whiteboard", "poster", "wall_art", "screen"
}



# ── Video Normalizer ─────────────────────────────────────────────────────────
VIDEO_TARGET_LONG_EDGE = 720    # Resize so the longer edge is this many pixels
VIDEO_TARGET_FPS       = 24     # Cap output frame rate (clamped to source FPS)

# ── Depth Inference (Pass 1) ─────────────────────────────────────────────────
DEPTH_MODEL_ID      = "depth-anything/da3-base"  # HuggingFace model repo ID
DEPTH_SAMPLE_STRIDE = 8    # Sample 1 frame every N frames for DA3 inference
DEPTH_MAX_FRAMES    = 60   # Max frames fed to DA3 joint inference (VRAM bound)


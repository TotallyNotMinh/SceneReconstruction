# =============================================================================
# kaggle_setup.py — Scene Reconstruction Pipeline: Kaggle Environment Setup
#
# Paste each cell into a Kaggle notebook, or run the whole file as a script.
# Requires: Internet ON (Settings → Internet → On) and GPU accelerator enabled.
# =============================================================================

import subprocess, sys, os

def _run(cmd, allow_fail=False):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout[-3000:])
    if result.returncode != 0:
        msg = result.stderr[-1000:] if result.stderr else "(no stderr)"
        if allow_fail:
            print(f"[WARN] Command exited {result.returncode}: {msg}")
        else:
            print(f"[ERROR] Command exited {result.returncode}: {msg}")
    return result.returncode

REPO_DIR = "/kaggle/working/SceneReconstruction"
DA3_DIR  = "/kaggle/working/Depth-Anything-3"
DA3_REPO = "https://github.com/ByteDance-Seed/Depth-Anything-3.git"
SCENE_REPO = "https://github.com/TotallyNotMinh/SceneReconstruction.git"

# ── Step 1: System dependencies ───────────────────────────────────────────────
print("=" * 60)
print("  Step 1: Installing system libraries...")
print("=" * 60)
_run("apt-get install -y -q libgl1-mesa-glx libglib2.0-0 ffmpeg")

# ── Step 2: Clone SceneReconstruction FIRST (needed for requirements.txt) ─────
print("\n" + "=" * 60)
print("  Step 2: Setting up SceneReconstruction project...")
print("=" * 60)

if not os.path.exists(REPO_DIR):
    print(f"[→] Cloning SceneReconstruction into {REPO_DIR}...")
    _run(f"git clone {SCENE_REPO} {REPO_DIR}")
else:
    print(f"[✓] Project already exists at {REPO_DIR} — pulling latest...")
    _run(f"git -C {REPO_DIR} pull", allow_fail=True)

# Add project root to Python path immediately
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
    print(f"[✓] Added {REPO_DIR} to sys.path")

# ── Step 3: Install Python dependencies from requirements.txt ─────────────────
print("\n" + "=" * 60)
print("  Step 3: Installing Python packages...")
print("=" * 60)

req_file = f"{REPO_DIR}/requirements.txt"
if os.path.exists(req_file):
    print(f"[→] Installing from {req_file}...")
    # Install without -q so errors are visible
    _run(f"pip install -r {req_file}")
else:
    print("[!] requirements.txt not found — installing core packages directly...")
    packages = [
        "numpy>=1.24.0",
        "opencv-python-headless>=4.8.0",
        "trimesh>=4.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "open3d",
        "ultralytics>=8.0.0",
        "transformers>=4.30.0",
        "pillow>=10.0.0",
        "tqdm>=4.65.0",
        "mobile-sam",
    ]
    _run(f"pip install {' '.join(packages)}")

# ── Step 4: Install Depth Anything V3 from source ────────────────────────────
print("\n" + "=" * 60)
print("  Step 4: Installing Depth Anything V3...")
print("=" * 60)

if os.path.exists(DA3_DIR):
    print(f"[!] {DA3_DIR} already exists — pulling latest...")
    _run(f"git -C {DA3_DIR} pull", allow_fail=True)
else:
    _run(f"git clone {DA3_REPO} {DA3_DIR}")

_run(f"pip install -e {DA3_DIR}")

# Editable installs don't always register in the current process —
# add the source dir directly to sys.path as a guaranteed fallback
if DA3_DIR not in sys.path:
    sys.path.insert(0, DA3_DIR)
    print(f"[✓] Added {DA3_DIR} to sys.path (editable install fallback)")

# ── Step 5: Verify all imports ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Step 5: Verifying imports...")
print("=" * 60)

checks = [
    ("numpy",                   "import numpy"),
    ("cv2 (headless)",          "import cv2"),
    ("torch",                   "import torch; print(f'      CUDA available: {torch.cuda.is_available()}')"),
    ("torchvision",             "import torchvision"),
    ("trimesh",                 "import trimesh"),
    ("scipy",                   "import scipy"),
    ("sklearn",                 "import sklearn"),
    ("open3d",                  "import open3d"),
    ("ultralytics (YOLO)",      "from ultralytics import YOLO"),
    ("transformers (DINOv2)",   "from transformers import AutoModel"),
    ("PIL",                     "from PIL import Image"),
    ("tqdm",                    "from tqdm import tqdm"),
    ("mobile_sam",              "from mobile_sam import sam_model_registry"),
    ("depth_anything_3",        "from depth_anything_3.api import DepthAnything3"),
    ("config (project)",        "import config"),
    ("video_normalizer",        "from core import video_normalizer"),
]

passed, failed = 0, []
for name, stmt in checks:
    try:
        exec(stmt)
        print(f"  [✓] {name}")
        passed += 1
    except Exception as e:
        print(f"  [✗] {name}: {e}")
        failed.append(name)

print(f"\n{'=' * 60}")
print(f"  {passed}/{len(checks)} imports OK")
if failed:
    print(f"  Failed: {', '.join(failed)}")
    print("  → Check install errors above for each failed package.")
else:
    print("  All imports OK — environment is ready!")
print("=" * 60)

# ── Step 6: Quick smoke test ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Step 6: Smoke test — config paths & directories...")
print("=" * 60)

try:
    import config
    print(f"  BASE_DIR               : {config.BASE_DIR}")
    print(f"  RAW_DATA_DIR           : {config.RAW_DATA_DIR}")
    print(f"  PROCESSED_DATA_DIR     : {config.PROCESSED_DATA_DIR}")
    print(f"  OUTPUT_DIR             : {config.OUTPUT_DIR}")
    print(f"  VIDEO_TARGET_LONG_EDGE : {config.VIDEO_TARGET_LONG_EDGE}px")
    print(f"  VIDEO_TARGET_FPS       : {config.VIDEO_TARGET_FPS} FPS")
    print(f"  DEPTH_MODEL_ID         : {config.DEPTH_MODEL_ID}")
    print(f"  DEPTH_SAMPLE_STRIDE    : every {config.DEPTH_SAMPLE_STRIDE} frames")
    print(f"  DEPTH_MAX_FRAMES       : {config.DEPTH_MAX_FRAMES} frames max")
    print(f"\n  [✓] Config loaded. Pipeline is ready to run.")
    print(f"\n  To run Pass 1 (depth inference):")
    print(f"    python {REPO_DIR}/pointcloud/depth_inference.py <path_to_video.mov>")
    print(f"\n  To run Pass 2 (point cloud builder):")
    print(f"    python {REPO_DIR}/pointcloud/pointcloud_builder.py")
except Exception as e:
    print(f"  [✗] Config error: {e}")

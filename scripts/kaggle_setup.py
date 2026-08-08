# =============================================================================
# kaggle_setup.py — Scene Reconstruction Pipeline: Kaggle Environment Setup
#
# Paste each cell into a Kaggle notebook, or run the whole file as a script.
# Requires: Internet ON (Settings → Internet → On) and GPU accelerator enabled.
# =============================================================================

# ── Cell 1: System dependencies (libGL for open3d headless) ──────────────────
import subprocess, sys

def _run(cmd):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.stdout: print(result.stdout[-2000:])
    if result.returncode != 0:
        print(f"[WARN] Command exited {result.returncode}: {result.stderr[-500:]}")

print("=" * 60)
print("  Installing system libraries...")
print("=" * 60)
_run("apt-get install -y -q libgl1-mesa-glx libglib2.0-0 ffmpeg")

# ── Cell 2: Python dependencies ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Installing Python packages from requirements.txt...")
print("=" * 60)

import os
REPO_DIR = "/kaggle/working/SceneReconstruction"

if os.path.exists(f"{REPO_DIR}/requirements.txt"):
    _run(f"pip install -q -r {REPO_DIR}/requirements.txt")
else:
    # Fallback: install packages directly
    packages = [
        "numpy>=1.24.0",
        "opencv-python-headless>=4.8.0",
        "torch>=2.0.0",
        "torchvision",
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
    _run(f"pip install -q {' '.join(packages)}")

# ── Cell 3: Install Depth Anything V3 from source ────────────────────────────
print("\n" + "=" * 60)
print("  Installing Depth Anything V3...")
print("=" * 60)

DA3_DIR = "/kaggle/working/Depth-Anything-3"
DA3_REPO = "https://github.com/ByteDance-Seed/Depth-Anything-3.git"

if os.path.exists(DA3_DIR):
    print(f"[!] {DA3_DIR} already exists — pulling latest...")
    _run(f"git -C {DA3_DIR} pull")
else:
    _run(f"git clone {DA3_REPO} {DA3_DIR}")

_run(f"pip install -q -e {DA3_DIR}")

# ── Cell 4: Clone the SceneReconstruction project (if not already present) ───
print("\n" + "=" * 60)
print("  Setting up SceneReconstruction project...")
print("=" * 60)

SCENE_REPO = "https://github.com/TotallyNotMinh/SceneReconstruction.git"

if not os.path.exists(REPO_DIR):
    print(f"[→] Cloning SceneReconstruction into {REPO_DIR}...")
    _run(f"git clone {SCENE_REPO} {REPO_DIR}")
else:
    print(f"[✓] Project directory already exists at {REPO_DIR}")

# Add project root to Python path
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
    print(f"[✓] Added {REPO_DIR} to sys.path")

# ── Cell 5: Verify all imports ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Verifying imports...")
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
    print("  → Fix errors above before running the pipeline.")
else:
    print("  All imports OK — environment is ready!")
print("=" * 60)

# ── Cell 6: Quick smoke test ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Smoke test: config paths & directories...")
print("=" * 60)

try:
    import config
    print(f"  BASE_DIR           : {config.BASE_DIR}")
    print(f"  RAW_DATA_DIR       : {config.RAW_DATA_DIR}")
    print(f"  PROCESSED_DATA_DIR : {config.PROCESSED_DATA_DIR}")
    print(f"  OUTPUT_DIR         : {config.OUTPUT_DIR}")
    print(f"  VIDEO_TARGET_LONG_EDGE : {config.VIDEO_TARGET_LONG_EDGE}px")
    print(f"  VIDEO_TARGET_FPS       : {config.VIDEO_TARGET_FPS} FPS")
    print(f"\n  [✓] Config loaded. Pipeline is ready to run.")
    print(f"\n  To run Pass 1:")
    print(f"    python {REPO_DIR}/pointcloud/depth_inference.py <path_to_video.mov>")
    print(f"\n  To run Pass 2:")
    print(f"    python {REPO_DIR}/pointcloud/pointcloud_builder.py")
except Exception as e:
    print(f"  [✗] Config error: {e}")

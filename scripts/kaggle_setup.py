# =============================================================================
# kaggle_setup.py — Scene Reconstruction Pipeline: Kaggle Environment Setup
#
# Paste each cell into a Kaggle notebook, or run the whole file as a script.
# Requires: Internet ON (Settings → Internet → On) and GPU accelerator enabled.
#
# Strategy: Kaggle pre-installs torch, torchvision, numpy, scipy, sklearn,
# transformers, PIL, tqdm. We only install what is missing to avoid ABI conflicts
# from numpy upgrades breaking pre-compiled binaries.
# =============================================================================

import subprocess, sys, os, importlib

def _run(cmd, allow_fail=False):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout[-3000:])
    if result.returncode != 0:
        msg = result.stderr[-1500:] if result.stderr else "(no stderr)"
        level = "[WARN]" if allow_fail else "[ERROR]"
        print(f"{level} exit {result.returncode}: {msg}")
    return result.returncode

def _is_importable(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

REPO_DIR   = "/kaggle/working/SceneReconstruction"
DA3_DIR    = "/kaggle/working/Depth-Anything-3"
DA3_REPO   = "https://github.com/ByteDance-Seed/Depth-Anything-3.git"
SCENE_REPO = "https://github.com/TotallyNotMinh/SceneReconstruction.git"

# ── Step 1: System dependencies ───────────────────────────────────────────────
print("=" * 60)
print("  Step 1: System libraries")
print("=" * 60)
_run("apt-get install -y -q libgl1-mesa-glx libglib2.0-0 ffmpeg")

# ── Step 2: Clone SceneReconstruction FIRST ───────────────────────────────────
print("\n" + "=" * 60)
print("  Step 2: SceneReconstruction project")
print("=" * 60)

if not os.path.exists(REPO_DIR):
    print(f"[→] Cloning into {REPO_DIR}...")
    _run(f"git clone {SCENE_REPO} {REPO_DIR}")
else:
    print(f"[✓] Already exists at {REPO_DIR}")
    _run(f"git -C {REPO_DIR} pull", allow_fail=True)

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
    print(f"[✓] Added {REPO_DIR} to sys.path")

# ── Step 3: Install only packages Kaggle does NOT pre-install ─────────────────
# Kaggle pre-installs: torch, torchvision, numpy, scipy, sklearn, transformers,
#                      PIL, tqdm, cv2 — do NOT upgrade these or numpy ABI breaks.
print("\n" + "=" * 60)
print("  Step 3: Installing missing packages")
print("=" * 60)

missing_packages = []

if not _is_importable("cv2"):
    missing_packages.append("opencv-python-headless>=4.8.0")
    print("[→] cv2 missing")
else:
    print("[✓] cv2 already installed")

if not _is_importable("trimesh"):
    missing_packages.append("trimesh>=4.0.0")
    print("[→] trimesh missing")
else:
    print("[✓] trimesh already installed")

if not _is_importable("open3d"):
    missing_packages.append("open3d")
    print("[→] open3d missing")
else:
    print("[✓] open3d already installed")

if not _is_importable("ultralytics"):
    missing_packages.append("ultralytics>=8.0.0")
    print("[→] ultralytics missing")
else:
    print("[✓] ultralytics already installed")

if not _is_importable("mobile_sam"):
    # PyPI mobile-sam only supports Python <=3.11; use git source for Python 3.12
    missing_packages.append("git+https://github.com/ChaoningZhang/MobileSAM.git")
    print("[→] mobile_sam missing (will install from git)")
else:
    print("[✓] mobile_sam already installed")

if missing_packages:
    print(f"\n[→] Installing {len(missing_packages)} missing package(s)...")
    for pkg in missing_packages:
        _run(f"pip install '{pkg}'")
else:
    print("\n[✓] All base packages already installed")

# ── Step 4: Install Depth Anything V3 from source ────────────────────────────
print("\n" + "=" * 60)
print("  Step 4: Depth Anything V3")
print("=" * 60)

if not _is_importable("depth_anything_3"):
    if os.path.exists(DA3_DIR):
        print(f"[!] {DA3_DIR} exists — pulling latest...")
        _run(f"git -C {DA3_DIR} pull", allow_fail=True)
    else:
        _run(f"git clone {DA3_REPO} {DA3_DIR}")

    _run(f"pip install -e {DA3_DIR}")

    # Editable installs don't always register in the current process — add manually
    if DA3_DIR not in sys.path:
        sys.path.insert(0, DA3_DIR)
        print(f"[✓] Added {DA3_DIR} to sys.path (editable install fallback)")

    # Debug: show DA3 directory structure to diagnose path issues
    if not _is_importable("depth_anything_3"):
        print("\n[DEBUG] DA3 directory contents:")
        _run(f"ls -la {DA3_DIR}", allow_fail=True)
        _run(f"find {DA3_DIR} -name '*.py' -maxdepth 3 | head -20", allow_fail=True)
        print("[!] depth_anything_3 still not importable after install.")
        print("    → You may need to restart the Kaggle kernel and re-run.")
else:
    print("[✓] depth_anything_3 already importable")

# ── Step 5: Verify all imports ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Step 5: Verifying imports")
print("=" * 60)

checks = [
    ("numpy",                  "import numpy as np; print(f'      version: {np.__version__}')"),
    ("cv2 (headless)",         "import cv2"),
    ("torch",                  "import torch; print(f'      CUDA: {torch.cuda.is_available()}, version: {torch.__version__}')"),
    ("torchvision",            "import torchvision"),
    ("trimesh",                "import trimesh"),
    ("scipy",                  "import scipy"),
    ("sklearn",                "import sklearn"),
    ("open3d",                 "import open3d"),
    ("ultralytics (YOLO)",     "from ultralytics import YOLO"),
    ("transformers (DINOv2)",  "from transformers import AutoModel"),
    ("PIL",                    "from PIL import Image"),
    ("tqdm",                   "from tqdm import tqdm"),
    ("mobile_sam",             "from mobile_sam import sam_model_registry"),
    ("depth_anything_3",       "from depth_anything_3.api import DepthAnything3"),
    ("config (project)",       "import config"),
    ("video_normalizer",       "from core import video_normalizer"),
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
    if any(x in failed for x in ["torchvision", "sklearn", "open3d", "transformers"]):
        print("\n  ⚠️  numpy ABI error detected.")
        print("     numpy was likely upgraded mid-session, breaking pre-compiled packages.")
        print("     → Restart the Kaggle kernel and re-run this script.")
    if "depth_anything_3" in failed:
        print("\n  ⚠️  depth_anything_3 not found.")
        print("     → Restart the kernel after the install completes and re-run.")
else:
    print("  All imports OK — environment is ready!")
print("=" * 60)

# ── Step 6: Smoke test ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Step 6: Config smoke test")
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
    print(f"\n  [✓] Config OK — pipeline is ready to run.")
    print(f"\n  Pass 1: python {REPO_DIR}/pointcloud/depth_inference.py <video.mov>")
    print(f"  Pass 2: python {REPO_DIR}/pointcloud/pointcloud_builder.py")
except Exception as e:
    print(f"  [✗] Config error: {e}")

# -*- coding: utf-8 -*-
"""
scripts/kaggle_mask3d_run.py — Official JonasSchult/Mask3D Kaggle GPU Inference Script.

Extracts individual 3D point clouds for ALL physical objects in the room:
- Table / Desk -> obj_001_table_pointcloud.ply
- Individual Chairs -> obj_002_chair_pointcloud.ply, obj_003_chair_pointcloud.ply, etc.
- Monitor / TV on Wall -> obj_005_monitor_pointcloud.ply

How to run in a Kaggle GPU Notebook (T4 / P100 / A100):
-------------------------------------------------------
Cell 1 (Setup dependencies):
    !pip install -q ninja
    !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
    !pip install -q MinkowskiEngine -v --no-deps
    !git clone https://github.com/JonasSchult/Mask3D.git

Cell 2 (Download Official ScanNet200 Benchmark Checkpoint):
    !mkdir -p weights
    !wget https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_benchmark.ckpt -O weights/mask3d_scannet200_benchmark.ckpt

Cell 3 (Execute Object Extraction):
    !python scripts/kaggle_mask3d_run.py --world-pcd data/processed/world_pointcloud.ply
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add cloned Mask3D directory to sys.path
for cand_dir in [PROJECT_ROOT / "Mask3D", Path("Mask3D"), Path("/kaggle/working/Mask3D")]:
    if cand_dir.exists() and str(cand_dir) not in sys.path:
        sys.path.insert(0, str(cand_dir))

import argparse
import numpy as np
import trimesh
import json
import config


def download_mask3d_checkpoint(weights_dir: Path) -> Path:
    """Download official Mask3D ScanNet200 checkpoint if not present."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = weights_dir / "mask3d_scannet200_benchmark.ckpt"
    if not ckpt_path.exists():
        url = "https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_benchmark.ckpt"
        print(f"[Kaggle] Downloading official Mask3D weights from {url}...")
        try:
            cmd = f"wget {url} -O {ckpt_path}"
            subprocess.run(cmd, shell=True, check=True)
            print(f"[Kaggle] Downloaded weights -> {ckpt_path}")
        except Exception as e:
            print(f"[Kaggle] Wget failed ({e}). Please manually download: {url}")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Official JonasSchult/Mask3D 3D Instance Segmentation Runner")
    parser.add_argument("--world-pcd", type=str, default=str(config.PROCESSED_DATA_DIR / "world_pointcloud.ply"),
                        help="Path to world_pointcloud.ply file")
    parser.add_argument("--out-dir", type=str, default=str(config.PROCESSED_DATA_DIR / "objects"),
                        help="Output directory for extracted object point clouds")
    parser.add_argument("--checkpoint", type=str, default=str(config.WEIGHTS_DIR / "mask3d_scannet200_benchmark.ckpt"),
                        help="Path to Mask3D checkpoint")
    parser.add_argument("--planes-json", type=str, default=str(config.PROCESSED_DATA_DIR / "detected_planes.json"),
                        help="Path to detected_planes.json file")
    args = parser.parse_args()

    world_pcd = Path(args.world_pcd)
    if not world_pcd.exists():
        print(f"[Error] World point cloud file '{world_pcd}' does not exist!")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure checkpoint exists
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        ckpt_path = download_mask3d_checkpoint(config.WEIGHTS_DIR)

    # 2. Run Mask3D Extraction
    from spatial.object_extractor import extract_object_pointclouds

    print(f"\n=======================================================")
    print(f"[Kaggle Mask3D] Starting 3D Instance Segmentation on:")
    print(f"               Input PCD : {world_pcd.resolve()}")
    print(f"               Output Dir: {out_dir.resolve()}")
    print(f"               Weights   : {ckpt_path}")
    print(f"=======================================================\n")

    results = extract_object_pointclouds(
        world_pcd_path=world_pcd,
        plane_data_path=args.planes_json,
        checkpoint_path=ckpt_path if ckpt_path.exists() else None,
        out_dir=out_dir,
    )

    print(f"\n[Kaggle Mask3D] Segmentation finished successfully!")
    print(f"[Kaggle Mask3D] Total extracted objects: {len(results)}")
    for obj_id, obj_info in results.items():
        print(f"  * {obj_id} ({obj_info['label']}): {obj_info['point_count']:,} points -> {Path(obj_info['pcd_path']).name}")


if __name__ == "__main__":
    main()

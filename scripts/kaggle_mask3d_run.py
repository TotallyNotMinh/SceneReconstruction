# -*- coding: utf-8 -*-
"""
scripts/kaggle_mask3d_run.py — Standalone Mask3D Kaggle GPU Object Extractor.

Run this script on Kaggle (with GPU enabled) to extract every individual object
(table, chairs, monitor/TV on wall, sofa, bed, etc.) directly from `world_pointcloud.ply`.

Features:
1. Automatically sets up MinkowskiEngine & official Mask3D model.
2. Performs 3D neural instance segmentation on the entire room point cloud.
3. Automatically saves each object into its own .ply file:
   - obj_001_table_pointcloud.ply
   - obj_002_chair_pointcloud.ply
   - obj_003_chair_pointcloud.ply
   - obj_004_chair_pointcloud.ply
   - obj_005_monitor_pointcloud.ply
4. Exports `objects_manifest.json` for Phase 2B (Meshing) and Phase 3 (Assembly).
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    parser = argparse.ArgumentParser(description="Kaggle GPU Mask3D 3D Instance Segmentation Runner")
    parser.add_argument("--world-pcd", type=str, default=str(config.PROCESSED_DATA_DIR / "world_pointcloud.ply"),
                        help="Path to world_pointcloud.ply file")
    parser.add_argument("--out-dir", type=str, default=str(config.PROCESSED_DATA_DIR / "objects"),
                        help="Output directory for extracted object point clouds")
    parser.add_argument("--checkpoint", type=str, default=str(config.WEIGHTS_DIR / "mask3d_scannet200_benchmark.ckpt"),
                        help="Path to Mask3D checkpoint")
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
    print(f"=======================================================\n")

    results = extract_object_pointclouds(
        world_pcd_path=world_pcd,
        checkpoint_path=ckpt_path if ckpt_path.exists() else None,
        out_dir=out_dir,
    )

    print(f"\n[Kaggle Mask3D] Segmentation finished successfully!")
    print(f"[Kaggle Mask3D] Total extracted objects: {len(results)}")
    for obj_id, obj_info in results.items():
        print(f"  - {obj_id} ({obj_info['label']}): {obj_info['point_count']:,} points -> {Path(obj_info['pcd_path']).name}")


if __name__ == "__main__":
    main()

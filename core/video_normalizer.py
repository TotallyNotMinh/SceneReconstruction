# -*- coding: utf-8 -*-
"""
video_normalizer.py

Implementation of the Video Resolution Normalization & Intrinsics Scaling Specification.

Features:
  - Step 0: Orientation normalization via ffprobe metadata (displaymatrix/side_data/rotate).
            Physically rotates frames into display orientation and clears container rotate flags.
  - Step 1: Orientation-agnostic target scaling via get_scaled_resolution() forcing even dimensions
            and deriving exact post-rounding per-axis scale_x and scale_y.
  - Step 2: High-quality Lanczos rescaling with zero cropping or aspect ratio distortion.
  - Step 3: Camera intrinsics update using exact scale_x and scale_y.
  - Step 4: Pose & Dual Intrinsics management (preserves original capture K alongside rescaled K).
  - Step 5: Dataset Exception check (skips processing for pre-calibrated lowres datasets).
"""

import os
import sys
import json
import math
import shutil
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional


def get_scaled_resolution(width: int, height: int, target_long_edge: int = 720) -> Tuple[int, int, float, float]:
    """
    Step 1 — Compute Target Resolution (orientation-agnostic).

    Parameters
    ----------
    width            : Image/video width in display orientation (px)
    height           : Image/video height in display orientation (px)
    target_long_edge : Maximum dimension for the longer edge (default 720)

    Returns
    -------
    (new_width, new_height, scale_x, scale_y)
      - new_width, new_height: Even integers
      - scale_x, scale_y: Actual post-rounding scale ratios
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid input dimensions: {width}x{height}")

    scale = float(target_long_edge) / float(max(width, height))

    new_width  = int(round(width  * scale / 2.0) * 2)  # force even
    new_height = int(round(height * scale / 2.0) * 2)  # force even

    scale_x = new_width  / float(width)
    scale_y = new_height / float(height)

    return new_width, new_height, scale_x, scale_y


def rotate_intrinsics(intrinsics: list, width: int, height: int, rotation_deg: int) -> Tuple[list, int, int]:
    """
    Rotate intrinsic matrix K into the new physically rotated image coordinate system.

    Parameters
    ----------
    intrinsics   : [fx, fy, cx, cy]
    width, height: Pre-rotation image dimensions
    rotation_deg : 0, 90, 180, or 270 (degrees clockwise)

    Returns
    -------
    (rotated_intrinsics [fx_rot, fy_rot, cx_rot, cy_rot], oriented_width, oriented_height)
    """
    fx, fy, cx, cy = intrinsics
    rot = int(rotation_deg) % 360

    if rot == 0:
        return [fx, fy, cx, cy], width, height
    elif rot == 90:
        # 90 deg CW: (x', y') = (height - 1 - y, x)
        return [fy, fx, height - 1.0 - cy, cx], height, width
    elif rot == 180:
        # 180 deg: (x', y') = (width - 1 - x, height - 1 - y)
        return [fx, fy, width - 1.0 - cx, height - 1.0 - cy], width, height
    elif rot == 270:
        # 270 deg CW (90 deg CCW): (x', y') = (y, width - 1 - x)
        return [fy, fx, cy, width - 1.0 - cx], height, width
    else:
        return [fx, fy, cx, cy], width, height


def rescale_intrinsics(intrinsics: list, scale_x: float, scale_y: float) -> list:
    """
    Step 3 — Rescale Intrinsics using actual post-rounding per-axis scales.

    Parameters
    ----------
    intrinsics : [fx, fy, cx, cy]
    scale_x    : new_width / width
    scale_y    : new_height / height

    Returns
    -------
    [fx_new, fy_new, cx_new, cy_new]
    """
    fx, fy, cx, cy = intrinsics
    return [
        float(fx * scale_x),
        float(fy * scale_y),
        float(cx * scale_x),
        float(cy * scale_y),
    ]


def probe_video_orientation_and_meta(video_path: Path) -> Dict[str, Any]:
    """
    Step 0 — Detect Orientation using ffprobe (displaymatrix, side_data, stream_tags.rotate).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH. Install ffmpeg package.")

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate:stream_tags=rotate:stream_side_data_list=rotation",
        "-of", "json",
        str(video_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed to probe video metadata: {res.stderr}")

    data = json.loads(res.stdout)
    stream = data.get("streams", [{}])[0]

    raw_w = int(stream.get("width", 1920))
    raw_h = int(stream.get("height", 1080))

    r_fps = stream.get("r_frame_rate", "30/1")
    num, _, den = r_fps.partition("/")
    fps = float(num) / float(den) if den and float(den) > 0 else float(num)

    rotation = 0
    side_data_list = stream.get("side_data_list", [])
    for sd in side_data_list:
        if "rotation" in sd:
            rotation = int(float(sd["rotation"])) % 360
            break

    if rotation == 0:
        tags = stream.get("tags", {})
        if "rotate" in tags:
            rotation = int(float(tags["rotate"])) % 360

    display_w, display_h = (raw_h, raw_w) if rotation in (90, 270) else (raw_w, raw_h)

    return {
        "raw_width": raw_w,
        "raw_height": raw_h,
        "display_width": display_w,
        "display_height": display_h,
        "rotation": rotation,
        "fps": fps,
    }


def normalize_and_preprocess_video(
    video_path: Path,
    orig_intrinsics: Optional[list] = None,
    target_long_edge: int = 720,
    target_fps: int = 24,
    is_prepackaged_dataset: bool = False,
) -> Dict[str, Any]:
    """
    Executes the full Video Resolution Normalization & Intrinsics Scaling Pipeline.

    Returns dict containing:
      - processed_video_path (Path)
      - orig_resolution (tuple)
      - processed_resolution (tuple)
      - orig_intrinsics (list)
      - rotated_intrinsics (list)
      - processed_intrinsics (list)
      - scale_x (float)
      - scale_y (float)
      - fps (float)
    """
    video_path = Path(video_path)
    meta = probe_video_orientation_and_meta(video_path)

    display_w = meta["display_width"]
    display_h = meta["display_height"]
    orig_fps  = meta["fps"]
    rotation  = meta["rotation"]

    # Step 5: Dataset Exception check
    if is_prepackaged_dataset:
        print(f"[Dataset Exception] Skipping downscaling for pre-packaged dataset: {video_path.name}")
        k_orig = orig_intrinsics if orig_intrinsics else [1.2*max(display_w, display_h), 1.2*max(display_w, display_h), display_w/2.0, display_h/2.0]
        return {
            "processed_video_path": video_path,
            "orig_resolution": (display_w, display_h),
            "processed_resolution": (display_w, display_h),
            "orig_intrinsics": k_orig,
            "rotated_intrinsics": k_orig,
            "processed_intrinsics": k_orig,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "fps": orig_fps,
        }

    # Step 0: Base intrinsics at original capture resolution
    if orig_intrinsics is None:
        fx = fy = 1.2 * max(meta["raw_width"], meta["raw_height"])
        cx, cy = meta["raw_width"] / 2.0, meta["raw_height"] / 2.0
        k_base = [fx, fy, cx, cy]
    else:
        k_base = list(orig_intrinsics)

    # Step 0: Rotate intrinsics if physical orientation rotation needed
    k_rotated, oriented_w, oriented_h = rotate_intrinsics(k_base, meta["raw_width"], meta["raw_height"], rotation)

    # Step 1: Determine target resolution & per-axis scale
    new_w, new_h, scale_x, scale_y = get_scaled_resolution(display_w, display_h, target_long_edge=target_long_edge)

    # Step 3: Rescale intrinsics using post-rounding scale_x, scale_y
    k_processed = rescale_intrinsics(k_rotated, scale_x, scale_y)

    fps_out = min(target_fps, orig_fps)

    # Step 2 & 6: Execute FFmpeg resizing + orientation normalization
    suffix = video_path.suffix or ".mp4"
    out_dir = Path(os.environ.get("PROCESSED_DATA_DIR", str(video_path.parent)))
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / f"{video_path.stem}_normalized{suffix}"

    print(f"[Video Normalizer] Processing: '{video_path.name}'")
    print(f"                   Raw: {meta['raw_width']}x{meta['raw_height']} (rotation={rotation}°)")
    print(f"                   Display Normalized: {display_w}x{display_h}")
    print(f"                   Target Scaled: {new_w}x{new_h} @ {fps_out:.1f} FPS")
    print(f"                   Per-axis Scale: scale_x={scale_x:.6f}, scale_y={scale_y:.6f}")
    print(f"                   Original K : {[round(v, 2) for v in k_base]}")
    print(f"                   Rescaled K : {[round(v, 2) for v in k_processed]}")

    # Build FFmpeg filter chain
    vf_filters = []
    if rotation == 90:
        vf_filters.append("transpose=1")
    elif rotation == 180:
        vf_filters.append("transpose=2,transpose=2")
    elif rotation == 270:
        vf_filters.append("transpose=2")

    vf_filters.append(f"scale={new_w}:{new_h}:flags=lanczos")
    vf_chain = ",".join(vf_filters)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", vf_chain,
        "-r", str(fps_out),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-metadata:s:v:0", "rotate=0",  # Clear container rotation flag to prevent downstream double-rotation
        "-an",
        str(tmp_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg preprocessing & normalization failed:\n{result.stderr[-2000:]}")

    return {
        "processed_video_path": tmp_path,
        "orig_resolution": (display_w, display_h),
        "processed_resolution": (new_w, new_h),
        "orig_intrinsics": k_base,
        "rotated_intrinsics": k_rotated,
        "processed_intrinsics": k_processed,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "fps": fps_out,
    }

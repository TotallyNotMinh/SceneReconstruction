# -*- coding: utf-8 -*-
"""
pointcloud/depth_inference.py  —  Pass 1: video preprocessing + Depth Anything V3 inference
"""

import sys
import io
import os
import json
import math
import shutil
import subprocess
import tempfile
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from core import video_normalizer

try:
    import torch
    from PIL import Image
    from depth_anything_3.api import DepthAnything3
except ImportError as _e:
    sys.exit(
        "[ERROR] Missing required packages.\n"
        "  Install with: pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git\n"
        f"  Detail: {_e}"
    )

DEFAULT_NPZ_PATH = config.PROCESSED_DATA_DIR / "raw_depths.npz"
RAW_META_PATH    = config.PROCESSED_DATA_DIR / "ar_metadata.json"


def preprocess_video(
    video_path: Path,
    target_long_edge: int = config.VIDEO_TARGET_LONG_EDGE,
    target_fps: int = config.VIDEO_TARGET_FPS,
) -> tuple[Path, dict]:
    res = video_normalizer.normalize_and_preprocess_video(
        video_path, target_long_edge=target_long_edge, target_fps=target_fps
    )
    return res["processed_video_path"], res


def load_depth_anything_model(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = config.DEPTH_MODEL_ID
    print(f"[+] Loading Depth Anything V3 Base model ({model_id}) on {device}...")
    model = DepthAnything3.from_pretrained(model_id).to(device)
    model.eval()
    return model, device


def run_depth_inference(
    video_path: Path,
    sample_stride: int = config.DEPTH_SAMPLE_STRIDE,
    max_frames: int = config.DEPTH_MAX_FRAMES,
    npz_out: Path | None = None,
    use_fp16: bool = config.DEPTH_USE_FP16,
) -> Path:
    video_path = Path(video_path)
    if not video_path.exists():
        sys.exit(f"[ERROR] Video file not found: {video_path}")

    if npz_out is None:
        npz_out = DEFAULT_NPZ_PATH

    preprocessed_path: Path | None = None
    prep_meta: dict = {}
    try:
        preprocessed_path, prep_meta = preprocess_video(video_path)
        working_path = preprocessed_path
    except RuntimeError as exc:
        print(f"[WARNING] Video preprocessing failed — using original video.\n  {exc}")
        working_path = video_path

    model, device = load_depth_anything_model()

    cap = cv2.VideoCapture(str(working_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open processed video: {working_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    print(f"[+] Processing '{working_path.name}' ({w}x{h} @ {fps:.1f} FPS, {total_frames} frames)")

    orig_intrinsics = prep_meta.get("orig_intrinsics", [1.2 * max(w, h), 1.2 * max(w, h), w / 2.0, h / 2.0])
    rescaled_intrinsics = prep_meta.get("processed_intrinsics", [1.2 * max(w, h), 1.2 * max(w, h), w / 2.0, h / 2.0])
    scale_x = prep_meta.get("scale_x", 1.0)
    scale_y = prep_meta.get("scale_y", 1.0)

    intrinsics = np.array(rescaled_intrinsics, dtype=np.float64)
    sampled_indices = list(range(0, total_frames, sample_stride))[:max_frames]

    print(f"[+] Decoding {len(sampled_indices)} frames for multi-view DA3 inference...")
    pil_images = []
    rgb_frames = []
    valid_indices = []

    pbar = tqdm(total=len(sampled_indices), desc="[Decoding Frames]", unit="frame")
    for idx, target_idx in enumerate(sampled_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            pbar.update(1)
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frames.append(frame_rgb)
        pil_images.append(Image.fromarray(frame_rgb))
        valid_indices.append(idx)
        pbar.update(1)

    pbar.close()
    cap.release()

    if not pil_images:
        sys.exit("[ERROR] Failed to extract any valid frames from video.")

    fp16_str = " [FP16 autocast enabled]" if (use_fp16 and device == "cuda") else " [FP32]"
    print(f"[+] Pass 1 — running Depth Anything V3 joint multi-view inference on {len(pil_images)} frames{fp16_str}...")
    with torch.no_grad():
        if use_fp16 and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                result = model.inference(pil_images)
        else:
            result = model.inference(pil_images)

    raw_depths = result.depth
    raw_exts   = result.extrinsics
    raw_ixts   = result.intrinsics
    raw_confs  = getattr(result, "conf", None)
    proc_imgs  = getattr(result, "processed_images", None)

    frames_meta_raw = []
    for idx, i in enumerate(valid_indices):
        ext_mat = np.eye(4, dtype=np.float32)
        if raw_exts is not None:
            e = raw_exts[idx]
            ext_mat[:e.shape[0], :e.shape[1]] = e

        video_frame_id = sampled_indices[idx]
        frame_dict = {
            "index": int(i),
            "video_frame_index": int(video_frame_id),
            "pose_matrix": ext_mat.tolist(),
        }
        if raw_ixts is not None:
            ixt = raw_ixts[idx]
            frame_dict["fl_x"] = float(ixt[0, 0])
            frame_dict["fl_y"] = float(ixt[1, 1])
            frame_dict["cx"]   = float(ixt[0, 2])
            frame_dict["cy"]   = float(ixt[1, 2])
            frame_dict["w"]    = int(w)
            frame_dict["h"]    = int(h)
        frames_meta_raw.append(frame_dict)

    if prep_meta:
        meta_payload = {
            "w": w,
            "h": h,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "orig_intrinsics": orig_intrinsics,
            "processed_intrinsics": rescaled_intrinsics,
            "frames": frames_meta_raw,
        }
    else:
        meta_payload = {
            "w": w,
            "h": h,
            "fl_x": float(intrinsics[0]),
            "fl_y": float(intrinsics[1]),
            "cx": float(intrinsics[2]),
            "cy": float(intrinsics[3]),
            "frames": frames_meta_raw,
        }

    raw_meta_path = RAW_META_PATH
    raw_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)
    print(f"[+] Saved AR metadata ({len(frames_meta_raw)} frames) -> {raw_meta_path}")

    if preprocessed_path is not None and preprocessed_path.exists():
        try:
            preprocessed_path.unlink()
        except OSError:
            pass

    print(f"[+] Pass 1 done — {len(raw_depths)} frames processed.")

    npz_out = Path(npz_out)
    npz_out.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict = {
        "video_w":         np.int64(w),
        "video_h":         np.int64(h),
        "video_fps":       np.float64(fps),
        "scale_x":         np.float64(scale_x),
        "scale_y":         np.float64(scale_y),
        "intrinsics":      intrinsics,
        "orig_intrinsics": np.array(orig_intrinsics, dtype=np.float64),
        "frames_meta":     np.bytes_(json.dumps(frames_meta_raw, separators=(",", ":"))
                                     .encode("utf-8")),
    }
    for idx, i in enumerate(valid_indices):
        d_arr = raw_depths[idx].astype(np.float32)
        if d_arr.shape[:2] != (h, w):
            d_arr = cv2.resize(d_arr, (w, h), interpolation=cv2.INTER_NEAREST)
        arrays[f"depth_{i}"] = d_arr

        if raw_exts is not None:
            arrays[f"ext_{i}"] = raw_exts[idx].astype(np.float32)

        if raw_ixts is not None:
            ixt_arr = raw_ixts[idx].astype(np.float32)
            if proc_imgs is not None:
                orig_h, orig_w = proc_imgs[idx].shape[:2]
            else:
                orig_h, orig_w = h, w
            sx = w / orig_w
            sy = h / orig_h
            ixt_arr[0, 0] *= sx
            ixt_arr[1, 1] *= sy
            ixt_arr[0, 2] *= sx
            ixt_arr[1, 2] *= sy
            arrays[f"ixt_{i}"] = ixt_arr

        if raw_confs is not None:
            c_arr = raw_confs[idx].astype(np.float32)
            if c_arr.shape[:2] != (h, w):
                c_arr = cv2.resize(c_arr, (w, h), interpolation=cv2.INTER_NEAREST)
            arrays[f"conf_{i}"] = c_arr

        rgb_img = proc_imgs[idx] if proc_imgs is not None else rgb_frames[idx]
        if rgb_img.shape[:2] != (h, w):
            rgb_img = cv2.resize(rgb_img, (w, h), interpolation=cv2.INTER_LINEAR)
        arrays[f"rgb_{i}"] = rgb_img

    np.savez_compressed(str(npz_out), **arrays)
    print(f"[+] Raw depths + predicted camera poses saved ({len(raw_depths)} frames) -> {npz_out}")

    return npz_out


def generate_pcd_from_video(
    video_path: Path,
    sample_stride: int = 8,
    max_frames: int = 60,
    point_step: int = 4,
    return_depth_maps: bool = False,
    use_fp16: bool = config.DEPTH_USE_FP16,
):
    npz_path = run_depth_inference(
        video_path,
        sample_stride=sample_stride,
        max_frames=max_frames,
        use_fp16=use_fp16,
    )

    from pointcloud.pointcloud_builder import build_pointcloud_from_npz
    return build_pointcloud_from_npz(
        npz_path,
        point_step=point_step,
        return_depth_maps=return_depth_maps,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Depth Anything V3 Multi-View Inference")
    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument("npz_out", type=str, nargs="?", default=None, help="Path to output NPZ file (optional)")
    parser.add_argument("--fp16", action="store_true", default=config.DEPTH_USE_FP16, help="Run inference in FP16 mixed precision")
    args = parser.parse_args()

    _video = args.video
    _npz_out = Path(args.npz_out) if args.npz_out else None
    run_depth_inference(_video, npz_out=_npz_out, use_fp16=args.fp16)

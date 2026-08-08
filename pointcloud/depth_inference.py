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


def preprocess_video(video_path: Path, target_long_edge: int = 720, target_fps: int = 24) -> tuple[Path, dict]:
    res = video_normalizer.normalize_and_preprocess_video(
        video_path, target_long_edge=target_long_edge, target_fps=target_fps
    )
    return res["processed_video_path"], res


def load_depth_anything_model(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "depth-anything/da3-base"
    print(f"[+] Loading Depth Anything V3 Base model ({model_id}) on {device}...")
    model = DepthAnything3.from_pretrained(model_id).to(device)
    model.eval()
    return model, device


def run_depth_inference(
    video_path: Path,
    sample_stride: int = 8,
    max_frames: int = 30,
    npz_out: Path | None = None,
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

    cap          = cv2.VideoCapture(str(working_path))
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

    chunk_size = 8
    overlap = 2
    stride = chunk_size - overlap

    chunk_ranges = []
    cs = 0
    while cs < len(pil_images):
        ce = min(cs + chunk_size, len(pil_images))
        chunk_ranges.append((cs, ce))
        if ce >= len(pil_images):
            break
        cs += stride

    n_chunks = len(chunk_ranges)
    chunk_data = []

    for ci, (cs, ce) in enumerate(chunk_ranges):
        chunk = pil_images[cs:ce]
        print(f"[+] Pass 1 — chunk {ci + 1}/{n_chunks} ({len(chunk)} frames)...")
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            result = model.inference(chunk)

        _exts = result.extrinsics
        _ixts = result.intrinsics
        _confs = getattr(result, 'conf', None)
        _proc = getattr(result, 'processed_images', None)

        cd = {
            'start': cs,
            'end': ce,
            'depths': list(result.depth),
            'exts': [np.array(e) for e in _exts] if _exts is not None else None,
            'ixts': [np.array(e) for e in _ixts] if _ixts is not None else None,
            'confs': [np.array(e) for e in _confs] if _confs is not None else None,
            'proc_imgs': [np.array(e) for e in _proc] if _proc is not None else None,
        }
        chunk_data.append(cd)
        del result
        torch.cuda.empty_cache()

    def _ext_to_4x4(e):
        e = np.array(e, dtype=np.float64)
        if e.shape == (3, 4):
            m = np.eye(4, dtype=np.float64)
            m[:3, :4] = e
            return m
        return e

    has_all_exts = all(cd['exts'] is not None for cd in chunk_data)
    if has_all_exts and len(chunk_data) > 1:
        corrections = [np.eye(4, dtype=np.float64)]
        for i in range(1, len(chunk_data)):
            prev_cd = chunk_data[i - 1]
            curr_cd = chunk_data[i]

            overlap_idx_in_prev = curr_cd['start'] - prev_cd['start']
            ext_prev = _ext_to_4x4(prev_cd['exts'][overlap_idx_in_prev])
            ext_curr = _ext_to_4x4(curr_cd['exts'][0])

            ext_prev_aligned = ext_prev @ corrections[i - 1]
            correction = np.linalg.inv(ext_curr) @ ext_prev_aligned
            corrections.append(correction)

        for i, cd in enumerate(chunk_data):
            if cd['exts'] is not None:
                for j in range(len(cd['exts'])):
                    cd['exts'][j] = _ext_to_4x4(cd['exts'][j]) @ corrections[i]

    raw_depths = list(chunk_data[0]['depths'])
    raw_exts_acc = list(chunk_data[0]['exts']) if chunk_data[0]['exts'] is not None else []
    raw_ixts_acc = list(chunk_data[0]['ixts']) if chunk_data[0]['ixts'] is not None else []
    raw_confs_acc = list(chunk_data[0]['confs']) if chunk_data[0]['confs'] is not None else []
    proc_imgs_acc = list(chunk_data[0]['proc_imgs']) if chunk_data[0]['proc_imgs'] is not None else []

    for i in range(1, len(chunk_data)):
        cd = chunk_data[i]
        skip = chunk_data[i - 1]['end'] - cd['start']
        raw_depths.extend(cd['depths'][skip:])
        if cd['exts'] is not None:
            raw_exts_acc.extend(cd['exts'][skip:])
        if cd['ixts'] is not None:
            raw_ixts_acc.extend(cd['ixts'][skip:])
        if cd['confs'] is not None:
            raw_confs_acc.extend(cd['confs'][skip:])
        if cd['proc_imgs'] is not None:
            proc_imgs_acc.extend(cd['proc_imgs'][skip:])

    raw_exts  = raw_exts_acc if len(raw_exts_acc) > 0 else None
    raw_ixts  = raw_ixts_acc if len(raw_ixts_acc) > 0 else None
    raw_confs = raw_confs_acc if len(raw_confs_acc) > 0 else None
    proc_imgs = proc_imgs_acc if len(proc_imgs_acc) > 0 else None

    frames_meta_raw = []
    for idx, i in enumerate(valid_indices):
        ext_mat = np.eye(4, dtype=np.float32)
        if raw_exts is not None:
            e = raw_exts[idx]
            ext_mat[:e.shape[0], :e.shape[1]] = e

        video_frame_id = sampled_indices[idx]
        frame_fps = fps if (fps is not None and fps > 0) else 30.0
        ts_ns = int(video_frame_id * (1e9 / frame_fps))

        frames_meta_raw.append({
            "frame_id":       i,
            "video_frame_id": video_frame_id,
            "timestamp_ns":   ts_ns,
            "tracking_state": "TRACKING",
            "pose_matrix":    ext_mat.tolist(),
        })

    if preprocessed_path is not None and preprocessed_path.exists():
        print(f"[+] Normalized video kept at: {preprocessed_path}")

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
    for i in range(len(raw_depths)):
        d_arr = raw_depths[i].astype(np.float32)
        dh, dw = d_arr.shape[:2]
        sx, sy = w / float(dw), h / float(dh)

        if (dh, dw) != (h, w):
            d_arr = cv2.resize(d_arr, (w, h), interpolation=cv2.INTER_NEAREST)
        arrays[f"depth_{i}"] = d_arr

        if raw_exts is not None:
            arrays[f"ext_{i}"] = raw_exts[i].astype(np.float32)

        if raw_ixts is not None:
            ixt_arr = raw_ixts[i].astype(np.float32).copy()
            ixt_arr[0, 0] *= sx
            ixt_arr[1, 1] *= sy
            ixt_arr[0, 2] *= sx
            ixt_arr[1, 2] *= sy
            arrays[f"ixt_{i}"] = ixt_arr

        if raw_confs is not None:
            c_arr = raw_confs[i].astype(np.float32)
            if c_arr.shape[:2] != (h, w):
                c_arr = cv2.resize(c_arr, (w, h), interpolation=cv2.INTER_NEAREST)
            arrays[f"conf_{i}"] = c_arr

        rgb_img = proc_imgs[i] if proc_imgs is not None else rgb_frames[i]
        if rgb_img.shape[:2] != (h, w):
            rgb_img = cv2.resize(rgb_img, (w, h), interpolation=cv2.INTER_LINEAR)
        arrays[f"rgb_{i}"] = rgb_img

    np.savez_compressed(str(npz_out), **arrays)
    print(f"[+] Raw depths + predicted camera poses saved ({len(raw_depths)} frames) → {npz_out}")

    return npz_out


def generate_pcd_from_video(
    video_path: Path,
    sample_stride: int = 8,
    max_frames: int = 60,
    point_step: int = 4,
    return_depth_maps: bool = False,
):
    npz_path = run_depth_inference(
        video_path,
        sample_stride=sample_stride,
        max_frames=max_frames,
    )

    from pointcloud.pointcloud_builder import build_pointcloud_from_npz
    return build_pointcloud_from_npz(
        npz_path,
        point_step=point_step,
        return_depth_maps=return_depth_maps,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(
            "[ERROR] Please specify an input video file.\n"
            "Usage: python pointcloud/depth_inference.py <video_file> [raw_depths.npz]"
        )
    _video   = sys.argv[1]
    _npz_out = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
    run_depth_inference(_video, npz_out=_npz_out)

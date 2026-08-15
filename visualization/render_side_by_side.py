# -*- coding: utf-8 -*-
"""
visualization/render_side_by_side.py — Renders side-by-side RGB vs Depth video
"""

import os
import sys
import io
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT_VIDEO_PATH = config.OUTPUT_DIR / "demo_side_by_side_depth.mp4"


def load_depth_model(device=None):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as _e:
        sys.exit(f"[ERROR] Missing required libraries: torch, transformers, pillow.\n  Detail: {_e}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "depth-anything/Depth-Anything-V2-Base-hf"
    print(f"[+] Loading Depth Anything V2 Base model ({model_id}) on {device}...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model, device


def render_side_by_side(video_path: Path, output_path: Path, sample_stride: int = 1):
    video_path = Path(video_path)
    if not video_path.exists():
        sys.exit(f"[ERROR] Input video not found: {video_path}")

    processor, model, device = load_depth_model()

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w = w * 2
    out_h = h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    print(f"[+] Input Video : {video_path.name} ({w}x{h} @ {fps:.1f} FPS, {total_frames} frames)")
    print(f"[+] Output Video: {output_path.name} ({out_w}x{out_h} @ {fps:.1f} FPS, Stride={sample_stride})")

    pbar = tqdm(total=total_frames, desc="[Rendering Side-by-Side Depth]", unit="frame")

    frame_idx = 0
    cached_depth_color = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1

        if cached_depth_color is None or (frame_idx % sample_stride == 0):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            inputs = processor(images=pil_img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            depth_tensor = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_map = depth_tensor.cpu().numpy()

            d_min, d_max = depth_map.min(), depth_map.max()
            if d_max > d_min:
                depth_norm = ((depth_map - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
            else:
                depth_norm = np.zeros((h, w), dtype=np.uint8)

            cached_depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

        depth_color_bgr = cached_depth_color.copy()

        cv2.putText(frame_bgr, "Original Video (RGB)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(depth_color_bgr, "Depth Anything V2 Base", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        stacked_frame = np.hstack([frame_bgr, depth_color_bgr])

        progress_pct = (frame_idx / max(1, total_frames)) * 100.0
        hud_text = f" Frame: {frame_idx}/{total_frames} ({progress_pct:.1f}%) | Model: Depth-Anything-V2-Base-hf "
        cv2.rectangle(stacked_frame, (0, out_h - 30), (out_w, out_h), (20, 20, 20), -1)
        cv2.putText(stacked_frame, hud_text, (20, out_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(stacked_frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    print("\n==================================================================")
    print(f"[+] SUCCESS: Side-by-side depth video saved to:\n    {output_path}")
    print("==================================================================\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("[ERROR] Please specify an input video file on the command line.\nUsage: python visualization/render_side_by_side.py <video_file> [stride]")

    input_video = Path(sys.argv[1])
    if not input_video.exists():
        sys.exit(f"[ERROR] Specified input video does not exist: {input_video}")

    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out_name = f"{input_video.stem}_side_by_side_depth.mp4"
    out_path = config.OUTPUT_DIR / out_name
    render_side_by_side(input_video, out_path, sample_stride=stride)

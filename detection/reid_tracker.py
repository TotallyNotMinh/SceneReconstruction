# -*- coding: utf-8 -*-
"""
detection/reid_tracker.py — YOLO11 + BoT-SORT + DINOv2 Object Tracking & Re-ID
"""

import os
os.environ["XFORMERS_DISABLED"] = "1"
import sys
import glob
import json
import shutil
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import trimesh
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from core import video_normalizer

OUTPUT_BASE_DIR = config.PROCESSED_DATA_DIR / "reid_output"
SEGMENTED_DIR = OUTPUT_BASE_DIR / "segmented"
BBOX_UNCLIPPED_DIR = OUTPUT_BASE_DIR / "bbox_unclipped"
TRACKED_VIDEO_PATH = OUTPUT_BASE_DIR / "demo_tracked.mp4"

SIMILARITY_THRESHOLD = config.SIMILARITY_THRESHOLD
SAMPLE_EVERY_N_FRAMES = config.SAMPLE_EVERY_N_FRAMES
MIN_CROP_SIZE = config.MIN_CROP_SIZE
TARGET_CLASSES = config.TARGET_CLASSES


def get_color(identifier):
    hash_val = hash(str(identifier))
    r = (hash_val & 0xFF0000) >> 16
    g = (hash_val & 0x00FF00) >> 8
    b = hash_val & 0x0000FF
    return (int(r * 0.7 + 50), int(g * 0.7 + 50), int(b * 0.7 + 50))


def generate_architectural_room_layout(pcd_pts: np.ndarray, out_path: Path):
    if len(pcd_pts) == 0:
        return
    y_min = float(pcd_pts[:, 1].min())
    x_min, x_max = float(pcd_pts[:, 0].min()) - 0.2, float(pcd_pts[:, 0].max()) + 0.2
    z_min, z_max = float(pcd_pts[:, 2].min()) - 0.2, float(pcd_pts[:, 2].max()) + 0.2
    fw = x_max - x_min
    fd = z_max - z_min

    slab = trimesh.creation.box(extents=[fw, 0.12, fd])
    slab.apply_translation([(x_min + x_max) / 2.0, y_min - 0.06, (z_min + z_max) / 2.0])
    slab.invert()
    slab.export(str(out_path))


def process_video_or_folder(input_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Running on device: {device}")
    if device == "cuda":
        print(f"[+] GPU Name: {torch.cuda.get_device_name(0)}")

    model_path = config.YOLO_MODEL_PATH
    if not model_path.exists():
        model_path = Path("yolo11m-seg.pt")

    print(f"[+] Loading YOLO11 Medium Segmentation model ({model_path}) with BoT-SORT tracking...")
    yolo_model = YOLO(str(model_path))

    print("[+] Loading DINOv2 Small feature extractor (dinov2_vits14)...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    dinov2.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if SEGMENTED_DIR.exists():
        shutil.rmtree(SEGMENTED_DIR)
    if BBOX_UNCLIPPED_DIR.exists():
        shutil.rmtree(BBOX_UNCLIPPED_DIR)
    SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    BBOX_UNCLIPPED_DIR.mkdir(parents=True, exist_ok=True)

    def extract_embedding(crop_rgb):
        pil_img = Image.fromarray(crop_rgb)
        tensor_img = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = dinov2(tensor_img)
            feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.cpu().numpy().flatten()

    objects_db = []
    track_to_obj_map = {}
    total_detections = 0

    input_path = Path(input_path)
    is_video = input_path.is_file() and input_path.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')

    video_writer = None

    if is_video:
        prep_res = video_normalizer.normalize_and_preprocess_video(input_path, target_long_edge=720, target_fps=24)
        working_video_path = prep_res["processed_video_path"]

        print(f"[+] Processing Normalized Video Input with BoT-SORT: {working_video_path.name}")
        cap = cv2.VideoCapture(str(working_video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0

        tracked_video_path = OUTPUT_BASE_DIR / f"{input_path.stem}_tracked.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(tracked_video_path), fourcc, fps, (width, height))
        print(f"[+] Rendering tracked video to: {tracked_video_path} ({width}x{height} @ {fps:.1f} FPS)")
    else:
        print(f"[+] Processing Folder Input with BoT-SORT: {input_path}")
        image_paths = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
        total_frames = len(image_paths)

    processed_count = 0
    pbar = tqdm(total=total_frames, desc="[BoT-SORT + Re-ID]", unit="frame", dynamic_ncols=True)

    while True:
        if is_video:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_idx += 1
            frame_name = f"frame_{frame_idx:05d}"
            should_sample = (frame_idx % SAMPLE_EVERY_N_FRAMES == 0)
            curr_step = frame_idx
        else:
            if processed_count >= total_frames:
                break
            img_path = image_paths[processed_count]
            frame_name = img_path.stem
            frame_bgr = cv2.imread(str(img_path))
            processed_count += 1
            should_sample = ((processed_count - 1) % SAMPLE_EVERY_N_FRAMES == 0)
            curr_step = processed_count

        if frame_bgr is None:
            pbar.update(1)
            continue

        results = yolo_model.track(frame_bgr, persist=True, tracker="botsort.yaml", verbose=False)[0]
        frame_annotated = frame_bgr.copy()

        if results.masks is not None and results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            masks = results.masks.data.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy() if results.boxes.id is not None else None
            names = yolo_model.names

            for i, (box, cls_id) in enumerate(zip(boxes, classes)):
                label = names[int(cls_id)]
                if label not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                if w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
                    continue

                track_id = int(track_ids[i]) if track_ids is not None else None
                track_key = (label, track_id) if track_id is not None else None

                matched_obj = None

                if should_sample:
                    unclipped_crop_bgr = frame_bgr[y1:y2, x1:x2]

                    mask_full = cv2.resize(masks[i], (frame_bgr.shape[1], frame_bgr.shape[0]))
                    mask_3ch = (mask_full[:, :, np.newaxis] > 0.5).astype(np.uint8)
                    masked_frame = frame_bgr * mask_3ch
                    segmented_crop_bgr = masked_frame[y1:y2, x1:x2]

                    crop_alpha = (mask_full[y1:y2, x1:x2] > 0.5).astype(np.uint8) * 255
                    segmented_crop_bgra = cv2.cvtColor(segmented_crop_bgr, cv2.COLOR_BGR2BGRA)
                    segmented_crop_bgra[:, :, 3] = crop_alpha

                    crop_rgb = cv2.cvtColor(segmented_crop_bgr, cv2.COLOR_BGR2RGB)
                    feat = extract_embedding(crop_rgb)

                    if track_key is not None and track_key in track_to_obj_map:
                        matched_obj = track_to_obj_map[track_key]
                    else:
                        best_sim = -1.0
                        for obj in objects_db:
                            if obj['label'] != label:
                                continue
                            avg_feat = np.mean(obj['features'], axis=0)
                            avg_feat /= np.linalg.norm(avg_feat)
                            sim = np.dot(feat, avg_feat)

                            if sim > best_sim:
                                best_sim = sim
                                if sim >= SIMILARITY_THRESHOLD:
                                    matched_obj = obj

                        if matched_obj is not None and track_key is not None:
                            track_to_obj_map[track_key] = matched_obj

                    if matched_obj is not None:
                        matched_obj['features'].append(feat)
                        matched_obj['image_count'] += 1
                        matched_obj['views'].append({
                            "frame_id": curr_step - 1,
                            "bbox_px": [x1, y1, x2, y2]
                        })
                        obj_id = matched_obj['id']
                    else:
                        label_count = sum(1 for o in objects_db if o['label'] == label) + 1
                        obj_id = f"{label}_{label_count}"
                        matched_obj = {
                            'id': obj_id,
                            'label': label,
                            'features': [feat],
                            'image_count': 1,
                            'views': [{
                                "frame_id": curr_step - 1,
                                "bbox_px": [x1, y1, x2, y2]
                            }]
                        }
                        objects_db.append(matched_obj)
                        if track_key is not None:
                            track_to_obj_map[track_key] = matched_obj

                    filename = f"{matched_obj['image_count']:03d}_{frame_name}.png"

                    obj_seg_dir = SEGMENTED_DIR / obj_id
                    obj_seg_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(obj_seg_dir / filename), segmented_crop_bgra)

                    obj_bbox_dir = BBOX_UNCLIPPED_DIR / obj_id
                    obj_bbox_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(obj_bbox_dir / filename), unclipped_crop_bgr)

                    total_detections += 1
                else:
                    if track_key is not None and track_key in track_to_obj_map:
                        matched_obj = track_to_obj_map[track_key]

                display_id = matched_obj['id'] if matched_obj is not None else (f"{label} [#{track_id}]" if track_id else label)
                color = get_color(display_id)

                mask_full = cv2.resize(masks[i], (frame_bgr.shape[1], frame_bgr.shape[0])) > 0.5
                colored_mask = np.zeros_like(frame_annotated, dtype=np.uint8)
                colored_mask[mask_full] = color
                frame_annotated = cv2.addWeighted(frame_annotated, 1.0, colored_mask, 0.35, 0)

                cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 2)

                label_text = f" {display_id} "
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_y1 = max(0, y1 - text_h - baseline - 4)
                cv2.rectangle(frame_annotated, (x1, text_y1), (x1 + text_w, text_y1 + text_h + baseline + 4), color, -1)
                cv2.putText(frame_annotated, label_text, (x1, text_y1 + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        img_h, img_w = frame_annotated.shape[:2]
        progress_pct = curr_step / total_frames if total_frames > 0 else 0
        bar_h = 10
        cv2.rectangle(frame_annotated, (0, img_h - bar_h), (img_w, img_h), (30, 30, 30), -1)
        filled_w = int(img_w * progress_pct)
        if filled_w > 0:
            cv2.rectangle(frame_annotated, (0, img_h - bar_h), (filled_w, img_h), (0, 220, 255), -1)

        hud_text = f" Frame: {curr_step}/{total_frames} ({progress_pct*100:.1f}%) | Unique Objects: {len(objects_db)} "
        (th_w, th_h), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame_annotated, (10, 10), (10 + th_w, 15 + th_h + 10), (20, 20, 20), -1)
        cv2.rectangle(frame_annotated, (10, 10), (10 + th_w, 15 + th_h + 10), (0, 220, 255), 1)
        cv2.putText(frame_annotated, hud_text, (10, 15 + th_h), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if video_writer is not None:
            video_writer.write(frame_annotated)

        pbar.update(1)
        pbar.set_postfix({"Objects": len(objects_db), "Detections": total_detections})

    pbar.close()
    if is_video:
        cap.release()
    if video_writer is not None:
        video_writer.release()

    print("\n==========================================")
    print(f"[+] Re-ID Pipeline with BoT-SORT Complete!")
    print(f"[+] Total Detections Processed: {total_detections}")
    print(f"[+] Total Unique Objects Discovered: {len(objects_db)}")
    print(f"[+] Segmented Output Dir: {SEGMENTED_DIR}")
    print(f"[+] Bounding-Box Unclipped Output Dir: {BBOX_UNCLIPPED_DIR}")
    if is_video:
        print(f"[+] Tracked Annotated Video Saved To: {TRACKED_VIDEO_PATH}")
    print("==========================================\n")

    detections_b_dict = {}
    for obj in objects_db:
        obj_id = obj['id']
        detections_b_dict[obj_id] = {"associated_views": obj['views']}

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PROCESSED_DATA_DIR / "detections.json", "w", encoding="utf-8") as f:
        json.dump(detections_b_dict, f, indent=2)

    print(f"[+] Exported detections for {len(objects_db)} objects to {config.PROCESSED_DATA_DIR / 'detections.json'}")

    if is_video:
        from pointcloud.depth_inference import generate_pcd_from_video
        pts_clean, _, depth_maps = generate_pcd_from_video(
            input_path, return_depth_maps=True
        )
        generate_architectural_room_layout(pts_clean, config.PROCESSED_DATA_DIR / "room_layout.obj")

        depth_npz_path = config.PROCESSED_DATA_DIR / "depth_maps.npz"
        np.savez_compressed(
            str(depth_npz_path),
            **{str(k): v for k, v in depth_maps.items()}
        )
        print(f"[+] Depth maps saved ({len(depth_maps)} frames) -> {depth_npz_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("[ERROR] Please specify an input video or folder on the command line.\nUsage: python detection/reid_tracker.py <path_to_video_or_folder>")
    target_path = sys.argv[1]
    process_video_or_folder(target_path)

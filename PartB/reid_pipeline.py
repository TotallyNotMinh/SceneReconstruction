import os
import sys
import glob
import shutil
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

# Script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Output base directory
OUTPUT_BASE_DIR = os.path.join(SCRIPT_DIR, "reid_objects_output")
SEGMENTED_DIR = os.path.join(OUTPUT_BASE_DIR, "segmented")
BBOX_UNCLIPPED_DIR = os.path.join(OUTPUT_BASE_DIR, "bbox_unclipped")
TRACKED_VIDEO_PATH = os.path.join(OUTPUT_BASE_DIR, "demo_tracked.mp4")

# Hyperparameters
SIMILARITY_THRESHOLD = 0.8  # Cosine similarity cutoff for matching
SAMPLE_EVERY_N_FRAMES = 5    # Sample 1 every 5 frames for Re-ID feature extraction
MIN_CROP_SIZE = 100           # Minimum px size to ignore noise

# Furniture classes in COCO
TARGET_CLASSES = {
    'chair', 'couch', 'bed', 'dining table', 'tv', 'refrigerator',
    'oven'}


def get_color(identifier):
    """Generate a consistent RGB color tuple from an object identifier string or int."""
    hash_val = hash(str(identifier))
    r = (hash_val & 0xFF0000) >> 16
    g = (hash_val & 0x00FF00) >> 8
    b = hash_val & 0x0000FF
    return (int(r * 0.7 + 50), int(g * 0.7 + 50), int(b * 0.7 + 50))

def process_video_or_folder(input_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Running on device: {device}")
    if device == "cuda":
        print(f"[+] GPU Name: {torch.cuda.get_device_name(0)}")

    model_path = os.path.join(SCRIPT_DIR, "yolo11m-seg.pt")
    if not os.path.exists(model_path):
        model_path = "yolo11m-seg.pt"

    print(f"[+] Loading YOLO11 Medium Segmentation model ({model_path}) with BoT-SORT tracking...")
    yolo_model = YOLO(model_path)

    print("[+] Loading DINOv2 Small feature extractor (dinov2_vits14)...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    dinov2.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Re-create fresh output directories
    if os.path.exists(SEGMENTED_DIR):
        shutil.rmtree(SEGMENTED_DIR)
    if os.path.exists(BBOX_UNCLIPPED_DIR):
        shutil.rmtree(BBOX_UNCLIPPED_DIR)
    os.makedirs(SEGMENTED_DIR, exist_ok=True)
    os.makedirs(BBOX_UNCLIPPED_DIR, exist_ok=True)

    def extract_embedding(crop_rgb):
        pil_img = Image.fromarray(crop_rgb)
        tensor_img = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = dinov2(tensor_img)
            feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.cpu().numpy().flatten()

    objects_db = []
    track_to_obj_map = {}  # Maps (label, track_id) -> obj in objects_db
    total_detections = 0

    # Determine input type (Video file or Image directory)
    is_video = os.path.isfile(input_path) and input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    video_writer = None

    if is_video:
        print(f"[+] Processing Video Input with BoT-SORT: {input_path}")
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(TRACKED_VIDEO_PATH, fourcc, fps, (width, height))
        print(f"[+] Rendering tracked video to: {TRACKED_VIDEO_PATH} ({width}x{height} @ {fps:.1f} FPS)")
    else:
        print(f"[+] Processing Folder Input with BoT-SORT: {input_path}")
        image_paths = sorted(glob.glob(os.path.join(input_path, "*.png")) + glob.glob(os.path.join(input_path, "*.jpg")))
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
            frame_name = os.path.basename(img_path).split('.')[0]
            frame_bgr = cv2.imread(img_path)
            processed_count += 1
            should_sample = ((processed_count - 1) % SAMPLE_EVERY_N_FRAMES == 0)
            curr_step = processed_count

        if frame_bgr is None:
            pbar.update(1)
            continue

        # Run YOLO with BoT-SORT tracking
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

                # Perform sampling for crops & DINOv2 embeddings
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
                        obj_id = matched_obj['id']
                    else:
                        label_count = sum(1 for o in objects_db if o['label'] == label) + 1
                        obj_id = f"{label}_{label_count}"
                        matched_obj = {
                            'id': obj_id,
                            'label': label,
                            'features': [feat],
                            'image_count': 1
                        }
                        objects_db.append(matched_obj)
                        if track_key is not None:
                            track_to_obj_map[track_key] = matched_obj

                    filename = f"{matched_obj['image_count']:03d}_{frame_name}.png"

                    obj_seg_dir = os.path.join(SEGMENTED_DIR, obj_id)
                    os.makedirs(obj_seg_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(obj_seg_dir, filename), segmented_crop_bgra)

                    obj_bbox_dir = os.path.join(BBOX_UNCLIPPED_DIR, obj_id)
                    os.makedirs(obj_bbox_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(obj_bbox_dir, filename), unclipped_crop_bgr)

                    total_detections += 1
                else:
                    # Non-sampled frame: lookup existing track_to_obj mapping
                    if track_key is not None and track_key in track_to_obj_map:
                        matched_obj = track_to_obj_map[track_key]

                # Draw Bounding Box & Label on frame_annotated
                display_id = matched_obj['id'] if matched_obj is not None else (f"{label} [#{track_id}]" if track_id else label)
                color = get_color(display_id)

                # Draw semi-transparent mask overlay
                mask_full = cv2.resize(masks[i], (frame_bgr.shape[1], frame_bgr.shape[0])) > 0.5
                colored_mask = np.zeros_like(frame_annotated, dtype=np.uint8)
                colored_mask[mask_full] = color
                frame_annotated = cv2.addWeighted(frame_annotated, 1.0, colored_mask, 0.35, 0)

                # Draw Bounding Box
                cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 2)

                # Draw Label Background & Text
                label_text = f" {display_id} "
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_y1 = max(0, y1 - text_h - baseline - 4)
                cv2.rectangle(frame_annotated, (x1, text_y1), (x1 + text_w, text_y1 + text_h + baseline + 4), color, -1)
                cv2.putText(frame_annotated, label_text, (x1, text_y1 + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw HUD Box and Visual Progress Bar on Video Frame
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

    for obj in objects_db:
        print(f"  - {obj['id']}: {obj['image_count']} views collected")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        demo_in_script_dir = os.path.join(SCRIPT_DIR, "demo.mp4")
        target_path = demo_in_script_dir if os.path.exists(demo_in_script_dir) else "demo.mp4"

    process_video_or_folder(target_path)




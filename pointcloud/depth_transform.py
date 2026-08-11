import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config


def extract_timestamp(path: Path) -> float:
    """
    Tên file ARKit dạng '{sessionID}_{timestamp}.png',
    ví dụ '41048190_3606.704.png' -> 3606.704
    """
    stem = path.stem  # bỏ đuôi .png -> "41048190_3606.704"
    ts_str = stem.rsplit("_", 1)[-1]
    try:
        return float(ts_str)
    except ValueError:
        raise ValueError(
            f"Không parse được timestamp từ tên file: '{path.name}'. "
            f"Kỳ vọng dạng 'sessionID_timestamp.png'."
        )
def remove_flying_pixels(depth: np.ndarray, rel_threshold: float = 0.15, median_ksize: int = 5) -> np.ndarray:
    valid_mask = depth > 0
    depth_for_median = depth.copy()
    depth_for_median[~valid_mask] = 0

    # median blur cần input uint8/uint16/float32, ksize lẻ
    local_median = cv2.medianBlur(depth_for_median.astype(np.float32), median_ksize)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.abs(depth - local_median) / np.maximum(local_median, 1e-6)

    edge_mask = (rel_diff > rel_threshold) & valid_mask
    depth_clean = depth.copy()
    depth_clean[edge_mask] = 0.0
    return depth_clean

import json

def process_to_single_npz(INPUT_DIR: Path = PROJECT_ROOT / "highres_depth"):
    OUTPUT_FILE = config.PROCESSED_DATA_DIR / "raw_depths.npz"
    DEBUG_FILE = config.PROCESSED_DATA_DIR / "depth_debug_stats.json"
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(INPUT_DIR.glob("*.png"), key=extract_timestamp)
    if not image_files:
        print(f"Không tìm thấy file ảnh .png nào trong {INPUT_DIR}")
        return

    print(f"Bắt đầu gộp {len(image_files)} file ảnh (sort theo timestamp)...")

    data_to_save = {}
    file_names = []
    ref_shape = None
    index = 0
    debug_stats = []  # lưu thống kê từng frame để xem lại sau

    for img_path in image_files:
        depth_map = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if depth_map is None:
            print(f"  [LỖI] Không thể đọc file: {img_path.name} -> bỏ qua")
            continue

        depth_array = np.asarray(depth_map, dtype=np.float32)

        if ref_shape is None:
            ref_shape = depth_array.shape
        elif depth_array.shape != ref_shape:
            print(f"  [CẢNH BÁO] {img_path.name} shape lệch -> bỏ qua")
            continue

        # thống kê TRƯỚC khi lọc, để biết rõ scale/đơn vị thật sự
        valid = depth_array[depth_array > 0]
        stats_before = {
            "file": img_path.name,
            "dtype_goc": str(depth_map.dtype),
            "min": float(valid.min()) if valid.size else None,
            "max": float(valid.max()) if valid.size else None,
            "mean": float(valid.mean()) if valid.size else None,
            "n_valid_px": int(valid.size),
        }

        depth_clean = remove_flying_pixels(depth_array, rel_threshold=0.15)

        n_valid_after = int((depth_clean > 0).sum())
        stats_before["n_valid_px_after_filter"] = n_valid_after
        stats_before["ty_le_giu_lai"] = round(n_valid_after / max(valid.size, 1), 3)
        debug_stats.append(stats_before)

        data_to_save[f"depth_{index}"] = depth_clean
        file_names.append(img_path.name)
        index += 1

    if data_to_save:
        data_to_save["filenames"] = np.asarray(file_names)
        np.savez_compressed(OUTPUT_FILE, **data_to_save)

        with open(DEBUG_FILE, "w") as f:
            json.dump(debug_stats, f, indent=2)

        print(f"\n[THÀNH CÔNG] Đã lưu {len(file_names)} depth map vào {OUTPUT_FILE}")
        print(f"[DEBUG] Thống kê chi tiết từng frame: {DEBUG_FILE}")
    else:
        print("[LỖI] Không có depth map hợp lệ nào để lưu.")


if __name__ == "__main__":
    process_to_single_npz()

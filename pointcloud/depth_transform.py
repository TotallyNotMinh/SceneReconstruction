import re
import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config


def extract_timestamp(path: Path) -> float:
    stem = path.stem
    ts_str = stem.rsplit("_", 1)[-1]
    try:
        return float(ts_str)
    except ValueError:
        raise ValueError(
            f"Không parse được timestamp từ tên file: '{path.name}'. "
            f"Kỳ vọng dạng 'sessionID_timestamp.png'."
        )


def process_to_single_npz(INPUT_DIR: Path = PROJECT_ROOT / "highres_depth"):
    OUTPUT_FILE = config.PROCESSED_DATA_DIR / "raw_depths.npz"
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(INPUT_DIR.glob("*.png"), key=extract_timestamp)
    if not image_files:
        print(f"Không tìm thấy file ảnh .png nào trong {INPUT_DIR}")
        return

    print(f"[+] Bắt đầu gộp {len(image_files)} file ảnh (sort theo timestamp)...")

    depths, file_names, timestamps = [], [], []
    ref_shape = None

    for img_path in image_files:
        depth_map = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if depth_map is None:
            print(f"  [LỖI] Không thể đọc file: {img_path.name}")
            continue

        depth_array = np.asarray(depth_map, dtype=np.float32)

        if ref_shape is None:
            ref_shape = depth_array.shape
        elif depth_array.shape != ref_shape:
            print(
                f"  [CẢNH BÁO] {img_path.name} có shape {depth_array.shape}, "
                f"khác với shape tham chiếu {ref_shape} -> bỏ qua file này."
            )
            continue

        ts = extract_timestamp(img_path)
        depths.append(depth_array)
        file_names.append(img_path.name)
        timestamps.append(ts)

        print(f"  -> Đã đọc: {img_path.name} (t={ts:.3f}s, shape={depth_array.shape}, dtype gốc={depth_map.dtype})")

    if not depths:
        print("[LỖI] Không có depth map hợp lệ nào để lưu.")
        return

    depths_arr = np.stack(depths, axis=0)
    timestamps_arr = np.asarray(timestamps, dtype=np.float64)
    filenames_arr = np.asarray(file_names)

    if not np.all(np.diff(timestamps_arr) > 0):
        print("[CẢNH BÁO] Timestamp không tăng dần đơn điệu — kiểm tra lại dữ liệu nguồn.")

    np.savez_compressed(
        OUTPUT_FILE,
        depths=depths_arr,
        filenames=filenames_arr,
        timestamps=timestamps_arr,
    )

    print(f"\n[THÀNH CÔNG] Đã lưu {depths_arr.shape[0]} depth map, shape mỗi ảnh {depths_arr.shape[1:]}")
    print(f"  Khoảng timestamp: {timestamps_arr[0]:.3f}s -> {timestamps_arr[-1]:.3f}s")
    print(f"  File: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_to_single_npz()

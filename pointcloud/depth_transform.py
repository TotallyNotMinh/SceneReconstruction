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
def remove_flying_pixels(depth: np.ndarray, threshold: float = 0.05, kernel_size: int = 3) -> np.ndarray:
    depth_clean = depth.copy()

    # tính gradient theo 2 hướng
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=kernel_size)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    edge_mask = grad_mag > threshold
    depth_clean[edge_mask] = 0.0

    return depth_clean
def process_to_single_npz(INPUT_DIR: Path = PROJECT_ROOT / "highres_depth"):
    OUTPUT_FILE = config.PROCESSED_DATA_DIR / "raw_depths.npz"
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(INPUT_DIR.glob("*.png"), key=extract_timestamp)
    if not image_files:
        print(f"Không tìm thấy file ảnh .png nào trong {INPUT_DIR}")
        return

    print(f"Bắt đầu gộp {len(image_files)} file ảnh (sort theo timestamp)...")

    data_to_save = {}
    file_names = []
    ref_shape = None
    index = 0  # index thực tế được gán, chỉ tăng khi đọc file thành công

    for img_path in image_files:
        depth_map = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if depth_map is None:
            print(f"  [LỖI] Không thể đọc file: {img_path.name} -> bỏ qua")
            continue

        depth_array = np.asarray(depth_map, dtype=np.float32)
        depth_array = remove_flying_pixels(depth_array, threshold=0.05)
        if ref_shape is None:
            ref_shape = depth_array.shape
        elif depth_array.shape != ref_shape:
            print(
                f"  [CẢNH BÁO] {img_path.name} có shape {depth_array.shape}, "
                f"khác shape tham chiếu {ref_shape} -> bỏ qua file này."
            )
            continue

        data_to_save[f"depth_{index}"] = depth_array
        file_names.append(img_path.name)
        print(f"  -> Đã đọc: {img_path.name} (t={extract_timestamp(img_path):.3f}s, lưu thành depth_{index})")
        index += 1

    if data_to_save:
        data_to_save["filenames"] = np.asarray(file_names)
        np.savez_compressed(OUTPUT_FILE, **data_to_save)
        print(f"\n[THÀNH CÔNG] Đã lưu {len(file_names)} mảng depth riêng biệt vào:")
        print(f"  {OUTPUT_FILE}")
    else:
        print("[LỖI] Không có depth map hợp lệ nào để lưu.")


if __name__ == "__main__":
    process_to_single_npz()

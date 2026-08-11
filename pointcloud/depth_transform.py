import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

def process_to_single_npz(INPUT_DIR = PROJECT_ROOT / "highres_depth"):
    OUTPUT_FILE = config.PROCESSED_DATA_DIR / "raw_depths.npz"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_depths = []
    file_names = []

    image_files = sorted(list(INPUT_DIR.glob("*.png")))

    if not image_files:
        print(f"Không tìm thấy file ảnh .png nào trong {INPUT_DIR}")
        return

    print(f"Bắt đầu gộp {len(image_files)} file ảnh...")

    for img_path in image_files:
        depth_map = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if depth_map is not None:
            depth_array = np.array(depth_map, dtype=np.float32)

            all_depths.append(depth_array)
            file_names.append(img_path.name)
            print(f"  -> Đã đọc: {img_path.name}")
        else:
            print(f"  [LỖI] Không thể đọc file: {img_path.name}")

    if all_depths:
        stacked_depths = np.stack(all_depths)

        np.savez_compressed(
            OUTPUT_FILE,
            depths=stacked_depths,
            filenames=file_names
        )

        print(f"\n[THÀNH CÔNG] Đã lưu mảng với kích thước {stacked_depths.shape} vào:")
        print(f"  {OUTPUT_FILE}")

if __name__ == "__main__":
    process_to_single_npz()

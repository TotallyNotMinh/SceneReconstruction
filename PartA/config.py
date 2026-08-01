from pathlib import Path

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "outputs"
MESH_INPUT_DIR = DATA_DIR / "meshes_from_c"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MESH_INPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- THÔNG SỐ HÌNH HỌC & LỌC ---
VOXEL_SIZE_PCD = 0.02  # Downsample Point Cloud 2cm
DBSCAN_EPS = 0.05  # Khoảng cách gom cụm 5cm
DBSCAN_MIN_SAMPLES = 10
OCCLUSION_MIN_CONSENSUS = 0.6  # Score voting >= 60%
MAX_DISTORTION_THRESH = 1.25  # Ngưỡng lệch tỉ lệ Mesh

# --- HỆ TỌA ĐỘ CHUẨN ---
WORLD_UNIT = "meters"
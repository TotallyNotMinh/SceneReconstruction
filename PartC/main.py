import os
import subprocess
import glob
import shutil
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PART_B_DIR = os.path.join(ROOT_DIR, 'PartB')
input_directory = os.path.join(PART_B_DIR, 'reid_objects_output')
instant_mesh_directory = os.path.join(SCRIPT_DIR, 'InstantMesh')
source_output_dir = os.path.join(instant_mesh_directory, 'outputs')
final_destination_dir = os.path.join(ROOT_DIR, 'Final_Outputs')


def run_dual_gpu(directory=input_directory, project_dir=instant_mesh_directory):
    os.chdir(project_dir)
    valid_extensions = ('*.jpg', '*.jpeg', '*.png')
    image_files = []

    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))

    if not image_files:
        print(f"Error: No images found in '{directory}'.")
        return

    print(f"Found {len(image_files)} images in total. Starting processing...")

    gpu_queue = Queue()
    gpu_queue.put(0)
    gpu_queue.put(1)

    def process_image(img_path):
        gpu_id = gpu_queue.get()
        file_name = os.path.basename(img_path)
        print(f"-> Processing {file_name} on GPU {gpu_id}...")

        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python run.py configs/instant-mesh-base.yaml {img_path} --save_video"
        subprocess.run(cmd, shell=True)

        gpu_queue.put(gpu_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(process_image, image_files)

    print("Completed: Processed all images!")

os.makedirs(final_destination_dir, exist_ok=True)
def extract_outputs():
    if not os.path.exists(source_output_dir):
        print(f"Error: can't find {source_output_dir}. Maybe it not in here?.")
        return

    for item in os.listdir(source_output_dir):
        source_item_path = os.path.join(source_output_dir, item)
        dest_item_path = os.path.join(final_destination_dir, item)

        try:
            shutil.move(source_item_path, dest_item_path)
            print(f"Đã di chuyển: {item} -> {final_destination_dir}")
        except Exception as e:
            print(f"Error when transfer {item}: {e}")

    print("Success!")

if __name__ == "__main__":
    print("Starting the Process :")
    run_dual_gpu()
    extract_outputs()
    print("Done!")

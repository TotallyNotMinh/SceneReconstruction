def run_depth_inference(*args, **kwargs):
    from .depth_inference import run_depth_inference as _fn
    return _fn(*args, **kwargs)


def generate_pcd_from_video(*args, **kwargs):
    from .depth_inference import generate_pcd_from_video as _fn
    return _fn(*args, **kwargs)


from .pointcloud_builder import build_pointcloud_from_npz
from .mesh_reconstructor import mesh_pointcloud


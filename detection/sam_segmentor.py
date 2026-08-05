# detection/sam_segmentor.py
"""
SAMSegmentor — thin wrapper around MobileSAM for per-object pixel masking.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import config

try:
    import torch
    from mobile_sam import sam_model_registry, SamPredictor as _SamPredictor
    _MOBILE_SAM_AVAILABLE = True
except ImportError:
    _MOBILE_SAM_AVAILABLE = False


class SAMSegmentor:
    """Wraps MobileSAM to produce a per-frame binary object mask from a bbox prompt."""

    def __init__(self, checkpoint_path: Path = config.SAM_CHECKPOINT_PATH, device: str = "cpu"):
        self._device = device
        self._predictor: Optional[object] = None

        if not _MOBILE_SAM_AVAILABLE:
            print(
                "[SAMSegmentor] WARNING: 'mobile_sam' not installed. "
                "Falling back to bbox-fill masks. "
                "Run: pip install mobile-sam"
            )
            return

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            checkpoint_path = self._download_checkpoint(checkpoint_path)

        sam = sam_model_registry["vit_t"](checkpoint=str(checkpoint_path))
        sam.to(device=device)
        sam.eval()
        self._predictor = _SamPredictor(sam)
        print(f"[SAMSegmentor] MobileSAM loaded on {device}.")

    def segment_object(
        self,
        frame_rgb: np.ndarray,
        bbox_xyxy: list,
    ) -> np.ndarray:
        if self._predictor is None:
            return self._bbox_fill_mask(frame_rgb.shape[:2], bbox_xyxy)

        self._predictor.set_image(frame_rgb)

        box = np.array(bbox_xyxy, dtype=np.float32)
        masks, scores, _ = self._predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=False,
        )
        return masks[0]

    @staticmethod
    def _bbox_fill_mask(shape_hw: tuple, bbox_xyxy: list) -> np.ndarray:
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=bool)
        x0, y0, x1, y1 = (
            max(0, int(bbox_xyxy[0])),
            max(0, int(bbox_xyxy[1])),
            min(w, int(bbox_xyxy[2])),
            min(h, int(bbox_xyxy[3])),
        )
        mask[y0:y1, x0:x1] = True
        return mask

    @staticmethod
    def _download_checkpoint(dest: Path) -> Path:
        url = (
            "https://huggingface.co/spaces/dhkim2810/MobileSAM"
            "/resolve/main/mobile_sam.pt"
        )
        import urllib.request
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[SAMSegmentor] Downloading MobileSAM weights to {dest} ...")
        urllib.request.urlretrieve(url, str(dest))
        print("[SAMSegmentor] Download complete.")
        return dest

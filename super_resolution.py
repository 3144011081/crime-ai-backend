"""
Super-Resolution & High-Altitude Pixel Zooming Module
Incorporates FSRCNN_x3 neural super-resolution model (models/FSRCNN_x3.pb)
and 3x3 full-frame pixel grid tile zooming for distant aerial surveillance.
"""

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
# pyrefly: ignore [missing-import]
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FSRCNN_MODEL_PATH = PROJECT_ROOT / "models" / "FSRCNN_x3.pb"

# Global loaded FSRCNN model reference
_FSRCNN_NET = None
_FSRCNN_LOADED = False


def load_fsrcnn_model():
    """
    Attempt to load FSRCNN_x3.pb super-resolution model.
    """
    global _FSRCNN_NET, _FSRCNN_LOADED
    if _FSRCNN_LOADED:
        return _FSRCNN_NET

    if FSRCNN_MODEL_PATH.exists():
        try:
            net = cv2.dnn.readNet(str(FSRCNN_MODEL_PATH))
            _FSRCNN_NET = net
            _FSRCNN_LOADED = True
            print(f"[SuperRes] Loaded FSRCNN_x3.pb neural super-resolution model from {FSRCNN_MODEL_PATH}")
        except Exception as e:
            print(f"[SuperRes] Note loading FSRCNN_x3.pb: {e}. Falling back to Lanczos4 + Unsharp High-Frequency Resolution Enhancement.")
            _FSRCNN_LOADED = True
            _FSRCNN_NET = None
    else:
        _FSRCNN_LOADED = True
        _FSRCNN_NET = None

    return _FSRCNN_NET


def enhance_resolution_3x(img, scale_factor=3):
    """
    Enhance low-resolution / high-altitude crop using 3x Super-Resolution.
    Uses FSRCNN_x3.pb when possible, or Lanczos4 + Unsharp High-Frequency sharpening.

    Args:
        img: OpenCV BGR image
        scale_factor: super-resolution multiplier (default 3x)

    Returns:
        Enhanced high-resolution BGR image
    """
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]

    # Try neural FSRCNN_x3 super-resolution
    net = load_fsrcnn_model()
    if net is not None:
        try:
            # Prepare luminance Y channel for neural super resolution
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)

            blob = cv2.dnn.blobFromImage(y, 1.0 / 255.0, (w, h), (0, 0, 0), swapRB=False, crop=False)
            net.setInput(blob)
            out = net.forward()

            # Rescale output Y channel
            out_y = out[0, 0] * 255.0
            out_y = np.clip(out_y, 0, 255).astype(np.uint8)

            out_h, out_w = out_y.shape[:2]
            cr_up = cv2.resize(cr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            cb_up = cv2.resize(cb, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

            enhanced_ycrcb = cv2.merge([out_y, cr_up, cb_up])
            enhanced_bgr = cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2BGR)
            return enhanced_bgr
        except Exception:
            pass

    # High-Frequency Unsharp Masking + Lanczos4 3x Resolution Enhancement
    resized = cv2.resize(img, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_LANCZOS4)
    gaussian = cv2.GaussianBlur(resized, (0, 0), 3.0)
    enhanced = cv2.addWeighted(resized, 1.5, gaussian, -0.5, 0)
    return enhanced


def create_all_pixel_grid_crops(frame, grid_rows=3, grid_cols=3, zoom_factor=2.0):
    """
    Divide frame across ALL pixel coordinates into a grid of overlapping tiles
    to guarantee no distant crime scene is missed regardless of height.

    Args:
        frame: OpenCV BGR frame
        grid_rows: number of row tiles (default 3)
        grid_cols: number of col tiles (default 3)
        zoom_factor: zoom expansion factor

    Returns:
        List of tuples: (tile_name, cropped_region, (x1, y1, x2, y2))
    """
    h, w = frame.shape[:2]
    crops = []

    # Whole frame
    crops.append(("original_full_frame", frame, (0, 0, w, h)))

    tile_h = h // grid_rows
    tile_w = w // grid_cols

    for r in range(grid_rows):
        for c in range(grid_cols):
            x1 = max(0, c * tile_w - tile_w // 4)
            y1 = max(0, r * tile_h - tile_h // 4)
            x2 = min(w, (c + 1) * tile_w + tile_w // 4)
            y2 = min(h, (r + 1) * tile_h + tile_h // 4)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                # Enhance resolution for distant high-altitude pixel crop
                crop_sr = enhance_resolution_3x(crop, scale_factor=3)
                tile_name = f"grid_pixel_tile_r{r+1}_c{c+1}_[{x1},{y1},{x2},{y2}]"
                crops.append((tile_name, crop_sr, (x1, y1, x2, y2)))

    return crops

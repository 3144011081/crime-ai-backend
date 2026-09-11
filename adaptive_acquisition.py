"""
Adaptive Video Acquisition Module

When aerial footage is too distant or the classifier is uncertain,
this module provides multi-scale zoom crops around detected people
and re-classifies to improve accuracy.

Used by video_inference1.py — not a standalone script.
"""



import cv2
import numpy as np


import torch
from super_resolution import enhance_resolution_3x, create_all_pixel_grid_crops
from image_preprocessing import DenoiseAndSharpen, IlluminationEnhancer, SceneQualityMetrics


# ============================================================
# CONFIGURATION
# ============================================================

# Classification thresholds
CONFIDENT_THRESHOLD = 0.75
UNCERTAIN_THRESHOLD = 0.55

# Person-size threshold.
# If detected people are very small in the aerial frame,
# zooming will be triggered.
MIN_PERSON_HEIGHT_RATIO = 0.035

# Zoom factors for person-based crops
ZOOM_FACTORS = [1.5, 2.0, 2.5, 3.0]


# ============================================================
# CHECK WHETHER AERIAL SCENE IS TOO SMALL
# ============================================================

def scene_needs_zoom(frame, boxes):
    """
    Check if detected people are too small in the frame,
    indicating the aerial camera is too far away.

    Args:
        frame: OpenCV BGR frame
        boxes: list of (x1, y1, x2, y2) person bounding boxes

    Returns:
        True if zoom is needed
    """

    height, width = frame.shape[:2]

    if len(boxes) == 0:
        return False

    person_heights = []

    for item in boxes:
        if isinstance(item, dict):
            box = item.get("box", item.get("bbox", [0, 0, 0, 0]))
        else:
            box = item
        x1, y1, x2, y2 = [float(v) for v in box]
        person_height = y2 - y1
        ratio = person_height / height
        person_heights.append(ratio)

    max_ratio = max(person_heights)

    if max_ratio < MIN_PERSON_HEIGHT_RATIO:
        return True

    return False


# ============================================================
# CREATE ZOOMED ROI
# ============================================================

def crop_zoomed_region(frame, box, zoom_factor=2.0):
    """
    Crop a zoomed region around a bounding box.

    Args:
        frame: OpenCV BGR frame
        box: (x1, y1, x2, y2) bounding box
        zoom_factor: how much to expand the crop

    Returns:
        Cropped frame region
    """

    h, w = frame.shape[:2]

    if isinstance(box, dict):
        box_coords = box.get("box", box.get("bbox", [0, 0, 0, 0]))
    else:
        box_coords = box
    x1, y1, x2, y2 = [int(float(v)) for v in box_coords]

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    box_w = x2 - x1
    box_h = y2 - y1

    new_w = int(box_w * zoom_factor)
    new_h = int(box_h * zoom_factor)

    nx1 = max(0, cx - new_w // 2)
    ny1 = max(0, cy - new_h // 2)

    nx2 = min(w, cx + new_w // 2)
    ny2 = min(h, cy + new_h // 2)

    crop = frame[ny1:ny2, nx1:nx2]

    if crop.size == 0:
        return frame

    return crop


# ============================================================
# FULL FRAME MULTI-SCALE CROPS
# ============================================================

def create_multiscale_frames(frame, boxes):
    """
    Create multiple zoomed and super-resolved views of the frame around detected people.

    Args:
        frame: OpenCV BGR frame
        boxes: list of (x1, y1, x2, y2) person bounding boxes

    Returns:
        list of (name, cropped_frame, (x1, y1, x2, y2)) tuples
    """
    crops = []
    h, w = frame.shape[:2]

    # Original frame
    crops.append(("original", frame, (0, 0, w, h)))

    # Person-based zoom with FSRCNN 3x Super-Resolution & Preprocessing
    for idx, box in enumerate(boxes):
        for zoom in ZOOM_FACTORS:
            crop = crop_zoomed_region(frame, box, zoom_factor=zoom)
            if crop.size > 0:
                # Preprocess zoomed crop to boost contrast and edge salience
                crop = IlluminationEnhancer.apply_clahe(crop, clip_limit=2.0)
                crop = DenoiseAndSharpen.unsharp_mask(crop, sigma=0.8, strength=1.0)
            crop_sr = enhance_resolution_3x(crop, scale_factor=3)
            if isinstance(box, dict):
                b_coords = box.get("box", box.get("bbox", [0, 0, 0, 0]))
            else:
                b_coords = box
            x1, y1, x2, y2 = [int(float(v)) for v in b_coords]
            crops.append((f"fsrcnn_person_zoom_{zoom}_[{x1},{y1},{x2},{y2}]", crop_sr, (x1, y1, x2, y2)))

    return crops


# ============================================================
# ADAPTIVE ACQUISITION PIPELINE
# ============================================================

def run_adaptive_acquisition(
    predict_fn,
    detect_people_fn,
    frames,
    original_label,
    original_conf,
    original_probs
):
    """
    Main adaptive acquisition logic.

    If original confidence is high enough, returns original.
    Otherwise, tries multi-scale zoomed crops around detected persons and picks the best.
    """

    print("\n====================================")
    print("ADAPTIVE VIDEO ACQUISITION")
    print("====================================")

    print(f"\nOriginal prediction: {original_label}")
    print(f"Original confidence: {original_conf:.3f}")

    # --------------------------------------------------------
    # STEP 1: If already confident, don't zoom
    # --------------------------------------------------------

    if original_conf >= CONFIDENT_THRESHOLD:

        print("\nScene is sufficiently visible.")
        print("Using original aerial video.")

        return {
            "prediction": original_label,
            "confidence": original_conf,
            "mode": "original",
            "probabilities": original_probs
        }

    # --------------------------------------------------------
    # STEP 2: Detect people on representative frame
    # --------------------------------------------------------

    middle_frame = frames[len(frames) // 2]

    boxes = detect_people_fn(middle_frame)

    print(f"\nDetected people: {len(boxes)}")

    # If no people detected, zooming into empty scenery would cause false positives
    if len(boxes) == 0:
        print("\nNo persons detected in aerial footage. Retaining original perspective.")
        return {
            "prediction": original_label,
            "confidence": original_conf,
            "mode": "original",
            "probabilities": original_probs
        }

    # --------------------------------------------------------
    # STEP 3: Determine whether scene needs zoom
    # --------------------------------------------------------

    needs_zoom = scene_needs_zoom(middle_frame, boxes)

    if not needs_zoom and original_conf >= UNCERTAIN_THRESHOLD:

        print("\nScene is reasonably visible.")
        print("No additional zoom required.")

        return {
            "prediction": original_label,
            "confidence": original_conf,
            "mode": "original",
            "probabilities": original_probs
        }

    # --------------------------------------------------------
    # STEP 4: Adaptive zoom
    # --------------------------------------------------------

    print("\nOriginal scene is uncertain.")
    print("Activating adaptive zoom...")

    zoom_results = []
    multiscale_frames = []

    for frame in frames:
        frame_boxes = detect_people_fn(frame)
        crops = create_multiscale_frames(frame, frame_boxes)
        multiscale_frames.append(crops)

    # --------------------------------------------------------
    # Evaluate each zoom level
    # --------------------------------------------------------

    num_zoom_levels = len(multiscale_frames[0])

    for zoom_index in range(num_zoom_levels):

        zoom_name = multiscale_frames[0][zoom_index][0]
        zoom_coords = multiscale_frames[0][zoom_index][2] if len(multiscale_frames[0][zoom_index]) > 2 else None

        zoom_clip = []

        for frame_crops in multiscale_frames:
            if zoom_index < len(frame_crops):
                crop = frame_crops[zoom_index][1]
            else:
                crop = frame_crops[0][1]  # fallback to original
            zoom_clip.append(crop)

        prediction, confidence, probs = predict_fn(zoom_clip)

        coord_str = f" [Pixel ROI: {zoom_coords}]" if zoom_coords else ""
        print(
            f"{zoom_name}{coord_str}: "
            f"{prediction} "
            f"({confidence:.3f})"
        )

        zoom_results.append({
            "mode": zoom_name,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probs,
            "pixel_coordinates": zoom_coords
        })

    # --------------------------------------------------------
    # STEP 5: Select strongest result
    # --------------------------------------------------------

    best_result = max(
        zoom_results,
        key=lambda x: x["confidence"]
    )

    # Compare against original
    if best_result["confidence"] > original_conf:

        print(f"\nZoomed scene produced a stronger prediction.")
        print(f"Final prediction: {best_result['prediction']}")
        print(f"Final confidence: {best_result['confidence']:.3f}")

        return best_result

    else:

        print("\nOriginal prediction remains strongest.")

        return {
            "prediction": original_label,
            "confidence": original_conf,
            "mode": "original",
            "probabilities": original_probs
        }

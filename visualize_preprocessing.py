"""
Visual Demonstration & Quality Comparison Tool for Preprocessing Suite
Extracts sample frames from videos/ directory and outputs side-by-side
comparisons demonstrating:
  1. Low-Light & Shadow Enhancement
  2. Atmospheric Dehazing & Contrast Boost
  3. High-Frequency Sharpening & Detail Booster
  4. Auto White Balance & Color Correction
  5. 16-frame Temporal Smoothing & Motion Gradient Visualization
"""

import os
from pathlib import Path
import cv2
import numpy as np

from image_preprocessing import (
    SceneQualityMetrics,
    IlluminationEnhancer,
    DehazingEngine,
    DenoiseAndSharpen,
    ColorBalanceEngine,
    TemporalPreprocessor,
    AdaptiveDronePreprocessor
)

OUTPUT_DIR = Path("outputs") / "preprocessing_comparisons"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_label(img, text, position=(10, 30), bg_color=(0, 0, 0), text_color=(0, 255, 255)):
    canvas = img.copy()
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, bg_color, 3, cv2.LINE_AA)
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)
    return canvas


def main():
    print("=== Generating Visual Preprocessing Comparisons ===")

    # Look for video files
    video_files = list(Path("videos").glob("*.mp4"))
    if not video_files:
        print("No videos found in videos/")
        return

    sample_video = None
    for target in ["drone.mp4", "hello.mp4", "hello1.mp4", "6.mp4"]:
        candidate = Path("videos") / target
        if candidate.exists():
            sample_video = candidate
            break

    if sample_video is None:
        sample_video = video_files[0]

    print(f"Loading sample frames from: {sample_video}")
    cap = cv2.VideoCapture(str(sample_video))
    frames = []
    while len(frames) < 32:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print("Failed to read frames from video.")
        return

    ref_frame = frames[min(len(frames) // 2, 5)]
    h, w = ref_frame.shape[:2]
    # Resize to standard width for side-by-side grid if large
    display_w = 480
    display_h = int(h * (display_w / w))
    base = cv2.resize(ref_frame, (display_w, display_h))

    # 1. Individual Enhancements
    clahe_img = IlluminationEnhancer.apply_clahe(base, clip_limit=2.5)
    shadow_img = IlluminationEnhancer.shadow_highlight_recovery(base, shadow_lift=0.4)
    gamma_img = IlluminationEnhancer.adaptive_gamma_correction(base, alpha=0.6)
    dehazed_img = DehazingEngine.dark_channel_dehaze(base, patch_size=11)
    retinex_img = DehazingEngine.multiscale_retinex(base)
    sharpened_img = DenoiseAndSharpen.unsharp_mask(base, strength=1.5)
    wb_img = ColorBalanceEngine.gray_world_white_balance(base)

    # 2. Preset Preprocessors
    drone_prep = AdaptiveDronePreprocessor(mode="aerial_drone").preprocess_single_frame(base)
    low_light_prep = AdaptiveDronePreprocessor(mode="low_light_night").preprocess_single_frame(base)
    yolo_prep = AdaptiveDronePreprocessor(mode="yolo_enhanced").preprocess_single_frame(base, for_yolo=True)

    # Compute diagnostics for original vs enhanced
    diag_orig = SceneQualityMetrics.diagnose_frame(base)
    diag_drone = SceneQualityMetrics.diagnose_frame(drone_prep)
    diag_lowlight = SceneQualityMetrics.diagnose_frame(low_light_prep)

    print(f"Original Frame -> Brightness: {diag_orig['brightness']:.1f}, Contrast: {diag_orig['contrast']:.1f}, Sharpness: {diag_orig['sharpness']:.1f}")
    print(f"Aerial Enhanced -> Brightness: {diag_drone['brightness']:.1f}, Contrast: {diag_drone['contrast']:.1f}, Sharpness: {diag_drone['sharpness']:.1f}")
    print(f"Low-Light Mode -> Brightness: {diag_lowlight['brightness']:.1f}, Contrast: {diag_lowlight['contrast']:.1f}, Sharpness: {diag_lowlight['sharpness']:.1f}")

    # Build 2x2 Grid Comparison 1: Core Techniques
    p1 = draw_label(base, f"Original (Sharp: {diag_orig['sharpness']:.0f})")
    p2 = draw_label(clahe_img, "Perceptual LAB CLAHE")
    p3 = draw_label(dehazed_img, "Atmospheric Dehazing")
    p4 = draw_label(sharpened_img, f"Unsharp Masked (Sharp: {SceneQualityMetrics.compute_sharpness_score(sharpened_img):.0f})")

    row1 = np.hstack([p1, p2])
    row2 = np.hstack([p3, p4])
    grid1 = np.vstack([row1, row2])
    out_path1 = OUTPUT_DIR / "comparison_core_filters.jpg"
    cv2.imwrite(str(out_path1), grid1)
    print(f"Saved: {out_path1}")

    # Build 2x2 Grid Comparison 2: Task-Specific Profiles
    g1 = draw_label(base, "1. Raw Surveillance Input")
    g2 = draw_label(yolo_prep, "2. YOLOv8 Enhanced (Edge+Contrast)")
    g3 = draw_label(low_light_prep, "3. Low-Light Shadow Recovery")
    g4 = draw_label(drone_prep, "4. Aerial Drone Master Profile")

    row3 = np.hstack([g1, g2])
    row4 = np.hstack([g3, g4])
    grid2 = np.vstack([row3, row4])
    out_path2 = OUTPUT_DIR / "comparison_task_profiles.jpg"
    cv2.imwrite(str(out_path2), grid2)
    print(f"Saved: {out_path2}")

    # 3. Temporal Motion dynamics on 16 frames
    if len(frames) >= 16:
        sub_clip = [cv2.resize(f, (display_w, display_h)) for f in frames[:16]]
        motion_clip = TemporalPreprocessor.enhance_motion_dynamics(sub_clip, motion_weight=0.35)
        m_frame = draw_label(motion_clip[len(motion_clip)//2], "16-Frame Motion Gradient Overlay")
        out_path3 = OUTPUT_DIR / "comparison_motion_dynamics.jpg"
        cv2.imwrite(str(out_path3), m_frame)
        print(f"Saved: {out_path3}")

    print("Visual comparisons successfully generated in outputs/preprocessing_comparisons/")


if __name__ == "__main__":
    main()

"""
AI Aerial Surveillance & Video Crime Detection - Image Preprocessing Suite
==========================================================================
Provides comprehensive, modular, high-performance image & video preprocessing
algorithms specifically engineered to enhance downstream computer vision tasks:
  1. Object & Weapon Detection / Tracking (YOLOv8 + ByteTrack)
  2. Spatio-Temporal Action & Crime Classification (R(2+1)D-18)
  3. Adaptive Multi-Scale Zoom & Super-Resolution (FSRCNN / Lanczos4)
  4. Evidence Extraction & Forensic Quality Enhancement

Author: Antigravity AI Pair Programmer
Project: ML_Project (Unified 2-Stage AI System)
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# =====================================================================
# 1. SCENE QUALITY METRICS & DIAGNOSTICS
# =====================================================================

class SceneQualityMetrics:
    """
    Computes statistical and perceptual quality indicators on video frames
    to guide adaptive preprocessing decisions.
    """

    @staticmethod
    def compute_brightness(image: np.ndarray) -> float:
        """Computes mean perceived luminance (0 to 255)."""
        if image is None or image.size == 0:
            return 0.0
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(np.mean(gray))

    @staticmethod
    def compute_rms_contrast(image: np.ndarray) -> float:
        """Computes Root Mean Square (RMS) standard deviation contrast."""
        if image is None or image.size == 0:
            return 0.0
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(np.std(gray))

    @staticmethod
    def compute_sharpness_score(image: np.ndarray) -> float:
        """Computes the variance of the Laplacian as a focus/sharpness metric."""
        if image is None or image.size == 0:
            return 0.0
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.var(laplacian))

    @staticmethod
    def compute_colorfulness(image: np.ndarray) -> float:
        """Computes Hasler & Süsstrunk metric for perceived image colorfulness."""
        if image is None or image.size == 0 or len(image.shape) < 3:
            return 0.0
        b, g, r = cv2.split(image.astype(np.float32))
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        std_rg, mean_rg = np.std(rg), np.mean(rg)
        std_yb, mean_yb = np.std(yb), np.mean(yb)
        std_rgyb = np.sqrt(std_rg ** 2 + std_yb ** 2)
        mean_rgyb = np.sqrt(mean_rg ** 2 + mean_yb ** 2)
        return float(std_rgyb + 0.3 * mean_rgyb)

    @classmethod
    def diagnose_frame(cls, image: np.ndarray) -> Dict[str, Union[float, bool, str]]:
        """
        Diagnoses environmental condition: low-light, overexposed, hazy, low-contrast, blurry.
        """
        brightness = cls.compute_brightness(image)
        contrast = cls.compute_rms_contrast(image)
        sharpness = cls.compute_sharpness_score(image)
        colorfulness = cls.compute_colorfulness(image)

        is_low_light = brightness < 70.0
        is_overexposed = brightness > 190.0
        is_low_contrast = contrast < 35.0
        is_blurry = sharpness < 80.0
        is_hazy = (contrast < 42.0 and brightness > 120.0 and colorfulness < 25.0)

        # Primary condition assessment
        if is_low_light:
            condition = "low_light"
        elif is_hazy:
            condition = "hazy"
        elif is_overexposed:
            condition = "overexposed"
        elif is_low_contrast:
            condition = "low_contrast"
        elif is_blurry:
            condition = "blurry"
        else:
            condition = "normal"

        return {
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "colorfulness": colorfulness,
            "is_low_light": is_low_light,
            "is_overexposed": is_overexposed,
            "is_low_contrast": is_low_contrast,
            "is_blurry": is_blurry,
            "is_hazy": is_hazy,
            "condition": condition
        }


# =====================================================================
# 2. ILLUMINATION & CONTRAST ENHANCER
# =====================================================================

class IlluminationEnhancer:
    """
    Performs perceptual illumination correction, CLAHE, shadow-highlight balancing,
    and adaptive gamma correction for surveillance and drone scenes.
    """

    @staticmethod
    def apply_clahe(
        image: np.ndarray,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8),
        color_space: str = "LAB"
    ) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Enhances local contrast while preserving color balance by operating on
        luminance channel (L in LAB, or Y in YCrCb).
        """
        if image is None or image.size == 0:
            return image

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        if len(image.shape) == 2:
            return clahe.apply(image)

        if color_space.upper() == "LAB":
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = clahe.apply(l)
            merged = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        elif color_space.upper() == "YCRCB":
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y_enhanced = clahe.apply(y)
            merged = cv2.merge([y_enhanced, cr, cb])
            return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

        elif color_space.upper() == "HSV":
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v_enhanced = clahe.apply(v)
            merged = cv2.merge([h, s, v_enhanced])
            return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)

        else:
            # Fallback per-channel
            b, g, r = cv2.split(image)
            return cv2.merge([clahe.apply(b), clahe.apply(g), clahe.apply(r)])

    @staticmethod
    def adaptive_gamma_correction(image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Adaptive Gamma Correction with Weighting Distribution (AGCWD).
        Dynamically calculates optimal gamma from image CDF to recover shadow detail
        without blowing out bright daylight areas.
        """
        if image is None or image.size == 0:
            return image

        # Work on luminance channel
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            channel = v
        else:
            channel = image

        hist, _ = np.histogram(channel.flatten(), 256, [0, 256])
        prob = hist / (channel.size + 1e-7)
        prob_max = np.max(prob)
        prob_min = np.min(prob)

        # Truncated weighting distribution
        prob_w = prob_max * (((prob - prob_min) / (prob_max - prob_min + 1e-7)) ** alpha)
        cdf_w = np.cumsum(prob_w) / (np.sum(prob_w) + 1e-7)

        # Gamma transformation mapping
        gamma_table = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            gamma = 1.0 - cdf_w[i]
            val = 255.0 * ((i / 255.0) ** gamma)
            gamma_table[i] = np.clip(val, 0, 255).astype(np.uint8)

        v_corrected = cv2.LUT(channel, gamma_table)

        if len(image.shape) == 3:
            merged = cv2.merge([h, s, v_corrected])
            return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
        return v_corrected

    @staticmethod
    def shadow_highlight_recovery(
        image: np.ndarray,
        shadow_lift: float = 0.35,
        highlight_compress: float = 0.15
    ) -> np.ndarray:
        """
        Lifts deep shadows (where concealed weapons / suspects hide) while
        protecting highlights from clipping.
        """
        if image is None or image.size == 0:
            return image

        float_img = image.astype(np.float32) / 255.0

        # Create shadow mask (inverted luminance)
        if len(image.shape) == 3:
            lum = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            lum = np.expand_dims(lum, axis=2)
        else:
            lum = float_img

        shadow_mask = np.clip(1.0 - lum * 1.5, 0.0, 1.0)
        highlight_mask = np.clip((lum - 0.7) * 3.33, 0.0, 1.0)

        # Smooth masks
        shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=15)
        if len(shadow_mask.shape) == 2 and len(float_img.shape) == 3:
            shadow_mask = np.expand_dims(shadow_mask, axis=2)

        # Apply non-linear tone curve
        enhanced = float_img + shadow_lift * shadow_mask * (1.0 - float_img)
        enhanced = enhanced - highlight_compress * highlight_mask * enhanced
        enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)

        return enhanced


# =====================================================================
# 3. ATMOSPHERIC DEHAZING & RETINEX ENGINE
# =====================================================================

class DehazingEngine:
    """
    Removes atmospheric haze, smog, and fog for high-altitude aerial drone footage
    using Dark Channel Prior (DCP) and Multi-Scale Retinex.
    """

    @staticmethod
    def dark_channel_dehaze(
        image: np.ndarray,
        patch_size: int = 15,
        omega: float = 0.85,
        guided_filter_radius: int = 40,
        epsilon: float = 0.001
    ) -> np.ndarray:
        """
        He et al. Dark Channel Prior Dehazing with fast guided smoothing.
        """
        if image is None or image.size == 0 or len(image.shape) < 3:
            return image

        img_norm = image.astype(np.float32) / 255.0
        h, w = image.shape[:2]

        # 1. Dark Channel
        min_channel = np.min(img_norm, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark_channel = cv2.erode(min_channel, kernel)

        # 2. Estimate Atmospheric Light (A)
        num_pixels = h * w
        top_k = max(int(num_pixels * 0.001), 1)
        flat_dark = dark_channel.flatten()
        flat_img = img_norm.reshape(-1, 3)
        indices = np.argpartition(flat_dark, -top_k)[-top_k:]
        atmospheric_light = np.mean(flat_img[indices], axis=0)
        atmospheric_light = np.clip(atmospheric_light, 0.2, 1.0)

        # 3. Estimate Transmission Map
        norm_img_by_a = img_norm / atmospheric_light
        dark_norm = cv2.erode(np.min(norm_img_by_a, axis=2), kernel)
        transmission = 1.0 - omega * dark_norm

        # 4. Fast Refinement using Bilateral Smoothing
        transmission_refined = cv2.bilateralFilter(
            transmission.astype(np.float32),
            d=9,
            sigmaColor=0.1,
            sigmaSpace=float(guided_filter_radius)
        )
        transmission_refined = np.clip(transmission_refined, 0.15, 1.0)
        t_3d = np.expand_dims(transmission_refined, axis=2)

        # 5. Recover Radiance: J = (I - A)/t + A
        recovered = (img_norm - atmospheric_light) / t_3d + atmospheric_light
        recovered = np.clip(recovered * 255.0, 0, 255).astype(np.uint8)

        return recovered

    @staticmethod
    def multiscale_retinex(
        image: np.ndarray,
        sigma_list: List[int] = [15, 80, 250],
        gain: float = 1.2,
        offset: float = 10.0
    ) -> np.ndarray:
        """
        Multi-Scale Retinex (MSR) color constancy and dynamic range expansion.
        """
        if image is None or image.size == 0 or len(image.shape) < 3:
            return image

        img_float = image.astype(np.float32) + 1.0
        retinex = np.zeros_like(img_float)

        for sigma in sigma_list:
            blur = cv2.GaussianBlur(img_float, (0, 0), float(sigma))
            retinex += np.log10(img_float) - np.log10(blur + 1e-6)

        retinex /= len(sigma_list)

        # Color restoration & dynamic normalization
        res = gain * retinex + offset
        for c in range(3):
            mean = np.mean(res[:, :, c])
            std = np.std(res[:, :, c]) + 1e-6
            res[:, :, c] = (res[:, :, c] - (mean - 2.0 * std)) / (4.0 * std) * 255.0

        return np.clip(res, 0, 255).astype(np.uint8)


# =====================================================================
# 4. EDGE-PRESERVING DENOISING & SHARPENING
# =====================================================================

class DenoiseAndSharpen:
    """
    Fast edge-preserving smoothing to remove sensor noise and compression artifacts,
    coupled with high-frequency unsharp masking to enhance small weapon and person boundaries.
    """

    @staticmethod
    def edge_preserving_denoise(
        image: np.ndarray,
        d: int = 5,
        sigma_color: float = 35.0,
        sigma_space: float = 35.0
    ) -> np.ndarray:
        """Fast Bilateral Filter preserving sharp edges while smoothing sensor noise."""
        if image is None or image.size == 0:
            return image
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    @staticmethod
    def unsharp_mask(
        image: np.ndarray,
        sigma: float = 1.0,
        strength: float = 1.5,
        threshold: int = 3
    ) -> np.ndarray:
        """
        High-Frequency Unsharp Masking.
        Calculates difference between frame and Gaussian blur to sharpen contours.
        """
        if image is None or image.size == 0:
            return image

        gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
        diff = cv2.absdiff(image, gaussian)

        # Apply threshold to avoid sharpening background noise
        mask = diff > threshold
        sharpened = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)

        result = np.where(mask, sharpened, image)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def detail_boost(image: np.ndarray, amount: float = 0.8) -> np.ndarray:
        """Laplacian-guided high-frequency edge booster."""
        if image is None or image.size == 0:
            return image
        kernel = np.array([
            [0, -1, 0],
            [-1, 4 + amount, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        filtered = cv2.filter2D(image, -1, kernel)
        return cv2.addWeighted(image, 0.7, filtered, 0.3, 0)


# =====================================================================
# 5. AUTO WHITE BALANCE & CHROMATICITY NORMALIZATION
# =====================================================================

class ColorBalanceEngine:
    """
    Corrects color casts caused by drone camera sensors, green terrain reflection,
    or artificial night lighting (sodium lamps).
    """

    @staticmethod
    def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
        """
        Gray-World assumption: the average color in a scene is neutral gray.
        Scales B, G, R channels so their means equalize.
        """
        if image is None or image.size == 0 or len(image.shape) < 3:
            return image

        b, g, r = cv2.split(image.astype(np.float32))
        mean_b = np.mean(b) + 1e-6
        mean_g = np.mean(g) + 1e-6
        mean_r = np.mean(r) + 1e-6

        mean_gray = (mean_b + mean_g + mean_r) / 3.0

        b = np.clip(b * (mean_gray / mean_b), 0, 255)
        g = np.clip(g * (mean_gray / mean_g), 0, 255)
        r = np.clip(r * (mean_gray / mean_r), 0, 255)

        return cv2.merge([b, g, r]).astype(np.uint8)

    @staticmethod
    def auto_color_equalization(image: np.ndarray, clip_percent: float = 1.0) -> np.ndarray:
        """
        Robust histogram stretching across RGB channels with percentile clipping.
        """
        if image is None or image.size == 0 or len(image.shape) < 3:
            return image

        channels = cv2.split(image)
        out_channels = []

        for ch in channels:
            low_val = np.percentile(ch, clip_percent)
            high_val = np.percentile(ch, 100.0 - clip_percent)

            if high_val > low_val:
                stretched = (ch.astype(np.float32) - low_val) * (255.0 / (high_val - low_val))
                out_channels.append(np.clip(stretched, 0, 255).astype(np.uint8))
            else:
                out_channels.append(ch)

        return cv2.merge(out_channels)


# =====================================================================
# 6. TEMPORAL VIDEO PREPROCESSING & MOTION GRADIENT ENHANCER
# =====================================================================

class TemporalPreprocessor:
    """
    Processes video clip sequences (e.g. 16-frame sliding windows) to:
      1. Eliminate temporal exposure flicker / sudden drone auto-exposure swings
      2. Maintain inter-frame luminance continuity
      3. Compute and blend motion gradients to accentuate dynamic violence/shooting cues
    """

    @staticmethod
    def temporal_exposure_smoothing(
        frames: List[np.ndarray],
        alpha: float = 0.7
    ) -> List[np.ndarray]:
        """
        Applies Exponential Moving Average (EMA) luminance normalization across a video clip.
        Prevents drone camera auto-exposure jumps from triggering false crime alerts.
        """
        if not frames:
            return frames

        # Compute mean luminance for all frames
        luminances = [
            float(np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))) if len(f.shape) == 3 else float(np.mean(f))
            for f in frames
        ]
        target_mean = float(np.median(luminances))

        smoothed_frames = []
        for f, lum in zip(frames, luminances):
            if lum < 1e-3:
                smoothed_frames.append(f)
                continue
            ratio = (alpha * (target_mean / lum) + (1.0 - alpha))
            ratio = np.clip(ratio, 0.7, 1.4)

            # Apply smooth scaling in Y/Luminance channel
            if len(f.shape) == 3:
                hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
                adj = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            else:
                adj = np.clip(f.astype(np.float32) * ratio, 0, 255).astype(np.uint8)

            smoothed_frames.append(adj)

        return smoothed_frames

    @staticmethod
    def enhance_motion_dynamics(
        frames: List[np.ndarray],
        motion_weight: float = 0.25
    ) -> List[np.ndarray]:
        """
        Blends frame-differencing motion gradients into the frame stream.
        Highlights rapid combat movements, running suspects, and weapon swings for 3D ConvNets.
        """
        if len(frames) < 2:
            return frames

        enhanced = []
        for t in range(len(frames)):
            curr = frames[t]
            if t == 0:
                diff = cv2.absdiff(frames[1], curr)
            elif t == len(frames) - 1:
                diff = cv2.absdiff(curr, frames[t - 1])
            else:
                diff1 = cv2.absdiff(curr, frames[t - 1])
                diff2 = cv2.absdiff(frames[t + 1], curr)
                diff = cv2.addWeighted(diff1, 0.5, diff2, 0.5, 0)

            # Soft edge mask
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(diff.shape) == 3 else diff
            diff_boost = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET) if len(curr.shape) == 3 else diff_gray

            blended = cv2.addWeighted(curr, 1.0 - motion_weight * 0.5, diff_boost, motion_weight * 0.5, 0)
            enhanced.append(blended)

        return enhanced


# =====================================================================
# 7. SPATIAL LETTERBOXING & ASPECT-RATIO PRESERVING TRANSFORMS
# =====================================================================

class SpatialTransformer:
    """
    Standardizes spatial dimensions without aspect-ratio distortion or bounding box squashing.
    """

    @staticmethod
    def letterbox_resize(
        image: np.ndarray,
        target_size: Tuple[int, int] = (112, 112),
        fill_color: Tuple[int, int, int] = (114, 114, 114),
        auto_pad: bool = False
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resizes image while keeping aspect ratio using letterbox padding.

        Returns:
            padded_img: Resized & padded image
            scale: Scale factor applied
            (pad_w, pad_h): Padding applied to left/right and top/bottom
        """
        if image is None or image.size == 0:
            return image, 1.0, (0, 0)

        target_w, target_h = target_size
        h, w = image.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        pad_w = (target_w - new_w) / 2
        pad_h = (target_h - new_h) / 2

        if auto_pad:
            pad_w = np.mod(pad_w, 32)
            pad_h = np.mod(pad_h, 32)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        top = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))
        left = int(round(pad_w - 0.1))
        right = int(round(pad_w + 0.1))

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=fill_color
        )

        return padded, scale, (left, top)

    @staticmethod
    def restore_bbox(
        box: Union[List[float], Tuple[float, ...]],
        scale: float,
        pad: Tuple[int, int],
        orig_shape: Tuple[int, int]
    ) -> List[float]:
        """
        Maps bounding box coordinates from letterboxed frame back to original image dimensions.
        """
        pad_left, pad_top = pad
        orig_h, orig_w = orig_shape

        x1, y1, x2, y2 = box[:4]

        x1 = max(0.0, min(float(orig_w), (x1 - pad_left) / scale))
        y1 = max(0.0, min(float(orig_h), (y1 - pad_top) / scale))
        x2 = max(0.0, min(float(orig_w), (x2 - pad_left) / scale))
        y2 = max(0.0, min(float(orig_h), (y2 - pad_top) / scale))

        return [x1, y1, x2, y2]


# =====================================================================
# 8. PYTORCH MODEL INPUT STANDARDIZER
# =====================================================================

class ModelInputStandardizer:
    """
    Converts preprocessed numpy arrays to standardized PyTorch tensors
    with configurable normalization parameters (ImageNet / Kinetics / Custom).
    """

    # Kinetics-400 video model normalization statistics
    KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
    KINETICS_STD = [0.22803, 0.22145, 0.216989]

    # Standard ImageNet statistics
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    @classmethod
    def frame_to_tensor(
        cls,
        frame: np.ndarray,
        target_size: int = 112,
        norm_type: str = "kinetics"
    ) -> torch.Tensor:
        """
        Converts single OpenCV BGR frame -> PyTorch Tensor [C, H, W] normalized.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != target_size or rgb.shape[1] != target_size:
            rgb = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

        if norm_type == "kinetics":
            mean = torch.tensor(cls.KINETICS_MEAN, dtype=torch.float32).view(3, 1, 1)
            std = torch.tensor(cls.KINETICS_STD, dtype=torch.float32).view(3, 1, 1)
            tensor = (tensor - mean) / std
        elif norm_type == "imagenet":
            mean = torch.tensor(cls.IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
            std = torch.tensor(cls.IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
            tensor = (tensor - mean) / std

        return tensor

    @classmethod
    def frames_to_clip_tensor(
        cls,
        frames: List[np.ndarray],
        target_size: int = 112,
        norm_type: str = "kinetics"
    ) -> torch.Tensor:
        """
        Converts list of T frames -> 5D Batch Tensor [1, C, T, H, W] for R(2+1)D models.
        """
        tensors = [cls.frame_to_tensor(f, target_size=target_size, norm_type=norm_type) for f in frames]
        # Stack into [T, C, H, W]
        clip = torch.stack(tensors)
        # Permute to [C, T, H, W]
        clip = clip.permute(1, 0, 2, 3)
        # Add batch dimension -> [1, C, T, H, W]
        clip = clip.unsqueeze(0)
        return clip


# =====================================================================
# 9. UNIFIED ADAPTIVE PREPROCESSOR PIPELINE
# =====================================================================

class AdaptiveDronePreprocessor:
    """
    Unified, end-to-end adaptive preprocessing engine for the Surveillance & Crime Detection Suite.
    Automatically inspects environmental scene properties and applies the optimal combination
    of illumination recovery, dehazing, edge sharpening, and temporal filtering.
    """

    def __init__(self, mode: str = "auto"):
        """
        Modes:
          - "auto": Automatically detects scene condition and tunes hyperparameters.
          - "aerial_drone": Standard drone profile (dehaze + CLAHE + unsharp mask).
          - "low_light_night": Low-light / night surveillance (shadow recovery + AGCWD + bilateral denoise).
          - "hazy_foggy": Dark channel prior dehazing + contrast expansion.
          - "yolo_enhanced": Optimized for small object/weapon detection recall.
          - "temporal_enhanced": Optimized for 16-frame R(2+1)D spatio-temporal classification.
          - "none" / "passthrough": Minimal resizing/color conversion only.
        """
        self.mode = mode
        self.illumination = IlluminationEnhancer()
        self.dehazer = DehazingEngine()
        self.denoiser = DenoiseAndSharpen()
        self.color_balancer = ColorBalanceEngine()
        self.temporal = TemporalPreprocessor()
        self.spatial = SpatialTransformer()
        self.standardizer = ModelInputStandardizer()
        self.diagnostics = SceneQualityMetrics()

    def preprocess_single_frame(
        self,
        frame: np.ndarray,
        for_yolo: bool = False,
        mode: str = None
    ) -> np.ndarray:
        """
        Preprocesses a single OpenCV BGR frame.

        Args:
            frame: Raw BGR input frame
            for_yolo: If True, tunes contrast and edge features specifically for YOLOv8
            mode: Optional processing mode override (e.g. "aerial_drone", "low_light_night")
        """
        if frame is None or frame.size == 0:
            return frame

        effective_mode = mode or self.mode
        if effective_mode in ("none", "passthrough"):
            return frame

        # Run scene diagnostics
        diag = self.diagnostics.diagnose_frame(frame)
        active_mode = effective_mode

        if active_mode == "auto":
            active_mode = diag["condition"]

        enhanced = frame.copy()

        # Step 1: Dehazing if hazy
        if active_mode in ("hazy", "hazy_foggy") or (self.mode == "aerial_drone" and diag["is_hazy"]):
            enhanced = self.dehazer.dark_channel_dehaze(enhanced, patch_size=11, omega=0.8)

        # Step 2: Illumination & Contrast correction
        if active_mode in ("low_light", "low_light_night"):
            # Recover deep shadows and apply adaptive gamma
            enhanced = self.illumination.shadow_highlight_recovery(enhanced, shadow_lift=0.35)
            enhanced = self.illumination.adaptive_gamma_correction(enhanced, alpha=0.5)
            enhanced = self.illumination.apply_clahe(enhanced, clip_limit=2.0, color_space="LAB")
            enhanced = self.denoiser.edge_preserving_denoise(enhanced, d=5, sigma_color=25.0, sigma_space=25.0)

        elif active_mode in ("aerial_drone", "yolo_enhanced"):
            # Enhance local contrast and micro-edges smoothly without pixelation
            enhanced = self.illumination.apply_clahe(enhanced, clip_limit=1.6, color_space="LAB")
            enhanced = self.color_balancer.gray_world_white_balance(enhanced)
            enhanced = self.denoiser.unsharp_mask(enhanced, sigma=0.8, strength=0.7, threshold=3)

        elif active_mode in ("low_contrast",):
            enhanced = self.illumination.apply_clahe(enhanced, clip_limit=1.8, color_space="LAB")
            enhanced = self.color_balancer.gray_world_white_balance(enhanced)
            enhanced = self.denoiser.unsharp_mask(enhanced, sigma=0.8, strength=0.7, threshold=3)

        elif active_mode in ("blurry",):
            enhanced = self.denoiser.unsharp_mask(enhanced, sigma=1.0, strength=0.9, threshold=2)
            enhanced = self.denoiser.detail_boost(enhanced, amount=0.3)

        else:
            # Normal / Balanced default
            enhanced = self.illumination.apply_clahe(enhanced, clip_limit=1.5, color_space="LAB")
            enhanced = self.denoiser.unsharp_mask(enhanced, sigma=0.8, strength=0.6, threshold=3)

        # YOLOv8 specific boost
        if for_yolo:
            # Subtle edge sharpness to make small guns/knives and distant people stand out
            enhanced = self.denoiser.unsharp_mask(enhanced, sigma=0.8, strength=0.8, threshold=2)

        return enhanced

    def preprocess_video_clip(
        self,
        frames: List[np.ndarray],
        enable_motion_enhancement: bool = False
    ) -> List[np.ndarray]:
        """
        Preprocesses a 16-frame sequence for the R(2+1)D crime classification model.
        Applies spatial enhancement followed by temporal exposure smoothing.
        """
        if not frames:
            return frames

        # 1. Spatial enhancement per frame
        spatially_enhanced = [self.preprocess_single_frame(f) for f in frames]

        # 2. Temporal exposure smoothing across the 16-frame window
        temporally_smoothed = self.temporal.temporal_exposure_smoothing(spatially_enhanced, alpha=0.75)

        # 3. Optional motion dynamic gradient blend
        if enable_motion_enhancement or self.mode == "temporal_enhanced":
            temporally_smoothed = self.temporal.enhance_motion_dynamics(temporally_smoothed, motion_weight=0.2)

        return temporally_smoothed

    def prepare_r2plus1d_tensor(
        self,
        frames: List[np.ndarray],
        target_size: int = 112,
        norm_type: str = "kinetics"
    ) -> torch.Tensor:
        """
        End-to-end pipeline from raw frames -> 5D PyTorch Tensor [1, C, 16, 112, 112].
        """
        preprocessed_frames = self.preprocess_video_clip(frames)
        return self.standardizer.frames_to_clip_tensor(
            preprocessed_frames,
            target_size=target_size,
            norm_type=norm_type
        )

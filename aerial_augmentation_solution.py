"""
AI Aerial Surveillance & Video Crime Detection - Fine-Tuning Fix & Aerial Augmentation Suite
==========================================================================================
Provides:
  1. Class-Balanced Focal Loss with Label Smoothing to eliminate False Positives for Shooting
     and increase Recall for minority classes like Vandalism.
  2. Weighted Random Sampler calculation for balanced mini-batch training.
  3. Aerial & Drone Spatio-Temporal Video Augmentor (3D/Clip-level consistent transformations):
     - Arbitrary 360-degree Top-Down / Oblique Rotation
     - Altitude Scaling (Random Resized Crop & Multi-scale Zoom)
     - Drone Gimbal Perspective & Homography Shear
     - Weather & Environmental Distortions (Haze, Thermal/Low-light Noise, Glare, Motion Blur)
     - Structural Occlusion Masking (Trees, Roofs, Poles)
"""

import math
import random
from typing import List, Tuple, Union, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


# =====================================================================
# 1. OPTIMIZED CLASS WEIGHTS & FOCAL LOSS
# =====================================================================

def compute_effective_class_weights(
    class_counts: Dict[str, int],
    beta: float = 0.999,
    scaling: str = "effective"
) -> Tuple[Dict[str, float], torch.Tensor]:
    """
    Computes smooth, robust class weights to prevent overconfident minority predictions.
    
    Methods:
    - 'effective': Class-Balanced Loss based on Effective Number of Samples (Cui et al., CVPR 2019).
    - 'sqrt': Square root smoothed inverse frequency weights.
    """
    classes = list(class_counts.keys())
    counts = np.array([class_counts[c] for c in classes], dtype=np.float32)
    
    if scaling == "effective":
        # E_n = (1 - beta^N) / (1 - beta)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    elif scaling == "sqrt":
        total_samples = np.sum(counts)
        weights = np.sqrt(total_samples / counts)
    else:
        # Standard inverse frequency
        total_samples = np.sum(counts)
        weights = total_samples / (len(classes) * counts)
        
    # Normalize weights so mean weight = 1.0
    weights = weights / np.mean(weights)
    
    weight_dict = {c: float(round(w, 3)) for c, w in zip(classes, weights)}
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    
    return weight_dict, weight_tensor


class ClassBalancedFocalLoss(nn.Module):
    """
    Focal Loss with Class Weights and optional Label Smoothing.
    
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Reduces the impact of easy negative samples and prevents overconfident predictions
    on heavily weighted classes like Shooting.
    """
    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C]
        targets: [B] (class indices)
        """
        num_classes = logits.size(1)
        
        # Apply Label Smoothing to target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weights = torch.pow(1.0 - probs, self.gamma)
        
        # Class alpha weight tensor [B, C]
        alpha_weights = self.alpha.to(logits.device).unsqueeze(0).expand_as(logits)
        
        # Cross entropy loss per class
        loss = -alpha_weights * focal_weights * smooth_targets * log_probs
        loss = torch.sum(loss, dim=-1) # Sum across classes [B]
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss


def get_weighted_sampler(targets: List[int], num_classes: int) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler to oversample rare classes (Vandalism, Shooting)
    at data loader level, ensuring balanced mini-batches.
    """
    class_counts = np.bincount(targets, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = np.array([class_weights[t] for t in targets], dtype=np.float64)
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


# =====================================================================
# 2. AERIAL & SURVEILLANCE SPATIO-TEMPORAL VIDEO AUGMENTOR
# =====================================================================

class AerialVideoAugmentor:
    """
    Applies clip-consistent spatial, geometric, aerial perspective, and atmospheric 
    augmentations across video frames [T, H, W, C] or [T, C, H, W].
    """
    def __init__(
        self,
        degrees: float = 180.0,
        scale_range: Tuple[float, float] = (0.5, 1.2),
        shear_deg: float = 15.0,
        p_flip: float = 0.5,
        p_weather: float = 0.4,
        p_occlusion: float = 0.3,
        p_motion_blur: float = 0.3
    ):
        self.degrees = degrees
        self.scale_range = scale_range
        self.shear_deg = shear_deg
        self.p_flip = p_flip
        self.p_weather = p_weather
        self.p_occlusion = p_occlusion
        self.p_motion_blur = p_motion_blur

    def _get_affine_matrix(self, height: int, width: int) -> np.ndarray:
        """Generates random 2D perspective / affine matrix for aerial rotation & scale."""
        angle = random.uniform(-self.degrees, self.degrees)
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        shear_x = random.uniform(-self.shear_deg, self.shear_deg)
        shear_y = random.uniform(-self.shear_deg, self.shear_deg)
        
        center = (width / 2.0, height / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Apply shear transformation
        tan_sx = math.tan(math.radians(shear_x))
        tan_sy = math.tan(math.radians(shear_y))
        shear_matrix = np.array([[1, tan_sx, 0], [tan_sy, 1, 0]], dtype=np.float32)
        
        # Combine M and shear
        M_combined = np.vstack([M, [0, 0, 1]])
        S_combined = np.vstack([shear_matrix, [0, 0, 1]])
        final_M = np.matmul(S_combined, M_combined)[:2, :]
        return final_M

    def _apply_haze(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Simulates atmospheric haze/fog on aerial surveillance camera."""
        h, w, _ = frame.shape
        overlay = np.full((h, w, 3), 220, dtype=np.uint8) # Whitish haze
        return cv2.addWeighted(frame, 1.0 - intensity, overlay, intensity, 0)

    def _apply_low_light_noise(self, frame: np.ndarray) -> np.ndarray:
        """Simulates low-light sensor noise (thermal/infrared/night vision noise)."""
        noise = np.random.normal(0, 15, frame.shape).astype(np.float32)
        noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    def _apply_motion_blur(self, frame: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        """Simulates drone movement or camera jitter blur."""
        kernel = np.zeros((kernel_size, kernel_size))
        # Random angle for blur kernel
        if random.random() > 0.5:
            kernel[int((kernel_size - 1) / 2), :] = 1.0
        else:
            kernel[:, int((kernel_size - 1) / 2)] = 1.0
        kernel /= kernel_size
        return cv2.filter2D(frame, -1, kernel)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Augments a sequence of frames [T, H, W, C] (uint8 0-255).
        Returns augmented frames of same shape [T, H, W, C].
        """
        if len(frames) == 0:
            return frames
            
        t, h, w, c = frames.shape
        
        # 1. Sample clip-consistent augmentation parameters
        do_hflip = random.random() < self.p_flip
        do_vflip = random.random() < self.p_flip
        do_affine = random.random() < 0.7
        affine_M = self._get_affine_matrix(h, w) if do_affine else None
        
        do_weather = random.random() < self.p_weather
        haze_intensity = random.uniform(0.1, 0.35) if do_weather else 0.0
        do_noise = do_weather and (random.random() < 0.5)
        
        do_blur = random.random() < self.p_motion_blur
        blur_k = random.choice([5, 7, 9]) if do_blur else 0
        
        do_occlusion = random.random() < self.p_occlusion
        if do_occlusion:
            occ_w = random.randint(int(w * 0.1), int(w * 0.3))
            occ_h = random.randint(int(h * 0.1), int(h * 0.3))
            occ_x = random.randint(0, w - occ_w)
            occ_y = random.randint(0, h - occ_h)
            occ_color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            
        augmented_frames = []
        for i in range(t):
            frame = frames[i].copy()
            
            # Flips
            if do_hflip:
                frame = cv2.flip(frame, 1)
            if do_vflip:
                frame = cv2.flip(frame, 0)
                
            # Affine / Top-Down Rotation / Scale
            if do_affine:
                frame = cv2.warpAffine(frame, affine_M, (w, h), borderMode=cv2.BORDER_REFLECT)
                
            # Weather & Sensor Noise
            if do_weather and haze_intensity > 0:
                frame = self._apply_haze(frame, haze_intensity)
            if do_noise:
                frame = self._apply_low_light_noise(frame)
                
            # Motion Blur
            if do_blur:
                frame = self._apply_motion_blur(frame, blur_k)
                
            # Structural Occlusion
            if do_occlusion:
                frame[occ_y:occ_y + occ_h, occ_x:occ_x + occ_w] = occ_color
                
            augmented_frames.append(frame)
            
        return np.stack(augmented_frames, axis=0)


# =====================================================================
# 3. DEMONSTRATION & VERIFICATION
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DEMO: CLASS WEIGHT & FOCAL LOSS RE-BALANCING FOR ML_PROJECT")
    print("=" * 70)
    
    # Original user dataset counts
    dataset_counts = {
        "Normal": 240,
        "Violence": 120,
        "Robbery": 320,
        "Shooting": 40,
        "FireExplosion": 80,
        "Accident": 120,
        "Vandalism": 40
    }
    
    # 1. Compute effective weights vs standard weights
    old_weights = {
        "Normal": 0.571, "Violence": 1.143, "Robbery": 0.429,
        "Shooting": 3.429, "FireExplosion": 1.714, "Accident": 1.143, "Vandalism": 3.429
    }
    
    eff_dict, eff_tensor = compute_effective_class_weights(dataset_counts, beta=0.999, scaling="effective")
    sqrt_dict, sqrt_tensor = compute_effective_class_weights(dataset_counts, scaling="sqrt")
    
    print("\nComparison of Class Weights:")
    print(f"{'Class':<15} | {'Old Weight':<12} | {'Effective Weight':<18} | {'Sqrt Weight':<12}")
    print("-" * 65)
    for cls_name in dataset_counts:
        print(f"{cls_name:<15} | {old_weights[cls_name]:<12.3f} | {eff_dict[cls_name]:<18.3f} | {sqrt_dict[cls_name]:<12.3f}")
        
    print("\n[✓] Class Balanced Loss & Aerial Augmentation Module loaded successfully.")

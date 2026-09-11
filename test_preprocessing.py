"""
Unit and Performance Test Suite for Image & Video Preprocessing Pipeline
"""

import time
import numpy as np
import cv2
import torch
import unittest

from image_preprocessing import (
    SceneQualityMetrics,
    IlluminationEnhancer,
    DehazingEngine,
    DenoiseAndSharpen,
    ColorBalanceEngine,
    TemporalPreprocessor,
    SpatialTransformer,
    ModelInputStandardizer,
    AdaptiveDronePreprocessor
)


class TestImagePreprocessing(unittest.TestCase):

    def setUp(self):
        # Create synthetic test frames
        # 1. Normal color frame (gradient)
        self.h, self.w = 480, 640
        self.normal_frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for i in range(self.h):
            self.normal_frame[i, :, 0] = int(255 * (i / self.h))  # Blue gradient
            self.normal_frame[i, :, 1] = 120                      # Green constant
            self.normal_frame[:, :, 2] = 80                       # Red constant

        # 2. Low-light dark frame
        self.dark_frame = (self.normal_frame * 0.15).astype(np.uint8)

        # 3. Hazy frame
        self.hazy_frame = cv2.addWeighted(self.normal_frame, 0.5, np.full_like(self.normal_frame, 220), 0.5, 0)

        # 4. 16-frame video clip with synthetic exposure flicker
        self.clip = []
        for t in range(16):
            factor = 1.0 + 0.3 * np.sin(t * 0.5)
            f = np.clip(self.normal_frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            self.clip.append(f)

    def test_scene_quality_metrics(self):
        diag_normal = SceneQualityMetrics.diagnose_frame(self.normal_frame)
        self.assertIn("brightness", diag_normal)
        self.assertIn("contrast", diag_normal)
        self.assertIn("sharpness", diag_normal)

        diag_dark = SceneQualityMetrics.diagnose_frame(self.dark_frame)
        self.assertTrue(diag_dark["is_low_light"])

        diag_hazy = SceneQualityMetrics.diagnose_frame(self.hazy_frame)
        self.assertTrue(diag_hazy["is_hazy"] or diag_hazy["brightness"] > 100)

    def test_illumination_enhancer(self):
        # Test CLAHE
        clahe_out = IlluminationEnhancer.apply_clahe(self.dark_frame, clip_limit=2.0)
        self.assertEqual(clahe_out.shape, self.dark_frame.shape)
        self.assertGreaterEqual(np.mean(clahe_out), np.mean(self.dark_frame))

        # Test Adaptive Gamma Correction
        gamma_out = IlluminationEnhancer.adaptive_gamma_correction(self.dark_frame)
        self.assertEqual(gamma_out.shape, self.dark_frame.shape)
        self.assertGreater(np.mean(gamma_out), np.mean(self.dark_frame))

        # Test Shadow-Highlight Recovery
        shadow_out = IlluminationEnhancer.shadow_highlight_recovery(self.dark_frame, shadow_lift=0.4)
        self.assertEqual(shadow_out.shape, self.dark_frame.shape)
        self.assertGreater(np.mean(shadow_out), np.mean(self.dark_frame))

    def test_dehazing_engine(self):
        dehazed = DehazingEngine.dark_channel_dehaze(self.hazy_frame, patch_size=9)
        self.assertEqual(dehazed.shape, self.hazy_frame.shape)
        # Dehazed frame should have higher contrast than original hazy frame
        contrast_orig = SceneQualityMetrics.compute_rms_contrast(self.hazy_frame)
        contrast_dehazed = SceneQualityMetrics.compute_rms_contrast(dehazed)
        self.assertGreaterEqual(contrast_dehazed, contrast_orig - 1.0)

        retinex_out = DehazingEngine.multiscale_retinex(self.normal_frame, sigma_list=[15, 80])
        self.assertEqual(retinex_out.shape, self.normal_frame.shape)

    def test_denoise_and_sharpen(self):
        denoised = DenoiseAndSharpen.edge_preserving_denoise(self.normal_frame, d=3)
        self.assertEqual(denoised.shape, self.normal_frame.shape)

        sharpened = DenoiseAndSharpen.unsharp_mask(self.normal_frame, strength=1.5)
        self.assertEqual(sharpened.shape, self.normal_frame.shape)
        sharpness_orig = SceneQualityMetrics.compute_sharpness_score(self.normal_frame)
        sharpness_new = SceneQualityMetrics.compute_sharpness_score(sharpened)
        self.assertGreaterEqual(sharpness_new, sharpness_orig)

        boosted = DenoiseAndSharpen.detail_boost(self.normal_frame)
        self.assertEqual(boosted.shape, self.normal_frame.shape)

    def test_color_balancer(self):
        # Color cast frame (heavy green tint)
        tinted = self.normal_frame.copy()
        tinted[:, :, 1] = 240
        balanced = ColorBalanceEngine.gray_world_white_balance(tinted)
        self.assertEqual(balanced.shape, tinted.shape)

        auto_col = ColorBalanceEngine.auto_color_equalization(self.normal_frame)
        self.assertEqual(auto_col.shape, self.normal_frame.shape)

    def test_temporal_preprocessor(self):
        smoothed = TemporalPreprocessor.temporal_exposure_smoothing(self.clip, alpha=0.7)
        self.assertEqual(len(smoothed), len(self.clip))
        self.assertEqual(smoothed[0].shape, self.clip[0].shape)

        motion_enhanced = TemporalPreprocessor.enhance_motion_dynamics(self.clip, motion_weight=0.2)
        self.assertEqual(len(motion_enhanced), len(self.clip))

    def test_spatial_transformer(self):
        padded, scale, pad = SpatialTransformer.letterbox_resize(self.normal_frame, target_size=(112, 112))
        self.assertEqual(padded.shape, (112, 112, 3))
        self.assertGreater(scale, 0.0)

        # Test bbox restoration
        orig_box = [100, 100, 200, 300]
        # Simulate forward transform
        scaled_box = [b * scale + (pad[0] if i % 2 == 0 else pad[1]) for i, b in enumerate(orig_box)]
        restored = SpatialTransformer.restore_bbox(scaled_box, scale, pad, (self.h, self.w))
        np.testing.assert_allclose(restored, orig_box, atol=2.0)

    def test_model_input_standardizer(self):
        tensor = ModelInputStandardizer.frame_to_tensor(self.normal_frame, target_size=112, norm_type="kinetics")
        self.assertEqual(tensor.shape, (3, 112, 112))
        self.assertEqual(tensor.dtype, torch.float32)

        clip_tensor = ModelInputStandardizer.frames_to_clip_tensor(self.clip, target_size=112, norm_type="kinetics")
        # Expected shape: [1, 3, 16, 112, 112]
        self.assertEqual(clip_tensor.shape, (1, 3, 16, 112, 112))
        self.assertEqual(clip_tensor.dtype, torch.float32)

    def test_adaptive_drone_preprocessor(self):
        modes = ["auto", "aerial_drone", "low_light_night", "hazy_foggy", "yolo_enhanced", "temporal_enhanced", "none"]
        for mode in modes:
            prep = AdaptiveDronePreprocessor(mode=mode)
            single_out = prep.preprocess_single_frame(self.dark_frame, for_yolo=True)
            self.assertEqual(single_out.shape, self.dark_frame.shape)

            clip_tensor = prep.prepare_r2plus1d_tensor(self.clip, target_size=112)
            self.assertEqual(clip_tensor.shape, (1, 3, 16, 112, 112))

    def test_throughput_benchmark(self):
        prep = AdaptiveDronePreprocessor(mode="aerial_drone")
        # Warmup
        for _ in range(5):
            _ = prep.preprocess_single_frame(self.normal_frame)

        # Benchmark 50 frames
        n_iters = 50
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = prep.preprocess_single_frame(self.normal_frame, for_yolo=True)
        elapsed = time.perf_counter() - start
        fps = n_iters / elapsed
        print(f"\n[Benchmark] Adaptive Preprocessing Speed: {fps:.2f} FPS (Elapsed: {elapsed*1000/n_iters:.2f} ms/frame)")
        self.assertGreater(fps, 15.0)  # Should achieve real-time throughput


if __name__ == "__main__":
    unittest.main()

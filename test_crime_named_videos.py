import os
import sys
from pathlib import Path
import time
import json
from collections import Counter

# Avoid matplotlib font cache rebuild delay
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.makedirs("/tmp/matplotlib_cache", exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
import numpy as np

from video_inference1 import (
    model,
    classes,
    DEVICE,
    predict_clip,
    ObjectActivityDetector,
    fuse_multimodal_crime_decision,
    CLIP_LEN,
    WINDOW_STRIDE,
    CRIME_THRESHOLD
)

VIDEOS_DIR = PROJECT_ROOT / "videos"

TEST_VIDEOS = [
    {"file": "accident.mp4", "expected": "Accident"},
    {"file": "fireexplosion.mp4", "expected": "FireExplosion"},
    {"file": "fireexplosion1.mp4", "expected": "FireExplosion"},
    {"file": "robbery6.mp4", "expected": "Robbery"},
    {"file": "shooting.mp4", "expected": "Shooting"},
    {"file": "vandalism.mp4", "expected": "Vandalism"},
]

print("=" * 80, flush=True)
print("TESTING PROJECT ON CRIME-NAMED VIDEOS", flush=True)
print(f"Device: {DEVICE} | Model Classes: {classes}", flush=True)
print("=" * 80, flush=True)

detector = ObjectActivityDetector()

results_summary = []

for item in TEST_VIDEOS:
    vname = item["file"]
    expected = item["expected"]
    vpath = VIDEOS_DIR / vname

    if not vpath.exists():
        print(f"\n[SKIP] {vname}: File not found at {vpath}", flush=True)
        continue

    print(f"\n{'='*70}", flush=True)
    print(f"Processing: {vname} (Expected Crime: {expected})", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    duration = total_frames / fps
    print(f"Specs: {frames[0].shape[1]}x{frames[0].shape[0]} | {fps:.1f} fps | {total_frames} frames ({duration:.2f}s)", flush=True)

    # 1. Detector tracking across sampled frames (1 frame per second)
    sample_step = max(1, int(fps))  # 1 frame per second
    all_people = []
    all_vehicles = []
    all_weapons = []
    all_activities = set()
    detector_cache = {}
    detector.reset_tracker()
    sample_indices = list(range(0, total_frames, sample_step))
    print(f"Sampling {len(sample_indices)} frames for physical detectors (1 frame/sec)...", flush=True)
    for step_i, f_idx in enumerate(sample_indices):
        p, v, w, acts = detector.detect_and_track(frames[f_idx])
        detector_cache[f_idx] = (p, v, w, acts)
        all_people.extend(p)
        all_vehicles.extend(v)
        all_weapons.extend(w)
        all_activities.update(acts)
        if (step_i + 1) % 10 == 0 or (step_i + 1) == len(sample_indices):
            print(f"  [Detector] Processed {step_i+1}/{len(sample_indices)} frames...", flush=True)

    valid_weapons = [wp for wp in all_weapons if wp.get("conf", 0) >= 0.30]
    weapon_names = sorted(list(set(wp.get("label", "Weapon") for wp in valid_weapons)))

    # 2. 3D CNN Predictions across temporal windows
    total_windows = (total_frames - CLIP_LEN) // WINDOW_STRIDE + 1
    print(f"Evaluating {total_windows} temporal 3D CNN windows on {DEVICE}...", flush=True)
    raw_window_labels = []
    fused_window_labels = []
    fused_reasons = []
    all_window_probs = []

    for w_idx, start in enumerate(range(0, total_frames - CLIP_LEN + 1, WINDOW_STRIDE)):
        end = start + CLIP_LEN
        clip = frames[start:end]

        raw_label, raw_conf, probs = predict_clip(clip)
        raw_window_labels.append(raw_label)
        all_window_probs.append(probs[0].cpu())

        # Nearest sampled detector frame for this window
        win_sample_idx = (start + end) // 2
        nearest_key = min(detector_cache.keys(), key=lambda k: abs(k - win_sample_idx))
        p_win, v_win, w_win, _ = detector_cache[nearest_key]

        # Multimodal fusion
        f_lbl, f_conf, f_reason = fuse_multimodal_crime_decision(
            label=raw_label,
            confidence=raw_conf,
            probabilities=probs,
            people_boxes=p_win,
            weapon_boxes=w_win,
            vehicle_boxes=v_win,
            classes=classes,
            crime_threshold=CRIME_THRESHOLD
        )
        fused_window_labels.append(f_lbl)
        fused_reasons.append(f_reason)

        if (w_idx + 1) % 100 == 0 or (w_idx + 1) == total_windows:
            print(f"  [3D CNN Windows] Classified {w_idx+1}/{total_windows} windows...", flush=True)

    raw_counter = Counter(raw_window_labels)
    fused_counter = Counter(fused_window_labels)

    # Video-level overall prediction
    mean_probs = torch.mean(torch.stack(all_window_probs), dim=0).numpy()
    dominant_3d_class = classes[np.argmax(mean_probs)]
    dominant_3d_conf = float(np.max(mean_probs))

    # Overall fused decision
    top_crime_counts = {c: count for c, count in fused_counter.items() if c != "Normal"}
    if top_crime_counts:
        final_crime_label = max(top_crime_counts, key=top_crime_counts.get)
        crime_window_count = top_crime_counts[final_crime_label]
        crime_pct = (crime_window_count / total_windows) * 100
    else:
        final_crime_label = "Normal"
        crime_window_count = 0
        crime_pct = 0.0

    # Overall video fusion with verified persistent weapons
    weapon_label_counts = Counter(wp.get("label", "") for wp in all_weapons if wp.get("conf", 0) >= 0.20)
    verified_weapons = [
        wp for wp in all_weapons
        if wp.get("conf", 0) >= 0.40 or (weapon_label_counts.get(wp.get("label", ""), 0) >= 2 and wp.get("conf", 0) >= 0.20)
    ]

    video_fused_label, video_fused_conf, video_fused_reason = fuse_multimodal_crime_decision(
        label=final_crime_label if final_crime_label != "Normal" else dominant_3d_class,
        confidence=dominant_3d_conf,
        probabilities=torch.from_numpy(mean_probs),
        people_boxes=all_people,
        weapon_boxes=verified_weapons,
        vehicle_boxes=all_vehicles,
        classes=classes,
        crime_threshold=CRIME_THRESHOLD
    )

    elapsed = time.time() - t0

    # Check match with expected
    is_match = (final_crime_label.lower() == expected.lower()) or (video_fused_label.lower() == expected.lower())
    status = "PASSED ✓" if is_match else "MISMATCH"

    res_record = {
        "video": vname,
        "expected": expected,
        "raw_3d_dominant": f"{dominant_3d_class} ({dominant_3d_conf*100:.1f}%)",
        "raw_3d_windows": dict(raw_counter),
        "fused_dominant_crime": final_crime_label,
        "fused_windows": dict(fused_counter),
        "video_level_verdict": f"{video_fused_label} ({video_fused_conf*100:.1f}%)",
        "video_fused_reason": video_fused_reason,
        "people_detected": len(all_people) > 0,
        "vehicles_detected": len(all_vehicles) > 0,
        "weapons_detected": weapon_names,
        "activities": list(all_activities)[:5],
        "processing_time": f"{elapsed:.1f}s",
        "status": status
    }
    results_summary.append(res_record)

    print(f"--- Results for {vname} ---", flush=True)
    print(f"  Expected:              {expected}", flush=True)
    print(f"  Raw 3D CNN Dominant:   {dominant_3d_class} ({dominant_3d_conf*100:.1f}%)", flush=True)
    print(f"  Raw Window Breakdown:  {dict(raw_counter)}", flush=True)
    print(f"  Fused Window Breakdown:{dict(fused_counter)}", flush=True)
    print(f"  Physical Entities:     People={len(all_people)>0}, Vehicles={len(all_vehicles)>0}, Weapons={weapon_names}", flush=True)
    print(f"  Video Fused Verdict:   {video_fused_label} ({video_fused_conf*100:.1f}%) -> {video_fused_reason}", flush=True)
    print(f"  Status:                {status} (Time: {elapsed:.1f}s)", flush=True)

# Save evaluation report
out_report = PROJECT_ROOT / "outputs" / "crime_named_videos_test_report.json"
with open(out_report, "w") as f:
    json.dump(results_summary, f, indent=2)

print("\n" + "=" * 90, flush=True)
print(f"{'VIDEO':<20} | {'EXPECTED':<14} | {'3D CNN DOMINANT':<18} | {'FUSED VERDICT':<22} | {'STATUS':<10}", flush=True)
print("=" * 90, flush=True)
for r in results_summary:
    print(f"{r['video']:<20} | {r['expected']:<14} | {r['raw_3d_dominant']:<18} | {r['video_level_verdict']:<22} | {r['status']:<10}", flush=True)
print("=" * 90, flush=True)
print(f"Report saved to: {out_report}", flush=True)

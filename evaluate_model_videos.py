import os
import sys
from pathlib import Path
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import torch
import numpy as np
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.crime_model import load_crime_model

CHECKPOINT = PROJECT_ROOT / "models" / "crime_aerial_augmented_best1.pth"
VIDEOS_DIR = PROJECT_ROOT / "videos"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Kinetics Normalization
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32).reshape(3, 1, 1)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32).reshape(3, 1, 1)

# CPU with native thread parallelism for 3D / LSTM Conv
DEVICE = torch.device("cpu")
torch.set_num_threads(os.cpu_count() or 4)

print(f"Using device: {DEVICE} (threads: {torch.get_num_threads()})", flush=True)

# Load model
print(f"Loading model checkpoint from {CHECKPOINT}...", flush=True)
model, classes, checkpoint = load_crime_model(CHECKPOINT, device=DEVICE)
config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

input_shape = config.get("input_shape", [8, 112, 112])
CLIP_LEN = input_shape[0] if len(input_shape) >= 3 else 8
FRAME_SIZE = input_shape[1] if len(input_shape) >= 3 else 112
MODEL_TEMPERATURE = float(config.get("temperature", 0.7))
WINDOW_STRIDE = max(1, CLIP_LEN // 2)
BATCH_SIZE = 32

print(f"Model classes ({len(classes)}): {classes} | CLIP_LEN: {CLIP_LEN} | Temp: {MODEL_TEMPERATURE}", flush=True)

def evaluate_video(video_path):
    print(f"\n==========================================", flush=True)
    print(f"Evaluating Video: {video_path.name}", flush=True)
    print(f"==========================================", flush=True)

    t0 = time.time()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}", flush=True)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Resolution: {width}x{height} | FPS: {fps:.2f} | Total Frames: {total_frames} | Duration: {duration:.2f}s", flush=True)

    # Store normalized CHW float32 frames
    frames_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE)).astype(np.float32) / 255.0
        frame_chw = (np.transpose(frame_resized, (2, 0, 1)) - KINETICS_MEAN) / KINETICS_STD
        frames_list.append(frame_chw)
    cap.release()

    if len(frames_list) < CLIP_LEN:
        print(f"Skipping {video_path.name}: insufficient frames ({len(frames_list)} < {CLIP_LEN})", flush=True)
        return None

    # Pre-build list of 8-frame clip float32 tensors [3, 8, 112, 112]
    clip_tensors = []
    window_meta = []
    for start in range(0, len(frames_list) - CLIP_LEN + 1, WINDOW_STRIDE):
        end = start + CLIP_LEN
        clip_frames = np.stack(frames_list[start:end]) # [8, 3, 112, 112]
        clip_tensor = torch.from_numpy(clip_frames).permute(1, 0, 2, 3) # [3, 8, 112, 112]
        clip_tensors.append(clip_tensor)
        window_meta.append((start, end - 1))

    total_windows = len(window_meta)
    all_probs = []

    # Fast CPU Batched forward pass
    with torch.no_grad():
        for i in range(0, total_windows, BATCH_SIZE):
            batch_tensors = torch.stack(clip_tensors[i : i + BATCH_SIZE]).to(DEVICE)
            outputs = model(batch_tensors)
            probs = torch.softmax(outputs / MODEL_TEMPERATURE, dim=1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.vstack(all_probs) # [total_windows, num_classes]

    window_results = []
    class_counts = {c: 0 for c in classes}
    prob_sums = np.zeros(len(classes))

    for idx, (start, end_frame) in enumerate(window_meta):
        probs = all_probs[idx]
        predicted_idx = np.argmax(probs)
        label = classes[predicted_idx]
        conf = float(probs[predicted_idx])

        window_results.append({
            "window_start": start,
            "window_end": end_frame,
            "top_label": label,
            "confidence": conf,
            "probabilities": {classes[k]: float(probs[k]) for k in range(len(classes))}
        })
        class_counts[label] += 1
        prob_sums += probs

    avg_probs = {classes[i]: float(prob_sums[i] / total_windows) for i in range(len(classes))}
    overall_class = max(avg_probs, key=avg_probs.get)

    crime_classes = {c for c in classes if c != "Normal"}
    crime_windows = [w for w in window_results if w["top_label"] in crime_classes and w["confidence"] >= 0.50]

    elapsed = time.time() - t0

    video_summary = {
        "video_name": video_path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": round(duration, 2),
        "processing_time_sec": round(elapsed, 2),
        "total_windows": total_windows,
        "overall_predicted_class": overall_class,
        "overall_confidence": round(avg_probs[overall_class], 4),
        "avg_probabilities": {k: round(v, 4) for k, v in avg_probs.items()},
        "class_distribution_window_counts": class_counts,
        "crime_detected": len(crime_windows) > 0,
        "crime_windows_count": len(crime_windows),
        "crime_windows_pct": round(len(crime_windows) / total_windows * 100, 2)
    }

    print(f"\nEvaluation Summary for {video_path.name} (Processed in {elapsed:.2f}s):", flush=True)
    print(f"  Dominant Predicted Class: {overall_class} (Avg Conf: {avg_probs[overall_class]*100:.1f}%)", flush=True)
    print(f"  Crime Detected: {'YES' if len(crime_windows) > 0 else 'NO'} ({len(crime_windows)}/{total_windows} windows)", flush=True)
    print(f"  Class breakdown (window count):", flush=True)
    for cls_name, count in class_counts.items():
        if count > 0:
            print(f"    - {cls_name}: {count} windows ({count/total_windows*100:.1f}%)", flush=True)

    return video_summary

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate video models on surveillance clips")
    parser.add_argument("--video", type=str, default="", help="Evaluate specific video file")
    parser.add_argument("--max_videos", type=int, default=0, help="Max videos to evaluate (0 for all)")
    args, _ = parser.parse_known_args()

    if args.video:
        candidate = Path(args.video)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / args.video if (PROJECT_ROOT / args.video).exists() else VIDEOS_DIR / args.video
        video_files = [candidate] if candidate.exists() else []
    else:
        video_files = [
            f for f in VIDEOS_DIR.glob("*.mp4")
            if f.stat().st_size > 100
        ]
        if args.max_videos > 0:
            video_files = sorted(video_files)[:args.max_videos]

    print(f"Found {len(video_files)} total videos to evaluate.", flush=True)

    results = []
    t_start = time.time()
    for vfile in sorted(video_files):
        res = evaluate_video(vfile)
        if res:
            results.append(res)

    t_total = time.time() - t_start

    report_path = OUTPUT_DIR / "all_videos_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n==========================================", flush=True)
    print(f"ALL {len(results)} VIDEOS EVALUATED IN {t_total:.2f} SECONDS", flush=True)
    print(f"Full evaluation results saved to: {report_path}", flush=True)
    print(f"==========================================", flush=True)

if __name__ == "__main__":
    main()

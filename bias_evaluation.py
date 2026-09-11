# bias_evaluation.py
import json
import pathlib
from collections import Counter

import torch
import numpy as np
try:
    from sklearn.metrics import confusion_matrix, classification_report
except ImportError:
    def confusion_matrix(y_true, y_pred, labels):
        label_to_idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            if t in label_to_idx and p in label_to_idx:
                cm[label_to_idx[t], label_to_idx[p]] += 1
        return cm

    def classification_report(y_true, y_pred, labels, output_dict=True):
        cm = confusion_matrix(y_true, y_pred, labels)
        report = {}
        for i, cls in enumerate(labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            report[cls] = {"precision": prec, "recall": rec, "f1-score": f1, "support": int(cm[i, :].sum())}
        return report

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
VALIDATION_VIDEOS_DIR = PROJECT_ROOT / "validation_videos"
ANNOTATIONS_FILE = PROJECT_ROOT / "validation_annotations.json"
REPORT_PATH = PROJECT_ROOT / "bias_report.md"

# -------------------------------------------------
# LOAD MODEL (same as video_inference1)
# -------------------------------------------------
CHECKPOINT = PROJECT_ROOT / "models" / "crime_aerial_augmented_best1.pth"
DEVICE = torch.device("cpu")  # change to cuda/mps if available

from models.crime_model import load_crime_model

model, classes, checkpoint = load_crime_model(CHECKPOINT, device=DEVICE)

# -------------------------------------------------
# HELPER: run a single clip (reuse from video_inference1)
# -------------------------------------------------
def preprocess_frame(frame):
    import cv2
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (112, 112))
    frame = (frame.astype(np.float32) / 255.0).transpose(2, 0, 1)
    return torch.from_numpy(frame)

def create_clip(frames):
    clips = [preprocess_frame(f) for f in frames]
    clip = torch.stack(clips).permute(1, 0, 2, 3).unsqueeze(0)
    return clip.to(DEVICE)

def predict_clip(frames):
    clip = create_clip(frames)
    with torch.no_grad():
        outputs = model(clip)
    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    normal_idx = classes.index("Normal")
    normal_prob = probs[normal_idx]
    crime_prob = 1.0 - normal_prob
    # top non‑normal class
    crime_indices = [i for i, c in enumerate(classes) if c != "Normal"]
    crime_probs = probs[crime_indices]
    top_idx = np.argmax(crime_probs)
    top_class = classes[crime_indices[top_idx]]
    conf = crime_probs[top_idx]
    if crime_prob >= 0.50 and crime_prob > normal_prob:
        return top_class, conf
    else:
        return "Normal", normal_prob

# -------------------------------------------------
# MAIN: evaluate on validation set
# -------------------------------------------------
def load_annotations():
    with open(ANNOTATIONS_FILE, "r") as f:
        return json.load(f)  # expects list of {"video": "name.mp4", "label": "Robbery", ...}

def evaluate():
    ann = load_annotations()
    y_true, y_pred = [], []
    for entry in ann:
        video_path = VALIDATION_VIDEOS_DIR / entry["video"]
        # naive sampling: read first 16 frames (could be improved)
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        for _ in range(16):
            ret, frm = cap.read()
            if not ret:
                break
            frames.append(frm)
        cap.release()
        if len(frames) < 16:
            continue
        pred_label, _ = predict_clip(frames)
        y_true.append(entry["label"])
        y_pred.append(pred_label)

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True)

    robbery_idx = classes.index("Robbery")
    robbery_pred_sum = cm[:, robbery_idx].sum()
    robbery_fp = (cm[robbery_idx].sum() - cm[robbery_idx, robbery_idx]) / robbery_pred_sum if robbery_pred_sum > 0 else 0.0
    with open(REPORT_PATH, "w") as f:
        f.write("# Bias Evaluation Report\n\n")
        f.write("## Confusion Matrix\n")
        f.write("| " + " | ".join(classes) + " |\n")
        f.write("|" + "---|" * len(classes) + "\n")
        for i, row in enumerate(cm):
            f.write("| " + " | ".join(map(str, row)) + " |\n")
        f.write("\n## Classification Metrics\n")
        for cls in classes:
            metrics = report[cls]
            f.write(f"- **{cls}** – Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}\n")
        f.write(f"\n## Robbery False‑Positive Rate\n")
        f.write(f"The model predicts *Robbery* on non‑robbery frames **{robbery_fp*100:.2f}%** of the time.\n")

if __name__ == "__main__":
    evaluate()
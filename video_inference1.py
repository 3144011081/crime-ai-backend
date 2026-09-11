
from typing import Tuple, Optional, Dict, List, Union
from adaptive_acquisition import crop_zoomed_region
import os
import json
import subprocess
import time
from pathlib import Path
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import torch
import numpy as np
from models.crime_model import CrimeR2Plus1D, DomainAdaptationCrimeModel, load_crime_model
from adaptive_acquisition import run_adaptive_acquisition
from image_preprocessing import AdaptiveDronePreprocessor, SceneQualityMetrics


def convert_to_h264(video_path):
    """Re-encode MP4 video to standard H.264 (libx264, yuv420p, faststart) for native HTML5 web browser playback."""
    vpath = Path(video_path)
    if not vpath.exists():
        return vpath
    temp_path = vpath.with_name(f"{vpath.stem}_h264_temp.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(vpath),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-crf", "18",
        "-preset", "medium",
        str(temp_path)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and temp_path.exists() and temp_path.stat().st_size > 0:
            temp_path.replace(vpath)
            print(f"H.264 conversion successful: {vpath.name}")
        else:
            if temp_path.exists():
                temp_path.unlink()
    except Exception as err:
        print(f"Warning: ffmpeg conversion failed for {vpath}: {err}")
        if temp_path.exists():
            temp_path.unlink()
    return vpath


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT = PROJECT_ROOT / "models" / "crime_aerial_augmented_best1.pth"

VIDEO_PATH = (
    PROJECT_ROOT
    / "videos"
    / "hello.mp4"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "Normal",
    "Violence",
    "Robbery",
    "Shooting",
    "FireExplosion",
    "Accident",
    "Vandalism"
]

# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple MPS")

elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA")

else:
    DEVICE = torch.device("cpu")
    print("Using CPU")


# ============================================================
# LOAD MODEL & CONFIGURATION
# ============================================================

print("\nLoading 3D Spatio-Temporal Crime Model (CrimeR2Plus1D)...")

model, classes, checkpoint = load_crime_model(CHECKPOINT, device=DEVICE)
config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

print(f"Architecture: {model.__class__.__name__}")
print("Classes:", classes)

# Dynamic input shape & temperature from trained checkpoint config
input_shape = config.get("input_shape", [8, 112, 112])
CLIP_LEN = input_shape[0] if len(input_shape) >= 3 else 8
FRAME_SIZE = input_shape[1] if len(input_shape) >= 3 else 112
MODEL_TEMPERATURE = float(config.get("temperature", 0.7))

# Process every 4 frames (50% temporal overlap for 8-frame clips)
WINDOW_STRIDE = max(1, CLIP_LEN // 2)

CRIME_THRESHOLD = 0.30
CONFIRMATION_WINDOWS = 3
SMOOTHING_WINDOWS = 5

# Kinetics dataset channel-wise normalization parameters
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32).reshape(3, 1, 1)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32).reshape(3, 1, 1)

print("Model loaded successfully.")


# ============================================================
# STAGE 1: OBJECT, VEHICLE, WEAPON & ACTIVITY DETECTOR
# ============================================================

def apply_nms(boxes, iou_threshold=0.40):
    """
    Suppresses overlapping bounding boxes based on IoU and confidence score.
    """
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: b.get("conf", b.get("confidence", 0.0)), reverse=True)
    selected = []
    for candidate in sorted_boxes:
        cx1, cy1, cx2, cy2 = candidate["box"]
        c_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)
        overlap = False
        for kept in selected:
            kx1, ky1, kx2, ky2 = kept["box"]
            ix1, iy1 = max(cx1, kx1), max(cy1, ky1)
            ix2, iy2 = min(cx2, kx2), min(cy2, ky2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            k_area = max(0, kx2 - kx1) * max(0, ky2 - ky1)
            union = c_area + k_area - inter
            if union > 0 and (inter / union) > iou_threshold:
                overlap = True
                break
        if not overlap:
            selected.append(candidate)
    return selected


def people_are_interacting(people_boxes):
    """Return True only when at least two detected people are physically close."""
    if len(people_boxes) < 2:
        return False

    for first_index, first_person in enumerate(people_boxes[:-1]):
        first_box = first_person["box"]
        first_width = max(1, first_box[2] - first_box[0])
        first_height = max(1, first_box[3] - first_box[1])
        first_center = (
            (first_box[0] + first_box[2]) / 2,
            (first_box[1] + first_box[3]) / 2,
        )

        for second_person in people_boxes[first_index + 1:]:
            second_box = second_person["box"]
            second_width = max(1, second_box[2] - second_box[0])
            second_height = max(1, second_box[3] - second_box[1])
            second_center = (
                (second_box[0] + second_box[2]) / 2,
                (second_box[1] + second_box[3]) / 2,
            )

            intersection_width = max(
                0,
                min(first_box[2], second_box[2]) - max(first_box[0], second_box[0])
            )
            intersection_height = max(
                0,
                min(first_box[3], second_box[3]) - max(first_box[1], second_box[1])
            )
            if intersection_width * intersection_height > 0:
                return True

            center_distance = np.hypot(
                first_center[0] - second_center[0],
                first_center[1] - second_center[1],
            )
            interaction_distance = 0.75 * max(
                first_width,
                first_height,
                second_width,
                second_height,
            )
            if center_distance <= interaction_distance:
                return True

    return False


def prefer_vandalism_for_non_interacting_damage(label, confidence, probabilities, interacting_people):
    """Keep property-damage predictions while rejecting unsupported violence labels."""
    if label != "Violence" or interacting_people or "Vandalism" not in classes:
        return label, confidence

    vandalism_index = classes.index("Vandalism")
    vandalism_confidence = float(probabilities[vandalism_index])
    normal_index = classes.index("Normal") if "Normal" in classes else 0
    normal_confidence = float(probabilities[normal_index])
    if vandalism_confidence >= CRIME_THRESHOLD and vandalism_confidence >= normal_confidence:
        return "Vandalism", vandalism_confidence

    return "Normal", 1.0 - float(confidence)


def is_weapon_held_or_near_person(w_box, people_boxes, max_dist=180):
    """
    Validates whether a detected weapon bounding box is spatially associated
    with an actual person in the scene. In aerial surveillance, small background objects
    (debris, vehicle parts, street markings) frequently trigger false weapon detections.
    A weapon is only an active operational threat if a person is in the scene and
    in physical proximity to the object.
    """
    if not people_boxes:
        return False

    if isinstance(w_box, dict):
        wx1, wy1, wx2, wy2 = w_box.get("box", (0, 0, 0, 0))
    elif isinstance(w_box, (list, tuple)) and len(w_box) == 4:
        wx1, wy1, wx2, wy2 = w_box
    else:
        return False

    w_area = max(1, (wx2 - wx1) * (wy2 - wy1))
    wcx, wcy = (wx1 + wx2) / 2.0, (wy1 + wy2) / 2.0

    for p in people_boxes:
        if isinstance(p, dict):
            px1, py1, px2, py2 = p.get("box", (0, 0, 0, 0))
        elif isinstance(p, (list, tuple)) and len(p) == 4:
            px1, py1, px2, py2 = p
        else:
            continue

        p_area = max(1, (px2 - px1) * (py2 - py1))

        # Check intersection / overlap with person bounding box
        ix1 = max(wx1, px1)
        iy1 = max(wy1, py1)
        ix2 = min(wx2, px2)
        iy2 = min(wy2, py2)
        if ix2 > ix1 and iy2 > iy1:
            return True

        margin = max(max_dist, int(max(px2 - px1, py2 - py1) * 1.5))
        if not (wx2 < px1 - margin or wx1 > px2 + margin or wy2 < py1 - margin or wy1 > py2 + margin):
            return True

        pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        if np.hypot(wcx - pcx, wcy - pcy) <= margin:
            return True

    return False


def fuse_multimodal_crime_decision(
    label: str,
    confidence: float,
    probabilities,
    people_boxes: list,
    weapon_boxes: list,
    vehicle_boxes: list,
    classes: list,
    crime_threshold: float = 0.30
) -> Tuple[str, float, str]:
    """
    Unified Multi-Modal Crime Reasoning & Gating Engine:
    
    1. Specialized Non-Human / Environmental / Structural Crimes:
       - FireExplosion: Gated on 3D CNN thermal/visual cues or explosives.
         Protected from false weapon overrides when no persons are involved.
       - Accident: Gated on vehicle presence and 3D crash dynamics.
         Protected from weapon overrides when vehicles collide.
       - Vandalism: Single or solitary property destruction / sabotage.
         Preserved even when people are not physically fighting each other.
         
    2. Weapon Association & Proximity Gating:
       - Full-frame false positive weapon detections on empty backgrounds are filtered out.
       - Weapons are strictly required to be in spatial proximity or overlap with a detected person.
       
    3. Person-to-Person Actions:
       - Interacting people + running/snatching -> Robbery.
       - Interacting people + struggle/hitting -> Violence.
       - Verified firearm + violent altercation / shooting -> Shooting.
       - Verified firearm + snatch/run -> Armed Robbery.
       - Solitary active person + property target -> Vandalism.
       - Walking / idle persons without weapons or struggle -> Normal.
    """
    p_valid = [p for p in (people_boxes or []) if p.get("conf", 1.0) >= 0.20]
    num_people = len(p_valid)

    # Filter weapon detections: must meet threshold and be associated with a person
    valid_weapons = []
    for w in (weapon_boxes or []):
        w_conf = w.get("conf", 1.0)
        near_person = is_weapon_held_or_near_person(w, p_valid) if num_people > 0 else False
        if near_person and w_conf >= 0.20:
            valid_weapons.append(w)
        elif w_conf >= 0.45 and num_people > 0:
            valid_weapons.append(w)

    w_labels = [w.get("label", "").lower() for w in valid_weapons]
    has_firearm = any(any(term in lbl for term in ("firearm", "gun", "pistol", "rifle", "shotgun")) for lbl in w_labels)
    has_knife = any(any(term in lbl for term in ("knife", "blade", "dagger", "sword", "machete")) for lbl in w_labels)
    has_explosive = any(any(term in (w.get("label", "").lower()) for term in ("explosive", "bomb", "grenade", "dynamite")) for w in (weapon_boxes or []) if w.get("conf", 1.0) >= 0.30)

    interacting = people_are_interacting(p_valid) if num_people >= 2 else False

    speeds = [p.get("speed", 0.0) for p in p_valid]
    has_running = any(s > 8.0 for s in speeds)
    has_running_activity = any("Running" in p.get("activity", "") for p in p_valid)
    running_present = has_running or has_running_activity

    snatch_and_run = (interacting and running_present)

    # Calculate total crime probability if probabilities vector is provided
    total_crime_prob = 0.0
    if probabilities is not None and "Normal" in classes:
        normal_idx = classes.index("Normal")
        if hasattr(probabilities, "shape") and len(probabilities.shape) > 1:
            normal_p = float(probabilities[0, normal_idx])
        elif hasattr(probabilities, "__getitem__"):
            normal_p = float(probabilities[normal_idx])
        else:
            normal_p = 0.5
        total_crime_prob = 1.0 - normal_p

    # 1. NON-HUMAN SCENES (0 People detected):
    if num_people == 0:
        if len(vehicle_boxes or []) > 0 and (label == "Accident" or float(confidence) >= 0.35):
            return "Accident", max(float(confidence), 0.75), "Accident (Vehicle Incident - No Humans)"
        elif label == "FireExplosion" or has_explosive or float(confidence) >= 0.35:
            return "FireExplosion", max(float(confidence), 0.75), "Fire/Explosion Incident (Unattended Scene)"
        else:
            return "Normal", 0.95, "Normal Scene (No Humans / No Threats)"

    # 2. VEHICLE-DOMINATED SCENES (Vehicles present, no firearm):
    if len(vehicle_boxes or []) > 0 and not has_firearm:
        if label in ["Accident", "Robbery", "Violence"] or float(confidence) >= 0.35:
            return "Accident", max(float(confidence), 0.75), "Accident (Vehicle Collision / Incident)"

    # 3. SOLITARY CRIME (Single person or no interaction):
    if not interacting and not snatch_and_run and num_people > 0:
        if has_firearm:
            return "Shooting", min(max(float(confidence) + 0.25, 0.75), 0.95), "Shooting (Active Firearm Threat)"
        elif has_knife:
            return "Violence", min(max(float(confidence) + 0.25, 0.70), 0.95), "Violence (Blade / Knife Threat)"
        elif label == "FireExplosion" or has_explosive:
            return "FireExplosion", max(float(confidence), 0.75), "Fire/Explosion Incident"
        elif len(vehicle_boxes or []) > 0:
            return "Accident", max(float(confidence), 0.70), "Accident (Vehicle Incident)"
        elif label == "Vandalism" or total_crime_prob >= 0.35 or float(confidence) >= 0.30:
            return "Vandalism", max(float(confidence), 0.70), "Vandalism (Property Damage / Sabotage)"
        else:
            return "Normal", 0.92, "Normal (No Interaction / No Weapon)"

    # 4. FIREARM THREAT WITH MULTIPLE / INTERACTING PERSONS:
    if has_firearm:
        if snatch_and_run:
            return "Robbery", min(max(float(confidence) + 0.25, 0.75), 0.95), "Robbery (Armed Firearm Threat)"
        else:
            return "Shooting", min(max(float(confidence) + 0.25, 0.75), 0.95), "Shooting (Active Firearm Detected)"

    # 5. KNIFE / BLADE THREAT:
    if has_knife:
        if snatch_and_run or label == "Robbery":
            return "Robbery", min(max(float(confidence) + 0.20, 0.70), 0.95), "Robbery (Knife Threat / Snatch)"
        else:
            return "Violence", min(max(float(confidence) + 0.25, 0.70), 0.95), "Violence (Blade / Knife Detected)"

    # 6. INTERPERSONAL FIGHTING & SNATCH-AND-RUN:
    if snatch_and_run or (label == "Robbery" and float(confidence) >= 0.30):
        return "Robbery", min(max(float(confidence) + 0.15, 0.70), 0.95), "Robbery (Snatch & Run Pattern)"

    if interacting or (label == "Violence" and float(confidence) >= 0.30):
        return "Violence", min(max(float(confidence) + 0.15, 0.70), 0.95), "Violence (Physical Fight / Altercation)"

    return label, float(confidence), "3D CNN Direct Prediction"

    # 7. DEFAULT:
    return "Normal", 0.92, "Normal Movement"


def predict_clip_with_adaptive_clarity(
    clip_frames: list,
    classes: list,
    preprocessor = None,
    conf_threshold: float = 0.48
) -> Tuple[str, float, torch.Tensor, bool]:
    """
    Evaluates a video clip. If the initial prediction has low confidence (< conf_threshold)
    or is ambiguous, applies aerial drone clarity filters (CLAHE, unsharp dehazing, edge boost)
    to reveal fine-grained visual details from the aerial perspective, and re-evaluates.
    """
    label, conf, probs = predict_clip(clip_frames)
    clarity_applied = False

    prob_vec = probs[0].cpu().numpy()
    sorted_probs = np.sort(prob_vec)[::-1]
    top1_prob = sorted_probs[0]
    top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin = top1_prob - top2_prob

    if (conf < conf_threshold or margin < 0.12) and preprocessor is not None:
        try:
            enhanced_frames = [
                preprocessor.preprocess_single_frame(f, mode="aerial_drone")
                for f in clip_frames
            ]
            e_label, e_conf, e_probs = predict_clip(enhanced_frames)

            if e_conf > conf or (e_label != "Normal" and e_conf >= 0.35):
                label, conf, probs = e_label, e_conf, e_probs
                clarity_applied = True
        except Exception:
            pass

    return label, conf, probs, clarity_applied


class ObjectActivityDetector:
    """
    High-Precision Stage 1 Detector: Detects People, Vehicles (Car, Motorcycle, Bus, Truck),
    and Weapons (Guns, Knives, Firearms) with ByteTrack tracking and class filtering.
    """

    def __init__(self):
        self.person_model_path = PROJECT_ROOT / "models" / "drone_person_detector_best.pt"
        self.weapon_model_path = PROJECT_ROOT / "models" / "weapon_detector_best.pt"
        if not self.weapon_model_path.exists() and (PROJECT_ROOT / "models" / "firearmbest.pt").exists():
            self.weapon_model_path = PROJECT_ROOT / "models" / "firearmbest.pt"
        self.coco_model_path = PROJECT_ROOT / "yolov8s.pt"

        self.person_model = None
        self.weapon_model = None
        self.coco_model = None

        self._load_models()
        self.track_history = {}

    def _load_models(self):
        try:
            # pyrefly: ignore [missing-import]
            from ultralytics import YOLO
            if self.person_model_path.exists():
                self.person_model = YOLO(str(self.person_model_path))
                print(f"Loaded Drone Person Detector model ({self.person_model_path.name}).")
            if self.weapon_model_path.exists():
                self.weapon_model = YOLO(str(self.weapon_model_path))
                print(f"Loaded Firearm Detector model ({self.weapon_model_path.name}).")
            if self.coco_model_path.exists():
                self.coco_model = YOLO(str(self.coco_model_path))
                print("Loaded YOLOv8s COCO Object Detector model.")
            else:
                self.coco_model = YOLO("yolov8n.pt")
                print("Loaded Fallback COCO Object Detector model.")
        except Exception as err:
            print(f"Warning initializing detector models: {err}")

    def reset_tracker(self):
        """Reset ByteTrack tracker history for a new video stream."""
        self.track_history = {}
        for m in [self.person_model, self.coco_model]:
            if m is not None and hasattr(m, "predictor") and m.predictor is not None:
                if hasattr(m.predictor, "trackers") and m.predictor.trackers:
                    for tr in m.predictor.trackers:
                        if hasattr(tr, "reset"):
                            tr.reset()

    def detect_and_track(self, frame):
        """
        Detect People, Vehicles, Weapons and estimate activities with ByteTrack persistent IDs.
        """
        raw_people_boxes = []
        vehicle_boxes = []
        weapon_boxes = []
        activities = set()

        # ----------------------------------------------------
        # 1. Weapon Detection (Specialized Model: imgsz=960, conf=0.35)
        # ----------------------------------------------------
        if self.weapon_model is not None:
            try:
                # Primary full-frame inference at native trained resolution (960px)
                w_results = self.weapon_model(frame, imgsz=960, conf=0.35, iou=0.45, verbose=False)
                if w_results and w_results[0].boxes is not None:
                    for box in w_results[0].boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        conf = float(box.conf[0])
                        if conf < 0.35:
                            continue
                        # Reject giant bounding boxes (vehicles/roads falsely detected as firearms)
                        h_f, w_f = frame.shape[:2]
                        box_w, box_h = x2 - x1, y2 - y1
                        if box_w > w_f * 0.30 or box_h > h_f * 0.35:
                            continue
                        cls_id = int(box.cls[0])
                        w_name = self.weapon_model.names.get(cls_id, "Weapon")
                        weapon_boxes.append({
                            "box": (x1, y1, x2, y2),
                            "conf": conf,
                            "label": w_name,
                            "activity": f"Exposed {w_name} / Threat"
                        })
                        activities.add(f"Weapon Threat ({w_name})")
            except Exception:
                pass

        # ----------------------------------------------------
        # 2. COCO Detection & ByteTrack (Person, Car, Motorcycle, Bus, Truck, Knife: classes=[0,2,3,5,7,43], conf=0.25)
        # ----------------------------------------------------
        if self.coco_model is not None:
            try:
                c_results = self.coco_model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0, 2, 3, 5, 7, 43],
                    conf=0.25,
                    verbose=False
                )
                if c_results and c_results[0].boxes is not None:
                    boxes = c_results[0].boxes
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        conf = float(boxes.conf[i])
                        cls_id = int(boxes.cls[i])
                        c_name = self.coco_model.names.get(cls_id, "").lower()

                        track_id = int(boxes.id[i]) if boxes.id is not None else None

                        # COCO Weapon (knife) - threshold at 25%
                        if c_name == "knife" and conf >= 0.25:
                            weapon_boxes.append({
                                "box": (x1, y1, x2, y2),
                                "conf": conf,
                                "label": "Knife",
                                "activity": "Exposed Knife / Threat"
                            })
                            activities.add("Weapon Threat (Knife)")

                        # Vehicle classes: car, motorcycle, bus, truck
                        elif c_name in ["car", "motorcycle", "bus", "truck"]:
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            speed = 0.0
                            if track_id is not None:
                                key = f"veh_{track_id}"
                                if key in self.track_history:
                                    prev_cx, prev_cy = self.track_history[key]
                                    speed = np.hypot(cx - prev_cx, cy - prev_cy)
                                self.track_history[key] = (cx, cy)

                            act = "Vehicle In Motion" if speed > 2.0 or conf > 0.45 else "Stationary Vehicle"
                            v_label = f"{c_name.capitalize()} #{track_id}" if track_id is not None else c_name.capitalize()
                            vehicle_boxes.append({
                                "box": (x1, y1, x2, y2),
                                "conf": conf,
                                "label": v_label,
                                "activity": act
                            })
                            activities.add(act)

                        # COCO Person
                        elif c_name == "person":
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            speed = 0.0
                            if track_id is not None:
                                key = f"person_{track_id}"
                                if key in self.track_history:
                                    prev_cx, prev_cy = self.track_history[key]
                                    speed = np.hypot(cx - prev_cx, cy - prev_cy)
                                self.track_history[key] = (cx, cy)

                            p_label = f"Person #{track_id}" if track_id is not None else "Person"
                            raw_people_boxes.append({
                                "box": (x1, y1, x2, y2),
                                "conf": conf,
                                "label": p_label,
                                "cx": cx,
                                "cy": cy,
                                "speed": speed
                            })
            except Exception:
                pass

        # ----------------------------------------------------
        # 3. Drone Person Detector & ByteTrack (conf=0.25)
        # ----------------------------------------------------
        if self.person_model is not None:
            try:
                p_results = self.person_model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=0.25,
                    verbose=False
                )
                if p_results and p_results[0].boxes is not None:
                    boxes = p_results[0].boxes
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        conf = float(boxes.conf[i])
                        track_id = int(boxes.id[i]) if boxes.id is not None else None

                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        speed = 0.0
                        if track_id is not None:
                            key = f"drone_person_{track_id}"
                            if key in self.track_history:
                                prev_cx, prev_cy = self.track_history[key]
                                speed = np.hypot(cx - prev_cx, cy - prev_cy)
                            self.track_history[key] = (cx, cy)

                        p_label = f"Person #{track_id}" if track_id is not None else "Person"
                        raw_people_boxes.append({
                            "box": (x1, y1, x2, y2),
                            "conf": conf,
                            "label": p_label,
                            "cx": cx,
                            "cy": cy,
                            "speed": speed
                        })
            except Exception:
                pass

        # ----------------------------------------------------
        # 4. IoU Non-Maximum Suppression & Activity Assignment
        # ----------------------------------------------------
        nms_people = apply_nms(raw_people_boxes, iou_threshold=0.40)

        # Contextual High-Resolution Weapon Search around detected people
        if self.weapon_model is not None and len(nms_people) > 0:
            h_f, w_f = frame.shape[:2]
            for p in nms_people:
                px1, py1, px2, py2 = p["box"]
                pw, ph = px2 - px1, py2 - py1
                if ph < 30 or pw < 20:
                    continue
                pad_w = int(pw * 0.35)
                pad_h = int(ph * 0.35)
                cx1, cy1 = max(0, px1 - pad_w), max(0, py1 - pad_h)
                cx2, cy2 = min(w_f, px2 + pad_w), min(h_f, py2 + pad_h)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    try:
                        cw_res = self.weapon_model(crop, imgsz=640, conf=0.20, iou=0.45, verbose=False)
                        if cw_res and cw_res[0].boxes is not None:
                            for wb in cw_res[0].boxes:
                                wx1, wy1, wx2, wy2 = map(int, wb.xyxy[0].cpu().numpy())
                                conf = float(wb.conf[0])
                                if conf < 0.20:
                                    continue
                                wb_w, wb_h = wx2 - wx1, wy2 - wy1
                                if wb_w > pw * 2.0 or wb_h > ph * 2.0:
                                    continue
                                cls_id = int(wb.cls[0])
                                w_name = self.weapon_model.names.get(cls_id, "Weapon")
                                weapon_boxes.append({
                                    "box": (cx1 + wx1, cy1 + wy1, cx1 + wx2, cy1 + wy2),
                                    "conf": conf,
                                    "label": w_name,
                                    "activity": f"Exposed {w_name} / Threat"
                                })
                                activities.add(f"Weapon Threat ({w_name})")
                    except Exception:
                        pass

        # Apply NMS on weapons & vehicles to eliminate duplicates
        weapon_boxes = apply_nms(weapon_boxes, iou_threshold=0.35)
        # Filter weapons to strictly those near detected people in THIS frame (or explosives)
        if len(nms_people) > 0:
            weapon_boxes = [
                wb for wb in weapon_boxes
                if is_weapon_held_or_near_person(wb, nms_people, max_dist=150) or "explosive" in wb.get("label", "").lower()
            ]
        else:
            # If no people in this frame, firearms and blades cannot float in empty air
            weapon_boxes = [
                wb for wb in weapon_boxes
                if "explosive" in wb.get("label", "").lower() and wb.get("conf", 0) >= 0.40
            ]
        vehicle_boxes = apply_nms(vehicle_boxes, iou_threshold=0.40)

        people_boxes = []
        for p in nms_people:
            cx, cy = p["cx"], p["cy"]
            speed = p["speed"]

            has_weapon = any(
                abs(cx - (wb["box"][0] + wb["box"][2]) // 2) < 130 and
                abs(cy - (wb["box"][1] + wb["box"][3]) // 2) < 130
                for wb in weapon_boxes
            )

            if has_weapon:
                act = "Armed / Threat"
            elif speed > 8.0:
                act = "Running / Fast Motion"
            else:
                act = "Walking / Active"

            p["activity"] = act
            people_boxes.append(p)
            activities.add(act)

        summary = {
            "people_count": len(people_boxes),
            "vehicle_count": len(vehicle_boxes),
            "weapon_count": len(weapon_boxes),
            "activities": list(activities)
        }

        return people_boxes, vehicle_boxes, weapon_boxes, summary





# Initialize detector instance
detector_engine = ObjectActivityDetector()

# Initialize global adaptive image/video preprocessor
drone_preprocessor = AdaptiveDronePreprocessor(mode="auto")


# ============================================================
# FRAME PREPROCESSING
# ============================================================

def preprocess_frame(frame, apply_enhancement=False):
    """
    OpenCV BGR frame
    ->
    Optional Environmental Enhancement
    ->
    RGB
    ->
    112x112
    ->
    Kinetics Normalization (mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ->
    Tensor [C,H,W]
    """
    if apply_enhancement and drone_preprocessor is not None:
        frame = drone_preprocessor.preprocess_single_frame(frame)

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frame = cv2.resize(
        frame,
        (FRAME_SIZE, FRAME_SIZE)
    )

    frame = frame.astype(
        np.float32
    ) / 255.0

    # H,W,C -> C,H,W
    frame = np.transpose(
        frame,
        (2, 0, 1)
    )

    # Apply Kinetics channel-wise normalization
    frame = (frame - KINETICS_MEAN) / KINETICS_STD

    return torch.from_numpy(frame)


def apply_temporal_majority_smoothing(frame_predictions, classes, window_radius=2, min_crime_votes=3):
    """
    Applies temporal majority voting and 1D temporal moving average smoothing across contiguous frames.
    Filters out transient single-frame spikes / glitches that do not persist across related frames.

    Args:
        frame_predictions: list of lists of (label, confidence, prob_tensor)
        classes: list of class names
        window_radius: radius K for temporal neighborhood [t - K, t + K] (default: 2 -> 5 frames)
        min_crime_votes: minimum number of frames in temporal neighborhood predicting crime

    Returns:
        smoothed_window_predictions: dict mapping frame_index -> (smoothed_label, smoothed_confidence)
    """
    num_frames = len(frame_predictions)
    normal_index = classes.index("Normal") if "Normal" in classes else 0

    raw_frame_probs = []
    for frame_index in range(num_frames):
        preds = frame_predictions[frame_index]
        if not preds:
            prob_vec = torch.zeros(len(classes), dtype=torch.float32)
            prob_vec[normal_index] = 1.0
        else:
            prob_stack = torch.stack([
                p[2] if isinstance(p[2], torch.Tensor) else torch.as_tensor(p[2], dtype=torch.float32)
                for p in preds
            ], dim=0)
            prob_vec = torch.mean(prob_stack, dim=0)
        raw_frame_probs.append(prob_vec)

    if not raw_frame_probs:
        return {}

    raw_matrix = torch.stack(raw_frame_probs, dim=0)

    smoothed_window_predictions = {}

    for t in range(num_frames):
        t_start = max(0, t - window_radius)
        t_end = min(num_frames, t + window_radius + 1)

        neighborhood_matrix = raw_matrix[t_start:t_end]
        mean_neighborhood_prob = torch.mean(neighborhood_matrix, dim=0)

        neighborhood_top_classes = [
            classes[torch.argmax(raw_matrix[i]).item()]
            for i in range(t_start, t_end)
        ]
        crime_vote_count = sum(1 for c in neighborhood_top_classes if c != "Normal")

        top_conf, top_class_id = torch.max(mean_neighborhood_prob, dim=0)
        top_label = classes[top_class_id.item()]

        # Temporal Majority Filter: If top class is a Crime, but fewer than min_crime_votes neighbor frames agree, suppress spike
        if top_label != "Normal" and crime_vote_count < min_crime_votes:
            final_label = "Normal"
            final_conf = mean_neighborhood_prob[normal_index].item()
        else:
            final_label = top_label
            final_conf = top_conf.item()

        smoothed_window_predictions[t] = (final_label, final_conf)

    return smoothed_window_predictions


# ============================================================
# CREATE CLIP
# ============================================================

def create_clip(frames, apply_temporal_smoothing=False):
    """
    Input:
        list of OpenCV frames (length: CLIP_LEN, e.g. 8)

    Output:
        Tensor [1, 3, CLIP_LEN, 112, 112]
    """
    if apply_temporal_smoothing and drone_preprocessor is not None:
        enhanced_frames = drone_preprocessor.preprocess_video_clip(frames)
    else:
        enhanced_frames = frames

    processed = []
    for frame in enhanced_frames:
        processed.append(
            preprocess_frame(frame, apply_enhancement=False)
        )

    # [T,C,H,W]
    clip = torch.stack(
        [
            p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float32)
            for p in processed
        ]
    )

    # [T,C,H,W] -> [C,T,H,W]
    clip = clip.permute(
        1, 0, 2, 3
    )

    # Add batch dimension -> [1, C, T, H, W]
    clip = clip.unsqueeze(0)

    return clip


# ============================================================
# PREDICT ONE CLIP
# ============================================================

@torch.no_grad()
def predict_clip(frames):

    clip = create_clip(
        frames,
        apply_temporal_smoothing=False
    )

    clip = clip.to(
        DEVICE
    )

    outputs = model(
        clip
    )

    # Apply trained temperature scaling for calibrated probabilities
    probabilities = torch.softmax(
        outputs / MODEL_TEMPERATURE,
        dim=1
    )

    normal_index = classes.index("Normal") if "Normal" in classes else 0
    normal_prob = probabilities[0, normal_index].item()

    crime_indices = [i for i, c in enumerate(classes) if c != "Normal"]
    crime_probs = probabilities[0, crime_indices]
    top_crime_conf, top_crime_sub_idx = torch.max(crime_probs, dim=0)
    top_crime_class = classes[crime_indices[top_crime_sub_idx.item()]]
    top_crime_val = top_crime_conf.item()

    top_idx = torch.argmax(probabilities, dim=1).item()
    top_label = classes[top_idx]
    top_conf = probabilities[0, top_idx].item()
    total_crime_prob = 1.0 - normal_prob

    if total_crime_prob >= CRIME_THRESHOLD and total_crime_prob >= normal_prob:
        class_name = top_crime_class
        confidence = min(total_crime_prob, 0.95)
    elif top_label != "Normal" and top_conf >= CRIME_THRESHOLD:
        class_name = top_label
        confidence = min(top_conf, 0.95)
    else:
        class_name = "Normal"
        confidence = normal_prob

    return (
        class_name,
        confidence,
        probabilities
    )



# ============================================================
# DRAW PREDICTION ON FRAME
# ============================================================


# ============================================================
# DRAW PREDICTION ON FRAME
# ============================================================

def draw_corner_accents(img, x1, y1, x2, y2, color, length=16, thickness=3):
    """Draw tactical HUD corner brackets around a bounding box."""
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


def draw_prediction(
    frame,
    label,
    confidence,
    crime_threshold=0.30,
    detector_data=None,
    acquisition_mode="original",
    pixel_coordinates=None,
    pip_crop=None,
    pip_title=None
):
    output = frame.copy()
    h, w = output.shape[:2]

    crime_classes = {
        "Violence",
        "Robbery",
        "Shooting",
        "FireExplosion",
        "Accident",
        "Vandalism"
    }

    # --------------------------------------------------------
    # Unified Multi-Modal Crime Reasoning & Gating Engine
    # --------------------------------------------------------
    p_boxes_frame = detector_data[0] if detector_data else []
    v_boxes_frame = detector_data[1] if detector_data else []
    w_boxes_frame = detector_data[2] if detector_data else []

    label, confidence, reason_tag = fuse_multimodal_crime_decision(
        label=label,
        confidence=confidence,
        probabilities=None,
        people_boxes=p_boxes_frame,
        weapon_boxes=w_boxes_frame,
        vehicle_boxes=v_boxes_frame,
        classes=classes,
        crime_threshold=crime_threshold
    )

    weapon_boxes_valid = [wp for wp in w_boxes_frame if wp.get("conf", 1.0) >= 0.30]
    crime = (
        label in crime_classes
        and confidence >= crime_threshold
    )

    # --------------------------------------------------------
    # 1. Stage 1 & Crime Bounding Boxes (Weapons, People, Vehicles, Crime Spot)
    # --------------------------------------------------------
    if detector_data:
        people_boxes, vehicle_boxes, weapon_boxes, summary = detector_data

        # Detect all weapons present in scene (threshold >= 30% confidence)
        scene_weapon_labels = []
        for wp in weapon_boxes:
            if wp.get("conf", 1.0) < 0.30:
                continue
            w_name = wp.get("label", "Weapon").strip()
            if w_name and w_name.upper() not in [x.upper() for x in scene_weapon_labels]:
                scene_weapon_labels.append(w_name)

        # ----------------------------------------------------
        # 1. Draw CRIME SPOT Bounding Box (Only when Crime is active)
        # ----------------------------------------------------
        if crime:
            spot_coords = []
            if people_boxes:
                spot_coords.extend([p["box"] for p in people_boxes])
            if not spot_coords and vehicle_boxes and label == "Accident":
                spot_coords.extend([v["box"] for v in vehicle_boxes])

            if spot_coords:
                cs_x1 = max(0, min([b[0] for b in spot_coords]) - 25)
                cs_y1 = max(0, min([b[1] for b in spot_coords]) - 30)
                cs_x2 = min(w - 1, max([b[2] for b in spot_coords]) + 25)
                cs_y2 = min(h - 1, max([b[3] for b in spot_coords]) + 25)
            elif pixel_coordinates and len(pixel_coordinates) == 4:
                cs_x1, cs_y1, cs_x2, cs_y2 = pixel_coordinates
            else:
                cs_x1, cs_y1, cs_x2, cs_y2 = int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)

            # Draw prominent Red Crime Spot Box with double border and corner accents
            cv2.rectangle(output, (cs_x1, cs_y1), (cs_x2, cs_y2), (0, 0, 255), 3)
            cv2.rectangle(output, (max(0, cs_x1 - 2), max(0, cs_y1 - 2)), (min(w - 1, cs_x2 + 2), min(h - 1, cs_y2 + 2)), (0, 0, 180), 1)
            draw_corner_accents(output, cs_x1, cs_y1, cs_x2, cs_y2, (0, 0, 255), length=25, thickness=4)

            # Top Banner for Crime Spot
            spot_txt = f"CRIME SPOT: {label.upper()} ({confidence * 100:.1f}%)"
            st_size = cv2.getTextSize(spot_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            sy2 = cs_y1 if cs_y1 - 28 >= 70 else min(cs_y2, cs_y1 + 28)
            sy1 = max(0, sy2 - 28)
            cv2.rectangle(output, (cs_x1, sy1), (min(w - 1, cs_x1 + st_size[0] + 16), sy2), (0, 0, 220), -1)
            cv2.putText(output, spot_txt, (cs_x1 + 8, sy2 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # --------------------------------------------------------
    # 2. Stage 2 Header HUD Banner & Corner Crime/Normal Status
    # --------------------------------------------------------
    banner_height = 54
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)

    status_color = (0, 0, 255) if crime else (0, 230, 0)
    status_text = f"CRIME: {label.upper()} ({confidence * 100:.1f}%)" if crime else f"NORMAL ({confidence * 100:.1f}%)"

    # Status Circle Indicator on Corner
    cv2.circle(output, (25, 27), 10, status_color, -1)
    cv2.circle(output, (25, 27), 12, (255, 255, 255), 1)

    # Corner Status Text
    cv2.putText(
        output,
        status_text,
        (45, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Acquisition Mode Badge (Top Right HUD)
    if acquisition_mode != "original":
        if "fsrcnn_person_zoom_" in acquisition_mode:
            try:
                zoom_str = acquisition_mode.split("person_zoom_")[1].split("_")[0]
                acq_badge_text = f"FSRCNN {zoom_str}x ZOOM"
            except (IndexError, ValueError):
                acq_badge_text = "FSRCNN ZOOM"
        elif "person_zoom_" in acquisition_mode:
            acq_badge_text = f"ADAPTIVE ZOOM ({acquisition_mode})"
        elif "grid_pixel_tile_" in acquisition_mode:
            acq_badge_text = "PIXEL GRID SCAN"
        else:
            acq_badge_text = f"ADAPTIVE ZOOM ({acquisition_mode})"
        acq_bg_color = (255, 140, 0)
    else:
        acq_badge_text = "ORIGINAL VIEW"
        acq_bg_color = (0, 180, 0)

    badge_w = len(acq_badge_text) * 9
    cv2.rectangle(output, (max(0, w - badge_w - 20), 12), (w - 15, 42), acq_bg_color, -1)
    cv2.putText(
        output,
        acq_badge_text,
        (max(5, w - badge_w - 12), 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # 3. Bottom Telemetry Bar (Entities & Activities)
    # --------------------------------------------------------
    if detector_data:
        _, _, _, summary = detector_data
        panel_y = h - 35
        panel_overlay = output.copy()
        cv2.rectangle(panel_overlay, (0, panel_y), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(panel_overlay, 0.85, output, 0.15, 0, output)

        acts_str = ", ".join(summary['activities'][:3]) if summary['activities'] else "Normal Behavior"
        telemetry_str = (
            f"People: {summary['people_count']}  |  "
            f"Vehicles: {summary['vehicle_count']}  |  "
            f"Weapons: {summary['weapon_count']}  |  "
            f"Activity: {acts_str}"
        )

        cv2.putText(
            output,
            telemetry_str,
            (15, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # 4. Picture-in-Picture (PiP) Inset for Adaptive Zoom / FSRCNN ROI
    # --------------------------------------------------------
    if pip_crop is not None and getattr(pip_crop, "size", 0) > 0:
        try:
            ph, pw = pip_crop.shape[:2]
            if pw > 0 and ph > 0:
                pip_target_w = min(240, max(140, int(w * 0.22)))
                pip_target_h = int(pip_target_w * (ph / pw))
                pip_target_h = max(80, min(160, pip_target_h))

                px2 = w - 15
                px1 = px2 - pip_target_w
                py1 = 56
                py2 = py1 + pip_target_h

                if py2 < h - 45 and px1 > 0:
                    pip_resized = cv2.resize(pip_crop, (pip_target_w, pip_target_h), interpolation=cv2.INTER_LANCZOS4)
                    pip_border_col = (0, 0, 255) if crime else (0, 200, 255)

                    # Inset image
                    output[py1:py2, px1:px2] = pip_resized
                    # Frame border
                    cv2.rectangle(output, (px1 - 2, py1 - 2), (px2 + 2, py2 + 2), pip_border_col, 2)

                    # Header label badge
                    tag_txt = pip_title or "AI ENHANCED ROI"
                    (tw, th), _ = cv2.getTextSize(tag_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                    cv2.rectangle(output, (px1 - 2, py1 - 18), (px1 + tw + 10, py1), (15, 15, 15), -1)
                    cv2.rectangle(output, (px1 - 2, py1 - 18), (px1 + tw + 10, py1), pip_border_col, 1)
                    cv2.putText(output, tag_txt, (px1 + 4, py1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

    return output



# helloed
def extract_evidence_clips(
    frames,
    fps,
    window_results,
    output_dir=None,
    crime_threshold=0.30,
    min_consecutive=2,
    padding_seconds=2.0,
    video_stem=None,
    detector_engine=None,
    pixel_coordinates=None
):
    """
    Extract consecutive crime detections as evidence videos.

    window_results must contain:
        start
        end
        label
        confidence
        probabilities
    """

    output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "outputs" / "evidence"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    crime_classes = {
        "Violence",
        "Robbery",
        "Shooting",
        "FireExplosion",
        "Accident",
        "Vandalism"
    }

    events = []

    current_event = None
    consecutive = 0

    # --------------------------------------------------------
    # Find consecutive crime windows
    # --------------------------------------------------------

    for result in window_results:

        label = result["label"]
        confidence = result["confidence"]

        is_crime = (
            label in crime_classes
            and confidence >= crime_threshold
        )

        if is_crime:

            consecutive += 1

            if current_event is None:

                current_event = {
                    "start": result["start"],
                    "end": result["end"],
                    "label": label,
                    "confidence": confidence
                }

            else:

                current_event["end"] = result["end"]

                # Keep the strongest prediction
                if confidence > current_event["confidence"]:

                    current_event["label"] = label
                    current_event["confidence"] = confidence

        else:

            if (
                current_event is not None
                and consecutive >= min_consecutive
            ):

                events.append(
                    current_event
                )

            current_event = None
            consecutive = 0

    # --------------------------------------------------------
    # Handle event reaching end of video
    # --------------------------------------------------------

    if (
        current_event is not None
        and consecutive >= min_consecutive
    ):

        events.append(
            current_event
        )

    # --------------------------------------------------------
    # Extract video clips
    # --------------------------------------------------------

    evidence_files = []

    print("\n======================================")
    print("CRIME EVENTS")
    print("======================================")

    if len(events) == 0:

        print(
            "No crime events met the threshold."
        )

        return []

    for event_number, event in enumerate(
        events,
        start=1
    ):

        # ----------------------------------------------------
        # Add padding
        # ----------------------------------------------------

        padding_frames = int(
            padding_seconds * fps
        )

        start_frame = max(
            0,
            event["start"] - padding_frames
        )

        end_frame = min(
            len(frames) - 1,
            event["end"] + padding_frames
        )

        # ----------------------------------------------------
        # Timestamps
        # ----------------------------------------------------

        start_time = (
            start_frame / fps
            if fps > 0
            else 0
        )

        end_time = (
            end_frame / fps
            if fps > 0
            else 0
        )

        # ----------------------------------------------------
        # Output filename
        # ----------------------------------------------------

        safe_label = (
            event["label"]
            .lower()
            .replace(" ", "_")
        )

        prefix = f"{video_stem}_" if video_stem else ""
        output_path = output_dir / f"{prefix}{safe_label}_{event_number:02d}.mp4"

        # ----------------------------------------------------
        # Video dimensions
        # ----------------------------------------------------

        height, width = frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():

            print(
                f"Could not create: {output_path}"
            )

            continue

        # ----------------------------------------------------
        # Write evidence frames
        # ----------------------------------------------------

        for frame_index in range(
            start_frame,
            end_frame + 1
        ):

            frame = frames[
                frame_index
            ].copy()
            h, w = frame.shape[:2]

            # -----------------------------------------------
            # Detect objects / crime spot for evidence box
            # -----------------------------------------------
            detector_data = None
            if detector_engine is not None:
                try:
                    p_boxes, v_boxes, w_boxes, summary = detector_engine.detect_and_track(frame)
                    detector_data = (p_boxes, v_boxes, w_boxes, summary)
                except Exception:
                    detector_data = None

            # -----------------------------------------------
            # 1. Draw CRIME SPOT Bounding Box (Red Box + Crime Type)
            # -----------------------------------------------
            spot_coords = []
            if detector_data:
                p_boxes, v_boxes, w_boxes, _ = detector_data
                if p_boxes:
                    spot_coords.extend([p["box"] for p in p_boxes])
                if not spot_coords and v_boxes and event["label"] == "Accident":
                    spot_coords.extend([v["box"] for v in v_boxes])

            if spot_coords:
                cs_x1 = max(0, min([b[0] for b in spot_coords]) - 25)
                cs_y1 = max(0, min([b[1] for b in spot_coords]) - 30)
                cs_x2 = min(w - 1, max([b[2] for b in spot_coords]) + 25)
                cs_y2 = min(h - 1, max([b[3] for b in spot_coords]) + 25)
            elif pixel_coordinates and len(pixel_coordinates) == 4:
                cs_x1, cs_y1, cs_x2, cs_y2 = pixel_coordinates
            else:
                cs_x1, cs_y1, cs_x2, cs_y2 = int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)

            # Draw prominent Red Crime Spot Box with double border and corner accents
            cv2.rectangle(frame, (cs_x1, cs_y1), (cs_x2, cs_y2), (0, 0, 255), 3)
            cv2.rectangle(frame, (max(0, cs_x1 - 2), max(0, cs_y1 - 2)), (min(w - 1, cs_x2 + 2), min(h - 1, cs_y2 + 2)), (0, 0, 180), 1)
            draw_corner_accents(frame, cs_x1, cs_y1, cs_x2, cs_y2, (0, 0, 255), length=25, thickness=4)

            # Top Banner for Crime Spot with Crime Type
            spot_txt = f"CRIME SPOT: {event['label'].upper()} ({event['confidence'] * 100:.1f}%)"
            st_size = cv2.getTextSize(spot_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            sy2 = cs_y1 if cs_y1 - 28 >= 140 else min(cs_y2, cs_y1 + 28)
            sy1 = max(0, sy2 - 28)
            cv2.rectangle(frame, (cs_x1, sy1), (min(w - 1, cs_x1 + st_size[0] + 16), sy2), (0, 0, 220), -1)
            cv2.putText(frame, spot_txt, (cs_x1 + 8, sy2 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            # -----------------------------------------------
            # Evidence HUD overlay (Top-Left metadata)
            # -----------------------------------------------
            cv2.rectangle(
                frame,
                (15, 15),
                (500, 125),
                (20, 20, 20),
                -1
            )

            cv2.putText(
                frame,
                "CRIME EVIDENCE",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Type: {event['label']}",
                (30, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Confidence: "
                f"{event['confidence'] * 100:.1f}%",
                (30, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            writer.write(
                frame
            )

        writer.release()

        evidence_files.append(
            output_path
        )

        print(
            f"\nEvent {event_number}"
        )

        print(
            f"Type: {event['label']}"
        )

        print(
            f"Confidence: "
            f"{event['confidence'] * 100:.2f}%"
        )

        print(
            f"Time: "
            f"{start_time:.2f}s -> "
            f"{end_time:.2f}s"
        )

        print(
            f"Saved: {output_path}"
        )

    print("\n======================================")

    return evidence_files


# ============================================================
# VIDEO PROCESSING
# ============================================================


def process_video(video_path, progress_callback=None):

    video_path = Path(video_path)

    # Reset ByteTrack tracking state for new video
    detector_engine.reset_tracker()

    if progress_callback:
        progress_callback(0.05, f"Opening video stream: {video_path.name}...")

    print("\nOpening video:")
    print(video_path)

    # Create output filename based on uploaded video
    video_name = video_path.stem

    output_video = (
        OUTPUT_DIR
        / f"{video_name}_result.mp4"
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    print(f"\nFPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Resolution: {width}x{height}")
    print(f"Duration: {duration:.2f} seconds")

    # ========================================================
    # READ ALL FRAMES
    # ========================================================

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    print(
        f"Frames loaded: {len(frames)}"
    )

    if progress_callback:
        progress_callback(0.15, f"Loaded {len(frames)} frames. Starting spatio-temporal sliding window classification...")

    if len(frames) < CLIP_LEN:

        raise RuntimeError(
            f"Video has only {len(frames)} frames. "
            f"Need at least {CLIP_LEN}."
        )

    # ========================================================
    # SLIDING WINDOW PREDICTIONS
    # ========================================================

    window_predictions = {}

    results = []
    # Store all predictions contributing to each frame
    frame_predictions = [
        [] for _ in range(len(frames))
    ]

    print("\n======================================")
    print("WINDOW PREDICTIONS")
    print("======================================")

    window_number = 0
    total_window_steps = max(1, len(range(0, len(frames) - CLIP_LEN + 1, WINDOW_STRIDE)))

    for start in range(
        0,
        len(frames) - CLIP_LEN + 1,
        WINDOW_STRIDE
    ):

        end = start + CLIP_LEN

        clip_frames = frames[start:end]

        window_number += 1

        # 1. Predict clip with adaptive aerial clarity enhancement for low-confidence windows
        raw_label, raw_confidence, probabilities, clarity_boost = predict_clip_with_adaptive_clarity(
            clip_frames,
            classes,
            preprocessor=drone_preprocessor,
            conf_threshold=0.48
        )

        # 2. Sample window frames to collect physical detector evidence
        win_sample_indices = list(set([start, (start + end) // 2, end - 1]))
        p_boxes_win = []
        w_boxes_win = []
        v_boxes_win = []
        for s_idx in win_sample_indices:
            pb_s, vb_s, wb_s, _ = detector_engine.detect_and_track(frames[s_idx])
            p_boxes_win.extend(pb_s)
            v_boxes_win.extend(vb_s)
            w_boxes_win.extend(wb_s)

        # 3. Apply Unified Multi-Modal Decision & Gating Engine
        label, confidence, reason_tag = fuse_multimodal_crime_decision(
            label=raw_label,
            confidence=raw_confidence,
            probabilities=probabilities,
            people_boxes=p_boxes_win,
            weapon_boxes=w_boxes_win,
            vehicle_boxes=v_boxes_win,
            classes=classes,
            crime_threshold=CRIME_THRESHOLD
        )

        probabilities_cpu = torch.zeros((1, len(classes)), dtype=torch.float32)
        label_index = classes.index(label) if label in classes else 0
        probabilities_cpu[0, label_index] = float(confidence)

        results.append({
            "label": label,
            "confidence": float(confidence),
            "probabilities": probabilities_cpu.squeeze(0).numpy().tolist(),
            "start": start,
            "end": end - 1,
        })

        if progress_callback:
            pct = 0.15 + 0.35 * (window_number / total_window_steps)
            progress_callback(pct, f"Classifying window {window_number}/{total_window_steps} — {label} ({confidence * 100:.1f}%)...")

        # ----------------------------------------------------
        # Store prediction for every frame in this window
        # ----------------------------------------------------

        for frame_index in range(start, end):

            frame_predictions[frame_index].append(
                (
                    label,
                    confidence,
                    probabilities_cpu.squeeze(0)
                )
            )
        # ----------------------------------------------------
        # Calculate averaged prediction for this frame
        # ----------------------------------------------------

        for frame_index in range(start, end):

            predictions = frame_predictions[frame_index]

            probability_stack = torch.stack(
                [
                    prediction[2] if isinstance(prediction[2], torch.Tensor)
                    else torch.as_tensor(prediction[2], dtype=torch.float32)
                    for prediction in predictions
                ],
                dim=0
            )

            mean_frame_probabilities = torch.mean(
                probability_stack,
                dim=0
            )

            frame_confidence, frame_class_id = torch.max(
                mean_frame_probabilities,
                dim=0
            )

            frame_label = classes[
                frame_class_id.item()
            ]

            window_predictions[frame_index] = (
                frame_label,
                frame_confidence.item()
            )

        print(
            f"Window {window_number:03d} | "
            f"Frames {start:05d}-{end - 1:05d} | "
            f"{label:<15} "
            f"{confidence * 100:.2f}%"
        )

    # ----------------------------------------------------
    # Apply Temporal Majority Voting & Window Smoothing
    # ----------------------------------------------------
    window_predictions = apply_temporal_majority_smoothing(
        frame_predictions,
        classes,
        window_radius=2,
        min_crime_votes=1
    )

    # ========================================================
    # WHOLE VIDEO PROBABILITIES
    # ========================================================

    prob_tensors = [
        res["probabilities"] if isinstance(res["probabilities"], torch.Tensor)
        else torch.as_tensor(res["probabilities"], dtype=torch.float32)
        for res in results
    ]
    all_probabilities = torch.stack(prob_tensors, dim=0)

    mean_probabilities = torch.mean(
        all_probabilities,
        dim=0
    )

    normal_index = classes.index("Normal") if "Normal" in classes else 0
    normal_prob = mean_probabilities[normal_index].item()

    crime_indices = [i for i, c in enumerate(classes) if c != "Normal"]
    crime_probs = mean_probabilities[crime_indices]
    top_crime_conf, top_crime_sub_idx = torch.max(crime_probs, dim=0)
    top_crime_class = classes[crime_indices[top_crime_sub_idx.item()]]
    top_crime_val = top_crime_conf.item()

    crime_results = [r for r in results if r["label"] != "Normal" and r["confidence"] >= CRIME_THRESHOLD]
    crime_ratio = len(crime_results) / max(len(results), 1)
    total_crime_prob = 1.0 - normal_prob
    is_crime = (crime_ratio >= 0.15) or (total_crime_prob >= 0.40 and total_crime_prob >= normal_prob)

    if is_crime and crime_results:
        from collections import Counter
        crime_labels = [r["label"] for r in crime_results]
        dominant_crime = Counter(crime_labels).most_common(1)[0][0]
        final_class = dominant_crime
        dom_confs = [r["confidence"] for r in crime_results if r["label"] == dominant_crime]
        final_confidence = torch.as_tensor(float(np.mean(dom_confs)), dtype=torch.float32)
        crime_probability = final_confidence.item()
    else:
        is_crime = False
        final_class = "Normal"
        final_confidence = torch.as_tensor(normal_prob if normal_prob >= 0.50 else 0.90, dtype=torch.float32)
        crime_probability = 0.0

    # ========================================================
    # ADAPTIVE ACQUISITION (if confidence is low)
    # ========================================================

    acquisition_mode = "original"

    def _detect_people_boxes(frame):
        """Extract person bounding boxes from detector engine for adaptive acquisition."""
        p_boxes, _, _, _ = detector_engine.detect_and_track(frame)
        return [(p['box'][0], p['box'][1], p['box'][2], p['box'][3]) for p in p_boxes]

    def _predict_clip_wrapper(clip_frames):
        """Wrapper for predict_clip that returns (label, confidence, probs_numpy)."""
        label, conf, probs = predict_clip(clip_frames)
        return label, conf, probs.cpu().numpy()

    # Sample representative contiguous frames around center of video for adaptive acquisition
    mid_idx = len(frames) // 2
    clip_start_idx = max(0, min(mid_idx - CLIP_LEN // 2, len(frames) - CLIP_LEN))
    sample_clip = frames[clip_start_idx : clip_start_idx + CLIP_LEN]

    adaptive_result = run_adaptive_acquisition(
        predict_fn=_predict_clip_wrapper,
        detect_people_fn=_detect_people_boxes,
        frames=sample_clip,
        original_label=final_class,
        original_conf=final_confidence.item(),
        original_probs=mean_probabilities.cpu().numpy()
    )

    acquisition_mode = adaptive_result["mode"]
    pixel_coordinates = adaptive_result.get("pixel_coordinates", None)

    # If adaptive acquisition found a better prediction, update
    if adaptive_result["mode"] != "original":
        coord_msg = f" at pixel ROI {pixel_coordinates}" if pixel_coordinates else ""
        print(f"\nAdaptive acquisition improved prediction via {acquisition_mode}{coord_msg}")
        final_class = adaptive_result["prediction"]
        final_confidence = torch.as_tensor(adaptive_result["confidence"], dtype=torch.float32)

        # Recalculate crime status
        is_crime = final_class != "Normal" and adaptive_result["confidence"] >= CRIME_THRESHOLD
        if is_crime:
            crime_probability = adaptive_result["confidence"]

    # ========================================================
    # 2-STAGE ENTITY GATING & FALSE POSITIVE SUPPRESSION
    # ========================================================
    # Verify entity presence across sampled frames to ensure physical consistency
    entity_check_indices = np.linspace(0, len(frames) - 1, min(len(frames), 25)).astype(int)
    all_sampled_people = []
    all_sampled_vehicles = []
    all_sampled_weapons = []

    for s_idx in entity_check_indices:
        pb, vb, wb, _ = detector_engine.detect_and_track(frames[s_idx])
        all_sampled_people.extend(pb)
        all_sampled_vehicles.extend(vb)
        all_sampled_weapons.extend(wb)

    has_detected_people = len(all_sampled_people) > 0
    has_interacting_people = people_are_interacting(all_sampled_people)
    has_detected_vehicles = len(all_sampled_vehicles) > 0
    valid_sampled_weapons = [w for w in all_sampled_weapons if w.get("conf", 1.0) >= 0.30]
    has_detected_weapons = len(valid_sampled_weapons) > 0
    has_firearm = any(any(t in w.get("label", "").lower() for t in ("firearm", "gun", "pistol", "rifle")) for w in valid_sampled_weapons)
    has_blade = any(any(t in w.get("label", "").lower() for t in ("knife", "blade", "dagger", "sword")) for w in valid_sampled_weapons)

    # Apply Unified Multi-Modal Decision & Gating Engine on video-level summary
    fused_class, fused_conf, fused_reason = fuse_multimodal_crime_decision(
        label=final_class,
        confidence=final_confidence.item() if isinstance(final_confidence, torch.Tensor) else float(final_confidence),
        probabilities=mean_probabilities,
        people_boxes=all_sampled_people,
        weapon_boxes=all_sampled_weapons,
        vehicle_boxes=all_sampled_vehicles,
        classes=classes,
        crime_threshold=CRIME_THRESHOLD
    )

    final_class = fused_class
    # Cap crime confidence at 95% maximum — models should never claim 100% certainty
    capped_conf = min(fused_conf, 0.95) if fused_class != "Normal" else fused_conf
    final_confidence = torch.as_tensor(capped_conf, dtype=torch.float32)
    is_crime = (final_class != "Normal" and capped_conf >= CRIME_THRESHOLD)
    crime_probability = min(capped_conf, 0.95) if is_crime else 0.0

    if fused_reason:
        print(f"\n[Multi-Modal Fusion Decision] {fused_reason} -> {final_class} ({fused_conf*100:.1f}%)")

    # ========================================================
    # SAVE CRIME CLIPS ONLY
    # ========================================================

    evidence_dir = OUTPUT_DIR / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    evidence_files = extract_evidence_clips(
        frames,
        fps,
        results,
        output_dir=evidence_dir,
        crime_threshold=CRIME_THRESHOLD,
        min_consecutive=2,
        padding_seconds=2.0,
        video_stem=video_name,
        detector_engine=detector_engine,
        pixel_coordinates=pixel_coordinates
    )

    acq_note = ""
    if acquisition_mode != "original":
        acq_note = f" Adaptive acquisition ({acquisition_mode}) was used to enhance detection."

    weapon_note = ""
    if has_firearm:
        weapon_note = " Confirmed firearm detected."
    elif has_detected_weapons:
        weapon_note = " Confirmed weapon threat detected."

    report_description = (
        f"The video contains a {final_class.lower()} incident. "
        f"The model identified this as {final_class} with "
        f"{final_confidence.item() * 100:.2f}% confidence and a crime probability "
        f"of {crime_probability * 100:.2f}%.{acq_note}{weapon_note}"
        if is_crime
        else f"The video appears normal. The model classified the scene as {final_class} "
        f"with {final_confidence.item() * 100:.2f}% confidence and a crime probability of "
        f"{crime_probability * 100:.2f}%.{acq_note}{weapon_note}"
    )

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        str(OUTPUT_DIR),
        exist_ok=True
    )

    # ========================================================
    # CREATE VIDEO WRITER
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        raise RuntimeError(
            "Could not create output video."
        )

    # ========================================================
    # GENERATE ANNOTATED VIDEO WITH 2-STAGE PIPELINE
    # ========================================================

    print("\n======================================")
    print("CREATING ANNOTATED VIDEO (2-STAGE AI)")
    print("======================================")

    max_people = 0
    max_vehicles = 0
    max_weapons = 0
    all_activities = set()

    for i, frame in enumerate(frames):
        if progress_callback and (i % 5 == 0 or i == len(frames) - 1):
            pct = 0.55 + 0.35 * ((i + 1) / len(frames))
            progress_callback(pct, f"Generating 2-Stage AI HUD frame {i + 1}/{len(frames)}...")

        # ----------------------------------------------------
        # Pristine High-Resolution Base Frame
        # Keep 100% native resolution for crisp video quality
        # ----------------------------------------------------
        display_frame = frame.copy()
        pip_crop = None
        pip_title = None

        if acquisition_mode != "original":
            zoom_val = 2.0
            if "person_zoom_" in acquisition_mode:
                try:
                    after_pz = acquisition_mode.split("person_zoom_")[1]
                    zoom_val = float(after_pz.split("_")[0])
                except Exception:
                    zoom_val = 2.0
            elif acquisition_mode.startswith("global_zoom_"):
                try:
                    zoom_val = float(acquisition_mode.replace("global_zoom_", ""))
                except Exception:
                    zoom_val = 2.0

            # Target crop region for Picture-in-Picture (PiP) inset
            target_box = None
            if pixel_coordinates and len(pixel_coordinates) == 4:
                target_box = pixel_coordinates
            else:
                p_boxes_raw, _, _, _ = detector_engine.detect_and_track(frame)
                if p_boxes_raw:
                    target_box = p_boxes_raw[0]["box"]

            if target_box is not None:
                try:
                    crop = crop_zoomed_region(frame, target_box, zoom_factor=zoom_val)
                    if crop is not None and crop.size > 0:
                        pip_crop = crop
                        if "fsrcnn" in acquisition_mode:
                            pip_title = f"FSRCNN ROI ({zoom_val:.1f}x)"
                        else:
                            pip_title = f"AI ZOOM ({zoom_val:.1f}x)"
                except Exception:
                    pip_crop = None

        # ----------------------------------------------------
        # Stage 1: Detect People, Vehicles, Weapons & Activities
        # ----------------------------------------------------
        p_boxes, v_boxes, w_boxes, summary = detector_engine.detect_and_track(display_frame)
        max_people = max(max_people, summary["people_count"])
        max_vehicles = max(max_vehicles, summary["vehicle_count"])
        max_weapons = max(max_weapons, summary["weapon_count"])
        all_activities.update(summary["activities"])

        detector_data = (p_boxes, v_boxes, w_boxes, summary)

        # ----------------------------------------------------
        # Stage 2: Get Crime Prediction for this frame
        # When adaptive acquisition found a crime, override
        # per-frame window predictions so the video shows the
        # actual crime classification instead of "Normal"
        # ----------------------------------------------------
        if is_crime and acquisition_mode != "original":
            # Adaptive acquisition confirmed crime — use the final verdict
            label = final_class
            confidence = final_confidence.item() if hasattr(final_confidence, 'item') else float(final_confidence)
        elif is_crime and i in window_predictions:
            label, confidence = window_predictions[i]
        elif not is_crime:
            label = "Normal"
            confidence = final_confidence.item() if hasattr(final_confidence, 'item') else float(final_confidence)
        else:
            label = final_class
            confidence = final_confidence.item() if hasattr(final_confidence, 'item') else float(final_confidence)

        # ----------------------------------------------------
        # Draw 2-Stage prediction HUD with optional PiP Zoom
        # ----------------------------------------------------
        annotated_frame = draw_prediction(
            display_frame,
            label,
            confidence,
            CRIME_THRESHOLD,
            detector_data=detector_data,
            acquisition_mode=acquisition_mode,
            pixel_coordinates=pixel_coordinates,
            pip_crop=pip_crop,
            pip_title=pip_title
        )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------
        timestamp = i / fps if fps > 0 else 0
        cv2.putText(
            annotated_frame,
            f"Time: {timestamp:.2f}s",
            (20, height - 45 if summary else height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        writer.write(annotated_frame)

    writer.release()

    # ========================================================
    # WHOLE VIDEO ANALYSIS
    # ========================================================

    print("\n======================================")
    print("WHOLE VIDEO ANALYSIS")
    print("======================================")

    if is_crime:
        print("Crime: YES")
        print(f"Type: {final_class}")
        print(f"Confidence: {final_confidence.item() * 100:.2f}%")
        print(f"Crime Probability: {crime_probability * 100:.2f}%")
    else:
        print("Crime: NO")
        print("Type: Normal")
        print(f"Confidence: {mean_probabilities[normal_index].item() * 100:.2f}%")

    print(f"Telemetry -> People: {max_people}, Vehicles: {max_vehicles}, Weapons: {max_weapons}")
    print(f"Activities Detected: {list(all_activities)}")

    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================

    print("\nClass probabilities:")
    for i, class_name in enumerate(classes):
        print(f"{class_name:<15}: {mean_probabilities[i].item() * 100:.2f}%")

    print("\n======================================")
    print("CONVERTING OUTPUT VIDEOS TO H.264 FOR WEB PLAYBACK")
    print("======================================")

    if progress_callback:
        progress_callback(0.92, "Transcoding output video & evidence clips to H.264 format...")

    convert_to_h264(output_video)
    for ev_file in evidence_files:
        convert_to_h264(ev_file)

    if progress_callback:
        progress_callback(1.00, "Analysis complete!")

    print("\n======================================")
    print("ANNOTATED VIDEO SAVED")
    print(f"Saved to: {output_video}")
    print("======================================")

    window_audits = []
    for idx, res in enumerate(results):
        sec = res["start"] / fps if fps > 0 else 0.0
        mins = int(sec // 60)
        secs = int(sec % 60)
        ts = f"{mins:02d}:{secs:02d}"
        window_audits.append({
            "id": idx,
            "timestamp": ts,
            "seconds": round(sec, 2),
            "classification": res["label"],
            "confidence": round(res["confidence"] * 100, 1)
        })

    report_data = {
        "results": results,
        "crime": bool(is_crime),
        "type": str(final_class) if is_crime else "Normal",
        "confidence": float(final_confidence.item()) if hasattr(final_confidence, 'item') else float(final_confidence),
        "crime_probability": float(crime_probability.item()) if hasattr(crime_probability, 'item') else float(crime_probability),
        "output_video": str(Path(output_video).name),
        "crime_clips": [
            str(Path(path).name)
            for path in evidence_files
        ],
        "description": report_description,
        "people_count": int(max_people),
        "vehicle_count": int(max_vehicles),
        "weapon_count": int(max_weapons),
        "detected_activities": list(all_activities),
        "acquisition_mode": str(acquisition_mode),
        "pixel_coordinates": pixel_coordinates,
        "window_audits": window_audits
    }

    report_json_path = OUTPUT_DIR / f"{video_name}_report.json"
    try:
        with open(report_json_path, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"Saved surveillance audit report to: {report_json_path}")
    except Exception as err:
        print(f"Warning saving report JSON: {err}")

    return report_data


def generate_live_frames(video_path):
    """
    Generate live video frames for Flask HTTP streaming with proper FPS pacing and infinite seamless looping.
    Prefers reading pre-annotated output video if available for smooth real-time web playback.
    """
    vpath = Path(video_path)
    video_stem = vpath.stem
    output_video_path = OUTPUT_DIR / f"{video_stem}_result.mp4"

    # Use annotated video if available, otherwise fallback to source video
    target_stream_path = output_video_path if output_video_path.exists() else vpath

    cap = cv2.VideoCapture(str(target_stream_path))
    ret_test, _ = cap.read() if cap.isOpened() else (False, None)
    if not ret_test:
        cap.release()
        cap = cv2.VideoCapture(str(vpath))
    else:
        # Rewind to start after initial test read
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    frame_delay = 1.0 / fps

    while True:
        ret, frame = cap.read()
        if not ret:
            # Rewind to start for continuous live stream playback
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
        time.sleep(frame_delay)

    cap.release()


if __name__ == "__main__":

    final_result = process_video(
        VIDEO_PATH
    )

    print("\n======================================")
    print("FINAL RESULT")
    print("======================================")

    print(
        f"Crime: "
        f"{'YES' if final_result['crime'] else 'NO'}"
    )

    print(
        f"Type: "
        f"{final_result['type']}"
    )

    print(
        f"Confidence: "
        f"{final_result['confidence'] * 100:.2f}%"
    )

    print(
        f"Output: "
        f"{final_result['output_video']}"
    )

    print("======================================")

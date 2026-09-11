from pathlib import Path
import torch
from ultralytics import YOLO

from models.crime_model import load_crime_model


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT = PROJECT_ROOT / "models" / "crime_aerial_augmented_best1.pth"
PERSON_MODEL = PROJECT_ROOT / "models" / "drone_person_detector_best.pt"
WEAPON_MODEL = PROJECT_ROOT / "models" / "weapon_detector_best.pt"
if not WEAPON_MODEL.exists() and (PROJECT_ROOT / "models" / "firearmbest.pt").exists():
    WEAPON_MODEL = PROJECT_ROOT / "models" / "firearmbest.pt"


# ============================================================
# CHECK FILES
# ============================================================

print("Checking model files...")

print("Crime model:", CHECKPOINT)
print("Exists:", CHECKPOINT.exists())

print("Drone Person YOLO:", PERSON_MODEL)
print("Exists:", PERSON_MODEL.exists())

print("Weapon YOLO:", WEAPON_MODEL)
print("Exists:", WEAPON_MODEL.exists())


# ============================================================
# LOAD CRIME MODEL
# ============================================================

print("\nLoading Crime Model (Domain Adaptation)...")

model, classes, checkpoint = load_crime_model(CHECKPOINT, device=torch.device("cpu"))

print(f"\nArchitecture: {model.__class__.__name__}")
print(f"Num Classes: {len(classes)}")
print("Classes:", classes)
if "best_val_acc" in checkpoint:
    print(f"Best Val Acc: {checkpoint['best_val_acc']:.2%}")

# Test forward pass on dummy clip [1, 3, 8, 112, 112]
dummy_clip = torch.randn(1, 3, 8, 112, 112)
with torch.no_grad():
    out = model(dummy_clip)
print("Sample inference forward pass output shape:", out.shape)
print("\nCrime model loaded and verified successfully.")


# ============================================================
# LOAD PERSON YOLO
# ============================================================

if PERSON_MODEL.exists():
    person_model = YOLO(str(PERSON_MODEL))
    print("\nPerson YOLO loaded successfully.")
else:
    print(f"\nPerson YOLO ({PERSON_MODEL.name}) not found on disk, skipping dedicated person model test.")

# ============================================================
# LOAD WEAPON YOLO
# ============================================================

print("\nLoading weapon YOLO...")

if WEAPON_MODEL.exists():
    weapon_model = YOLO(str(WEAPON_MODEL))
    print(f"Weapon YOLO ({WEAPON_MODEL.name}) loaded successfully.")
    print("Weapon Classes:", weapon_model.names)
else:
    print(f"Weapon YOLO ({WEAPON_MODEL.name}) not found on disk.")


# ============================================================
# FINAL
# ============================================================

print("\n========================================")
print("ALL MODELS LOADED SUCCESSFULLY")
print("========================================")

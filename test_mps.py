from pathlib import Path
import torch
import torch.nn.functional as F

from models.crime_model import load_crime_model

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT = PROJECT_ROOT / "models" / "crime_aerial_augmented_best1.pth"

NUM_FRAMES = 8
IMAGE_SIZE = 112

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
# LOAD MODEL
# ============================================================
print("\nLoading checkpoint...")
model, classes, checkpoint = load_crime_model(CHECKPOINT, device=DEVICE)

print("Classes:")
print(classes)
print("Number of classes:", len(classes))
print("Architecture:", model.__class__.__name__)
print("Device:", DEVICE)

# ============================================================
# CREATE DUMMY VIDEO CLIP
# ============================================================
# Expected model input: [Batch, Channels, Frames, Height, Width]
print("\nCreating test video clip...")
clip = torch.randn(
    1,
    3,
    NUM_FRAMES,
    IMAGE_SIZE,
    IMAGE_SIZE,
    dtype=torch.float32
).to(DEVICE)
print("Input shape:", clip.shape)
print("Input device:", clip.device)

# ============================================================
# RUN INFERENCE
# ============================================================
print("\nRunning inference...")
with torch.no_grad():
    logits = model(clip)
    probabilities = F.softmax(logits, dim=1)
    confidence, prediction = torch.max(probabilities, dim=1)

# ============================================================
# RESULTS
# ============================================================
predicted_index = prediction.item()
predicted_class = classes[predicted_index]
confidence_value = confidence.item()

print("\n======================================")
print("DEVICE MODEL TEST")
print("======================================")
print("Input shape:", clip.shape)
print("Output shape:", logits.shape)
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence_value * 100:.2f}%\n")
print("All probabilities:")
for class_name, prob in zip(classes, probabilities[0].detach().cpu().tolist()):
    print(f"  {class_name:<15} {prob * 100:.2f}%")

# ============================================================
# FINAL STATUS
# ============================================================
if logits.shape[1] == len(classes):
    print("\n======================================")
    print("SUCCESS: Device inference passed cleanly.")
    print("======================================")
else:
    print("\nERROR: Output shape mismatch.")

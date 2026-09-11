import torch
import os

CHECKPOINT = "models/crime_aerial_augmented_best1.pth"
if not os.path.exists(CHECKPOINT):
    for alt in ["models/crime_r2plus1d_ucf_finetuned.pth", "models/domain_adaptation_model_aerial.pth"]:
        if os.path.exists(alt):
            CHECKPOINT = alt
            break

print("Checkpoint:", CHECKPOINT)
print("Exists:", os.path.exists(CHECKPOINT))

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

print("Checkpoint type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nCheckpoint keys:")
    print(checkpoint.keys())

    classes = checkpoint.get("class_names") or checkpoint.get("classes")
    if classes:
        print("\nClasses:")
        print(classes)

    if "model_state_dict" in checkpoint:
        print("\nNumber of parameters:", len(checkpoint["model_state_dict"]))

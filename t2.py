import torch
import os

CHECKPOINT = "models/crime_aerial_augmented_best1.pth"
if not os.path.exists(CHECKPOINT):
    for alt in ["models/crime_r2plus1d_ucf_finetuned.pth", "models/domain_adaptation_model_aerial.pth"]:
        if os.path.exists(alt):
            CHECKPOINT = alt
            break

print("Inspecting checkpoint:", CHECKPOINT)
checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

state_dict = checkpoint["model_state_dict"]

print("\nDOMAIN ADAPTATION MODEL STATE DICT:\n")

for key, value in state_dict.items():
    print(f"{key:60s} {tuple(value.shape)}")

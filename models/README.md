# Model Weights Directory

This directory stores the trained deep learning weights:

1. `crime_aerial_augmented_best1.pth`: Primary 3D Spatio-Temporal R(2+1)D model fine-tuned on aerial surveillance crime datasets.
2. `drone_person_detector_best.pt`: Aerial person detection YOLO model.
3. `weapon_detector_best.pt`: Specialized firearm/weapon detector YOLO model.
4. `yolov8s.pt` & `yolov8n.pt`: COCO object detectors and ByteTrack trackers.

> **Note on Large Checkpoint Files**:
> Checkpoints exceeding 100MB (`*.pth`) are excluded from GitHub git tracking via `.gitignore`. Keep your `.pth` weights in this directory for local inference or deploy them using cloud storage/release assets.

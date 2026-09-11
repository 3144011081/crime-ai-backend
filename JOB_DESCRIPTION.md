# 📄 Software Engineer Job Description (AI & Computer Vision)

**Department:** Software Development  
**Organization:** Neuronix Technologies  
**Project:** AI Aerial Surveillance & Crime Detection System from Drone (UAV) Video Feeds  
**Track:** Artificial Intelligence & Computer Vision Engineering  

---

## 📌 Executive Summary (Single Paragraph)

> "As a **Software Engineer / AI Developer** in the **Software Development Department**, my role involves designing, building, and deploying an end-to-end intelligent aerial surveillance system for drone footage. I engineered a multi-stage Python pipeline combining fine-tuned **YOLOv8** (people, vehicles, and weapons), **ByteTrack** multi-object tracking, and a **3D R(2+1)D CNN** for temporal crime classification. To eliminate false alarms, I developed a **domain-specific gating engine** with strict verification logic: personal physical crimes (violence, robbery) require confirmed human presence and physical interaction, while lethal threats like **Shooting are strictly verified by dedicated firearm detection** (suppressing shooting false alarms unless a physical gun is detected). I also integrated adaptive image enhancement and deep learning **FSRCNN super-resolution** to resolve low-resolution, high-altitude targets. Additionally, I built full-stack **Flask and Streamlit** operational dashboards, automated evidence clipping with **DOCX/JSON** forensic report generation, and optimized the pipeline for GPU acceleration on **Apple Silicon MPS** and **NVIDIA CUDA**."

---

## 🎯 Core Responsibilities in Software Development

### 1. Multi-Stage AI Inference Pipeline
- Engineered the core 4-stage video processing pipeline in Python (`video_inference1.py`).
- Integrated custom-trained YOLOv8 models (`models/drone_person_detector_best.pt`, `models/weapon_detector_best.pt`) with real-time multi-target tracking (`ByteTrack`).
- Built spatio-temporal video action recognition using R(2+1)D 3D ResNets for 7 action classes: *Normal, Violence, Robbery, Shooting, FireExplosion, Accident, Vandalism*.

### 2. Multi-Modal Decision & Logical Gating Engine
To address real-world surveillance challenges where standalone 3D CNNs produce false alarms:
- **No Person $\rightarrow$ No Personal Crime:** Suppresses personal crime classifications (Violence, Robbery, Shooting) to `Normal` if no humans are detected or interacting.
- **Firearm-Gated Shooting:** A shooting incident is strictly gated on the detection of a physical firearm (handgun, pistol, rifle). If the 3D CNN predicts "Shooting" without a firearm detected, the scene is routed to `Violence` (if physical fighting is observed) or suppressed to `Normal`.
- **Isolated Non-Human Incidents:** Permits vehicle `Accident` alerts only when motor vehicles are present, and flags `FireExplosion` based on high-confidence thermal/optical criteria.
- **Physical Contact & Snatch-and-Run:** Distinguishes normal walking from fighting via human bounding box proximity and velocity analysis.

### 3. Adaptive Preprocessing & Super-Resolution
- Built the `image_preprocessing.py` suite supporting dynamic modes (`aerial_drone`, `low_light_night`, `yolo_enhanced`).
- Implemented CLAHE, AGCWD gamma adjustment, and dark channel dehazing for high-altitude haze.
- Embedded FSRCNN 3x super-resolution (`super_resolution.py`) for low-resolution target enhancement.

### 4. Full-Stack Web Development & Forensic Tooling
- **Flask Web Platform (`app.py`):** Live MJPEG video streaming, file upload handling, and timeline event rendering.
- **Streamlit Interactive App (`streamlit_app.py`):** Real-time slider adjustments for confidence thresholds and visual comparison overlays.
- **Forensic Report Generator:** Automatic generation of audit-ready DOCX reports (`generate_docx_report.py`) and JSON event logs.

### 5. Hardware Acceleration & Performance Optimization
- Native cross-platform execution with **Apple Silicon MPS**, **NVIDIA CUDA**, and CPU fallback.
- Applied FP16 half-precision tensor calculations and frame downsampling for low-latency video streaming.

---

## 🛠️ Technical Competencies & Tool Stack

- **Languages:** Python 3.9+
- **Deep Learning / CV:** PyTorch, torchvision, Ultralytics YOLOv8, OpenCV, NumPy, ByteTrack
- **Action Recognition & Upscaling:** R(2+1)D 3D CNN, FSRCNN Super-Resolution
- **Web & Interface:** Flask, Streamlit, Jinja2, HTML5, CSS3, REST APIs, ngrok Tunneling
- **Hardware & Cloud Platforms:** Google Colab, Kaggle GPU Environments, Apple Silicon MPS, NVIDIA CUDA
- **Documentation & Reporting:** `python-docx`, JSON, Git

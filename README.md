# 🚁 Aerial Crime Detection Suite

An advanced AI-powered system for detecting suspicious activity in drone and aerial surveillance footage. The project combines multiple deep learning models to analyze video content in real-time:

- **Multi-stage YOLO object detection**: People, vehicles, and weapons detection with aerial-specific fine-tuning
- **Temporal crime classification**: R(2+1)D 3D CNN model for spatio-temporal action recognition
- **Adaptive preprocessing**: Specialized image enhancement for aerial, low-light, and distant scenes
- **Super resolution enhancement**: FSRCNN-based upscaling for low-quality footage
- **Intelligent evidence extraction**: Automatic detection and clipping of suspicious events
- **Comprehensive reporting**: JSON and DOCX report generation with detailed analysis
- **Dual web interfaces**: Flask and Streamlit dashboards for local analysis and visualization

The system is designed for comprehensive surveillance analysis in crime monitoring scenarios including violence, robbery, shooting, accidents, vandalism, and fire-related incidents.

---

## ✅ Key Features

- **Video Input Handling**: Process surveillance or drone videos from local repository or uploaded files
- **Multi-Stage Detection**: Frame-by-frame detection of people, vehicles, and weapons using fine-tuned YOLO models
- **Adaptive Preprocessing**: Multiple preprocessing modes optimized for aerial, low-light, and distant scenes
- **Temporal Analysis**: Spatio-temporal crime classification using R(2+1)D neural network
- **Super Resolution**: Optional FSRCNN-based image upscaling to improve detection quality on low-resolution footage
- **Adaptive Acquisition**: Multi-scale zoom and cropping for improving detection in distant or low-confidence regions
- **Evidence Extraction**: Automatic detection and extraction of suspicious activity clips
- **Report Generation**: Comprehensive JSON and DOCX reports with detection metrics, confidence scores, and activity summaries
- **Dual Dashboards**: Flask and Streamlit web interfaces for interactive analysis and visualization
- **GPU Acceleration**: Support for MPS (Apple Silicon), CUDA (NVIDIA), and CPU fallback

---

## 🧠 AI Model Pipeline

The repository implements a comprehensive multi-stage AI workflow optimized for aerial surveillance analysis:

### Stage 1: Preprocessing & Enhancement
- **Scene Analysis**: Automatic detection of camera type (aerial/drone) and lighting conditions
- **Adaptive Preprocessing**: Multiple modes to enhance visibility
  - `aerial_drone` - Contrast enhancement and noise reduction for high-altitude footage
  - `low_light_night` - Histogram equalization and brightness adjustment
  - `yolo_enhanced` - Specialized enhancement for object detection
  - `auto` - Automatic mode selection based on scene characteristics
  - `none` - No preprocessing
- **Super Resolution**: Optional FSRCNN upscaling (`models/FSRCNN_x3.pb`) to improve low-resolution footage quality

### Stage 2: Object Detection
- **People Detector**: `models/drone_person_detector_best.pt` (custom YOLO trained on aerial/drone footage)
- **Weapon Detector**: `models/weapon_detector_best.pt` (specialized firearm and blade detection)
- **General Detector**: `yolov8s.pt` with ByteTrack for persistent multi-object tracking across frames
- **Adaptive Acquisition**: Multi-scale zoom and cropping to improve detection in distant regions

### Stage 3: Temporal Crime Classification
- **Crime Model**: `models/crime_aerial_augmented_best1.pth` (Augmented Aerial 3D R(2+1)D action recognition network)
- **Fallback Model**: `models/crime_r2plus1d_ucf_finetuned.pth`
- **Detected Classes**: 
  - Normal
  - Violence
  - Robbery
  - Shooting
  - FireExplosion
  - Accident
  - Vandalism

### Stage 4: Multi-Modal Decision & Logical Gating Engine
To prevent action-recognition false alarms from camera motion or optical flow noise, the system runs an intelligent rule-based arbitration engine (`fuse_multimodal_crime_decision`):
- **Human Presence Requirement:** Personal crimes (*Violence, Robbery, Shooting*) require confirmed person presence and physical interaction. If no humans or weapons are present, the scene strictly defaults to `Normal`.
- **Firearm-Gated Shooting:** The model's *Shooting* predictions are strictly gated on detecting an actual firearm (gun, pistol, rifle). If *Shooting* is predicted without a weapon, the alert is suppressed to `Violence` (if fighting) or `Normal` (92% confidence).
- **Physical Contact & Snatch-and-Run:** Distinguishes normal walking from *Violence* based on bounding-box proximity and intersection, and detects *Robbery* when proximity is followed by fleeing/running velocity.
- **Independent Non-Human Incidents:** Isolates non-human incidents—validating *Accident* only when vehicles are detected and *FireExplosion* on verified thermal/high-confidence triggers.

### Stage 5: Evidence & Report Generation
- **Evidence Extraction**: Automatic clipping and saving of suspicious activity clips
- **JSON Reports**: Detailed frame-by-frame analysis with confidence scores
- **DOCX Reports**: Professional formatted reports with statistics and visual summaries
- **Bias Evaluation**: Tools for assessing model performance across different scenarios

---

## 📁 Repository Structure

```text
ML_Project/
├── 🚀 Monoserver & Launchers
│   ├── run.py                         # Unified CLI Launcher (python run.py)
│   ├── app.py                         # Root entrypoint (delegates to backend/monoserver.py)
│   ├── package.json                   # Root package scripts (npm run build, npm start)
│   └── requirements.txt               # Backend Python dependencies (flask, flask-cors, torch, etc.)
│
├── ⚡ Frontend (React 18 + Vite SPA)
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── api/client.ts          # Typed REST API & streaming client
│   │   │   ├── components/            # Sidebar, Navbar, VideoUploader, VideoPlayer,
│   │   │   │                          # AlertBanner, StatsGrid, IncidentChart, AuditTable
│   │   │   ├── pages/                 # DashboardPage, LiveFeedPage, ReportsPage, SettingsPage
│   │   │   ├── types/index.ts         # TypeScript interfaces (ReportData, WindowAudit, etc.)
│   │   │   ├── App.tsx                # Main SPA application layout & tabs
│   │   │   ├── main.tsx               # React DOM root
│   │   │   └── index.css              # Tailwind CSS & dark surveillance theme
│   │   ├── vite.config.ts             # Vite configuration with /api proxy to :5005
│   │   ├── package.json               # Frontend dependencies (React, Tailwind, Lucide)
│   │   └── dist/                      # Compiled production SPA bundle served by Monoserver
│
├── 🐍 Backend & REST API (Python Monoserver)
│   ├── backend/
│   │   ├── config.py                  # Environment paths & runtime setup
│   │   ├── monoserver.py              # Unified server hosting API + React SPA
│   │   ├── api/
│   │   │   ├── __init__.py            # Flask API blueprint registration (/api/*)
│   │   │   ├── videos.py              # Video listing, upload, and media streaming
│   │   │   ├── inference.py           # 2-Stage AI surveillance trigger & parameter tuning
│   │   │   ├── live.py                # Real-time MJPEG live detection stream
│   │   │   ├── reports.py             # JSON reports retrieval & CSV exports
│   │   │   └── settings.py            # Model parameters, thresholds, and accelerator telemetry
│   │   └── core/                      # Package initialization for ML modules
│
├── 🤖 Core AI Pipeline
│   ├── video_inference1.py            # Main 4-stage detection & classification pipeline
│   ├── adaptive_acquisition.py        # Adaptive zoom and multi-scale processing logic
│   ├── image_preprocessing.py         # Drone and scene preprocessing utilities
│   ├── super_resolution.py            # FSRCNN super resolution enhancement
│   ├── aerial_augmentation_solution.py # Aerial-specific data augmentation
│   └── bias_evaluation.py             # Model bias and performance evaluation
│
├── 📦 Pre-trained Models
│   ├── models/
│   │   ├── crime_model.py             # R(2+1)D crime classification model definition
│   │   ├── crime_aerial_augmented_best1.pth # Primary crime classifier checkpoint
│   │   ├── crime_r2plus1d_ucf_finetuned.pth # Fallback crime classifier
│   │   ├── drone_person_detector_best.pt    # Custom YOLO for aerial person detection
│   │   ├── weapon_detector_best.pt          # YOLO for weapon detection
│   │   └── FSRCNN_x3.pb              # Super resolution model (3x upscaling)
│   ├── yolov8n.pt                     # YOLOv8 Nano general detector
│   └── yolov8s.pt                     # YOLOv8 Small general detector
│
├── 📊 Project Data & Outputs
│   ├── videos/                        # Sample surveillance videos
│   └── outputs/                       # Analysis results, reports, evidence clips, and uploads
│       ├── evidence/                  # Extracted suspicious activity clips
│       ├── uploads/                   # User-uploaded video files
│       └── *.json                     # Video analysis reports
```

---

## ⚙️ Environment Setup

### Prerequisites
- Python 3.8+
- FFmpeg (for video processing)
- GPU support (optional, recommended for faster inference)

### 1. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies** include:
- Flask (web framework)
- Streamlit (interactive dashboard)
- PyTorch & TorchVision (deep learning)
- Ultralytics YOLO (object detection)
- OpenCV (video processing)
- NumPy, Pandas, Pillow (data processing)
- python-docx (report generation)

### 3. Install FFmpeg

FFmpeg is required for H.264 video conversion and browser-compatible playback:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (if using Chocolatey)
choco install ffmpeg
```

Verify installation:
```bash
ffmpeg -version
```

### 4. Download/Verify Model Files

Ensure all model files are present in the `models/` directory:
- `crime_aerial_augmented_best1.pth`
- `drone_person_detector_best.pt`
- `weapon_detector_best.pt`
- `FSRCNN_x3.pb`

Additional YOLO models (auto-downloaded if missing):
- `yolov8n.pt`
- `yolov8s.pt`

---

## ▶️ Running the Application

### Flask Web Dashboard

Start the Flask web server:

```bash
python app.py
```

Then open your browser to:

```
http://127.0.0.1:5005
```

**Features:**
- Upload or select videos from the `videos/` folder
- Configure detection parameters
- View real-time processing results
- Access generated reports
- Download evidence clips and annotated videos

### Streamlit Interactive Dashboard

Start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

Then open your browser to:

```
http://localhost:8501
```

**Features:**
- Interactive parameter adjustment with real-time preview
- Confidence threshold configuration
- Preprocessing mode selection
- Model checkpoint selection
- Detailed detection visualization
- Evidence extraction interface
- Report viewing and download

---

## 🧪 Testing & Evaluation

### Test Model Inference

Test the crime classification model on a single video:

```bash
python test_model.py --video path/to/video.mp4 --checkpoint models/crime_aerial_augmented_best1.pth
```

### Test YOLO Detection

Verify object detection pipeline:

```bash
python test_yolo.py --video path/to/video.mp4 --confidence 0.5
```

### Test Preprocessing

Visualize preprocessing effects on sample frames:

```bash
python test_preprocessing.py --image path/to/image.jpg --mode aerial_drone
python visualize_preprocessing.py
```

### Test GPU Acceleration (MPS)

Verify Apple Silicon GPU support:

```bash
python test_mps.py
```

### Batch Evaluation

Evaluate model performance on a dataset of videos:

```bash
python evaluate_model_videos.py --video_dir videos/ --output outputs/model_evaluation/
```

### Bias Evaluation

Assess model performance across different scenarios:

```bash
python bias_evaluation.py --video_dir videos/ --output outputs/
```

### Generate DOCX Reports

Create professional formatted reports from analysis results:

```bash
python generate_docx_report.py --json_report outputs/video_name_report.json --output outputs/report.docx
```

---

## 📖 Workflow Guide

### Basic Video Analysis Workflow

1. **Start the Application**
   - Choose Flask (`python app.py`) or Streamlit (`streamlit run streamlit_app.py`)
   - Open the web interface in your browser

2. **Select or Upload Video**
   - Choose a sample video from `videos/` folder, or
   - Upload your own surveillance/drone footage

3. **Configure Analysis Parameters**
   - **Preprocessing Mode**: Select preprocessing strategy
     - `auto` (recommended): Automatic mode selection
     - `aerial_drone`: For high-altitude drone footage
     - `low_light_night`: For dark/night scenes
     - `yolo_enhanced`: Optimized for object detection
   - **Confidence Threshold**: Set alert sensitivity (10-90%)
   - **Crime Model**: Select checkpoint (primary or fallback)
   - **Enable Features**: Optional super resolution, adaptive acquisition

4. **Start Analysis**
   - Click "Analyze" or "Run Detection"
   - Monitor progress in real-time

5. **Review Results**
   - **Annotated Video**: Output with bounding boxes and labels
   - **Evidence Clips**: Automatically extracted suspicious segments
   - **JSON Report**: Detailed frame-by-frame analysis
   - **Statistics**: People count, weapon detections, crime classifications

6. **Generate Reports**
   - Export as DOCX professional report
   - Download evidence clips
   - View visualizations and statistics

### Advanced Features

**Super Resolution Enhancement**: Enable for improved detection quality on low-resolution footage
- Uses FSRCNN 3x upscaling
- Recommended for aerial footage at high altitudes

**Adaptive Acquisition**: Automatic zoom on low-confidence regions
- Improves detection in distant or partially visible subjects
- Recommended for wide-angle drone shots

**Batch Processing**: Evaluate on multiple videos
- Use `evaluate_model_videos.py` for systematic analysis
- Generate comparison reports across dataset

---

## 📦 Output Artifacts

After video analysis, the following files are generated in the `outputs/` directory:

### Annotated Videos
- **Format**: MP4 with H.264 codec (browser-compatible)
- **Contents**: Original video with bounding boxes, confidence scores, and labels
- **Location**: `outputs/<video_name>_output.mp4`

### Evidence Clips
- **Format**: MP4 video segments
- **Contents**: Extracted suspicious activity clips with detected crimes
- **Location**: `outputs/evidence/<crime_type>_<timestamp>.mp4`

### JSON Reports
- **File**: `outputs/<video_name>_report.json`
- **Contents**: 
  - Frame-by-frame detection results
  - Object counts (people, weapons, vehicles)
  - Crime classifications with confidence scores
  - Timeline of detected suspicious events
  - Preprocessing mode and parameters used

### DOCX Reports
- **File**: `outputs/<video_name>_report.docx` (generated on-demand)
- **Contents**:
  - Executive summary
  - Detection statistics and charts
  - Evidence clip timestamps
  - Detailed findings per detected crime class
  - Model performance metrics

### Additional Outputs
- **Uploaded Videos**: `outputs/uploads/<filename>`
- **Preprocessing Comparisons**: `outputs/preprocessing_comparisons/`
- **Model Evaluation Results**: `outputs/model_evaluation/all_videos_evaluation_report.json`

### Report Structure Example

```json
{
  "video_name": "surveillance_footage.mp4",
  "processing_date": "2024-01-15",
  "total_frames": 1200,
  "fps": 30,
  "duration_seconds": 40,
  "preprocessing_mode": "aerial_drone",
  "detections": [
    {
      "frame": 150,
      "objects": {
        "people": 3,
        "weapons": 1,
        "vehicles": 0
      },
      "crime_classification": {
        "label": "Shooting",
        "confidence": 0.87,
        "timestamp": "00:05:00"
      }
    },
    ...
  ],
  "evidence_clips": [
    {
      "crime_type": "Shooting",
      "start_frame": 145,
      "end_frame": 200,
      "confidence": 0.87,
      "file": "evidence/Shooting_145_200.mp4"
    }
  ],
  "summary": {
    "total_crime_detections": 2,
    "people_count": 5,
    "weapons_count": 1,
    "crime_types": ["Violence", "Shooting"]
  }
}
```

---

## 🔧 Configuration & Customization

### Model Configuration

Modify model paths and parameters in the dashboard:

- **Crime Model Checkpoint**: Change the R(2+1)D checkpoint
- **Confidence Threshold**: Adjust detection sensitivity (10-90%)
- **Preprocessing Mode**: Select scene-specific preprocessing
- **Super Resolution**: Enable/disable FSRCNN upscaling
- **Adaptive Acquisition**: Enable/disable multi-scale zoom

### Hardware Acceleration

The system automatically detects and uses available hardware:

```
Apple Silicon: MPS (Metal Performance Shaders) - Fastest
NVIDIA GPU: CUDA - Fast
CPU Fallback: Standard PyTorch CPU inference - Slower
```

To force CPU mode:
```python
import torch
torch.device("cpu")
```

### Environment Variables

Optional configuration:

```bash
# Disable GPU acceleration
CUDA_VISIBLE_DEVICES=-1

# Set YOLO cache directory
YOLOv8_CACHE=/custom/path

# Set temporary file location
TEMP=/custom/temp/path
```

---

## 📝 Notes

### Project Design
- Built for **local analysis workflows**, not production cloud deployment
- Models are pre-trained and optimized for aerial/drone surveillance scenarios
- Supports both real-time streaming and batch video analysis

### Requirements
- All model checkpoint files must be present in `models/` directory
- FFmpeg must be installed for video processing
- Minimum 8GB RAM recommended (16GB+ recommended for GPU)
- NVIDIA GPU (optional): Requires CUDA 11.8+
- Apple Silicon (M1/M2/M3): Recommended for fastest inference

### Performance Considerations
- First run downloads YOLO models (300MB+ download)
- Analysis time depends on video length and resolution
- GPU acceleration can provide 5-10x speedup vs CPU
- Super resolution adds ~20-30% processing time but improves detection accuracy

### Troubleshooting

**Issue**: Model files not found
- **Solution**: Ensure all `.pth` and `.pt` files are in `models/` directory

**Issue**: Out of memory errors
- **Solution**: Reduce batch size or disable super resolution

**Issue**: Slow processing
- **Solution**: Enable GPU acceleration or reduce video resolution

**Issue**: Poor detection quality
- **Solution**: Try `aerial_drone` preprocessing, enable super resolution, or adjust confidence threshold

---

## 🔍 Architecture Overview

### Pipeline Flow

```
Video Input
    ↓
Frame Extraction (30+ fps handling)
    ↓
Preprocessing (Aerial/Low-light enhancement)
    ↓
Super Resolution (Optional FSRCNN upscaling)
    ↓
Parallel Detection
  ├─ People Detection (Custom YOLO)
  ├─ Weapon Detection (Specialized YOLO)
  └─ General Detection (YOLOv8 Fallback)
    ↓
Adaptive Acquisition (Optional multi-scale zoom)
    ↓
Temporal Crime Classification (R(2+1)D)
    ↓
Evidence Extraction (Suspicious clip detection)
    ↓
Report Generation (JSON & DOCX)
    ↓
Visualization & Web Interface (Flask/Streamlit)
```

### Model Sizes
- **Crime Classifier**: ~150MB
- **Person Detector**: ~80MB
- **Weapon Detector**: ~80MB
- **YOLOv8 Models**: ~30-100MB each
- **Super Resolution**: ~10MB

Total: ~350-400MB of model files

---

## 🚀 Development & Extension

### Current Capabilities
- Multi-stage AI pipeline with 3-4 models
- Adaptive preprocessing for various scenes
- Super resolution enhancement
- Evidence-based report generation
- Dual web interfaces (Flask + Streamlit)
- Batch evaluation and bias testing

### Future Enhancement Areas
- **Model Improvements**: Retrain on larger/diverse datasets, improve accuracy
- **New Crime Categories**: Add new classification categories
- **Advanced Analytics**: Real-time dashboards, historical trend analysis
- **Cloud Deployment**: Containerization (Docker), server deployment, API scaling
- **Performance**: Model quantization, edge deployment support
- **Integration**: RTMP stream support, CCTV system integration
- **Multi-GPU**: Distributed processing for large-scale deployments

### Extending the Project

**Add Custom Crime Categories**:
1. Prepare training dataset
2. Fine-tune `models/crime_model.py`
3. Update class labels in `video_inference1.py`
4. Test with `evaluate_model_videos.py`

**Improve Detection**:
1. Collect more training data
2. Fine-tune YOLO models for your use case
3. Augment with `aerial_augmentation_solution.py`
4. Evaluate with `bias_evaluation.py`

**Customize Web Interface**:
1. Modify Flask templates in `templates/`
2. Update styling in `static/style.css`
3. Add new Streamlit components in `streamlit_app.py`

---

## 📖 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch 3D Vision](https://pytorch.org/vision/stable/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 📄 License & Attribution

This project uses pre-trained models and follows respective licensing terms for:
- Ultralytics YOLO (AGPL-3.0)
- PyTorch (BSD)
- OpenCV (Apache 2.0)

---

## 👥 Support & Feedback

For issues, feature requests, or contributions:
- Review existing test files for usage examples
- Check output reports for diagnostic information
- Use `bias_evaluation.py` to test model performance
- Verify preprocessing effects with `visualize_preprocessing.py`


import os
import sys
from pathlib import Path

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core directories
VIDEOS_DIR = PROJECT_ROOT / "videos"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
UPLOADS_DIR = OUTPUT_DIR / "uploads"
EVIDENCE_DIR = OUTPUT_DIR / "evidence"
MODELS_DIR = PROJECT_ROOT / "models"
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"

# Ensure runtime directories exist
for directory in [VIDEOS_DIR, OUTPUT_DIR, UPLOADS_DIR, EVIDENCE_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Add project root and backend to sys.path so core ML modules can be imported directly
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "backend"), str(PROJECT_ROOT / "backend" / "core")]:
    if p not in sys.path:
        sys.path.insert(0, p)

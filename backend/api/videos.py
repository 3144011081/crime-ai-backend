import os
from pathlib import Path
from flask import request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from backend.config import VIDEOS_DIR, UPLOADS_DIR, OUTPUT_DIR, EVIDENCE_DIR
from . import api_bp

try:
    import video_inference1
except ImportError:
    video_inference1 = None

def get_all_videos():
    """List videos in videos/ and outputs/uploads/."""
    extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    samples = []
    for ext in extensions:
        samples.extend(VIDEOS_DIR.glob(ext))
    
    uploads = []
    for ext in extensions:
        uploads.extend(UPLOADS_DIR.glob(ext))
    
    samples_list = sorted([v.name for v in samples if not v.name.startswith(".")])
    uploads_list = sorted([v.name for v in uploads if not v.name.startswith(".")])
    all_unique = sorted(list(set(samples_list + uploads_list)))
    return {
        "all": all_unique,
        "samples": samples_list,
        "uploads": uploads_list
    }

@api_bp.route("/videos", methods=["GET"])
def list_videos():
    """Return available sample and uploaded videos."""
    return jsonify(get_all_videos())

@api_bp.route("/upload", methods=["POST"])
def upload_video_file():
    """Handle video file upload and conversion."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided in request."}), 400
    
    file = request.files["video"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    filename = secure_filename(file.filename)
    if not filename:
        filename = "uploaded_video.mp4"

    target_path = UPLOADS_DIR / filename
    file.save(str(target_path))

    # Convert to browser-compatible H.264 if needed
    if video_inference1 and hasattr(video_inference1, "convert_to_h264"):
        try:
            video_inference1.convert_to_h264(target_path)
        except Exception as e:
            print(f"[API] H.264 conversion note: {e}")

    file_size_mb = round(target_path.stat().st_size / (1024 * 1024), 2)

    return jsonify({
        "success": True,
        "filename": filename,
        "size_mb": file_size_mb,
        "url": f"/api/media/video/{filename}"
    })

@api_bp.route("/media/video/<path:filename>", methods=["GET"])
def serve_video(filename):
    """Serve source video from uploads or videos directory."""
    if (UPLOADS_DIR / filename).exists():
        return send_from_directory(UPLOADS_DIR, filename)
    elif (VIDEOS_DIR / filename).exists():
        return send_from_directory(VIDEOS_DIR, filename)
    elif (OUTPUT_DIR / filename).exists():
        return send_from_directory(OUTPUT_DIR, filename)
    return jsonify({"error": f"Video '{filename}' not found."}), 404

@api_bp.route("/media/output/<path:filename>", methods=["GET"])
def serve_output_video(filename):
    """Serve processed/annotated video from outputs directory."""
    if (OUTPUT_DIR / filename).exists():
        return send_from_directory(OUTPUT_DIR, filename)
    elif (EVIDENCE_DIR / filename).exists():
        return send_from_directory(EVIDENCE_DIR, filename)
    return jsonify({"error": f"Output file '{filename}' not found."}), 404

@api_bp.route("/media/evidence/<path:filename>", methods=["GET"])
def serve_evidence_video(filename):
    """Serve evidence clip from evidence directory."""
    if (EVIDENCE_DIR / filename).exists():
        return send_from_directory(EVIDENCE_DIR, filename)
    elif (OUTPUT_DIR / filename).exists():
        return send_from_directory(OUTPUT_DIR, filename)
    return jsonify({"error": f"Evidence clip '{filename}' not found."}), 404

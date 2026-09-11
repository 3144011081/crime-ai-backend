import os
from pathlib import Path
from flask import request, jsonify
from backend.config import VIDEOS_DIR, UPLOADS_DIR, OUTPUT_DIR
from . import api_bp

try:
    import video_inference1
except ImportError:
    video_inference1 = None

@api_bp.route("/inference/process", methods=["POST"])
def process_video_endpoint():
    """Trigger 2-Stage AI crime detection & spatio-temporal video analysis."""
    if not video_inference1:
        return jsonify({"error": "Video inference engine is not available."}), 500

    data = request.get_json(silent=True) or request.form.to_dict()
    filename = data.get("filename")
    preprocess_mode = data.get("preprocess_mode", "auto")
    conf_threshold = data.get("confidence_threshold")
    stride = data.get("stride")

    if not filename:
        return jsonify({"error": "Filename is required."}), 400

    # Locate source video
    target_path = None
    if (UPLOADS_DIR / filename).exists():
        target_path = UPLOADS_DIR / filename
    elif (VIDEOS_DIR / filename).exists():
        target_path = VIDEOS_DIR / filename
    elif (OUTPUT_DIR / filename).exists():
        target_path = OUTPUT_DIR / filename
    else:
        return jsonify({"error": f"Video '{filename}' not found on server."}), 404

    # Apply preprocessing mode if supported
    if hasattr(video_inference1, "drone_preprocessor") and video_inference1.drone_preprocessor is not None:
        video_inference1.drone_preprocessor.mode = preprocess_mode

    # Configure custom parameters if provided
    if conf_threshold is not None:
        try:
            val = float(conf_threshold)
            if val > 1.0:
                val /= 100.0
            if hasattr(video_inference1, "CRIME_ALERT_CONF_THRESHOLD"):
                video_inference1.CRIME_ALERT_CONF_THRESHOLD = val
        except Exception:
            pass

    if stride is not None:
        try:
            val = int(stride)
            if hasattr(video_inference1, "WINDOW_STRIDE"):
                video_inference1.WINDOW_STRIDE = val
        except Exception:
            pass

    print(f"\n[Monoserver] Starting 2-Stage AI Video Analysis on: {target_path} (Mode: {preprocess_mode})")

    try:
        results = video_inference1.process_video(target_path)
        return jsonify({
            "success": True,
            "filename": filename,
            "report": results
        })
    except Exception as err:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(err)
        }), 500

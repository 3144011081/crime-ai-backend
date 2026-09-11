import torch
from flask import jsonify, request
from . import api_bp

try:
    import video_inference1
except ImportError:
    video_inference1 = None

@api_bp.route("/settings", methods=["GET"])
def get_settings():
    """Return current model configuration, runtime hardware, and thresholds."""
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "Apple Silicon MPS (Metal Performance Shaders)"

    conf_thresh = 35
    stride = 8
    model_path = "/models/crime_aerial_augmented_best1.pth"
    classes = [
        "Abuse", "Arrest", "Arson", "Assault", "Burglary",
        "Explosion", "Fighting", "RoadAccidents", "Robbery",
        "Shooting", "Shoplifting", "Stealing", "Vandalism", "Normal"
    ]

    if video_inference1:
        if hasattr(video_inference1, "CRIME_ALERT_CONF_THRESHOLD"):
            conf_thresh = int(video_inference1.CRIME_ALERT_CONF_THRESHOLD * 100)
        if hasattr(video_inference1, "WINDOW_STRIDE"):
            stride = video_inference1.WINDOW_STRIDE
        if hasattr(video_inference1, "CLASSES"):
            classes = list(video_inference1.CLASSES)

    return jsonify({
        "device": device_name,
        "checkpoint": model_path,
        "confidence_threshold": conf_thresh,
        "stride": stride,
        "classes": classes,
        "preprocess_modes": ["auto", "none", "histogram", "superres"]
    })

@api_bp.route("/settings", methods=["POST"])
def update_settings():
    """Update active inference parameters."""
    data = request.get_json(silent=True) or {}
    updated = {}

    if "confidence_threshold" in data and video_inference1:
        try:
            val = float(data["confidence_threshold"])
            if val > 1.0:
                val /= 100.0
            video_inference1.CRIME_ALERT_CONF_THRESHOLD = val
            updated["confidence_threshold"] = int(val * 100)
        except Exception:
            pass

    if "stride" in data and video_inference1:
        try:
            val = int(data["stride"])
            video_inference1.WINDOW_STRIDE = val
            updated["stride"] = val
        except Exception:
            pass

    return jsonify({"success": True, "updated": updated})

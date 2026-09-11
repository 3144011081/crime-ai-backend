import os
import sys
import socket
from pathlib import Path
from flask import Flask, send_from_directory, jsonify, Response, request
try:
    from flask_cors import CORS
except (ImportError, TypeError):
    CORS = None

# Initialize configuration and sys.path
from backend.config import (
    PROJECT_ROOT,
    VIDEOS_DIR,
    OUTPUT_DIR,
    UPLOADS_DIR,
    EVIDENCE_DIR,
    FRONTEND_DIST_DIR,
)
from backend.api import api_bp

def create_app():
    """Application factory for the Monoserver."""
    app = Flask(__name__, static_folder=None)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "crime_aerial_monoserver_secret_2026")

    # Enable Cross-Origin Resource Sharing (allows Vercel frontend to query backend API)
    cors_env = os.environ.get("CORS_ORIGINS", "*")
    cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()] if "," in cors_env else cors_env
    if CORS:
        CORS(app, resources={r"/*": {"origins": cors_origins}}, supports_credentials=True)
    else:
        @app.before_request
        def handle_preflight():
            if request.method == "OPTIONS":
                response = Response()
                origin = request.headers.get("Origin", "*")
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
                response.headers["Access-Control-Allow-Credentials"] = "true"
                return response

        @app.after_request
        def add_cors_headers(response):
            origin = request.headers.get("Origin", "*")
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            return response

    @app.route("/health")
    def health_check():
        return jsonify({"status": "ok", "service": "crime-ai-backend"}), 200

    # Register REST API blueprint
    app.register_blueprint(api_bp)

    # ---------------------------------------------------------
    # Legacy Backward-Compatibility Routes
    # ---------------------------------------------------------
    @app.route("/outputs/<path:filename>")
    @app.route("/output_file/<path:filename>")
    def legacy_output(filename):
        if (EVIDENCE_DIR / filename).exists():
            return send_from_directory(EVIDENCE_DIR, filename)
        return send_from_directory(OUTPUT_DIR, filename)

    @app.route("/evidence/<path:filename>")
    def legacy_evidence(filename):
        return send_from_directory(EVIDENCE_DIR, filename)

    @app.route("/videos/<path:filename>")
    @app.route("/video_file/<path:filename>")
    def legacy_video(filename):
        if (UPLOADS_DIR / filename).exists():
            return send_from_directory(UPLOADS_DIR, filename)
        return send_from_directory(VIDEOS_DIR, filename)

    @app.route("/live_video/<path:filename>")
    def legacy_live(filename):
        try:
            import video_inference1
            video_path = UPLOADS_DIR / filename
            if not video_path.exists():
                video_path = VIDEOS_DIR / filename
            if not video_path.exists():
                video_path = VIDEOS_DIR / "hello.mp4"
            return Response(
                video_inference1.generate_live_frames(video_path),
                mimetype="multipart/x-mixed-replace; boundary=frame"
            )
        except Exception as e:
            return Response(f"Live video error: {e}", status=500)

    # ---------------------------------------------------------
    # Frontend Static & SPA Routing (Monoserver Core)
    # ---------------------------------------------------------
    @app.route("/assets/<path:path>")
    def serve_assets(path):
        assets_dir = FRONTEND_DIST_DIR / "assets"
        if assets_dir.exists() and (assets_dir / path).exists():
            return send_from_directory(assets_dir, path)
        return jsonify({"error": f"Asset '{path}' not found."}), 404

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_spa(path):
        # Never intercept API calls
        if path.startswith("api/"):
            return jsonify({"error": "Endpoint not found"}), 404

        # Serve existing static files in dist root (favicon, icons, etc.)
        target_file = FRONTEND_DIST_DIR / path
        if path and target_file.is_file():
            return send_from_directory(FRONTEND_DIST_DIR, path)

        # Serve React SPA index.html
        index_file = FRONTEND_DIST_DIR / "index.html"
        if index_file.exists():
            return send_from_directory(FRONTEND_DIST_DIR, "index.html")

        # Fallback if frontend has not been compiled yet
        return (
            "<html><head><title>Aerial CrimeAI Monoserver</title>"
            "<style>body{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#f8fafc;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;text-align:center;}"
            ".box{max-width:560px;padding:32px;background:#1e293b;border-radius:12px;border:1px solid #334155;}"
            "h1{color:#ef4444;margin-top:0;}code{background:#0f172a;padding:4px 8px;border-radius:6px;color:#38bdf8;font-size:14px;}"
            "a{color:#38bdf8;text-decoration:none;}</style></head>"
            "<body><div class='box'><h1>🚁 CrimeAI Monoserver Running</h1>"
            "<p>API is active at <code>/api/videos</code>, <code>/api/reports</code>, etc.</p>"
            "<p>Frontend distribution bundle not built yet.</p>"
            "<p>Run: <code>cd frontend && npm run build</code> to compile the React + Vite frontend.</p>"
            "</div></body></html>"
        ), 200

    return app

def get_available_port(start_port=5005, max_attempts=20):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return start_port

def start_monoserver(host="0.0.0.0", port=None, debug=False):
    app = create_app()
    if port is None:
        port = int(os.environ.get("PORT", get_available_port(5005)))

    print("\n" + "=" * 65)
    print("  🚁 Aerial Surveillance & Crime Detection Suite (Monoserver)  ")
    print("=" * 65)
    print(f"  [+] Unified Monoserver listening on: http://127.0.0.1:{port}")
    print(f"  [+] REST API prefix:                http://127.0.0.1:{port}/api")
    print(f"  [+] React SPA build directory:      {FRONTEND_DIST_DIR}")
    print("=" * 65 + "\n")

    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    start_monoserver()

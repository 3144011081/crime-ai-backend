import os
import json
from pathlib import Path
from flask import jsonify, send_from_directory, request, Response
from backend.config import OUTPUT_DIR
from . import api_bp

def get_all_reports():
    """Load all JSON report files from outputs/ directory."""
    reports = []
    report_files = sorted(OUTPUT_DIR.glob("*_report.json"), key=os.path.getmtime, reverse=True)
    for r_file in report_files:
        try:
            with open(r_file, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data["report_file"] = r_file.name
                    data["mtime"] = os.path.getmtime(r_file)
                    if "name" not in data or not data["name"]:
                        data["name"] = r_file.stem.replace("_report", "")
                    reports.append(data)
        except Exception as err:
            print(f"[API] Error loading report {r_file.name}: {err}")
    return reports

@api_bp.route("/reports", methods=["GET"])
def list_reports():
    """Return all audit reports."""
    reports = get_all_reports()
    return jsonify({
        "total": len(reports),
        "reports": reports
    })

@api_bp.route("/reports/<path:filename>", methods=["GET"])
def get_report(filename):
    """Get a specific report by report_file name or source video name."""
    stem = Path(filename).stem
    if stem.endswith("_report"):
        report_name = f"{stem}.json"
    else:
        report_name = f"{stem}_report.json"

    report_path = OUTPUT_DIR / report_name
    if not report_path.exists():
        # Fallback search
        matches = list(OUTPUT_DIR.glob(f"*{stem}*_report.json"))
        if matches:
            report_path = matches[0]

    if report_path.exists():
        try:
            with open(report_path, "r") as f:
                data = json.load(f)
                data["report_file"] = report_path.name
                if "name" not in data or not data["name"]:
                    data["name"] = report_path.stem.replace("_report", "")
                return jsonify(data)
        except Exception as err:
            return jsonify({"error": f"Failed to read report: {err}"}), 500

    return jsonify({"error": f"Report for '{filename}' not found."}), 404

@api_bp.route("/reports/download/<path:filename>", methods=["GET"])
def download_report_file(filename):
    """Download JSON report file."""
    if (OUTPUT_DIR / filename).exists():
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

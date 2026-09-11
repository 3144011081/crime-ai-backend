#!/usr/bin/env python3
"""
CLI Launcher for CrimeAI Monoserver.
Usage:
    python run.py
    python run.py --port 5005
    python run.py --debug
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.monoserver import start_monoserver

def main():
    parser = argparse.ArgumentParser(description="Aerial Surveillance & Crime Detection Monoserver")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on (default: auto/5005)")
    parser.add_argument("--debug", action="store_true", help="Run with Flask debug mode enabled")
    args = parser.parse_args()

    start_monoserver(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()

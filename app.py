#!/usr/bin/env python3
"""
Application entrypoint for the Aerial Surveillance & Crime Detection Suite.
Delegates to the modular Monoserver architecture.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.monoserver import start_monoserver

if __name__ == "__main__":
    start_monoserver()

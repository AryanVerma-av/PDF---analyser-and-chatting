import sys
from pathlib import Path

# Add backend directory to sys.path so "app.*" modules are discoverable
backend_dir = str(Path(__file__).resolve().parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

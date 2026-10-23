"""Run the sidon-rocm CLI from a checkout without installing this package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidon_rocm_windows.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

"""Restore one WAV on CPU (reference) and GPU, then compare them. See docs/findings.md."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidon_rocm_windows.restore import main

if __name__ == "__main__":
    raise SystemExit(main())

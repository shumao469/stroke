"""Run the source checkout without installation or external dependencies."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from strokeweave.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())

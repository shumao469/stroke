from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from istbi_behavior_decoder.open_field import run_open_field


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode open-field videos under a base directory.")
    parser.add_argument("--base-dir", required=True, help="Root directory containing the experimental folders and mp4 files.")
    parser.add_argument("--skip-seconds", type=float, default=5.0, help="Seconds to skip at the beginning of each video.")
    parser.add_argument("--analyze-seconds", type=float, default=300.0, help="Maximum number of seconds to analyze per video.")
    parser.add_argument("--pixel-to-cm", type=float, default=None, help="Optional scale factor. Example: 0.045 means 1 pixel = 0.045 cm.")
    parser.add_argument("--center-ratio", type=float, default=0.5, help="Central rectangle size relative to ROI bounding box.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = run_open_field(
        base_dir=args.base_dir,
        skip_seconds=args.skip_seconds,
        analyze_seconds=args.analyze_seconds,
        pixel_to_cm=args.pixel_to_cm,
        center_ratio=args.center_ratio,
    )
    print(df.head())
    print(f"Finished. Processed {len(df)} open-field videos.")


if __name__ == "__main__":
    main()

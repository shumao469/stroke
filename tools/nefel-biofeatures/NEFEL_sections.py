"""NEFEL Section 1–7 (organized single-file entry)

This file mirrors the notebook structure and is convenient for:
- sharing with collaborators who prefer a single .py
- quick grepping of "which section computes what"

Preferred usage is via the installed package:
    from nefel.sections import section1_inos_arg, ...

But you can also run this file as a script to batch-process a folder.

Example
-------
python NEFEL_sections.py --section 4 --image_dir ./CD31 --out cd31.csv --qc qc_masks
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import cv2
import pandas as pd

from nefel.sections import (
    section1_inos_arg,
    section2_iba1_day1,
    section3_claudin5,
    section4_cd31,
    section5_synapse,
    section6_gap43,
    section7_tunel,
)

def load_rgb(p: Path):
    bgr = cv2.imread(str(p))
    if bgr is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", type=int, required=True, choices=range(1,8),
                    help="Which section to run (1..7)")
    ap.add_argument("--image_dir", type=str, required=True,
                    help="Folder containing images")
    ap.add_argument("--pattern", type=str, default=r".*\.(png|jpg|jpeg|tif|tiff)$",
                    help="Regex pattern for image filenames")
    ap.add_argument("--out", type=str, default="metrics.csv",
                    help="Output CSV path")
    ap.add_argument("--qc", type=str, default=None,
                    help="Optional QC output directory (overlays/masks)")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    regex = re.compile(args.pattern, re.IGNORECASE)

    rows = []
    for p in sorted(image_dir.iterdir()):
        if not p.is_file() or not regex.match(p.name):
            continue

        if args.section == 7:
            metrics = section7_tunel(p, prefix=p.stem, out_dir=args.qc)
        else:
            rgb = load_rgb(p)
            if args.section == 1:
                metrics = section1_inos_arg(rgb)
            elif args.section == 2:
                metrics = section2_iba1_day1(rgb, prefix=p.stem, out_dir=args.qc)
            elif args.section == 3:
                metrics = section3_claudin5(rgb, prefix=p.stem, out_dir=args.qc)
            elif args.section == 4:
                metrics = section4_cd31(rgb, prefix=p.stem, out_dir=args.qc)
            elif args.section == 5:
                metrics = section5_synapse(rgb)
            elif args.section == 6:
                metrics = section6_gap43(rgb, prefix=p.stem, out_dir=args.qc)
            else:
                raise ValueError("Invalid section")

        rows.append({"file": p.name, **metrics})

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out} ({len(df)} images)")

if __name__ == "__main__":
    main()

"""nefel.markers.gap43

Section 6: GAP43 Analysis

Main API
- extract_gap43_fibers(rgb, tissue_mask)
- quantify_gap43(rgb, prefix="...", out_dir=None)
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from ..core import tissue_mask

def extract_gap43_fibers(rgb, tissue):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]

    # GAP-43 DAB 褐色经验范围
    fib = (
        (H >= 5) & (H <= 35) &
        (S >= 25) &
        (V <= 230) &
        tissue
    )

    # 形态学增强细纤维
    fib = cv2.morphologyEx(fib.astype(np.uint8),
                           cv2.MORPH_OPEN,
                           np.ones((3,3), np.uint8))
    fib = cv2.morphologyEx(fib,
                           cv2.MORPH_CLOSE,
                           np.ones((5,5), np.uint8))
    return fib.astype(bool)


def quantify_gap43(rgb, prefix: str = "sample", out_dir=None):
    """Quantify GAP43 fiber metrics; optionally save QC overlays."""
    tissue = tissue_mask(rgb)
    fibers = extract_gap43_fibers(rgb, tissue)

    tissue_area = int(tissue.sum())
    fiber_area = int(fibers.sum())

    skel = skeletonize(fibers)
    skel_len = int(skel.sum())

    lab = label(skel)
    regions = regionprops(lab)
    seg_lens = [r.area for r in regions if r.area > 15]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean_int = float(gray[fibers].mean()) if fiber_area > 0 else 0.0

    if out_dir is not None:
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        overlay = rgb.copy()
        overlay[skel] = [255, 0, 0]
        cv2.imwrite(str(out_dir / f"{prefix}_skeleton_overlay.jpg"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{prefix}_fiber_mask.jpg"),
                    (fibers*255).astype(np.uint8))

    return {
        "gap43_area_ratio": fiber_area / (tissue_area + 1e-9),
        "skeleton_total_length": skel_len,
        "mean_fiber_length": float(np.mean(seg_lens)) if seg_lens else 0.0,
        "p90_fiber_length": float(np.percentile(seg_lens, 90)) if seg_lens else 0.0,
        "segment_count": int(len(seg_lens)),
        "mean_intensity": mean_int,
    }


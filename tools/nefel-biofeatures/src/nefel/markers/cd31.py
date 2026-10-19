"""nefel.markers.cd31

Section 4: CD31 Analysis

Main API
- extract_cd31_mask(rgb, tissue_mask)
- quantify_cd31(rgb, prefix="...", out_dir=None)

The QC outputs (overlay/mask) are only written when `out_dir` is provided.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import cv2
from skimage.color import rgb2hed
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from ..core import tissue_mask

def extract_cd31_mask(rgb, tissue):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]

    # CD31 DAB 褐色经验范围（可微调）
    brown = (
        (H >= 5) & (H <= 35) &
        (S >= 40) &
        (V <= 220) &
        tissue
    )

    # 去噪
    brown = cv2.morphologyEx(brown.astype(np.uint8),
                             cv2.MORPH_OPEN,
                             np.ones((3,3), np.uint8))
    brown = cv2.morphologyEx(brown,
                             cv2.MORPH_CLOSE,
                             np.ones((5,5), np.uint8))
    return brown.astype(bool)


def quantify_cd31(rgb, prefix: str = "sample", out_dir=None):
    """Quantify CD31 vasculature metrics; optionally save QC overlays."""
    tissue = tissue_mask(rgb)
    vessel = extract_cd31_mask(rgb, tissue)

    tissue_area = int(tissue.sum())
    vessel_area = int(vessel.sum())

    skel = skeletonize(vessel)
    skel_len = int(skel.sum())

    lab = label(skel)
    regions = regionprops(lab)
    seg_lens = [r.area for r in regions if r.area > 10]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean_int = float(gray[vessel].mean()) if vessel_area > 0 else 0.0

    if out_dir is not None:
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        overlay = rgb.copy()
        overlay[skel] = [255, 0, 0]
        cv2.imwrite(str(out_dir / f"{prefix}_skeleton_overlay.jpg"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{prefix}_vessel_mask.jpg"),
                    (vessel*255).astype(np.uint8))

    return {
        "cd31_area_ratio": vessel_area / (tissue_area + 1e-9),
        "skeleton_total_length": skel_len,
        "mean_segment_length": float(np.mean(seg_lens)) if seg_lens else 0.0,
        "p90_segment_length": float(np.percentile(seg_lens, 90)) if seg_lens else 0.0,
        "segment_count": int(len(seg_lens)),
        "mean_intensity": mean_int,
    }


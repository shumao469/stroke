"""nefel.markers.iba1_day1

Section 2: Iba1 Analysis (Day 1)

This module provides Iba1 quantification for microglial activation.

Main API
- quantify_iba1(rgb, prefix="...", out_dir=None)
- quantify_iba1_morphology(rgb, od_thr)

Key outputs
- iba1_area_ratio
- cell_count
- skeleton_total_length
- branching_index
- mean_intensity

Notes
- 'Day 1' indicates the experimental timepoint; encode it in metadata.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import cv2
from skimage.color import rgb2hed
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from ..core import tissue_mask

def extract_iba1_mask(rgb, tissue):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]

    iba1 = (
        (H >= 5) & (H <= 35) &
        (S >= 30) &
        (V <= 230) &
        tissue
    )

    iba1 = cv2.morphologyEx(iba1.astype(np.uint8),
                            cv2.MORPH_OPEN,
                            np.ones((3,3), np.uint8))
    iba1 = cv2.morphologyEx(iba1,
                            cv2.MORPH_CLOSE,
                            np.ones((5,5), np.uint8))
    return iba1.astype(bool)

def quantify_iba1_morphology(rgb, od_thr):
    hed = rgb2hed(rgb)
    dab_od = np.clip(-hed[..., 2], 0, None)

    tissue = tissue_mask(rgb)
    mask = (dab_od >= od_thr) & tissue

    lab = label(mask)
    regions = regionprops(lab)

    soma_areas = []
    branch_pixels = []

    for r in regions:
        if r.area < 30:
            continue
        soma_areas.append(r.area)
        skel = skeletonize(r.image)
        branch_pixels.append(skel.sum())

    return {
        "cell_count": len(soma_areas),
        "mean_soma_area": np.mean(soma_areas) if soma_areas else 0,
        "mean_branch_pixels": np.mean(branch_pixels) if branch_pixels else 0
    }


def quantify_iba1(rgb, prefix: str = "sample", out_dir=None):
    """Quantify Iba1 mask and morphology metrics.

    Parameters
    ----------
    rgb:
        RGB uint8 image.
    prefix:
        Prefix for saved masks/overlays when `out_dir` is provided.
    out_dir:
        If provided (str/Path), write diagnostic overlay and mask images.
        If None, no files are written.

    Returns
    -------
    dict:
        iba1_area_ratio, cell_count, skeleton_total_length, branching_index, mean_intensity
    """
    tissue = tissue_mask(rgb)
    iba1 = extract_iba1_mask(rgb, tissue)

    tissue_area = int(tissue.sum())
    iba1_area = int(iba1.sum())

    lab = label(iba1)
    cells = [r for r in regionprops(lab) if r.area > 50]
    cell_count = len(cells)

    skel = skeletonize(iba1)
    skel_len = int(skel.sum())

    branching_index = skel_len / (cell_count + 1e-9)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean_int = float(gray[iba1].mean()) if iba1_area > 0 else 0.0

    if out_dir is not None:
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        overlay = rgb.copy()
        overlay[skel] = [255, 0, 0]
        cv2.imwrite(str(out_dir / f"{prefix}_skeleton_overlay.jpg"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{prefix}_iba1_mask.jpg"),
                    (iba1*255).astype(np.uint8))

    return {
        "iba1_area_ratio": iba1_area / (tissue_area + 1e-9),
        "cell_count": int(cell_count),
        "skeleton_total_length": int(skel_len),
        "branching_index": float(branching_index),
        "mean_intensity": float(mean_int),
    }


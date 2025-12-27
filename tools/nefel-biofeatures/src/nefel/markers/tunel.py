"""nefel.markers.tunel

Section 7: TUNEL Analysis

Main API
- quantify_tunel_image(path, prefix="...", out_dir=None)
- detect_dapi_nuclei(rgb)
- detect_tunel_signal(rgb)

QC output is only written when `out_dir` is provided.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import cv2
from skimage.measure import label, regionprops

def detect_dapi_nuclei(rgb, min_area=20, max_area=500):
    B = rgb[...,2]  # 蓝色
    thr = np.percentile(B[B > 0], 80)
    mask = B >= thr

    lab = label(mask)
    nuclei = []

    for r in regionprops(lab):
        if min_area <= r.area <= max_area:
            nuclei.append(r)

    return nuclei

def detect_tunel_signal(rgb):
    R = rgb[...,0]  # 红色
    thr = np.percentile(R[R > 0], 95)
    mask = R >= thr
    return mask


def quantify_tunel_image(path, prefix: str = "sample", out_dir=None):
    """Quantify TUNEL from an image path; optionally save QC mask overlay."""
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    nuclei = detect_dapi_nuclei(rgb)
    tunel_mask = detect_tunel_signal(rgb)

    total_cells = len(nuclei)
    tunel_pos = 0

    for n in nuclei:
        rr, cc = n.coords[:,0], n.coords[:,1]
        if tunel_mask[rr, cc].mean() > 0.2:  # ≥20% coverage
            tunel_pos += 1

    if out_dir is not None:
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        vis = rgb.copy()
        vis[tunel_mask] = [255, 0, 0]
        cv2.imwrite(str(out_dir / f"{prefix}_tunel_mask.jpg"),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    return {
        "total_cells": int(total_cells),
        "tunel_pos_cells": int(tunel_pos),
        "tunel_ratio": float(tunel_pos / (total_cells + 1e-9)),
    }


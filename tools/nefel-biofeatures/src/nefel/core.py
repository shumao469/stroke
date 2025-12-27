"""nefel.core

Shared utilities used across biomarker analyses.

Key conventions
- Image arrays are expected as **RGB** (H, W, 3) with dtype uint8.
- If you load images with OpenCV (`cv2.imread`) you get BGR; convert with:
  `rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)`.

These helpers intentionally avoid any hard-coded file paths to make the
package safe to import and easier to run in batch pipelines.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import cv2
from skimage.measure import label, regionprops


def norm01(x):
    x = np.array(x, dtype=float)
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)

def tissue_mask(rgb, thr=245):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray < thr
    mask = cv2.morphologyEx(mask.astype(np.uint8),
                             cv2.MORPH_CLOSE,
                             np.ones((5,5), np.uint8))
    return mask.astype(bool)

def split_channels(rgb):
    # OpenCV 读进来是 RGB
    R = rgb[...,0]
    G = rgb[...,1]
    B = rgb[...,2]
    return R, G, B

def effect_ratio(df, value_col, day, invert=False):
    """
    (TI - Control) / Control
    invert=True 用于“越低越好”的指标（炎症）
    """
    g = df.groupby("group")[value_col].mean()
    eff = (g["ti"] - g["control"]) / (g["control"] + 1e-9)
    if invert:
        eff = -eff
    return eff, day

def detect_nuclei(rgb, min_area=20, max_area=400):
    B = rgb[...,2]  # blue
    thr = np.percentile(B[B > 0], 80)
    mask = B >= thr

    lab = label(mask)
    nuclei = [r for r in regionprops(lab)
              if min_area <= r.area <= max_area]
    return nuclei

def detect_signal(channel, q=95):
    thr = np.percentile(channel[channel > 0], q)
    return channel >= thr

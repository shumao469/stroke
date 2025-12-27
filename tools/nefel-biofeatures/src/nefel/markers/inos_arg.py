"""nefel.markers.inos_arg

Section 1: iNOS / Arg analysis

Typical use-case
- Dual-channel immunofluorescence where iNOS is encoded in **red** and Arg/Arg1
  in **green**.
- We quantify:
  - area ratio (positive pixels / tissue pixels)
  - mean intensity in positive mask
  - integrated intensity (sum of intensities in positive mask)
  - Red/Green ratios (area and integrated intensity)

Implementation notes
- The original notebook used HSV thresholding for red/green separation.
- Defaults below replicate that logic but are exposed as parameters for robustness.

Output dictionary keys
- red_area_ratio, red_mean_intensity, red_integrated_intensity
- green_area_ratio, green_mean_intensity, green_integrated_intensity
- RG_ratio_area, RG_ratio_intensity
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import cv2
import pandas as pd
import re

# Default HSV thresholds (from the notebook)
RED_LOWER1 = np.array([0, 80, 50])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([160, 80, 50])
RED_UPPER2 = np.array([180, 255, 255])
GREEN_LOWER = np.array([35, 50, 50])
GREEN_UPPER = np.array([85, 255, 255])

def analyze_image(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- 红色 mask ---
    red_mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    red_mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # --- 绿色 mask ---
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    # --- 组织区域（非黑色）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tissue_mask = gray > 10

    def stats(mask):
        area = np.sum(mask > 0)
        intensity = np.mean(gray[mask > 0]) if area > 0 else 0
        integrated = np.sum(gray[mask > 0]) if area > 0 else 0
        return area, intensity, integrated

    red_area, red_mean, red_int = stats(red_mask)
    green_area, green_mean, green_int = stats(green_mask)

    tissue_area = np.sum(tissue_mask)

    return {
        "red_area_ratio": red_area / tissue_area,
        "red_mean_intensity": red_mean,
        "red_integrated_intensity": red_int,
        "green_area_ratio": green_area / tissue_area,
        "green_mean_intensity": green_mean,
        "green_integrated_intensity": green_int,
        "RG_ratio_area": red_area / (green_area + 1e-6),
        "RG_ratio_intensity": red_int / (green_int + 1e-6)
    }

def analyze_inos_arg(
    rgb: np.ndarray,
    red_lower1: np.ndarray = RED_LOWER1,
    red_upper1: np.ndarray = RED_UPPER1,
    red_lower2: np.ndarray = RED_LOWER2,
    red_upper2: np.ndarray = RED_UPPER2,
    green_lower: np.ndarray = GREEN_LOWER,
    green_upper: np.ndarray = GREEN_UPPER,
    tissue_gray_thr: int = 10,
) -> Dict[str, Any]:
    """Analyze a single RGB image array for iNOS (red) / Arg (green).

    Parameters
    ----------
    rgb:
        RGB uint8 image.
    red_lower1, red_upper1, red_lower2, red_upper2, green_lower, green_upper:
        HSV bounds used by OpenCV `inRange`.
    tissue_gray_thr:
        Tissue mask threshold: pixels with gray > thr are treated as tissue.

    Returns
    -------
    dict:
        Metrics described in the module docstring.
    """
    # convert RGB->BGR because the notebook used cv2.imread (BGR)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tissue_mask = gray > tissue_gray_thr

    def stats(mask):
        # intensity in original BGR (for consistency with notebook)
        vals = gray[mask.astype(bool)]
        area = int(mask.astype(bool).sum())
        mean = float(vals.mean()) if area > 0 else 0.0
        integ = float(vals.sum()) if area > 0 else 0.0
        return area, mean, integ

    red_area, red_mean, red_int = stats(red_mask & tissue_mask.astype(np.uint8))
    green_area, green_mean, green_int = stats(green_mask & tissue_mask.astype(np.uint8))

    tissue_area = int(tissue_mask.sum()) if tissue_mask is not None else 0

    return {
        "red_area_ratio": red_area / (tissue_area + 1e-9),
        "red_mean_intensity": red_mean,
        "red_integrated_intensity": red_int,
        "green_area_ratio": green_area / (tissue_area + 1e-9),
        "green_mean_intensity": green_mean,
        "green_integrated_intensity": green_int,
        "RG_ratio_area": red_area / (green_area + 1e-6),
        "RG_ratio_intensity": red_int / (green_int + 1e-6),
    }

def batch_inos_arg(
    image_dir: str | Path,
    pattern: str = r".*\.(png|jpg|jpeg|tif|tiff)$",
    save_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Batch-quantify iNOS/Arg for a directory of images.

    This helper is intentionally conservative:
    - It only reads image files matching `pattern`.
    - It does *not* assume any filename conventions; you can post-process the
      returned DataFrame to add metadata.

    Parameters
    ----------
    image_dir:
        Folder containing images.
    pattern:
        Regex pattern for file matching.
    save_csv:
        If provided, save the resulting table.

    Returns
    -------
    pd.DataFrame:
        One row per image.
    """
    image_dir = Path(image_dir)
    regex = re.compile(pattern, re.IGNORECASE)

    rows = []
    for p in sorted(image_dir.iterdir()):
        if not p.is_file():
            continue
        if not regex.match(p.name):
            continue
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        info = analyze_inos_arg(rgb)
        rows.append({"file": p.name, **info})

    df = pd.DataFrame(rows)
    if save_csv is not None:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
    return df

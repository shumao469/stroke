"""nefel.sections

A thin, human-readable wrapper that maps the paper/notebook "Sections" to code.

Sections
--------
1) iNOS / Arg analysis
2) Iba1 analysis (Day 1)
3) Claudin-5 analysis
4) CD31 analysis
5) Synapse analysis
6) GAP43 analysis
7) TUNEL analysis

Why this module exists
----------------------
- Reviewers and collaborators often want a one-to-one mapping between a methods
  section and an implementation entry point.
- Each section function here calls the corresponding marker module and returns
  a flat metrics dict (easy to save as CSV).

All functions accept in-memory RGB arrays (uint8). For path-based workflows,
use the marker modules that provide `*_image(path, ...)` helpers.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import cv2

from .markers.inos_arg import analyze_inos_arg
from .markers.iba1_day1 import quantify_iba1
from .markers.claudin5 import quantify_claudin5
from .markers.cd31 import quantify_cd31
from .markers.synapse import quantify_synapse
from .markers.gap43 import quantify_gap43
from .markers.tunel import quantify_tunel_image


def _load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --- Section 1 ---
def section1_inos_arg(rgb: np.ndarray) -> Dict[str, Any]:
    return analyze_inos_arg(rgb)


# --- Section 2 ---
def section2_iba1_day1(rgb: np.ndarray, prefix: str = "sample", out_dir=None) -> Dict[str, Any]:
    return quantify_iba1(rgb, prefix=prefix, out_dir=out_dir)


# --- Section 3 ---
def section3_claudin5(rgb: np.ndarray, prefix: str = "sample", q_thr: int = 85, out_dir=None) -> Dict[str, Any]:
    return quantify_claudin5(rgb, prefix=prefix, q_thr=q_thr, out_dir=out_dir)


# --- Section 4 ---
def section4_cd31(rgb: np.ndarray, prefix: str = "sample", out_dir=None) -> Dict[str, Any]:
    return quantify_cd31(rgb, prefix=prefix, out_dir=out_dir)


# --- Section 5 ---
def section5_synapse(
    rgb: np.ndarray,
    pre_channel: int = 1,
    post_channel: int = 0,
    thr_percentile: int = 99,
    min_area: int = 5,
    max_area: int = 200,
    coloc_radius_px: float = 3.0,
    use_tissue_mask: bool = True,
) -> Dict[str, Any]:
    """Synapse analysis from an RGB image array."""
    return quantify_synapse(
        rgb,
        pre_channel=pre_channel,
        post_channel=post_channel,
        thr_percentile=thr_percentile,
        min_area=min_area,
        max_area=max_area,
        max_dist=coloc_radius_px,
        use_tissue_mask=use_tissue_mask,
    )



# --- Section 6 ---
def section6_gap43(rgb: np.ndarray, prefix: str = "sample", out_dir=None) -> Dict[str, Any]:
    return quantify_gap43(rgb, prefix=prefix, out_dir=out_dir)


# --- Section 7 ---
def section7_tunel(path: str | Path, prefix: str = "sample", out_dir=None) -> Dict[str, Any]:
    return quantify_tunel_image(str(path), prefix=prefix, out_dir=out_dir)

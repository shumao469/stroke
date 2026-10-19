"""nefel.markers.synapse

Section 5: Synapse Analysis

Typical markers
- Presynaptic: Synaptophysin / Synapsin (often Green)
- Postsynaptic: PSD-95 / Homer1 (often Red)

Two entry points
1) quantify_synapse(rgb, pre_channel=1, post_channel=0, ...)
   - recommended for pipelines working with in-memory RGB arrays
2) quantify_synapse_image(path, ...)
   - convenience wrapper that reads an image and calls quantify_synapse

Detection & colocalization logic
- Puncta are detected by percentile thresholding (default q=99) on each channel,
  followed by connected-component size filtering.
- Colocalization is computed by nearest-neighbor matching within `max_dist` pixels
  using a k-d tree.

Outputs
- pre_count, post_count
- coloc_count, coloc_ratio
- pre_mean_area_px, post_mean_area_px
- pre_mean_intensity, post_mean_intensity
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import cv2
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops

from ..core import tissue_mask, split_channels

def detect_puncta(channel, thr_percentile=99, min_area=5, max_area=200):
    # 自适应阈值（非常关键）
    thr = np.percentile(channel[channel > 0], thr_percentile)
    mask = channel >= thr

    lab = label(mask)
    props = regionprops(lab, intensity_image=channel)

    centers = []
    areas = []
    intensities = []

    for p in props:
        if min_area <= p.area <= max_area:
            centers.append(p.centroid)
            areas.append(p.area)
            intensities.append(p.mean_intensity)

    return np.array(centers), np.array(areas), np.array(intensities)

def colocalization(pre_centers, post_centers, max_dist=3.0):
    if len(pre_centers) == 0 or len(post_centers) == 0:
        return 0.0, 0

    tree = cKDTree(post_centers)
    dists, _ = tree.query(pre_centers, distance_upper_bound=max_dist)

    coloc = dists < max_dist
    ratio = coloc.sum() / len(pre_centers)

    return ratio, coloc.sum()

def quantify_synapse(
    rgb: np.ndarray,
    pre_channel: int = 1,
    post_channel: int = 0,
    thr_percentile: int = 99,
    min_area: int = 5,
    max_area: int = 200,
    max_dist: float = 3.0,
    use_tissue_mask: bool = True,
) -> Dict[str, Any]:
    """Quantify synapse-like puncta and colocalization from RGB array.

    Parameters
    ----------
    rgb:
        RGB uint8 image.
    pre_channel, post_channel:
        Channel indices in RGB: 0=R, 1=G, 2=B.
    thr_percentile:
        Percentile threshold for puncta detection.
    min_area, max_area:
        Size filtering for connected components.
    max_dist:
        Max centroid distance (pixels) for colocalization.
    use_tissue_mask:
        If True, compute tissue area for density normalization.

    Returns
    -------
    dict
    """
    pre = rgb[..., pre_channel]
    post = rgb[..., post_channel]

    pre_cent, pre_area, pre_int = detect_puncta(pre, thr_percentile, min_area, max_area)
    post_cent, post_area, post_int = detect_puncta(post, thr_percentile, min_area, max_area)

    coloc_ratio, coloc_count = colocalization(pre_cent, post_cent, max_dist)

    # densities (optional)
    if use_tissue_mask:
        tissue = tissue_mask(rgb)
        area = float(tissue.sum()) + 1e-9
        pre_density = len(pre_cent) / area
        post_density = len(post_cent) / area
        coloc_density = coloc_count / area
    else:
        pre_density = post_density = coloc_density = 0.0

    return {
        "pre_count": int(len(pre_cent)),
        "post_count": int(len(post_cent)),
        "coloc_count": int(coloc_count),
        "coloc_ratio": float(coloc_ratio),
        "pre_mean_area_px": float(np.mean(pre_area)) if len(pre_area) else 0.0,
        "post_mean_area_px": float(np.mean(post_area)) if len(post_area) else 0.0,
        "pre_mean_intensity": float(np.mean(pre_int)) if len(pre_int) else 0.0,
        "post_mean_intensity": float(np.mean(post_int)) if len(post_int) else 0.0,
        "pre_density_per_px": float(pre_density),
        "post_density_per_px": float(post_density),
        "coloc_density_per_px": float(coloc_density),
    }

def quantify_synapse_image(
    path: str,
    pre_channel: int = 1,
    post_channel: int = 0,
    **kwargs,
) -> Dict[str, Any]:
    """Read image from `path` and run `quantify_synapse`."""
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return quantify_synapse(rgb, pre_channel=pre_channel, post_channel=post_channel, **kwargs)

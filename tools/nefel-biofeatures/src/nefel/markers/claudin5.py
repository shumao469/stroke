"""nefel.markers.claudin5

Section 3: Claudin-5 Analysis

Main API
- quantify_claudin5(rgb, prefix="...", q_thr=85, out_dir=None)
- quantify_claudin5_dab(rgb, od_thr)
- quantify_claudin5_intensity(rgb, channel="G")
- quantify_vessel_continuity(rgb, pos_mask, prefix="...", out_dir=None)
- dab_signal_auto(rgb, tissue_mask)

The default `quantify_claudin5` follows the notebook logic:
adaptive DAB OD extraction -> percentile threshold -> skeleton metrics.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import cv2
from skimage.color import rgb2hed
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from ..core import tissue_mask

def quantify_claudin5_dab(rgb):
    hed = rgb2hed(rgb)
    dab_od = np.clip(-hed[...,2], 0, None)

    tissue = tissue_mask(rgb)
    dab_t = dab_od[tissue]

    # 用高分位数，保证只抓到“血管线”
    thr = np.percentile(dab_t, 85)

    pos = (dab_od >= thr) & tissue

    pos_area = pos.sum()
    tissue_area = tissue.sum()

    mean_od_pos = dab_od[pos].mean() if pos_area > 0 else 0
    iod_pos = dab_od[pos].sum()

    return {
        "dab_pos_area_px": pos_area,
        "dab_pos_area_ratio": pos_area / tissue_area,
        "dab_meanOD_pos": mean_od_pos,
        "dab_IOD_pos": iod_pos,
        "dab_IOD_per_pos_area": iod_pos / (pos_area + 1e-9),
        "thr85": thr,
        "pos_mask": pos
    }

def quantify_claudin5_intensity(rgb):
    hed = rgb2hed(rgb)
    dab_od = np.clip(-hed[..., 2], 0, None)

    tissue = tissue_mask(rgb)
    dab_t = dab_od[tissue]

    # 取高分位，只保留血管线
    thr85 = np.percentile(dab_t, 85)
    pos = (dab_od >= thr85) & tissue

    pos_area = pos.sum()
    mean_od = dab_od[pos].mean() if pos_area > 0 else 0.0

    return {
        "dab_pos_area_px": int(pos_area),
        "dab_meanOD_pos": mean_od,
        "dab_meanOD_pos_x1000": mean_od * 1000,  # ★画图用
        "thr85": thr85,
        "pos_mask": pos
    }


def quantify_vessel_continuity(rgb, pos_mask, prefix: str = "sample", out_dir=None):
    """Quantify vessel continuity metrics from a binary positive mask."""
    skel = skeletonize(pos_mask)
    skel_len = int(skel.sum())

    lab = label(skel)
    regions = regionprops(lab)
    seg_lengths = [r.area for r in regions if r.area > 10]

    if out_dir is not None:
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        overlay = rgb.copy()
        overlay[skel] = [255, 0, 0]
        cv2.imwrite(str(out_dir / f"{prefix}_skeleton_overlay.jpg"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{prefix}_pos_mask.jpg"),
                    (pos_mask * 255).astype(np.uint8))

    return {
        "skeleton_total_length": skel_len,
        "segment_count": int(len(seg_lengths)),
        "mean_segment_length": float(np.mean(seg_lengths)) if seg_lengths else 0.0
    }


def dab_signal_auto(rgb, tissue):
    # --- 尝试 HED ---
    hed = rgb2hed(rgb)  # float
    d1 = -hed[..., 2]   # 方向1
    d2 =  hed[..., 2]   # 方向2

    # 用组织内 99分位判断“有没有动态范围”
    p99_1 = np.percentile(d1[tissue], 99)
    p99_2 = np.percentile(d2[tissue], 99)

    # 选择动态范围更大的一支
    if max(p99_1, p99_2) > 1e-4:  # 经验阈值：太小就认为 HED 失败
        dab = d1 if p99_1 >= p99_2 else d2
        dab = np.clip(dab, 0, None)
        method = "HED_D"
        return dab, method

    # --- 退回 HSV 褐色检测（对亮场 DAB 很稳）---
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]


def quantify_claudin5(rgb, prefix: str = "sample", q_thr: int = 85, out_dir=None):
    """Quantify Claudin-5 from DAB OD using an adaptive + percentile threshold.

    Parameters
    ----------
    rgb:
        RGB uint8 image.
    prefix:
        Used for saving overlays when out_dir is provided.
    q_thr:
        Percentile threshold applied on DAB OD within tissue.
    out_dir:
        If provided, saves overlay and mask for QC.

    Returns
    -------
    dict:
        method, thr_q, thr_value, dab_pos_area_px, dab_pos_area_ratio, dab_meanOD_pos,
        skeleton_total_length, segment_count, mean_segment_length, p99_dab_tissue
    """
    tissue = tissue_mask(rgb)
    dab, method = dab_signal_auto(rgb, tissue)

    dab_t = dab[tissue]
    if dab_t.size < 200:
        raise ValueError("Too little tissue.")

    thr = np.percentile(dab_t, q_thr)
    pos = (dab >= thr) & tissue

    pos_area = int(pos.sum())
    tissue_area = int(tissue.sum())

    mean_od = float(dab[pos].mean()) if pos_area > 0 else 0.0

    skel = skeletonize(pos)
    skel_len = int(skel.sum())

    lab = label(skel)
    regions = regionprops(lab)
    seg_lens = [r.area for r in regions if r.area > 10]

    if out_dir is not None:
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        overlay = rgb.copy()
        overlay[skel] = [255, 0, 0]
        cv2.imwrite(str(out_dir / f"{prefix}_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{prefix}_pos_mask.jpg"), (pos*255).astype(np.uint8))

    return {
        "method": method,
        "thr_q": int(q_thr),
        "thr_value": float(thr),
        "dab_pos_area_px": pos_area,
        "dab_pos_area_ratio": pos_area/(tissue_area+1e-9),
        "dab_meanOD_pos": mean_od,
        "dab_meanOD_pos_x1000": mean_od*1000,
        "skeleton_total_length": skel_len,
        "segment_count": int(len(seg_lens)),
        "mean_segment_length": float(np.mean(seg_lens)) if seg_lens else 0.0,
        "p99_dab_tissue": float(np.percentile(dab_t, 99))
    }


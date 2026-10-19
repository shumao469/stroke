from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

@dataclass(frozen=True)
class SegmentSpec:
    """Time segmentation spec for pre/stim/post.

    All times are in seconds.

    Two supported modes:

    1) Absolute times on a global time axis:
       - pre:  [pre_start, pre_end)
       - stim: [stim_start, stim_end)
       - post: [post_start, post_end)

    2) Relative-to-stim-onset:
       Provide stim_onset and durations:
         pre_dur, stim_dur, post_dur
       then segments are:
         pre  = [onset - pre_dur, onset)
         stim = [onset, onset + stim_dur)
         post = [onset + stim_dur, onset + stim_dur + post_dur)

    Notes
    -----
    In most TI experiments, it is recommended to define **stim_onset = 0 s**
    after alignment (see docs/ALIGNMENT.md).
    """
    # mode A: explicit bounds
    pre_start: Optional[float] = None
    pre_end: Optional[float] = None
    stim_start: Optional[float] = None
    stim_end: Optional[float] = None
    post_start: Optional[float] = None
    post_end: Optional[float] = None

    # mode B: onset + durations
    stim_onset: Optional[float] = None
    pre_dur: Optional[float] = None
    stim_dur: Optional[float] = None
    post_dur: Optional[float] = None

def _mask(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return (t >= a) & (t < b)

def segment_by_time(t_sec: np.ndarray, X: np.ndarray, spec: SegmentSpec) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split features `X` by pre/stim/post according to time vector `t_sec`.

    Parameters
    ----------
    t_sec: (T,)
    X: (T, F) or (T,)
    spec: SegmentSpec

    Returns
    -------
    dict with keys 'pre','stim','post' → (t_seg, X_seg)
    """
    t = np.asarray(t_sec, dtype=float).reshape(-1)
    if X.ndim == 1:
        Xv = X.reshape(-1, 1)
    else:
        Xv = X
    if Xv.shape[0] != t.shape[0]:
        raise ValueError(f"Time and data length mismatch: t={t.shape}, X={Xv.shape}")

    # Resolve bounds
    if spec.stim_onset is not None and spec.pre_dur is not None and spec.stim_dur is not None and spec.post_dur is not None:
        onset = float(spec.stim_onset)
        pre_a, pre_b = onset - float(spec.pre_dur), onset
        st_a, st_b = onset, onset + float(spec.stim_dur)
        po_a, po_b = st_b, st_b + float(spec.post_dur)
    else:
        need = [spec.pre_start, spec.pre_end, spec.stim_start, spec.stim_end, spec.post_start, spec.post_end]
        if any(v is None for v in need):
            raise ValueError("SegmentSpec incomplete. Use either explicit bounds or onset+durations.")
        pre_a, pre_b = float(spec.pre_start), float(spec.pre_end)
        st_a, st_b = float(spec.stim_start), float(spec.stim_end)
        po_a, po_b = float(spec.post_start), float(spec.post_end)

    seg = {}
    for name, (a, b) in {
        "pre": (pre_a, pre_b),
        "stim": (st_a, st_b),
        "post": (po_a, po_b),
    }.items():
        m = _mask(t, a, b)
        seg[name] = (t[m], Xv[m])
    return seg

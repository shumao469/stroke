from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

@dataclass(frozen=True)
class SegmentSpec:
    """Define pre/stim/post segments.

    Use either:
    - onset+durations (recommended after alignment to stim_onset=0)
    or
    - explicit absolute bounds.
    """
    # Explicit bounds
    pre_start: Optional[float] = None
    pre_end: Optional[float] = None
    stim_start: Optional[float] = None
    stim_end: Optional[float] = None
    post_start: Optional[float] = None
    post_end: Optional[float] = None

    # Onset + durations
    stim_onset: Optional[float] = None
    pre_dur: Optional[float] = None
    stim_dur: Optional[float] = None
    post_dur: Optional[float] = None

def _mask(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return (t >= a) & (t < b)

def segment_by_time(t_sec: np.ndarray, X: np.ndarray, spec: SegmentSpec) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    t = np.asarray(t_sec, dtype=float).reshape(-1)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != t.shape[0]:
        raise ValueError(f"Time and features length mismatch: t={t.shape}, X={X.shape}")

    if spec.stim_onset is not None and spec.pre_dur is not None and spec.stim_dur is not None and spec.post_dur is not None:
        onset = float(spec.stim_onset)
        pre_a, pre_b = onset - float(spec.pre_dur), onset
        st_a, st_b = onset, onset + float(spec.stim_dur)
        po_a, po_b = st_b, st_b + float(spec.post_dur)
    else:
        need = [spec.pre_start, spec.pre_end, spec.stim_start, spec.stim_end, spec.post_start, spec.post_end]
        if any(v is None for v in need):
            raise ValueError("SegmentSpec incomplete. Provide onset+durations or all explicit bounds.")
        pre_a, pre_b = float(spec.pre_start), float(spec.pre_end)
        st_a, st_b = float(spec.stim_start), float(spec.stim_end)
        po_a, po_b = float(spec.post_start), float(spec.post_end)

    out = {}
    for name, (a, b) in {"pre": (pre_a, pre_b), "stim": (st_a, st_b), "post": (po_a, po_b)}.items():
        m = _mask(t, a, b)
        out[name] = (t[m], X[m])
    return out

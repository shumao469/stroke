from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .preprocess import robust_mad

@dataclass
class DetectConfig:
    fs: float = 20000.0
    thresh_sd: float = 4.0      # threshold in robust SD units
    polarity: str = "negative"  # 'negative' or 'positive'
    refractory_ms: float = 1.0  # minimum separation

def detect_spikes_threshold(x_bp: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    """Simple threshold-based spike detection with a refractory period.

    Returns spike indices (sample indices).
    """
    sigma = robust_mad(x_bp)
    thr = cfg.thresh_sd * sigma
    if cfg.polarity.lower().startswith("neg"):
        cand = np.where(x_bp < -thr)[0]
    else:
        cand = np.where(x_bp > thr)[0]

    if cand.size == 0:
        return cand.astype(np.int64)

    refractory = int(cfg.refractory_ms * 1e-3 * cfg.fs)
    clean = [int(cand[0])]
    last = int(cand[0])
    for s in cand[1:]:
        s = int(s)
        if s - last > refractory:
            clean.append(s)
            last = s
    return np.asarray(clean, dtype=np.int64)

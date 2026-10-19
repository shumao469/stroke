from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, filtfilt

@dataclass
class BandpassConfig:
    fs: float = 20000.0
    low_hz: float = 300.0
    high_hz: float = 6000.0
    order: int = 3

def bandpass_filter(x: np.ndarray, cfg: BandpassConfig) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * cfg.fs
    low = cfg.low_hz / nyq
    high = cfg.high_hz / nyq
    b, a = butter(cfg.order, [low, high], btype="band")
    return filtfilt(b, a, x).astype(np.float32)

def robust_mad(x: np.ndarray) -> float:
    """Robust estimate of scale using MAD (median absolute deviation)."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return 1.4826 * mad

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class QualityConfig:
    fs: float = 20000.0
    refractory_ms: float = 1.5  # for ISI violation
    snr_window: slice = slice(None)

def isi_violation_percent(spike_idx: np.ndarray, cfg: QualityConfig) -> float:
    """Percentage of ISIs below refractory."""
    if spike_idx.size < 2:
        return 0.0
    isi = np.diff(spike_idx) / cfg.fs
    min_isi = cfg.refractory_ms * 1e-3
    return float((isi < min_isi).sum() / len(isi) * 100.0)

def estimate_snr(wfs: np.ndarray) -> float:
    """Approximate SNR: peak-to-peak / (2*noise_std), where noise_std is from baseline."""
    if wfs.size == 0:
        return float("nan")
    # baseline from first 20% samples
    n = wfs.shape[1]
    base = wfs[:, : max(1, n//5)]
    noise = np.std(base)
    if noise <= 1e-12:
        return float("inf")
    p2p = np.mean(wfs.max(axis=1) - wfs.min(axis=1))
    return float(p2p / (2.0 * noise))

def firing_rate(spike_idx: np.ndarray, fs: float, t0: float, t1: float) -> float:
    """Firing rate (Hz) in [t0, t1) seconds."""
    if spike_idx.size == 0:
        return 0.0
    s0 = int(t0 * fs); s1 = int(t1 * fs)
    n = int(((spike_idx >= s0) & (spike_idx < s1)).sum())
    dur = max(t1 - t0, 1e-6)
    return float(n / dur)

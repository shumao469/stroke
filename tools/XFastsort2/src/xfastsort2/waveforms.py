from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA

@dataclass
class WaveformConfig:
    fs: float = 20000.0
    pre_ms: float = 1.0
    post_ms: float = 1.0

def extract_waveforms(x_bp: np.ndarray, spike_idx: np.ndarray, cfg: WaveformConfig) -> np.ndarray:
    """Extract aligned waveforms around each spike index.

    Returns
    -------
    wfs: (n_spikes, n_samples_win)
    """
    pre = int(cfg.pre_ms * 1e-3 * cfg.fs)
    post = int(cfg.post_ms * 1e-3 * cfg.fs)
    win = pre + post
    wfs = []
    for s in spike_idx:
        s = int(s)
        if s - pre >= 0 and s + post < len(x_bp):
            wfs.append(x_bp[s-pre:s+post])
    if len(wfs) == 0:
        return np.zeros((0, win), dtype=np.float32)
    return np.asarray(wfs, dtype=np.float32)

def waveform_features(wfs: np.ndarray) -> np.ndarray:
    """Compute simple waveform features.

    Features per spike:
      - p2p: peak-to-peak
      - trough: min
      - peak: max
      - energy: sum(w^2)
    """
    if wfs.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    p2p = (wfs.max(axis=1) - wfs.min(axis=1)).astype(np.float32)
    trough = wfs.min(axis=1).astype(np.float32)
    peak = wfs.max(axis=1).astype(np.float32)
    energy = (wfs ** 2).sum(axis=1).astype(np.float32)
    return np.stack([p2p, trough, peak, energy], axis=1)

def pca_embed(wfs: np.ndarray, n_components: int = 3) -> np.ndarray:
    """PCA embedding for clustering."""
    if wfs.shape[0] < max(5, n_components):
        return np.zeros((wfs.shape[0], n_components), dtype=np.float32)
    pca = PCA(n_components=n_components, random_state=0)
    return pca.fit_transform(wfs).astype(np.float32)

from __future__ import annotations
import numpy as np
from scipy.signal import welch
from typing import Tuple, Dict, List, Optional

def compute_erd_ers_ratio(
    eeg_data: np.ndarray,
    sfreq: float,
    freq_band: Tuple[float, float],
    baseline_samples: Tuple[int, int] = (0, 1000),
    task_samples: Tuple[int, int] = (1000, 3500),
    win: int = 500,
    overlap: int = 300,
) -> float:
    """
    Compute ERD/ERS ratio (%) for one channel (port of calculate_erd_ers in task.m).

    eeg_data: (n_times, n_epochs) or (n_times,).
    """
    if eeg_data.ndim == 1:
        eeg_data = eeg_data[:, None]
    if eeg_data.ndim != 2:
        raise ValueError("eeg_data must be (n_times, n_epochs)")

    n_times = eeg_data.shape[0]
    b0, b1 = baseline_samples
    t0, t1 = task_samples
    b1 = min(b1, n_times)
    t1 = min(t1, n_times)

    if t0 >= t1 or b0 >= b1:
        raise ValueError("baseline/task windows are invalid for this signal length")

    nfft = 1 << int(np.ceil(np.log2(win)))

    def avg_psd(x: np.ndarray):
        psds = []
        freqs = None
        for tr in range(x.shape[1]):
            f, pxx = welch(
                x[:, tr],
                fs=sfreq,
                nperseg=win,
                noverlap=overlap,
                nfft=nfft,
                detrend="constant",
            )
            psds.append(pxx)
            freqs = f
        return freqs, np.mean(np.stack(psds, axis=0), axis=0)

    fb, psd_b = avg_psd(eeg_data[b0:b1, :])
    ft, psd_t = avg_psd(eeg_data[t0:t1, :])

    f_lo, f_hi = freq_band
    idx_b = (fb >= f_lo) & (fb <= f_hi)
    idx_t = (ft >= f_lo) & (ft <= f_hi)

    baseline_power = np.trapz(psd_b[idx_b], fb[idx_b]) if np.any(idx_b) else 0.0
    task_power = np.trapz(psd_t[idx_t], ft[idx_t]) if np.any(idx_t) else 0.0

    if baseline_power == 0:
        return 0.0
    return (task_power - baseline_power) / baseline_power * 100.0

def compute_subject_metrics(
    eegdata: np.ndarray,
    sfreq: float = 500.0,
    stim_side: str = "left",
    original_order: Optional[List[str]] = None,
    target_order: Optional[List[str]] = None,
    freq_bands: Optional[List[Tuple[float, float]]] = None,
    band_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute ERD/ERS per band x channel and LI per band for one subject.
    eegdata: (n_channels, n_times, n_epochs)
    """
    if original_order is None:
        original_order = ["FT8", "TP8", "FC4", "CP4", "CP3", "FC3", "TP7", "FT7"]
    if target_order is None:
        target_order = ["CP3", "FC3", "TP7", "FT7", "CP4", "FC4", "TP8", "FT8"]
    if freq_bands is None:
        freq_bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    if band_names is None:
        band_names = ["delta", "theta", "alpha", "beta", "gamma"]

    idx_map = [original_order.index(ch) for ch in target_order]
    eeg = eegdata[idx_map, :, :]

    if stim_side.lower() == "right":
        eeg = eeg[[4, 5, 6, 7, 0, 1, 2, 3], :, :]

    n_bands = len(freq_bands)
    n_ch = eeg.shape[0]
    erd = np.zeros((n_bands, n_ch), dtype=float)
    ers = np.zeros((n_bands, n_ch), dtype=float)
    li = np.zeros((n_bands,), dtype=float)

    for k, band in enumerate(freq_bands):
        erd_ipsi = []
        erd_contra = []
        for ch in range(n_ch):
            ratio = compute_erd_ers_ratio(eeg[ch].T, sfreq=sfreq, freq_band=band)
            if ratio < 0:
                erd[k, ch] = ratio
                ers[k, ch] = 0.0
            else:
                ers[k, ch] = ratio
                erd[k, ch] = 0.0

            if ch < 4:
                erd_ipsi.append(erd[k, ch])
            else:
                erd_contra.append(erd[k, ch])

        a = float(np.mean(erd_ipsi)) if len(erd_ipsi) else 0.0
        b = float(np.mean(erd_contra)) if len(erd_contra) else 0.0
        denom = abs(a) + abs(b)
        li[k] = -(a - b) / denom if denom != 0 else 0.0

    return {
        "erd": erd,
        "ers": ers,
        "li": li,
        "band_names": np.array(band_names),
        "ch_names": np.array(target_order),
    }

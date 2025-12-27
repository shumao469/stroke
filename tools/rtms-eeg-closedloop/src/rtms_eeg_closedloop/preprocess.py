from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import mne

@dataclass
class PreprocessConfig:
    sfreq: float = 500.0
    l_freq: float = 0.5
    h_freq: float = 40.0
    tmin: float = -2.0
    tmax: float = 5.0
    baseline: Tuple[Optional[float], Optional[float]] = (-2.0, 0.0)
    reref: str = "average"  # or None

def epoch_from_trigger_channel(
    data: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    trigger_ch: str,
    threshold: Optional[float] = None,
    tmin: float = -2.0,
    tmax: float = 5.0,
    baseline: Tuple[Optional[float], Optional[float]] = (-2.0, 0.0),
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    reref: str = "average",
) -> mne.Epochs:
    """
    Create epochs using a trigger channel's rising edges (EEGLAB pop_chanevent equivalent).

    data must be shape (n_channels, n_samples). The trigger channel should be included in ch_names.
    """
    if data.ndim != 2:
        raise ValueError("data must be 2D (n_channels, n_samples)")
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose="ERROR")

    if reref == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    elif reref is None:
        pass
    else:
        raw.set_eeg_reference(reref, projection=False, verbose="ERROR")

    trig_idx = ch_names.index(trigger_ch)
    trig = raw.get_data(picks=[trig_idx])[0]

    if threshold is None:
        med = np.median(trig)
        mad = np.median(np.abs(trig - med)) + 1e-12
        threshold = med + 3.0 * mad

    above = trig > threshold
    edges = np.where(np.logical_and(above[1:], ~above[:-1]))[0] + 1
    if edges.size == 0:
        raise RuntimeError("No trigger edges detected. Check trigger channel and threshold.")

    events = np.zeros((edges.size, 3), dtype=int)
    events[:, 0] = edges
    events[:, 2] = 1
    event_id = {"trigger": 1}

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose="ERROR",
    )
    return epochs

def preprocess_task_eeg(
    eegdata_3d: np.ndarray,
    sfreq: float = 500.0,
    ch_names: Optional[List[str]] = None,
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    baseline: Tuple[Optional[float], Optional[float]] = (-2.0, 0.0),
    reref: str = "average",
) -> np.ndarray:
    """
    Basic preprocessing for already-epoched data shaped (n_channels, n_times, n_epochs).
    """
    if eegdata_3d.ndim != 3:
        raise ValueError("eegdata_3d must be 3D (n_channels, n_times, n_epochs)")
    n_ch = eegdata_3d.shape[0]
    if ch_names is None:
        ch_names = [f"EEG{c+1:02d}" for c in range(n_ch)]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_ch)
    epochs = mne.EpochsArray(np.transpose(eegdata_3d, (2,0,1)), info, verbose="ERROR")
    epochs.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose="ERROR")

    if reref == "average":
        epochs.set_eeg_reference("average", verbose="ERROR")

    if baseline is not None:
        epochs.apply_baseline(baseline, verbose="ERROR")

    return np.transpose(epochs.get_data(), (1,2,0))

from __future__ import annotations
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import mne
from typing import Tuple, Optional, Dict
from .io import save_mat

def compute_ersp_morlet(
    epochs_data: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    n_cycles: float | np.ndarray = 3.0,
    baseline: Tuple[Optional[float], Optional[float]] = (-2.0, 0.0),
    mode: str = "logratio",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ERSP (time-frequency power) using Morlet wavelets.

    epochs_data: (n_epochs, n_times) for one channel.
    Returns (power[n_freq, n_time], times_ms[n_time]).
    """
    if epochs_data.ndim != 2:
        raise ValueError("epochs_data must be 2D (n_epochs, n_times)")
    info = mne.create_info(ch_names=["ch"], sfreq=sfreq, ch_types=["eeg"])
    ep = mne.EpochsArray(epochs_data[:, None, :], info, tmin=0.0, verbose="ERROR")
    tfr = mne.time_frequency.tfr_morlet(
        ep, freqs=freqs, n_cycles=n_cycles, average=True, return_itc=False, verbose="ERROR"
    )
    tfr.apply_baseline(baseline=baseline, mode=mode, verbose="ERROR")
    power = tfr.data[0]
    times_ms = tfr.times * 1000.0
    return power, times_ms

def save_ersp_outputs(
    out_dir: str | Path,
    channel_idx: int,
    power: np.ndarray,
    times_ms: np.ndarray,
    freqs: np.ndarray,
    vmin: float = -5.0,
    vmax: float = 5.0,
    save_png: bool = True,
    save_matfile: bool = True,
) -> Dict[str, str]:
    """Save ERSP image and .mat file (port of eeglabtask.m)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    if save_png:
        fig = plt.figure()
        ax = plt.gca()
        im = ax.imshow(
            power,
            aspect="auto",
            origin="lower",
            extent=[times_ms[0], times_ms[-1], freqs[0], freqs[-1]],
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"ERSP - Channel {channel_idx}")
        im.set_clim(vmin, vmax)
        plt.colorbar(im, ax=ax)
        png_path = out_dir / f"ERSP_Channel_{channel_idx}.png"
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths["png"] = str(png_path)

    if save_matfile:
        mat_path = out_dir / f"ERSP_Channel_{channel_idx}.mat"
        save_mat(mat_path, ersp=power, times=times_ms, freqs=freqs)
        paths["mat"] = str(mat_path)

    return paths

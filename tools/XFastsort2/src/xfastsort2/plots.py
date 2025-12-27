from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_mean_waveforms(wfs: np.ndarray, labels: np.ndarray, out_png: str | Path, title: str = "Mean Waveforms"):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,5))
    for cid in np.unique(labels):
        w = wfs[labels == cid]
        if w.shape[0] == 0:
            continue
        plt.plot(w.mean(axis=0), label=f"C{cid}(n={w.shape[0]})")
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_snr_isi_qc(snr: np.ndarray, isi_v: np.ndarray, out_png: str | Path):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1)
    plt.scatter(isi_v, snr, s=10, alpha=0.8)
    plt.xlabel("ISI violation (%)")
    plt.ylabel("SNR")
    plt.title("SNR vs ISI violation")

    plt.subplot(1,3,2)
    plt.hist(snr, bins=10)
    plt.xlabel("SNR")
    plt.ylabel("Count")
    plt.title("SNR distribution")

    plt.subplot(1,3,3)
    plt.hist(isi_v, bins=10)
    plt.xlabel("ISI violation (%)")
    plt.ylabel("Count")
    plt.title("ISI violation distribution")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_raster(spike_times_sec: np.ndarray, labels: np.ndarray, out_png: str | Path, title: str = "Raster"):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12,3))
    for cid in np.unique(labels):
        st = spike_times_sec[labels == cid]
        plt.vlines(st, cid-0.4, cid+0.4)
    plt.xlabel("Time (s)")
    plt.ylabel("Cluster ID")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import os

import numpy as np
import pandas as pd

from .io import load_csv_trace
from .preprocess import BandpassConfig, bandpass_filter
from .detect import DetectConfig, detect_spikes_threshold
from .waveforms import WaveformConfig, extract_waveforms, waveform_features, pca_embed
from .cluster import ClusterConfig, cluster_kmeans
from .quality import QualityConfig, isi_violation_percent, estimate_snr
from .plots import plot_mean_waveforms, plot_snr_isi_qc, plot_raster

def sort_csv_trace_cpu(
    csv_path: str | Path,
    outdir: str | Path,
    *,
    fs: float = 20000.0,
    n_clusters: int = 3,
    bp_low: float = 300.0,
    bp_high: float = 6000.0,
    thresh_sd: float = 4.0,
    polarity: str = "negative",
    refractory_ms: float = 1.0,
) -> Dict[str, Any]:
    """End-to-end CPU spike sorting for a **single-channel CSV trace**.

    Steps
    -----
    1) Load trace
    2) Bandpass (300–6000 Hz by default)
    3) Threshold detection + refractory
    4) Extract waveforms
    5) Feature (PCA on waveforms + basic waveform features)
    6) Cluster (KMeans)
    7) QC: SNR + ISI violation
    8) Save results + plots

    Outputs (in outdir)
    -------------------
    - spikes.csv: spike index/time + cluster labels
    - cluster_quality.csv: per-cluster n_spikes, SNR, ISI violation
    - mean_waveforms.png
    - snr_isi_qc.png
    - raster.png
    """
    csv_path = Path(csv_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t, x = load_csv_trace(csv_path)
    # If timestamps are irregular/missing, fall back to uniform sampling
    if (len(t) != len(x)) or (np.any(np.diff(t) <= 0)):
        t = np.arange(len(x), dtype=np.float64) / fs

    x_bp = bandpass_filter(x, BandpassConfig(fs=fs, low_hz=bp_low, high_hz=bp_high))
    spike_idx = detect_spikes_threshold(x_bp, DetectConfig(fs=fs, thresh_sd=thresh_sd, polarity=polarity, refractory_ms=refractory_ms))
    wfs = extract_waveforms(x_bp, spike_idx, WaveformConfig(fs=fs, pre_ms=1.0, post_ms=1.0))

    # Features for clustering
    emb = pca_embed(wfs, n_components=3)
    feat = waveform_features(wfs)
    X = np.concatenate([emb, feat], axis=1) if wfs.shape[0] > 0 else np.zeros((0, 7), dtype=np.float32)

    labels = cluster_kmeans(X, ClusterConfig(n_clusters=n_clusters, random_state=0))
    spike_times = spike_idx / fs

    # QC per cluster
    rows = []
    for cid in np.unique(labels) if labels.size else []:
        idx_c = spike_idx[labels == cid]
        wfs_c = wfs[labels == cid]
        qc = QualityConfig(fs=fs, refractory_ms=1.5)
        rows.append({
            "cluster": int(cid),
            "n_spikes": int(idx_c.size),
            "snr": estimate_snr(wfs_c),
            "isi_violation_percent": isi_violation_percent(idx_c, qc),
        })
    qdf = pd.DataFrame(rows).sort_values(["cluster"]).reset_index(drop=True)

    # Save tables
    spikes_df = pd.DataFrame({"spike_index": spike_idx, "spike_time_sec": spike_times, "cluster": labels})
    spikes_df.to_csv(outdir / "spikes.csv", index=False)
    qdf.to_csv(outdir / "cluster_quality.csv", index=False)

    # Plots
    plot_mean_waveforms(wfs, labels, outdir / "mean_waveforms.png")
    if len(rows) > 0:
        plot_snr_isi_qc(qdf["snr"].to_numpy(), qdf["isi_violation_percent"].to_numpy(), outdir / "snr_isi_qc.png")
    plot_raster(spike_times, labels, outdir / "raster.png")

    return {
        "csv": str(csv_path),
        "outdir": str(outdir),
        "n_spikes": int(spike_idx.size),
        "n_clusters": int(n_clusters),
    }

def batch_sort_csv_cpu(
    root_dir: str | Path,
    out_root: str | Path,
    *,
    pattern: str = ".csv",
    **kwargs,
) -> Dict[str, Any]:
    """Batch-run CPU sorting over a directory tree."""
    root_dir = Path(root_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    failures = []

    for p in root_dir.rglob(f"*{pattern}"):
        try:
            rel = p.relative_to(root_dir)
            outdir = out_root / rel.parent / p.stem
            sort_csv_trace_cpu(p, outdir, **kwargs)
            processed += 1
        except Exception as e:
            failures.append({"file": str(p), "error": str(e)})

    return {"processed": processed, "failures": failures}

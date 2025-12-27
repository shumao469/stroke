from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Optional, List

def _normality_p(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if np.allclose(np.std(x), 0):
        return 0.0
    z = (x - np.mean(x)) / (np.std(x) + 1e-12)
    return stats.kstest(z, "norm").pvalue

def _paired_test(before: np.ndarray, after: np.ndarray) -> float:
    p1 = _normality_p(before)
    p2 = _normality_p(after)
    if (p1 > 0.05) and (p2 > 0.05):
        return stats.ttest_rel(before, after, nan_policy="omit").pvalue
    try:
        return stats.wilcoxon(before, after, zero_method="wilcox", correction=False).pvalue
    except ValueError:
        return 1.0

def _cohens_d_paired(before: np.ndarray, after: np.ndarray) -> float:
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if (np.mean(after) > 0 and np.mean(before) > 0) or (np.mean(after) < 0 and np.mean(before) < 0):
        mean_diff = np.mean(after) - np.mean(before)
    else:
        mean_diff = 0.0
    pooled_std = np.sqrt(((np.std(before, ddof=1)**2 + np.std(after, ddof=1)**2) / 2.0))
    return (mean_diff / pooled_std) if pooled_std != 0 else 0.0

def compute_significance_matrices(
    results_before_xlsx: str,
    results_after_xlsx: str,
    li_before_xlsx: str,
    li_after_xlsx: str,
    p_threshold: float = 0.05,
    d_threshold: float = 0.5,
    n_subjects: int = 9,
    n_channels: int = 8,
    n_bands: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Compute matrices used in ptest1219.m and apply thresholds.

    Returns dict:
      sig_diff: (n_bands, n_channels) mean_diff where significant else 0
      effect_d: (n_bands, n_channels) Cohen's d thresholded by |d|>=d_threshold else 0
      li_sig:   (n_bands,) LI mean_diff where significant else 0
      li_d:     (n_bands,) LI Cohen's d thresholded
    """
    rb = pd.read_excel(results_before_xlsx, index_col=0)
    ra = pd.read_excel(results_after_xlsx, index_col=0)
    lb = pd.read_excel(li_before_xlsx, index_col=0)
    la = pd.read_excel(li_after_xlsx, index_col=0)

    before = np.zeros((rb.shape[0], n_channels, n_subjects), dtype=float)
    after = np.zeros((ra.shape[0], n_channels, n_subjects), dtype=float)
    for i in range(n_subjects):
        start = i * n_channels
        before[:, :, i] = rb.iloc[:, start:start+n_channels].to_numpy()
        after[:, :, i] = ra.iloc[:, start:start+n_channels].to_numpy()

    sig = np.zeros((n_bands, n_channels), dtype=float)
    dmat = np.zeros((n_bands, n_channels), dtype=float)

    for i in range(n_bands):
        for j in range(n_channels):
            b = before[i, j, :]
            a = after[i, j, :]
            p = _paired_test(b, a)
            d = _cohens_d_paired(b, a)
            if p < p_threshold:
                if (np.mean(a) > 0 and np.mean(b) > 0) or (np.mean(a) < 0 and np.mean(b) < 0):
                    sig[i, j] = np.mean(a) - np.mean(b)
                else:
                    sig[i, j] = 0.0
            dmat[i, j] = d

    li_sig = np.zeros((n_bands,), dtype=float)
    li_d = np.zeros((n_bands,), dtype=float)
    for i in range(n_bands):
        b = lb.iloc[i, :].to_numpy(dtype=float)
        a = la.iloc[i, :].to_numpy(dtype=float)
        p = _paired_test(b, a)
        d = _cohens_d_paired(b, a)
        if p < p_threshold:
            li_sig[i] = np.mean(a) - np.mean(b)
        li_d[i] = d

    d_thr = dmat.copy()
    d_thr[np.abs(d_thr) < d_threshold] = 0.0
    li_d_thr = li_d.copy()
    li_d_thr[np.abs(li_d_thr) < d_threshold] = 0.0

    return {"sig_diff": sig, "effect_d": d_thr, "li_sig": li_sig, "li_d": li_d_thr}

def plot_threshold_heatmaps(
    sig_diff: np.ndarray,
    effect_d: np.ndarray,
    li_sig: np.ndarray,
    li_d: np.ndarray,
    metric_labels: Optional[List[str]] = None,
    channel_labels: Optional[List[str]] = None,
    li_labels: Optional[List[str]] = None,
    out_png: Optional[str] = None,
    dpi: int = 200,
):
    """Plot 2x2 heatmaps similar to ptest1219.m (non-interactive)."""
    if metric_labels is None:
        metric_labels = ["ERD_delta","ERD_theta","ERD_alpha","ERD_beta","ERD_gamma"]
    if channel_labels is None:
        channel_labels = ["CP3","FC3","TP7","FT7","CP4","FC4","TP8","FT8"]
    if li_labels is None:
        li_labels = ["LI_delta","LI_theta","LI_alpha","LI_beta","LI_gamma"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    def _hm(ax, mat, title, xlab, ylab, xticks, yticks):
        im = ax.imshow(mat, aspect="auto", origin="upper")
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, rotation=45, ha="right")
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks)
        plt.colorbar(im, ax=ax)

    _hm(axs[0, 0], sig_diff, "Significant Difference Matrix", "Channels", "Metrics", channel_labels, metric_labels)
    _hm(axs[0, 1], li_sig[:, None], "LI Significant Difference", " ", "LI Metrics", ["Δ"], li_labels)
    _hm(axs[1, 0], effect_d, "Filtered Effect Size (Cohen's d)", "Channels", "Metrics", channel_labels, metric_labels)
    _hm(axs[1, 1], li_d[:, None], "Filtered LI Effect Size", " ", "LI Metrics", ["d"], li_labels)

    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    return fig

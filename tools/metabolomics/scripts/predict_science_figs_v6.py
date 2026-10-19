#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_science_figs.py (v5)
A + B full pipeline for small-sample multiclass metabolomics-style prediction with:
- Strict out-of-fold (OOF) probabilities from nested CV (optional)
- Optional training-only augmentation / oversampling (none/smote/adasyn/noise)
- Optional in-fold probability calibration (Platt / Isotonic) evaluated strictly OOF
- Robust calibration plot (reliability curves) + Brier + ECE + slope/intercept
- Robust AUROC/AUPRC + CI (stratified bootstrap w/ skip + fold-level fallback)
- Feature selection stability (outer-fold frequency)
- Mechanistic figures:
  - Module/signature score heatmap + compact group comparison (top modules only)
  - Partial-correlation network on top biomarkers (shrinkage precision; top edges only; discrete class colors)
  - Module eigenscore association bars (HS-NC, ZS-NC; stratified bootstrap CI)

Usage examples
--------------
# 1) Main analysis: strict OOF, no oversampling
python3 predict_science_figs.py --xlsx data.xlsx --oversample none --do_nested_tuning

# 2) Sensitivity: drop samples + training-only augmentation
python3 predict_science_figs.py --xlsx data.xlsx --drop NC1 NC3 --oversample noise --augment_n 100 --noise_scale 0.10 --do_nested_tuning

Notes
-----
- This script is designed for SMALL datasets. Many uncertainty estimates will be wide; that's expected.
- If some classes are extremely rare, per-class CIs may be unstable. The script will:
  (i) stratified bootstrap with skip-if-single-class, and
  (ii) fall back to fold-level quantile CI when bootstrap yields too few valid draws.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    average_precision_score, precision_recall_curve,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state

try:
    from scipy.stats import kruskal
except Exception:
    kruskal = None

# Optional: imbalanced-learn for SMOTE / ADASYN
_IMBLEARN_OK = True
try:
    from imblearn.over_sampling import SMOTE, ADASYN
except Exception:
    _IMBLEARN_OK = False

# Optional: graphical lasso for partial correlation
_GLASSO_OK = True
try:
    from sklearn.covariance import GraphicalLassoCV, LedoitWolf
except Exception:
    _GLASSO_OK = False


# -----------------------------
# Utilities
# -----------------------------
def _mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _softmax(z: np.ndarray, axis: int = 1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta (nonparametric effect size)."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # O(n*m) but small samples
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (len(x) * len(y))


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR. Returns q-values."""
    p = np.asarray(pvals, dtype=float)
    n = np.sum(np.isfinite(p))
    q = np.full_like(p, np.nan)
    if n == 0:
        return q
    idx = np.argsort(p)
    ranked = p[idx]
    m = len(ranked)
    # handle NaN: move to end
    finite_mask = np.isfinite(ranked)
    finite = ranked[finite_mask]
    m_f = len(finite)
    if m_f == 0:
        return q
    q_f = finite * m_f / (np.arange(1, m_f + 1))
    q_f = np.minimum.accumulate(q_f[::-1])[::-1]
    out = np.full(m, np.nan)
    out[np.where(finite_mask)[0]] = q_f
    q[idx] = out
    return q


@dataclass
class CalMetrics:
    brier: float
    ece: float
    slope: float
    intercept: float
    n: int


def _ece_binary(y: np.ndarray, p: np.ndarray, bin_edges: np.ndarray) -> float:
    """Expected calibration error for binary."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    ece = 0.0
    n = len(y)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        m = (p >= lo) & (p < hi) if i < len(bin_edges) - 2 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        acc = y[m].mean()
        conf = p[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def _adaptive_bins(p: np.ndarray, n_bins: int, min_bin_n: int) -> np.ndarray:
    """
    Build bin edges in [0,1] such that each bin has at least min_bin_n samples when possible.
    Strategy:
      - Start with uniform edges.
      - If any bin < min_bin_n, merge with neighbor greedily.
    """
    p = np.asarray(p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # compute counts
    def counts_for(edges_):
        cnt = []
        for i in range(len(edges_) - 1):
            lo, hi = edges_[i], edges_[i + 1]
            m = (p >= lo) & (p < hi) if i < len(edges_) - 2 else (p >= lo) & (p <= hi)
            cnt.append(int(m.sum()))
        return cnt

    edges = edges.tolist()
    while True:
        cnt = counts_for(edges)
        bad = [i for i, c in enumerate(cnt) if c < min_bin_n]
        if len(bad) == 0:
            break
        if len(edges) <= 3:  # cannot merge further (<2 bins)
            break
        i = bad[0]
        # Merge bin i with neighbor with smaller count increase penalty
        if i == 0:
            # merge with next: remove edge 1
            edges.pop(1)
        elif i == len(cnt) - 1:
            # merge with prev: remove last-1 edge
            edges.pop(-2)
        else:
            # decide merge left or right based on neighbor counts
            if cnt[i - 1] <= cnt[i + 1]:
                edges.pop(i)      # remove left boundary of bin i
            else:
                edges.pop(i + 1)  # remove right boundary of bin i
    return np.array(edges, dtype=float)


def _reliability_points(y: np.ndarray, p: np.ndarray, n_bins: int, min_bin_n: int, strategy: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (mean_pred, frac_pos, n_in_bin) for reliability curve.
    strategy: 'uniform' or 'quantile' or 'adaptive'
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(y) == 0:
        return np.array([]), np.array([]), np.array([])
    if strategy == "quantile":
        # quantile edges; protect small samples by unique probabilities
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(p, qs)
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.array([0.0, 0.5, 1.0])
    elif strategy == "adaptive":
        edges = _adaptive_bins(p, n_bins=n_bins, min_bin_n=min_bin_n)
    else:  # uniform
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    mean_pred, frac_pos, n_bin = [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < len(edges) - 2 else (p >= lo) & (p <= hi)
        if m.sum() < min_bin_n:
            continue
        mean_pred.append(p[m].mean())
        frac_pos.append(y[m].mean())
        n_bin.append(m.sum())
    return np.array(mean_pred), np.array(frac_pos), np.array(n_bin)


def _calibration_metrics_binary(y: np.ndarray, p: np.ndarray, n_bins: int, min_bin_n: int, strategy: str) -> CalMetrics:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    # Brier
    try:
        brier = float(brier_score_loss(y, p))
    except Exception:
        brier = np.nan

    # ECE using edges
    if strategy == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(p, qs)
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.array([0.0, 0.5, 1.0])
    elif strategy == "adaptive":
        edges = _adaptive_bins(p, n_bins=n_bins, min_bin_n=max(1, min_bin_n))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = _ece_binary(y, p, edges)

    # Calibration slope & intercept via logistic regression: y ~ logit(p)
    slope = np.nan
    intercept = np.nan
    try:
        x = _safe_logit(p).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(x, y)
        slope = float(lr.coef_.ravel()[0])
        intercept = float(lr.intercept_.ravel()[0])
    except Exception:
        pass

    return CalMetrics(brier=brier, ece=ece, slope=slope, intercept=intercept, n=int(len(y)))


def _stratified_bootstrap_binary_metric(
    y: np.ndarray,
    p: np.ndarray,
    metric_fn,
    n_boot: int,
    seed: int,
    min_pos: int = 1,
    min_neg: int = 1,
) -> np.ndarray:
    """
    Stratified bootstrap for binary metrics:
      - sample positives with replacement from positives
      - sample negatives with replacement from negatives
    Skips draws that don't meet min_pos/min_neg.
    """
    rng = check_random_state(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) < min_pos or len(neg_idx) < min_neg:
        return np.array([])
    out = []
    for _ in range(n_boot):
        bpos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bneg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        bidx = np.concatenate([bpos, bneg])
        yb, pb = y[bidx], p[bidx]
        # safety
        if yb.min() == yb.max():
            continue
        try:
            out.append(metric_fn(yb, pb))
        except Exception:
            continue
    return np.array(out, dtype=float)


def _stratified_bootstrap_multiclass_metric(
    y: np.ndarray,
    P: np.ndarray,
    metric_fn,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """
    Stratified bootstrap for multiclass:
      - sample within each class label with replacement, preserving class counts.
    """
    rng = check_random_state(seed)
    y = np.asarray(y).astype(int)
    P = np.asarray(P).astype(float)
    classes = np.unique(y)
    idx_by_c = {c: np.where(y == c)[0] for c in classes}
    out = []
    for _ in range(n_boot):
        bidx = []
        for c in classes:
            idx = idx_by_c[c]
            if len(idx) == 0:
                continue
            bidx.append(rng.choice(idx, size=len(idx), replace=True))
        bidx = np.concatenate(bidx)
        yb, Pb = y[bidx], P[bidx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            out.append(metric_fn(yb, Pb))
        except Exception:
            continue
    return np.array(out, dtype=float)


# -----------------------------
# Data I/O
# -----------------------------
def load_xlsx_matrix(
    xlsx_path: str,
    label_col: str = "Group",
    id_col: str = "ID",
    sheet: Optional[Union[str, int]] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load data matrix from an Excel file.

    Supports two common layouts:

    (A) Sample-rows layout (classic ML table):
        rows = samples, columns = features (+ label_col, id_col).

    (B) Feature-rows layout (metabolomics-style):
        rows = metabolites/features, columns = samples (e.g., HS1/NC1/ZS1/QC1...),
        plus feature annotation columns (e.g., Metabolites, Class, KEGG, ...).
        In this case, we auto-transpose into (samples × features) and infer labels from sample IDs.

    Returns
    -------
    X_df : pd.DataFrame
        samples × features (float)
    y_str : pd.Series
        sample labels (string)
    meta_feat : dict
        feature-level metadata (DataFrame under key 'feature_table') if available
    """
    xls = pd.ExcelFile(xlsx_path)
    if sheet is None:
        # default: first sheet
        sheet = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)

    # --- Case A: sample-rows layout ---
    if label_col in df.columns:
        # Extract y + IDs
        y = df[label_col].astype(str)
        if id_col in df.columns:
            sample_ids = df[id_col].astype(str)
        else:
            sample_ids = pd.Series([f"S{i+1}" for i in range(len(df))], name="SampleID")
        # Select numeric feature columns (exclude label/id)
        exclude = {label_col}
        if id_col in df.columns:
            exclude.add(id_col)
        feat_cols = [c for c in df.columns if c not in exclude]
        X = df[feat_cols].copy()
        # Coerce to numeric where possible
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X.index = sample_ids.values
        X.index.name = "SampleID"
        y.index = X.index

        meta_feat = {"sheet": sheet, "layout": "sample_rows"}
        return X, y, meta_feat

    # --- Case B: feature-rows layout (auto-detect sample columns) ---
    # sample columns typically look like "HS1", "NC3", "ZS4", "QC2"
    sample_col_pattern = re.compile(r"^(NC|HS|ZS|QC)\d+$", re.IGNORECASE)
    sample_cols = [c for c in df.columns if sample_col_pattern.match(str(c).strip())]
    if len(sample_cols) < 3:
        raise ValueError(
            f"Cannot find label column '{label_col}' in sheet '{sheet}', and also cannot auto-detect "
            f"sample columns like HS1/NC1/ZS1/QC1. Columns head: {list(df.columns)[:15]}..."
        )

    # feature annotations: everything else
    annot_cols = [c for c in df.columns if c not in sample_cols]

    # choose a feature-name column (prefer metabolite name)
    preferred_name_cols = [
        "Metabolites", "Metabolite", "Compound", "Name",
        "Metabolites_cn", "ID", "m/z"
    ]
    name_col = None
    for c in preferred_name_cols:
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        name_col = annot_cols[0] if len(annot_cols) > 0 else None

    if name_col is not None:
        feat_names = df[name_col].astype(str).fillna("").replace("nan", "")
        # fallback when blank names exist
        blank = feat_names.str.strip().eq("")
        if blank.any():
            if "ID" in df.columns:
                feat_names.loc[blank] = df.loc[blank, "ID"].astype(str)
            else:
                feat_names.loc[blank] = [f"F{i+1}" for i in np.where(blank)[0]]
    else:
        feat_names = pd.Series([f"F{i+1}" for i in range(len(df))])

    # make feature names unique
    feat_names = feat_names.astype(str).tolist()
    seen = {}
    uniq = []
    for i, nm in enumerate(feat_names):
        nm0 = nm.strip() if nm is not None else ""
        if nm0 == "" or nm0.lower() == "nan":
            nm0 = f"F{i+1}"
        if nm0 not in seen:
            seen[nm0] = 1
            uniq.append(nm0)
        else:
            seen[nm0] += 1
            uniq.append(f"{nm0}__{seen[nm0]}")
    feat_names = uniq

    # Build X as samples × features (transpose)
    X = df[sample_cols].copy().T
    X.index = [str(s).strip() for s in X.index]
    X.index.name = "SampleID"
    X.columns = feat_names

    # numeric coercion
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Infer labels from sample IDs (prefix)
    # e.g., "HS1" -> "HS"
    y = pd.Series([re.match(r"^([A-Za-z]+)", sid).group(1).upper() if re.match(r"^([A-Za-z]+)", sid) else "UNK"
                   for sid in X.index], index=X.index, name="Group")

    # drop QC by default (keep only NC/HS/ZS)
    keep_mask = y.isin(["NC", "HS", "ZS"])
    X = X.loc[keep_mask].copy()
    y = y.loc[keep_mask].copy()

    # Feature metadata table (optional, useful for Class coloring, pathway evidence, etc.)
    feature_table = df[annot_cols].copy()
    feature_table.index = feat_names
    feature_table.index.name = "Feature"
    meta_feat = {"sheet": sheet, "layout": "feature_rows", "feature_table": feature_table}

    return X, y, meta_feat

def apply_drop_samples(X: pd.DataFrame, y: pd.Series, drop_list: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    if not drop_list:
        return X, y
    # try match by index or by a column? We only have index. We'll assume index contains sample names.
    # If the provided xlsx has a sample name column, user should set it as index before exporting.
    idx = X.index.astype(str)
    mask = ~idx.isin(drop_list)
    return X.loc[mask].copy(), y.loc[mask].copy()


# -----------------------------
# Augmentation / Oversampling (training only)
# -----------------------------
def augment_noise_by_feature_variance(X_train: np.ndarray, y_train: np.ndarray, n_new: int, noise_scale: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate n_new synthetic samples by adding Gaussian noise to randomly selected training samples.
    Noise sigma per feature is std(feature) * noise_scale.
    """
    rng = check_random_state(seed)
    if n_new <= 0:
        return X_train, y_train
    X = np.asarray(X_train, float)
    y = np.asarray(y_train)
    std = np.nanstd(X, axis=0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    idx = rng.choice(np.arange(X.shape[0]), size=n_new, replace=True)
    X_base = X[idx]
    noise = rng.normal(loc=0.0, scale=std * noise_scale, size=X_base.shape)
    X_new = X_base + noise
    y_new = y[idx]
    X_aug = np.vstack([X, X_new])
    y_aug = np.concatenate([y, y_new])
    return X_aug, y_aug


def oversample_train_only(X_train: np.ndarray, y_train: np.ndarray, method: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oversample using SMOTE/ADASYN. Requires imblearn.
    """
    if method == "none":
        return X_train, y_train
    if method in ("smote", "adasyn"):
        if not _IMBLEARN_OK:
            raise RuntimeError("imblearn is required for SMOTE/ADASYN. Please install imbalanced-learn or use --oversample noise/none.")
        if method == "smote":
            sampler = SMOTE(random_state=seed)
        else:
            sampler = ADASYN(random_state=seed)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        return np.asarray(X_res), np.asarray(y_res)
    raise ValueError(f"Unknown oversample method: {method}")


# -----------------------------
# Model / CV
# -----------------------------
def build_base_model(seed: int) -> LogisticRegression:
    # Elastic-net multinomial logistic regression (good baseline for small N, many features)
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=1.0,
        multi_class="multinomial",
        max_iter=5000,
        random_state=seed,
        n_jobs=1,
    )


def build_pipeline(seed: int, do_feature_select: bool, select_C: float = 0.5) -> Pipeline:
    # Preprocess: impute + scale
    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    base = build_base_model(seed)

    steps = [("pre", pre)]
    if do_feature_select:
        # L1-ish selector via elastic net logistic with high sparsity
        selector_model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=select_C,
            multi_class="ovr",
            max_iter=5000,
            random_state=seed,
            n_jobs=1,
        )
        steps.append(("select", SelectFromModel(selector_model, max_features=None)))
    steps.append(("clf", base))
    return Pipeline(steps)


def tune_hyperparams(pipe: Pipeline, X_train: np.ndarray, y_train: np.ndarray, seed: int, inner_splits: int) -> Pipeline:
    """
    Nested tuning inside outer fold. Keeps it light for small datasets.
    """
    param_grid = {
        "clf__C": [0.1, 0.3, 1.0, 3.0, 10.0],
        "clf__l1_ratio": [0.2, 0.5, 0.8],
    }
    # Also tune selector sparsity if present
    if "select" in pipe.named_steps:
        param_grid["select__estimator__C"] = [0.05, 0.1, 0.2, 0.5, 1.0]

    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc_ovr",
        cv=inner,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_


def fit_calibrator_binary(method: str, p_train: np.ndarray, y_train_bin: np.ndarray, seed: int):
    """
    Fit a calibrator g(p)->p_cal on training data only for a binary OVR problem.
    method: 'none'|'platt'|'isotonic'
    Returns a function.
    """
    method = method.lower()
    p_train = np.asarray(p_train).reshape(-1, 1)
    y_train_bin = np.asarray(y_train_bin).astype(int)

    if method == "none":
        return lambda p: np.asarray(p)

    if method == "platt":
        lr = LogisticRegression(solver="lbfgs", random_state=seed)
        lr.fit(_safe_logit(p_train.ravel()).reshape(-1, 1), y_train_bin)
        return lambda p: lr.predict_proba(_safe_logit(np.asarray(p).ravel()).reshape(-1, 1))[:, 1]

    if method == "isotonic":
        try:
            from sklearn.isotonic import IsotonicRegression
        except Exception as e:
            raise RuntimeError("Isotonic calibration requires scikit-learn isotonic module.") from e
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_train.ravel(), y_train_bin)
        return lambda p: iso.transform(np.asarray(p).ravel())

    raise ValueError(f"Unknown calibration method: {method}")


# -----------------------------
# Plotting helpers
# -----------------------------
def savefig(path: str, dpi: int = 300) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_auc_distribution(auc_outer: List[float], out_path: str, title: str) -> None:
    plt.figure(figsize=(4.2, 4.2))
    x = np.array(auc_outer, dtype=float)
    x = x[np.isfinite(x)]
    # Violin + jitter
    parts = plt.violinplot([x], showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_alpha(0.25)
    plt.scatter(np.ones_like(x), x, s=25, alpha=0.7)
    plt.boxplot([x], widths=0.25)
    plt.xticks([1], ["macro-AUC"])
    plt.ylabel("AUC (outer folds)")
    plt.title(title)
    savefig(out_path)


def plot_feature_frequency(freq: pd.Series, out_path: str, top_n: int = 20) -> None:
    top = freq.sort_values(ascending=False).head(top_n)[::-1]  # reverse for barh
    plt.figure(figsize=(6.5, 5.2))
    plt.barh(top.index.astype(str), top.values)
    plt.xlabel("Selection frequency (outer folds)")
    plt.title("Top biomarkers by selection stability (OOF)")
    savefig(out_path)


def plot_multiclass_roc(
    y_true_int: np.ndarray,
    P_oof: np.ndarray,
    classes: List[str],
    out_path: str,
    auc_ci: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> None:
    plt.figure(figsize=(5.0, 5.0))
    # diagonal
    plt.plot([0, 1], [0, 1], color="gray", lw=1)

    for k, name in enumerate(classes):
        yk = (y_true_int == k).astype(int)
        pk = P_oof[:, k]
        if yk.min() == yk.max():
            continue
        fpr, tpr, _ = roc_curve(yk, pk)
        ak = auc(fpr, tpr)
        label = f"{name} (AUC={ak:.3f})"
        if auc_ci and name in auc_ci and np.isfinite(auc_ci[name][1]) and np.isfinite(auc_ci[name][2]):
            label = f"{name} (AUC={auc_ci[name][0]:.3f}, 95%CI {auc_ci[name][1]:.3f}-{auc_ci[name][2]:.3f})"
        plt.plot(fpr, tpr, lw=2, label=label)

    # macro AUROC
    try:
        macro_auc = roc_auc_score(y_true_int, P_oof, multi_class="ovr", average="macro")
        plt.text(0.62, 0.08, f"Macro-AUROC={macro_auc:.3f}", transform=plt.gca().transAxes, fontsize=10)
    except Exception:
        pass

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Multiclass ROC (one-vs-rest, out-of-fold)")
    plt.legend(loc="lower right", fontsize=9)
    savefig(out_path)


def plot_multiclass_pr(
    y_true_int: np.ndarray,
    P_oof: np.ndarray,
    classes: List[str],
    out_path: str,
) -> None:
    plt.figure(figsize=(5.0, 5.0))
    for k, name in enumerate(classes):
        yk = (y_true_int == k).astype(int)
        pk = P_oof[:, k]
        if yk.min() == yk.max():
            continue
        prec, rec, _ = precision_recall_curve(yk, pk)
        ap = average_precision_score(yk, pk)
        plt.plot(rec, prec, lw=2, label=f"{name} (AP={ap:.3f})")

    # macro AUPRC (macro of per-class AP)
    aps = []
    for k in range(len(classes)):
        yk = (y_true_int == k).astype(int)
        if yk.min() == yk.max():
            continue
        aps.append(average_precision_score(yk, P_oof[:, k]))
    if len(aps) > 0:
        plt.text(0.60, 0.08, f"Macro-AUPRC={np.mean(aps):.3f}", transform=plt.gca().transAxes, fontsize=10)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multiclass PR (one-vs-rest, out-of-fold)")
    plt.legend(loc="lower left", fontsize=9)
    savefig(out_path)


def plot_calibration_ovr(
    y_true_int: np.ndarray,
    P_oof: np.ndarray,
    classes: List[str],
    out_path: str,
    n_bins: int,
    min_bin_n: int,
    strategy: str,
) -> Dict[str, CalMetrics]:
    """
    Plot OVR calibration reliability curves for each class + diagonal.
    Returns per-class calibration metrics.
    """
    plt.figure(figsize=(5.2, 5.2))
    plt.plot([0, 1], [0, 1], color="gray", lw=1, label="Perfectly calibrated")

    metrics = {}
    for k, name in enumerate(classes):
        yk = (y_true_int == k).astype(int)
        pk = P_oof[:, k]
        if yk.min() == yk.max():
            continue

        x, y, nbin = _reliability_points(yk, pk, n_bins=n_bins, min_bin_n=min_bin_n, strategy=strategy)
        if len(x) == 0:
            # fallback: reduce bins on the fly (robustness)
            x, y, nbin = _reliability_points(yk, pk, n_bins=max(2, n_bins - 1), min_bin_n=max(1, min_bin_n - 1), strategy="adaptive")
        if len(x) == 0:
            continue

        plt.plot(x, y, marker="o", lw=2, label=name)

        metrics[name] = _calibration_metrics_binary(yk, pk, n_bins=n_bins, min_bin_n=min_bin_n, strategy=strategy)

    # macro summary (mean across classes with metrics)
    if len(metrics) > 0:
        briers = [m.brier for m in metrics.values() if np.isfinite(m.brier)]
        eces = [m.ece for m in metrics.values() if np.isfinite(m.ece)]
        txt = []
        if len(briers) > 0:
            txt.append(f"Macro Brier={np.mean(briers):.3f}")
        if len(eces) > 0:
            txt.append(f"Macro ECE={np.mean(eces):.3f}")
        if txt:
            plt.text(0.04, 0.92, " | ".join(txt), transform=plt.gca().transAxes, fontsize=10)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction")
    plt.title("Calibration (one-vs-rest, out-of-fold)")
    plt.legend(loc="upper left", fontsize=9)
    savefig(out_path)
    return metrics


def plot_dca_binary(
    y: np.ndarray,
    p: np.ndarray,
    out_path: str,
    title: str,
    thresholds: Optional[np.ndarray] = None,
) -> None:
    """
    Classic binary Decision Curve Analysis (net benefit).
    y: 0/1
    p: predicted probability of class=1
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    n = len(y)
    prev = y.mean()
    nb_model = []
    nb_all = []
    nb_none = np.zeros_like(thresholds)

    for pt in thresholds:
        pred_pos = (p >= pt).astype(int)
        tp = np.sum((pred_pos == 1) & (y == 1))
        fp = np.sum((pred_pos == 1) & (y == 0))
        # net benefit
        w = pt / (1 - pt)
        nb = (tp / n) - (fp / n) * w
        nb_model.append(nb)

        # treat-all
        tp_all = np.sum(y == 1)
        fp_all = np.sum(y == 0)
        nbA = (tp_all / n) - (fp_all / n) * w
        nb_all.append(nbA)

    plt.figure(figsize=(5.2, 4.6))
    plt.plot(thresholds, nb_none, linestyle=":", lw=2, label="Treat none")
    plt.plot(thresholds, nb_all, linestyle="--", lw=2, label="Treat all (baseline)")
    plt.plot(thresholds, nb_model, lw=2, label="Model")

    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(title)
    plt.legend(loc="lower left", fontsize=9)
    savefig(out_path)


# -----------------------------
# Mechanistic / module analysis
# -----------------------------
def build_modules_unsupervised(
    X: pd.DataFrame,
    n_modules: int,
    method: str = "spearman_hclust",
    seed: int = 1,
) -> Dict[str, List[str]]:
    """
    Unsupervised feature modules based on absolute Spearman correlation + hierarchical clustering cut.
    Returns dict module_name -> feature list
    """
    # compute correlation (features x features)
    # for stability: use Spearman (rank)
    Xv = X.values
    # rank-transform per feature
    Xr = pd.DataFrame(Xv).rank(axis=0).values
    corr = np.corrcoef(Xr, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)

    # hierarchical clustering
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        Z = linkage(squareform(dist, checks=False), method="average")
        # labels 1..n_modules
        clusters = fcluster(Z, t=n_modules, criterion="maxclust")
    except Exception:
        # fallback: random assignment (shouldn't happen if scipy exists)
        rng = check_random_state(seed)
        clusters = rng.randint(1, n_modules + 1, size=X.shape[1])

    modules = {}
    cols = list(X.columns.astype(str))
    for m in range(1, n_modules + 1):
        feats = [cols[i] for i in np.where(clusters == m)[0]]
        if len(feats) == 0:
            continue
        modules[f"M{len(modules)+1}"] = feats
    return modules


def compute_module_scores(X: pd.DataFrame, modules: Dict[str, List[str]]) -> pd.DataFrame:
    """
    module score = mean z-score of features in module (z-score per feature across samples)
    """
    Xz = (X - X.mean(axis=0)) / X.std(axis=0).replace(0, np.nan)
    Xz = Xz.replace([np.inf, -np.inf], np.nan)
    # impute missing module values with 0 after z-score
    Xz = Xz.fillna(0.0)

    scores = {}
    for m, feats in modules.items():
        feats = [f for f in feats if f in Xz.columns]
        if len(feats) == 0:
            continue
        scores[m] = Xz[feats].mean(axis=1)
    return pd.DataFrame(scores, index=X.index)


def compute_module_eigenscores(X: pd.DataFrame, modules: Dict[str, List[str]], seed: int = 1) -> pd.DataFrame:
    """
    module eigenscore = PC1 of z-scored features within module
    """
    Xz = (X - X.mean(axis=0)) / X.std(axis=0).replace(0, np.nan)
    Xz = Xz.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = {}
    for m, feats in modules.items():
        feats = [f for f in feats if f in Xz.columns]
        if len(feats) < 2:
            out[f"Eig{m}"] = Xz[feats].mean(axis=1)
            continue
        pca = PCA(n_components=1, random_state=seed)
        pc1 = pca.fit_transform(Xz[feats].values).ravel()
        out[f"Eig{m}"] = pc1
    return pd.DataFrame(out, index=X.index)


def plot_module_heatmap(scores: pd.DataFrame, y: pd.Series, out_path: str, sort_by: str = "group_then_pc1") -> None:
    """
    Compact heatmap: samples x modules.
    Sorting:
      - group_then_pc1: sort by group then mean score
    """
    df = scores.copy()
    df["__group__"] = y.values
    mods = [c for c in df.columns if c != "__group__"]

    if sort_by == "group_then_mean":
        df["__key__"] = df[mods].mean(axis=1)
        df = df.sort_values(["__group__", "__key__"], ascending=[True, False])
    else:
        df["__key__"] = df[mods].mean(axis=1)
        df = df.sort_values(["__group__", "__key__"], ascending=[True, False])

    mat = df[mods].values
    plt.figure(figsize=(6.2, 4.8))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="Module score (z)")
    plt.yticks([])
    plt.xticks(np.arange(len(mods)), mods)
    plt.xlabel("Modules")
    plt.title("Module-like signature scores (samples × modules)")
    savefig(out_path)


def plot_top_module_group_panels(
    scores: pd.DataFrame,
    y: pd.Series,
    out_path: str,
    groups_order: List[str],
    top_k: int = 5,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Plot top_k modules (by Kruskal/ANOVA proxy) with compact per-module group comparison.
    Returns stats table (module, p, q, effect sizes).
    """
    rng = check_random_state(seed)
    mods = list(scores.columns)

    rows = []
    for m in mods:
        vals = scores[m]
        # group arrays
        gvals = [vals[y == g].values for g in groups_order if np.any(y == g)]
        # Kruskal as robust test (if available)
        p = np.nan
        if kruskal is not None and len(gvals) >= 2 and all(len(v) >= 2 for v in gvals):
            try:
                p = float(kruskal(*gvals).pvalue)
            except Exception:
                p = np.nan
        # effect sizes for HS-NC and ZS-NC when possible
        eff_hs = np.nan
        eff_zs = np.nan
        if "HS" in groups_order and "NC" in groups_order and np.any(y == "HS") and np.any(y == "NC"):
            eff_hs = _cliffs_delta(vals[y == "HS"].values, vals[y == "NC"].values)
        if "ZS" in groups_order and "NC" in groups_order and np.any(y == "ZS") and np.any(y == "NC"):
            eff_zs = _cliffs_delta(vals[y == "ZS"].values, vals[y == "NC"].values)

        rows.append({"module": m, "p_kw": p, "cliffs_HS_vs_NC": eff_hs, "cliffs_ZS_vs_NC": eff_zs})

    stat = pd.DataFrame(rows).sort_values("p_kw", na_position="last")
    stat["q_kw"] = _fdr_bh(stat["p_kw"].values)

    top = stat.head(top_k)["module"].tolist()
    # plot as small multiples in one row (1 x top_k)
    n = len(top)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.3), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, top):
        vals = scores[m]
        xs = np.arange(len(groups_order))
        for i, g in enumerate(groups_order):
            v = vals[y == g].values
            if len(v) == 0:
                continue
            # jitter scatter
            jitter = (rng.rand(len(v)) - 0.5) * 0.18
            ax.scatter(np.full(len(v), i) + jitter, v, s=25, alpha=0.85)
            # median + IQR
            med = np.median(v)
            q1, q3 = np.percentile(v, [25, 75])
            ax.plot([i - 0.18, i + 0.18], [med, med], lw=2)
            ax.plot([i, i], [q1, q3], lw=2)

        ax.set_xticks(xs)
        ax.set_xticklabels(groups_order, rotation=45)
        ax.set_title(m)
        ax.axhline(0, color="gray", lw=1, alpha=0.5)

    fig.suptitle("Top module scores by group (compact)")
    fig.text(0.04, 0.5, "Module score (z)", va="center", rotation=90)
    savefig(out_path)
    return stat


def partial_corr_from_precision(P: np.ndarray) -> np.ndarray:
    """Convert precision matrix to partial correlation matrix."""
    D = np.sqrt(np.outer(np.diag(P), np.diag(P)))
    pc = -P / D
    np.fill_diagonal(pc, 1.0)
    return pc


def plot_partialcorr_network(
    X: pd.DataFrame,
    features: List[str],
    meta_feat: pd.DataFrame,
    out_path: str,
    top_edges: int = 20,
    seed: int = 1,
) -> None:
    """
    Partial correlation network for selected features.
    - Node colors are DISCRETE by 'class' in meta_feat (if present).
    - Only plot top |partial corr| edges to avoid poster-like clutter.
    """
    feats = [f for f in features if f in X.columns]
    if len(feats) < 4:
        warnings.warn("Not enough features for network plot.")
        return

    Xs = X[feats].copy()
    # z-score and impute
    Xs = (Xs - Xs.mean(axis=0)) / Xs.std(axis=0).replace(0, np.nan)
    Xs = Xs.replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    if not _GLASSO_OK:
        warnings.warn("Graphical model tools not available; skipping network.")
        return

    # fit shrinkage precision (GraphicalLassoCV if possible, else LedoitWolf inverse)
    prec = None
    try:
        gl = GraphicalLassoCV()
        gl.fit(Xs)
        prec = gl.precision_
    except Exception:
        try:
            lw = LedoitWolf().fit(Xs)
            cov = lw.covariance_
            prec = np.linalg.pinv(cov)
        except Exception:
            prec = None
    if prec is None:
        warnings.warn("Failed to estimate precision.")
        return

    pc = partial_corr_from_precision(prec)
    # collect edges
    edges = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            edges.append((i, j, pc[i, j]))
    edges = sorted(edges, key=lambda t: abs(t[2]), reverse=True)
    edges = edges[: min(top_edges, len(edges))]

    # Build graph via networkx if available
    try:
        import networkx as nx
    except Exception:
        warnings.warn("networkx not installed; skipping network plot.")
        return

    G = nx.Graph()
    for i, f in enumerate(feats):
        G.add_node(i, feature=f)

    for i, j, w in edges:
        G.add_edge(i, j, weight=w)

    # Node labeling: prefer short name if available
    name_map = {f: f for f in feats}
    class_map = {f: "NA" for f in feats}

    if meta_feat is not None and not meta_feat.empty:
        cols = {c.lower(): c for c in meta_feat.columns}
        fcol = cols.get("feature", None)
        ncol = cols.get("name", None) or cols.get("metabolite", None)
        ccol = cols.get("class", None)
        if fcol is not None:
            mdf = meta_feat.copy()
            mdf[fcol] = mdf[fcol].astype(str)
            mdf = mdf.set_index(fcol, drop=False)
            for f in feats:
                if f in mdf.index:
                    if ncol is not None and pd.notna(mdf.loc[f, ncol]):
                        name_map[f] = str(mdf.loc[f, ncol])
                    if ccol is not None and pd.notna(mdf.loc[f, ccol]):
                        class_map[f] = str(mdf.loc[f, ccol])

    # discrete palette by class
    classes = sorted(set(class_map.values()))
    # use tab10-like cycle
    palette = plt.cm.get_cmap("tab10", max(3, len(classes)))
    color_by_class = {c: palette(i) for i, c in enumerate(classes)}

    node_colors = [color_by_class[class_map[f]] for f in feats]

    # layout
    pos = nx.spring_layout(G, seed=seed, k=None)

    plt.figure(figsize=(6.0, 5.4))
    # edges
    weights = [abs(G[u][v]["weight"]) for u, v in G.edges()]
    if len(weights) == 0:
        return
    wmin, wmax = min(weights), max(weights)
    widths = [1.0 + 4.0 * (w - wmin) / (wmax - wmin + 1e-9) for w in weights]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.6, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, linewidths=1.2, edgecolors="white")

    labels = {i: name_map[f] for i, f in enumerate(feats)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    # legend
    handles = []
    for c in classes:
        handles.append(plt.Line2D([0], [0], marker="o", color="w", label=c,
                                  markerfacecolor=color_by_class[c], markersize=9))
    plt.legend(handles=handles, loc="lower left", fontsize=8, frameon=False)

    plt.title("Top biomarkers partial-correlation network (shrinkage precision)")
    plt.axis("off")
    savefig(out_path)


def plot_module_eigenscore_assoc(
    eig: pd.DataFrame,
    y: pd.Series,
    out_path: str,
    top_k: int = 6,
    seed: int = 1,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """
    For each module eigenscore, compute mean differences (HS-NC, ZS-NC) with stratified bootstrap CI.
    Plot top_k modules by max |effect|.
    """
    rng = check_random_state(seed)
    groups = ["NC", "HS", "ZS"]
    for g in groups:
        if not np.any(y == g):
            # keep order but handle missing
            pass

    rows = []
    for col in eig.columns:
        v = eig[col]
        # differences
        def boot_diff(g1, g0):
            x1 = v[y == g1].values
            x0 = v[y == g0].values
            if len(x1) < 2 or len(x0) < 2:
                return np.array([])
            # stratified bootstrap within each group
            out = []
            for _ in range(n_boot):
                b1 = rng.choice(x1, size=len(x1), replace=True)
                b0 = rng.choice(x0, size=len(x0), replace=True)
                out.append(b1.mean() - b0.mean())
            return np.array(out)

        diff_hs = np.nan
        ci_hs = (np.nan, np.nan)
        if np.any(y == "HS") and np.any(y == "NC"):
            diff_hs = float(v[y == "HS"].mean() - v[y == "NC"].mean())
            bd = boot_diff("HS", "NC")
            if len(bd) >= 200:
                ci_hs = (float(np.quantile(bd, 0.025)), float(np.quantile(bd, 0.975)))
        diff_zs = np.nan
        ci_zs = (np.nan, np.nan)
        if np.any(y == "ZS") and np.any(y == "NC"):
            diff_zs = float(v[y == "ZS"].mean() - v[y == "NC"].mean())
            bd = boot_diff("ZS", "NC")
            if len(bd) >= 200:
                ci_zs = (float(np.quantile(bd, 0.025)), float(np.quantile(bd, 0.975)))

        score = np.nanmax([abs(diff_hs) if np.isfinite(diff_hs) else np.nan,
                           abs(diff_zs) if np.isfinite(diff_zs) else np.nan])
        rows.append({
            "module": col,
            "diff_HS_NC": diff_hs, "ci_HS_lo": ci_hs[0], "ci_HS_hi": ci_hs[1],
            "diff_ZS_NC": diff_zs, "ci_ZS_lo": ci_zs[0], "ci_ZS_hi": ci_zs[1],
            "rank_score": score
        })

    tab = pd.DataFrame(rows).sort_values("rank_score", ascending=False, na_position="last")
    show = tab.head(top_k).copy()
    # Plot grouped bars: two bars per module
    plt.figure(figsize=(1.6 * len(show) + 2.6, 4.2))
    x = np.arange(len(show))
    w = 0.35
    plt.axhline(0, color="gray", lw=1)

    # HS-NC
    hs = show["diff_HS_NC"].values.astype(float)
    hs_lo = show["ci_HS_lo"].values.astype(float)
    hs_hi = show["ci_HS_hi"].values.astype(float)
    hs_err = np.vstack([hs - hs_lo, hs_hi - hs])
    hs_err[:, ~np.isfinite(hs_err).all(axis=0)] = np.nan
    plt.bar(x - w/2, hs, width=w, yerr=hs_err, capsize=4, label="HS - NC")

    # ZS-NC
    zs = show["diff_ZS_NC"].values.astype(float)
    zs_lo = show["ci_ZS_lo"].values.astype(float)
    zs_hi = show["ci_ZS_hi"].values.astype(float)
    zs_err = np.vstack([zs - zs_lo, zs_hi - zs])
    zs_err[:, ~np.isfinite(zs_err).all(axis=0)] = np.nan
    plt.bar(x + w/2, zs, width=w, yerr=zs_err, capsize=4, label="ZS - NC")

    plt.xticks(x, show["module"].astype(str), rotation=45, ha="right")
    plt.ylabel("Eigenscore difference (mean, bootstrap 95% CI)")
    plt.title("Module eigenscore association with subtype")
    plt.legend(fontsize=9)
    savefig(out_path)
    return tab


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Input Excel file containing data matrix.")
    ap.add_argument("--sheet", default=None, help="Sheet name for the data matrix (default: first non-meta sheet).")
    ap.add_argument("--label_col", default="Group", help="Label column name.")
    ap.add_argument("--id_col", default=None, help="Optional sample-id column to drop from features.")
    ap.add_argument("--drop", nargs="*", default=[], help="Sample IDs to drop (match index strings).")

    ap.add_argument("--out_dir", default="out_figs_v5", help="Output directory for figures/tables.")
    ap.add_argument("--seed", type=int, default=1, help="Random seed.")
    ap.add_argument("--outer_splits", type=int, default=8, help="Outer CV splits (StratifiedKFold).")
    ap.add_argument("--inner_splits", type=int, default=4, help="Inner CV splits for tuning.")
    ap.add_argument("--do_nested_tuning", action="store_true", help="Enable nested hyperparameter tuning inside each outer fold.")
    ap.add_argument("--do_feature_select", action="store_true", help="Enable per-fold feature selection (stability freq).")

    ap.add_argument("--oversample", choices=["none", "smote", "adasyn", "noise"], default="none", help="Training-only oversampling/augmentation method.")
    ap.add_argument("--augment_n", type=int, default=0, help="For noise augmentation: how many synthetic samples to add per fold.")
    ap.add_argument("--noise_scale", type=float, default=0.10, help="Noise strength relative to feature std (only for --oversample noise).")

    ap.add_argument("--cal_method", choices=["none", "platt", "isotonic"], default="none", help="In-fold probability calibration method (OVR).")
    ap.add_argument("--n_bins_cal", type=int, default=3, help="Calibration bins (small-N friendly: 3-4).")
    ap.add_argument("--min_bin_n", type=int, default=2, help="Min samples per calibration bin (small-N friendly: 2).")
    ap.add_argument("--cal_strategy", choices=["uniform", "quantile", "adaptive"], default="adaptive", help="Calibration binning strategy.")

    ap.add_argument("--n_boot", type=int, default=3000, help="Bootstrap iterations for CIs (skip invalid draws).")
    ap.add_argument("--min_boot_valid", type=int, default=400, help="Minimum valid bootstrap draws; else fall back to fold-level CI.")

    # Mechanistic settings
    ap.add_argument("--n_modules", type=int, default=8, help="Number of unsupervised modules for signature score.")
    ap.add_argument("--top_modules", type=int, default=5, help="Number of top modules to show in group panels.")
    ap.add_argument("--top_edges", type=int, default=20, help="Top edges to keep in partial-corr network.")
    ap.add_argument("--top_eig_modules", type=int, default=6, help="Top eigenscore modules to show in association bar plot.")

    # DCA
    ap.add_argument("--dca_binary", default="HSZS_vs_NC", choices=["off", "HSZS_vs_NC", "ZS_vs_others", "HS_vs_others"], help="Binary DCA task for clinical interpretability.")
    args = ap.parse_args()

    _mkdir(args.out_dir)
    seed = args.seed

    # ----- Load data
    X_df, y_str, meta_feat = load_xlsx_matrix(args.xlsx, label_col=args.label_col, id_col=args.id_col, sheet=args.sheet)
    X_df, y_str = apply_drop_samples(X_df, y_str, args.drop)

    # basic cleanup
    X_df = X_df.loc[:, X_df.notna().sum(axis=0) >= 2]  # remove near-empty features

    # ----- Encode labels
    le = LabelEncoder()
    y_int = le.fit_transform(y_str.values)
    classes = list(le.classes_)
    n_classes = len(classes)

    # Sanity: ensure outer splits feasible
    counts = pd.Series(y_int).value_counts().to_dict()
    min_count = min(counts.values())
    if args.outer_splits > min_count:
        warnings.warn(
            f"outer_splits={args.outer_splits} > min class count={min_count}. "
            f"Reducing outer_splits to {max(2, min_count)} to avoid folds with missing classes."
        )
        args.outer_splits = max(2, min_count)

    # ----- Outer CV
    outer = StratifiedKFold(n_splits=args.outer_splits, shuffle=True, random_state=seed)

    P_oof = np.zeros((len(y_int), n_classes), dtype=float)
    fold_macro_auc = []
    fold_macro_auprc = []
    feature_selected_counts = pd.Series(0, index=X_df.columns.astype(str), dtype=float)

    # For robust CI fallback
    fold_auc_by_class = {c: [] for c in classes}
    fold_ap_by_class = {c: [] for c in classes}

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for fold, (tr, te) in enumerate(outer.split(X_df.values, y_int), start=1):
        Xtr, Xte = X_df.values[tr], X_df.values[te]
        ytr, yte = y_int[tr], y_int[te]

        # training-only oversampling / augmentation
        if args.oversample in ("smote", "adasyn"):
            Xtr2, ytr2 = oversample_train_only(Xtr, ytr, method=args.oversample, seed=seed + fold)
        else:
            Xtr2, ytr2 = Xtr, ytr

        if args.oversample == "noise" and args.augment_n > 0:
            Xtr2, ytr2 = augment_noise_by_feature_variance(Xtr2, ytr2, n_new=args.augment_n, noise_scale=args.noise_scale, seed=seed + 1000 + fold)

        pipe = build_pipeline(seed=seed + fold, do_feature_select=args.do_feature_select)

        if args.do_nested_tuning:
            pipe = tune_hyperparams(pipe, Xtr2, ytr2, seed=seed + 10 * fold, inner_splits=min(args.inner_splits, max(2, min(pd.Series(ytr2).value_counts().min(), args.inner_splits))))
        else:
            pipe.fit(Xtr2, ytr2)

        # Feature selection stability (count selected features)
        if args.do_feature_select and "select" in pipe.named_steps:
            try:
                selector = pipe.named_steps["select"]
                mask = selector.get_support()
                selected = X_df.columns.astype(str)[mask]
                feature_selected_counts.loc[selected] += 1.0
            except Exception:
                pass

        # Predict probabilities (uncalibrated)
        try:
            proba_te = pipe.predict_proba(Xte)
        except Exception:
            # If classifier lacks predict_proba, use decision_function + softmax
            scores = pipe.decision_function(Xte)
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
            proba_te = _softmax(scores, axis=1)

        # In-fold OVR calibration (train-only, evaluated on test fold)
        if args.cal_method != "none":
            # Need training probabilities too
            try:
                proba_tr = pipe.predict_proba(Xtr)
            except Exception:
                scores = pipe.decision_function(Xtr)
                if scores.ndim == 1:
                    scores = np.vstack([-scores, scores]).T
                proba_tr = _softmax(scores, axis=1)

            proba_cal = np.zeros_like(proba_te)
            for k in range(n_classes):
                ytr_bin = (ytr == k).astype(int)
                cal = fit_calibrator_binary(args.cal_method, proba_tr[:, k], ytr_bin, seed=seed + 999 + fold + k)
                proba_cal[:, k] = cal(proba_te[:, k])
            # renormalize to simplex
            s = proba_cal.sum(axis=1, keepdims=True)
            s = np.where(s > 0, s, 1.0)
            proba_te = proba_cal / s

        P_oof[te] = proba_te

        # Fold-level metrics
        try:
            fold_auc = roc_auc_score(yte, proba_te, multi_class="ovr", average="macro")
        except Exception:
            fold_auc = np.nan
        fold_macro_auc.append(fold_auc)

        # Fold-level macro AUPRC (mean AP across classes)
        aps = []
        for k in range(n_classes):
            yk = (yte == k).astype(int)
            if yk.min() == yk.max():
                continue
            try:
                aps.append(average_precision_score(yk, proba_te[:, k]))
            except Exception:
                continue
        fold_macro_auprc.append(float(np.mean(aps)) if len(aps) else np.nan)

        # per-class fold AUC/AP (for fallback CI)
        for k, name in enumerate(classes):
            yk = (yte == k).astype(int)
            if yk.min() == yk.max():
                continue
            try:
                fold_auc_by_class[name].append(roc_auc_score(yk, proba_te[:, k]))
            except Exception:
                pass
            try:
                fold_ap_by_class[name].append(average_precision_score(yk, proba_te[:, k]))
            except Exception:
                pass

        print(f"[fold {fold:02d}/{args.outer_splits}] macro-AUROC={fold_auc:.3f} macro-AUPRC={fold_macro_auprc[-1]:.3f}  n_test={len(te)}")

    # ----- Save core metrics table
    metrics_rows = []
    macro_auc_full = float(roc_auc_score(y_int, P_oof, multi_class="ovr", average="macro")) if n_classes > 1 else np.nan
    aps_full = []
    for k in range(n_classes):
        yk = (y_int == k).astype(int)
        if yk.min() == yk.max():
            continue
        aps_full.append(average_precision_score(yk, P_oof[:, k]))
    macro_auprc_full = float(np.mean(aps_full)) if len(aps_full) else np.nan

    # CI via stratified bootstrap (multiclass)
    def _macro_auc_fn(yb, Pb):
        return roc_auc_score(yb, Pb, multi_class="ovr", average="macro")

    def _macro_auprc_fn(yb, Pb):
        # macro mean AP
        apv = []
        for kk in range(n_classes):
            yk = (yb == kk).astype(int)
            if yk.min() == yk.max():
                continue
            apv.append(average_precision_score(yk, Pb[:, kk]))
        return float(np.mean(apv)) if len(apv) else np.nan

    boot_macro_auc = _stratified_bootstrap_multiclass_metric(y_int, P_oof, _macro_auc_fn, n_boot=args.n_boot, seed=seed + 123)
    boot_macro_auprc = _stratified_bootstrap_multiclass_metric(y_int, P_oof, _macro_auprc_fn, n_boot=args.n_boot, seed=seed + 456)

    def ci_from_boot_or_folds(boot: np.ndarray, folds: List[float]):
        boot = boot[np.isfinite(boot)]
        folds = np.asarray([v for v in folds if np.isfinite(v)], float)
        if len(boot) >= args.min_boot_valid:
            return (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), "bootstrap")
        if len(folds) >= 3:
            return (float(np.quantile(folds, 0.025)), float(np.quantile(folds, 0.975)), "fold-quantile")
        return (np.nan, np.nan, "NA")

    macro_auc_ci_lo, macro_auc_ci_hi, macro_auc_ci_src = ci_from_boot_or_folds(boot_macro_auc, fold_macro_auc)
    macro_auprc_ci_lo, macro_auprc_ci_hi, macro_auprc_ci_src = ci_from_boot_or_folds(boot_macro_auprc, fold_macro_auprc)

    metrics_rows.append({
        "metric": "macro_AUROC",
        "value": macro_auc_full,
        "ci_lo": macro_auc_ci_lo,
        "ci_hi": macro_auc_ci_hi,
        "ci_source": macro_auc_ci_src,
        "n": len(y_int),
    })
    metrics_rows.append({
        "metric": "macro_AUPRC",
        "value": macro_auprc_full,
        "ci_lo": macro_auprc_ci_lo,
        "ci_hi": macro_auprc_ci_hi,
        "ci_source": macro_auprc_ci_src,
        "n": len(y_int),
    })

    # Per-class AUC CIs (binary stratified bootstrap; fallback to fold-quantile)
    auc_ci_by_class = {}
    for k, name in enumerate(classes):
        yk = (y_int == k).astype(int)
        pk = P_oof[:, k]
        if yk.min() == yk.max():
            continue
        boot = _stratified_bootstrap_binary_metric(
            yk, pk,
            metric_fn=lambda yy, pp: roc_auc_score(yy, pp),
            n_boot=args.n_boot,
            seed=seed + 100 + k,
            min_pos=1,
            min_neg=1,
        )
        folds = fold_auc_by_class.get(name, [])
        lo, hi, src = ci_from_boot_or_folds(boot, folds)
        auc_point = float(roc_auc_score(yk, pk))
        auc_ci_by_class[name] = (auc_point, lo, hi)
        metrics_rows.append({
            "metric": f"AUROC_{name}_OVR",
            "value": auc_point,
            "ci_lo": lo,
            "ci_hi": hi,
            "ci_source": src,
            "n_pos": int(yk.sum()),
            "n_neg": int((1 - yk).sum()),
        })
        # AUPRC per class
        try:
            ap = float(average_precision_score(yk, pk))
        except Exception:
            ap = np.nan
        boot_ap = _stratified_bootstrap_binary_metric(
            yk, pk,
            metric_fn=lambda yy, pp: average_precision_score(yy, pp),
            n_boot=args.n_boot,
            seed=seed + 200 + k,
            min_pos=1,
            min_neg=1,
        )
        folds_ap = fold_ap_by_class.get(name, [])
        lo2, hi2, src2 = ci_from_boot_or_folds(boot_ap, folds_ap)
        metrics_rows.append({
            "metric": f"AUPRC_{name}_OVR",
            "value": ap,
            "ci_lo": lo2,
            "ci_hi": hi2,
            "ci_source": src2,
            "n_pos": int(yk.sum()),
            "n_neg": int((1 - yk).sum()),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(args.out_dir, "OOF_metrics_with_CI.csv"), index=False)

    # ----- Figures: ExtData AUC distribution, Feature frequency
    plot_auc_distribution(fold_macro_auc, os.path.join(args.out_dir, "ExtData_AUC_distribution_OOF.png"), title=f"CV stability (n={args.outer_splits} outer folds)")
    if args.do_feature_select:
        freq = feature_selected_counts / float(args.outer_splits)
        plot_feature_frequency(freq, os.path.join(args.out_dir, "ExtData_FeatureFreq_Top20_OOF.png"), top_n=20)

    # ----- Figures: ROC, PR, Calibration
    plot_multiclass_roc(y_int, P_oof, classes, os.path.join(args.out_dir, "Fig2A_ROC_multiclass_OVR_OOF.png"), auc_ci=auc_ci_by_class)
    plot_multiclass_pr(y_int, P_oof, classes, os.path.join(args.out_dir, "Fig2A_PR_multiclass_OVR_OOF.png"))
    cal_metrics = plot_calibration_ovr(y_int, P_oof, classes, os.path.join(args.out_dir, "Fig2B_Calibration_multiclass_OVR_OOF.png"),
                                       n_bins=args.n_bins_cal, min_bin_n=args.min_bin_n, strategy=args.cal_strategy)

    # Save per-class calibration metrics table
    if cal_metrics:
        rows = []
        for k, v in cal_metrics.items():
            rows.append({
                "class": k,
                "Brier": v.brier,
                "ECE": v.ece,
                "cal_slope": v.slope,
                "cal_intercept": v.intercept,
                "n": v.n
            })
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "OOF_calibration_metrics.csv"), index=False)

    # ----- DCA (binary, more defensible than multiclass)
    if args.dca_binary != "off":
        if args.dca_binary == "HSZS_vs_NC":
            y_bin = np.isin(y_str.values, ["HS", "ZS"]).astype(int)
            p_bin = P_oof[:, classes.index("HS")] + P_oof[:, classes.index("ZS")] if "HS" in classes and "ZS" in classes else P_oof.max(axis=1)
            title = "Decision curve analysis (HS/ZS vs NC, out-of-fold)"
        elif args.dca_binary == "ZS_vs_others":
            y_bin = (y_str.values == "ZS").astype(int)
            p_bin = P_oof[:, classes.index("ZS")] if "ZS" in classes else P_oof.max(axis=1)
            title = "Decision curve analysis (ZS vs others, out-of-fold)"
        else:  # HS_vs_others
            y_bin = (y_str.values == "HS").astype(int)
            p_bin = P_oof[:, classes.index("HS")] if "HS" in classes else P_oof.max(axis=1)
            title = "Decision curve analysis (HS vs others, out-of-fold)"
        plot_dca_binary(y_bin, p_bin, os.path.join(args.out_dir, "Fig2C_DCA_binary_OOF.png"), title=title)

    # ----- Mechanistic: module scores, network, eigenscore association
    modules = build_modules_unsupervised(X_df, n_modules=args.n_modules, seed=seed)
    module_scores = compute_module_scores(X_df, modules)
    module_scores.to_csv(os.path.join(args.out_dir, "Module_scores.csv"))

    plot_module_heatmap(module_scores, y_str, os.path.join(args.out_dir, "Fig4A_ModuleScore_heatmap.png"))
    stat_tab = plot_top_module_group_panels(module_scores, y_str, os.path.join(args.out_dir, "Fig4B_ModuleScore_top_panels.png"),
                                            groups_order=[g for g in ["NC", "HS", "ZS"] if g in set(y_str.values)],
                                            top_k=args.top_modules, seed=seed)
    stat_tab.to_csv(os.path.join(args.out_dir, "Module_score_stats.csv"), index=False)

    # Partial correlation network on top selected biomarkers (or fallback to top variance)
    if args.do_feature_select and args.outer_splits > 0:
        freq = (feature_selected_counts / float(args.outer_splits)).sort_values(ascending=False)
        top_feats = freq.head(15).index.astype(str).tolist()
    else:
        # fallback: top variance features
        top_feats = X_df.var(axis=0).sort_values(ascending=False).head(15).index.astype(str).tolist()

    plot_partialcorr_network(
        X_df, top_feats, meta_feat, os.path.join(args.out_dir, "Fig5A_TopBiomarker_partialcorr_network.png"),
        top_edges=args.top_edges, seed=seed
    )

    eig = compute_module_eigenscores(X_df, modules, seed=seed)
    eig.to_csv(os.path.join(args.out_dir, "Module_eigenscores.csv"))
    eig_assoc = plot_module_eigenscore_assoc(eig, y_str, os.path.join(args.out_dir, "Fig5B_ModuleEigenscore_assoc_bar.png"),
                                             top_k=args.top_eig_modules, seed=seed, n_boot=2000)
    eig_assoc.to_csv(os.path.join(args.out_dir, "Module_eigenscore_assoc_table.csv"), index=False)

    print("\nDone.")
    print(f"Outputs saved to: {args.out_dir}")
    print("Key outputs:")
    print(" - Fig2A_ROC_multiclass_OVR_OOF.png")
    print(" - Fig2B_Calibration_multiclass_OVR_OOF.png (now non-empty, with Brier/ECE)")
    print(" - Fig2C_DCA_binary_OOF.png (defensible binary DCA)")
    print(" - Fig4A_ModuleScore_heatmap.png + Fig4B_ModuleScore_top_panels.png")
    print(" - Fig5A_TopBiomarker_partialcorr_network.png + Fig5B_ModuleEigenscore_assoc_bar.png")
    print(" - OOF_metrics_with_CI.csv (CI source indicated)")
    print(" - OOF_calibration_metrics.csv")

if __name__ == "__main__":
    main()
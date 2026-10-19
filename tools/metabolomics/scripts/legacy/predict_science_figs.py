#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict_science_figs.py (v4)

A-line (Prediction, strict OOF):
- Multiclass ROC (OVR) + bootstrap CI
- Calibration (OVR) with binomial bootstrap CI
- Decision Curve Analysis (OVR)
- Nested CV stability (AUC distribution + feature selection frequency)
- Fig S1C + Table S1 Part II: log2FC concordance (Full vs Sensitivity)

B-line (Mechanistic / biology, robust but not exaggerated):
4) Module/signature scores:
   - Build 5-10 modules from features (data-driven correlation clustering)
   - Module score = mean(z-scored features) per module; also eigenscore = PC1
   - Heatmap (samples x modules) + violin per group
5) Top biomarkers partial-correlation network + module eigenscore association
6) Pathway evidence bars (optional; requires compound↔pathway mapping table)

Key design:
- Synthetic augmentation (SMOTE/ADASYN/noise) is TRAIN-only for A-line.
- For B-line, augmentation can be used ONLY to stabilize module/network estimation;
  plots default to REAL samples unless --mech_plot_augmented is set.
"""

import os
import re
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

# ----------------------------
# optional deps
# ----------------------------
IMBLEARN_OK = True
try:
    from imblearn.over_sampling import SMOTE, ADASYN
except Exception:
    IMBLEARN_OK = False

SHAP_OK = True
try:
    import shap  # noqa
except Exception:
    SHAP_OK = False

NETWORKX_OK = True
try:
    import networkx as nx
except Exception:
    NETWORKX_OK = False

# scipy optional (for clustering)
SCIPY_OK = True
try:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from scipy.stats import ttest_ind, pearsonr, spearmanr
except Exception:
    SCIPY_OK = False

# ----------------------------
# utils
# ----------------------------
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def find_sheet(xl: pd.ExcelFile, candidates):
    for s in candidates:
        if s in xl.sheet_names:
            return s
    return xl.sheet_names[0]

def safe_str(x, fallback="UnknownFeature"):
    if x is None:
        return fallback
    try:
        if isinstance(x, float) and np.isnan(x):
            return fallback
    except Exception:
        pass
    s = str(x)
    if s.strip() == "" or s.strip().lower() == "nan":
        return fallback
    return s

def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    q = np.empty(n, float)
    q[order] = p[order] * n / ranks
    q[order[::-1]] = np.minimum.accumulate(q[order[::-1]])
    return np.clip(q, 0, 1)

def bootstrap_ci(y_true_bin, y_prob, fn, n_boot=400, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true_bin)
    idx = np.arange(n)
    stats = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=n, replace=True)
        try:
            stats.append(fn(y_true_bin[b], y_prob[b]))
        except Exception:
            continue
    if len(stats) < 20:
        return np.array([np.nan, np.nan])
    stats = np.array(stats)
    return np.percentile(stats, [2.5, 97.5])

def net_benefit_binary(y_true, y_prob, pt):
    y_pred = (y_prob >= pt).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)
    return (tp / n) - (fp / n) * (pt / (1 - pt))

def treat_all_nb(prevalence, pt):
    return prevalence - (1 - prevalence) * (pt / (1 - pt))

def _min_class_count(y):
    y = np.asarray(y)
    _, cnt = np.unique(y, return_counts=True)
    return int(cnt.min())

def pick_cv_params(y, target_outer_folds=50, max_splits=5, inner_max_splits=4):
    minc = _min_class_count(y)
    if minc < 2:
        raise ValueError(f"Min class count={minc} < 2, cannot do stratified CV.")
    n_splits_outer = min(max_splits, minc)
    repeats = max(1, int(math.ceil(target_outer_folds / n_splits_outer)))
    inner_splits = min(inner_max_splits, max(2, minc))
    return n_splits_outer, repeats, inner_splits

# ----------------------------
# augmentation (TRAIN-only for A-line)
# ----------------------------
def classwise_noise_augment(X, y, target_n=None, seed=0, scale=0.10):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    n, p = X.shape
    if (target_n is None) or (target_n <= n):
        return X, y

    classes, counts = np.unique(y, return_counts=True)
    extra = int(target_n - n)

    frac = counts / counts.sum()
    extra_per = np.floor(extra * frac).astype(int)
    rem = extra - extra_per.sum()
    for i in range(rem):
        extra_per[i % len(extra_per)] += 1

    X_new = [X]
    y_new = [y]

    for cls, k in zip(classes, extra_per):
        if k <= 0:
            continue
        idx = np.where(y == cls)[0]
        if idx.size == 0:
            continue

        Xc = X[idx, :]
        sd = np.nanstd(Xc, axis=0, ddof=0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1e-6)

        pick = rng.integers(0, Xc.shape[0], size=k)
        base = Xc[pick, :]
        noise = rng.normal(0.0, sd * scale, size=base.shape)
        Xs = base + noise
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

        X_new.append(Xs)
        y_new.append(np.array([cls] * k, dtype=y.dtype))

    return np.vstack(X_new), np.concatenate(y_new)

def _safe_k_neighbors(y_tr, default=5):
    _, cnt = np.unique(y_tr, return_counts=True)
    minc = int(cnt.min())
    return max(1, min(default, minc - 1))

def oversample_train(X_tr, y_tr, method="none", seed=0, augment_n=0, noise_scale=0.10):
    method = (method or "none").lower()
    X_tr = np.asarray(X_tr, float)
    y_tr = np.asarray(y_tr)

    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)

    if method == "none" and (augment_n is None or augment_n <= 0):
        return X_tr, y_tr, "none"

    if method == "noise":
        if augment_n and augment_n > len(y_tr):
            X_aug, y_aug = classwise_noise_augment(X_tr, y_tr, target_n=augment_n, seed=seed, scale=noise_scale)
            return X_aug, y_aug, f"noise(var-based,scale={noise_scale})"
        return X_tr, y_tr, "noise(no-op)"

    if method in ["smote", "adasyn"] and IMBLEARN_OK:
        try:
            k = _safe_k_neighbors(y_tr, default=5)
            if method == "smote":
                sampler = SMOTE(random_state=seed, k_neighbors=k)
            else:
                sampler = ADASYN(random_state=seed, n_neighbors=k)
            X_res, y_res = sampler.fit_resample(X_tr, y_tr)
            X_res = np.nan_to_num(X_res, nan=0.0, posinf=0.0, neginf=0.0)

            if augment_n and augment_n > len(y_res):
                X_res, y_res = classwise_noise_augment(
                    X_res, y_res, target_n=augment_n, seed=seed + 999, scale=noise_scale
                )
                return X_res, y_res, f"{method}+noise_fill"
            return X_res, y_res, method
        except Exception:
            pass

    if augment_n and augment_n > len(y_tr):
        X_aug, y_aug = classwise_noise_augment(X_tr, y_tr, target_n=augment_n, seed=seed, scale=noise_scale)
        return X_aug, y_aug, f"noise_fallback(scale={noise_scale})"

    return X_tr, y_tr, "none"

# ----------------------------
# B-line augmentation (mechanism only)
# ----------------------------
def augment_for_mechanism(X, y, method="none", target_n=100, seed=0, noise_scale=0.10):
    """
    Used ONLY for module/network estimation (not for prediction evaluation).
    """
    if target_n is None or target_n <= 0 or target_n <= len(y):
        return X, y, "none"
    method = (method or "none").lower()

    if method == "none":
        return X, y, "none"

    if method in ["smote", "adasyn"] and IMBLEARN_OK:
        try:
            k = _safe_k_neighbors(y, default=5)
            if method == "smote":
                sampler = SMOTE(random_state=seed, k_neighbors=k)
            else:
                sampler = ADASYN(random_state=seed, n_neighbors=k)
            X_res, y_res = sampler.fit_resample(np.nan_to_num(X), y)
            # if still short, fill with noise
            if len(y_res) < target_n:
                X_res, y_res = classwise_noise_augment(X_res, y_res, target_n=target_n, seed=seed+777, scale=noise_scale)
            return X_res, y_res, method
        except Exception:
            # fallback to noise
            X_aug, y_aug = classwise_noise_augment(X, y, target_n=target_n, seed=seed, scale=noise_scale)
            return X_aug, y_aug, "noise_fallback"

    # noise
    X_aug, y_aug = classwise_noise_augment(X, y, target_n=target_n, seed=seed, scale=noise_scale)
    return X_aug, y_aug, f"noise(scale={noise_scale})"

# ----------------------------
# DE / sensitivity (HS vs NC)
# ----------------------------
def diff_stats(df_wide, group_a, group_b, sample_to_group):
    if not SCIPY_OK:
        raise RuntimeError("scipy is required for DE stats (ttest_ind). Please install scipy.")

    samples = [c for c in df_wide.columns if c in sample_to_group]
    A = [s for s in samples if sample_to_group[s] == group_a]
    B = [s for s in samples if sample_to_group[s] == group_b]

    Xa = df_wide[A].to_numpy(float)
    Xb = df_wide[B].to_numpy(float)

    mean_a = np.nanmean(Xa, axis=1)
    mean_b = np.nanmean(Xb, axis=1)
    log2fc = mean_a - mean_b

    p = np.array([
        ttest_ind(Xa[i, :], Xb[i, :], equal_var=False, nan_policy="omit").pvalue
        for i in range(Xa.shape[0])
    ])
    q = bh_fdr(p)

    out = pd.DataFrame({
        "feature": [safe_str(i) for i in df_wide.index],
        "log2FC": log2fc,
        "p": p,
        "q": q
    })
    return out

def concordance_table(full_df, sens_df, topk=20, q_thr=0.05):
    if not SCIPY_OK:
        raise RuntimeError("scipy is required for concordance stats (pearson/spearman). Please install scipy.")

    f = full_df.set_index("feature")
    s = sens_df.set_index("feature")
    common = f.index.intersection(s.index)
    f = f.loc[common]
    s = s.loc[common]

    sig_f = set(f.index[f["q"] < q_thr])
    sig_s = set(s.index[s["q"] < q_thr])
    inter = sig_f & sig_s
    union = sig_f | sig_s
    jacc = (len(inter) / len(union)) if len(union) else np.nan

    r_p = pearsonr(f["log2FC"].values, s["log2FC"].values)[0]
    r_s = spearmanr(f["log2FC"].values, s["log2FC"].values)[0]

    top_f = list(f.loc[f["log2FC"].abs().sort_values(ascending=False).head(topk).index].index)
    top_s = list(s.loc[s["log2FC"].abs().sort_values(ascending=False).head(topk).index].index)
    top_inter = sorted(set(top_f) & set(top_s))
    top_union = set(top_f) | set(top_s)
    top_j = len(set(top_inter)) / len(top_union) if len(top_union) else np.nan

    if len(common) > 0:
        sign_same = np.mean(np.sign(f["log2FC"].values) == np.sign(s["log2FC"].values))
    else:
        sign_same = np.nan

    return {
        "n_common_features": int(len(common)),
        "n_sig_full(q<%.3f)" % q_thr: int(len(sig_f)),
        "n_sig_sens(q<%.3f)" % q_thr: int(len(sig_s)),
        "n_overlap_sig": int(len(inter)),
        "jaccard_sig": float(jacc) if np.isfinite(jacc) else np.nan,
        "pearson_log2FC": float(r_p),
        "spearman_log2FC": float(r_s),
        "sign_concordance_all": float(sign_same) if np.isfinite(sign_same) else np.nan,
        f"top{topk}_overlap": int(len(top_inter)),
        f"top{topk}_jaccard": float(top_j) if np.isfinite(top_j) else np.nan,
        f"top{topk}_overlap_list": ";".join(top_inter[:200]),
    }

# ----------------------------
# plotting helpers (style)
# ----------------------------
def mpl_science_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "legend.frameon": False,
    })

# ----------------------------
# SHAP (optional)
# ----------------------------
def try_shap_outputs(model, X, y, feature_names, outdir, topk=10, seed=0):
    if not SHAP_OK:
        return
    try:
        import shap
        model.fit(X, y)
        Xt = model.named_steps["imputer"].transform(X)
        Xt = model.named_steps["scaler"].transform(Xt)
        clf = model.named_steps["clf"]

        rng = np.random.default_rng(seed)
        bg_idx = rng.choice(np.arange(Xt.shape[0]), size=min(50, Xt.shape[0]), replace=False)
        background = Xt[bg_idx, :]

        explainer = shap.LinearExplainer(clf, background)
        shap_values = explainer.shap_values(Xt)  # list[n_classes]

        abs_mean = np.zeros(Xt.shape[1], float)
        for k in range(len(shap_values)):
            abs_mean += np.mean(np.abs(shap_values[k]), axis=0)
        abs_mean /= max(1, len(shap_values))
        order = np.argsort(abs_mean)[::-1]
        top_idx = order[:min(topk, len(order))]

        plt.figure(figsize=(6.5, 3.6))
        plt.barh([feature_names[i] for i in top_idx][::-1], abs_mean[top_idx][::-1])
        plt.xlabel("Mean |SHAP| (avg across classes)")
        plt.title(f"SHAP summary (Top {len(top_idx)})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "ExtData_SHAP_summary_Top10.png"), dpi=300)
        plt.savefig(os.path.join(outdir, "ExtData_SHAP_summary_Top10.pdf"))
        plt.close()

        for j in top_idx[:3]:
            plt.figure(figsize=(5.2, 4.2))
            sv = np.zeros(Xt.shape[0], float)
            for k in range(len(shap_values)):
                sv += shap_values[k][:, j]
            sv /= max(1, len(shap_values))
            plt.scatter(Xt[:, j], sv, s=16, alpha=0.55)
            plt.xlabel(f"{feature_names[j]} (scaled)")
            plt.ylabel("SHAP (avg across classes)")
            plt.title(f"SHAP dependence: {feature_names[j]}")
            plt.tight_layout()
            fname = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature_names[j])[:80]
            plt.savefig(os.path.join(outdir, f"ExtData_SHAP_depend_{fname}.png"), dpi=300)
            plt.savefig(os.path.join(outdir, f"ExtData_SHAP_depend_{fname}.pdf"))
            plt.close()
    except Exception:
        return

# ----------------------------
# B-line: modules / network / pathway evidence
# ----------------------------
def zscore_cols(X):
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return (X - mu) / sd

def zscore_rows(X):
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return (X - mu) / sd

def build_modules_corr_clustering(X_samples_by_features, feature_names, n_modules=8, corr_method="spearman"):
    """
    Returns module_id per feature index.
    Uses |corr| distance + hierarchical clustering.
    """
    if not SCIPY_OK:
        raise RuntimeError("scipy is required for correlation clustering modules. Please install scipy.")

    X = np.nan_to_num(X_samples_by_features, nan=0.0, posinf=0.0, neginf=0.0)
    # compute correlation across samples between features
    # shape: features x samples
    F = X.T
    if corr_method == "spearman":
        # spearman: rank transform along samples
        ranks = np.apply_along_axis(lambda v: pd.Series(v).rank(method="average").to_numpy(), 1, F)
        C = np.corrcoef(ranks)
    else:
        C = np.corrcoef(F)

    C = np.nan_to_num(C, nan=0.0)
    # distance = 1 - |corr|
    D = 1.0 - np.abs(C)
    np.fill_diagonal(D, 0.0)

    Z = linkage(squareform(D, checks=False), method="average")
    # cut into n_modules
    cl = fcluster(Z, t=n_modules, criterion="maxclust")
    # cl is 1..n_modules
    return cl

def compute_module_scores(X_samples_by_features, module_ids, n_modules):
    """
    module score per sample: mean(zscore(feature across samples) within module)
    module eigenscore per sample: PC1 scores in each module (on zscored features)
    """
    X = np.nan_to_num(X_samples_by_features, nan=0.0, posinf=0.0, neginf=0.0)
    # z-score features across samples (column-wise)
    Xz = zscore_cols(X)

    scores = np.zeros((X.shape[0], n_modules), float)
    eig = np.zeros((X.shape[0], n_modules), float)

    for m in range(1, n_modules + 1):
        idx = np.where(module_ids == m)[0]
        if idx.size == 0:
            scores[:, m - 1] = np.nan
            eig[:, m - 1] = np.nan
            continue
        Xm = Xz[:, idx]
        scores[:, m - 1] = np.mean(Xm, axis=1)

        # eigenscore = PC1 score
        if idx.size >= 2:
            pca = PCA(n_components=1, random_state=0)
            eig[:, m - 1] = pca.fit_transform(Xm).ravel()
        else:
            eig[:, m - 1] = Xm.ravel()

    return scores, eig

def plot_module_heatmap(scores, sample_groups, title, out_png, out_pdf):
    # sort samples by group then by first module score
    order = np.argsort(sample_groups.astype(str))
    S = scores[order]
    G = sample_groups[order]

    # row-wise z-score for heatmap (samples x modules -> normalize per module)
    Hz = zscore_cols(S)

    plt.figure(figsize=(7.6, 6.6))
    im = plt.imshow(Hz, aspect="auto")
    plt.title(title)
    plt.ylabel("Samples")
    plt.xlabel("Modules")
    plt.xticks(np.arange(Hz.shape[1]), [f"M{i+1}" for i in range(Hz.shape[1])], rotation=0)

    # add group separators
    # compute boundaries in sorted G
    bounds = []
    for i in range(1, len(G)):
        if G[i] != G[i-1]:
            bounds.append(i - 0.5)
    for b in bounds:
        plt.axhline(b, color="k", lw=0.6, alpha=0.6)

    cbar = plt.colorbar(im)
    cbar.set_label("Module score (z)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

def plot_module_violin(score_df, out_png, out_pdf, classes):
    # score_df columns: sample, group, M1..Mk
    module_cols = [c for c in score_df.columns if c.startswith("M")]
    k = len(module_cols)

    # compact: one subplot-like figure (no subplots rule not required here, but keep single panel with grouped violins)
    # We'll draw grouped violins manually on one axis to stay "Science compact".
    plt.figure(figsize=(1.2*k + 3.0, 5.0))

    x_positions = []
    data_list = []
    labels = []
    pos = 1
    gap = 1.0

    for mi, mcol in enumerate(module_cols):
        for gi, g in enumerate(classes):
            vals = score_df.loc[score_df["group"] == g, mcol].astype(float).values
            data_list.append(vals)
            x_positions.append(pos)
            labels.append(f"{mcol}\n{g}")
            pos += 1
        pos += gap

    parts = plt.violinplot(data_list, positions=x_positions, showmeans=False, showmedians=True, widths=0.8)
    for pc in parts["bodies"]:
        pc.set_alpha(0.20)

    # overlay points (jitter)
    rng = np.random.default_rng(0)
    for x, vals in zip(x_positions, data_list):
        if len(vals) == 0:
            continue
        jitter = rng.normal(0, 0.07, size=len(vals))
        plt.scatter(np.full_like(vals, x, dtype=float) + jitter, vals, s=12, alpha=0.55)

    plt.xticks(x_positions, labels, rotation=90)
    plt.ylabel("Module score")
    plt.title("Module scores by group")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

def partial_corr_from_precision(P):
    # partial corr = -P_ij / sqrt(P_ii P_jj)
    d = np.sqrt(np.outer(np.diag(P), np.diag(P)))
    d = np.where(d > 0, d, 1e-9)
    pc = -P / d
    np.fill_diagonal(pc, 1.0)
    return pc

def pick_top_biomarkers(feature_names, feat_freq_table_path=None, topn=20):
    # use selection frequency table if available
    if feat_freq_table_path and os.path.exists(feat_freq_table_path):
        df = pd.read_csv(feat_freq_table_path)
        if "feature" in df.columns:
            return [safe_str(x) for x in df["feature"].head(topn).tolist()]
    # fallback: first topn features
    return [safe_str(x) for x in feature_names[:topn]]

def plot_partialcorr_network(X_samples_by_features, feature_names, sample_groups,
                             top_features, out_png, out_pdf, edge_thr=0.15):
    if not NETWORKX_OK:
        return False

    # build subset matrix
    name_to_idx = {safe_str(n): i for i, n in enumerate(feature_names)}
    idx = [name_to_idx.get(safe_str(n)) for n in top_features]
    idx = [i for i in idx if i is not None]
    if len(idx) < 5:
        return False

    Xsub = np.nan_to_num(X_samples_by_features[:, idx], nan=0.0, posinf=0.0, neginf=0.0)
    Xsub = zscore_cols(Xsub)

    # precision via LedoitWolf
    lw = LedoitWolf().fit(Xsub)
    P = lw.precision_
    pc = partial_corr_from_precision(P)

    # node color by "which group has max mean z"
    groups = np.array(sample_groups, dtype=str)
    uniq = sorted(list(set(groups)))
    mean_by_g = {}
    for g in uniq:
        mean_by_g[g] = np.mean(Xsub[groups == g], axis=0)
    # choose argmax group per feature
    node_group = []
    for j in range(Xsub.shape[1]):
        best = max(uniq, key=lambda g: mean_by_g[g][j])
        node_group.append(best)

    # build graph
    G = nx.Graph()
    nodes = [safe_str(feature_names[i]) for i in idx]
    for n, g in zip(nodes, node_group):
        G.add_node(n, group=g)

    edges = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            w = pc[i, j]
            if np.isfinite(w) and abs(w) >= edge_thr:
                G.add_edge(nodes[i], nodes[j], weight=float(w))
                edges.append((nodes[i], nodes[j], float(w)))

    if len(edges) == 0:
        return False

    # draw
    plt.figure(figsize=(7.6, 6.2))
    pos = nx.spring_layout(G, seed=0, k=0.7)

    # color map by group
    uniq = sorted(list(set(node_group)))
    palette = {g: i for i, g in enumerate(uniq)}
    node_colors = [palette[G.nodes[n]["group"]] for n in G.nodes()]

    # edge width by |weight|
    ewidth = [1.0 + 2.5*abs(G.edges[e]["weight"]) for e in G.edges()]

    nx.draw_networkx_edges(G, pos, width=ewidth, alpha=0.35)
    nx.draw_networkx_nodes(G, pos, node_size=520, node_color=node_colors, alpha=0.90)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Top biomarkers partial-correlation network (shrinkage precision)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

    # save nodes/edges table
    return True

def bootstrap_effect_ci(a, b, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return (np.nan, np.nan, np.nan)
    # effect = mean(a) - mean(b)
    obs = float(np.mean(a) - np.mean(b))
    stats = []
    for _ in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        stats.append(float(np.mean(aa) - np.mean(bb)))
    lo, hi = np.percentile(np.array(stats), [2.5, 97.5])
    return (obs, float(lo), float(hi))

def plot_module_assoc_bar(score_df, out_png, out_pdf, classes):
    # association: for each module, compare HS vs NC and ZS vs NC (mean difference + CI)
    module_cols = [c for c in score_df.columns if c.startswith("EigM")]
    if len(module_cols) == 0:
        return

    # pick baseline
    if "NC" in classes:
        base = "NC"
    else:
        base = classes[0]

    rows = []
    for mcol in module_cols:
        for tgt in [c for c in classes if c != base]:
            a = score_df.loc[score_df["group"] == tgt, mcol].values
            b = score_df.loc[score_df["group"] == base, mcol].values
            obs, lo, hi = bootstrap_effect_ci(a, b, n_boot=600, seed=0)
            rows.append({"module": mcol, "contrast": f"{tgt}-{base}", "effect": obs, "lo": lo, "hi": hi})

    df = pd.DataFrame(rows)
    # plot compact bars
    plt.figure(figsize=(7.4, 4.8))
    x = np.arange(len(df))
    y = df["effect"].values
    yerr = np.vstack([y - df["lo"].values, df["hi"].values - y])
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=3, lw=1.2)
    plt.axhline(0, color="0.2", lw=0.8)
    plt.xticks(x, [f"{r.module}\n{r.contrast}" for r in df.itertuples()], rotation=45, ha="right")
    plt.ylabel("Eigenscore difference (mean, bootstrap 95% CI)")
    plt.title("Module eigenscore association with subtype")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

def pathway_evidence_from_mapping(mapping_path, hits_list, out_csv, out_png, out_pdf, top_show=20):
    """
    mapping table expected columns (flexible):
      pathway_id, pathway_name, compound_id(or feature), compound_name(optional)
    hits_list: list of "hit compounds/features" (strings)
    Evidence = hit_count / total_mappable_in_pathway
    And list top <=5 hit compounds.
    """
    if mapping_path is None or not os.path.exists(mapping_path):
        return False

    mp = pd.read_csv(mapping_path)
    # infer columns
    cols = {c.lower(): c for c in mp.columns}
    pid = cols.get("pathway_id", None) or cols.get("pathway", None)
    pname = cols.get("pathway_name", None) or cols.get("name", None)
    cmpd = cols.get("compound_id", None) or cols.get("compound", None) or cols.get("feature", None)
    if pid is None or cmpd is None:
        return False
    if pname is None:
        mp["pathway_name"] = mp[pid].astype(str)
        pname = "pathway_name"

    mp["_hit"] = mp[cmpd].astype(str).isin(set([str(x) for x in hits_list]))

    g = mp.groupby([pid, pname])
    out = []
    for (p_id, p_nm), sub in g:
        total = int(sub[cmpd].nunique())
        hit = int(sub.loc[sub["_hit"], cmpd].nunique())
        if total <= 0:
            continue
        ratio = hit / total
        if hit == 0:
            continue
        # top supporting compounds (<=5)
        supp = sub.loc[sub["_hit"], cmpd].astype(str).drop_duplicates().head(5).tolist()
        out.append({
            "pathway_id": p_id,
            "pathway_name": p_nm,
            "hit": hit,
            "total_mappable": total,
            "hit_ratio": ratio,
            "top_support_compounds": ";".join(supp)
        })

    if len(out) == 0:
        return False

    df = pd.DataFrame(out).sort_values("hit_ratio", ascending=False)
    df.to_csv(out_csv, index=False)

    # plot top_show
    d2 = df.head(top_show).iloc[::-1]
    plt.figure(figsize=(8.4, 0.33*len(d2) + 2.2))
    plt.barh(d2["pathway_name"].astype(str), d2["hit_ratio"].values)
    plt.xlabel("Hit density (hit compounds / total mappable)")
    plt.title("Pathway evidence (density + top supporting compounds)")
    # annotate support compounds lightly (avoid clutter)
    for i, r in enumerate(d2.itertuples()):
        txt = str(r.top_support_compounds)
        if len(txt) > 70:
            txt = txt[:67] + "..."
        plt.text(r.hit_ratio + 0.01, i, txt, va="center", fontsize=8)
    plt.xlim(0, min(1.02, max(0.2, float(d2["hit_ratio"].max()) + 0.15)))
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    return True

# ----------------------------
# main
# ----------------------------
def main():
    mpl_science_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="数据矩阵.xlsx")
    ap.add_argument("--sheet_data", default=None)
    ap.add_argument("--sheet_group", default=None)

    ap.add_argument("--drop", nargs="*", default=[], help="sample IDs to drop, e.g. NC1 NC3")

    ap.add_argument("--classes", nargs="*", default=["NC", "HS", "ZS"])
    ap.add_argument("--target_outer_folds", type=int, default=50)
    ap.add_argument("--max_splits", type=int, default=5)
    ap.add_argument("--inner_max_splits", type=int, default=4)

    ap.add_argument("--oversample", default="none", choices=["none", "smote", "adasyn", "noise"])
    ap.add_argument("--augment_n", type=int, default=0, help="TRAIN-only target size for A-line. 0 disables.")
    ap.add_argument("--noise_scale", type=float, default=0.10)

    ap.add_argument("--do_nested_tuning", action="store_true")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="Predict_Figures_v4")

    # A-line: CI & calibration
    ap.add_argument("--n_boot_auc", type=int, default=400)
    ap.add_argument("--n_bins_cal", type=int, default=5)
    ap.add_argument("--n_boot_cal", type=int, default=300)

    # DCA
    ap.add_argument("--dca_min", type=float, default=0.05)
    ap.add_argument("--dca_max", type=float, default=0.95)
    ap.add_argument("--dca_n", type=int, default=19)

    # Sensitivity (DE)
    ap.add_argument("--fc_groupA", default="HS")
    ap.add_argument("--fc_groupB", default="NC")
    ap.add_argument("--q_thr", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=20)

    # SHAP
    ap.add_argument("--shap", action="store_true")
    ap.add_argument("--shap_topk", type=int, default=10)

    # ---------------- B-line flags ----------------
    ap.add_argument("--mechanism", action="store_true", help="Enable mechanistic figures B-line.")
    ap.add_argument("--n_modules", type=int, default=8, help="5-10 recommended.")
    ap.add_argument("--module_feature_top", type=int, default=500,
                    help="Use top variable features for module building (to avoid huge p).")
    ap.add_argument("--corr_method", default="spearman", choices=["spearman", "pearson"])

    # mechanism augmentation (for module/network estimation)
    ap.add_argument("--mech_aug_method", default="none", choices=["none", "smote", "adasyn", "noise"])
    ap.add_argument("--mech_aug_n", type=int, default=100)
    ap.add_argument("--mech_plot_augmented", action="store_true",
                    help="Plot heatmap/violin using augmented samples (not recommended, but available).")

    # network parameters
    ap.add_argument("--net_top", type=int, default=20)
    ap.add_argument("--net_edge_thr", type=float, default=0.15)

    # pathway evidence mapping file
    ap.add_argument("--pathway_map", default=None,
                    help="CSV mapping: pathway_id,pathway_name,compound_id(or feature). If not provided, will auto-search common names in cwd.")
    ap.add_argument("--pathway_top_show", type=int, default=20)

    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    figdir = ensure_dir(os.path.join(outdir, "figures"))
    tabdir = ensure_dir(os.path.join(outdir, "tables"))
    mech_figdir = ensure_dir(os.path.join(outdir, "figures_mechanism"))
    mech_tabdir = ensure_dir(os.path.join(outdir, "tables_mechanism"))

    # -----------------------
    # Read Excel
    # -----------------------
    xl = pd.ExcelFile(args.xlsx)
    sheet_data = args.sheet_data or find_sheet(xl, ["数据矩阵", "缺失值数据矩阵"])
    sheet_group = args.sheet_group or find_sheet(xl, ["分组", "Group", "group"])

    df = xl.parse(sheet_data)
    df_group = xl.parse(sheet_group)

    pat = re.compile(r"^(QC|NC|HS|ZS)\d+", re.IGNORECASE)
    sample_cols = []
    for c in df.columns:
        if isinstance(c, str) and pat.match(c.strip()):
            sample_cols.append(c.strip())

    drop_set = set([str(x) for x in args.drop])
    sample_cols = [c for c in sample_cols if c not in drop_set]
    if len(sample_cols) < 3:
        raise ValueError("Too few sample columns found. Please check sheet format / sample naming.")

    cols_lower = {c: str(c).lower() for c in df_group.columns}
    sample_col = None
    group_col = None
    for c in df_group.columns:
        if any(k in cols_lower[c] for k in ["sample", "样本"]):
            sample_col = c
        if any(k in cols_lower[c] for k in ["group", "分组", "类别"]):
            group_col = c
    if sample_col is None or group_col is None:
        sample_col = df_group.columns[0]
        group_col = df_group.columns[1]

    gmap = dict(zip(df_group[sample_col].astype(str), df_group[group_col].astype(str)))

    # feature names
    feature_names = [safe_str(df.index[i], fallback=f"F{i:05d}") for i in range(df.shape[0])]

    # matrix: samples x features
    X_all = df[sample_cols].T.to_numpy(float)
    s_all = np.array(sample_cols, dtype=str)
    y_all = np.array([gmap.get(s, "Unknown") for s in s_all], dtype=str)

    classes = [str(c) for c in args.classes]
    keep = np.isin(y_all, classes)
    X = X_all[keep]
    y = y_all[keep]
    s = s_all[keep]

    minc = _min_class_count(y)
    if minc < 2:
        raise ValueError(f"After dropping samples, min class count={minc} <2; cannot run CV.")

    # choose outer CV
    n_splits_outer, repeats, inner_splits_global = pick_cv_params(
        y,
        target_outer_folds=args.target_outer_folds,
        max_splits=args.max_splits,
        inner_max_splits=args.inner_max_splits
    )
    outer = RepeatedStratifiedKFold(n_splits=n_splits_outer, n_repeats=repeats, random_state=args.seed)

    # model pipeline
    base_clf = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=8000, n_jobs=-1)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", base_clf),
    ])

    param_grid = {"clf__C": [0.05, 0.1, 0.2, 0.5, 1.0], "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}
    fixed_params = {"clf__C": 0.2, "clf__l1_ratio": 0.5}

    # OOF prediction + stability
    oof_prob = np.full((len(y), len(classes)), np.nan, float)
    fold_auc = []
    feat_counts = {}
    best_params_list = []

    for fold_id, (tr, te) in enumerate(outer.split(X, y)):
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        # TRAIN-only oversample / augmentation (A-line)
        X_tr2, y_tr2, used_os = oversample_train(
            X_tr, y_tr,
            method=args.oversample,
            seed=args.seed + 1000 + fold_id,
            augment_n=int(args.augment_n) if args.augment_n else 0,
            noise_scale=float(args.noise_scale)
        )

        X_tr2 = np.nan_to_num(X_tr2, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

        # nested tuning
        if args.do_nested_tuning:
            minc_tr = _min_class_count(y_tr2)
            inner_splits = min(inner_splits_global, minc_tr)
            inner_splits = max(2, inner_splits)
            inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=args.seed + fold_id)
            gs = GridSearchCV(pipe, param_grid=param_grid, scoring="roc_auc_ovr_weighted", cv=inner, n_jobs=-1, refit=True)
            gs.fit(X_tr2, y_tr2)
            model = gs.best_estimator_
            best_params_list.append(gs.best_params_)
        else:
            model = pipe
            model.set_params(**fixed_params)
            model.fit(X_tr2, y_tr2)
            best_params_list.append({"fixed": True, **fixed_params, "oversample": used_os})

        prob_te = model.predict_proba(X_te)
        cls_order = list(model.named_steps["clf"].classes_)
        prob_aligned = np.zeros((prob_te.shape[0], len(classes)), float)
        for k, cls in enumerate(classes):
            prob_aligned[:, k] = prob_te[:, cls_order.index(cls)]
        oof_prob[te] = prob_aligned

        y_te_int = np.array([classes.index(v) for v in y_te])
        y_te_oh = np.eye(len(classes))[y_te_int]
        try:
            auc_macro = roc_auc_score(y_te_oh, prob_aligned, average="macro", multi_class="ovr")
        except Exception:
            auc_macro = np.nan
        fold_auc.append(auc_macro)

        coef = model.named_steps["clf"].coef_
        nonzero = np.where(np.max(np.abs(coef), axis=0) > 1e-8)[0]
        for j in nonzero:
            nm = safe_str(feature_names[j], fallback=f"F{j:05d}")
            feat_counts[nm] = feat_counts.get(nm, 0) + 1

    fold_auc = np.array(fold_auc, float)

    # Save summaries
    pd.DataFrame(best_params_list).to_csv(os.path.join(tabdir, "Table_NestedCV_BestParams_perFold.csv"), index=False)
    pd.DataFrame({
        "metric": [
            "outer_splits", "outer_repeats", "n_outer_folds",
            "macroAUC_mean", "macroAUC_median", "macroAUC_min", "macroAUC_max",
            "n_samples_used", "oversample", "augment_n", "noise_scale",
            "nested_tuning"
        ],
        "value": [
            n_splits_outer, repeats, len(fold_auc),
            float(np.nanmean(fold_auc)), float(np.nanmedian(fold_auc)),
            float(np.nanmin(fold_auc)), float(np.nanmax(fold_auc)),
            int(len(y)), args.oversample, int(args.augment_n), float(args.noise_scale),
            bool(args.do_nested_tuning)
        ]
    }).to_csv(os.path.join(tabdir, "Table_ModelPerformanceSummary.csv"), index=False)

    # Prepare one-hot
    y_int = np.array([classes.index(v) for v in y])
    y_oh = np.eye(len(classes))[y_int]

    # Fig2A ROC
    plt.figure(figsize=(6, 6))
    for k, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_oh[:, k], oof_prob[:, k])
        a = auc(fpr, tpr)
        ci = bootstrap_ci(
            y_oh[:, k].astype(int), oof_prob[:, k],
            lambda yt, yp: roc_auc_score(yt, yp),
            n_boot=args.n_boot_auc, seed=args.seed + 10 + k
        )
        if np.isfinite(ci).all():
            lab = f"{cls} (AUC={a:.3f}, 95%CI {ci[0]:.3f}-{ci[1]:.3f})"
        else:
            lab = f"{cls} (AUC={a:.3f})"
        plt.plot(fpr, tpr, lw=2, label=lab)
    plt.plot([0, 1], [0, 1], color="0.5", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Multiclass ROC (one-vs-rest, out-of-fold)")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "Fig2A_ROC_multiclass_OVR_OOF.png"), dpi=300)
    plt.savefig(os.path.join(figdir, "Fig2A_ROC_multiclass_OVR_OOF.pdf"))
    plt.close()

    # Fig2B Calibration + binomial bootstrap CI
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], color="0.5", lw=1)
    for k, cls in enumerate(classes):
        prob = oof_prob[:, k]
        yt = y_oh[:, k].astype(int)
        n_bins = int(args.n_bins_cal)
        edges = np.quantile(prob, np.linspace(0, 1, n_bins + 1))
        edges[0] = -1e9
        edges[-1] = 1e9
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-9

        xs, ys, lo, hi = [], [], [], []
        rng = np.random.default_rng(args.seed + 100 + k)
        for i in range(n_bins):
            m = (prob > edges[i]) & (prob <= edges[i + 1])
            if m.sum() < 5:
                continue
            x = float(np.mean(prob[m]))
            yobs = float(np.mean(yt[m]))
            idx = np.where(m)[0]
            boot = []
            for _ in range(int(args.n_boot_cal)):
                b = rng.choice(idx, size=len(idx), replace=True)
                boot.append(float(np.mean(yt[b])))
            ci = np.percentile(np.array(boot), [2.5, 97.5])
            xs.append(x); ys.append(yobs); lo.append(ci[0]); hi.append(ci[1])
        if len(xs) == 0:
            continue
        xs = np.array(xs); ys = np.array(ys); lo = np.array(lo); hi = np.array(hi)
        plt.plot(xs, ys, marker="o", lw=2, label=f"{cls}")
        plt.fill_between(xs, lo, hi, alpha=0.15)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction")
    plt.title("Calibration (one-vs-rest, out-of-fold)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "Fig2B_Calibration_multiclass_OVR_OOF.png"), dpi=300)
    plt.savefig(os.path.join(figdir, "Fig2B_Calibration_multiclass_OVR_OOF.pdf"))
    plt.close()

    # Fig2C DCA
    pts = np.linspace(float(args.dca_min), float(args.dca_max), int(args.dca_n))
    plt.figure(figsize=(6.8, 5.2))
    plt.plot(pts, np.zeros_like(pts), color="0.2", lw=1.0, ls=":", label="Treat none")
    treatall_labeled = False
    for k, cls in enumerate(classes):
        yt = y_oh[:, k].astype(int)
        prob = oof_prob[:, k]
        prev = float(np.mean(yt))
        nb_model = [net_benefit_binary(yt, prob, pt) for pt in pts]
        nb_all = [treat_all_nb(prev, pt) for pt in pts]
        plt.plot(pts, nb_model, lw=2, label=f"Model: {cls}")
        if not treatall_labeled:
            plt.plot(pts, nb_all, lw=1.2, color="0.5", ls="--", label="Treat all (baseline)")
            treatall_labeled = True
        else:
            plt.plot(pts, nb_all, lw=1.2, color="0.5", ls="--")
    plt.axhline(0, color="0.2", lw=0.8)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision curve analysis (one-vs-rest, out-of-fold)")
    plt.legend(fontsize=9, loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "Fig2C_DCA_multiclass_OVR_OOF.png"), dpi=300)
    plt.savefig(os.path.join(figdir, "Fig2C_DCA_multiclass_OVR_OOF.pdf"))
    plt.close()

    # ExtData AUC distribution
    plt.figure(figsize=(4.6, 5.0))
    data = fold_auc[np.isfinite(fold_auc)]
    if len(data) >= 2:
        plt.violinplot(data, showmeans=False, showmedians=True)
    else:
        plt.violinplot([data if len(data) else [np.nan]], showmeans=False, showmedians=True)
    rng = np.random.default_rng(args.seed)
    plt.scatter(np.ones_like(data) + rng.normal(0, 0.03, size=len(data)), data, s=22, alpha=0.65)
    plt.xticks([1], ["macro-AUC"])
    plt.ylabel("AUC (outer folds)")
    plt.title(f"CV stability (n={len(fold_auc)} outer folds)")
    ymin = max(0.0, float(np.nanmin(fold_auc)) - 0.05) if np.isfinite(np.nanmin(fold_auc)) else 0.0
    ymax = min(1.01, float(np.nanmax(fold_auc)) + 0.02) if np.isfinite(np.nanmax(fold_auc)) else 1.01
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "ExtData_AUC_distribution_OOF.png"), dpi=300)
    plt.savefig(os.path.join(figdir, "ExtData_AUC_distribution_OOF.pdf"))
    plt.close()

    # ExtData Feature freq Top20
    feat_df = pd.DataFrame({"feature": list(feat_counts.keys()), "selected_folds": list(feat_counts.values())})
    feat_df["freq"] = feat_df["selected_folds"] / max(1, len(fold_auc))
    feat_df = feat_df.sort_values("freq", ascending=False).reset_index(drop=True)
    feat_df.to_csv(os.path.join(tabdir, "Table_FeatureSelectionFrequency_all.csv"), index=False)

    top = feat_df.head(20).iloc[::-1]
    plt.figure(figsize=(6.9, 5.7))
    plt.barh(top["feature"], top["freq"])
    plt.xlabel("Selection frequency (outer folds)")
    plt.title("Top biomarkers by selection stability (OOF)")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "ExtData_FeatureFreq_Top20_OOF.png"), dpi=300)
    plt.savefig(os.path.join(figdir, "ExtData_FeatureFreq_Top20_OOF.pdf"))
    plt.close()

    # -----------------------
    # Fig S1C + Table S1 Part II (Full vs Sensitivity)
    # -----------------------
    df_full = xl.parse(sheet_data)
    full_sample_cols = []
    for c in df_full.columns:
        if isinstance(c, str) and pat.match(c.strip()):
            full_sample_cols.append(c.strip())

    df_full_wide = df_full[full_sample_cols].copy()
    df_full_wide.index = [safe_str(x, fallback=f"F{i:05d}") for i, x in enumerate(df_full_wide.index)]
    sample_to_group_full = {ss: gmap.get(ss, "Unknown") for ss in full_sample_cols}

    df_sens_wide = df[sample_cols].copy()
    df_sens_wide.index = [safe_str(x, fallback=f"F{i:05d}") for i, x in enumerate(df_sens_wide.index)]
    sample_to_group_sens = {ss: gmap.get(ss, "Unknown") for ss in sample_cols}

    if SCIPY_OK:
        full_stats = diff_stats(df_full_wide, args.fc_groupA, args.fc_groupB, sample_to_group_full)
        sens_stats = diff_stats(df_sens_wide, args.fc_groupA, args.fc_groupB, sample_to_group_sens)

        m = full_stats.set_index("feature").join(
            sens_stats.set_index("feature"),
            lsuffix="_full", rsuffix="_sens", how="inner"
        )

        xfc = m["log2FC_full"].values
        yfc = m["log2FC_sens"].values

        plt.figure(figsize=(6, 6))
        plt.scatter(xfc, yfc, s=12, alpha=0.45)
        lim = np.nanmax(np.abs(np.r_[xfc, yfc]))
        lim = max(1.0, float(lim))
        plt.plot([-lim, lim], [-lim, lim], color="0.2", lw=1.6)
        plt.axhline(0, color="0.3", lw=0.9)
        plt.axvline(0, color="0.3", lw=0.9)
        plt.xlim(-lim, lim); plt.ylim(-lim, lim)
        plt.xlabel(f"log2FC ({args.fc_groupA} vs {args.fc_groupB}) - Full")
        plt.ylabel(f"log2FC ({args.fc_groupA} vs {args.fc_groupB}) - Sensitivity(drop)")
        plt.title("Fig S1C. log2FC concordance (Full vs Sensitivity)")
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, "FigS1C_log2FC_concordance_Full_vs_Sens.png"), dpi=300)
        plt.savefig(os.path.join(figdir, "FigS1C_log2FC_concordance_Full_vs_Sens.pdf"))
        plt.close()

        metrics = concordance_table(full_stats, sens_stats, topk=int(args.topk), q_thr=float(args.q_thr))
        pd.DataFrame([metrics]).to_csv(
            os.path.join(tabdir, f"TableS1_PartII_Full_vs_Sensitivity_{args.fc_groupA}vs{args.fc_groupB}.csv"),
            index=False
        )
        full_stats.to_csv(os.path.join(tabdir, f"Table_DE_Full_{args.fc_groupA}vs{args.fc_groupB}.csv"), index=False)
        sens_stats.to_csv(os.path.join(tabdir, f"Table_DE_Sensitivity_{args.fc_groupA}vs{args.fc_groupB}.csv"), index=False)

    # -----------------------
    # Optional SHAP
    # -----------------------
    if args.shap and SHAP_OK:
        final_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga", max_iter=8000, n_jobs=-1,
                C=fixed_params["clf__C"], l1_ratio=fixed_params["clf__l1_ratio"]
            ))
        ])
        try_shap_outputs(final_model, X, y, feature_names, outdir=figdir, topk=int(args.shap_topk), seed=args.seed)

    # ==========================================================
    # B-line: Mechanism figures
    # ==========================================================
    if args.mechanism:
        # Choose features for module building: top by variance (real samples)
        X_real = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        var = np.var(X_real, axis=0)
        top_p = min(int(args.module_feature_top), X_real.shape[1])
        idx_var = np.argsort(var)[::-1][:top_p]

        X_mod_real = X_real[:, idx_var]
        feat_mod_names = [safe_str(feature_names[i], fallback=f"F{i:05d}") for i in idx_var]

        # Augmented data used ONLY for module/network estimation (optional)
        X_for_build, y_for_build, used_mech_aug = augment_for_mechanism(
            X_mod_real, y, method=args.mech_aug_method, target_n=int(args.mech_aug_n),
            seed=args.seed + 2024, noise_scale=float(args.noise_scale)
        )

        # Build modules on X_for_build (stabilized)
        module_ids = build_modules_corr_clustering(
            X_for_build, feat_mod_names, n_modules=int(args.n_modules), corr_method=args.corr_method
        ).astype(int)

        # Compute scores on REAL samples for plotting (default)
        scores_real, eig_real = compute_module_scores(X_mod_real, module_ids, int(args.n_modules))

        # Optionally compute on augmented samples for plotting
        if args.mech_plot_augmented:
            scores_plot, eig_plot = compute_module_scores(X_for_build, module_ids, int(args.n_modules))
            groups_plot = y_for_build
            sample_ids_plot = np.array([f"AUG{i:03d}" for i in range(len(y_for_build))], dtype=str)
            plot_tag = "_AUG"
        else:
            scores_plot, eig_plot = scores_real, eig_real
            groups_plot = y
            sample_ids_plot = s
            plot_tag = ""

        # Save module assignment table
        mod_assign = pd.DataFrame({
            "feature": feat_mod_names,
            "module": module_ids
        })
        mod_assign.to_csv(os.path.join(mech_tabdir, "Table_ModuleAssignments.csv"), index=False)

        # Save module scores
        df_scores = pd.DataFrame(scores_plot, columns=[f"M{i+1}" for i in range(int(args.n_modules))])
        df_eig = pd.DataFrame(eig_plot, columns=[f"EigM{i+1}" for i in range(int(args.n_modules))])
        df_ms = pd.concat([pd.DataFrame({"sample": sample_ids_plot, "group": groups_plot}), df_scores, df_eig], axis=1)
        df_ms.to_csv(os.path.join(mech_tabdir, f"Table_ModuleScores{plot_tag}.csv"), index=False)

        # Fig4A heatmap
        plot_module_heatmap(
            df_scores.values, np.array(groups_plot, dtype=str),
            title=f"Module-like signature scores (samples × modules){plot_tag}",
            out_png=os.path.join(mech_figdir, f"Fig4A_ModuleScore_heatmap{plot_tag}.png"),
            out_pdf=os.path.join(mech_figdir, f"Fig4A_ModuleScore_heatmap{plot_tag}.pdf"),
        )

        # Fig4B violin
        plot_module_violin(
            df_ms[["sample", "group"] + [c for c in df_scores.columns]].copy(),
            out_png=os.path.join(mech_figdir, f"Fig4B_ModuleScore_violin{plot_tag}.png"),
            out_pdf=os.path.join(mech_figdir, f"Fig4B_ModuleScore_violin{plot_tag}.pdf"),
            classes=classes
        )

        # ----------------- Fig5A network (Top biomarkers) -----------------
        # prefer feature selection frequency table from A-line
        freq_path = os.path.join(tabdir, "Table_FeatureSelectionFrequency_all.csv")
        top_nodes = pick_top_biomarkers(feature_names, feat_freq_table_path=freq_path, topn=int(args.net_top))

        # For network estimation, use module feature space? Better use the ORIGINAL feature space for those nodes
        name_to_fullidx = {safe_str(n): i for i, n in enumerate(feature_names)}
        idx_nodes = [name_to_fullidx.get(safe_str(n)) for n in top_nodes]
        idx_nodes = [i for i in idx_nodes if i is not None]
        if len(idx_nodes) >= 5:
            X_net_real = X_real[:, idx_nodes]
            feat_net_names = [safe_str(feature_names[i]) for i in idx_nodes]

            # Optionally stabilize network estimation with augmented samples (mechanism aug)
            X_net_build, y_net_build, _ = augment_for_mechanism(
                X_net_real, y, method=args.mech_aug_method, target_n=int(args.mech_aug_n),
                seed=args.seed + 909, noise_scale=float(args.noise_scale)
            )

            ok = plot_partialcorr_network(
                X_net_build, feat_net_names, y_net_build,
                top_features=feat_net_names,
                out_png=os.path.join(mech_figdir, "Fig5A_TopBiomarker_partialcorr_network.png"),
                out_pdf=os.path.join(mech_figdir, "Fig5A_TopBiomarker_partialcorr_network.pdf"),
                edge_thr=float(args.net_edge_thr)
            )
            if ok and NETWORKX_OK:
                # Save nodes/edges for transparency (optional)
                # Recompute edge list quickly (same steps)
                Xsub = zscore_cols(np.nan_to_num(X_net_build))
                lw = LedoitWolf().fit(Xsub)
                pc = partial_corr_from_precision(lw.precision_)
                edges = []
                for i in range(len(feat_net_names)):
                    for j in range(i+1, len(feat_net_names)):
                        w = float(pc[i, j])
                        if np.isfinite(w) and abs(w) >= float(args.net_edge_thr):
                            edges.append((feat_net_names[i], feat_net_names[j], w))
                pd.DataFrame(edges, columns=["node1", "node2", "partial_corr"]).to_csv(
                    os.path.join(mech_tabdir, "Table_TopNetwork_edges.csv"), index=False
                )
                pd.DataFrame({"node": feat_net_names}).to_csv(
                    os.path.join(mech_tabdir, "Table_TopNetwork_nodes.csv"), index=False
                )

        # ----------------- Fig5B module eigenscore association -----------------
        plot_module_assoc_bar(
            df_ms[["sample", "group"] + [c for c in df_eig.columns]].copy(),
            out_png=os.path.join(mech_figdir, f"Fig5B_ModuleEigenscore_assoc_bar{plot_tag}.png"),
            out_pdf=os.path.join(mech_figdir, f"Fig5B_ModuleEigenscore_assoc_bar{plot_tag}.pdf"),
            classes=classes
        )

        # ----------------- ExtData Pathway evidence (optional) -----------------
        # auto-detect mapping file if not provided
        map_path = args.pathway_map
        if map_path is None:
            for cand in ["compound_pathway_map.csv", "kegg_compound_pathway.csv", "pathway_mapping.csv"]:
                if os.path.exists(cand):
                    map_path = cand
                    break

        # choose hits list: use top stable biomarkers (avoid “C00416反复出现”)
        hits = pick_top_biomarkers(feature_names, feat_freq_table_path=freq_path, topn=50)

        ok = pathway_evidence_from_mapping(
            mapping_path=map_path,
            hits_list=hits,
            out_csv=os.path.join(mech_tabdir, "Table_PathwayEvidence.csv"),
            out_png=os.path.join(mech_figdir, "ExtData_PathwayEvidence_bar.png"),
            out_pdf=os.path.join(mech_figdir, "ExtData_PathwayEvidence_bar.pdf"),
            top_show=int(args.pathway_top_show)
        )
        if not ok:
            print("⚠️ Pathway evidence skipped: mapping table not found or format mismatch.")

        # record how modules were built
        pd.DataFrame([{
            "module_build_features": int(top_p),
            "n_modules": int(args.n_modules),
            "corr_method": args.corr_method,
            "mech_aug_method": args.mech_aug_method,
            "mech_aug_n": int(args.mech_aug_n),
            "mech_aug_used": used_mech_aug,
            "plot_augmented": bool(args.mech_plot_augmented)
        }]).to_csv(os.path.join(mech_tabdir, "Table_ModuleBuildConfig.csv"), index=False)

    # -----------------------
    # Print run info
    # -----------------------
    print("✅ Done (v4).")
    print(f"Data sheet: {sheet_data} | Group sheet: {sheet_group}")
    print(f"Classes used: {classes} | n={len(y)} | min class count={_min_class_count(y)}")
    print(f"Outer CV: n_splits={n_splits_outer}, repeats={repeats}, total folds={len(fold_auc)}")
    print(f"A-line oversample(train-only): {args.oversample} | augment_n={args.augment_n} | noise_scale={args.noise_scale}")
    if args.oversample in ['smote', 'adasyn'] and not IMBLEARN_OK:
        print("⚠️ imblearn not found: SMOTE/ADASYN unavailable for A-line; will fallback to noise if augment_n>0.")
    if args.mechanism:
        print(f"B-line mechanism enabled: n_modules={args.n_modules}, module_feature_top={args.module_feature_top}, mech_aug={args.mech_aug_method}, mech_aug_n={args.mech_aug_n}")
    if args.shap and not SHAP_OK:
        print("⚠️ shap not installed: SHAP outputs skipped.")
    if not NETWORKX_OK and args.mechanism:
        print("⚠️ networkx not installed: network plot skipped.")
    print(f"Figures: {figdir}")
    print(f"Tables : {tabdir}")
    if args.mechanism:
        print(f"Mechanism Figures: {mech_figdir}")
        print(f"Mechanism Tables : {mech_tabdir}")

if __name__ == "__main__":
    main()
# 在脚本末尾（oof_prob 已经填完之后）加入：
oof_df = pd.DataFrame({
    "SampleID": s,          # 你的样本名数组（你代码里一般叫 s 或 s_all[keep]）
    "y_true": y,
    "p_NC": oof_prob[:, 0],
    "p_HS": oof_prob[:, 1],
    "p_ZS": oof_prob[:, 2],
})
oof_df["risk_stroke"] = 1.0 - oof_df["p_NC"]          # 连续谱常用：stroke risk
oof_df["risk_ischemic"] = oof_df["p_ZS"]              # 可选：某一亚型概率
oof_df["risk_hemorrhagic"] = oof_df["p_HS"]           # 可选
oof_df.to_csv(os.path.join(tabdir, "OOF_pred_probs.csv"), index=False, encoding="utf-8-sig")


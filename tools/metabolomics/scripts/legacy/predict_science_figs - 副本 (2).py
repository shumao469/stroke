#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict_science_figs.py (v3)

What this script does (Science-style pipeline):
1) Read metabolomics matrix from Excel (features x samples) + group sheet.
2) Optional drop sample IDs (e.g., NC1 NC3).
3) Strict out-of-fold (OOF) evaluation for multiclass (NC/HS/ZS):
   - Fig2A: ROC (one-vs-rest) with bootstrap 95% CI (OOF)
   - Fig2B: Calibration (OOF) with binomial-bootstrap CI
   - Fig2C: Decision Curve Analysis (OOF, one-vs-rest)
4) CV stability:
   - ExtData: AUC distribution (violin + points)
   - ExtData: Feature selection stability (Top20 by non-zero coefficient frequency)
5) Sensitivity analysis for DE (HS vs NC) "Full vs Sensitivity(drop outlier NCs)":
   - Fig S1C: log2FC concordance scatter
   - Table S1 Part II: Jaccard(sig overlap)/FC correlations/Top20 overlap
6) Optional SHAP (only if shap is installed):
   - SHAP summary (Top10) and optional dependence for a few features.

Oversampling / augmentation:
- ONLY applied on TRAIN folds.
- Supports:
  --oversample smote | adasyn | noise | none
  --augment_n N  (target training size; for smote/adasyn it's "at least", then noise fill if needed)
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
    """Benjamini–Hochberg FDR"""
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
        # some boot samples can be all-0 or all-1; handle robustly
        try:
            stats.append(fn(y_true_bin[b], y_prob[b]))
        except Exception:
            continue
    if len(stats) < 20:
        return np.array([np.nan, np.nan])
    stats = np.array(stats)
    return np.percentile(stats, [2.5, 97.5])

def net_benefit_binary(y_true, y_prob, pt):
    """Net benefit for binary decision at threshold pt."""
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
    """
    choose outer n_splits and repeats based on min class count (global).
    ensure n_splits <= min_class_count, and also >=2.
    """
    minc = _min_class_count(y)
    if minc < 2:
        raise ValueError(f"Min class count={minc} < 2, cannot do stratified CV.")
    n_splits_outer = min(max_splits, minc)
    repeats = max(1, int(math.ceil(target_outer_folds / n_splits_outer)))
    inner_splits = min(inner_max_splits, max(2, minc))  # will refine per-fold too
    return n_splits_outer, repeats, inner_splits

# ----------------------------
# augmentation (TRAIN only)
# ----------------------------
def classwise_noise_augment(X, y, target_n=None, seed=0, scale=0.10):
    """
    Class-wise bootstrap + Gaussian noise (TRAIN only).
    noise sigma = std_per_feature * scale (computed within each class)
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    n, p = X.shape
    if (target_n is None) or (target_n <= n):
        return X, y

    classes, counts = np.unique(y, return_counts=True)
    extra = int(target_n - n)

    # allocate extras proportional to counts
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
    """For SMOTE/ADASYN: k must be <= min_class_count-1 and >=1."""
    _, cnt = np.unique(y_tr, return_counts=True)
    minc = int(cnt.min())
    return max(1, min(default, minc - 1))

def oversample_train(X_tr, y_tr, method="none", seed=0, augment_n=0, noise_scale=0.10):
    """
    Oversample only TRAIN.
    - method: none|smote|adasyn|noise
    - augment_n: if >0, ensure training size at least augment_n
      * smote/adasyn: first resample (balanced), then if still < augment_n, fill with noise augmentation.
      * noise: directly noise augment to augment_n.
    """
    method = (method or "none").lower()
    X_tr = np.asarray(X_tr, float)
    y_tr = np.asarray(y_tr)

    # always guard against NaN/inf before any sampler
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

            # ensure at least augment_n by noise fill if requested
            if augment_n and augment_n > len(y_res):
                X_res, y_res = classwise_noise_augment(X_res, y_res, target_n=augment_n, seed=seed + 999, scale=noise_scale)
                return X_res, y_res, f"{method}+noise_fill"
            return X_res, y_res, method
        except Exception:
            # fall back to noise if smote/adasyn fails
            pass

    # fallback: noise augmentation (if requested)
    if augment_n and augment_n > len(y_tr):
        X_aug, y_aug = classwise_noise_augment(X_tr, y_tr, target_n=augment_n, seed=seed, scale=noise_scale)
        return X_aug, y_aug, f"noise_fallback(scale={noise_scale})"

    return X_tr, y_tr, "none"

# ----------------------------
# DE / sensitivity (HS vs NC)
# ----------------------------
def diff_stats(df_wide, group_a, group_b, sample_to_group):
    """
    df_wide: features x samples
    log2FC defined as mean_a - mean_b (assumes input already log2-like or at least comparable).
    """
    from scipy.stats import ttest_ind

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
    """
    full_df/sens_df must have: feature, log2FC, q
    """
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

    from scipy.stats import pearsonr, spearmanr
    r_p = pearsonr(f["log2FC"].values, s["log2FC"].values)[0]
    r_s = spearmanr(f["log2FC"].values, s["log2FC"].values)[0]

    # topK by abs(log2FC)
    top_f = list(f.loc[f["log2FC"].abs().sort_values(ascending=False).head(topk).index].index)
    top_s = list(s.loc[s["log2FC"].abs().sort_values(ascending=False).head(topk).index].index)
    top_inter = sorted(set(top_f) & set(top_s))
    top_union = set(top_f) | set(top_s)
    top_j = len(set(top_inter)) / len(top_union) if len(top_union) else np.nan

    # sign concordance among common significant
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
def try_shap_outputs(model, X, y, feature_names, classes, outdir, topk=10, seed=0):
    """
    Train a final model on full data (no OOF), and output SHAP if available.
    For elasticnet LogisticRegression (linear), use shap.LinearExplainer.
    """
    if not SHAP_OK:
        return

    # fit
    model.fit(X, y)

    # transform X to the space used by the model:
    # Pipeline: imputer -> scaler -> clf
    Xt = model.named_steps["imputer"].transform(X)
    Xt = model.named_steps["scaler"].transform(Xt)

    clf = model.named_steps["clf"]

    # SHAP for multiclass linear model:
    # LinearExplainer expects the "model" and background data.
    try:
        import shap
        rng = np.random.default_rng(seed)
        bg_idx = rng.choice(np.arange(Xt.shape[0]), size=min(50, Xt.shape[0]), replace=False)
        background = Xt[bg_idx, :]

        explainer = shap.LinearExplainer(clf, background)
        shap_values = explainer.shap_values(Xt)  # list[n_classes] (n_samples, n_features)

        # Summary: use mean(|shap|) across classes
        # Build a single importance ranking
        abs_mean = np.zeros(Xt.shape[1], float)
        for k in range(len(shap_values)):
            abs_mean += np.mean(np.abs(shap_values[k]), axis=0)
        abs_mean /= max(1, len(shap_values))
        order = np.argsort(abs_mean)[::-1]
        top_idx = order[:min(topk, len(order))]

        # Save a compact bar plot (Science-style)
        plt.figure(figsize=(6.5, 3.6))
        plt.barh([feature_names[i] for i in top_idx][::-1], abs_mean[top_idx][::-1])
        plt.xlabel("Mean |SHAP| (avg across classes)")
        plt.title(f"SHAP summary (Top {len(top_idx)})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "ExtData_SHAP_summary_Top10.png"), dpi=300)
        plt.savefig(os.path.join(outdir, "ExtData_SHAP_summary_Top10.pdf"))
        plt.close()

        # Optional: dependence for 3 features
        for j in top_idx[:3]:
            plt.figure(figsize=(5.2, 4.2))
            # plot class-averaged SHAP value
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
        # if shap fails, silently skip
        return

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
    ap.add_argument("--augment_n", type=int, default=0, help="TRAIN-only target size (>=). 0 disables.")
    ap.add_argument("--noise_scale", type=float, default=0.10)

    ap.add_argument("--do_nested_tuning", action="store_true",
                    help="Enable inner-CV tuning for LogisticRegression (recommended for robustness).")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--outdir", default="Predict_Figures_v3")

    # Calibration / CI
    ap.add_argument("--n_boot_auc", type=int, default=400)
    ap.add_argument("--n_bins_cal", type=int, default=5)
    ap.add_argument("--n_boot_cal", type=int, default=300)

    # DCA
    ap.add_argument("--dca_min", type=float, default=0.05)
    ap.add_argument("--dca_max", type=float, default=0.95)
    ap.add_argument("--dca_n", type=int, default=19)

    # Sensitivity
    ap.add_argument("--fc_groupA", default="HS")
    ap.add_argument("--fc_groupB", default="NC")
    ap.add_argument("--q_thr", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=20)

    # SHAP
    ap.add_argument("--shap", action="store_true", help="If set: try SHAP. (Will still auto-skip if not installed)")
    ap.add_argument("--shap_topk", type=int, default=10)

    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    figdir = ensure_dir(os.path.join(outdir, "figures"))
    tabdir = ensure_dir(os.path.join(outdir, "tables"))

    # -----------------------
    # Read Excel
    # -----------------------
    xl = pd.ExcelFile(args.xlsx)
    sheet_data = args.sheet_data or find_sheet(xl, ["数据矩阵", "缺失值数据矩阵"])
    sheet_group = args.sheet_group or find_sheet(xl, ["分组", "Group", "group"])

    df = xl.parse(sheet_data)
    df_group = xl.parse(sheet_group)

    # identify sample columns robustly
    # samples typically look like: QC1, NC3, HS2, ZS4...
    pat = re.compile(r"^(QC|NC|HS|ZS)\d+", re.IGNORECASE)

    sample_cols = []
    for c in df.columns:
        if isinstance(c, str) and pat.match(c.strip()):
            sample_cols.append(c.strip())

    # apply drop
    drop_set = set([str(x) for x in args.drop])
    sample_cols = [c for c in sample_cols if c not in drop_set]

    if len(sample_cols) < 3:
        raise ValueError("Too few sample columns found. Please check sheet format / sample naming.")

    # group mapping
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

    # feature names: prefer a meaningful ID column if present; else row index
    # common: "Retention time (min)" exists but is numeric; not ideal as unique ID.
    # We'll fallback to row index; if it's nan, generate F00001...
    raw_feat_names = []
    for i in range(df.shape[0]):
        nm = None
        # try first column if it looks like an ID/Name column (not numeric RT)
        # but do not assume; safest: use index and fallback.
        nm = df.index[i]
        nm = safe_str(nm, fallback=f"F{i:05d}")
        raw_feat_names.append(nm)
    feature_names = raw_feat_names

    # matrix: samples x features
    X_all = df[sample_cols].T.to_numpy(float)
    s_all = np.array(sample_cols, dtype=str)
    y_all = np.array([gmap.get(s, "Unknown") for s in s_all], dtype=str)

    # keep only specified ML classes
    classes = [str(c) for c in args.classes]
    keep = np.isin(y_all, classes)
    X = X_all[keep]
    y = y_all[keep]
    s = s_all[keep]

    # sanity
    minc = _min_class_count(y)
    if minc < 2:
        raise ValueError(f"After dropping samples, min class count={minc} <2; cannot run CV.")

    # choose outer CV params based on min class count
    n_splits_outer, repeats, inner_splits_global = pick_cv_params(
        y,
        target_outer_folds=args.target_outer_folds,
        max_splits=args.max_splits,
        inner_max_splits=args.inner_max_splits
    )
    outer = RepeatedStratifiedKFold(n_splits=n_splits_outer, n_repeats=repeats, random_state=args.seed)

    # base model
    base_clf = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=8000,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", base_clf),
    ])

    # optional nested tuning grid (kept small and "Science-friendly")
    # NOTE: avoid huge grids; reviewers care about leakage + stability, not over-optimization.
    param_grid = {
        "clf__C": [0.05, 0.1, 0.2, 0.5, 1.0],
        "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    fixed_params = {"clf__C": 0.2, "clf__l1_ratio": 0.5}

    # OOF predictions container
    oof_prob = np.full((len(y), len(classes)), np.nan, float)
    fold_auc = []
    feat_counts = {}
    best_params_list = []

    for fold_id, (tr, te) in enumerate(outer.split(X, y)):
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        # TRAIN-only oversample / augmentation
        X_tr2, y_tr2, used_os = oversample_train(
            X_tr, y_tr,
            method=args.oversample,
            seed=args.seed + 1000 + fold_id,
            augment_n=int(args.augment_n) if args.augment_n else 0,
            noise_scale=float(args.noise_scale)
        )

        # guard
        X_tr2 = np.nan_to_num(X_tr2, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

        # NESTED: inner CV tuning on the (possibly augmented) training set
        if args.do_nested_tuning:
            # inner splits must also be <= min class count in TRAIN
            minc_tr = _min_class_count(y_tr2)
            inner_splits = min(inner_splits_global, minc_tr)
            inner_splits = max(2, inner_splits)

            inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=args.seed + fold_id)

            # set fixed penalty + solver
            pipe.set_params(clf__penalty="elasticnet", clf__solver="saga")

            gs = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring="roc_auc_ovr_weighted",  # stable for multiclass
                cv=inner,
                n_jobs=-1,
                refit=True
            )
            gs.fit(X_tr2, y_tr2)
            model = gs.best_estimator_
            best_params_list.append(gs.best_params_)
        else:
            model = pipe
            model.set_params(clf__penalty="elasticnet", clf__solver="saga", **fixed_params)
            model.fit(X_tr2, y_tr2)
            best_params_list.append({"fixed": True, **fixed_params, "oversample": used_os})

        # predict on test fold
        prob_te = model.predict_proba(X_te)
        cls_order = list(model.named_steps["clf"].classes_)  # learned class order
        prob_aligned = np.zeros((prob_te.shape[0], len(classes)), float)
        for k, cls in enumerate(classes):
            if cls not in cls_order:
                raise RuntimeError(f"Class {cls} not in model.classes_: {cls_order}")
            prob_aligned[:, k] = prob_te[:, cls_order.index(cls)]

        oof_prob[te] = prob_aligned

        # fold macro-AUC for stability stats (OVR)
        y_te_int = np.array([classes.index(v) for v in y_te])
        y_te_oh = np.eye(len(classes))[y_te_int]
        try:
            auc_macro = roc_auc_score(y_te_oh, prob_aligned, average="macro", multi_class="ovr")
        except Exception:
            auc_macro = np.nan
        fold_auc.append(auc_macro)

        # selection stability: non-zero coefficients
        coef = model.named_steps["clf"].coef_  # (n_classes, n_features)
        nonzero = np.where(np.max(np.abs(coef), axis=0) > 1e-8)[0]
        for j in nonzero:
            nm = feature_names[j] if j < len(feature_names) else f"F{j:05d}"
            nm = safe_str(nm, fallback=f"F{j:05d}")
            feat_counts[nm] = feat_counts.get(nm, 0) + 1

    fold_auc = np.array(fold_auc, float)

    # -----------------------
    # Save summary tables
    # -----------------------
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

    # -----------------------
    # Prepare OOF one-hot
    # -----------------------
    y_int = np.array([classes.index(v) for v in y])
    y_oh = np.eye(len(classes))[y_int]

    # -----------------------
    # Fig2A: ROC (OOF) + bootstrap CI
    # -----------------------
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

    # -----------------------
    # Fig2B: Calibration (OOF) + bootstrap CI
    # -----------------------
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], color="0.5", lw=1)

    for k, cls in enumerate(classes):
        prob = oof_prob[:, k]
        yt = y_oh[:, k].astype(int)

        # quantile bins (robust), but ensure unique edges
        n_bins = int(args.n_bins_cal)
        edges = np.quantile(prob, np.linspace(0, 1, n_bins + 1))
        # make strictly increasing (handles many identical probs)
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

    # -----------------------
    # Fig2C: DCA (OOF) one-vs-rest
    # -----------------------
    pts = np.linspace(float(args.dca_min), float(args.dca_max), int(args.dca_n))

    plt.figure(figsize=(6.8, 5.2))

    # plot treat-none once
    plt.plot(pts, np.zeros_like(pts), color="0.2", lw=1.0, ls=":", label="Treat none")

    # plot treat-all per class once (dashed gray), but avoid clutter by not repeating label
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

    # -----------------------
    # ExtData: AUC distribution
    # -----------------------
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

    # -----------------------
    # ExtData: Feature frequency (Top20)
    # -----------------------
    feat_df = pd.DataFrame({
        "feature": list(feat_counts.keys()),
        "selected_folds": list(feat_counts.values())
    })
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
    # Fig S1C + Table S1 Part II: Full vs Sensitivity (HS vs NC)
    # -----------------------
    # "Full": using all original sample columns (before drop) where possible
    df_full = xl.parse(sheet_data)
    full_sample_cols = []
    for c in df_full.columns:
        if isinstance(c, str) and pat.match(c.strip()):
            full_sample_cols.append(c.strip())

    df_full_wide = df_full[full_sample_cols].copy()
    df_full_wide.index = [safe_str(x, fallback=f"F{i:05d}") for i, x in enumerate(df_full_wide.index)]
    sample_to_group_full = {ss: gmap.get(ss, "Unknown") for ss in full_sample_cols}

    # "Sensitivity": current df_wide after drop
    df_sens_wide = df[sample_cols].copy()
    df_sens_wide.index = [safe_str(x, fallback=f"F{i:05d}") for i, x in enumerate(df_sens_wide.index)]
    sample_to_group_sens = {ss: gmap.get(ss, "Unknown") for ss in sample_cols}

    full_stats = diff_stats(df_full_wide, args.fc_groupA, args.fc_groupB, sample_to_group_full)
    sens_stats = diff_stats(df_sens_wide, args.fc_groupA, args.fc_groupB, sample_to_group_sens)

    m = full_stats.set_index("feature").join(
        sens_stats.set_index("feature"),
        lsuffix="_full", rsuffix="_sens", how="inner"
    )

    x = m["log2FC_full"].values
    yv = m["log2FC_sens"].values

    plt.figure(figsize=(6, 6))
    plt.scatter(x, yv, s=12, alpha=0.45)
    lim = np.nanmax(np.abs(np.r_[x, yv]))
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

    # also save the DE lists (optional but useful)
    full_stats.to_csv(os.path.join(tabdir, f"Table_DE_Full_{args.fc_groupA}vs{args.fc_groupB}.csv"), index=False)
    sens_stats.to_csv(os.path.join(tabdir, f"Table_DE_Sensitivity_{args.fc_groupA}vs{args.fc_groupB}.csv"), index=False)

    # -----------------------
    # Optional SHAP
    # -----------------------
    if args.shap and SHAP_OK:
        # train a final model on ALL data (no OOF) for interpretability only
        final_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga", max_iter=8000, n_jobs=-1,
                C=fixed_params["clf__C"], l1_ratio=fixed_params["clf__l1_ratio"]
            ))
        ])
        try_shap_outputs(
            final_model, X, y, feature_names, classes,
            outdir=figdir, topk=int(args.shap_topk), seed=args.seed
        )

    # -----------------------
    # Print run info
    # -----------------------
    print("✅ Done (v3).")
    print(f"Data sheet: {sheet_data} | Group sheet: {sheet_group}")
    print(f"Classes used: {classes} | n={len(y)} | min class count={_min_class_count(y)}")
    print(f"Outer CV: n_splits={n_splits_outer}, repeats={repeats}, total folds={len(fold_auc)}")
    print(f"Oversample (train-only): {args.oversample} | augment_n={args.augment_n} | noise_scale={args.noise_scale}")
    if args.oversample in ["smote", "adasyn"] and not IMBLEARN_OK:
        print("⚠️ imblearn not found: SMOTE/ADASYN unavailable; will fallback to noise if augment_n>0.")
    if args.shap and not SHAP_OK:
        print("⚠️ shap not installed: SHAP outputs skipped.")
    print(f"Figures: {figdir}")
    print(f"Tables : {tabdir}")

if __name__ == "__main__":
    main()

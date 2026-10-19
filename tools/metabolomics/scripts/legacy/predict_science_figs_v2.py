#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
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

# ----------------------------
# basic utils
# ----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def find_sheet(xl, candidates):
    for s in candidates:
        if s in xl.sheet_names:
            return s
    return xl.sheet_names[0]

def safe_str(x):
    if x is None:
        return "UnknownFeature"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "UnknownFeature"
    except Exception:
        pass
    return str(x)

def bh_fdr(p):
    """Benjamini–Hochberg FDR"""
    p = np.asarray(p, float)
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
        stats.append(fn(y_true_bin[b], y_prob[b]))
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

def pick_cv_params(y, target_outer_folds=50, max_splits=5, inner_max_splits=4):
    """choose outer splits/repeats based on min class count"""
    y = np.asarray(y)
    _, cnt = np.unique(y, return_counts=True)
    minc = int(cnt.min())
    if minc < 2:
        raise ValueError(f"Min class count={minc} < 2, cannot do stratified CV.")

    n_splits_outer = min(max_splits, minc)
    repeats = max(1, int(math.ceil(target_outer_folds / n_splits_outer)))
    return n_splits_outer, repeats, inner_max_splits

# ----------------------------
# oversampling / augmentation
# ----------------------------
def classwise_noise_augment(X, y, target_n=None, seed=0, scale=0.10):
    """
    Class-wise bootstrap + Gaussian noise (TRAIN only).
    X: (n_samples, n_features)
    y: (n_samples,)
    target_n: total target number of samples after augmentation (across all classes)
    scale: noise sigma = std_per_feature * scale (computed within each class)
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

        Xc = X[idx, :]  # (n_cls, p)

        # per-feature std, ddof=0 avoids NaN when n_cls==1
        sd = np.nanstd(Xc, axis=0, ddof=0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1e-6)

        # bootstrap pick existing samples + jitter
        pick = rng.integers(0, Xc.shape[0], size=k)
        base = Xc[pick, :]
        noise = rng.normal(0.0, sd * scale, size=base.shape)
        Xs = base + noise

        # final guard: no NaN/inf
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

        X_new.append(Xs)
        y_new.append(np.array([cls] * k, dtype=y.dtype))

    return np.vstack(X_new), np.concatenate(y_new)

def _safe_k_neighbors(y_tr, default=5):
    """For SMOTE/ADASYN: k must be < min_class_count."""
    _, cnt = np.unique(y_tr, return_counts=True)
    minc = int(cnt.min())
    # need at least 2 samples in every class to do any neighbor-based oversampling
    # k_neighbors must be <= minc-1 and >=1
    k = max(1, min(default, minc - 1))
    return k

def oversample_train(X_tr, y_tr, method="none", seed=0, fallback_target_n=0):
    """
    Oversample only TRAIN:
    - smote/adasyn if available
    - else fallback to classwise_noise_augment (bootstrap+jitter)
    """
    method = (method or "none").lower()

    # SMOTE/ADASYN path
    if method in ["smote", "adasyn"] and IMBLEARN_OK:
        try:
            k = _safe_k_neighbors(y_tr, default=5)
            if method == "smote":
                osamp = SMOTE(random_state=seed, k_neighbors=k)
            else:
                osamp = ADASYN(random_state=seed, n_neighbors=k)

            X_res, y_res = osamp.fit_resample(X_tr, y_tr)
            # guard
            X_res = np.nan_to_num(X_res, nan=0.0, posinf=0.0, neginf=0.0)
            return X_res, y_res, method
        except Exception:
            # fall through to noise fallback
            pass

    # fallback noise augmentation
    if fallback_target_n and fallback_target_n > len(y_tr):
        X_aug, y_aug = classwise_noise_augment(
            X_tr, y_tr, target_n=fallback_target_n, seed=seed, scale=0.10
        )
        return X_aug, y_aug, "noise(var-based)"
    else:
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        return X_tr, y_tr, "none"

# ----------------------------
# stats for supplement
# ----------------------------
def diff_stats(df_wide, group_a, group_b, sample_to_group):
    """
    df_wide: features x samples (original sheet columns)
    returns: dataframe with log2FC(a vs b), p, q
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
    full_df/sens_df: must have feature, log2FC, q
    returns metrics dict + top overlap list
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

    top_f = list(f.reindex(f["log2FC"].abs().sort_values(ascending=False).index).head(topk).index)
    top_s = list(s.reindex(s["log2FC"].abs().sort_values(ascending=False).index).head(topk).index)
    top_inter = sorted(set(top_f) & set(top_s))
    top_j = len(set(top_f) & set(top_s)) / len(set(top_f) | set(top_s))

    return {
        "n_sig_full": len(sig_f),
        "n_sig_sens": len(sig_s),
        "n_overlap": len(inter),
        "jaccard_sig": jacc,
        "pearson_log2FC": r_p,
        "spearman_log2FC": r_s,
        "top20_overlap": len(top_inter),
        "top20_jaccard": top_j,
        "top20_overlap_list": ";".join(top_inter[:50])
    }

# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="数据矩阵.xlsx")
    ap.add_argument("--sheet_data", default=None)
    ap.add_argument("--sheet_group", default=None)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--target_outer_folds", type=int, default=50)
    ap.add_argument("--oversample", default="none", choices=["none", "smote", "adasyn", "noise"])
    ap.add_argument("--augment_n", type=int, default=0, help="If >0: TRAIN only to this size via oversample/fallback noise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="Predict_Figures_v2")
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    figdir = ensure_dir(os.path.join(outdir, "figures"))
    tabdir = ensure_dir(os.path.join(outdir, "tables"))

    xl = pd.ExcelFile(args.xlsx)
    sheet_data = args.sheet_data or find_sheet(xl, ["数据矩阵", "缺失值数据矩阵"])
    sheet_group = args.sheet_group or find_sheet(xl, ["分组", "Group", "group"])

    df = xl.parse(sheet_data)
    df_group = xl.parse(sheet_group)

    # sample columns (include QC but QC excluded from ML)
    sample_cols = [
        c for c in df.columns
        if isinstance(c, str) and (c.startswith("QC") or c.startswith("HS") or c.startswith("NC") or c.startswith("ZS"))
    ]
    sample_cols = [c for c in sample_cols if c not in set(args.drop)]

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

    # matrix: samples x features
    X_all = df[sample_cols].T.to_numpy(float)
    s_all = np.array(sample_cols, dtype=str)
    y_all = np.array([gmap.get(s, "Unknown") for s in s_all], dtype=str)

    # ML keep only NC/HS/ZS
    classes = ["NC", "HS", "ZS"]
    keep = np.isin(y_all, classes)
    X = X_all[keep]
    y = y_all[keep]
    s = s_all[keep]

    # choose outer CV params based on min class count
    n_splits_outer, repeats, inner_max = pick_cv_params(
        y, target_outer_folds=args.target_outer_folds, max_splits=5, inner_max_splits=4
    )
    outer = RepeatedStratifiedKFold(n_splits=n_splits_outer, n_repeats=repeats, random_state=args.seed)

    # base model
    base_clf = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=6000,
        n_jobs=-1,
        C=0.2,
        l1_ratio=0.5
    )

    # IMPORTANT: include imputer to avoid NaN crash
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", base_clf)
    ])

    # OOF predictions
    oof_prob = np.full((len(y), len(classes)), np.nan, float)
    fold_auc = []
    feat_counts = {}

    # feature names
    feature_names = [safe_str(i) for i in df.index]

    for fold_id, (tr, te) in enumerate(outer.split(X, y)):
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        fallback_target_n = args.augment_n if args.augment_n and args.augment_n > 0 else 0

        if args.oversample == "noise":
            # force noise augmentation only
            if fallback_target_n and fallback_target_n > len(y_tr):
                X_tr2, y_tr2 = classwise_noise_augment(
                    X_tr, y_tr, target_n=fallback_target_n, seed=args.seed + fold_id, scale=0.10
                )
            else:
                X_tr2, y_tr2 = X_tr, y_tr
        else:
            X_tr2, y_tr2, used = oversample_train(
                X_tr, y_tr,
                method=args.oversample,
                seed=args.seed + fold_id,
                fallback_target_n=fallback_target_n
            )

        # guard again
        X_tr2 = np.nan_to_num(X_tr2, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

        model = pipe
        model.fit(X_tr2, y_tr2)

        prob_te = model.predict_proba(X_te)

        cls_order = list(model.named_steps["clf"].classes_)
        prob_aligned = np.zeros((prob_te.shape[0], len(classes)), float)
        for k, cls in enumerate(classes):
            prob_aligned[:, k] = prob_te[:, cls_order.index(cls)]

        oof_prob[te] = prob_aligned

        y_te_int = np.array([classes.index(v) for v in y_te])
        y_te_oh = np.eye(len(classes))[y_te_int]
        auc_macro = roc_auc_score(y_te_oh, prob_aligned, average="macro", multi_class="ovr")
        fold_auc.append(auc_macro)

        coef = model.named_steps["clf"].coef_
        nonzero = np.where(np.max(np.abs(coef), axis=0) > 1e-8)[0]
        for j in nonzero:
            name = feature_names[j] if j < len(feature_names) else f"F{j}"
            feat_counts[name] = feat_counts.get(name, 0) + 1

    fold_auc = np.array(fold_auc)

    # Save performance summary
    pd.DataFrame({
        "metric": ["outer_splits", "outer_repeats", "n_outer_folds",
                   "macroAUC_mean", "macroAUC_median", "macroAUC_min", "macroAUC_max",
                   "n_samples_used"],
        "value": [n_splits_outer, repeats, len(fold_auc),
                  float(fold_auc.mean()), float(np.median(fold_auc)),
                  float(fold_auc.min()), float(fold_auc.max()),
                  int(len(y))]
    }).to_csv(os.path.join(tabdir, "Table_ModelPerformanceSummary.csv"), index=False)

    # -----------------------
    # Fig2A: ROC (OOF)
    # -----------------------
    y_int = np.array([classes.index(v) for v in y])
    y_oh = np.eye(len(classes))[y_int]

    plt.figure(figsize=(6, 6))
    for k, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_oh[:, k], oof_prob[:, k])
        a = auc(fpr, tpr)
        ci = bootstrap_ci(
            y_oh[:, k], oof_prob[:, k],
            lambda yt, yp: roc_auc_score(yt, yp),
            n_boot=400, seed=args.seed + 10 + k
        )
        plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={a:.3f}, 95%CI {ci[0]:.3f}-{ci[1]:.3f})")
    plt.plot([0, 1], [0, 1], color="0.5", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Multiclass ROC (one-vs-rest, out-of-fold)")
    plt.legend(frameon=False, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "Fig2A_ROC_multiclass_OVR_OOF.png"), dpi=300)
    plt.savefig(os.path.join(figdir, "Fig2A_ROC_multiclass_OVR_OOF.pdf"))
    plt.close()

    # -----------------------
    # Fig2B: Calibration (OOF)
    # -----------------------
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], color="0.5", lw=1)

    for k, cls in enumerate(classes):
        prob = oof_prob[:, k]
        yt = y_oh[:, k].astype(int)

        n_bins = 5
        qbins = np.quantile(prob, np.linspace(0, 1, n_bins + 1))
        qbins[0] = -1e9
        qbins[-1] = 1e9

        xs, ys, lo, hi = [], [], [], []
        rng = np.random.default_rng(args.seed + 100 + k)

        for i in range(n_bins):
            m = (prob > qbins[i]) & (prob <= qbins[i+1])
            if m.sum() < 3:
                continue
            x = prob[m].mean()
            yobs = yt[m].mean()

            idx = np.where(m)[0]
            boot = []
            for _ in range(300):
                b = rng.choice(idx, size=len(idx), replace=True)
                boot.append(yt[b].mean())
            ci = np.percentile(boot, [2.5, 97.5])

            xs.append(x); ys.append(yobs); lo.append(ci[0]); hi.append(ci[1])

        xs, ys, lo, hi = map(np.array, (xs, ys, lo, hi))
        plt.plot(xs, ys, marker="o", lw=2, label=f"{cls}")
        plt.fill_between(xs, lo, hi, alpha=0.15)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction")
    plt.title("Calibration (one-vs-rest, out-of-fold)")
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "Fig2B_Calibration_multiclass_OVR_OOF.png"), dpi=300)
    plt.savefig(os.path.join(figdir, "Fig2B_Calibration_multiclass_OVR_OOF.pdf"))
    plt.close()

    # -----------------------
    # Fig2C: DCA (OOF)
    # -----------------------
    pts = np.linspace(0.05, 0.95, 19)
    plt.figure(figsize=(6.6, 5.2))
    for k, cls in enumerate(classes):
        yt = y_oh[:, k].astype(int)
        prob = oof_prob[:, k]
        prev = yt.mean()

        nb = [net_benefit_binary(yt, prob, pt) for pt in pts]
        nb_all = [treat_all_nb(prev, pt) for pt in pts]
        nb_none = [0.0 for _ in pts]

        plt.plot(pts, nb, lw=2, label=f"Model: {cls}")
        plt.plot(pts, nb_all, lw=1, color="0.5", ls="--")
        plt.plot(pts, nb_none, lw=1, color="0.2", ls=":")

    plt.axhline(0, color="0.2", lw=0.8)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision curve analysis (one-vs-rest, out-of-fold)")
    plt.legend(frameon=False, fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "Fig2C_DCA_multiclass_OVR_OOF.png"), dpi=300)
    plt.savefig(os.path.join(figdir, "Fig2C_DCA_multiclass_OVR_OOF.pdf"))
    plt.close()

    # -----------------------
    # ExtData: AUC distribution
    # -----------------------
    plt.figure(figsize=(4.6, 5.0))
    plt.violinplot(fold_auc, showmeans=False, showmedians=True)
    rng = np.random.default_rng(args.seed)
    plt.scatter(np.ones_like(fold_auc) + rng.normal(0, 0.03, size=len(fold_auc)), fold_auc, s=18, alpha=0.6)
    plt.xticks([1], ["macro-AUC"])
    plt.ylabel("AUC (outer folds)")
    plt.title(f"CV stability (n={len(fold_auc)} outer folds)")
    plt.ylim(max(0.4, float(fold_auc.min()) - 0.05), min(1.01, float(fold_auc.max()) + 0.02))
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
    feat_df["freq"] = feat_df["selected_folds"] / len(fold_auc)
    feat_df = feat_df.sort_values("freq", ascending=False).reset_index(drop=True)
    feat_df.to_csv(os.path.join(tabdir, "Table_FeatureSelectionFrequency_all.csv"), index=False)

    top = feat_df.head(20).iloc[::-1]
    plt.figure(figsize=(6.8, 5.6))
    plt.barh(top["feature"], top["freq"])
    plt.xlabel("Selection frequency (outer folds)")
    plt.title("Top biomarkers by selection stability")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "ExtData_FeatureFreq_Top20_OOF.png"), dpi=300)
    plt.savefig(os.path.join(figdir, "ExtData_FeatureFreq_Top20_OOF.pdf"))
    plt.close()

    # -----------------------
    # Fig S1C + Table S1 Part II: Full vs Sensitivity (drop NC outliers)
    # -----------------------
    df_wide = df[sample_cols].copy()
    df_wide.index = [safe_str(i) for i in df_wide.index]
    sample_to_group = {ss: gmap.get(ss, "Unknown") for ss in sample_cols}

    try:
        df_full = xl.parse(sheet_data)
        full_sample_cols = [
            c for c in df_full.columns
            if isinstance(c, str) and (c.startswith("QC") or c.startswith("HS") or c.startswith("NC") or c.startswith("ZS"))
        ]
        df_full_wide = df_full[full_sample_cols].copy()
        df_full_wide.index = [safe_str(i) for i in df_full_wide.index]
        sample_to_group_full = {ss: gmap.get(ss, "Unknown") for ss in full_sample_cols}
        full_stats = diff_stats(df_full_wide, "HS", "NC", sample_to_group_full)
    except Exception:
        full_stats = diff_stats(df_wide, "HS", "NC", sample_to_group)

    sens_stats = diff_stats(df_wide, "HS", "NC", sample_to_group)

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
    plt.plot([-lim, lim], [-lim, lim], color="0.2", lw=1.5)
    plt.axhline(0, color="0.3", lw=0.8)
    plt.axvline(0, color="0.3", lw=0.8)
    plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.xlabel("log2FC (HS vs NC) - Full")
    plt.ylabel("log2FC (HS vs NC) - Sensitivity (drop NC outliers)")
    plt.title("Fig S1C. log2FC concordance (Full vs Sensitivity)")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "FigS1C_log2FC_concordance_Full_vs_Sens.png"), dpi=300)
    plt.savefig(os.path.join(figdir, "FigS1C_log2FC_concordance_Full_vs_Sens.pdf"))
    plt.close()

    metrics = concordance_table(full_stats, sens_stats, topk=20, q_thr=0.05)
    pd.DataFrame([metrics]).to_csv(
        os.path.join(tabdir, "TableS1_PartII_Full_vs_Sensitivity_HSvsNC.csv"),
        index=False
    )

    print("✅ Done (v2).")
    print(f"Outer CV: n_splits={n_splits_outer}, repeats={repeats}, total folds={len(fold_auc)}")
    if args.oversample in ["smote", "adasyn"] and not IMBLEARN_OK:
        print("⚠️ imblearn not found: SMOTE/ADASYN skipped, used noise fallback when augment_n>0.")
    print(f"Figures: {figdir}")
    print(f"Tables : {tabdir}")

if __name__ == "__main__":
    main()

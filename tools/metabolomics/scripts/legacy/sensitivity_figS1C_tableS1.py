import os, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, spearmanr

def is_sample_col(c):
    return bool(re.match(r"^(QC|HS|NC|ZS)\d+$", str(c), flags=re.I))

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out

def pick_feature_id(df, meta_cols):
    candidates = ["Metabolite", "metabolite", "Name", "NAME", "Compound", "compound",
                  "KEGG", "kegg", "HMDB", "hmdb", "m/z", "mz", "Retention time (min)",
                  "RT", "rt", "Feature", "Peak", "ID", "id"]
    for c in candidates:
        if c in df.columns and c in meta_cols:
            return df[c].astype(str)

    mz_col = None
    rt_col = None
    for c in df.columns:
        if str(c).lower() in ["m/z","mz"]:
            mz_col = c
        if str(c).lower() in ["retention time (min)","rt"]:
            rt_col = c
    if mz_col is not None and rt_col is not None:
        return (df[mz_col].astype(str) + "_RT" + df[rt_col].astype(str))

    return pd.Series([f"F{i+1}" for i in range(len(df))])

def make_augmented_samples(X_group, n_aug=100, noise_scale=0.03, seed=0):
    rng = np.random.default_rng(seed)
    A = np.asarray(X_group, dtype=float)
    n_feat, n_real = A.shape
    idx = rng.integers(0, n_real, size=n_aug)
    base = A[:, idx]
    sd = np.nanstd(A, axis=1, ddof=1)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1e-6)
    noise = rng.normal(0, sd[:, None] * noise_scale, size=(n_feat, n_aug))
    return base + noise

def diff_stats(X, g1_cols, g2_cols, use_aug=False, n_aug=100, noise_scale=0.03, seed=0):
    A1 = X[g1_cols].values
    A2 = X[g2_cols].values

    if use_aug:
        A1 = make_augmented_samples(A1, n_aug=n_aug, noise_scale=noise_scale, seed=seed+1)
        A2 = make_augmented_samples(A2, n_aug=n_aug, noise_scale=noise_scale, seed=seed+2)

    m1 = np.nanmean(A1, axis=1)
    m2 = np.nanmean(A2, axis=1)
    log2fc = m1 - m2

    pvals = np.array([ttest_ind(A1[i, :], A2[i, :], equal_var=False, nan_policy="omit").pvalue
                      for i in range(A1.shape[0])], dtype=float)
    qvals = bh_fdr(pvals)
    return log2fc, pvals, qvals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="/mnt/h/Data/Yuchun-yanshi/数据矩阵.xlsx")
    ap.add_argument("--sheet", default="数据矩阵")
    ap.add_argument("--outdir", default="/mnt/h/Data/Yuchun-yanshi/QC_Figures")
    ap.add_argument("--exclude_nc", nargs="*", default=["NC1", "NC3"])
    ap.add_argument("--augment_n", type=int, default=100, help="pseudo-samples per group for effect stability")
    ap.add_argument("--noise_scale", type=float, default=0.03)
    ap.add_argument("--use_aug_for_stats", action="store_true",
                    help="If set, t-tests will be performed on augmented pseudo-samples (NOT recommended).")
    ap.add_argument("--sig_q", type=float, default=0.05)
    ap.add_argument("--sig_abs_fc", type=float, default=0.0, help="threshold on |log2FC|")
    ap.add_argument("--topN", type=int, default=20, help="top features to assess direction consistency")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_excel(args.xlsx, sheet_name=args.sheet)
    sample_cols = [c for c in df.columns if is_sample_col(c)]
    meta_cols = [c for c in df.columns if c not in sample_cols]

    feat_id = pick_feature_id(df, meta_cols)
    feat_id = feat_id.fillna("").astype(str)
    feat_id = feat_id.where(feat_id.str.len() > 0, pd.Series([f"F{i+1}" for i in range(len(df))]))
    feat_id = feat_id.astype(str)

    X = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.apply(lambda r: r.fillna(r.median()), axis=1)

    hs = [c for c in sample_cols if str(c).upper().startswith("HS")]
    nc_all = [c for c in sample_cols if str(c).upper().startswith("NC")]
    nc_keep = [c for c in nc_all if c not in set(args.exclude_nc)]

    if len(hs) < 2 or len(nc_all) < 2 or len(nc_keep) < 2:
        raise RuntimeError(f"Need enough samples. HS={hs}, NC_all={nc_all}, NC_keep={nc_keep}")

    log2fc_full, p_full, q_full = diff_stats(
        X, hs, nc_all,
        use_aug=args.use_aug_for_stats,
        n_aug=args.augment_n, noise_scale=args.noise_scale, seed=0
    )
    log2fc_sens, p_sens, q_sens = diff_stats(
        X, hs, nc_keep,
        use_aug=args.use_aug_for_stats,
        n_aug=args.augment_n, noise_scale=args.noise_scale, seed=10
    )

    # augmented effect stability only
    X_hs_aug = make_augmented_samples(X[hs].values, n_aug=args.augment_n,
                                      noise_scale=args.noise_scale, seed=123)
    X_nc_all_aug = make_augmented_samples(X[nc_all].values, n_aug=args.augment_n,
                                          noise_scale=args.noise_scale, seed=456)
    X_nc_keep_aug = make_augmented_samples(X[nc_keep].values, n_aug=args.augment_n,
                                           noise_scale=args.noise_scale, seed=789)

    log2fc_full_aug = np.mean(X_hs_aug, axis=1) - np.mean(X_nc_all_aug, axis=1)
    log2fc_sens_aug = np.mean(X_hs_aug, axis=1) - np.mean(X_nc_keep_aug, axis=1)

    sig_full = (q_full < args.sig_q) & (np.abs(log2fc_full) >= args.sig_abs_fc)
    sig_sens = (q_sens < args.sig_q) & (np.abs(log2fc_sens) >= args.sig_abs_fc)

    set_full = set(np.where(sig_full)[0].tolist())
    set_sens = set(np.where(sig_sens)[0].tolist())
    inter = set_full & set_sens
    union = set_full | set_sens
    jacc = (len(inter) / len(union)) if len(union) else np.nan
    overlap_full = (len(inter) / len(set_full)) if len(set_full) else np.nan
    overlap_sens = (len(inter) / len(set_sens)) if len(set_sens) else np.nan

    rho_real, _ = spearmanr(log2fc_full, log2fc_sens)
    rho_aug, _ = spearmanr(log2fc_full_aug, log2fc_sens_aug)

    top_idx = np.argsort(q_full)[:args.topN]
    dir_full = np.sign(log2fc_full[top_idx])
    dir_sens = np.sign(log2fc_sens[top_idx])
    dir_cons = float(np.mean(dir_full == dir_sens))

    # Save Table S1 Part II (suffix _2)
    tbl = pd.DataFrame([{
        "comparison": "HS vs NC",
        "full_NC_samples": ",".join(nc_all),
        "sens_NC_samples": ",".join(nc_keep),
        "HS_samples": ",".join(hs),
        "sig_definition": f"q<{args.sig_q} & |log2FC|>={args.sig_abs_fc}",
        "n_sig_full": int(len(set_full)),
        "n_sig_sensitivity": int(len(set_sens)),
        "n_intersection": int(len(inter)),
        "n_union": int(len(union)),
        "jaccard(sig_sets)": jacc,
        "intersection/Full": overlap_full,
        "intersection/Sensitivity": overlap_sens,
        "spearman_rho_log2FC_real": rho_real,
        "spearman_rho_log2FC_augMean": rho_aug,
        f"top{args.topN}_direction_consistency": dir_cons,
        "notes": ("p/q computed on augmented pseudo-samples" if args.use_aug_for_stats
                  else "p/q computed on real samples; augmentation used only for FC stability")
    }])

    out_csv = os.path.join(args.outdir, "TableS1_PartII_sensitivity_summary_2.csv")
    tbl.to_csv(out_csv, index=False)

    detail = pd.DataFrame({
        "feature": feat_id.values,
        "log2FC_full": log2fc_full,
        "q_full": q_full,
        "log2FC_sensitivity": log2fc_sens,
        "q_sensitivity": q_sens,
        "log2FC_full_augMean": log2fc_full_aug,
        "log2FC_sens_augMean": log2fc_sens_aug,
        "sig_full": sig_full,
        "sig_sensitivity": sig_sens
    })
    out_detail = os.path.join(args.outdir, "HSvsNC_FC_concordance_detail_2.csv")
    detail.to_csv(out_detail, index=False)

    # ===== Fig S1C: concordance scatter (NO TEXT ANNOTATION) =====
    x = log2fc_full_aug
    y = log2fc_sens_aug

    fig, ax = plt.subplots(figsize=(6.0, 5.6), dpi=300)
    ax.scatter(x, y, s=18, alpha=0.75, edgecolors="none")

    # y=x line
    lim = np.nanmax(np.abs(np.r_[x, y]))
    lim = max(lim, 1e-6)
    ax.plot([-lim, lim], [-lim, lim], lw=1.0, color="0.2")

    ax.axhline(0, lw=0.8, color="0.2")
    ax.axvline(0, lw=0.8, color="0.2")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # 去除任何文字：不设置 xlabel/ylabel/title，不添加 ax.text，不添加 feature label
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 去掉刻度文字也可以（如果你希望完全无文字）
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_png = os.path.join(args.outdir, "FigS1C_HSvsNC_log2FC_concordance_2.png")
    out_pdf = os.path.join(args.outdir, "FigS1C_HSvsNC_log2FC_concordance_2.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("✅ Saved:")
    print("  ", out_png)
    print("  ", out_pdf)
    print("  ", out_csv)
    print("  ", out_detail)
    print("NC excluded in sensitivity:", args.exclude_nc)

if __name__ == "__main__":
    main()

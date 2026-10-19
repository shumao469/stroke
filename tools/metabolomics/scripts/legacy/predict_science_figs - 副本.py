#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.calibration import calibration_curve

# ----------------------------
# Utils
# ----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def find_sheet(xl, candidates):
    for s in candidates:
        if s in xl.sheet_names:
            return s
    return xl.sheet_names[0]

def bootstrap_ci(y_true_bin, y_prob, fn, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true_bin)
    stats = []
    idx = np.arange(n)
    for _ in range(n_boot):
        b = rng.choice(idx, size=n, replace=True)
        stats.append(fn(y_true_bin[b], y_prob[b]))
    stats = np.array(stats)
    return np.percentile(stats, [2.5, 97.5])

def net_benefit_binary(y_true, y_prob, pt):
    # net benefit = TP/n - FP/n * (pt/(1-pt))
    y_pred = (y_prob >= pt).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)
    return (tp / n) - (fp / n) * (pt / (1 - pt))

def treat_all_nb(prevalence, pt):
    # treat-all net benefit in binary setting
    return prevalence - (1 - prevalence) * (pt / (1 - pt))

def safe_feature_names(cols):
    # 防止出现 nan / None
    out = []
    for c in cols:
        if c is None or (isinstance(c, float) and np.isnan(c)):
            out.append("UnknownFeature")
        else:
            out.append(str(c))
    return out

def augment_train_only(X_train, y_train, target_n=100, noise_scale=0.02, seed=0):
    """
    训练集内部做简单增强：在标准化前，对原始特征加小高斯噪声。
    注意：增强只用于训练拟合，不用于评估。
    """
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    if target_n <= n:
        return X_train, y_train
    extra = target_n - n

    idx = rng.choice(np.arange(n), size=extra, replace=True)
    X_extra = X_train[idx] + rng.normal(0, noise_scale, size=X_train[idx].shape)
    y_extra = y_train[idx]
    X_aug = np.vstack([X_train, X_extra])
    y_aug = np.concatenate([y_train, y_extra])
    return X_aug, y_aug

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="数据矩阵.xlsx", help="Excel file path")
    ap.add_argument("--sheet_data", default=None, help="Data sheet name (auto if None)")
    ap.add_argument("--sheet_group", default=None, help="Group sheet name (auto if None)")
    ap.add_argument("--drop", nargs="*", default=[], help="Samples to drop, e.g., NC1 NC3")
    ap.add_argument("--augment_n", type=int, default=0, help="If >0, augment TRAIN only to this size per outer fold")
    ap.add_argument("--outdir", default="Predict_Figures", help="Output dir")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    figdir = ensure_dir(os.path.join(outdir, "figures"))
    tabdir = ensure_dir(os.path.join(outdir, "tables"))

    xl = pd.ExcelFile(args.xlsx)
    sheet_data = args.sheet_data or find_sheet(xl, ["数据矩阵", "缺失值数据矩阵"])
    sheet_group = args.sheet_group or find_sheet(xl, ["分组", "Group", "group"])

    df = xl.parse(sheet_data)
    df_group = xl.parse(sheet_group)

    # 识别样本列（你已经知道：QC1-3, HS1-4, NC1-4, ZS1-4）
    sample_cols = [c for c in df.columns if isinstance(c, str) and (c.startswith("QC") or c.startswith("HS") or c.startswith("NC") or c.startswith("ZS"))]
    sample_cols = [c for c in sample_cols if c not in set(args.drop)]

    # feature cols：去掉“Retention time (min)”等注释列
    anno_cols = [c for c in df.columns if c not in sample_cols]
    # 常见注释列可能很多，这里保留整行作为 feature id；用 index 做特征名
    # 如果你有明确 metabolite name 列，可在这里替换 feature_names
    feature_names = safe_feature_names(df.index.to_list())

    X = df[sample_cols].T.values.astype(float)  # shape: (n_samples, n_features)
    sample_names = np.array(sample_cols)

    # 从分组表映射标签（要求分组表至少包含：样本名、组别）
    # 自动猜列名
    cols_lower = {c: str(c).lower() for c in df_group.columns}
    sample_col = None
    group_col = None
    for c in df_group.columns:
        if any(k in cols_lower[c] for k in ["sample", "样本"]):
            sample_col = c
        if any(k in cols_lower[c] for k in ["group", "分组", "类别"]):
            group_col = c
    if sample_col is None or group_col is None:
        # fallback: assume first 2 columns
        sample_col = df_group.columns[0]
        group_col = df_group.columns[1]

    gmap = dict(zip(df_group[sample_col].astype(str), df_group[group_col].astype(str)))
    y = np.array([gmap.get(s, "Unknown") for s in sample_names])

    # 只保留 NC/HS/ZS（QC 不进入预测模型；QC 用于质量体系图）
    keep = np.isin(y, ["NC", "HS", "ZS"])
    X_ml = X[keep]
    y_ml = y[keep]
    s_ml = sample_names[keep]

    classes = ["NC", "HS", "ZS"]
    assert set(np.unique(y_ml)).issubset(set(classes)), f"Unexpected labels: {np.unique(y_ml)}"

    # 模型：多分类 LogisticRegression（elastic-net）+ 标准化
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", multi_class="multinomial",
            max_iter=5000, n_jobs=-1
        ))
    ])

    param_grid = {
        "clf__C": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        "clf__l1_ratio": [0.0, 0.2, 0.5, 0.8, 1.0],
    }

    # 外层：50 folds（5-fold × 10 repeats）
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=args.seed)
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.seed)

    # out-of-fold 预测容器（严格：只存 outer test 的预测）
    oof_prob = np.full((len(y_ml), len(classes)), np.nan, dtype=float)
    oof_fold = np.full(len(y_ml), -1, dtype=int)

    fold_auc = []
    feat_counts = {}  # feature -> count selected (non-zero coef) across outer folds

    for fold_id, (tr, te) in enumerate(outer.split(X_ml, y_ml)):
        X_tr, y_tr = X_ml[tr], y_ml[tr]
        X_te, y_te = X_ml[te], y_ml[te]

        # 训练集内部可选“增强到100”（不影响 te）
        if args.augment_n and args.augment_n > 0:
            X_tr, y_tr = augment_train_only(X_tr, y_tr, target_n=args.augment_n, seed=args.seed + fold_id)

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc_ovr_weighted",
            cv=inner,
            n_jobs=-1,
            refit=True
        )
        gs.fit(X_tr, y_tr)

        prob_te = gs.predict_proba(X_te)
        oof_prob[te] = prob_te
        oof_fold[te] = fold_id

        # fold AUC（macro）
        y_te_int = np.array([classes.index(v) for v in y_te])
        y_te_onehot = np.eye(len(classes))[y_te_int]
        auc_macro = roc_auc_score(y_te_onehot, prob_te, average="macro", multi_class="ovr")
        fold_auc.append(auc_macro)

        # 特征稳定性：统计非零系数（multinomial -> coef_ shape (K, p)）
        best = gs.best_estimator_.named_steps["clf"]
        coef = best.coef_
        nonzero = np.where(np.max(np.abs(coef), axis=0) > 1e-8)[0]
        for j in nonzero:
            name = feature_names[j] if j < len(feature_names) else f"F{j}"
            feat_counts[name] = feat_counts.get(name, 0) + 1

    # -----------------------
    # Fig 2A: ROC (OVR) using OOF preds
    # -----------------------
    y_int = np.array([classes.index(v) for v in y_ml])
    y_oh = np.eye(len(classes))[y_int]

    plt.figure(figsize=(6, 6))
    for k, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_oh[:, k], oof_prob[:, k])
        a = auc(fpr, tpr)
        # CI (bootstrap)
        ci = bootstrap_ci(y_oh[:, k], oof_prob[:, k], lambda yt, yp: roc_auc_score(yt, yp), n_boot=400, seed=args.seed + k)
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
    # Fig 2B: Calibration (OVR) using OOF preds + bootstrap CI per bin
    # -----------------------
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], color="0.5", lw=1)
    for k, cls in enumerate(classes):
        prob = oof_prob[:, k]
        yt = y_oh[:, k].astype(int)

        # quantile bins
        n_bins = 5
        q = np.quantile(prob, np.linspace(0, 1, n_bins + 1))
        q[0] = -1e9
        q[-1] = 1e9

        bin_centers = []
        obs = []
        lo = []
        hi = []

        rng = np.random.default_rng(args.seed + 100 + k)
        for i in range(n_bins):
            m = (prob > q[i]) & (prob <= q[i + 1])
            if m.sum() < 3:
                continue
            p_mean = prob[m].mean()
            o_mean = yt[m].mean()

            # bootstrap CI for observed fraction in this bin
            boot = []
            idx = np.where(m)[0]
            for _ in range(300):
                b = rng.choice(idx, size=len(idx), replace=True)
                boot.append(yt[b].mean())
            lo_i, hi_i = np.percentile(boot, [2.5, 97.5])

            bin_centers.append(p_mean)
            obs.append(o_mean)
            lo.append(lo_i)
            hi.append(hi_i)

        bin_centers = np.array(bin_centers)
        obs = np.array(obs)
        lo = np.array(lo)
        hi = np.array(hi)

        plt.plot(bin_centers, obs, marker="o", lw=2, label=f"{cls}")
        plt.fill_between(bin_centers, lo, hi, alpha=0.15)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction")
    plt.title("Calibration (one-vs-rest, out-of-fold)")
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "Fig2B_Calibration_multiclass_OVR_OOF.png"), dpi=300)
    plt.savefig(os.path.join(figdir, "Fig2B_Calibration_multiclass_OVR_OOF.pdf"))
    plt.close()

    # -----------------------
    # Fig 2C: Decision Curve Analysis (OVR) using OOF preds (with treat-all/none)
    # -----------------------
    pts = np.linspace(0.05, 0.95, 19)
    plt.figure(figsize=(6.5, 5.2))
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
    # ExtData: AUC distribution (50 outer folds)
    # -----------------------
    fold_auc = np.array(fold_auc)
    plt.figure(figsize=(4.5, 5))
    plt.violinplot(fold_auc, showmeans=False, showmedians=True)
    # jitter points
    x = np.ones_like(fold_auc)
    rng = np.random.default_rng(args.seed)
    plt.scatter(x + rng.normal(0, 0.03, size=len(fold_auc)), fold_auc, s=18, alpha=0.6)
    plt.xticks([1], ["macro-AUC"])
    plt.ylabel("AUC (outer folds)")
    plt.title(f"Nested CV stability (n={len(fold_auc)} outer folds)")
    plt.ylim(max(0.4, fold_auc.min()-0.05), min(1.01, fold_auc.max()+0.02))
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "ExtData_AUC_distribution_OOF.png"), dpi=300)
    plt.savefig(os.path.join(figdir, "ExtData_AUC_distribution_OOF.pdf"))
    plt.close()

    # -----------------------
    # ExtData: Feature selection frequency (Top 20)
    # -----------------------
    feat_df = pd.DataFrame({
        "feature": list(feat_counts.keys()),
        "selected_folds": list(feat_counts.values()),
    })
    feat_df["freq"] = feat_df["selected_folds"] / len(fold_auc)
    feat_df = feat_df.sort_values("freq", ascending=False).reset_index(drop=True)

    feat_df.to_csv(os.path.join(tabdir, "Table_FeatureSelectionFrequency_all.csv"), index=False)

    top = feat_df.head(20).iloc[::-1]
    plt.figure(figsize=(6.8, 5.6))
    plt.barh(top["feature"], top["freq"])
    plt.xlabel("Selection frequency (outer folds)")
    plt.title("Top biomarkers by selection stability (nested CV)")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "ExtData_FeatureFreq_Top20_OOF.png"), dpi=300)
    plt.savefig(os.path.join(figdir, "ExtData_FeatureFreq_Top20_OOF.pdf"))
    plt.close()

    # -----------------------
    # (Optional) SHAP
    # -----------------------
    shap_ok = True
    try:
        import shap
    except Exception as e:
        shap_ok = False

    if shap_ok:
        # 用全数据拟合一个最终模型用于解释（注意：解释不是性能评估）
        final = GridSearchCV(pipe, param_grid, scoring="roc_auc_ovr_weighted", cv=inner, n_jobs=-1, refit=True)
        final.fit(X_ml, y_ml)
        model = final.best_estimator_

        # 取标准化后的特征矩阵
        Xz = model.named_steps["scaler"].transform(X_ml)

        explainer = shap.LinearExplainer(model.named_steps["clf"], Xz, feature_perturbation="interventional")
        shap_values = explainer.shap_values(Xz)  # list(K) for multiclass

        # summary bar: mean |shap| across classes
        mean_abs = np.zeros(Xz.shape[1])
        for k in range(len(classes)):
            mean_abs += np.mean(np.abs(shap_values[k]), axis=0)
        mean_abs /= len(classes)

        idx = np.argsort(mean_abs)[::-1][:10]
        top_feat = [feature_names[i] for i in idx]
        top_val = mean_abs[idx][::-1]

        plt.figure(figsize=(6.5, 4.2))
        plt.barh(top_feat[::-1], top_val)
        plt.xlabel("Mean |SHAP| (avg across classes)")
        plt.title("SHAP summary (Top 10)")
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, "ExtData_SHAP_summary_top10.png"), dpi=300)
        plt.savefig(os.path.join(figdir, "ExtData_SHAP_summary_top10.pdf"))
        plt.close()

        # dependence plots for top 3 features (colored by class)
        top3 = idx[:3]
        for j in top3:
            plt.figure(figsize=(5.6, 4.6))
            for cls in classes:
                m = (y_ml == cls)
                plt.scatter(Xz[m, j], oof_prob[m, classes.index(cls)], s=18, alpha=0.7, label=cls)
            plt.xlabel(f"Z-scored feature: {feature_names[j]}")
            plt.ylabel("OOF predicted probability (class)")
            plt.title("Dependence (feature vs predicted prob)")
            plt.legend(frameon=False)
            plt.tight_layout()
            fn = f"ExtData_SHAP_dependence_{feature_names[j].replace('/','_')}.png"
            plt.savefig(os.path.join(figdir, fn), dpi=300)
            plt.close()

    # summary table
    summary = pd.DataFrame({
        "metric": ["macroAUC_mean", "macroAUC_median", "macroAUC_min", "macroAUC_max", "n_samples_used"],
        "value": [fold_auc.mean(), np.median(fold_auc), fold_auc.min(), fold_auc.max(), len(y_ml)]
    })
    summary.to_csv(os.path.join(tabdir, "Table_ModelPerformanceSummary.csv"), index=False)

    print("✅ Done.")
    print(f"Figures: {figdir}")
    print(f"Tables : {tabdir}")

if __name__ == "__main__":
    main()

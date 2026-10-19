#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import median_abs_deviation

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


# ----------------------------
# Utils
# ----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def find_sample_columns(df, pattern=r"^(NC|HS|ZS|QC)\d+$"):
    sample_col_pattern = re.compile(pattern, re.IGNORECASE)
    sample_cols = [c for c in df.columns if sample_col_pattern.match(str(c).strip())]
    return sample_cols


def pick_feature_name_series(df):
    # Prefer metabolite name for feature id; fallback to ID; else index
    for col in ["Metabolites", "Metabolites_cn", "ID"]:
        if col in df.columns:
            s = df[col].astype(str).fillna("")
            # if empty too much, fallback
            if (s.str.len() > 0).mean() > 0.2:
                return s
    return pd.Series([f"F{i}" for i in range(df.shape[0])])


def load_label_map_from_hs_related(xlsx_path):
    """
    Try to find columns like SampleID + Group/label/组别 from any sheet.
    Returns dict sample->group_str or empty dict.
    """
    if not xlsx_path or (not os.path.exists(xlsx_path)):
        return {}

    xls = pd.ExcelFile(xlsx_path)
    label_map = {}
    candidate_sample_cols = ["SampleID", "sample_id", "ID", "样本", "样本ID", "样本编号"]
    candidate_group_cols = ["Group", "group", "Label", "label", "组别", "分组", "类别"]

    for sh in xls.sheet_names:
        try:
            d = pd.read_excel(xlsx_path, sheet_name=sh)
        except Exception:
            continue
        cols = {c: c for c in d.columns}
        sample_col = None
        group_col = None
        for c in candidate_sample_cols:
            if c in cols:
                sample_col = c
                break
        for c in candidate_group_cols:
            if c in cols:
                group_col = c
                break
        if sample_col and group_col:
            tmp = d[[sample_col, group_col]].dropna()
            for sid, g in tmp.values:
                sid = str(sid).strip()
                g = str(g).strip()
                if sid and g:
                    label_map[sid] = g
    return label_map


def infer_group_from_sample_id(sample_id):
    m = re.match(r"^(NC|HS|ZS|QC)", str(sample_id).strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return "UNK"


def ece_binary(y_true01, p, n_bins=10, strategy="uniform"):
    """
    Expected calibration error for binary probabilities.
    """
    y_true01 = np.asarray(y_true01).astype(int)
    p = np.asarray(p).astype(float)
    if strategy == "quantile":
        # quantile bins may collapse in tiny sample; handle duplicates safely
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(p, qs)
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
        if len(edges) <= 2:
            edges = np.array([0.0, 1.0])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    n = len(p)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        acc = y_true01[mask].mean()
        conf = p[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def compute_dca_binary(y_true01, p, thresholds):
    """
    Net benefit for binary classifier:
      NB(t) = TP/n - FP/n * (t/(1-t))
    """
    y_true01 = np.asarray(y_true01).astype(int)
    p = np.asarray(p).astype(float)
    n = len(y_true01)
    out = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = ((pred == 1) & (y_true01 == 1)).sum()
        fp = ((pred == 1) & (y_true01 == 0)).sum()
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        out.append(nb)
    return np.array(out, dtype=float)


def stratified_jitter_augment(X_train, y_train, n_copies_per_sample=20, jitter_strength=0.03, seed=0):
    """
    Training-fold-only feature-wise jitter:
      x_aug = x + Normal(0, jitter_strength * MAD_feature)
    Works even when a class has only 1 sample.
    """
    rng = np.random.default_rng(seed)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    mad = median_abs_deviation(X_train, axis=0, nan_policy="omit")
    mad = np.where((mad == 0) | ~np.isfinite(mad), 1.0, mad)

    X_aug_list = []
    y_aug_list = []

    for x, lab in zip(X_train, y_train):
        noise = rng.standard_normal(size=(n_copies_per_sample, X_train.shape[1])) * (jitter_strength * mad)
        X_aug_list.append(x[None, :] + noise)
        y_aug_list.append(np.full((n_copies_per_sample,), lab))

    X_aug = np.vstack(X_aug_list)
    y_aug = np.concatenate(y_aug_list)
    return X_aug, y_aug


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx_matrix", type=str, required=True, help="数据矩阵.xlsx 路径")
    ap.add_argument("--sheet", type=str, default="缺失值数据矩阵", help="数据矩阵 sheet 名称")
    ap.add_argument("--xlsx_hs_related", type=str, default="", help="HS_related_all_tables.xlsx (可选，用于SampleID->Group映射)")
    ap.add_argument("--out_dir", type=str, default="out_v6_6", help="输出目录")

    ap.add_argument("--drop_qc", action="store_true", help="是否丢弃 QC 样本 (推荐)")
    ap.add_argument("--n_splits", type=int, default=2, help="2-fold 推荐（小样本更稳）")
    ap.add_argument("--n_repeats", type=int, default=100, help="重复次数，50~200")
    ap.add_argument("--random_state", type=int, default=42)

    ap.add_argument("--jitter_strength", type=float, default=0.03, help="jitter 强度，0.02~0.05")
    ap.add_argument("--copies_per_sample", type=int, default=25, help="每个训练样本复制多少次")
    ap.add_argument("--rf_estimators", type=int, default=500, help="RF 树数")
    ap.add_argument("--do_shap", action="store_true", help="是否做 SHAP（会慢一点，但 n 小问题不大）")

    ap.add_argument("--dca_task", type=str, default="HSZS_vs_NC", choices=["HSZS_vs_NC"], help="目前仅实现 HSZS_vs_NC")
    ap.add_argument("--dca_tmin", type=float, default=0.05, help="DCA阈值下限（建议0.05或0.1）")
    ap.add_argument("--dca_tmax", type=float, default=0.80, help="DCA阈值上限（小样本不建议画到>0.8）")
    ap.add_argument("--dca_n", type=int, default=40, help="DCA阈值点数量")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # 1) Load matrix
    df = pd.read_excel(args.xlsx_matrix, sheet_name=args.sheet)
    sample_cols = find_sample_columns(df)
    if len(sample_cols) == 0:
        raise ValueError("未识别到样本列（NC1/HS1/ZS1/QC1...）。请检查 sheet 是否正确。")

    feat_names = pick_feature_name_series(df).tolist()

    X_samples = df[sample_cols].T  # samples x features
    X_samples.columns = feat_names
    sample_ids = X_samples.index.astype(str).tolist()

    # 2) Labels: from HS_related mapping if possible, else from prefix
    label_map = load_label_map_from_hs_related(args.xlsx_hs_related) if args.xlsx_hs_related else {}
    y_str = []
    for sid in sample_ids:
        if sid in label_map:
            y_str.append(str(label_map[sid]).upper())
        else:
            y_str.append(infer_group_from_sample_id(sid))

    meta = pd.DataFrame({"SampleID": sample_ids, "Group": y_str}).set_index("SampleID")

    # drop QC if required
    if args.drop_qc:
        keep = meta["Group"].isin(["NC", "HS", "ZS"])
        X_samples = X_samples.loc[keep.values]
        meta = meta.loc[keep.values]

    # final X,y
    groups = meta["Group"].values
    classes = np.unique(groups)
    # stable order
    classes = np.array([c for c in ["NC", "ZS", "HS"] if c in classes] + [c for c in classes if c not in ["NC", "ZS", "HS"]])
    class_to_int = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_int[g] for g in groups], dtype=int)

    X = X_samples.values.astype(float)

    # sanity
    vc = pd.Series(groups).value_counts()
    print("Class counts:\n", vc.to_string())
    if vc.min() < 2 and args.n_splits >= 2:
        warnings.warn("某个类别样本数<2，RepeatedStratifiedKFold 可能会失败或非常不稳。建议不要 drop 太多样本。")

    # 3) Repeated Stratified KFold
    cv = RepeatedStratifiedKFold(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state
    )

    # Collect per-iteration metrics + OOF probability accumulator
    n_classes = len(classes)
    proba_sum = np.zeros((X.shape[0], n_classes), dtype=float)
    proba_cnt = np.zeros((X.shape[0],), dtype=int)

    iter_rows = []

    for it, (tr, te) in enumerate(cv.split(X, y)):
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        # training-fold-only augmentation (jitter)
        X_aug, y_aug = stratified_jitter_augment(
            X_tr, y_tr,
            n_copies_per_sample=args.copies_per_sample,
            jitter_strength=args.jitter_strength,
            seed=args.random_state + it
        )

        # model
        model = RandomForestClassifier(
            n_estimators=args.rf_estimators,
            random_state=args.random_state + it,
            class_weight="balanced_subsample",
            n_jobs=-1
        )
        model.fit(X_aug, y_aug)

        p_te = model.predict_proba(X_te)  # shape (n_test, n_classes)

        # accumulate OOF (mean over times a sample appears in test across repeats)
        proba_sum[te] += p_te
        proba_cnt[te] += 1

        # metrics for this iteration
        try:
            auc_ovr = roc_auc_score(y_te, p_te, multi_class="ovr")
        except Exception:
            auc_ovr = np.nan

        # macro AUPRC (OVR)
        aps = []
        for k in range(n_classes):
            yk = (y_te == k).astype(int)
            if yk.sum() == 0 or yk.sum() == len(yk):
                aps.append(np.nan)
            else:
                aps.append(average_precision_score(yk, p_te[:, k]))
        auprc_macro = np.nanmean(aps)

        iter_rows.append({"iter": it, "AUROC_ovr": auc_ovr, "AUPRC_macro": auprc_macro})

    iter_df = pd.DataFrame(iter_rows)
    iter_df.to_csv(os.path.join(args.out_dir, "RepeatedCV_iter_metrics.csv"), index=False)

    # 4) Final OOF prob (per-sample)
    if (proba_cnt == 0).any():
        warnings.warn("有样本从未进入测试折（理论上不该发生）。检查 n_splits/n_repeats 设置。")
    proba_mean = proba_sum / np.maximum(proba_cnt[:, None], 1)

    oof_df = pd.DataFrame(proba_mean, columns=[f"p_{c}" for c in classes], index=X_samples.index)
    oof_df.insert(0, "y_true", y)
    oof_df.insert(0, "Group", groups)
    oof_df.insert(0, "SampleID", X_samples.index.astype(str).values)
    oof_df.to_csv(os.path.join(args.out_dir, "OOF_pred_probs.csv"), index=False)

    # 5) Plot 1: AUROC distribution (defensive / robust)
    plt.figure(figsize=(6, 4))
    vals = iter_df["AUROC_ovr"].dropna().values
    plt.boxplot(vals, vert=True, showfliers=False)
    plt.axhline(0.5, linestyle="--")
    plt.ylabel("AUROC (OVR) per split")
    plt.title(f"Repeated {args.n_splits}-fold CV × {args.n_repeats} (jitter train-only)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "Fig_AUROC_distribution_repeatedCV.png"), dpi=300)
    plt.close()

    # 6) Plot 2: ROC curves from aggregated OOF probs (OVR)
    plt.figure(figsize=(6, 5))
    for k, c in enumerate(classes):
        yk = (y == k).astype(int)
        if yk.sum() == 0 or yk.sum() == len(yk):
            continue
        fpr, tpr, _ = roc_curve(yk, proba_mean[:, k])
        auc_k = roc_auc_score(yk, proba_mean[:, k])
        plt.plot(fpr, tpr, label=f"{c} (AUC={auc_k:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Multiclass ROC (OVR, aggregated OOF probs)")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "Fig_ROC_OVR_OOF_aggregated.png"), dpi=300)
    plt.close()

    # 7) Plot 3: PR curves from aggregated OOF probs (OVR)
    plt.figure(figsize=(6, 5))
    ap_list = []
    for k, c in enumerate(classes):
        yk = (y == k).astype(int)
        if yk.sum() == 0 or yk.sum() == len(yk):
            continue
        prec, rec, _ = precision_recall_curve(yk, proba_mean[:, k])
        ap_k = average_precision_score(yk, proba_mean[:, k])
        ap_list.append(ap_k)
        plt.plot(rec, prec, label=f"{c} (AP={ap_k:.3f})")
    if len(ap_list) > 0:
        plt.text(0.55, 0.05, f"Macro-AUPRC={np.mean(ap_list):.3f}", fontsize=12)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multiclass PR (OVR, aggregated OOF probs)")
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "Fig_PR_OVR_OOF_aggregated.png"), dpi=300)
    plt.close()

    # 8) Plot 4: Calibration (preferred small-n form + optional binned curve)
    # Use slope/intercept style metrics (approx) + binned reliability
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")

    brier_list = []
    ece_list = []
    for k, c in enumerate(classes):
        yk = (y == k).astype(int)
        pk = proba_mean[:, k]
        if yk.sum() == 0 or yk.sum() == len(yk):
            continue

        # Brier
        brier = brier_score_loss(yk, pk)
        brier_list.append(brier)

        # ECE (use uniform bins; tiny sample, keep bins small)
        ece = ece_binary(yk, pk, n_bins=3, strategy="uniform")
        ece_list.append(ece)

        # reliability curve
        prob_true, prob_pred = calibration_curve(yk, pk, n_bins=3, strategy="uniform")
        plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label=f"{c}")

    if len(brier_list) > 0:
        plt.text(0.05, 0.92, f"Macro Brier={np.mean(brier_list):.3f}", transform=plt.gca().transAxes, fontsize=11)
    if len(ece_list) > 0:
        plt.text(0.05, 0.86, f"Macro ECE={np.mean(ece_list):.3f}", transform=plt.gca().transAxes, fontsize=11)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction")
    plt.title("Calibration (OVR, aggregated OOF probs)")
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "Fig_Calibration_OVR_OOF_aggregated.png"), dpi=300)
    plt.close()

    # 9) Plot 5: DCA for HSZS vs NC
    # Define positive = HS or ZS ; negative = NC
    if args.dca_task == "HSZS_vs_NC":
        # p_case = 1 - p_NC (if NC exists)
        if "NC" not in classes:
            warnings.warn("DCA 需要 NC 类，但当前 classes 中没有 NC。跳过 DCA。")
        else:
            idx_nc = np.where(classes == "NC")[0][0]
            p_case = 1.0 - proba_mean[:, idx_nc]
            y_case = (groups != "NC").astype(int)  # HS/ZS=1, NC=0

            ts = np.linspace(args.dca_tmin, args.dca_tmax, args.dca_n)
            nb_model = compute_dca_binary(y_case, p_case, ts)

            # Treat-all baseline: predict everyone positive
            # NB_all(t) = prevalence - (1-prevalence)*t/(1-t)
            prev = y_case.mean()
            nb_all = prev - (1 - prev) * (ts / (1 - ts))
            nb_none = np.zeros_like(ts)

            plt.figure(figsize=(6, 5))
            plt.plot(ts, nb_none, linestyle=":", linewidth=2, label="Treat none")
            plt.plot(ts, nb_all, linestyle="--", linewidth=2, label="Treat all")
            plt.plot(ts, nb_model, linewidth=3, label="Model")

            plt.xlabel("Threshold probability")
            plt.ylabel("Net benefit")
            plt.title("Decision curve analysis (HS/ZS vs NC, aggregated OOF)")
            plt.legend(loc="lower left", frameon=True)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "Fig_DCA_HSZS_vs_NC_OOF_aggregated.png"), dpi=300)
            plt.close()

    # 10) Optional SHAP: fit one final model on full data (train-only augmentation on full set)
    if args.do_shap:
        try:
            import shap
            # Full augmentation (for explanation only; strictly speaking this is "refit" not OOF)
            X_aug_full, y_aug_full = stratified_jitter_augment(
                X, y,
                n_copies_per_sample=args.copies_per_sample,
                jitter_strength=args.jitter_strength,
                seed=args.random_state + 999
            )
            model_full = RandomForestClassifier(
                n_estimators=args.rf_estimators,
                random_state=args.random_state + 999,
                class_weight="balanced_subsample",
                n_jobs=-1
            )
            model_full.fit(X_aug_full, y_aug_full)

            explainer = shap.TreeExplainer(model_full)
            shap_values = explainer.shap_values(X)  # list[n_classes] of (n_samples, n_features)

            # Save mean(|SHAP|) per class
            rows = []
            for k, c in enumerate(classes):
                sv = shap_values[k]
                imp = np.mean(np.abs(sv), axis=0)
                for j, fn in enumerate(feat_names):
                    rows.append({"Class": c, "Feature": fn, "MeanAbsSHAP": float(imp[j])})
            shap_df = pd.DataFrame(rows)
            shap_df.to_csv(os.path.join(args.out_dir, "SHAP_mean_abs_per_class.csv"), index=False)

            # Plot top-20 for each class
            for k, c in enumerate(classes):
                sub = shap_df[shap_df["Class"] == c].sort_values("MeanAbsSHAP", ascending=False).head(20)
                plt.figure(figsize=(7, 5))
                plt.barh(np.arange(len(sub))[::-1], sub["MeanAbsSHAP"].values[::-1])
                plt.yticks(np.arange(len(sub))[::-1], sub["Feature"].values[::-1])
                plt.xlabel("Mean |SHAP|")
                plt.title(f"Top features by SHAP (refit model) – {c}")
                plt.tight_layout()
                plt.savefig(os.path.join(args.out_dir, f"Fig_SHAP_Top20_{c}.png"), dpi=300)
                plt.close()

        except Exception as e:
            warnings.warn(f"SHAP skipped due to: {e}")

    print(f"\n✅ Done. Outputs saved to: {args.out_dir}")
    print("Key files:")
    print(" - OOF_pred_probs.csv (逐样本 OOF 概率，后续重画所有曲线/LOO influence 的核心)")
    print(" - RepeatedCV_iter_metrics.csv (每次split的AUROC/AUPRC，用于稳定性分布图)")
    print(" - Fig_*png (ROC/PR/Calibration/DCA/分布图)")
    if args.do_shap:
        print(" - SHAP_mean_abs_per_class.csv + Fig_SHAP_Top20_*.png (refit解释用)")

if __name__ == "__main__":
    main()

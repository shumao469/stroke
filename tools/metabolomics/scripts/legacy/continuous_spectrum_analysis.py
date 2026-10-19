#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
import statsmodels.api as sm

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _norm_colname(c):
    c0 = str(c).strip()
    c0 = re.sub(r"\s+", "", c0)
    return c0

def load_table(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")

def parse_dt(x):
    if pd.isna(x):
        return pd.NaT
    return pd.to_datetime(x, errors="coerce")

def coerce_binary(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ["1","y","yes","true","是","有","阳性","+"]:
        return 1
    if s in ["0","n","no","false","否","无","阴性","-"]:
        return 0
    return np.nan

def make_reg_df(clin, oof):
    # 统一列名
    clin = clin.copy()
    clin.columns = [_norm_colname(c) for c in clin.columns]

    # 兼容一些常见写法（你如果列名不同，在这里补映射）
    rename_map = {}
    for cand in ["SampleID","样本ID","样本编号","样本号","样本"]:
        if cand in clin.columns:
            rename_map[cand] = "SampleID"
            break
    for cand in ["组别","Group","group","分组"]:
        if cand in clin.columns:
            rename_map[cand] = "Group"
            break
    for cand in ["NIHSS","入院NIHSS","NIHSS评分"]:
        if cand in clin.columns:
            rename_map[cand] = "NIHSS"
            break
    for cand in ["年龄","Age"]:
        if cand in clin.columns:
            rename_map[cand] = "Age"
            break
    for cand in ["性别","Sex","Gender"]:
        if cand in clin.columns:
            rename_map[cand] = "Sex"
            break
    for cand in ["发病时间","起病时间","OnsetTime"]:
        if cand in clin.columns:
            rename_map[cand] = "OnsetTime"
            break
    for cand in ["采样时间","取样时间","SamplingTime","SampleTime"]:
        if cand in clin.columns:
            rename_map[cand] = "SamplingTime"
            break
    for cand in ["首次到院时间","到院时间","ArrivalTime","FirstArrivalTime"]:
        if cand in clin.columns:
            rename_map[cand] = "ArrivalTime"
            break
    for cand in ["HTN","高血压","Hypertension"]:
        if cand in clin.columns:
            rename_map[cand] = "HTN"
            break
    for cand in ["DM","糖尿病","Diabetes"]:
        if cand in clin.columns:
            rename_map[cand] = "DM"
            break
    for cand in ["取栓溶栓","取栓/溶栓","ThrombectomyOrThrombolysis","EVT_tPA"]:
        if cand in clin.columns:
            rename_map[cand] = "Recanalization"
            break

    clin = clin.rename(columns=rename_map)
    assert "SampleID" in clin.columns, "clinical 表中找不到 SampleID（或样本编号）列，请在脚本 rename_map 里加映射"

    # datetime
    for c in ["OnsetTime","SamplingTime","ArrivalTime"]:
        if c in clin.columns:
            clin[c] = clin[c].apply(parse_dt)

    # onset_hours：优先用显式列，否则用 SamplingTime - OnsetTime
    if "OnsetHours" in clin.columns:
        clin["OnsetHours"] = pd.to_numeric(clin["OnsetHours"], errors="coerce")
    else:
        clin["OnsetHours"] = np.nan
        if "OnsetTime" in clin.columns and "SamplingTime" in clin.columns:
            dt = (clin["SamplingTime"] - clin["OnsetTime"]).dt.total_seconds() / 3600.0
            clin["OnsetHours"] = dt

    # 处理二值协变量
    if "HTN" in clin.columns:
        clin["HTN"] = clin["HTN"].apply(coerce_binary)
    if "DM" in clin.columns:
        clin["DM"] = clin["DM"].apply(coerce_binary)
    if "Recanalization" in clin.columns:
        clin["Recanalization"] = clin["Recanalization"].apply(coerce_binary)

    # Sex -> 0/1（女=0 男=1；识别不了就 NaN）
    if "Sex" in clin.columns:
        def _sex(x):
            if pd.isna(x): return np.nan
            s = str(x).strip().lower()
            if s in ["m","male","男","1"]: return 1
            if s in ["f","female","女","0"]: return 0
            return np.nan
        clin["Sex01"] = clin["Sex"].apply(_sex)
    else:
        clin["Sex01"] = np.nan

    # 合并 OOF
    oof = oof.copy()
    oof.columns = [_norm_colname(c) for c in oof.columns]
    if "SampleID" not in oof.columns:
        raise ValueError("OOF_pred_probs.csv 里必须有 SampleID 列")
    merged = clin.merge(oof, on="SampleID", how="left")

    # risk（默认：1 - p_NC）
    if "risk_stroke" not in merged.columns:
        if "p_NC" in merged.columns:
            merged["risk_stroke"] = 1.0 - merged["p_NC"]
        else:
            raise ValueError("OOF 文件里缺少 risk_stroke 或 p_NC")

    return merged

def spearman_table(df, ycol="risk_stroke", xcols=("NIHSS","OnsetHours"), group_col="Group"):
    rows = []
    for g, d in [("ALL", df)] + list(df.groupby(group_col, dropna=False)):
        for x in xcols:
            if x not in d.columns: 
                continue
            dd = d[[x, ycol]].dropna()
            if len(dd) < 5:
                rows.append({"Group": g, "X": x, "n": len(dd), "rho": np.nan, "p": np.nan})
                continue
            rho, p = spearmanr(dd[x].values, dd[ycol].values)
            rows.append({"Group": g, "X": x, "n": len(dd), "rho": float(rho), "p": float(p)})
    return pd.DataFrame(rows)

def ols_adjusted(df, y="risk_stroke", x="NIHSS", covars=("Age","Sex01","HTN","DM"), group_col="Group"):
    use_cols = [y, x, group_col] + [c for c in covars if c in df.columns]
    d = df[use_cols].copy().dropna(subset=[y, x])
    # 构造设计矩阵：x + covars（可选）
    X = pd.DataFrame({"Intercept": 1.0, x: pd.to_numeric(d[x], errors="coerce")})
    for c in covars:
        if c in d.columns:
            X[c] = pd.to_numeric(d[c], errors="coerce")
    X = X.dropna()
    d = d.loc[X.index]
    yv = pd.to_numeric(d[y], errors="coerce")
    m = sm.OLS(yv, X).fit(cov_type="HC3")  # robust SE
    out = pd.DataFrame({
        "term": m.params.index,
        "coef": m.params.values,
        "se_robust": m.bse.values,
        "p": m.pvalues.values
    })
    return out, m

def scatter_regplot(df, x, y="risk_stroke", hue="Group", outpath="plot.png", title=None):
    d = df[[x, y, hue]].copy()
    d = d.dropna(subset=[x, y])
    if d.empty:
        print(f"[WARN] No data for {x} vs {y}")
        return

    # overall regression (OLS) for CI band
    X = sm.add_constant(pd.to_numeric(d[x], errors="coerce"))
    yv = pd.to_numeric(d[y], errors="coerce")
    m = sm.OLS(yv, X).fit()

    x_grid = np.linspace(np.nanmin(d[x]), np.nanmax(d[x]), 200)
    Xg = sm.add_constant(x_grid)
    pred = m.get_prediction(Xg).summary_frame(alpha=0.05)

    plt.figure(figsize=(6.2, 5.2))
    # scatter by group
    for g, gg in d.groupby(hue, dropna=False):
        plt.scatter(gg[x], gg[y], s=35, alpha=0.75, label=str(g))

    # CI band + line
    plt.plot(x_grid, pred["mean"].values, lw=2)
    plt.fill_between(x_grid, pred["mean_ci_lower"].values, pred["mean_ci_upper"].values, alpha=0.15)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title or f"{y} vs {x} (OOF)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True, help="clinical_extract_out.csv/xlsx")
    ap.add_argument("--oof", required=True, help="OOF_pred_probs.csv")
    ap.add_argument("--outdir", default="Clinical_ContinuousSpectrum_Out")
    ap.add_argument("--y", default="risk_stroke", help="risk column in merged table")
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    figdir = ensure_dir(os.path.join(outdir, "figures"))
    tabdir = ensure_dir(os.path.join(outdir, "tables"))

    clin = load_table(args.clinical)
    oof = load_table(args.oof)

    df = make_reg_df(clin, oof)
    df.to_csv(os.path.join(tabdir, "clinical_plus_oof_merged.csv"), index=False, encoding="utf-8-sig")

    # Spearman
    sp = spearman_table(df, ycol=args.y, xcols=("NIHSS","OnsetHours"), group_col="Group" if "Group" in df.columns else "Group")
    sp.to_csv(os.path.join(tabdir, "Table_Spearman_risk_vs_clinical.csv"), index=False, encoding="utf-8-sig")

    # Adjusted regression (optional)
    if "NIHSS" in df.columns:
        reg_tab, _ = ols_adjusted(df, y=args.y, x="NIHSS", covars=("Age","Sex01","HTN","DM"))
        reg_tab.to_csv(os.path.join(tabdir, "Table_OLS_adjusted_risk_vs_NIHSS.csv"), index=False, encoding="utf-8-sig")

    # Plots
    if "NIHSS" in df.columns:
        scatter_regplot(
            df, x="NIHSS", y=args.y, hue=("Group" if "Group" in df.columns else "Group"),
            outpath=os.path.join(figdir, "Fig_OOF_risk_vs_NIHSS.png"),
            title="OOF stroke risk vs NIHSS (with 95% CI)"
        )

    if "OnsetHours" in df.columns:
        scatter_regplot(
            df, x="OnsetHours", y=args.y, hue=("Group" if "Group" in df.columns else "Group"),
            outpath=os.path.join(figdir, "Fig_OOF_risk_vs_OnsetHours.png"),
            title="OOF stroke risk vs onset-to-sampling (hours) (with 95% CI)"
        )

    print("✅ Continuous spectrum analysis done.")
    print("Merged:", os.path.join(tabdir, "clinical_plus_oof_merged.csv"))
    print("Tables:", tabdir)
    print("Figs  :", figdir)

if __name__ == "__main__":
    main()

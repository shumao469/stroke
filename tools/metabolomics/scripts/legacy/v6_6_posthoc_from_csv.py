
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v6_6_posthoc_from_csv.py

Post-hoc "publishable" plots from OOF summary CSVs only.

Required inputs:
  - OOF_metrics_with_CI.csv
  - OOF_calibration_metrics.csv

Produces:
  - Fig2A_AUROC_forest_OOF.(png|pdf)
  - Fig2A_AUPRC_forest_OOF.(png|pdf)
  - Fig2B_Calibration_summary_OOF.(png|pdf)

Limitation:
- Without per-sample OOF probabilities, this cannot redraw ROC/PR curves or a true reliability diagram.
"""
import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_dir(d): os.makedirs(d, exist_ok=True)

def _read_required(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _metric_name(metric: str) -> str:
    m = metric.upper()
    if m.startswith("MACRO"):
        return "Macro"
    # patterns
    for cls in ["HS","NC","ZS"]:
        if re.search(rf"\b{cls}\b", m) or f"_{cls}_" in m or m.endswith(f"_{cls}") or m.startswith(f"{cls}_"):
            return cls
    return metric

def forest_plot(met_df: pd.DataFrame, kind: str, out_base: str, ref_line: float | None = None):
    d = met_df.copy()
    # normalize
    d.columns = [c.strip() for c in d.columns]
    metric_col = _pick_col(d, ["metric","Metric","name"])
    val_col    = _pick_col(d, ["value","Value","mean"])
    lo_col     = _pick_col(d, ["ci_low","ci_lo","CI_low","CI_lo","lower","lo"])
    hi_col     = _pick_col(d, ["ci_high","ci_hi","CI_high","CI_hi","upper","hi"])
    if metric_col is None or val_col is None:
        raise ValueError(f"metrics csv must contain metric + value columns. columns={list(d.columns)}")
    if lo_col is None or hi_col is None:
        raise ValueError(f"metrics csv must contain CI columns (ci_lo/ci_hi or ci_low/ci_high). columns={list(d.columns)}")

    d = d[d[metric_col].astype(str).str.contains(kind, case=False, regex=False)].copy()
    if d.empty:
        raise ValueError(f"No rows matching kind={kind}.")

    d["name"] = d[metric_col].astype(str).apply(_metric_name)
    order = ["Macro","HS","NC","ZS"]
    d["name"] = pd.Categorical(d["name"], categories=order, ordered=True)
    d = d.sort_values("name")

    y = np.arange(len(d))
    x = d[val_col].to_numpy(dtype=float)
    lo = d[lo_col].to_numpy(dtype=float)
    hi = d[hi_col].to_numpy(dtype=float)

    lo = np.minimum(lo, x)
    hi = np.maximum(hi, x)

    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=300)
    ax.errorbar(x, y, xerr=[x-lo, hi-x], fmt="o", capsize=3, elinewidth=1.6, markersize=6)

    ax.set_yticks(y)
    ax.set_yticklabels(d["name"].astype(str))

    xmax = float(np.nanmax(hi))
    ax.set_xlim(0.0, min(1.05, max(0.9, xmax + 0.05)))

    # numeric annotation at right
    xlim = ax.get_xlim()
    for yi, xi, loi, hii in zip(y, x, lo, hi):
        ax.text(xlim[1] + 0.01, yi, f"{xi:.3f} [{loi:.3f}, {hii:.3f}]",
                va="center", ha="left", fontsize=9, clip_on=False)

    if ref_line is not None:
        ax.axvline(ref_line, linestyle="--", linewidth=1.2, alpha=0.7, color="0.5")

    ax.set_xlabel(kind)
    ax.set_title(f"Out-of-fold {kind} (95% CI)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_ylim(-0.6, len(d)-0.4)

    fig.subplots_adjust(right=0.80, left=0.18, top=0.86, bottom=0.18)

    for ext in ["png","pdf"]:
        fig.savefig(f"{out_base}.{ext}", bbox_inches="tight")
    plt.close(fig)

def calibration_summary(cal_df: pd.DataFrame, out_base: str, show_implied_curve: bool = True):
    df = cal_df.copy()
    df.columns = [c.strip() for c in df.columns]

    class_col = _pick_col(df, ["class","Class","label","Label","group","Group"])
    brier_col = _pick_col(df, ["Brier","brier","brier_score"])
    ece_col   = _pick_col(df, ["ECE","ece"])
    slope_col = _pick_col(df, ["cal_slope","slope","Slope"])
    intc_col  = _pick_col(df, ["cal_intercept","intercept","Intercept"])
    if class_col is None or slope_col is None or intc_col is None:
        raise ValueError(f"calibration csv must contain class + cal_slope + cal_intercept. columns={list(df.columns)}")

    df[class_col] = df[class_col].astype(str).str.upper()
    sub = df[df[class_col].isin(["HS","NC","ZS"])].copy()
    if sub.empty:
        sub = df.copy()

    macro_brier = float(sub[brier_col].mean()) if brier_col else np.nan
    macro_ece   = float(sub[ece_col].mean())   if ece_col   else np.nan

    fig = plt.figure(figsize=(7.2, 3.2), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.35)
    axL = fig.add_subplot(gs[0,0])
    axR = fig.add_subplot(gs[0,1])

    # Left: implied mapping curve from slope/intercept (not a binned reliability diagram)
    axL.plot([0,1],[0,1], color="0.6", linewidth=1.5, label="Perfect")
    if show_implied_curve:
        xs = np.linspace(0.001, 0.999, 400)
        for _, r in sub.iterrows():
            cls = r[class_col]
            slope = float(r[slope_col])
            intercept = float(r[intc_col])
            logit = np.log(xs/(1-xs))
            z = intercept + slope * logit
            p_obs = 1/(1+np.exp(-z))
            axL.plot(xs, p_obs, linewidth=2.5, label=cls)

    axL.set_xlim(0,1); axL.set_ylim(0,1)
    axL.set_xlabel("Predicted probability")
    axL.set_ylabel("Observed probability (implied)")
    axL.set_title("Calibration mapping (OOF)")
    axL.grid(alpha=0.25)
    axL.legend(frameon=True, fontsize=9, loc="lower right")

    # Right: barh slope & intercept
    sub2 = sub.set_index(class_col).reindex(["HS","NC","ZS"]).reset_index()
    y = np.arange(len(sub2))
    slope = sub2[slope_col].to_numpy(dtype=float)
    intercept = sub2[intc_col].to_numpy(dtype=float)

    axR.axvline(1.0, linestyle="--", linewidth=1.2, color="0.6")
    axR.axvline(0.0, linestyle="-", linewidth=1.0, color="0.85")
    axR.barh(y+0.15, slope, height=0.28, label="Slope", alpha=0.85)
    axR.barh(y-0.15, intercept, height=0.28, label="Intercept", alpha=0.55)

    axR.set_yticks(y)
    axR.set_yticklabels(sub2[class_col].astype(str))
    axR.set_xlabel("Value")
    axR.set_title("Slope / intercept")
    axR.grid(axis="x", alpha=0.25)
    axR.legend(frameon=True, fontsize=9, loc="lower right")

    txt=[]
    if np.isfinite(macro_brier): txt.append(f"Macro Brier={macro_brier:.3f}")
    if np.isfinite(macro_ece):   txt.append(f"Macro ECE={macro_ece:.3f}")
    if txt:
        axR.text(0.02, 0.98, "  |  ".join(txt), transform=axR.transAxes,
                 ha="left", va="top", fontsize=10)

    for ext in ["png","pdf"]:
        fig.savefig(f"{out_base}.{ext}", bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal_csv", default="OOF_calibration_metrics.csv")
    ap.add_argument("--metrics_csv", default="OOF_metrics_with_CI.csv")
    ap.add_argument("--out_dir", default="out_posthoc")
    ap.add_argument("--no_implied_curve", action="store_true")
    args = ap.parse_args()

    _ensure_dir(args.out_dir)

    cal_df = _read_required(args.cal_csv)
    met_df = _read_required(args.metrics_csv)

    forest_plot(met_df, "AUROC", os.path.join(args.out_dir, "Fig2A_AUROC_forest_OOF"), ref_line=0.5)
    forest_plot(met_df, "AUPRC", os.path.join(args.out_dir, "Fig2A_AUPRC_forest_OOF"), ref_line=None)
    calibration_summary(cal_df, os.path.join(args.out_dir, "Fig2B_Calibration_summary_OOF"),
                        show_implied_curve=(not args.no_implied_curve))

    print("[DONE] wrote:", os.path.abspath(args.out_dir))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-hoc remake of v6.x figures from exported CSVs (reviewer-friendly for tiny N).

Usage:
  python3 v6_5_posthoc_from_csv.py --csv_dir out_figs_v6_4 --out_dir out_figs_v6_5_posthoc

Expected CSVs (any subset works; script degrades gracefully):
  - OOF_calibration_metrics.csv (required for calibration slope/intercept plot)
  - OOF_metrics_with_CI.csv     (required for AUROC/AUPRC forest plots)
Optional (if you have them, this script can also replot curves):
  - DCA_curve.csv               columns: threshold, net_benefit_model, net_benefit_all, net_benefit_none
  - OOF_pred_proba.csv          (or similar) with per-sample probs to regenerate ROC/PR/calibration bins
"""
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def sigmoid(x): return 1/(1+np.exp(-x))
def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def plot_calibration_slope(cal_df, out_png):
    p = np.linspace(0.001, 0.999, 400)
    fig = plt.figure(figsize=(7.2, 6.2), dpi=240)
    ax = plt.gca()
    ax.plot([0,1],[0,1], color="0.5", lw=2, label="Perfectly calibrated")

    for _, r in cal_df.iterrows():
        cls = str(r["class"])
        a = float(r["cal_intercept"])
        b = float(r["cal_slope"])
        y = sigmoid(a + b*logit(p))
        lab = f"{cls} (Brier={r['Brier']:.3f}, ECE={r['ECE']:.3f}, slope={b:.2f})"
        ax.plot(p, y, lw=3, label=lab)

    macro_brier = float(cal_df["Brier"].mean())
    macro_ece = float(cal_df["ECE"].mean())
    ax.text(
        0.03, 0.97, f"Macro Brier={macro_brier:.3f} | Macro ECE={macro_ece:.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85", alpha=0.95)
    )

    ax.set_title("Calibration (one-vs-rest, out-of-fold)\nlogistic line from slope/intercept (no binning)", fontsize=16, pad=10)
    ax.set_xlabel("Predicted probability", fontsize=13)
    ax.set_ylabel("Observed probability (model-implied)", fontsize=13)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(True, alpha=0.25)
    leg = ax.legend(loc="lower right", fontsize=9, frameon=True)
    leg.get_frame().set_alpha(0.92)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_forest(met_df, kind, out_png, ref_line=None):
    d = met_df[met_df["metric"].str.contains(kind)].copy()

    def name_from_metric(m):
        if m.startswith("macro_"):
            return "Macro"
        parts = m.split("_")
        return parts[1] if len(parts) >= 2 else m

    d["name"] = d["metric"].apply(name_from_metric)
    order = ["Macro","HS","NC","ZS"]
    d["name"] = pd.Categorical(d["name"], categories=order, ordered=True)
    d = d.sort_values("name")

    y = np.arange(len(d))
    fig = plt.figure(figsize=(7.2, 3.9), dpi=240)
    ax = plt.gca()
    ax.errorbar(
        d["value"], y,
        xerr=[d["value"]-d["ci_lo"], d["ci_hi"]-d["value"]],
        fmt="o", capsize=4, lw=2
    )
    if ref_line is not None:
        ax.axvline(ref_line, color="0.8", lw=1.5)

    ax.set_yticks(y)
    ylab = [f"{n}: {v:.3f} [{lo:.3f}, {hi:.3f}]" for n,v,lo,hi in zip(d["name"].astype(str), d["value"], d["ci_lo"], d["ci_hi"])]
    ax.set_yticklabels(ylab, fontsize=10)
    ax.set_xlabel(kind, fontsize=13)
    ax.set_title(f"Out-of-fold discrimination summary ({kind}, 95% CI)", fontsize=15)
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_dca_from_curve(dca_df, out_png, x_max=0.8):
    fig = plt.figure(figsize=(7.2, 5.6), dpi=240)
    ax = plt.gca()
    ax.plot(dca_df["threshold"], dca_df["net_benefit_none"], ls=":", lw=3, label="Treat none")
    ax.plot(dca_df["threshold"], dca_df["net_benefit_all"], ls="--", lw=3, label="Treat all")
    ax.plot(dca_df["threshold"], dca_df["net_benefit_model"], lw=3.5, label="Model")

    ax.set_xlim(0, x_max)
    ax.set_xlabel("Threshold probability", fontsize=13)
    ax.set_ylabel("Net benefit", fontsize=13)
    ax.set_title("Decision curve analysis (binary, out-of-fold)", fontsize=15)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True, help="Folder containing exported CSVs from v6.x run.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dca_xmax", type=float, default=0.8, help="Trim DCA x-axis for readability.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    cal_path = os.path.join(args.csv_dir, "OOF_calibration_metrics.csv")
    met_path = os.path.join(args.csv_dir, "OOF_metrics_with_CI.csv")
    dca_path = os.path.join(args.csv_dir, "DCA_curve.csv")

    if os.path.exists(cal_path):
        cal_df = pd.read_csv(cal_path)
        plot_calibration_slope(cal_df, os.path.join(args.out_dir, "Fig2B_Calibration_slope_intercept_OOF.png"))
        print("[OK] Calibration remade from slope/intercept.")
    else:
        print("[WARN] Missing OOF_calibration_metrics.csv -> skip calibration plot.")

    if os.path.exists(met_path):
        met_df = pd.read_csv(met_path)
        plot_forest(met_df, "AUROC", os.path.join(args.out_dir, "Fig2A_AUROC_forest_OOF.png"), ref_line=0.5)
        plot_forest(met_df, "AUPRC", os.path.join(args.out_dir, "Fig2A_AUPRC_forest_OOF.png"))
        print("[OK] AUROC/AUPRC forest plots remade.")
    else:
        print("[WARN] Missing OOF_metrics_with_CI.csv -> skip forest plots.")

    if os.path.exists(dca_path):
        dca_df = pd.read_csv(dca_path)
        required = {"threshold","net_benefit_model","net_benefit_all","net_benefit_none"}
        if required.issubset(set(dca_df.columns)):
            plot_dca_from_curve(dca_df, os.path.join(args.out_dir, "Fig2C_DCA_binary_OOF.png"), x_max=args.dca_xmax)
            print("[OK] DCA remade from DCA_curve.csv.")
        else:
            print(f"[WARN] DCA_curve.csv exists but missing columns: {required - set(dca_df.columns)}")
    else:
        print("[WARN] Missing DCA_curve.csv -> keep your current DCA PNG or export curve points in v6.x.")

if __name__ == "__main__":
    main()

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

def _plot_roc_pr(y_sets, p_sets, names, title_prefix, out_prefix, out_dir):
    fig = plt.figure(figsize=(7.2,6.2))
    ax = plt.gca()
    for y, p, nm in zip(y_sets, p_sets, names):
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, linewidth=2, label=f"{nm} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0,1],[0,1], linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title_prefix} | ROC (methods demo)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for ext in ["png","pdf"]:
        fig.savefig(os.path.join(out_dir, f"{out_prefix}_ROC.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7.2,6.2))
    ax = plt.gca()
    for y, p, nm in zip(y_sets, p_sets, names):
        if len(np.unique(y)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(rec, prec, linewidth=2, label=f"{nm} (AP={ap:.2f})")
    base = float(np.mean(np.concatenate(y_sets))) if len(y_sets)>0 else 0.5
    ax.hlines(base, 0, 1, linestyles="--", linewidth=1, label="Prevalence")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{title_prefix} | PR (methods demo)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    for ext in ["png","pdf"]:
        fig.savefig(os.path.join(out_dir, f"{out_prefix}_PR.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)

def _plot_calibration(y_sets, p_sets, names, title_prefix, out_prefix, out_dir, n_bins=2):
    fig = plt.figure(figsize=(7.0,6.2))
    ax = plt.gca()
    ax.plot([0,1],[0,1], linestyle="--", linewidth=1, label="Ideal")
    for y, p, nm in zip(y_sets, p_sets, names):
        if len(np.unique(y)) < 2:
            continue
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, label=nm)
    ax.set_xlabel("Predicted risk"); ax.set_ylabel("Observed event rate")
    ax.set_title(f"{title_prefix} | Calibration (methods demo)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.text(0.99, 0.01, "Note: n=4/group (demo only)", ha="right", va="bottom",
            transform=ax.transAxes, fontsize=9)
    for ext in ["png","pdf"]:
        fig.savefig(os.path.join(out_dir, f"{out_prefix}_Calibration.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)

def _decision_curve(y, p, title_prefix, out_prefix, out_dir):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
    ts = np.linspace(0.05, 0.95, 19)
    n = len(y)
    prev = float(np.mean(y)) if n>0 else 0.0
    nb = []
    for t in ts:
        pred = (p >= t).astype(int)
        TP = int(np.sum((pred==1) & (y==1)))
        FP = int(np.sum((pred==1) & (y==0)))
        nb.append((TP/n) - (FP/n) * (t/(1-t)) if n>0 else 0.0)
    nb = np.array(nb)
    nb_all = prev - (1-prev)*(ts/(1-ts))
    nb_none = np.zeros_like(ts)

    fig = plt.figure(figsize=(7.2,6.2))
    ax = plt.gca()
    ax.plot(ts, nb, linewidth=2, label="Model")
    ax.plot(ts, nb_all, linestyle="--", linewidth=1.5, label="Treat-all")
    ax.plot(ts, nb_none, linestyle="--", linewidth=1.5, label="Treat-none")
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title(f"{title_prefix} | Decision Curve (methods demo)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(0.99, 0.01, "Note: tiny n → visualization only", ha="right", va="bottom",
            transform=ax.transAxes, fontsize=9)
    for ext in ["png","pdf"]:
        fig.savefig(os.path.join(out_dir, f"{out_prefix}_DCA.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)

def _shap_like_linear(clf, scaler, X):
    coef = clf.coef_.ravel()
    X_std = scaler.transform(X)
    return X_std * coef[None, :]

def _plot_shap_bar(shap_vals, feature_names, title_prefix, out_prefix, out_dir, with_text=True, top_k=20):
    m = np.mean(np.abs(shap_vals), axis=0)
    idx = np.argsort(m)[::-1][:min(top_k, len(feature_names))]
    feats = [feature_names[i] for i in idx]
    vals = m[idx]
    fig = plt.figure(figsize=(7.6,6.6))
    ax = plt.gca()
    ax.barh(range(len(feats))[::-1], vals, height=0.8)
    if with_text:
        ax.set_yticks(range(len(feats))[::-1]); ax.set_yticklabels(feats, fontsize=8)
    else:
        ax.set_yticks([]); ax.set_yticklabels([])
    ax.set_xlabel("Mean |contribution| (SHAP-like)")
    ax.set_title(f"{title_prefix} | Explainability (demo)")
    ax.text(0.99, 0.01, "Linear SHAP-like: coef × standardized value", ha="right", va="bottom",
            transform=ax.transAxes, fontsize=9)
    for ext in ["png","pdf"]:
        fig.savefig(os.path.join(out_dir, f"{out_prefix}_SHAPbar_{'text' if with_text else 'notext'}.{ext}"),
                    dpi=600, bbox_inches="tight")
    plt.close(fig)

def _plot_waterfall(shap_row, feature_names, title, out_prefix, out_dir, with_text=True, top_k=12):
    v = np.asarray(shap_row)
    idx = np.argsort(np.abs(v))[::-1][:min(top_k, len(v))]
    feats = [feature_names[i] for i in idx]
    contrib = v[idx]
    order = np.argsort(contrib)
    feats = [feats[i] for i in order]; contrib = contrib[order]
    fig = plt.figure(figsize=(7.6,6.2))
    ax = plt.gca()
    ax.barh(range(len(feats)), contrib, height=0.75)
    if with_text:
        ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=8)
    else:
        ax.set_yticks([]); ax.set_yticklabels([])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Contribution to log-odds (SHAP-like)"); ax.set_title(title)
    for ext in ["png","pdf"]:
        fig.savefig(os.path.join(out_dir, f"{out_prefix}_Waterfall_{'text' if with_text else 'notext'}.{ext}"),
                    dpi=600, bbox_inches="tight")
    plt.close(fig)

def demo_binary_prediction(X: pd.DataFrame, pos_prefix: str, out_dir: str, out_prefix: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    sample_ids = list(X.index)
    prefixes = sorted({re.match(r"^([A-Za-z]+)\d+$", s).group(1).upper() for s in sample_ids})
    neg_prefix = [p for p in prefixes if p != pos_prefix][0]

    split = {
        "train": [f"{pos_prefix}1", f"{pos_prefix}2", f"{neg_prefix}1", f"{neg_prefix}2"],
        "test":  [f"{pos_prefix}3", f"{neg_prefix}3"],
        "ext":   [f"{pos_prefix}4", f"{neg_prefix}4"],
    }

    Xtr = X.loc[split["train"]].values
    Xte = X.loc[split["test"]].values
    Xex = X.loc[split["ext"]].values
    ytr = np.array([1 if s.startswith(pos_prefix) else 0 for s in split["train"]], dtype=int)
    yte = np.array([1 if s.startswith(pos_prefix) else 0 for s in split["test"]], dtype=int)
    yex = np.array([1 if s.startswith(pos_prefix) else 0 for s in split["ext"]], dtype=int)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    Xex_s = scaler.transform(Xex)

    clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=500)
    clf.fit(Xtr_s, ytr)

    ptr = clf.predict_proba(Xtr_s)[:,1]
    pte = clf.predict_proba(Xte_s)[:,1]
    pex = clf.predict_proba(Xex_s)[:,1]

    title = out_prefix.replace("_"," ")
    _plot_roc_pr([ytr,yte,yex],[ptr,pte,pex],["Train(1-2)","Test(3)","External(4)"], title, out_prefix, out_dir)
    _plot_calibration([ytr,yte,yex],[ptr,pte,pex],["Train(1-2)","Test(3)","External(4)"], title, out_prefix, out_dir, n_bins=2)
    _decision_curve(np.concatenate([yte,yex]), np.concatenate([pte,pex]), title, out_prefix, out_dir)

    shap_vals = _shap_like_linear(clf, scaler, X.values)
    _plot_shap_bar(shap_vals, list(X.columns), title, out_prefix, out_dir, with_text=True)
    _plot_shap_bar(shap_vals, list(X.columns), title, out_prefix, out_dir, with_text=False)

    sid = split["ext"][0]
    idx = sample_ids.index(sid)
    _plot_waterfall(shap_vals[idx], list(X.columns), f"{title} | Waterfall ({sid}, demo)", out_prefix, out_dir, with_text=True)
    _plot_waterfall(shap_vals[idx], list(X.columns), f"{title} | Waterfall ({sid}, demo)", out_prefix, out_dir, with_text=False)

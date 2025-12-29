import re, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PATH = "/mnt/h/Data/Yuchun-yanshi/数据矩阵.xlsx"
SHEET = "数据矩阵"  # 或 "缺失值数据矩阵"
OUT_DIR = "/mnt/h/Data/Yuchun-yanshi/QC_Figures"
os.makedirs(OUT_DIR, exist_ok=True)

def is_sample_col(c):
    return bool(re.match(r"^(QC|HS|NC|ZS)\d+$", str(c), flags=re.I))

def group_of_sample(s):
    m = re.match(r"^(QC|HS|NC|ZS)", str(s), flags=re.I)
    return m.group(1).upper() if m else "UNK"

def save_fig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------- Load ----------
df = pd.read_excel(PATH, sheet_name=SHEET)
sample_cols = [c for c in df.columns if is_sample_col(c)]
qc_cols = [c for c in sample_cols if str(c).upper().startswith("QC")]

print("sample cols:", sample_cols)
print("QC cols:", qc_cols)
if len(qc_cols) < 2:
    raise RuntimeError("QC列不足（至少2个）")

X = df[sample_cols].apply(pd.to_numeric, errors="coerce")  # features x samples

vals = X.values
print("value range (nanmin/median/nanmax):",
      float(np.nanmin(vals)), float(np.nanmedian(vals)), float(np.nanmax(vals)))
print("missing rate:", float(np.isnan(vals).mean()))

# Missingness
feat_missing = X.isna().mean(axis=1)
samp_missing = X.isna().mean(axis=0)

# Impute missing by per-feature median
X_imp = X.apply(lambda r: r.fillna(r.median()), axis=1)

# samples x features
M = X_imp.T
samples = M.index.tolist()
groups = [group_of_sample(s) for s in samples]

# Autoscaling
Ms = StandardScaler(with_mean=True, with_std=True).fit_transform(M)

# ---------- PCA ----------
pca = PCA(n_components=2, random_state=0)
Z = pca.fit_transform(Ms)
pc1 = pca.explained_variance_ratio_[0] * 100.0
pc2 = pca.explained_variance_ratio_[1] * 100.0

def plot_scatter(coords, title, xlabel, ylabel, labeled, outname):
    fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=300)
    order = ["QC", "NC", "HS", "ZS", "UNK"]
    for g in order:
        idx = [i for i, v in enumerate(groups) if v == g]
        if not idx:
            continue
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   s=65, alpha=0.92,
                   edgecolors="black", linewidths=0.35,
                   label=g)
        if labeled:
            for i in idx:
                ax.text(coords[i, 0], coords[i, 1], samples[i],
                        fontsize=7, ha="left", va="bottom")
    ax.axhline(0, lw=0.8, color="0.2")
    ax.axvline(0, lw=0.8, color="0.2")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, outname)

plot_scatter(Z,
             "PCA of all samples (QC should cluster tightly)",
             f"PC1 ({pc1:.1f}%)",
             f"PC2 ({pc2:.1f}%)",
             True,
             "QC_PCA_labeled")
plot_scatter(Z,
             "PCA of all samples (QC should cluster tightly)",
             f"PC1 ({pc1:.1f}%)",
             f"PC2 ({pc2:.1f}%)",
             False,
             "QC_PCA_nolabel")

# ---------- UMAP (optional) ----------
try:
    import umap
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.15, random_state=0)
    U = reducer.fit_transform(Ms)
    plot_scatter(U, "UMAP of all samples (QC should cluster tightly)",
                 "UMAP1", "UMAP2", True, "QC_UMAP_labeled")
    plot_scatter(U, "UMAP of all samples (QC should cluster tightly)",
                 "UMAP1", "UMAP2", False, "QC_UMAP_nolabel")
except Exception as e:
    print("UMAP skipped:", repr(e))

# ---------- QC-RSD ----------
Xqc = df[qc_cols].apply(pd.to_numeric, errors="coerce")
mean = Xqc.mean(axis=1)
std = Xqc.std(axis=1, ddof=1)
rsd = (std / mean).replace([np.inf, -np.inf], np.nan) * 100.0
rsd = rsd.dropna()

p20 = float((rsd < 20).mean() * 100.0)
p30 = float((rsd < 30).mean() * 100.0)

fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=300)
ax.hist(rsd, bins=60, edgecolor="black", linewidth=0.3)
ax.axvline(20, lw=1.0, color="0.2")
ax.axvline(30, lw=1.0, color="0.2")
ax.set_xlabel("RSD% across QC replicates")
ax.set_ylabel("Number of features")
ax.set_title(f"QC reproducibility (QC n={len(qc_cols)})\nRSD<20%: {p20:.1f}% ; RSD<30%: {p30:.1f}%",
             fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save_fig(fig, "QC_RSD_hist")

# ---------- Sample correlation heatmap ----------
corr = pd.DataFrame(Ms, index=samples).T.corr()
fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=300)
im = ax.imshow(corr.values, aspect="auto")
ax.set_xticks(range(len(samples)))
ax.set_yticks(range(len(samples)))
ax.set_xticklabels(samples, rotation=90, fontsize=8)
ax.set_yticklabels(samples, fontsize=8)
ax.set_title("Sample–sample correlation (after scaling)", fontsize=11)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)
save_fig(fig, "QC_sample_correlation_heatmap")

# ---------- Missingness ----------
fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=300)
ax.bar(range(len(samp_missing)), samp_missing.values,
       color="0.8", edgecolor="black", linewidth=0.6)
ax.set_xticks(range(len(samp_missing)))
ax.set_xticklabels(samp_missing.index.tolist(),
                   rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Missing rate")
ax.set_title("Per-sample missingness", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save_fig(fig, "QC_missingness_by_sample")

fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=300)
ax.hist(feat_missing.values, bins=60,
        edgecolor="black", linewidth=0.3)
ax.set_xlabel("Missing rate per feature")
ax.set_ylabel("Number of features")
ax.set_title("Per-feature missingness", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save_fig(fig, "QC_missingness_by_feature")

# ---------- Summary ----------
summary = pd.DataFrame({
    "metric": [
        "n_features",
        "n_samples",
        "n_QC",
        "overall_missing_rate",
        "RSD<20% (features)",
        "RSD<30% (features)",
    ],
    "value": [
        int(df.shape[0]),
        int(len(sample_cols)),
        int(len(qc_cols)),
        float(np.isnan(X.values).mean()),
        f"{p20:.2f}%",
        f"{p30:.2f}%",
    ],
})
summary.to_csv(os.path.join(OUT_DIR, "QC_summary_metrics.csv"), index=False)

print("Saved figures to:", OUT_DIR)

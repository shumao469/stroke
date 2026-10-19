import re, os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PATH = "/mnt/h/Data/Yuchun-yanshi/数据矩阵.xlsx"
SHEET = "数据矩阵"
OUT_DIR = "/mnt/h/Data/Yuchun-yanshi/QC_Figures"
os.makedirs(OUT_DIR, exist_ok=True)

def is_sample_col(c):
    return bool(re.match(r"^(QC|HS|NC|ZS)\d+$", str(c), flags=re.I))

def group_of_sample(s):
    m = re.match(r"^(QC|HS|NC|ZS)", str(s), flags=re.I)
    return m.group(1).upper() if m else "UNK"

# ===== load matrix =====
df = pd.read_excel(PATH, sheet_name=SHEET)
sample_cols = [c for c in df.columns if is_sample_col(c)]
qc_cols = [c for c in sample_cols if str(c).upper().startswith("QC")]
X = df[sample_cols].apply(pd.to_numeric, errors="coerce")  # features x samples

# ===== optional: compute QC-RSD and filter features (more standard for QC diagnosis) =====
Xqc = df[qc_cols].apply(pd.to_numeric, errors="coerce")
mean = Xqc.mean(axis=1)
std  = Xqc.std(axis=1, ddof=1)
rsd = (std/mean).replace([np.inf, -np.inf], np.nan) * 100.0
# Keep features with QC-RSD < 30% (you have 100%, so this keeps all; still OK)
mask = rsd.notna() & (rsd < 30)
Xf = X.loc[mask].copy()

# ===== prepare sample x feature matrix =====
M = Xf.T  # samples x features
samples = M.index.tolist()
groups = [group_of_sample(s) for s in samples]

# For outlier diagnostics, do autoscaling (common) but we'll ALSO store non-scaled similarity later
Ms = StandardScaler(with_mean=True, with_std=True).fit_transform(M)

# ===== PCA model =====
# Use more PCs for T2/Q (not just 2), e.g., keep 5 or <= n_samples-1
n_samples = Ms.shape[0]
n_comp = min(5, n_samples - 1)
pca = PCA(n_components=n_comp, random_state=0)
T = pca.fit_transform(Ms)          # scores (n_samples x n_comp)
P = pca.components_                # loadings (n_comp x n_features)
lam = pca.explained_variance_      # eigenvalues length n_comp

# Reconstruction and Q residual (SPE)
Ms_hat = T @ P
E = Ms - Ms_hat
Q = np.sum(E**2, axis=1)

# Hotelling's T2
T2 = np.sum((T**2) / lam, axis=1)

# ===== thresholds (practical) =====
# Use empirical 95th/99th percentiles (robust with small n)
T2_95, T2_99 = np.quantile(T2, [0.95, 0.99])
Q_95,  Q_99  = np.quantile(Q,  [0.95, 0.99])

# ===== PCA 2D for plotting + "leftmost NC" =====
pca2 = PCA(n_components=2, random_state=0)
Z2 = pca2.fit_transform(Ms)

pc1 = pca2.explained_variance_ratio_[0]*100
pc2 = pca2.explained_variance_ratio_[1]*100

# Identify leftmost NC samples on PC1
nc_idx = [i for i,s in enumerate(samples) if s.upper().startswith("NC")]
left_nc = sorted([(samples[i], Z2[i,0], Z2[i,1]) for i in nc_idx], key=lambda x: x[1])[:2]

# ===== distance-to-centroid metrics (in PCA2 space) =====
def centroid(idxs):
    return np.mean(Z2[idxs, :], axis=0)

# group centroids
centroids = {}
for g in sorted(set(groups)):
    idxs = [i for i,v in enumerate(groups) if v==g]
    centroids[g] = centroid(idxs)

# distances
dist_to_group = []
dist_to_qc = []
qc_center = centroids.get("QC", np.mean(Z2, axis=0))

for i,s in enumerate(samples):
    g = groups[i]
    d_g  = float(np.linalg.norm(Z2[i,:] - centroids[g]))
    d_qc = float(np.linalg.norm(Z2[i,:] - qc_center))
    dist_to_group.append(d_g)
    dist_to_qc.append(d_qc)

# ===== report table =====
rep = pd.DataFrame({
    "sample": samples,
    "group": groups,
    "PC1": Z2[:,0],
    "PC2": Z2[:,1],
    "dist_to_group_centroid(PCA2)": dist_to_group,
    "dist_to_QC_centroid(PCA2)": dist_to_qc,
    "T2": T2,
    "Q": Q
})
rep["flag_T2_99"] = rep["T2"] > T2_99
rep["flag_Q_99"]  = rep["Q"]  > Q_99
rep["flag_any_99"] = rep["flag_T2_99"] | rep["flag_Q_99"]
rep.to_csv(os.path.join(OUT_DIR, "outlier_report.csv"), index=False)

print("Leftmost 2 NC on PC1 (likely 'split' NC):")
for s,x,y in left_nc:
    print("  ", s, "PC1=", round(x,3), "PC2=", round(y,3))

print("Saved:", os.path.join(OUT_DIR, "outlier_report.csv"))

# ===== plot: PCA with labels + 95% ellipses (simple) =====
def plot_ellipse(ax, pts, n_std=2.0):
    # 2D covariance ellipse; n_std~2 roughly 95% for normal
    if pts.shape[0] < 3:
        return
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:,order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2*n_std*np.sqrt(vals)
    from matplotlib.patches import Ellipse
    ell = Ellipse(xy=pts.mean(axis=0), width=width, height=height, angle=theta,
                  fill=False, lw=1.0, ec="0.35")
    ax.add_patch(ell)

fig, ax = plt.subplots(figsize=(6.0,5.4), dpi=300)
color = {"QC":"#1f77b4","NC":"#ff7f0e","HS":"#2ca02c","ZS":"#d62728","UNK":"0.5"}

for g in ["QC","NC","HS","ZS","UNK"]:
    idx = [i for i,v in enumerate(groups) if v==g]
    if not idx:
        continue
    pts = Z2[idx,:]
    ax.scatter(pts[:,0], pts[:,1], s=70, alpha=0.92,
               edgecolors="black", linewidths=0.35,
               c=color.get(g,"0.5"), label=g)
    plot_ellipse(ax, pts, n_std=2.0)
    for i in idx:
        ax.text(Z2[i,0], Z2[i,1], samples[i], fontsize=7, ha="left", va="bottom")

ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
ax.set_title("PCA score plot of all samples and QC replicates", fontsize=11)
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(os.path.join(OUT_DIR, "QC_PCA_ellipses_labeled.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "QC_PCA_ellipses_labeled.pdf"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ===== plot: T2 vs Q (outlier map) =====
fig, ax = plt.subplots(figsize=(6.2,5.2), dpi=300)
for g in ["QC","NC","HS","ZS","UNK"]:
    sub = rep[rep["group"]==g]
    if sub.empty:
        continue
    ax.scatter(sub["T2"], sub["Q"], s=65, alpha=0.9,
               edgecolors="black", linewidths=0.35,
               c=color.get(g,"0.5"), label=g)

# threshold lines
ax.axvline(T2_99, lw=1.0, color="0.2")
ax.axhline(Q_99,  lw=1.0, color="0.2")

# label flagged
flag = rep[rep["flag_any_99"]]
for _, r in flag.iterrows():
    ax.text(r["T2"], r["Q"], r["sample"], fontsize=7, ha="left", va="bottom")

ax.set_xlabel("Hotelling's T² (PCA space)")
ax.set_ylabel("Q residual / SPE (reconstruction error)")
ax.set_title("Outlier map (empirical 99% thresholds)", fontsize=11)
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(os.path.join(OUT_DIR, "QC_outlier_map_T2_Q.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "QC_outlier_map_T2_Q.pdf"), dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved PCA+ellipse and T2-Q outlier map to:", OUT_DIR)

# ===== optional: similarity heatmap WITHOUT scaling (more interpretable) =====
# Spearman correlation on raw log-intensity scale (your matrix already looks log-like)
M0 = Xf.T  # samples x features, no scaling
corr0 = M0.corr(method="spearman")
fig, ax = plt.subplots(figsize=(6.4,5.6), dpi=300)
im = ax.imshow(corr0.values, aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(samples))); ax.set_yticks(range(len(samples)))
ax.set_xticklabels(samples, rotation=90, fontsize=8)
ax.set_yticklabels(samples, fontsize=8)
ax.set_title("Sample–sample Spearman correlation (no scaling)", fontsize=11)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)
fig.savefig(os.path.join(OUT_DIR, "QC_sample_spearman_noscaling.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "QC_sample_spearman_noscaling.pdf"), dpi=300, bbox_inches="tight")
plt.close(fig)

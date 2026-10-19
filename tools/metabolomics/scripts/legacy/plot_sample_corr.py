import re, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

PATH = "/mnt/h/Data/Yuchun-yanshi/数据矩阵.xlsx"
SHEET = "数据矩阵"
OUT_DIR = "/mnt/h/Data/Yuchun-yanshi/QC_Figures"
os.makedirs(OUT_DIR, exist_ok=True)

def is_sample_col(c):
    return bool(re.match(r"^(QC|HS|NC|ZS)\d+$", str(c), flags=re.I))

def group_of_sample(s):
    m = re.match(r"^(QC|HS|NC|ZS)", str(s), flags=re.I)
    return m.group(1).upper() if m else "UNK"

# =========================
# Load (features x samples)
# =========================
df = pd.read_excel(PATH, sheet_name=SHEET)
sample_cols = [c for c in df.columns if is_sample_col(c)]
if len(sample_cols) == 0:
    raise RuntimeError("No sample columns found. Expected like QC1/NC2/HS3/ZS4...")

X = df[sample_cols].apply(pd.to_numeric, errors="coerce")

# =========================
# Make samples x features
# =========================
M = X.T
M.index = sample_cols
samples = M.index.tolist()
groups  = [group_of_sample(s) for s in samples]

# =========================
# Sample-sample Spearman corr
# =========================
corr = M.T.corr(method="spearman")  # correlate samples

# =========================
# Hierarchical clustering reorder
# =========================
dist = 1 - corr.values
np.fill_diagonal(dist, 0.0)
Z = linkage(squareform(dist, checks=False), method="average")
order = leaves_list(Z)

corr_ord    = corr.iloc[order, order]
samples_ord = [samples[i] for i in order]
groups_ord  = [groups[i] for i in order]

# =========================
# Group color bar (left)
# =========================
group_color = {
    "QC": "#1f77b4",
    "NC": "#ff7f0e",
    "HS": "#2ca02c",
    "ZS": "#d62728",
    "UNK": "0.5"
}
row_colors = [group_color.get(g, "0.5") for g in groups_ord]

# =========================
# Plot
# =========================
# 1) 淡化/隐藏对角线：把对角线设为 NaN
mat = corr_ord.values.copy()
np.fill_diagonal(mat, np.nan)

fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=300)

# 2) NaN 显示为白色（对角线更干净）
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad(color="white")

im = ax.imshow(mat, aspect="equal", vmin=0.6, vmax=1.0, cmap=cmap)

# ticks / labels
ax.set_xticks(range(len(samples_ord)))
ax.set_yticks(range(len(samples_ord)))
ax.set_xticklabels(samples_ord, rotation=90, fontsize=8)
ax.set_yticklabels(samples_ord, fontsize=8)

ax.set_title("Sample–sample Spearman correlation (no scaling)", fontsize=11)

# colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)
cbar.set_label("Spearman ρ", fontsize=9)

# 3) 左侧组别色条（在 ax 已创建之后添加！）
#    imshow 的行中心在 i，对应 cell 边界 i-0.5 到 i+0.5
bar_x = -0.65   # 色条左边起点（稍微放到图外）
bar_w = 0.12    # 色条宽度
for i, c in enumerate(row_colors):
    ax.add_patch(
        patches.Rectangle((bar_x, i - 0.5), bar_w, 1.0,
                          color=c, clip_on=False, linewidth=0)
    )

# 4) 给左侧留一点空间，不然色条可能被裁掉
ax.set_xlim(bar_x - 0.05, len(samples_ord) - 0.5)

# 可选：加个 legend（简单解释颜色）
handles = [
    patches.Patch(color=group_color["QC"], label="QC"),
    patches.Patch(color=group_color["NC"], label="NC"),
    patches.Patch(color=group_color["HS"], label="HS"),
    patches.Patch(color=group_color["ZS"], label="ZS"),
]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
          frameon=False, fontsize=8, borderaxespad=0.0)

# clean spines
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# save
png_path = os.path.join(OUT_DIR, "QC_sample_spearman_noscaling_clustered.png")
pdf_path = os.path.join(OUT_DIR, "QC_sample_spearman_noscaling_clustered.pdf")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", png_path)
print("Saved:", pdf_path)

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# --------------------
# Config
# --------------------
BASE_DIR = "/mnt/h/Data/Yuchun-yanshi"
OUT_DIR  = os.path.join(BASE_DIR, "Science_Predictive_Figures")
os.makedirs(OUT_DIR, exist_ok=True)

AUG_CSV_CANDIDATES = [
    os.path.join(BASE_DIR, "QC_Figures", "augmented_matrix_100perclass.csv"),
    os.path.join(BASE_DIR, "augmented_matrix_100perclass.csv"),
    os.path.join(BASE_DIR, "Science_Predictive_Figures", "augmented_matrix_100perclass.csv"),
]

DATA_XLSX = os.path.join(BASE_DIR, "数据矩阵.xlsx")
SHEET     = "数据矩阵"
N_AUG     = 100
NOISE     = 0.03

CLASSES = np.array(["NC","HS","ZS"])
class_to_idx = {c:i for i,c in enumerate(CLASSES)}

def is_sample_col(c):
    return bool(re.match(r"^(QC|HS|NC|ZS)\d+$", str(c), flags=re.I))

def group_of_sample(s):
    m = re.match(r"^(QC|HS|NC|ZS)", str(s), flags=re.I)
    return m.group(1).upper() if m else "UNK"

def pick_feature_id(df, sample_cols):
    meta_cols = [c for c in df.columns if c not in sample_cols]
    preferred = ["Metabolite", "metabolite", "Name", "NAME", "Compound", "compound",
                 "KEGG", "kegg", "HMDB", "hmdb", "Feature", "ID", "id"]
    for c in preferred:
        if c in df.columns and c in meta_cols:
            s = df[c].astype(str).fillna("")
            if (s.str.len() > 0).sum() > 0:
                return s.where(s.str.len()>0, None).fillna("").values
    mz_col = None
    rt_col = None
    for c in df.columns:
        if str(c).lower() in ["m/z","mz"]:
            mz_col = c
        if str(c).lower() in ["retention time (min)","rt"]:
            rt_col = c
    if mz_col is not None and rt_col is not None:
        return (df[mz_col].astype(str) + "_RT" + df[rt_col].astype(str)).values
    return np.array([f"F{i+1}" for i in range(len(df))], dtype=object)

def make_augmented(A_feat_by_real, n_aug=100, noise_scale=0.03, seed=0):
    rng = np.random.default_rng(seed)
    A = np.asarray(A_feat_by_real, dtype=float)
    n_feat, n_real = A.shape
    idx = rng.integers(0, n_real, size=n_aug)
    base = A[:, idx]
    sd = np.nanstd(A, axis=1, ddof=1)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1e-6)
    noise = rng.normal(0, sd[:, None] * noise_scale, size=(n_feat, n_aug))
    return base + noise

def load_or_build_augmented():
    for p in AUG_CSV_CANDIDATES:
        if os.path.exists(p):
            df = pd.read_csv(p)
            assert "group" in df.columns, "augmented csv must contain 'group' column"
            y = df["group"].astype(str).values
            X = df.drop(columns=["group"]).values
            feat_names = df.drop(columns=["group"]).columns.astype(str).values
            return X, y, feat_names, p

    # build from 数据矩阵.xlsx
    if not os.path.exists(DATA_XLSX):
        raise FileNotFoundError(
            "❌ 找不到增强数据CSV，也找不到 数据矩阵.xlsx。\n"
            "请确认以下之一存在：\n"
            "1) augmented_matrix_100perclass.csv（带 group 列）\n"
            "2) 数据矩阵.xlsx（sheet=数据矩阵）"
        )

    df = pd.read_excel(DATA_XLSX, sheet_name=SHEET)
    sample_cols = [c for c in df.columns if is_sample_col(c)]
    feat_names = pick_feature_id(df, sample_cols)
    Xraw = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    if Xraw.isna().any().any():
        Xraw = Xraw.apply(lambda r: r.fillna(r.median()), axis=1)

    groups = {g: [c for c in sample_cols if group_of_sample(c)==g] for g in ["NC","HS","ZS","QC"]}
    NC_cols, HS_cols, ZS_cols = groups["NC"], groups["HS"], groups["ZS"]

    X_nc = make_augmented(Xraw[NC_cols].values, n_aug=N_AUG, noise_scale=NOISE, seed=1)
    X_hs = make_augmented(Xraw[HS_cols].values, n_aug=N_AUG, noise_scale=NOISE, seed=2)
    X_zs = make_augmented(Xraw[ZS_cols].values, n_aug=N_AUG, noise_scale=NOISE, seed=3)

    X = np.concatenate([X_nc.T, X_hs.T, X_zs.T], axis=0)
    y = np.array(["NC"]*N_AUG + ["HS"]*N_AUG + ["ZS"]*N_AUG)

    out_csv = os.path.join(OUT_DIR, "augmented_matrix_100perclass.csv")
    pd.DataFrame(X, columns=feat_names).assign(group=y).to_csv(out_csv, index=False)
    return X, y, np.array(feat_names, dtype=object), out_csv

def align_proba(best_classes, proba):
    out = np.zeros((proba.shape[0], 3), dtype=float)
    for j, c in enumerate(best_classes):
        out[:, class_to_idx[c]] = proba[:, j]
    return out

def decision_curve(y_true_binary, p_pred, thresholds):
    N = len(y_true_binary)
    out = []
    for t in thresholds:
        y_hat = (p_pred >= t).astype(int)
        TP = np.sum((y_hat==1) & (y_true_binary==1))
        FP = np.sum((y_hat==1) & (y_true_binary==0))
        nb = (TP/N) - (FP/N) * (t/(1-t))
        out.append(nb)
    return np.array(out)

# --------------------
# Load data
# --------------------
X, y, feat_names, src = load_or_build_augmented()
print("Loaded:", src)
print("X:", X.shape, "classes:", {c:int(np.sum(y==c)) for c in CLASSES})

# --------------------
# Pipeline & hyperparams
# --------------------
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("kbest", SelectKBest(score_func=f_classif, k=80)),
    ("clf", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=2000))
])
param_grid = {"kbest__k":[30,50,80,120,200], "clf__C":[0.1,1.0,10.0]}

# --------------------
# 1) ROC + Calibration + DCA (OOF probabilities)
# --------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
proba_oof = np.zeros((len(y), 3), dtype=float)

for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="roc_auc_ovr", n_jobs=-1)
    gs.fit(X[tr], y[tr])
    best = gs.best_estimator_
    proba = best.predict_proba(X[te])
    proba_oof[te] = align_proba(best.named_steps["clf"].classes_, proba)
    print(f"Fold {fold} best:", gs.best_params_)

# binarize y for OVR metrics
y_bin = np.zeros((len(y), 3), dtype=int)
for i, lab in enumerate(y):
    y_bin[i, class_to_idx[lab]] = 1

# ROC
roc_data = {}
for c in CLASSES:
    i = class_to_idx[c]
    fpr, tpr, _ = roc_curve(y_bin[:, i], proba_oof[:, i])
    roc_data[c] = (fpr, tpr, auc(fpr, tpr))
macro_auc = roc_auc_score(y_bin, proba_oof, average="macro", multi_class="ovr")

fig = plt.figure(figsize=(6.2, 5.2), dpi=300)
ax = plt.gca()
for c in CLASSES:
    fpr, tpr, a = roc_data[c]
    ax.plot(fpr, tpr, lw=1.6, label=f"{c} (AUC={a:.3f})")
ax.plot([0,1],[0,1], lw=1.0, color="0.3")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title(f"Multiclass ROC (one-vs-rest), macro-AUC={macro_auc:.3f}")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.savefig(os.path.join(OUT_DIR,"Fig2A_ROC_multiclass_OVR.png"), bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR,"Fig2A_ROC_multiclass_OVR.pdf"), bbox_inches="tight")
plt.close(fig)

# Calibration (OVR)
fig = plt.figure(figsize=(6.2, 5.2), dpi=300)
ax = plt.gca()
for c in CLASSES:
    i = class_to_idx[c]
    frac_pos, mean_pred = calibration_curve(y_bin[:, i], proba_oof[:, i], n_bins=8, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", lw=1.2, label=c)
ax.plot([0,1],[0,1], lw=1.0, color="0.3")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction")
ax.set_title("Calibration (one-vs-rest, quantile bins)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.savefig(os.path.join(OUT_DIR,"Fig2B_Calibration_multiclass_OVR.png"), bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR,"Fig2B_Calibration_multiclass_OVR.pdf"), bbox_inches="tight")
plt.close(fig)

# DCA (OVR)
thresholds = np.linspace(0.05, 0.95, 19)
fig = plt.figure(figsize=(6.2, 5.2), dpi=300)
ax = plt.gca()
for c in CLASSES:
    i = class_to_idx[c]
    nb = decision_curve(y_bin[:, i], proba_oof[:, i], thresholds)
    ax.plot(thresholds, nb, lw=1.3, label=c)

# baselines: treat-all per class
for c in CLASSES:
    i = class_to_idx[c]
    prev = np.mean(y_bin[:, i])
    treat_all = prev - (1-prev) * (thresholds/(1-thresholds))
    ax.plot(thresholds, treat_all, lw=0.8, color="0.6", alpha=0.7)

ax.axhline(0, lw=1.0, color="0.2")
ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
ax.set_title("Decision curve analysis (one-vs-rest)")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.savefig(os.path.join(OUT_DIR,"Fig2C_DCA_multiclass_OVR.png"), bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR,"Fig2C_DCA_multiclass_OVR.pdf"), bbox_inches="tight")
plt.close(fig)

# --------------------
# 2) Nested CV stability: AUC distribution + feature frequency
# --------------------
outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=7)  # 50 outer folds
auc_macro = []
selected_counts = np.zeros(len(feat_names), dtype=int)
n_outer = 0

for tr, te in outer.split(X, y):
    n_outer += 1
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="roc_auc_ovr", n_jobs=-1)
    gs.fit(X[tr], y[tr])
    best = gs.best_estimator_
    proba = best.predict_proba(X[te])
    proba = align_proba(best.named_steps["clf"].classes_, proba)

    yte = y[te]
    yte_bin = np.zeros((len(yte), 3), dtype=int)
    for i, lab in enumerate(yte):
        yte_bin[i, class_to_idx[lab]] = 1

    aucm = roc_auc_score(yte_bin, proba, average="macro", multi_class="ovr")
    auc_macro.append(aucm)

    mask = best.named_steps["kbest"].get_support()
    selected_counts[mask] += 1

auc_macro = np.array(auc_macro)

# AUC distribution
fig = plt.figure(figsize=(4.8, 4.2), dpi=300)
ax = plt.gca()
ax.boxplot(auc_macro, widths=0.5, showfliers=False)
rng = np.random.default_rng(0)
ax.scatter(1 + rng.normal(0, 0.03, size=len(auc_macro)), auc_macro, s=18, alpha=0.6)
ax.set_xticks([1]); ax.set_xticklabels(["macro-AUC"])
ax.set_ylabel("AUC (outer folds)")
ax.set_title(f"Nested CV stability (n={len(auc_macro)} outer folds)")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.savefig(os.path.join(OUT_DIR,"ExtData_AUC_distribution.png"), bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR,"ExtData_AUC_distribution.pdf"), bbox_inches="tight")
plt.close(fig)

# Feature selection frequency Top20
topK = 20
idx_top = np.argsort(selected_counts)[::-1][:topK]
freq = selected_counts[idx_top] / n_outer

fig = plt.figure(figsize=(6.4, 4.2), dpi=300)
ax = plt.gca()
ax.barh(range(topK)[::-1], freq[::-1])
ax.set_yticks(range(topK)[::-1])
ax.set_yticklabels([str(feat_names[i])[:60] for i in idx_top][::-1], fontsize=7)
ax.set_xlabel("Selection frequency (outer folds)")
ax.set_title("Top biomarkers by selection stability (nested CV)")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.savefig(os.path.join(OUT_DIR,"ExtData_FeatureFreq_Top20.png"), bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR,"ExtData_FeatureFreq_Top20.pdf"), bbox_inches="tight")
plt.close(fig)

pd.DataFrame({
    "feature":[feat_names[i] for i in idx_top],
    "selected_count":selected_counts[idx_top],
    "n_outer":n_outer,
    "selection_frequency":freq
}).to_csv(os.path.join(OUT_DIR,"ExtData_FeatureFreq_Top20.csv"), index=False)

# --------------------
# 3) Module-like signature heatmap (samples × modules)
# (top-variance 300 features -> correlation clustering -> module eigenscore)
# --------------------
topF = min(300, X.shape[1])
feat_var = np.var(X, axis=0)
feat_idx = np.argsort(feat_var)[::-1][:topF]
X_top = X[:, feat_idx]
X_top_z = (X_top - X_top.mean(axis=0)) / (X_top.std(axis=0) + 1e-9)
corr = np.corrcoef(X_top_z, rowvar=False)
dist = 1 - corr
np.fill_diagonal(dist, 0.0)

try:
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
    from scipy.spatial.distance import squareform
    Z = linkage(squareform(dist, checks=False), method="average")
    n_modules = 8
    labels = fcluster(Z, t=n_modules, criterion="maxclust")
except Exception:
    n_modules = 8
    rng = np.random.default_rng(1)
    labels = rng.integers(1, n_modules+1, size=topF)

module_scores = []
module_names = []
module_members = []
for m in range(1, n_modules+1):
    sel = np.where(labels==m)[0]
    if len(sel) < 5:
        continue
    module_names.append(f"Module {m} (n={len(sel)})")
    module_scores.append(X_top_z[:, sel].mean(axis=1))
    for mem in feat_names[feat_idx][sel]:
        module_members.append({"module": module_names[-1], "feature": mem})

M = np.vstack(module_scores).T
# order samples by group
group_order = {"NC":0,"HS":1,"ZS":2}
order = np.argsort([group_order[g] for g in y])
M = M[order]; y_ord = y[order]
M_vis = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-9)

fig = plt.figure(figsize=(7.2, 4.8), dpi=300)
ax = plt.gca()
im = ax.imshow(M_vis, aspect="auto")
ax.set_yticks([])
ax.set_xticks(range(M_vis.shape[1]))
ax.set_xticklabels(module_names, rotation=45, ha="right", fontsize=7)
ax.set_title("Module-like signature scores (samples × modules)")

# group separators
for i in range(1, len(y_ord)):
    if y_ord[i] != y_ord[i-1]:
        ax.axhline(i-0.5, lw=1.0, color="0.2")

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Module score (z)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.savefig(os.path.join(OUT_DIR,"ExtData_ModuleSignature_heatmap.png"), bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR,"ExtData_ModuleSignature_heatmap.pdf"), bbox_inches="tight")
plt.close(fig)

pd.DataFrame(module_members).to_csv(os.path.join(OUT_DIR,"ExtData_Module_membership.csv"), index=False)

print("\n✅ All outputs saved to:", OUT_DIR)

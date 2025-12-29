# Metabolomics (Stroke Prediction from Ocular Discharge Samples)

**Main script:** `scripts/predict_science_figs_v6_4.py`  
**Stable wrapper (recommended):** `scripts/predict_science_figs.py` → forwards to `v6_4`

This repository provides a reproducible LC–MS metabolomics workflow for:

- **QC / data integrity**: PCA (QC clustering), QC-RSD, sample correlation heatmaps  
- **Differential visualization**: class-aware volcano + raincloud distributions  
- **Pathway enrichment**: KEGG dotplot + “compound evidence” supplementary tables  
- **Overlap & direction consistency**: UpSet + quadrant log2FC concordance  
- **Prediction (internal validation)**: strict **OOF/CV** ROC/PR + calibration + DCA  
- **Robustness / audit**: stability, outlier influence, sensitivity analyses  

Groups:
- **NC**: normal control  
- **HS**: hemorrhagic stroke  
- **ZS**: ischemic stroke  
- **QC**: pooled QC replicates  

---

## 1) Inputs

### A) Intensity matrix (main input)
**Recommended filename (English):** `intensity_matrix.xlsx` (or `data_matrix.xlsx`)  
**Current filename in your project:** `数据矩阵.xlsx`  
> Raw data files are **not tracked by git**.

Recommended sheet:
- `缺失值数据矩阵` (recommended)

Alternative:
- `数据矩阵`

Expected columns:
- Annotation columns (e.g., `ID`, `m/z`, `Retention time (min)`, `Metabolites`, `Class`, `KEGG`, …)
- Sample columns named like: `QC1 QC2 QC3 HS1.. NC1.. ZS1..`

Sample columns are auto-detected by regex:
`^(NC|HS|ZS|QC)\d+$`

### B) Summary bundle (optional for some plots)
- `HS_related_all_tables.xlsx` (not tracked by git)

---

## 2) Installation (WSL / Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
Optional dependencies:

bash

pip install shap umap-learn imbalanced-learn
3) Reproduce figures (publication workflow)
Note: In commands below, keep the sheet names exactly as they appear in the Excel file
(e.g., 缺失值数据矩阵). Translating sheet names will break loading unless you rename them inside Excel.

3.1 QC figures (Fig.1 / Extended Data “entry ticket”)
bash

python3 scripts/qc_figs_clean.py \
  --xlsx "/mnt/h/Data/Yuchun-yanshi/数据矩阵.xlsx" \
  --sheet "缺失值数据矩阵" \
  --out_dir "/mnt/h/Data/Yuchun-yanshi/QC_Figures"
3.2 Main analysis (recommended): strict OOF / internal validation (no sample removal)
bash

python3 scripts/predict_science_figs.py \
  --xlsx "/mnt/h/Data/Yuchun-yanshi/数据矩阵.xlsx" \
  --sheet "缺失值数据矩阵" \
  --out_dir "/mnt/h/Data/Yuchun-yanshi/out_main_OOF" \
  --oversample none \
  --do_nested_tuning \
  --do_feature_select
3.3 Sensitivity analysis (audit): drop suspected outliers + training-fold-only augmentation
For tiny n, SMOTE/ADASYN may fail if any class has ≤1 sample inside a training fold.
Prefer noise/jitter augmentation (feature-wise) and/or class weights.

bash

python3 scripts/predict_science_figs.py \
  --xlsx "/mnt/h/Data/Yuchun-yanshi/数据矩阵.xlsx" \
  --sheet "缺失值数据矩阵" \
  --out_dir "/mnt/h/Data/Yuchun-yanshi/out_sensitivity" \
  --drop NC1 NC3 \
  --oversample noise \
  --augment_n 100 \
  --noise_scale 0.05 \
  --do_nested_tuning \
  --do_feature_select
4) What “core figures” this repo produces
Prediction core (main text / Extended Data):

ROC (one-vs-rest) and/or AUROC forest plot

PR curves / AUPRC forest plot (recommended for class-imbalance)

Calibration summary (slope/intercept + Brier/ECE) + reliability curves

DCA (recommended as a clearly-defined binary clinical decision, e.g., HS/ZS vs NC)

Robustness (Supplement):

AUC distribution across repeated CV

Feature selection frequency (Top 20)

Influence / leave-one-out audit for suspected outliers (e.g., NC1/NC3)

5) GitHub publishing checklist (recommended)
Do not commit raw patient data (*.xlsx/*.csv)

Keep outputs (*.png/*.pdf) out of the repo unless you intentionally publish figures

Add a data/demo/ synthetic dataset if you want others to run the pipeline end-to-end

Tag a release:

bash
git tag -a v0.1.0 -m "Initial public release"
git push origin v0.1.0
6) Repo layout
text

metabolomics/
  scripts/
    predict_science_figs.py           # stable wrapper → v6_4
    predict_science_figs_v6_4.py      # MAIN SCRIPT
    legacy/                           # (optional) older scripts collected from WSL
  src/metabolomics/                   # minimal installable package scaffold
  configs/ docs/ tests/
License
See LICENSE.

Citation
If you use this code, please cite your paper / preprint (replace placeholder):

bibtex

@article{your2026metabolomics,
  title   = {...},
  author  = {...},
  journal = {...},
  year    = {2026}
}
Contact
For questions or contributions, please open a GitHub issue or contact the maintainers.

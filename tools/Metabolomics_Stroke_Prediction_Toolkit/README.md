# Metabolomics (Ocular Discharge) — Stroke Prediction Toolkit

A reproducible pipeline to analyze **ocular discharge metabolomics** for **stroke phenotyping/prediction** with
Science-style figures (taxonomy shift, networks, and prediction-core panels).

> ⚠️ With **n=4 per group**, any model metrics are **methods demonstration only**.

## Functional modules (what this repo does)

- **Chemical taxonomy shift**
  - Sunburst/Treemap-like nested donut: `Super Class → Class → Sub Class`
  - Alluvial/Sankey-like: `SuperClass → SubClass → Regulation (Up/Down)`
  - Both **with text** and **no-text** versions

- **Metabolite association network**
  - Spearman correlation across samples for Top-N metabolites (VIP / |log2FC| / p)
  - Network: node color=Class, size=VIP; edge=|rho|≥threshold
  - Chord-like: Class-to-Class edge density (**structure visualization only**)

- **Prediction core figure (DEMO)**
  - ROC + PR + Calibration + DCA + SHAP-like (linear) + individual waterfall
  - Pseudo split: Train(1–2) / Test(3) / External(4)

## Install

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Quick start (CLI)

### Taxonomy shift
```bash
ms-taxonomy --excel "/path/to/HS_related_all_tables.xlsx" --out outputs/taxonomy
```

### Network + chord-like density
```bash
ms-network --excel "/path/to/HS_related_all_tables.xlsx" --comparison HSvsNC --out outputs/network
ms-network --excel "/path/to/HS_related_all_tables.xlsx" --comparison ZSvsHS --out outputs/network
```

### Prediction core (demo)
```bash
ms-predict-demo --excel "/path/to/HS_related_all_tables.xlsx" --out outputs/predict_demo --top_k 30
```

## Data conventions

Groups:
- QC = quality control
- NC = normal control
- HS = hemorrhagic stroke
- ZS = ischemic stroke

Sample IDs expected: `NC1..`, `HS1..`, `ZS1..`, `QC1..`

## Notes / limitations (must state in manuscript)

- Correlation networks with tiny n are **visual structure only** (no mechanistic claims).
- True external validation should be an **independent center** or **independent time window** cohort.

## License
MIT

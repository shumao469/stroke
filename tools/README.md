# Ocular Metabolomics Clinical Pipeline for Stroke Management (Demo CDSS)

A **modular, publication-style** Python pipeline that demonstrates how *non-invasive ocular secretion metabolites* can support hemorrhagic stroke (HS) management in three clinical scenarios:

1. **ED Triage** (HS vs non-HS): ROC curve + Decision Curve Analysis (DCA)  
2. **Early Risk Assessment** (hematoma expansion): calibration curve + risk stratification  
3. **Prognostic Prediction** (90-day outcome / mRS): patient-level explainability via a SHAP-style waterfall plot (no SHAP dependency)

> **Important**: This repository is a **demonstration / educational** implementation using **synthetic data**.  
> It is **NOT** a medical device and **NOT** intended for clinical diagnosis or treatment decisions.

---

## Features

- Interactive CLI optimized for **screen recording / demo videos**
- **Publication-ready** Matplotlib/Seaborn figures with academic English labels
- Simple, readable modules:
  - `DataEngine` (synthetic data generator)
  - `Visualizer` (plots)
  - 3 runnable modes: `Triage`, `Expansion Risk`, `Outcome`

---

## Biomarker Panels (Example)

**HS vs NC (Hemorrhagic signature)**  
- RvD5, DTA, DDA, 8-MKNA, N-AcCad

**ZS vs HS (Differential signature)**  
- UDC, Sph, OroA, ImPA

> These panels are included as *documentation context*; the demo code uses a subset to keep the workflow concise.

---

## Installation

### Prerequisites
- Python **3.8+**
- `pip`

### Setup
Clone or download this repository, then install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the interactive demo:

```bash
python main_pipeline.py
```

You will see a menu:

- **[1] ED Triage** → ROC + DCA  
- **[2] Hematoma Expansion Risk** → Calibration + Risk Strata  
- **[3] Outcome Prediction (mRS)** → SHAP-style Waterfall Explanation  
- **[Q] Quit**

---

## Methodology (Demo)

- **Models**
  - ED triage / expansion risk: Logistic Regression
  - Outcome prediction: Random Forest (demo)

- **Validation visuals**
  - ROC curve (AUC)
  - Decision Curve Analysis (net benefit)
  - Calibration curve (reliability)
  - Risk stratification (low / moderate / high)

- **Explainability**
  - A **SHAP-like** waterfall plot is generated using a simple contribution heuristic (for demo purposes).

---

## Repository Structure

```
Ocular_Stroke_Pipeline/
├── main_pipeline.py
├── requirements.txt
└── README.md
```

---

## Citation

If you use this repository for academic demonstration, please cite your related manuscript / preprint and include a software citation such as:

> Ocular Metabolomics Clinical Pipeline (Demo CDSS). Version 1.0. GitHub repository.

---

## License

Choose a license suitable for your release (e.g., MIT, Apache-2.0).  
This demo package does not include a license file by default.

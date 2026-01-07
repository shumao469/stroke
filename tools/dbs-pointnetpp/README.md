# PointNet++ for Personalized Deep Brain Stimulation (DBS) Efficacy Prediction

Official PyTorch implementation of a **PointNet++-based 3D point-cloud framework** for **personalized DBS efficacy prediction** in Parkinson’s disease (PD).

This framework integrates:
- **Patient-specific neuroanatomy** derived from MRI/CT (surfaces / labels)
- **Lead-DBS / FEM electric-field simulation outputs** (e-field magnitude and vectors)
- **Geometric deep learning** (PointNet++ with multi-scale set abstraction)

The model predicts clinical motor outcome (e.g., **MDS-UPDRS III improvement**) and supports multi-center generalization evaluation (e.g., **Leave-Center-Out**).

> Note: The repository provides a *research pipeline*. For clinical use, additional validation, governance, and regulatory processes are required.

---

## Highlights

- **End-to-end demo pipeline**: data preparation → point cloud fusion → training → evaluation  
- **Strict center-level split** (Leave-Center-Out) for external generalization testing  
- **Calibration + Decision Curve Analysis (DCA)** for clinical utility assessment  
- **Reproducibility utilities** (seed locking for NumPy/PyTorch/CPU/GPU)

---

## Repository Layout

Typical folder structure:

- `dbs_integrated_pipeline.py`  
  Main entry: runs the integrated “digital-twin” workflow (demo mode supported).

- `dbs_pipeline_demo.py`  
  Lightweight demo script (quick sanity checks / example usage).

- `dbs_external_validation.py`  
  External validation utilities (Leave-Center-Out, calibration, DCA).

- `mock_dbs_data/`  
  Synthetic data for demonstration/testing (when MATLAB/Lead-DBS is unavailable).

- `leaddbs_output/`  
  Placeholder directory for Lead-DBS outputs (MATLAB pipeline results).

- `processed_pcd/`  
  Output directory for fused point clouds (`.npz`) and intermediate artifacts.

See `docs/DATA_SPEC.md` for strict input/output format definitions.

---

## Installation

```bash
# recommended: create a clean env
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\\Scripts\\activate  # Windows

pip install -r requirements.txt
Requirements (typical)

Python 3.8+

PyTorch (GPU recommended)

scikit-learn

numpy, scipy

tqdm, matplotlib

(Optional)

MATLAB Engine API for Python (only needed if you run real Lead-DBS via MATLAB)

Quick Start
Run the integrated pipeline (demo mode supported):

bash
复制代码
python dbs_integrated_pipeline.py
Expected console flow (example):

Phase 1: generate multi-center mock data (or call MATLAB Lead-DBS if available)

Phase 2: convert outputs to unified point clouds (.npz)

Phase 3: Leave-Center-Out split

Phase 4: train PointNet++ model

Phase 5: external validation (discrimination + calibration + DCA)

Outputs are written under:

processed_pcd/ (point clouds)

runs/ or outputs/ (depending on your config)

Data Format: Point Cloud .npz
Each patient sample is stored as a single .npz file.

Required keys (recommended strict spec):

points: (N, C) float32

xyz coordinates + anatomical labels + e-field features, etc.

y_reg: float (or shape (1,))

continuous clinical outcome (e.g., ΔMDS-UPDRS III)

y_cls: int {0,1}

responder label (optional; depends on your study definition)

center: str

center/site ID for Leave-Center-Out split

Please refer to: docs/DATA_SPEC.md for the full strict schema and conventions.

Validation (Multi-center)
This repo supports Leave-Center-Out evaluation:

Train: Center A + Center B

External validation: Center C

Metrics (typical):

Regression: R² / MAE / RMSE

Classification (if enabled): AUC

Calibration: Brier score + calibration curve

Clinical utility: Decision Curve Analysis (DCA)

Citation
If you find this code useful, please cite:

bibtex
复制代码
@article{Xu2026high,
  title={PointNet++-based 3D point cloud deep learning for personalized deep brain stimulation efficacy prediction in Parkinson disease},
  author={Yinghao Zhu and Ru Wang and Minyan Ge and Yuchun Wang and Zihao Liu and Gongyi Zhu and Shugeng Chen and Bo Shen and Yimin Sun and Fengtao Liu and Jue Zhao and Narasimha M. Beeraka and Virak Sorn and Haiyin Wang and Vladimir N. Nikolenko and Jianjun Wu and Shumao Xu},
  year={2026},
  copyright={AAAS}
}
Contact
Shumao Xu, Ph.D.
Institute of Science and Technology for Brain-inspired Intelligence, Fudan University
📩 shumaoxu@fudan.edu.cn
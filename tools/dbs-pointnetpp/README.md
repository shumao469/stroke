# PointNet++ for Personalized Deep Brain Stimulation (DBS) Efficacy Prediction

Official PyTorch implementation of a **PointNet++-based 3D point-cloud framework** for **personalized DBS efficacy prediction** in Parkinson’s disease (PD).

This framework integrates patient-specific neuroanatomy and biophysical electric-field simulations into unified point clouds, enabling the model to directly learn spatial field–tissue interaction patterns for predicting clinical improvement (e.g., **MDS-UPDRS III**).

> **Compliance note**: The evaluation utilities are designed to support TRIPOD+AI-style reporting (center-level split, calibration, decision curve analysis). This repository is for research use and does not constitute a clinical decision system.

---

## Key Ideas

- **Geometry-preserving learning**: rather than compressing electric-field patterns into scalar features (e.g., VTA), we keep **high-dimensional spatial fidelity**.
- **Unified patient representation**: fuse **anatomy + tissue labels + FEM electric field** into a single **3D point cloud** per patient.
- **Multi-center generalization**: support **Leave-Center-Out** validation to reduce site-specific overfitting.
- **Clinical utility**: provide **calibration** (Brier score/curves) and **Decision Curve Analysis (DCA)**.

---

## Repository Structure

Typical files and folders (may vary depending on your local setup):

- `dbs_integrated_pipeline.py`  
  Main entry point. Runs an end-to-end pipeline from (mock or real) Lead-DBS outputs → point-cloud fusion → training → external validation.

- `dbs_pointnetpp_model.py` (or similar)  
  PointNet++ backbone implementation and heads (regression / optional classification).

- `dbs_external_validation.py` (or similar)  
  External validation suite: center split, metrics, calibration, DCA.

- `docs/`  
  - `DATA_SPEC.md`: strict `.npz/.npy` input/output specs (recommended)
  - other notes

- `processed_pcd/` (generated)  
  Point cloud `.npz` files per patient (**should not be committed to Git**).

> **Important**: generated artifacts such as `processed_pcd/`, `runs/`, `outputs/`, and `.npz` datasets should be ignored via `.gitignore`.

---

## Installation

### 1) Create environment (recommended)

```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

Typical requirements:
- Python 3.8+
- PyTorch (GPU strongly recommended)
- numpy, scipy
- scikit-learn (metrics, calibration)
- tqdm, matplotlib

Optional:
- MATLAB Engine API for Python (only if you call real Lead-DBS via MATLAB)

---

## Quick Start (Demo Mode)

Run the integrated pipeline:

```bash
python dbs_integrated_pipeline.py
```

If MATLAB / Lead-DBS is not available, the pipeline may switch to **simulation mode** (synthetic demo data) for reproducibility testing and tutorial use.

Expected workflow (conceptually):
1) **(Optional) Lead-DBS execution** (MATLAB): coregistration → normalization → electrode reconstruction → FEM E-field simulation  
2) **ETL & Fusion**: convert anatomy/mesh/voxels + FEM nodes to a unified point cloud `.npz`  
3) **Study design**: Leave-Center-Out split  
4) **Training**: PointNet++ with regression (+ optional classification)  
5) **External validation**: discrimination + calibration + DCA

---

## Data Format (`.npz`) — Point Cloud Sample

Each patient corresponds to a single `.npz` file.

Recommended required keys:
- `points`: `float32`, shape `(N, C)`  
  Example channel layout (recommended):
  - `0:3`  → XYZ coordinates (mm, same coordinate system as fusion output)
  - `3`    → tissue / structure ID (int-like stored as float ok)
  - `4`    → E-field magnitude
  - `5:8`  → E-field vector components (Ex, Ey, Ez)
  - `8:11` → optional geometric / normalized features (normals, distances, etc.)

- `y_reg`: float (scalar)  
  Continuous outcome (e.g., ΔMDS-UPDRS III)

Optional keys:
- `y_cls`: int {0,1}  
  Binary responder label (threshold defined in your study)

- `center`: string  
  Center/site ID used for Leave-Center-Out splitting

**Strict specs and file conventions are documented in**: `docs/DATA_SPEC.md`.

---

## Multi-Center Validation (Leave-Center-Out)

A recommended external validation design:
- Train: Center A + Center B
- Test (external): Center C

This simulates deployment to a previously unseen hospital and reduces center-leakage.

Typical metrics:
- Regression: R², MAE, RMSE  
- Classification (if enabled): AUC  
- Calibration: Brier score + calibration curve  
- Clinical utility: Decision Curve Analysis (DCA)

---

## Reproducibility

The pipeline includes deterministic/reproducibility utilities (seed locking across Python/NumPy/PyTorch; optional CUDA determinism). Exact reproducibility may still vary across GPU drivers and CUDA versions.

---

## Citation

If you use this code, please cite:

```bibtex
@article{Xu2026high,
  title={PointNet++-based 3D point cloud deep learning for personalized deep brain stimulation efficacy prediction in Parkinson disease},
  author={Shumao Xu et al.},
  year={2026},
  copyright={ }
}
```

---

## Contact
Shumao Xu, Institute of Science and Technology for Brain-inspired Intelligence, Fudan University  
📩 shumaoxu@fudan.edu.cn

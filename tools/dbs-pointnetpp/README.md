PointNet++ for Personalized Deep Brain Stimulation (DBS) Efficacy Prediction

This repository contains the official PyTorch implementation of the PointNet++ based 3D point cloud deep learning framework for predicting Deep Brain Stimulation (DBS) outcomes in Parkinson's disease (PD).

By integrating Lead-DBS biophysical simulations with geometric deep learning, this framework directly processes patient-specific neuroanatomy and electric field distributions to predict clinical motor improvements (MDS-UPDRS III).

Compliance: This codebase adheres to the TRIPOD+AI 2024 Reporting Guidelines, featuring strict data leakage control (Leave-Center-Out), calibration analysis, and decision curve analysis (DCA).

📂 Key Components & Pipelines

This repository is structured around two core workflows implemented in dbs_integrated_pipeline.py.

1. 🚀 The Integrated Pipeline (dbs_integrated_pipeline.py)

This is the main entry point that automates the "End-to-End" process: from raw neuroimaging data processing to AI model training. It serves as a bridge between clinical data processing (MATLAB) and deep learning (Python).

Workflow Stages:

Lead-DBS Interface (MATLAB/Mock): * Invokes the MATLAB engine to run the Lead-DBS pipeline (Coregistration -> Normalization -> PaCER Electrode Reconstruction -> FEM Electric Field Simulation).

Note: If MATLAB is not detected, it automatically switches to Simulation Mode, generating synthetic patient data for demonstration purposes.

ETL & Point Cloud Fusion:

Converts Lead-DBS outputs (Voxels/Meshes) into unified 3D Point Clouds (.npz).

Fusion Strategy: Merges anatomical coordinates, tissue labels (STN/GPi), and electric field vectors ($E, E_x, E_y, E_z$) into a dense point set (~227k points/patient, downsampled for training).

PointNet++ Model Training:

Initializes the fixed PointNet++ architecture (specifically tuned to handle 11-channel inputs: 3 coords + 8 features).

Trains on the internal development set using a multi-task loss (Regression + Classification).

2. 🛡️ The External Validation Pipeline (TRIPOD-AI Compliant)

Implemented as Phase 5 within the integrated script, this module rigorously evaluates model generalizability and clinical utility, strictly following TRIPOD-AI standards.

Key Validation Features:

Leave-Center-Out Cross-Validation (Data Leakage Control): * The script automatically identifies Center_ID tags in the data.

Training: Performed strictly on Center A and Center B.

Validation: Performed strictly on Center C (External).

Benefit: This simulates real-world deployment where the model sees patients from a completely new hospital, preventing "site-specific" overfitting.

Model Calibration: * Calculates the Brier Score and generates Calibration Curves.

Ensures that a predicted 80% success probability truly corresponds to an 80% real-world success rate.

Decision Curve Analysis (DCA):

Computes the Net Benefit across different probability thresholds.

Answers the clinical question: "Does using this AI model lead to better patient outcomes compared to treating everyone or treating no one?"

Reproducibility:

Includes a set_reproducibility() function that locks random seeds (CPU, GPU, Numpy) to ensure exact replication of results.

🛠️ Installation

# Clone the repository
git clone [https://github.com/yourlab/dbs-pointnetpp.git](https://github.com/yourlab/dbs-pointnetpp.git)
cd dbs-pointnetpp

# Install dependencies
pip install -r requirements.txt


Requirements:

Python 3.8+

PyTorch 1.12+

scikit-learn (for metrics and calibration)

tqdm

(Optional) MATLAB Engine API for Python (for real Lead-DBS execution)

🏃 Usage

Running the Full Pipeline

To run the complete simulation—including data generation, training, and TRIPOD-AI validation report—execute:

python dbs_integrated_pipeline.py


Expected Output:
The script acts as a "Digital Twin" demo, printing progress for each phase:

======== Phase 1: Multi-Center Simulation (Lead-DBS) ========
🏥 Generating Data for Center_A...
🏥 Generating Data for Center_C... (External)

======== Phase 3: Study Design (Leave-Center-Out) ========
📚 Internal Training Set (Center A+B): 8 samples
🛡️ External Validation Set (Center C): 4 samples
⚠️ Status: Strict Separation Enforced (No Data Leakage)

======== Phase 5: External Validation (TRIPOD-AI Report) ========
📝 [Discrimination] External AUC: 0.8750
📈 [Calibration] Brier Score: 0.1240 (Ideal: 0.0)
⚖️ [Decision Curve Analysis]
   Threshold | Net Benefit (Model)
   0.40      | 0.3500


📊 Data Format (.npz)

The pipeline generates/expects .npz files for each patient containing:

Key

Shape

Description

points

(N, 11)

Cols 0-2: XYZ coords



Cols 3: Tissue ID (0=STN, 1=GPi...)



Cols 4: E-Field Magnitude



Cols 5-7: E-Field Vector ($V_x, V_y, V_z$)



Cols 8-10: Normalized Geometry

y_reg

Scalar

MDS-UPDRS III improvement score (0-100)

y_cls

0 or 1

Responder binary label (Threshold > 40%)

center

String

Hospital/Center ID (e.g., "Center_C") for validation splitting

📝 Citation

If you use this code or the TRIPOD-AI validation framework, please cite:

@article{Xu2026high,
  title={PointNet++-based 3D point cloud deep learning for personalized deep brain stimulation efficacy prediction in Parkinson disease},
  author={Xu, Shumao and et al.},
  journal={},
  year={2026}
}


📧 Contact

For technical questions regarding the pipeline integration:
Shumao Xu, Ph.D. Institute of Science and Technology for Brain-inspired Intelligence, Fudan University

Email: shumaoxu@fudan.edu.cn

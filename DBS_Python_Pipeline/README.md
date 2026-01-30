AI-Enabled Personalization of DBS Pipeline

This repository contains a Python implementation of the integrative computational pipeline for Deep Brain Stimulation (DBS) planning and visualization

Overview

Precise targeting in DBS is often complicated by postoperative brain shift and manual electrode localization errors. This tool provides an automated, Python-based workflow (compatible with WSL/Linux) that replaces traditional manual steps with AI-driven solutions.

Key Features

Brain Shift Correction & Registration

Uses ANTsPy (Advanced Normalization Tools) for high-precision registration.

Implements a pipeline to align Post-op CT with Pre-op MRI, including subcortical refinement to account for brain shift (pneumocephalus/CSF loss).

AI-Automated Electrode Reconstruction

PaCER-like Algorithm: Automates electrode localization using high-intensity voxel thresholding and Principal Component Analysis (PCA) for trajectory fitting.

Achieves high precision without manual contact selection.

Patient-Specific FEM Field Prediction

Finite Element Modeling (FEM): Simulates the Volume of Tissue Activated (VTA) based on tissue conductivity.

Solves the Laplace equation ($\nabla \cdot (\sigma \nabla V) = 0$) to predict the electric field distribution around the active contacts.

Installation

This tool relies on Python scientific computing libraries and ANTsPy for image processing.

Prerequisites

Python 3.8+

WSL (Windows Subsystem for Linux) or a native Linux environment is highly recommended due to antspyx dependencies.

Install Dependencies

pip install -r requirements.txt


Usage

Prepare Data: Organize your data in BIDS format or standard directories (T1, T2 MRI, and CT).

Run Pipeline:

python dbs_pipeline.py


Note: If no input files are found, the script will automatically generate synthetic "phantom" data to demonstrate the registration, reconstruction, and simulation steps.

File Structure

dbs_pipeline.py: Main execution script containing the DBSPipeline class.

requirements.txt: Python dependencies.

data/: Directory for storing patient imaging data (optional).

Reference

If you use this code in your research, please refer to the original publication.

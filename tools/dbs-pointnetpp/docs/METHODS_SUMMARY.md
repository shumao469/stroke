# Methods-aligned summary (high level)

This document paraphrases key *implementation-relevant* items from the Methods section
for reproducibility of the **point-cloud representation + PointNet++ training**, without reproducing
institution-/software-specific steps (e.g., medical image segmentation, electrode localization UI workflows,
or FEM solver project files).

## Unified point cloud construction (concept)

1. **Anatomy point cloud**
   - Dense 3D points sampled from relevant anatomical structures around the DBS target (e.g., STN context).

2. **Electrode point cloud**
   - Contact-aligned surface points along the implanted lead.

3. **E-field simulation**
   - FEM simulation (e.g., COMSOL) provides node-wise electric-field magnitude values.

4. **Fusion**
   - KD-tree nearest-neighbor fusion with k=32 to associate E-field values to geometry points.
   - FPS used to enforce spatial uniformity.

5. **Optional mapping step**
   - RBF interpolation can be used to map FEM node values to dense query points.

## PointNet++ (local-to-global)

- Set abstraction stages perform FPS and local grouping, producing hierarchical features.
- Global feature projection aggregates into a fixed-length embedding for prediction heads.
- Output supports:
  - regression (clinical improvement score)
  - optional responder classification

See `dbs_pointnetpp/model.py` for the architecture and `dbs_pointnetpp/train.py` for training/evaluation utilities.

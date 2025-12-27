# DATA_SPEC — DBS Point Cloud I/O Specification (MRI/CT → Surface Point Cloud → COMSOL FEM → RBF/FPS)

This document defines **strict, machine-checkable** input/output formats for producing **per-patient unified 3D point clouds**
used by the PointNet++ model (`dbs_pointnetpp`).

The goal is a reproducible pipeline:

1. **MRI/CT → anatomy & electrode geometry**
2. **COMSOL FEM → E-field node samples**
3. **E-field mapping to geometry points** (RBF and/or kNN / KD-tree)
4. **Farthest Point Sampling (FPS)** to a fixed size (e.g., ~227,500 points/patient)
5. Save as `.npz` for training/inference

> This repo does **not** implement imaging segmentation, lead localization UI, or COMSOL solving.  
> It **does** standardize what those upstream tools should export.

---

## 0) Coordinate convention (REQUIRED)

All exported arrays MUST be in the **same patient-specific coordinate space**.

### Recommended convention
- **Right-handed** coordinate system
- Units: **millimeters (mm)**
- Origin: any consistent origin (e.g., image origin or AC-PC aligned origin), but must be consistent across:
  - anatomy surface points
  - electrode points
  - FEM node coordinates

### REQUIRED metadata keys (see `meta.json`)
- `coord_system`: string, e.g., `"RAS"` or `"LPS"` or `"ACPC"`
- `units`: `"mm"`
- `space_id`: unique ID for the patient space (e.g., `"sub-001_acpc"`)

If you must convert between RAS/LPS/voxel indices, perform conversion **before** exporting `.npy/.npz`,
and record it in metadata.

---

## 1) File layout per patient (RECOMMENDED)

For each patient `sub-XXX/`:

```
sub-XXX/
  anatomy_xyz.npy
  anatomy_label.npy            (optional)
  electrode_xyz.npy
  electrode_contact_id.npy     (optional)
  fem_nodes_xyz.npy
  fem_nodes_E.npy              (or fem_nodes_Ex,Ey,Ez.npy)
  meta.json
  unified_points_227500.npz    (final training input)
```

---

## 2) Required inputs

### 2.1 `anatomy_xyz.npy` (REQUIRED)
Dense surface point cloud representing relevant anatomy around the DBS target.

- shape: `(Na, 3)`
- dtype: `float32` (float64 allowed but will be cast)
- columns: `[x, y, z]` in **mm**

**Recommended Na**: 100k–500k (dense enough to cover ROI)

Optional:
- `anatomy_label.npy`: shape `(Na,)` int32  
  - tissue/structure label per point (e.g., STN/GPi/WM/CSF)

---

### 2.2 `electrode_xyz.npy` (REQUIRED)
Point cloud representing DBS lead geometry (shaft + contacts). This should be **aligned** to anatomy space.

- shape: `(Ne, 3)`
- dtype: `float32`
- columns: `[x, y, z]` in **mm**

**Recommended Ne**: 2k–20k depending on how finely you sample the lead.

Optional:
- `electrode_contact_id.npy`: shape `(Ne,)` int32  
  - contact index per point (e.g., 0–3) or -1 for shaft

---

### 2.3 `fem_nodes_xyz.npy` (REQUIRED)
COMSOL-exported FEM mesh node coordinates for the E-field simulation.

- shape: `(Mf, 3)`
- dtype: `float32`
- columns: `[x, y, z]` in **mm**

**Recommended Mf**: >= 10k (more is better; depends on mesh resolution)

---

### 2.4 E-field values (REQUIRED; choose ONE representation)

#### Option A: magnitude only (simplest)
`fem_nodes_E.npy`
- shape: `(Mf,)` or `(Mf, 1)`
- dtype: `float32`
- units: V/m or V/mm (MUST be recorded in metadata)

#### Option B: vector field
`fem_nodes_Ex.npy`, `fem_nodes_Ey.npy`, `fem_nodes_Ez.npy`
- each shape: `(Mf,)`
- dtype: `float32`

Magnitude can be computed downstream: `E = sqrt(Ex^2 + Ey^2 + Ez^2)`.

**REQUIRED metadata keys**
- `efield_units`: `"V/m"` or `"V/mm"`
- `efield_type`: `"magnitude"` or `"vector"`

---

## 3) Mapping FEM E-field to geometry points

You may use either or both:
- **RBF interpolation** (continuous interpolation)
- **kNN / KD-tree fusion** (local neighborhood statistics)

### 3.1 RBF interpolation (SUPPORTED)
Use `dbs_pointnetpp.build_pointcloud.rbf_interpolate_field(...)`

Inputs:
- `sim_nodes_xyz = fem_nodes_xyz` `(Mf, 3)`
- `sim_nodes_E = fem_nodes_E` `(Mf,)`
- `query_xyz = geometry_xyz` where `geometry_xyz = vstack([anatomy_xyz, electrode_xyz])` `(Nq, 3)`

Output:
- `E_query`: `(Nq,)`

Recommended RBF settings:
- kernel: `"multiquadric"` or `"linear"`
- downsample FEM nodes if extremely large to manage memory

**Quality control (RECOMMENDED)**
- report correlation between RBF-mapped E and local kNN mean E at a subset of points
- check for extrapolation artifacts outside mesh domain

---

### 3.2 KD-tree kNN fusion (SUPPORTED)
Use `dbs_pointnetpp.build_pointcloud.fuse_pointcloud_kdtree(...)`

This attaches **local statistics** of FEM E-field to each geometry point:
- `E_mean`, `E_std`, `E_max` over k nearest FEM nodes (default `k=32`)

Output points: `(Nq, 6)`
- `[x, y, z, E_mean, E_std, E_max]`

---

## 4) FPS downsampling to fixed size (REQUIRED for training)

PointNet++ training is simplified when each patient has a fixed number of points.

### Recommended sizes
- ~227,500 points/patient (as you noted)
- or 200,000 points/patient for memory efficiency

**FPS definition**
- run FPS on xyz coordinates (uniform spatial coverage)
- keep corresponding feature channels

Implementation options:
- use your existing PointNet++ FPS in PyTorch (see `dbs_pointnetpp/sampling.py`)
- or offline FPS (preferred for saving final `.npz`) using GPU if available

---

## 5) Final training file: `unified_points_227500.npz` (REQUIRED)

This is the only file strictly required by `DBSPointCloudDataset`.

### Required keys

#### `points`
- shape: `(N, C)` where `N ≈ 227500`
- dtype: `float32`
- columns:
  - `points[:,0:3]` = xyz (mm)
  - `points[:,3:]` = features

**Minimum viable C**: 6  
Example: `[x,y,z, E_mean, E_std, E_max]`

Recommended additional features (optional):
- `E_rbf` (RBF mapped magnitude)
- `E_grad` (magnitude gradient norm)
- `tissue_onehot` or `tissue_id`
- `distance_to_contact_i` (per contact)
- `signed_distance_to_target_surface` (if available)

#### `y_reg`
- scalar float32
- meaning: clinical efficacy target (e.g., **ΔMDS-UPDRS III**, improvement positive)

#### `y_cls` (optional)
- scalar int64 in `{0,1}` (responder label), or `-1` if not available

### Optional keys (highly recommended)

#### `meta`
- JSON string or small dict-like fields (depending on how you save)
- must include:
  - `subject_id`
  - `coord_system`
  - `units`
  - `efield_units`
  - `efield_type`
  - `fps_n`
  - `k_knn` (if KD-tree used)
  - `rbf_function` (if RBF used)

---

## 6) Naming and sign conventions for targets (RECOMMENDED)

For clinical interpretability:
- define whether **improvement** is positive (recommended)
- e.g., `y_reg = pre_UPDRS - post_UPDRS` (so higher is better), document in `meta.json`

---

## 7) Minimal example

### Create a unified point cloud using KD-tree fusion

```python
import numpy as np
from dbs_pointnetpp.build_pointcloud import fuse_pointcloud_kdtree

anatomy = np.load("anatomy_xyz.npy")         # (Na,3)
electrode = np.load("electrode_xyz.npy")     # (Ne,3)
fxyz = np.load("fem_nodes_xyz.npy")          # (Mf,3)
fE   = np.load("fem_nodes_E.npy")            # (Mf,)

points = fuse_pointcloud_kdtree(anatomy, electrode, fxyz, fE, k=32)  # (Nq,6)

# TODO: FPS to N=227500 offline (recommended)
np.savez_compressed("unified_points_227500.npz", points=points, y_reg=np.float32(0.0), y_cls=np.int64(-1))
```

---

## 8) Validation checklist (RECOMMENDED)

Before training, verify:
1. **Coordinate alignment**: electrode sits anatomically correct relative to target
2. **Units**: xyz in mm; E-field in documented units
3. **Mesh domain**: geometry points mostly inside/near FEM domain
4. **Point counts**: each `unified_points_*.npz` has identical `N`
5. **No NaNs/inf**: in points/features
6. **Label distribution**: responder vs non-responder is not collapsed

---

## 9) Suggested extensions (optional)

- Add `scripts/validate_npz.py` to enforce schema checks automatically
- Add `scripts/offline_fps.py` to FPS-downsample large clouds to exactly `N=227500`
- Add support for multi-contact stimulation settings as structured metadata (e.g., amplitude, active contacts)

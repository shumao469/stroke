# NEFEL Biofeature Extraction

A lightweight, **paper-oriented** toolkit for extracting quantitative biomarkers from
histology / immunofluorescence (IF) / immunohistochemistry (IHC) images.

This repository is a GitHub-ready packaging of the original notebook:
`Biofeature_extraction_NEFEL.ipynb`.

---

## What you get

Section-aligned analyses (matching the notebook structure):

- **Section 1:** iNOS / Arg analysis
- **Section 2:** Iba1 analysis (Day 1)
- **Section 3:** Claudin-5 analysis
- **Section 4:** CD31 analysis
- **Section 5:** Synapse analysis
- **Section 6:** GAP43 analysis
- **Section 7:** TUNEL analysis

Each section returns a **flat dict** of metrics (easy to append into a CSV).

📘 **Detailed documentation**
- English: `docs/sections.md`
- 中文说明：`docs/README_CN.md`

---

## Repository layout

```
nefel-biofeatures/
├─ src/
│  └─ nefel/
│     ├─ core.py                 # shared helpers (tissue mask, split channels, etc.)
│     ├─ sections.py             # section-aligned wrappers (Section 1–7)
│     ├─ feature_extraction.py   # compatibility layer (re-exports common functions)
│     └─ markers/
│        ├─ inos_arg.py
│        ├─ iba1_day1.py
│        ├─ claudin5.py
│        ├─ cd31.py
│        ├─ synapse.py
│        ├─ gap43.py
│        └─ tunel.py
├─ examples/
│  └─ quickstart.py
├─ tests/
├─ requirements.txt
├─ setup.py
├─ setup.cfg
└─ README.md
```

---

## Installation

### Option A) Install from source (recommended for development)

```bash
pip install -r requirements.txt
pip install -e .
```

### Option B) Minimal dependencies (manual)

Core dependencies:

- numpy
- pandas
- scipy
- scikit-image
- opencv-python
- matplotlib

---

## Quickstart

Run the example script:

```bash
python examples/quickstart.py --image /path/to/your/image.jpg --out metrics.csv
```

---

## Usage patterns

### 1) In-memory workflow (recommended)

```python
import cv2
from nefel.sections import (
    section1_inos_arg,
    section2_iba1_day1,
    section3_claudin5,
    section4_cd31,
    section5_synapse,
    section6_gap43,
)

bgr = cv2.imread("sample.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

m1 = section1_inos_arg(rgb)
m2 = section2_iba1_day1(rgb, prefix="day1_m1", out_dir="qc_masks")
m3 = section3_claudin5(rgb, prefix="day1_m1", out_dir="qc_masks")
m4 = section4_cd31(rgb, prefix="day1_m1", out_dir="qc_masks")
m5 = section5_synapse(rgb, pre_channel=1, post_channel=0)  # e.g., G=Syn, R=PSD95
m6 = section6_gap43(rgb, prefix="day3_m1", out_dir="qc_masks")
```

### 2) Optional QC outputs

Several analyses can write mask overlays (skeleton / positive masks) **only if** you set `out_dir=...`:

- Iba1 (Section 2)
- Claudin-5 (Section 3)
- CD31 (Section 4)
- GAP43 (Section 6)
- TUNEL (Section 7)

This makes the library safe to import and run in batch mode.

---

## Reproducibility checklist (recommended)

- Keep imaging parameters consistent within a cohort.
- Validate thresholds (percentile / OD cutoff) on a subset, then lock them.
- Always spot-check QC overlays for ~5–10% of images per batch.

---

## Citation

If you use this code, please cite your paper / preprint here (replace the placeholder):

```bibtex
@article{your2025nefel,
  title={...},
  author={...},
  journal={...},
  year={2025}
}
```

---

## Contact

For questions and contributions, please open a GitHub issue or contact the maintainers.

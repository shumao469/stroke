# UMAP State

UMAP State is a reproducible toolkit for **UMAP-based state-space analysis** of **LFP- or spike-derived feature time series**.
It is built for TI / task paradigms with clear epochs (**pre / stimulation / post**) and focuses on:

- **State-space embedding** using UMAP (CPU via `umap-learn`, optional GPU via RAPIDS cuML)
- **Discrete state identification** by clustering in the embedding (HDBSCAN or KMeans)
- **State dynamics**: state sequence, occupancy, transition matrix

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Optional extras:
- `hdbscan` for clustering (otherwise use `--cluster kmeans`)
- GPU UMAP: RAPIDS cuML (`cuml`) + compatible CUDA stack (optional)

---

## Quick start (10 Hz example)

Run on your feature array:

```bash
umap-state \
  --features path/to/features.npy \
  --fs 1 \
  --outdir outputs/umap_10hz \
  --stim-onset 0 \
  --pre-dur 60 --stim-dur 180 --post-dur 600 \
  --n-neighbors 30 --min-dist 0.1 \
  --cluster kmeans --kmeans-k 5
```

> `--fs` is **feature sampling rate** (bins/sec), not raw LFP 20 kHz.

If you have a time vector:

```bash
umap-state --features features.npy --time time.npy --outdir outputs/umap_10hz ...
```

---

## How to cut pre / stim / post (recommended)

After alignment (`stim onset = 0 s`):

- pre:  [-60, 0) seconds
- stim: [0, 180) seconds
- post: [180, 780) seconds

See:
- `docs/ALIGNMENT.md`
- `docs/USAGE.md`

---

## Outputs

The run folder contains:

- `umap_embedding.npy`
- `state_labels.npy`
- `umap_states.png`
- `umap_segments.png`
- `state_sequence.png`
- `transition_matrix.npy/.png`
- `state_occupancy.npy/.png`
- `run_meta.json`

---

## Data format

UMAP State takes **feature time series**, not raw LFP.

See `docs/DATA_SPEC.md` for the strict input format and recommended windowing.

---

## Citation

If you use this code, please cite your paper / preprint here (replace the placeholder):

```bibtex
@article{your2026nefel,
  title={...},
  author={...},
  journal={...},
  year={2026}
}
```

---

## Contact

For questions and contributions, please open a GitHub issue or contact the maintainers.
